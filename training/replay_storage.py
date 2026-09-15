"""Immutable replay chunks shared by checkpoints, with relocatable references."""

import os
import re
from collections import deque
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import torch

from .replay_buffer import TrainingSample, atomic_torch_save

CHUNK_SIZE = 256
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def training_data_root(config):
    if config.get("training_data_dir"):
        return Path(config["training_data_dir"]).expanduser().resolve()
    checkpoint_dir = Path(config["checkpoint_dir"]).resolve()
    if checkpoint_dir.is_relative_to(PROJECT_ROOT):
        return PROJECT_ROOT / "训练数据"
    shared = Path(
        os.path.commonpath([checkpoint_dir, Path(config["output_dir"]).resolve()])
    )
    return shared / "训练数据"


def file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReplayStorage:
    def __init__(self, config):
        root = training_data_root(config)
        checkpoint_dir = Path(config["checkpoint_dir"]).resolve()
        # Relative keys survive moving the whole project to a different folder.
        key = os.path.relpath(checkpoint_dir, root.parent)
        self.directory = (
            root / f"{checkpoint_dir.name}-{sha256(key.encode()).hexdigest()[:12]}"
        )

    def save(self, buffer, checkpoint_path):
        """Write new rows first, then return a small per-checkpoint manifest.

        Existing chunks are never overwritten or deleted. Historical checkpoints
        retain their own sample order and replay RNG even when the ring rolls over.
        """
        samples = list(buffer.samples)
        refs = list(buffer._storage_refs)
        if len(refs) != len(samples):
            raise ValueError("Replay samples and storage references are misaligned")
        present = {}
        for index, ref in enumerate(refs):
            if ref is None:
                continue
            path = ref[0]
            if path not in present:
                present[path] = path.parent == self.directory and path.is_file()
            if not present[path]:
                refs[index] = None
        index = 0
        while index < len(samples):
            if refs[index] is not None:
                index += 1
                continue
            end = index + 1
            while end < len(samples) and refs[end] is None and end - index < CHUNK_SIZE:
                end += 1
            path = self.directory / f"samples-{uuid4().hex}.pt"
            rows = [sample.to_dict() for sample in samples[index:end]]
            atomic_torch_save({"storage_version": 1, "samples": rows}, path)
            digest = file_digest(path)
            for offset in range(end - index):
                refs[index + offset] = (path, digest, offset)
            index = end
        chunks = []
        checkpoint_parent = Path(checkpoint_path).resolve().parent
        for path, digest, offset in refs:
            relative = os.path.relpath(path, checkpoint_parent)
            if (
                chunks
                and chunks[-1]["path"] == relative
                and chunks[-1]["stop"] == offset
            ):
                chunks[-1]["stop"] += 1
            else:
                chunks.append(
                    {
                        "path": relative,
                        "sha256": digest,
                        "start": offset,
                        "stop": offset + 1,
                    }
                )
        buffer._storage_refs = deque(refs, maxlen=buffer.capacity)
        return {
            "storage_version": 1,
            "metadata": buffer.state_dict(include_samples=False),
            "sample_count": len(samples),
            "chunks": chunks,
        }

    @staticmethod
    def load(buffer, manifest, checkpoint_path):
        """Return True only for missing chunks; corrupt data fails explicitly."""
        if manifest.get("storage_version") != 1:
            raise ValueError("Unsupported external replay storage version")
        metadata = manifest["metadata"]
        count = manifest["sample_count"]
        chunks = manifest["chunks"]
        if type(count) is not int or not 0 <= count <= metadata["capacity"]:
            raise ValueError("Invalid external replay sample count")
        paths = []
        total = 0
        for chunk in chunks:
            relative = Path(chunk["path"])
            if relative.is_absolute() or not re.fullmatch(
                r"[0-9a-f]{64}", chunk["sha256"]
            ):
                raise ValueError("Invalid external replay reference")
            start, stop = chunk["start"], chunk["stop"]
            if (
                type(start) is not int
                or type(stop) is not int
                or not 0 <= start < stop <= CHUNK_SIZE
            ):
                raise ValueError("Invalid external replay chunk range")
            total += stop - start
            paths.append((Path(checkpoint_path).resolve().parent / relative).resolve())
        if total != count:
            raise ValueError("External replay sample count does not match its manifest")
        # Validate metadata even if the user has removed every sample file.
        buffer.load_state_dict({**metadata, "samples": []})
        if any(not path.exists() for path in paths):
            return True
        refs = []
        for path, chunk in zip(paths, chunks):
            try:
                if file_digest(path) != chunk["sha256"]:
                    raise ValueError(
                        f"训练数据校验失败：{path}。请恢复备份，或清理该任务的训练数据后重新积累样本。"
                    )
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except FileNotFoundError:
                buffer.load_state_dict({**metadata, "samples": []})
                return True
            if (
                payload.get("storage_version") != 1
                or len(payload["samples"]) < chunk["stop"]
            ):
                raise ValueError(f"Invalid replay chunk: {path}")
            rows = payload["samples"][chunk["start"] : chunk["stop"]]
            buffer.add_episode(TrainingSample.from_dict(row) for row in rows)
            refs.extend(
                (path, chunk["sha256"], offset)
                for offset in range(chunk["start"], chunk["stop"])
            )
        buffer._storage_refs = deque(refs, maxlen=buffer.capacity)
        return False
