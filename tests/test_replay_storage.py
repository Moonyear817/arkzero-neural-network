"""Persistence acceptance: deletion, old saves, relocation and exact continuation."""

import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from test_training_alphazero import env_factory, sample_from, tiny_config

from training import AlphaZeroTrainer, TrainingControl
from training.replay_buffer import ReplayBuffer
from training.replay_storage import ReplayStorage


def same_tree(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_tree(a, b)
    else:
        assert left == right


def chunk_paths(manifest, checkpoint):
    return [(checkpoint.parent / c["path"]).resolve() for c in manifest["chunks"]]


def test_replay_chunks_reused_and_history_survives_rollover(simple, operator, tmp_path):
    storage = ReplayStorage(tiny_config(tmp_path))
    sample = sample_from(env_factory(simple, operator)())
    replay = ReplayBuffer(3)
    replay.add_episode([sample, sample, sample])
    checkpoint = tmp_path / "checkpoints/latest.pt"
    first = storage.save(replay, checkpoint)
    files = list(storage.directory.glob("*.pt"))
    assert len(files) == 1
    original = files[0].read_bytes()
    replay.sample(1)  # Advancing replay RNG must not duplicate sample files.
    second = storage.save(replay, checkpoint)
    assert first["chunks"] == second["chunks"]
    assert first["metadata"]["rng_state"] != second["metadata"]["rng_state"]
    replay.add_episode([sample_from(env_factory(simple, operator)(), z=0.5)])
    third = storage.save(replay, checkpoint)
    assert len(list(storage.directory.glob("*.pt"))) == 2
    assert files[0].read_bytes() == original
    old = ReplayBuffer()
    assert storage.load(old, first, checkpoint) is False
    assert [s.z for s in old.samples] == [-1, -1, -1]
    current = ReplayBuffer()
    assert storage.load(current, third, checkpoint) is False
    assert [s.z for s in current.samples] == [-1, -1, 0.5]
    same_tree(current.state_dict(), replay.state_dict())
    storage.save(current, checkpoint)
    assert len(list(storage.directory.glob("*.pt"))) == 2


def test_legacy_embedded_replay_converts_without_changing_weights(
    simple, operator, tmp_path
):
    config = tiny_config(tmp_path)
    factory = env_factory(simple, operator)
    trainer = AlphaZeroTrainer(config, factory)
    trainer.buffer.add_episode([sample_from(factory())])
    legacy = trainer._checkpoint_payload()
    legacy.update(format_version=1, replay_buffer=trainer.buffer.state_dict())
    old_path = tmp_path / "legacy.pt"
    torch.save(legacy, old_path)
    old_bytes = old_path.read_bytes()
    loaded = AlphaZeroTrainer({**config, "resume_from": str(old_path)}, factory)
    # An alternate checkpoint location must resolve its own relative references.
    new_path = loaded.save_checkpoint(tmp_path / "export/copied.pt")
    payload = torch.load(new_path, weights_only=True)
    assert payload["format_version"] == 2
    assert "samples" not in payload["replay_buffer"]
    assert "samples" not in payload["replay_buffer"]["metadata"]
    restored = AlphaZeroTrainer({**config, "resume_from": str(new_path)}, factory)
    same_tree(restored.model.state_dict(), trainer.model.state_dict())
    same_tree(restored.buffer.state_dict(), trainer.buffer.state_dict())
    assert old_path.read_bytes() == old_bytes


@pytest.mark.parametrize("delete_all", [False, True])
def test_deleted_samples_keep_parameters_and_regenerate_mid_iteration(
    simple, operator, tmp_path, delete_all
):
    factory = env_factory(simple, operator)
    config = tiny_config(tmp_path)
    trainer = AlphaZeroTrainer(config, factory)
    control = TrainingControl()
    step = trainer._training_step

    def stop_after_update():
        result = step()
        control.stop()
        return result

    trainer._training_step = stop_after_update
    trainer.train(control=control)
    assert len(trainer.pending["losses"]) == 1
    assert len(trainer.buffer) > 0
    checkpoint = trainer.checkpoint_dir / "latest.pt"
    if delete_all:
        shutil.rmtree(trainer.replay_storage.directory)
    else:
        next(trainer.replay_storage.directory.glob("*.pt")).unlink()
    restored = AlphaZeroTrainer({**config, "resume_from": str(checkpoint)}, factory)
    assert len(restored.buffer) == 0
    same_tree(restored.model.state_dict(), trainer.model.state_dict())
    same_tree(restored.optimizer.state_dict(), trainer.optimizer.state_dict())
    assert len(restored.pending["losses"]) == 1
    # Inference loads only the model, independently of deleted replay.
    expected = trainer.evaluate(seeds=[19])
    actual = restored.evaluate(checkpoint, seeds=[19])
    for key in ("success_rate", "average_return", "average_kills", "average_leaks"):
        assert actual[key] == expected[key]
    events = []
    restored.train(callback=events.append)
    assert any(e.kind == "replay_data_missing" for e in events)
    assert restored.iteration == 1 and restored.pending is None
    assert len(restored.buffer) > 0
    assert (
        restored.history[0]["optimizer_steps"] == config["training_steps_per_iteration"]
    )
    assert restored.total_generated_states > trainer.total_generated_states
    assert list(restored.replay_storage.directory.glob("*.pt"))
    reloaded = AlphaZeroTrainer({**config, "resume_from": str(checkpoint)}, factory)
    same_tree(reloaded.buffer.state_dict(), restored.buffer.state_dict())
    same_tree(reloaded.model.state_dict(), restored.model.state_dict())


def test_move_whole_workspace_preserves_exact_replay(simple, operator, tmp_path):
    original_root = tmp_path / "original"
    moved_root = tmp_path / "moved"
    factory = env_factory(simple, operator)
    trainer = AlphaZeroTrainer(tiny_config(original_root), factory)
    trainer.buffer.add_episode([sample_from(factory())])
    trainer.save_checkpoint()
    shutil.move(original_root, moved_root)
    config = tiny_config(
        moved_root, resume_from=str(moved_root / "checkpoints/latest.pt")
    )
    restored = AlphaZeroTrainer(config, factory)
    same_tree(restored.buffer.state_dict(), trainer.buffer.state_dict())
    files_before = set((moved_root / "训练数据").rglob("*.pt"))
    restored.save_checkpoint()
    assert set((moved_root / "训练数据").rglob("*.pt")) == files_before


def test_corrupt_replay_fails_clearly_instead_of_silently_resetting(
    simple, operator, tmp_path
):
    storage = ReplayStorage(tiny_config(tmp_path))
    replay = ReplayBuffer()
    replay.add_episode([sample_from(env_factory(simple, operator)())])
    checkpoint = tmp_path / "checkpoints/latest.pt"
    manifest = storage.save(replay, checkpoint)
    chunk_paths(manifest, checkpoint)[0].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="训练数据校验失败"):
        storage.load(ReplayBuffer(), manifest, checkpoint)


def test_failed_checkpoint_write_preserves_previous_resume(
    simple, operator, tmp_path, monkeypatch
):
    import training.trainer as trainer_module

    factory = env_factory(simple, operator)
    config = tiny_config(tmp_path)
    trainer = AlphaZeroTrainer(config, factory)
    trainer.buffer.add_episode([sample_from(factory())])
    checkpoint = trainer.save_checkpoint()
    before = checkpoint.read_bytes()
    old_replay = deepcopy(trainer.buffer.state_dict())
    trainer.buffer.add_episode([sample_from(factory(), z=0.3)])

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(trainer_module, "atomic_torch_save", fail)
    with pytest.raises(OSError, match="disk full"):
        trainer.save_checkpoint()
    assert checkpoint.read_bytes() == before
    restored = AlphaZeroTrainer({**config, "resume_from": str(checkpoint)}, factory)
    same_tree(restored.buffer.state_dict(), old_replay)


def test_desktop_resolves_training_data_inside_workspace(tmp_path):
    from desktop.services.config_service import ConfigService

    config = ConfigService(tmp_path).prepare_runtime({})
    assert Path(config["training_data_dir"]) == tmp_path / "训练数据"


def test_desktop_surfaces_deleted_replay_message(qtbot):
    from desktop.workers.training_worker import TrainingWorker
    from training.events import TrainingEvent

    worker = TrainingWorker(None, {})
    with qtbot.waitSignal(worker.log) as signal:
        worker._receive(
            TrainingEvent("replay_data_missing", {"message": "历史训练数据已清理"})
        )
    assert signal.args == ["历史训练数据已清理"]


def test_large_replay_is_chunked_and_partial_deletion_discards_all(
    simple, operator, tmp_path
):
    storage = ReplayStorage(tiny_config(tmp_path))
    sample = sample_from(env_factory(simple, operator)())
    replay = ReplayBuffer(600)
    replay.add_episode([sample] * 600)
    checkpoint = tmp_path / "checkpoints/latest.pt"
    manifest = storage.save(replay, checkpoint)
    paths = chunk_paths(manifest, checkpoint)
    assert len(paths) == 3
    loaded = ReplayBuffer()
    assert storage.load(loaded, manifest, checkpoint) is False
    assert len(loaded) == 600
    paths[-1].unlink()
    assert storage.load(loaded, manifest, checkpoint) is True
    assert len(loaded) == 0


def test_missing_data_does_not_bypass_version_check(simple, operator, tmp_path):
    config = tiny_config(tmp_path)
    factory = env_factory(simple, operator)
    trainer = AlphaZeroTrainer(config, factory)
    trainer.buffer.add_episode([sample_from(factory())])
    checkpoint = trainer.save_checkpoint()
    shutil.rmtree(trainer.replay_storage.directory)
    payload = torch.load(checkpoint, weights_only=True)
    payload["skill_engine_version"] = -1
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="技能实现已更新"):
        AlphaZeroTrainer({**config, "resume_from": str(checkpoint)}, factory)
