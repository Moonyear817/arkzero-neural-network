"""Lazy CPU replay with explicit dynamic action identities and masked batches."""

import json
import math
import os
import random
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
from arknights_sim.environment.rewards import REWARD_VERSION
from .schema import OBSERVATION_VERSION, EVENT_GRAPH_VERSION


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            name = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        name.replace(path)
    finally:
        if name is not None and name.exists():
            name.unlink()
    return path


@dataclass(frozen=True)
class TrainingSample:
    state_features: dict
    actions: tuple[str, ...]
    action_features: torch.Tensor
    policy: torch.Tensor
    z: float

    def __post_init__(self):
        features = {k: v.detach().cpu().clone() for k, v in self.state_features.items()}
        actions = tuple(
            json.dumps(a, sort_keys=True, separators=(",", ":"))
            if isinstance(a, dict)
            else str(a)
            for a in self.actions
        )
        for identity in actions:
            if not isinstance(json.loads(identity), dict):
                raise TypeError("Action identities must be JSON objects")
        action_features = self.action_features.detach().cpu().float().clone()
        policy = self.policy.detach().cpu().float().clone()
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("Samples require distinct ordered legal actions")
        if (
            action_features.ndim != 2
            or action_features.shape[0] != len(actions)
            or policy.shape != (len(actions),)
        ):
            raise ValueError(
                "Policy, features and ordered action identities do not align"
            )
        if (
            not torch.isfinite(policy).all()
            or (policy < 0).any()
            or not torch.isclose(policy.sum(), torch.tensor(1.0), atol=1e-5)
        ):
            raise ValueError("Policy must be a finite probability distribution")
        if not math.isfinite(self.z) or not -1.0000001 <= self.z <= 1.0000001:
            raise ValueError("Remaining-return labels must be finite and in [-1, 1]")
        object.__setattr__(self, "state_features", features)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "action_features", action_features)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "z", float(self.z))

    @property
    def action_dicts(self):
        return tuple(json.loads(a) for a in self.actions)

    def to_dict(self):
        return {
            "state_features": self.state_features,
            "actions": list(self.actions),
            "action_features": self.action_features,
            "policy": self.policy,
            "z": self.z,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


class ReplayBuffer:
    def __init__(self, capacity=50000, seed=12345):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.samples = deque(maxlen=capacity)
        # Parallel disk references; new samples are persisted on the next save.
        self._storage_refs = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.samples)

    def add_episode(self, samples):
        samples = tuple(samples)
        if any(not isinstance(sample, TrainingSample) for sample in samples):
            raise TypeError("Replay accepts TrainingSample values")
        self.samples.extend(samples)
        self._storage_refs.extend(None for _ in samples)

    def sample(self, batch_size):
        if not self.samples or batch_size <= 0:
            raise ValueError("Cannot sample an empty replay or nonpositive batch")
        count = min(batch_size, len(self.samples))
        choices = [s for s in self.samples if len(s.actions) > 1]
        forced = [s for s in self.samples if len(s.actions) == 1]
        choice_count = min(len(choices), max(math.ceil(count * 0.8), count - len(forced)))
        selected = self.rng.sample(choices, choice_count) + self.rng.sample(forced, count - choice_count)
        self.rng.shuffle(selected)
        return selected

    def state_dict(self, *, include_samples=True):
        state = {
            "version": 1,
            "reward_version": REWARD_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "event_graph_version": EVENT_GRAPH_VERSION,
            "capacity": self.capacity,
            "rng_state": self.rng.getstate(),
        }
        if include_samples:
            state["samples"] = [s.to_dict() for s in self.samples]
        return state

    def load_state_dict(self, state):
        if state["version"] != 1:
            raise ValueError("Unsupported replay format")
        if state.get('reward_version') != REWARD_VERSION:
            raise ValueError('Replay reward version mismatch; start with an empty replay')
        if (state.get('observation_version') != OBSERVATION_VERSION
                or state.get('event_graph_version') != EVENT_GRAPH_VERSION):
            raise ValueError('Replay observation/event graph version mismatch; start with an empty replay')
        self.capacity = state["capacity"]
        self.samples = deque(
            (TrainingSample.from_dict(s) for s in state["samples"]),
            maxlen=self.capacity,
        )
        self._storage_refs = deque((None for _ in self.samples), maxlen=self.capacity)
        self.rng.setstate(state["rng_state"])

    def save(self, path):
        return atomic_torch_save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        buffer = cls(state["capacity"])
        buffer.load_state_dict(state)
        return buffer


def collate_samples(samples, device="cpu"):
    from network import collate_states

    if not samples:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(s.actions) for s in samples)
    action_features = torch.zeros(
        len(samples), width, samples[0].action_features.shape[1]
    )
    policy = torch.zeros(len(samples), width)
    mask = torch.zeros(len(samples), width, dtype=torch.bool)
    for i, sample in enumerate(samples):
        count = len(sample.actions)
        action_features[i, :count] = sample.action_features
        policy[i, :count] = sample.policy
        mask[i, :count] = True
    states = {
        k: v.to(device)
        for k, v in collate_states([s.state_features for s in samples]).items()
    }
    return {
        "state_features": states,
        "action_features": action_features.to(device),
        "policy": policy.to(device),
        "mask": mask.to(device),
        "z": torch.tensor([s.z for s in samples], device=device),
    }


def masked_policy_loss(logits, target, mask):
    if not mask.any(dim=-1).all():
        raise ValueError("Every policy row needs a legal action")
    log_policy = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    safe_log = torch.where(mask, log_policy, torch.zeros_like(log_policy))
    per_row = -(target * safe_log).sum(dim=-1)
    choices = mask.sum(dim=-1) > 1
    # Forced rows contribute neither loss nor dilution of real decisions.
    return per_row[choices].mean() if choices.any() else per_row.sum() * 0
