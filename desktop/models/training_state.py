"""Framework-neutral immutable training events passed across worker boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def immutable_snapshot(value):
    """Detach trainer-owned state; never expose a mutable payload to Qt views."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(k): immutable_snapshot(v) for k, v in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(immutable_snapshot(v) for v in value)
    if isinstance(value, set):
        return frozenset(immutable_snapshot(v) for v in value)
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return value
    raise TypeError(
        f"Training events require simple DTO values, got {type(value).__name__}"
    )


@dataclass(frozen=True)
class TrainingEvent:
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Training event kind must be a nonempty string")
        object.__setattr__(self, "payload", immutable_snapshot(self.payload))


def normalize_event(event):
    """Allow the future core to use its own event DTO without importing desktop."""
    if isinstance(event, TrainingEvent):
        return event
    if isinstance(event, Mapping):
        return TrainingEvent(str(event["kind"]), event.get("payload", {}))
    return TrainingEvent(str(event.kind), event.payload)
