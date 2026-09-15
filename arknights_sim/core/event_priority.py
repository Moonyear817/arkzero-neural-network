"""Explicit V0.2 ordering policy. All priorities are ASSUMED, not official rules.

The simulator has two queue drains per boundary: the existing queue before the
combat tick, then newly generated zero-windup hits after it. EVENT_PRIORITY
orders events *within* a drain; it does not move a new hit before its cause.
"""

from dataclasses import fields, is_dataclass
from enum import Enum
import math
from types import MappingProxyType


EVENT_PRIORITY = MappingProxyType(
    {
        "MODIFIER_BOUNDARY": 10,
        "SKILL_END": 20,
        "SPAWN": 30,
        "HIT": 40,
        "AREA_HIT": 41,
        "SELF_DAMAGE": 42,
        "BURN_TICK": 43,
        "DEVICE_PULSE": 44,
        "TALENT_EVENT": 45,
    }
)

# Executed by Simulator._advance(), run_until(), _tick(), and _finish().
# Movement/DP/SP stay continuous; death is synchronous inside movement/HIT.
# External deployment, retreat and skill activation occur at public boundaries.
SIMULATION_PHASES = (
    "CONTINUOUS_UPDATES",  # DP, SP, then enemy movement (including escapes).
    "QUEUED_EVENTS",  # EVENT_PRIORITY, with deaths resolved in their HIT.
    "BLOCKING_UPDATE",  # Capacity release, then new blocking, by unit id.
    "OPERATOR_ATTACK",  # Attack starts, by operator id.
    "ENEMY_ATTACK",  # Attack starts, by enemy id.
    "ZERO_WINDUP_EVENTS",  # Newly queued events due at this timestamp.
    "TERMINAL_CHECK",
    "EXTERNAL_ACTION",  # Ordered by the caller's action sequence.
)


def _semantic_value(value):
    """Comparable, address-free values for deterministic queue tie breaking.

    Unknown objects are rejected instead of falling back to a repr containing
    a process-specific address. Current event payloads contain only frozen
    dataclasses and primitive values.
    """
    if value is None:
        return ("none",)
    if isinstance(value, Enum):
        return ("enum", type(value).__qualname__, _semantic_value(value.value))
    if isinstance(value, bool):
        return ("bool", int(value))
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Non-finite event payload")
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "dataclass",
            type(value).__qualname__,
            tuple((f.name, _semantic_value(getattr(value, f.name))) for f in fields(value)),
        )
    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_semantic_value(v) for v in value))
    if isinstance(value, dict):
        return (
            "mapping",
            tuple(sorted((_semantic_value(k), _semantic_value(v)) for k, v in value.items())),
        )
    raise TypeError(f"Unsupported deterministic event payload: {type(value).__name__}")


def event_order_key(kind, payload):
    """Return priority and semantic tie key, independently of push order."""
    if kind not in EVENT_PRIORITY:
        raise ValueError(f"Event kind requires an explicit priority: {kind}")
    if kind == "SPAWN":
        spawn = payload[0]
        # Give wave/fragment metadata precedence over enemy naming. The full
        # payload suffix distinguishes every remaining semantic difference.
        primary = (spawn.wave, spawn.fragment, spawn.route_index, spawn.enemy_id)
    elif kind == "HIT":
        hit = payload[0]
        primary = (hit.attacker, hit.target, hit.generation, hit.target_generation)
    else:
        primary = payload  # Unit id, plus deployment generation for SKILL_END.
    return EVENT_PRIORITY[kind], (_semantic_value(primary), _semantic_value(payload))
