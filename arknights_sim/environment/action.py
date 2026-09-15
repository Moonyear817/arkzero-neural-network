"""Typed, immutable actions; no stage-dependent action IDs."""

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    DEPLOY = "DEPLOY"
    ACTIVATE_SKILL = "ACTIVATE_SKILL"
    RETREAT = "RETREAT"
    WAIT = "WAIT"
    PLACE_DEVICE = "PLACE_DEVICE"
    ACTIVATE_DEVICE = "ACTIVATE_DEVICE"
    REMOVE_DEVICE = "REMOVE_DEVICE"
    DEPLOY_SUMMON = "DEPLOY_SUMMON"
    RETREAT_SUMMON = "RETREAT_SUMMON"


class Direction(str, Enum):
    RIGHT = "RIGHT"
    UP = "UP"
    LEFT = "LEFT"
    DOWN = "DOWN"

    @property
    def index(self):
        return tuple(Direction).index(self)


@dataclass(frozen=True)
class Action:
    type: ActionType
    operator_id: str | None = None
    tile: tuple[int, int] | None = None
    direction: Direction | None = None
    wait_seconds: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "type", ActionType(self.type))
        if self.direction is not None:
            object.__setattr__(self, "direction", Direction(self.direction))
        if self.tile is not None:
            if len(self.tile) != 2 or any(type(v) is not int for v in self.tile):
                raise ValueError("Tile must contain two integers")
            object.__setattr__(self, "tile", tuple(self.tile))
        if self.wait_seconds is not None:
            if self.type != ActionType.WAIT or type(self.wait_seconds) not in (int, float) or self.wait_seconds not in (0.5, 1, 2, 5):
                raise ValueError("WAIT duration must be 0.5, 1, 2 or 5 seconds")
            object.__setattr__(self, "wait_seconds", float(self.wait_seconds))
        if self.type != ActionType.WAIT and (
            not isinstance(self.operator_id, str) or not self.operator_id
        ):
            raise ValueError("Operator id must be a nonempty string")
        if self.type == ActionType.PLACE_DEVICE:
            if self.tile is None or self.direction is not None:raise ValueError('PLACE_DEVICE needs device id and tile')
        elif self.type in (ActionType.DEPLOY, ActionType.DEPLOY_SUMMON):
            if not self.operator_id or self.tile is None or self.direction is None:
                raise ValueError("DEPLOY needs operator, tile and direction")
        elif self.type in (ActionType.ACTIVATE_SKILL, ActionType.RETREAT, ActionType.ACTIVATE_DEVICE, ActionType.REMOVE_DEVICE, ActionType.RETREAT_SUMMON):
            if (
                not self.operator_id
                or self.tile is not None
                or self.direction is not None
            ):
                raise ValueError("Unit action only needs operator")
        elif (
            self.operator_id is not None
            or self.tile is not None
            or self.direction is not None
        ):
            raise ValueError("WAIT has no arguments")

    def to_dict(self):
        result = {
            "type": self.type.value,
            "operator_id": self.operator_id,
            "tile": list(self.tile) if self.tile is not None else None,
            "direction": self.direction.value if self.direction is not None else None,
        }
        if self.wait_seconds is not None:
            result["wait_seconds"] = self.wait_seconds
        return result

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def __str__(self):
        if self.type == ActionType.WAIT and self.wait_seconds is not None:
            return f"WAIT {self.wait_seconds:g}s"
        if self.type == ActionType.PLACE_DEVICE:return f'PLACE_DEVICE {self.operator_id} @ {self.tile}'
        if self.type in (ActionType.DEPLOY, ActionType.DEPLOY_SUMMON):
            return f"{self.type.value} {self.operator_id} @ {self.tile} {self.direction.value}"
        return self.type.value + (f" {self.operator_id}" if self.operator_id else "")


WAIT = Action(ActionType.WAIT)
