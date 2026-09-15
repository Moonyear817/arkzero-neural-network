"""Serializable root search statistics and discovered-plan provenance."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionStatistics:
    action: Any
    visit_count: int
    total_value: float
    mean_value: float

    def to_dict(self):
        return {
            "action": self.action.to_dict(),
            "N": self.visit_count,
            "W": self.total_value,
            "Q": self.mean_value,
        }


@dataclass
class SearchStatistics:
    root_state_key: str
    time: float
    simulations: int = 0
    root_visits: int = 0
    nodes: int = 1
    average_depth: float = 0.0
    maximum_depth: int = 0
    rollout_steps: int = 0
    average_legal_actions: float = 0.0
    maximum_legal_actions: int = 0
    wall_time: float = 0.0
    successful_rollouts: int = 0
    selected_action: Any = None
    actions: tuple[ActionStatistics, ...] = ()
    selection_reason: str = "most_visited"
    reused_discovered_plan: bool = False

    @property
    def simulations_per_second(self):
        return self.simulations / self.wall_time if self.wall_time else 0.0

    @property
    def nodes_per_second(self):
        return self.nodes / self.wall_time if self.wall_time else 0.0

    def to_dict(self):
        result = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"actions", "selected_action"}
        }
        result["selected_action"] = (
            self.selected_action.to_dict() if self.selected_action is not None else None
        )
        result["actions"] = [item.to_dict() for item in self.actions]
        result["simulations_per_second"] = self.simulations_per_second
        result["nodes_per_second"] = self.nodes_per_second
        return result

    def format(self):
        lines = [
            f"Time = {self.time:.3f}",
            f"Root visits = {self.root_visits}; nodes = {self.nodes}; "
            f"latency = {self.wall_time:.4f}s",
            "Actions:",
        ]
        lines.extend(
            f"  {item.action}  N={item.visit_count} Q={item.mean_value:.5f}"
            for item in self.actions
        )
        lines.append(f"Selected: {self.selected_action} ({self.selection_reason})")
        return "\n".join(lines)
