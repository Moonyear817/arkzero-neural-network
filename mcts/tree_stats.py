"""Serializable neural-search diagnostics, including actual computation counts."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ActionStatistics:
    action: object
    neural_prior: float
    search_prior: float
    visit_count: int
    total_value: float
    mean_value: float
    policy: float
    visit_fraction: float

    def to_dict(self):
        return {
            **{key: value for key, value in asdict(self).items() if key != "action"},
            "action": self.action.to_dict()
            if hasattr(self.action, "to_dict")
            else str(self.action),
        }


@dataclass(frozen=True)
class SearchStatistics:
    root_state_key: str
    requested_simulations: int
    simulations: int
    nodes: int
    root_visits: int
    average_depth: float
    maximum_depth: int
    evaluator_calls: int
    cache_hits: int
    wall_time: float
    legal_action_count: int
    forced_action: bool
    training: bool
    root_value: float
    actions: tuple[ActionStatistics, ...] = ()
    active_action_count: int = 0
    expanded_action_count: int = 0
    planner_enabled: bool = True

    @property
    def latency(self):
        return self.wall_time

    @property
    def simulations_per_second(self):
        return self.simulations / self.wall_time if self.wall_time else 0.0

    @property
    def nodes_per_second(self):
        return self.nodes / self.wall_time if self.wall_time else 0.0

    def to_dict(self):
        return {
            **{key: value for key, value in asdict(self).items() if key != "actions"},
            "actions": [item.to_dict() for item in self.actions],
        }

    def format(self):
        lines = [
            f"PUCT | {'training' if self.training else 'evaluation'} | root value {self.root_value:+.4f}",
            (
                f"Simulations {self.simulations}/{self.requested_simulations} | nodes {self.nodes} | "
                f"depth {self.average_depth:.2f}/{self.maximum_depth} | {self.wall_time:.4f}s"
            ),
            "Action | Neural P | Search P | Visit fraction | Action pi | N | Q",
        ]
        lines.extend(
            f"{item.action} | {item.neural_prior:.5f} | {item.search_prior:.5f} | "
            f"{item.visit_fraction:.5f} | {item.policy:.5f} | {item.visit_count} | {item.mean_value:+.5f}"
            for item in sorted(self.actions, key=lambda row: -row.visit_count)
        )
        return "\n".join(lines)
