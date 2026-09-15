"""A single-player UCT node; values always use the same agent perspective."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class MCTSNode:
    state_key: str
    state: Any = field(repr=False)
    parent: "MCTSNode | None" = field(default=None, repr=False)
    action: Any = None
    children: dict = field(default_factory=dict, repr=False)
    visit_count: int = 0
    total_value: float = 0.0
    terminal: bool = False
    untried_actions: list = field(default_factory=list, repr=False)
    depth: int = 0

    @property
    def mean_value(self):
        return self.total_value / self.visit_count if self.visit_count else 0.0

    @property
    def N(self):
        return self.visit_count

    @property
    def W(self):
        return self.total_value

    @property
    def Q(self):
        return self.mean_value


def backup(node: MCTSNode, value: float):
    """No alternating sign: this simulator has one decision-making player."""
    while node is not None:
        node.visit_count += 1
        node.total_value += value
        node = node.parent


Node = MCTSNode
