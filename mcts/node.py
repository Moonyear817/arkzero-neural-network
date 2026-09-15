"""Single-player PUCT state nodes and action edges, independent of any UI."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False)
class PUCTEdge:
    action: Any
    prior: float
    visit_count: int = 0
    total_value: float = 0.0
    child_state_key: str | None = None
    child: "PUCTNode | None" = field(default=None, repr=False)
    reward: float = 0.0

    @property
    def P(self):
        return self.prior

    @property
    def N(self):
        return self.visit_count

    @property
    def W(self):
        return self.total_value

    @property
    def Q(self):
        return self.total_value / self.visit_count if self.visit_count else 0.0


@dataclass(eq=False)
class PUCTNode:
    state_key: str
    state: Any = field(repr=False)
    terminal: bool = False
    depth: int = 0
    edges: dict[Any, PUCTEdge] = field(default_factory=dict, repr=False)
    visit_count: int = 0
    total_value: float = 0.0
    network_value: float | None = None
    neural_policy: tuple[float, ...] = ()
    expanded: bool = False
    expansion_order: tuple = ()
    active_actions: tuple = ()
    group_priors: dict = field(default_factory=dict, repr=False)

    @property
    def N(self):
        return self.visit_count

    @property
    def W(self):
        return self.total_value

    @property
    def Q(self):
        return self.total_value / self.visit_count if self.visit_count else 0.0

    @property
    def children(self):
        """Only instantiated children; all legal actions remain in ``edges``."""
        return {
            action: edge.child
            for action, edge in self.edges.items()
            if edge.child is not None
        }


def backup(nodes, edges, value: float):
    """Back up remaining return plus each transition, without a sign flip."""
    nodes[-1].visit_count += 1
    nodes[-1].total_value += value
    for node, edge in reversed(list(zip(nodes[:-1], edges))):
        value = edge.reward + value
        edge.visit_count += 1
        edge.total_value += value
        node.visit_count += 1
        node.total_value += value


Node = PUCTNode
Edge = PUCTEdge
