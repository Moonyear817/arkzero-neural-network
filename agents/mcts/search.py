"""Plain single-player Monte Carlo tree search with classical UCT."""

import math
import random
import time
from dataclasses import dataclass

from .heuristic import evaluate_state
from .node import MCTSNode, backup
from .rollout import RandomRolloutPolicy, TacticalRolloutPolicy, rollout
from .stats import ActionStatistics, SearchStatistics


@dataclass(frozen=True)
class MCTSConfig:
    mcts_simulations: int = 64
    exploration_constant: float = math.sqrt(2.0)
    max_depth: int = 64
    rollout_limit: int = 160
    seed: int = 12345
    rollout_policy: str = "tactical"
    reuse_successful_plan: bool = True

    def __post_init__(self):
        if self.mcts_simulations < 1 or self.max_depth < 1 or self.rollout_limit < 0:
            raise ValueError("Invalid MCTS simulation/depth/rollout budget")
        if not math.isfinite(self.exploration_constant) or self.exploration_constant < 0:
            raise ValueError("Invalid exploration constant")
        if self.rollout_policy not in {"random", "tactical"}:
            raise ValueError("rollout_policy must be random or tactical")

    @property
    def simulations(self):
        return self.mcts_simulations


@dataclass(frozen=True)
class DiscoveredStep:
    state_key: str
    action: object


@dataclass
class SearchResult:
    action: object
    root: MCTSNode
    stats: SearchStatistics
    discovered_plan: tuple[DiscoveredStep, ...] = ()


class MCTSSearch:
    def __init__(self, config=None, *, policy=None, evaluator=evaluate_state):
        self.config = config or MCTSConfig()
        self.rng = random.Random(self.config.seed)
        self.policy = policy or (
            TacticalRolloutPolicy() if self.config.rollout_policy == "tactical"
            else RandomRolloutPolicy()
        )
        self.evaluator = evaluator
        self.last_result = None

    def _node(self, env, state, parent=None, action=None):
        terminal = env.is_terminal(state)
        legal = () if terminal else env.legal_actions(state)
        return MCTSNode(
            state_key=env.state_key(state), state=state,
            parent=parent, action=action, terminal=terminal,
            untried_actions=self.policy.order_actions(env, state, legal, self.rng),
            depth=parent.depth + 1 if parent is not None else 0,
        )

    def _select_child(self, node):
        log_parent = math.log(max(1, node.visit_count))
        scores = [
            (
                child.mean_value + self.config.exploration_constant * math.sqrt(
                    log_parent / (1 + child.visit_count)
                ),
                child,
            )
            for child in node.children.values()
        ]
        best = max(score for score, _ in scores)
        return self.rng.choice([child for score, child in scores if score == best])

    def search(self, env, state):
        started = time.perf_counter()
        search_state = env.clone_state(state)
        if hasattr(search_state, "trace_enabled") and hasattr(search_state, "trace"):
            # Trace is observational metadata excluded from state_key; copying
            # complete combat logs through every hypothetical future is waste.
            search_state.trace = ()
            search_state.trace_enabled = False
        root = self._node(env, search_state)
        if root.terminal:
            raise ValueError("Cannot search a terminal state")
        if not root.untried_actions:
            raise ValueError("A nonterminal state has no legal actions")
        stats = SearchStatistics(
            root_state_key=root.state_key,
            time=getattr(getattr(state, "game", None), "current_time", 0.0),
        )
        legal_counts = [len(root.untried_actions)]
        depth_sum = 0
        discovered_plan = ()
        for _ in range(self.config.mcts_simulations):
            node = root
            while (
                not node.terminal
                and not node.untried_actions
                and node.children
                and node.depth < self.config.max_depth
            ):
                node = self._select_child(node)
            if (
                not node.terminal and node.untried_actions
                and node.depth < self.config.max_depth
            ):
                action = node.untried_actions.pop()
                child_state = env.step(node.state, action)
                child = self._node(env, child_state, parent=node, action=action)
                node.children[action] = child
                node = child
                stats.nodes += 1
                if not child.terminal:
                    legal_counts.append(len(child.untried_actions))
            outcome = rollout(
                env, node.state, self.policy, self.rng,
                self.config.rollout_limit, self.evaluator,
            )
            backup(node, outcome.value)
            stats.simulations += 1
            stats.rollout_steps += len(outcome.actions)
            depth_sum += node.depth
            stats.maximum_depth = max(stats.maximum_depth, node.depth)
            legal_counts.extend(outcome.legal_counts)
            if outcome.success:
                stats.successful_rollouts += 1
                candidate_length = node.depth + len(outcome.actions)
                if discovered_plan and candidate_length >= len(discovered_plan):
                    continue
                path = []
                cursor = node
                while cursor.parent is not None:
                    path.append(DiscoveredStep(cursor.parent.state_key, cursor.action))
                    cursor = cursor.parent
                path.reverse()
                # Hash only successful rollout states: unsuccessful samples do
                # not need persistent replay keys or retained trajectory data.
                path.extend(
                    DiscoveredStep(env.state_key(before), action)
                    for before, action in zip(outcome.states, outcome.actions)
                )
                if not discovered_plan or len(path) < len(discovered_plan):
                    discovered_plan = tuple(path)
        stats.root_visits = root.visit_count
        stats.average_depth = depth_sum / stats.simulations
        stats.average_legal_actions = sum(legal_counts) / len(legal_counts)
        stats.maximum_legal_actions = max(legal_counts)
        # Proven terminal continuations are useful in single-agent planning.
        # This plan consists solely of transitions discovered by this search.
        if discovered_plan and self.config.reuse_successful_plan:
            selected = discovered_plan[0].action
            stats.selection_reason = "found_solution"
        else:
            best = max((child.visit_count, child.mean_value)
                       for child in root.children.values())
            selected = self.rng.choice([
                child.action for child in root.children.values()
                if (child.visit_count, child.mean_value) == best
            ])
        stats.selected_action = selected
        stats.actions = tuple(sorted(
            (ActionStatistics(
                action, root.children[action].visit_count if action in root.children else 0,
                root.children[action].total_value if action in root.children else 0.0,
                root.children[action].mean_value if action in root.children else 0.0,
            ) for action in env.legal_actions(state)),
            key=lambda item: (-item.visit_count, -item.mean_value, str(item.action)),
        ))
        stats.wall_time = time.perf_counter() - started
        result = SearchResult(selected, root, stats, discovered_plan)
        self.last_result = result
        return result


MCTS = MCTSSearch
