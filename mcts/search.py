"""Neural single-player PUCT. No rollout policy or precomputed battle plan."""

import math
import random
import time
from dataclasses import dataclass
from typing import Protocol

from .node import PUCTEdge, PUCTNode, backup
from .puct import action_group, sample_policy, select_edge, visit_policy
from .tree_stats import ActionStatistics, SearchStatistics


class Evaluator(Protocol):
    def evaluate(
        self, env, state, ordered_legal_actions
    ) -> tuple[tuple[float, ...], float]: ...


@dataclass(frozen=True, init=False)
class PUCTConfig:
    simulations: int = 64
    c_puct: float = 1.5
    max_depth: int = 128
    seed: int = 12345
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    prior_uniform_mix: float = 0.05
    temperature: float = 1.0
    skip_forced_actions: bool = True
    top_k_actions: int = 8
    progressive_widening: bool = True
    widening_coefficient: float = 2.0
    widening_exponent: float = 0.5
    enabled: bool = True

    def __init__(
        self,
        simulations=64,
        c_puct=1.5,
        max_depth=128,
        seed=12345,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        prior_uniform_mix=0.05,
        temperature=1.0,
        skip_forced_actions=True,
        top_k_actions=8,
        progressive_widening=True,
        widening_coefficient=2.0,
        widening_exponent=0.5,
        enabled=True,
        *,
        mcts_simulations=None,
    ):
        if mcts_simulations is not None:
            if simulations != 64 and simulations != mcts_simulations:
                raise ValueError("Conflicting simulations and mcts_simulations budgets")
            simulations = mcts_simulations
        values = locals().copy()
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, values[name])
        self.__post_init__()

    def __post_init__(self):
        if type(self.simulations) is not int or self.simulations < 1:
            raise ValueError("simulations must be a positive integer")
        if type(self.max_depth) is not int or self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("c_puct", "temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be finite and positive")
        for name in ("dirichlet_epsilon", "prior_uniform_mix"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if type(self.skip_forced_actions) is not bool:
            raise ValueError("skip_forced_actions must be a boolean")
        if type(self.top_k_actions) is not int or self.top_k_actions < 1:
            raise ValueError("top_k_actions must be a positive integer")
        for key in ('progressive_widening', 'enabled'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be a boolean')
        if not math.isfinite(self.widening_coefficient) or self.widening_coefficient <= 0:
            raise ValueError('widening_coefficient must be positive and finite')
        if not math.isfinite(self.widening_exponent) or not 0 < self.widening_exponent <= 1:
            raise ValueError('widening_exponent must be in (0, 1]')

    @property
    def mcts_simulations(self):
        return self.simulations


@dataclass(frozen=True)
class SearchResult:
    actions: tuple
    visit_counts: tuple[int, ...]
    policy: tuple[float, ...]
    selected_action: object | None
    root: PUCTNode
    stats: SearchStatistics
    neural_policy: tuple[float, ...]
    root_value: float

    @property
    def action(self):
        return self.selected_action

    @property
    def visit_policy(self):
        """Raw visit fractions for inspection, distinct from temperature choice."""
        total = sum(self.visit_counts)
        if total:
            return tuple(count / total for count in self.visit_counts)
        return self.policy


def terminal_value(env, state) -> float:
    if not env.is_terminal(state):
        raise ValueError("Exact terminal value requires a terminal state")
    if hasattr(env, 'terminal_reward'):
        return env.terminal_reward(state)
    result = env.result(state)
    success = result["success"] if isinstance(result, dict) else result.success
    return 1.0 if success else -1.0


class PUCTSearch:
    """One independent tree per call; the seeded RNG advances across searches.

    Cached predictions live only for this search. State keys and the exact
    ordered legal-action tuple both participate, preventing policy misalignment.
    The simulator retains its own deterministic RNG, untouched by this RNG.
    """

    def __init__(self, evaluator: Evaluator, config: PUCTConfig | None = None):
        self.evaluator = evaluator
        self.config = config or PUCTConfig()
        self.rng = random.Random(self.config.seed)
        self.last_result = None

    def search(self, env, state, training=False) -> SearchResult:
        started = time.perf_counter()
        private_state = env.clone_state(state)
        if hasattr(private_state, "trace_enabled") and hasattr(private_state, "trace"):
            # Observational data is excluded from environment state keys. Keep
            # it out of private search branches, never alter the caller's trace.
            private_state.trace = ()
            private_state.trace_enabled = False
        root = PUCTNode(
            env.state_key(private_state), private_state, env.is_terminal(private_state)
        )
        cache = {}
        evaluator_calls = 0
        cache_hits = 0
        nodes_created = 1

        def prediction(node, actions):
            nonlocal evaluator_calls, cache_hits
            cache_key = (node.state_key, actions)
            if cache_key in cache:
                cache_hits += 1
                return cache[cache_key]
            priors, value = self.evaluator.evaluate(env, node.state, actions)
            priors = tuple(float(prior) for prior in priors)
            value = float(value)
            if len(priors) != len(actions):
                raise ValueError(
                    "Evaluator priors must match the ordered legal actions"
                )
            if any(not math.isfinite(prior) or prior < 0 for prior in priors):
                raise ValueError("Evaluator priors must be finite and nonnegative")
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("Evaluator value must be finite and in [-1, 1]")
            total = math.fsum(priors)
            if not math.isfinite(total) or total <= 0:
                raise ValueError("Evaluator priors must have positive finite mass")
            normalized = tuple(prior / total for prior in priors)
            cache[cache_key] = (normalized, value)
            evaluator_calls += 1
            return normalized, value

        def expand(node):
            if node.terminal:
                node.network_value = terminal_value(env, node.state)
                node.expanded = True
                return node.network_value
            actions = tuple(env.legal_actions(node.state))
            if not actions:
                raise ValueError("Nonterminal environment state has no legal actions")
            if len(set(actions)) != len(actions):
                raise ValueError("Environment returned duplicate legal actions")
            neural, value = prediction(node, actions)
            mix = self.config.prior_uniform_mix
            priors = tuple((1 - mix) * p + mix / len(actions) for p in neural)
            # Noise is restricted to the actual root of a training search.
            if node is root and training and self.config.dirichlet_epsilon > 0:
                noise = tuple(
                    self.rng.gammavariate(self.config.dirichlet_alpha, 1.0)
                    for _ in actions
                )
                noise_mass = math.fsum(noise)
                if noise_mass <= 0 or not math.isfinite(noise_mass):
                    raise ValueError("Dirichlet sampling underflow; increase alpha")
                epsilon = self.config.dirichlet_epsilon
                priors = tuple(
                    (1 - epsilon) * p + epsilon * sample / noise_mass
                    for p, sample in zip(priors, noise)
                )
            node.edges = {
                action: PUCTEdge(action, prior)
                for action, prior in zip(actions, priors)
            }
            # First represent each legal action type with its strongest leaf.
            # A type with hundreds of placements must not be excluded simply
            # because a WAIT type spreads equal mass over only five choices.
            # Type ranking uses summed policy mass; both ties preserve order.
            groups = {}
            for action in actions:
                groups.setdefault(action_group(action), []).append(action)
            node.group_priors = {key: math.fsum(node.edges[a].P for a in members)
                                 for key, members in groups.items()}
            ranked_groups = sorted(groups, key=lambda key: -node.group_priors[key])
            representatives = [max(groups[key], key=lambda a: node.edges[a].P)
                               for key in ranked_groups]
            remaining = sorted((a for a in actions if a not in representatives),
                               key=lambda a: -node.edges[a].P)
            node.expansion_order = tuple(representatives + remaining)
            activate_candidates(node)
            node.neural_policy = neural
            node.network_value = value
            node.expanded = True
            return value

        def activate_candidates(node):
            width = self.config.top_k_actions
            if self.config.progressive_widening:
                width += int(self.config.widening_coefficient * node.N ** self.config.widening_exponent)
            node.active_actions = node.expansion_order[:min(len(node.edges), width)]

        root_value = expand(root)
        forced = bool(
            not root.terminal
            and len(root.edges) == 1
            and self.config.skip_forced_actions
        )
        budget = 0 if root.terminal or forced or not self.config.enabled else self.config.mcts_simulations
        depth_sum = 0
        maximum_depth = 0
        for _ in range(budget):
            node = root
            path_nodes = [root]
            path_edges = []
            while True:
                if node.terminal:
                    value = terminal_value(env, node.state)
                    break
                if node.depth >= self.config.max_depth:
                    value = node.network_value
                    break
                activate_candidates(node)
                edge = select_edge(node, self.config.c_puct, self.rng)
                path_edges.append(edge)
                created = edge.child is None
                if created:
                    child_state = env.step(node.state, edge.action)
                    edge.reward = env.transition_reward(node.state, child_state) if hasattr(env, 'transition_reward') else 0.0
                    edge.child_state_key = env.state_key(child_state)
                    edge.child = PUCTNode(
                        edge.child_state_key,
                        child_state,
                        env.is_terminal(child_state),
                        node.depth + 1,
                    )
                    nodes_created += 1
                node = edge.child
                path_nodes.append(node)
                if created:
                    value = expand(node)
                    # A forced wait is not a tactical choice. Continue through
                    # it within the same simulation, bounded by max_depth.
                    if node.terminal or len(node.edges) != 1 or not self.config.skip_forced_actions:
                        break
            backup(path_nodes, path_edges, value)
            depth_sum += node.depth
            maximum_depth = max(maximum_depth, node.depth)

        actions = tuple(root.edges)
        counts = tuple(edge.N for edge in root.edges.values())
        temperature = self.config.temperature if training else 0.0
        if actions and not budget and not forced:
            if temperature == 0:
                largest = max(root.neural_policy)
                winner = self.rng.choice([i for i, p in enumerate(root.neural_policy) if p == largest])
                policy = tuple(float(i == winner) for i in range(len(actions)))
            else:
                logs = [math.log(p) / temperature if p > 0 else -math.inf for p in root.neural_policy]
                maximum = max(logs)
                weights = [math.exp(v - maximum) for v in logs]
                policy = tuple(v / math.fsum(weights) for v in weights)
        else:
            policy = visit_policy(counts, temperature, self.rng) if actions else ()
        selected = sample_policy(actions, policy, self.rng) if actions else None
        total_visits = sum(counts)
        action_stats = tuple(
            ActionStatistics(
                action,
                neural,
                edge.P,
                edge.N,
                edge.W,
                edge.Q,
                probability,
                edge.N / total_visits if total_visits else probability,
            )
            for action, neural, edge, probability in zip(
                actions, root.neural_policy, root.edges.values(), policy
            )
        )
        stats = SearchStatistics(
            root_state_key=root.state_key,
            requested_simulations=self.config.mcts_simulations,
            simulations=budget,
            nodes=nodes_created,
            root_visits=root.N,
            average_depth=depth_sum / budget if budget else 0.0,
            maximum_depth=maximum_depth,
            evaluator_calls=evaluator_calls,
            cache_hits=cache_hits,
            wall_time=time.perf_counter() - started,
            legal_action_count=len(actions),
            forced_action=forced,
            training=bool(training),
            root_value=root_value,
            actions=action_stats,
            active_action_count=len(root.active_actions),
            expanded_action_count=sum(edge.child is not None for edge in root.edges.values()),
            planner_enabled=self.config.enabled,
        )
        result = SearchResult(
            actions,
            counts,
            policy,
            selected,
            root,
            stats,
            root.neural_policy,
            root_value,
        )
        self.last_result = result
        return result
