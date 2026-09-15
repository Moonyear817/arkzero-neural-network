"""Behavioral checks for one-player UCT, independent of a stage solution."""

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from agents.mcts import MCTSConfig, MCTSNode, MCTSSearch, backup, evaluate_state
from agents.mcts_agent import PlainMCTSAgent


@dataclass(frozen=True)
class ToyAction:
    choice: int

    def to_dict(self):
        return {"choice": self.choice}


@dataclass
class ToyState:
    history: list = field(default_factory=list)
    game: object = field(default_factory=lambda: SimpleNamespace(current_time=0))


class ToyEnv:
    """A four-decision one-player search problem, with one successful leaf."""

    def clone_state(self, state):
        return copy.deepcopy(state)

    def state_key(self, state):
        return str(state.history)

    def is_terminal(self, state):
        return len(state.history) >= 4

    def legal_actions(self, state):
        return () if self.is_terminal(state) else (ToyAction(0), ToyAction(1))

    def step(self, state, action):
        assert action in self.legal_actions(state), "Search attempted an illegal action"
        child = self.clone_state(state)
        child.history.append(action.choice)
        child.game.current_time += 1
        return child

    def result(self, state):
        return SimpleNamespace(success=self.is_terminal(state) and all(state.history))


def toy_evaluation(env, state):
    if env.is_terminal(state):
        return 1.0 if env.result(state).success else -1.0
    return sum(state.history) / 4.0 - 0.5


def test_single_player_backup_has_no_sign_flip():
    root = MCTSNode("root", None)
    child = MCTSNode("child", None, parent=root)
    leaf = MCTSNode("leaf", None, parent=child)
    backup(leaf, 1.0)
    assert [(node.N, node.W, node.Q) for node in (root, child, leaf)] == [
        (1, 1.0, 1.0), (1, 1.0, 1.0), (1, 1.0, 1.0),
    ]
    backup(leaf, -1.0)
    assert all(node.N == 2 and node.W == 0 and node.Q == 0
               for node in (root, child, leaf))


def test_terminal_success_value():
    env = ToyEnv()
    assert evaluate_state(env, ToyState([1, 1, 1, 1])) == 1.0


def test_terminal_failure_value():
    env = ToyEnv()
    assert evaluate_state(env, ToyState([1, 0, 1, 1])) == -1.0


def test_mcts_visits_legality_and_child_isolation():
    env, state = ToyEnv(), ToyState()
    search = MCTSSearch(MCTSConfig(mcts_simulations=32, rollout_policy="random"),
                        evaluator=toy_evaluation)
    result = search.search(env, state)
    assert result.action in env.legal_actions(state)
    assert result.root.N == result.stats.root_visits == 32
    assert sum(child.N for child in result.root.children.values()) == 32
    assert set(result.root.children) == set(env.legal_actions(state))
    assert state.history == result.root.state.history == []
    children = list(result.root.children.values())
    children[0].state.history.append(99)
    assert state.history == result.root.state.history == []
    assert 99 not in children[1].state.history


def test_search_reproducibility_with_seed():
    env = ToyEnv()
    config = MCTSConfig(mcts_simulations=48, seed=963, rollout_policy="random")
    first = MCTSSearch(config, evaluator=toy_evaluation).search(env, ToyState())
    second = MCTSSearch(config, evaluator=toy_evaluation).search(env, ToyState())
    assert first.action == second.action
    assert first.discovered_plan == second.discovered_plan
    assert first.stats.actions == second.stats.actions
    assert first.stats.nodes == second.stats.nodes
    assert first.stats.rollout_steps == second.stats.rollout_steps


def test_real_stage_search_is_legal_and_preserves_parent(real_stage, squad):
    from arknights_sim.environment import ArknightsEnv

    env = ArknightsEnv(real_stage, squad, trace=True)
    state = env.reset(seed=101)
    original = env.state_key(state)
    config = MCTSConfig(mcts_simulations=3, rollout_limit=6, seed=101)
    result = MCTSSearch(config).search(env, state)
    assert result.action in env.legal_actions(state)
    assert result.root.N == 3
    assert env.state_key(state) == original == result.root.state_key
    assert state.trace and state.trace_enabled
    assert result.root.state.trace == () and not result.root.state.trace_enabled
    for action, child in result.root.children.items():
        assert action in env.legal_actions(state)
        assert child.state.game is not state.game
        assert env.state_key(child.state) == child.state_key


def test_tactical_policy_keeps_every_legal_action(real_stage, squad):
    import random
    from agents.mcts.rollout import TacticalRolloutPolicy
    from arknights_sim.environment import ArknightsEnv

    env = ArknightsEnv(real_stage, squad)
    state = env.reset()
    actions = env.legal_actions(state)
    policy = TacticalRolloutPolicy()
    assert all(weight > 0 for weight in policy.weights(env, state, actions))
    assert set(policy.order_actions(env, state, actions, random.Random(7))) == set(actions)


def test_discovered_solution_replay_and_plan_provenance():
    env, state = ToyEnv(), ToyState()
    agent = PlainMCTSAgent(MCTSConfig(mcts_simulations=64, rollout_policy="random"))
    history = []
    while not env.is_terminal(state):
        action = agent.select_action(env, state)
        assert action in env.legal_actions(state)
        history.append(action)
        state = env.step(state, action)
    assert env.result(state).success
    assert agent.stats_history[0].selection_reason == "found_solution"
    assert all(item.reused_discovered_plan for item in agent.stats_history[1:])
    replay_env, replay = ToyEnv(), ToyState()
    for action in history:
        replay = replay_env.step(replay, action)
    assert replay_env.state_key(replay) == env.state_key(state)
    assert replay_env.result(replay).success


def test_discovered_plan_is_discarded_on_state_divergence():
    env, state = ToyEnv(), ToyState()
    agent = PlainMCTSAgent(MCTSConfig(mcts_simulations=32, rollout_policy="random"))
    chosen = agent.select_action(env, state)
    assert chosen == ToyAction(1)
    altered = env.step(state, ToyAction(0))
    action = agent.select_action(env, altered)
    assert action in env.legal_actions(altered)
    assert not agent.last_stats.reused_discovered_plan
    assert agent.last_stats.simulations == 32


@pytest.mark.parametrize("options", [
    {"mcts_simulations": 0}, {"max_depth": 0}, {"rollout_limit": -1},
    {"exploration_constant": -1}, {"exploration_constant": float("nan")},
    {"rollout_policy": "scripted"},
])
def test_invalid_search_configuration(options):
    with pytest.raises(ValueError):
        MCTSConfig(**options)
