"""Neural PUCT invariants, separate from the historical UCT baselines."""

import copy
import dataclasses
import json
import math
import random
from types import SimpleNamespace

import pytest

from arknights_sim.environment import ArknightsEnv
from mcts import PUCTConfig, PUCTSearch, backup
from mcts.node import PUCTEdge, PUCTNode
from mcts.puct import puct_score, visit_policy


class ToyEnvironment:
    def clone_state(self, state):
        return copy.deepcopy(state)

    def state_key(self, state):
        return str(state)

    def legal_actions(self, state):
        return () if self.is_terminal(state) else ("win", "lose")

    def step(self, state, action):
        assert action in self.legal_actions(state)
        return action

    def is_terminal(self, state):
        return state in ("win", "lose")

    def result(self, state):
        return SimpleNamespace(success=state == "win")


class ConstantEvaluator:
    def __init__(self, value=0.3, prior=None):
        self.calls = []
        self.value = value
        self.prior = prior

    def evaluate(self, env, state, actions):
        assert not env.is_terminal(state), "Terminal states must bypass the model"
        self.calls.append((env.state_key(state), actions))
        return (self.prior or tuple(1 / len(actions) for _ in actions), self.value)


def test_terminal_states_bypass_network_and_return_exact_values():
    env = ToyEnvironment()
    evaluator = ConstantEvaluator()
    search = PUCTSearch(evaluator)
    for state, expected in (("win", 1.0), ("lose", -1.0)):
        result = search.search(env, state)
        assert result.root_value == expected
        assert result.root.terminal
        assert result.selected_action is None
        assert result.actions == result.policy == result.visit_counts == ()
        assert result.stats.simulations == 0
    assert not evaluator.calls


def test_puct_visits_backup_terminal_values_and_all_legal_edges():
    env = ToyEnvironment()
    result = PUCTSearch(
        ConstantEvaluator(-0.9), PUCTConfig(simulations=24, c_puct=2)
    ).search(env, "root")
    assert result.actions == env.legal_actions("root")
    assert result.root.N == sum(result.visit_counts) == result.stats.simulations == 24
    assert result.selected_action == "win"
    win, lose = result.root.edges["win"], result.root.edges["lose"]
    assert win.N > lose.N > 0
    assert win.Q == 1 and lose.Q == -1
    assert win.child.N == win.N
    assert win.child_state_key == "win"
    assert result.root.W == sum(edge.W for edge in result.root.edges.values())
    assert (
        result.root_value == -0.9
    )  # Initial network value remains separately visible.
    assert result.visit_policy == tuple(n / 24 for n in result.visit_counts)


def test_single_player_backup_never_alternates_sign():
    root, child, leaf = (PUCTNode(str(i), None) for i in range(3))
    edges = [PUCTEdge("a", 1), PUCTEdge("b", 1)]
    backup([root, child, leaf], edges, 0.75)
    assert [node.Q for node in (root, child, leaf)] == [0.75, 0.75, 0.75]
    assert [edge.Q for edge in edges] == [0.75, 0.75]
    backup([root, child, leaf], edges, -1)
    assert [node.W for node in (root, child, leaf)] == [-0.25] * 3
    assert [edge.W for edge in edges] == [-0.25] * 2


class InfiniteEnvironment(ToyEnvironment):
    def legal_actions(self, state):
        return ("left", "right")

    def is_terminal(self, state):
        return False

    def step(self, state, action):
        assert action in self.legal_actions(state)
        return state + action[0]


def test_depth_is_bounded_and_network_leaf_value_has_same_perspective():
    env = InfiniteEnvironment()
    result = PUCTSearch(
        ConstantEvaluator(0.4), PUCTConfig(simulations=24, max_depth=3)
    ).search(env, "")
    assert result.stats.maximum_depth == 3
    assert result.root.Q == pytest.approx(0.4)
    assert all(
        edge.Q == pytest.approx(0.4) for edge in result.root.edges.values() if edge.N
    )
    assert result.stats.nodes <= 1 + result.stats.simulations


def test_formula_uses_prior_and_parent_visits():
    edge = PUCTEdge("a", 0.4, visit_count=3, total_value=1.5)
    assert puct_score(edge, 16, 2) == pytest.approx(0.5 + 2 * 0.4 * 4 / 4)


def test_root_dirichlet_noise_only_in_training_and_only_at_root():
    env = InfiniteEnvironment()
    config = PUCTConfig(simulations=8, dirichlet_epsilon=1, prior_uniform_mix=0.1)
    evaluator = ConstantEvaluator(prior=(0.9, 0.1))
    evaluation = PUCTSearch(evaluator, config).search(env, "", training=False)
    training = PUCTSearch(evaluator, config).search(env, "", training=True)
    expected = (0.86, 0.14)
    assert tuple(edge.P for edge in evaluation.root.edges.values()) == pytest.approx(
        expected
    )
    assert tuple(edge.P for edge in training.root.edges.values()) != pytest.approx(
        expected
    )
    assert training.neural_policy == evaluation.neural_policy == (0.9, 0.1)
    for child in training.root.children.values():
        assert tuple(edge.P for edge in child.edges.values()) == pytest.approx(expected)
    assert sum(edge.P for edge in training.root.edges.values()) == pytest.approx(1)


def test_eval_ignores_temperature_and_chooses_most_visits():
    result = PUCTSearch(
        ConstantEvaluator(), PUCTConfig(simulations=16, temperature=100)
    ).search(ToyEnvironment(), "root")
    index = result.actions.index(result.selected_action)
    assert result.visit_counts[index] == max(result.visit_counts)
    assert sum(result.policy) == 1
    assert result.policy[index] == 1
    assert result.stats.actions[index].visit_fraction < 1


def test_temperature_policy_comes_only_from_visit_counts():
    assert visit_policy((1, 2, 0), 1, random.Random(0)) == pytest.approx(
        (1 / 3, 2 / 3, 0)
    )
    assert visit_policy((1, 2, 0), 0.5, random.Random(0)) == pytest.approx(
        (0.2, 0.8, 0)
    )
    assert visit_policy((1, 2), 1e-20, random.Random(0)) == (0.0, 1.0)
    assert visit_policy((4, 4), 0, random.Random(1)) == visit_policy(
        (4, 4), 0, random.Random(1)
    )
    with pytest.raises(ValueError):
        visit_policy((0, 0), 1, random.Random(0))


def test_training_policy_is_count_normalized_at_temperature_one():
    result = PUCTSearch(
        ConstantEvaluator(), PUCTConfig(simulations=16, temperature=1)
    ).search(ToyEnvironment(), "root", training=True)
    assert result.policy == pytest.approx(tuple(n / 16 for n in result.visit_counts))


def test_seed_reproducibility_includes_noise_visits_and_selection():
    config = PUCTConfig(simulations=16, seed=91)
    results = [
        PUCTSearch(ConstantEvaluator(), config).search(
            InfiniteEnvironment(), "", training=True
        )
        for _ in range(2)
    ]
    a, b = results
    assert a.selected_action == b.selected_action
    assert a.visit_counts == b.visit_counts
    assert a.policy == b.policy
    assert [edge.P for edge in a.root.edges.values()] == [
        edge.P for edge in b.root.edges.values()
    ]


class ForcedEnvironment(InfiniteEnvironment):
    def legal_actions(self, state):
        return ("wait",)


def test_forced_action_skip_reports_zero_actual_simulations():
    env = ForcedEnvironment()
    result = PUCTSearch(ConstantEvaluator(), PUCTConfig(simulations=64)).search(env, "")
    assert result.actions == ("wait",)
    assert result.policy == result.visit_policy == (1.0,)
    assert result.visit_counts == (0,)
    assert result.stats.forced_action
    assert result.stats.requested_simulations == 64
    assert result.stats.simulations == result.root.N == 0
    assert result.stats.nodes == 1
    assert result.selected_action == "wait"


def test_forced_action_can_be_searched_without_skip():
    result = PUCTSearch(
        ConstantEvaluator(),
        PUCTConfig(simulations=6, skip_forced_actions=False, max_depth=3),
    ).search(ForcedEnvironment(), "")
    assert result.visit_counts == (6,)
    assert not result.stats.forced_action
    assert result.stats.maximum_depth == 3


class DiamondEnvironment(InfiniteEnvironment):
    def state_key(self, state):
        return str(len(state))


def test_transposition_prediction_cache_is_local_to_one_search():
    evaluator = ConstantEvaluator(0)
    search = PUCTSearch(evaluator, PUCTConfig(simulations=2, c_puct=10))
    result = search.search(DiamondEnvironment(), "")
    assert result.stats.cache_hits == 1
    assert result.stats.evaluator_calls == 2
    assert len(evaluator.calls) == 2
    search.search(DiamondEnvironment(), "")
    assert len(evaluator.calls) == 4


class OrderedDiamondEnvironment(DiamondEnvironment):
    def legal_actions(self, state):
        return ("left", "right") if state != "r" else ("right", "left")


def test_cache_key_includes_exact_ordered_action_tuple():
    evaluator = ConstantEvaluator(0)
    result = PUCTSearch(evaluator, PUCTConfig(simulations=2, c_puct=10)).search(
        OrderedDiamondEnvironment(), ""
    )
    assert result.stats.cache_hits == 0
    assert result.stats.evaluator_calls == 3
    assert evaluator.calls[1][1] != evaluator.calls[2][1]


def test_real_simulator_legality_and_parent_isolation(real_stage, squad):
    env = ArknightsEnv(real_stage, squad, trace=True)
    state = env.reset()
    state = env.advance_to_time(state, 17)
    original_key, original_trace = env.state_key(state), copy.deepcopy(state.trace)
    result = PUCTSearch(ConstantEvaluator(), PUCTConfig(simulations=6)).search(
        env, state
    )
    assert env.state_key(state) == original_key
    assert state.trace == original_trace and state.trace_enabled
    assert result.selected_action in env.legal_actions(state)
    assert result.actions == env.legal_actions(state)
    assert result.root.state is not state
    assert result.root.state.trace == () and not result.root.state.trace_enabled
    for edge in result.root.edges.values():
        if edge.child:
            assert edge.action in env.legal_actions(result.root.state)
            assert edge.child_state_key == env.state_key(edge.child.state)
            edge.child.state.game.dp = -999
    assert result.root.state.game.dp == state.game.dp
    assert env.state_key(state) == original_key


@pytest.mark.parametrize(
    "prior,value",
    [
        ((1,), 0),
        ((-1, 2), 0),
        ((math.nan, 1), 0),
        ((0, 0), 0),
        ((1, 1), math.nan),
        ((1, 1), 1.1),
    ],
)
def test_invalid_evaluator_output_fails_explicitly(prior, value):
    with pytest.raises(ValueError):
        PUCTSearch(ConstantEvaluator(value, prior)).search(ToyEnvironment(), "root")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"simulations": 0},
        {"simulations": 1.5},
        {"max_depth": 0},
        {"c_puct": -1},
        {"temperature": math.inf},
        {"dirichlet_alpha": 0},
        {"dirichlet_epsilon": 2},
        {"prior_uniform_mix": -0.1},
        {"seed": 1.5},
    ],
)
def test_invalid_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        PUCTConfig(**kwargs)


def test_statistics_serialization_and_config_alias():
    config = PUCTConfig(mcts_simulations=4)
    assert config.simulations == config.mcts_simulations == 4
    assert dataclasses.replace(config, temperature=0.5).simulations == 4
    result = PUCTSearch(ConstantEvaluator(), config).search(ToyEnvironment(), "root")
    serialized = json.loads(json.dumps(result.stats.to_dict()))
    assert serialized["simulations"] == 4
    assert "Neural P" in result.stats.format()


class WideEnvironment(ToyEnvironment):
    def legal_actions(self, state):
        return tuple(range(20)) if state == 'root' else ()

    def is_terminal(self, state):
        return state != 'root'

    def result(self, state):
        return SimpleNamespace(success=True)


def test_top_k_keeps_complete_policy_identities_without_expanding_every_action():
    env = WideEnvironment()
    result = PUCTSearch(ConstantEvaluator(), PUCTConfig(
        simulations=5, top_k_actions=3, progressive_widening=False,
        prior_uniform_mix=0, dirichlet_epsilon=0,
    )).search(env, 'root', training=True)
    assert result.actions == tuple(range(20))
    assert result.root.active_actions == (0, 1, 2)
    assert all(count == 0 for count in result.visit_counts[3:])
    assert all(p == 0 for p in result.policy[3:])
    assert sum(result.policy) == pytest.approx(1)
    assert result.stats.expanded_action_count <= 3


def test_progressive_widening_eventually_permits_all_actions():
    result = PUCTSearch(ConstantEvaluator(), PUCTConfig(
        simulations=64, top_k_actions=2, widening_coefficient=1,
        widening_exponent=1, prior_uniform_mix=0, c_puct=10,
    )).search(WideEnvironment(), 'root')
    assert result.root.active_actions == tuple(range(20))
    assert all(n > 0 for n in result.visit_counts)


def test_policy_only_mode_is_legal_normalized_and_skips_planning():
    result = PUCTSearch(ConstantEvaluator(prior=(.8, .2)), PUCTConfig(
        simulations=8, enabled=False, temperature=1,
    )).search(ToyEnvironment(), 'root', training=True)
    assert result.policy == pytest.approx((.8, .2))
    assert result.stats.simulations == 0
    assert result.visit_counts == (0, 0)
    assert result.selected_action in result.actions


class HierarchicalWideEnvironment(ToyEnvironment):
    def __init__(self):
        from arknights_sim.environment.action import Action, ActionType, Direction
        self.actions = tuple(Action(ActionType.WAIT, wait_seconds=t) for t in (None,.5,1,2,5)) + tuple(
            Action(ActionType.DEPLOY, 'op', (x,0), d) for x in range(30) for d in Direction)

    def legal_actions(self,state):
        return self.actions if state == 'root' else ()

    def is_terminal(self,state):
        return state != 'root'

    def result(self,state):
        return SimpleNamespace(success=True)


@pytest.mark.parametrize('budget',[4,8])
def test_hierarchical_shortlist_explores_deploy_despite_many_placement_leaves(budget):
    env=HierarchicalWideEnvironment()
    priors=tuple(.1 if a.type == 'WAIT' else .5/120 for a in env.actions)
    result=PUCTSearch(ConstantEvaluator(prior=priors),PUCTConfig(
        simulations=budget,top_k_actions=budget,progressive_widening=False,
        prior_uniform_mix=0,dirichlet_epsilon=0,c_puct=2,
    )).search(env,'root',training=True)
    assert result.actions==env.actions
    assert result.neural_policy==pytest.approx(priors)
    assert len(result.root.active_actions)==budget
    assert {a.type for a in result.root.active_actions}=={'WAIT','DEPLOY'}
    assert sum(edge.N for a,edge in result.root.edges.items() if a.type=='DEPLOY')>0
    assert result.stats.expanded_action_count<=budget
    assert len(result.policy)==len(env.actions) and sum(result.policy)==pytest.approx(1)
    assert all(p==0 for a,p in zip(result.actions,result.policy) if a not in result.root.active_actions)


def test_shortlist_smaller_than_type_count_uses_total_type_probability():
    env=HierarchicalWideEnvironment()
    priors=tuple(.08 if a.type=='WAIT' else .6/120 for a in env.actions)
    result=PUCTSearch(ConstantEvaluator(prior=priors),PUCTConfig(
        simulations=1,top_k_actions=1,progressive_widening=False,
        prior_uniform_mix=0,dirichlet_epsilon=0,
    )).search(env,'root')
    assert result.root.active_actions[0].type=='DEPLOY'
    assert result.selected_action.type=='DEPLOY'
