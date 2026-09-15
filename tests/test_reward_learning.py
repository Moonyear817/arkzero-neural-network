"""Behavioral checks for progress credit, truncation and training isolation."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import random

import pytest
import torch

from arknights_sim.environment import ArknightsEnv, WAIT
from arknights_sim.environment.rewards import (
    REWARD_VERSION, episode_reward, terminal_bonus,
)
from mcts import PUCTConfig, PUCTSearch
from network import PolicyValueNetwork
from training.config import validate_config
from training.replay_buffer import ReplayBuffer, masked_policy_loss
from training.self_play import play_episode
from training.squad_selection import SquadCoordinator
from training.stage_sampler import StageSampler
from training.trainer import AlphaZeroTrainer, promotion_decision


def outcome(kills=0, life=0, *, success=False, result='LOSS', termination='battle_end'):
    return dict(total_enemies=10, kills=kills, life=life, initial_life=10,
                leaks=10-life, success=success, simulator_result=result,
                termination=termination)


def config(tmp_path, **updates):
    return validate_config({**dict(device='cpu', iterations=1, episodes_per_iteration=1,
        mcts_simulations=2, max_depth=4, batch_size=2, training_steps_per_iteration=1,
        evaluation_episodes=1, checkpoint_dir=str(tmp_path/'checkpoints'),
        output_dir=str(tmp_path/'outputs')), **updates})


class WaitEvaluator:
    def evaluate(self, env, state, actions):
        return tuple(float(a == WAIT) for a in actions), 0.0


def test_early_failure_improves_without_floor_and_wins_dominate():
    failures = [episode_reward(outcome(kills=k)) for k in range(10)]
    assert all(a < b for a, b in zip(failures, failures[1:]))
    ordinary = [episode_reward(outcome(kills=k, life=1, result='WIN')) for k in range(10)]
    perfect = episode_reward(outcome(kills=10, life=10, success=True, result='WIN'))
    assert max(failures) < min(ordinary) < max(ordinary) < perfect
    assert perfect == pytest.approx(0.8)


def test_truncation_bound_cannot_beat_a_natural_loss():
    for k in range(11):
        for life in range(11):
            cut = outcome(k, life, termination='time_limit')
            assert episode_reward(cut) == pytest.approx(-0.8)
            assert -1.000001 <= terminal_bonus(cut) <= 1
            assert episode_reward(cut) <= episode_reward(outcome(k, life)) + 1e-9


def test_actual_life_cost_drives_penalty(simple):
    values = []
    for cost in (1, 2):
        stage = replace(simple, enemies=(replace(simple.enemies[0], life_cost=cost),))
        env = ArknightsEnv(stage, squad=(), horizon=30)
        state = env.reset()
        while not env.is_terminal(state):
            state = env.step(state, WAIT)
        metrics = env.result(state).to_dict()
        assert metrics['leaks'] == 1
        assert metrics['life'] == stage.life-cost
        values.append(episode_reward(metrics))
    assert values[0] - values[1] == pytest.approx(0.2/simple.life)


def test_deploy_retreat_is_legal_but_does_not_mint_reward(simple, operator):
    env = ArknightsEnv(simple, [operator])
    start = env.reset()
    deploy = next(a for a in env.legal_actions(start) if a.type == 'DEPLOY')
    deployed = env.step(start, deploy)
    retreat = next(a for a in env.legal_actions(deployed) if a.type == 'RETREAT')
    withdrawn = env.step(deployed, retreat)
    assert start.game.current_time == withdrawn.game.current_time
    assert env.transition_reward(start, deployed) == 0
    assert env.transition_reward(deployed, withdrawn) == 0


def test_return_to_go_matches_observed_rewards(simple, tmp_path):
    # Two enemies leak at different times; states after the first leak must not
    # be charged for that already-observed life loss a second time.
    spawn = simple.waves[0].spawns[0]
    wave = replace(simple.waves[0], spawns=(spawn, replace(spawn, time=3)))
    stage = replace(simple, waves=(wave,))
    env = ArknightsEnv(stage, squad=(), horizon=30)
    episode = play_episode(env, WaitEvaluator(), config(tmp_path), seed=1)
    assert not episode.metrics['truncated']
    rewards = [r['reward'] for r in episode.metrics['history']]
    remaining = episode.metrics['terminal_reward']
    expected = []
    for reward in reversed(rewards):
        remaining += reward
        expected.append(remaining)
    assert [s.z for s in episode.samples] == pytest.approx(list(reversed(expected)))
    assert len(set(round(s.z, 8) for s in episode.samples)) > 1
    assert sum(rewards) + episode.metrics['terminal_reward'] == pytest.approx(episode.metrics['return'])


def test_truncated_episode_never_updates_either_network(simple, tmp_path):
    factory = lambda: ArknightsEnv(simple, squad=(), horizon=30, max_decisions=1)
    settings = config(tmp_path, auto_squad=True, squad_size=2, joint_training_mode='joint')
    trainer = AlphaZeroTrainer(settings, factory)
    before = deepcopy(trainer.model.state_dict())
    squad_before = deepcopy(trainer.squad_coordinator.selector.state_dict())
    result = trainer.train()
    assert len(trainer.buffer) == 0
    assert trainer.squad_coordinator.trace is None
    assert result['history'][0]['selfplay_trainable_episodes'] == 0
    assert result['history'][0]['optimizer_steps'] == 0
    assert all(torch.equal(v, trainer.model.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, trainer.squad_coordinator.selector.state_dict()[k]) for k, v in squad_before.items())
    resumed = AlphaZeroTrainer({**settings, 'resume_from':str(tmp_path/'checkpoints/latest.pt')}, factory)
    assert len(resumed.buffer) == 0 and resumed.iteration == 1


def test_squad_uses_shared_reward_and_frozen_selection_has_no_gradient(simple):
    env = ArknightsEnv(simple, squad=())
    coordinator = SquadCoordinator(env.data_dir, size=2)
    coordinator.choose(env, '0-1', 1, True, learn=False)
    assert coordinator.trace is None
    coordinator.choose(env, '0-1', 1, True)
    metrics = outcome(kills=3)
    update = coordinator.finish(metrics, True)
    assert update['squad_selection_reward'] == episode_reward(metrics)


def test_alternating_training_updates_only_the_active_network(simple, tmp_path):
    factory = lambda: ArknightsEnv(simple, squad=(), horizon=30, max_decisions=128)
    trainer = AlphaZeroTrainer(config(tmp_path, auto_squad=True, squad_size=2, iterations=2), factory)
    snapshots = [(deepcopy(trainer.model.state_dict()), deepcopy(trainer.squad_coordinator.selector.state_dict()))]
    def record(event):
        if event.kind == 'training_metrics':
            snapshots.append((deepcopy(trainer.model.state_dict()), deepcopy(trainer.squad_coordinator.selector.state_dict())))
    result = trainer.train(callback=record)
    assert [r['training_phase'] for r in result['history']] == ['battle', 'squad']
    assert any(not torch.equal(v, snapshots[1][0][k]) for k, v in snapshots[0][0].items())
    assert all(torch.equal(v, snapshots[1][1][k]) for k, v in snapshots[0][1].items())
    assert all(torch.equal(v, snapshots[2][0][k]) for k, v in snapshots[1][0].items())
    assert any(not torch.equal(v, snapshots[2][1][k]) for k, v in snapshots[1][1].items())


def test_forced_policy_rows_do_not_dilute_decisions():
    logits = torch.tensor([[1., 2.]], requires_grad=True)
    target = torch.tensor([[0., 1.]])
    mask = torch.tensor([[True, True]])
    loss = masked_policy_loss(logits, target, mask)
    padded_logits = torch.cat((logits, torch.zeros(20, 2)), dim=0)
    padded_target = torch.cat((target, torch.tensor([[1., 0.]]).repeat(20, 1)), dim=0)
    padded_mask = torch.cat((mask, torch.tensor([[True, False]]).repeat(20, 1)), dim=0)
    assert masked_policy_loss(padded_logits, padded_target, padded_mask) == loss


def test_curriculum_requires_repeated_success_and_resumes():
    sampler = StageSampler('easy', ['easy', 'hard'], curriculum=True, evaluations=2)
    assert {sampler.sample(random.Random(i)) for i in range(10)} == {'easy'}
    assert not sampler.observe([dict(stage='easy', success=True)])
    restored = StageSampler('easy', ['easy', 'hard'], curriculum=True, evaluations=2)
    restored.load_state_dict(sampler.state_dict())
    assert not restored.observe([dict(stage='easy', success=False)])
    assert not restored.observe([dict(stage='easy', success=True)])
    assert restored.observe([dict(stage='easy', success=True)])
    assert restored.active_stages == ('easy', 'hard')


def test_old_training_and_replay_are_rejected_but_weights_can_migrate(simple, tmp_path):
    settings = config(tmp_path)
    factory = lambda: ArknightsEnv(simple, squad=())
    trainer = AlphaZeroTrainer(settings, factory)
    payload = trainer._checkpoint_payload()
    payload.pop('reward_version')
    path = tmp_path/'old.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='奖励规则'):
        trainer.load_checkpoint(path)
    replay = ReplayBuffer().state_dict()
    replay.pop('reward_version')
    with pytest.raises(ValueError, match='reward version'):
        ReplayBuffer().load_state_dict(replay)
    warmed = AlphaZeroTrainer({**settings, 'warm_start':str(path)}, factory)
    assert torch.count_nonzero(warmed.model.value_head[-2].weight) == 0
    assert len(warmed.buffer) == 0
    assert torch.equal(warmed.model.map_input.weight, trainer.model.map_input.weight)


def test_truncated_evaluation_cannot_promote():
    best = dict(success_rate=0., average_return=-0.8, average_leaks=10., average_kills=0.)
    candidate = dict(best, average_return=0.7, truncated_fraction=1.)
    assert promotion_decision(candidate, best)[0] is False
    assert promotion_decision(candidate, None)[0] is False


def test_gui_distinguishes_ordinary_clear_and_truncation():
    from desktop.views.training_view import _episode_result_source
    assert _episode_result_source({'win': True}).startswith('Clear with leaks')
    assert _episode_result_source({'truncated': True}).startswith('Truncated')
    assert _episode_result_source({'simulator_result': 'LOSS'}).startswith('Failure')


class RewardTree:
    def clone_state(self, state): return state
    def state_key(self, state): return state
    def is_terminal(self, state): return state in ('win', 'loss')
    def legal_actions(self, state):
        return ('keep', 'retreat') if state == 'root' else ('wait',)
    def step(self, state, action):
        if state == 'root': return 'waiting' if action == 'keep' else 'loss'
        return 'win'
    def transition_reward(self, before, after): return 0.2 if after == 'win' else 0.
    def terminal_reward(self, state): return 0.6 if state == 'win' else -0.6


class UniformEvaluator:
    def evaluate(self, env, state, actions): return tuple(1/len(actions) for _ in actions), 0.


def test_search_sees_reward_beyond_forced_wait_and_backs_up_to_parent():
    env = RewardTree()
    search = PUCTSearch(UniformEvaluator(), PUCTConfig(simulations=8, max_depth=4))
    result = search.search(env, 'root')
    assert result.selected_action == 'keep'
    assert result.root.edges['keep'].Q == pytest.approx(0.8)
    assert result.root.edges['retreat'].Q == pytest.approx(-0.6)
    assert result.root.edges['keep'].child.Q == pytest.approx(0.8)
    assert result.root.edges['keep'].child.edges['wait'].child.Q == pytest.approx(0.6)
