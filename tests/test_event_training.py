"""Integrated event observation, diagnostics and safe training persistence."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from arknights_sim.environment import ArknightsEnv, WAIT
from arknights_sim.environment.rewards import progress, terminal_bonus
from network import PolicyValueNetwork, NeuralEvaluator
from training.config import validate_config
from training.replay_buffer import ReplayBuffer
from training.self_play import play_episode
from training.trainer import AlphaZeroTrainer, summarize_episodes, summarize_updates
from training.events import TrainingControl
from training.self_play import SelfPlayEpisode


def settings(tmp_path, **kwargs):
    return validate_config(dict(device='cpu',num_threads=1,iterations=1,
        episodes_per_iteration=1,mcts_simulations=2,max_depth=4,
        batch_size=2,training_steps_per_iteration=1,evaluation_episodes=1,
        checkpoint_dir=str(tmp_path/'checkpoints'),output_dir=str(tmp_path/'outputs'),
        **kwargs))


class WaitEvaluator:
    def evaluate(self, env, state, actions):
        return tuple(float(a == WAIT) for a in actions), 0.


def test_delayed_kills_and_resource_waits_have_equal_total_return():
    initial = dict(total_enemies=2, initial_life=10, kills=0, life=10)
    partial = dict(initial, kills=1)
    final = dict(initial, kills=2, success=True, simulator_result='WIN', termination='battle_end')
    def score(sequence):
        return sum(progress(b) - progress(a) for a,b in zip(sequence,sequence[1:])) + terminal_bonus(final)
    immediate = [initial,partial,final]
    held = [initial] * 8 + [partial] * 20 + [final]
    assert score(immediate) == pytest.approx(score(held))
    # No kill time, stored DP/SP, action count, or waiting duration can mint
    # reward independently of preserved life and the eventual battle result.
    assert score(held) == pytest.approx(.8)


def test_episode_records_decision_nodes_action_mix_and_future_explanations(simple,tmp_path):
    stage = replace(simple, initial_cost=0)
    env = ArknightsEnv(stage, squad=(), horizon=30)
    episode = play_episode(env,WaitEvaluator(),settings(tmp_path),seed=1)
    row=episode.metrics
    assert row['decision_nodes'] == row['decisions'] == len(row['history'])
    assert row['simulation_ticks'] > row['decision_nodes']
    assert row['wait_percentage'] == 100
    assert sum(row['action_counts'].values()) == row['decision_nodes']
    assert sum(row['action_percentages'].values()) == pytest.approx(100)
    explanation=row['history'][0]['decision']
    assert {'current_event_id','upcoming','policy','value','decision_reasons'} <= explanation.keys()
    assert all('probability' in item and 'search_probability' in item for item in explanation['policy'])
    assert sum(explanation['action_type_policy'].values())==pytest.approx(1)
    assert explanation['search_candidates']
    summary=summarize_episodes([row])
    assert summary['average_decision_nodes'] == row['decisions']


def test_replay_refuses_old_observation_even_when_reward_matches():
    state=ReplayBuffer().state_dict()
    state.pop('event_graph_version')
    with pytest.raises(ValueError,match='observation/event graph'):
        ReplayBuffer().load_state_dict(state)


def test_event_observation_trains_and_checkpoint_restores_exactly(simple,operator,tmp_path):
    factory=lambda: ArknightsEnv(simple,[operator],horizon=30,max_decisions=128)
    trainer=AlphaZeroTrainer(settings(tmp_path),factory)
    before=deepcopy(trainer.model.state_dict())
    result=trainer.train()
    assert len(trainer.buffer)>0
    assert {'events','event_relations','progression'} <= next(iter(trainer.buffer.samples)).state_features.keys()
    assert result['history'][0]['gradient_norm']>0
    assert any(not torch.equal(v,trainer.model.state_dict()[k]) for k,v in before.items())
    restored=AlphaZeroTrainer(settings(tmp_path,resume_from=str(tmp_path/'checkpoints/latest.pt')),factory)
    assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in trainer.model.state_dict().items())
    env=factory();state=env.reset()
    a=NeuralEvaluator(trainer.model).predict(env,state)
    b=NeuralEvaluator(restored.model).predict(env,state)
    assert a['policy']==b['policy'] and a['value']==b['value']
    assert torch.equal(a['state_embedding'],b['state_embedding'])


def test_future_ablation_resume_preserves_configuration(simple,operator,tmp_path):
    factory=lambda: ArknightsEnv(simple,[operator])
    trainer=AlphaZeroTrainer(settings(tmp_path,future_events_enabled=False,planner_enabled=False),factory)
    path=trainer.save_checkpoint()
    restored=AlphaZeroTrainer(settings(tmp_path,resume_from=str(path)),factory)
    assert restored.model.future_events_enabled is False
    assert restored.config['planner_enabled'] is False


def test_interrupted_episode_retains_sampled_stage_and_rng(simple,operator,tmp_path,monkeypatch):
    import training.trainer as trainer_module
    factory=lambda: ArknightsEnv(simple,[operator])
    config=settings(tmp_path,stage_pool=['easy','hard'])
    trainer=AlphaZeroTrainer(config,factory)
    trainer.best_metrics={}  # Isolate interruption after the initial evaluation.
    calls=[]
    def interrupted_episode(*args,**kwargs):
        calls.append((kwargs['seed'],kwargs['stage']))
        kwargs['control'].stop()
        return SelfPlayEpisode([],{'interrupted':True},False)
    monkeypatch.setattr(trainer_module,'play_episode',interrupted_episode)
    trainer.train(control=TrainingControl())
    rng=trainer.rng.getstate()
    restored=AlphaZeroTrainer(settings(tmp_path,resume_from=str(tmp_path/'checkpoints/latest.pt')),factory)
    restored.train(control=TrainingControl())
    assert len(calls)==2 and calls[0]==calls[1]
    assert rng==restored.rng.getstate()


def test_squad_iterations_report_active_losses_without_claiming_battle_updates():
    episode=dict(squad_selection_loss=.3,squad_selection_policy_loss=.1,
                 squad_selection_value_loss=.4,squad_selection_gradient_norm=2.,
                 squad_selection_step_time=.02)
    metrics=summarize_updates([], [episode, {'truncated':True}])
    assert metrics['policy_loss']==.1 and metrics['value_loss']==.4
    assert metrics['total_loss']==.3 and metrics['gradient_norm']==2.
    assert metrics['loss_network']=='squad'
    assert metrics['battle_optimizer_steps']==0
    assert metrics['squad_optimizer_steps']==metrics['total_optimizer_steps']==1
    inactive=summarize_updates([],[])
    assert inactive['loss_network']=='none' and inactive['total_optimizer_steps']==0


def test_squad_only_episode_does_not_build_unused_battle_samples(simple,tmp_path):
    env=ArknightsEnv(simple,squad=(),horizon=30)
    episode=play_episode(env,WaitEvaluator(),settings(tmp_path),seed=1,battle_training=False)
    assert episode.metrics['reward_trainable']
    assert not episode.metrics['battle_trainable']
    assert episode.samples==[] and episode.metrics['battle_samples']==0


def test_neural_entropy_is_distinct_from_greedy_search_entropy(simple,operator,tmp_path):
    class UniformEvaluator:
        def evaluate(self,env,state,actions):
            return tuple(1/len(actions) for _ in actions),0.
    env=ArknightsEnv(simple,[operator],horizon=30,max_decisions=1)
    episode=play_episode(env,UniformEvaluator(),settings(tmp_path),seed=1,training=False)
    assert episode.metrics['policy_entropy']>0
    assert episode.metrics['search_policy_entropy']==0
    assert not episode.metrics['battle_trainable'] and not episode.metrics['squad_trainable']
