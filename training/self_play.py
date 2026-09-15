"""Neural PUCT episodes. No scripted actions or tactical rollout policies."""

import math
from dataclasses import dataclass
from time import perf_counter

import torch

from mcts import PUCTConfig, PUCTSearch
from network import ActionEncoder, StateEncoder
from arknights_sim.environment.rewards import REWARD_VERSION, episode_reward, is_truncated

from .replay_buffer import TrainingSample
from .decision_logging import action_metrics, decision_explanation


@dataclass
class SelfPlayEpisode:
    samples: list[TrainingSample]
    metrics: dict
    complete: bool


def play_episode(
    env,
    evaluator,
    config,
    *,
    seed,
    stage="0-1",
    training=True,
    control=None,
    callback=None,
    iteration=0,
    episode_index=0,
    squad_coordinator=None,
    squad_training=True,
    battle_training=True,
):
    """Only genuinely terminal episodes produce replay samples and z labels."""
    state_encoder, action_encoder = StateEncoder(), ActionEncoder()
    chosen = squad_coordinator.choose(env, stage, seed, training, learn=squad_training) if squad_coordinator else None
    state = env.reset(stage_id=stage, seed=seed, squad=chosen)
    search = PUCTSearch(
        evaluator,
        PUCTConfig(
            mcts_simulations=config["mcts_simulations"],
            c_puct=config["c_puct"],
            max_depth=config["max_depth"],
            seed=seed,
            dirichlet_alpha=config["dirichlet_alpha"],
            dirichlet_epsilon=config["dirichlet_epsilon"],
            prior_uniform_mix=config["prior_uniform_mix"],
            temperature=config["temperature"],
            skip_forced_actions=config["skip_forced_actions"],
            top_k_actions=config.get('top_k_actions', 8),
            progressive_widening=config.get('progressive_widening', True),
            widening_coefficient=config.get('widening_coefficient', 2.0),
            widening_exponent=config.get('widening_exponent', 0.5),
            enabled=config.get('planner_enabled', True),
        ),
    )
    trajectory, history, potentials = [], [], []
    initial_potential = env.reward_potential(state)
    initial_tick = state.game.clock.tick
    depths, latencies, values, entropies, neural_entropies, counts = [], [], [], [], [], []
    forced = actual_simulations = total_nodes = root_visits = 0
    encode_seconds = transition_seconds = 0.0
    evaluations_before = getattr(evaluator, "evaluations", 0)
    started = perf_counter()
    while not env.is_terminal(state):
        if control is not None and not control.checkpoint():
            if squad_coordinator:
                squad_coordinator.trace = None
            return SelfPlayEpisode(
                [],
                {
                    "seed": seed,
                    "stage": stage,
                    "interrupted": True,
                    "decisions": len(history),
                },
                False,
            )
        tick = perf_counter()
        result = search.search(env, state, training=training)
        latencies.append(perf_counter() - tick)
        actions = result.actions
        if (
            result.selected_action not in actions
            or result.selected_action not in env.legal_actions(state)
        ):
            raise RuntimeError("PUCT returned an illegal action")
        policy = torch.tensor(result.policy, dtype=torch.float32)
        if training and battle_training:
            encode_start = perf_counter()
            trajectory.append(
                (
                    state_encoder.encode(state),
                    tuple(a.to_dict() for a in actions),
                    action_encoder.encode(state, actions),
                    policy,
                )
            )
            potentials.append(env.reward_potential(state))
            encode_seconds += perf_counter() - encode_start
        counts.append(len(actions))
        forced += len(actions) == 1
        stats = (
            result.stats.to_dict()
            if hasattr(result.stats, "to_dict")
            else vars(result.stats)
        )
        actual_simulations += stats.get(
            "simulations", stats.get("actual_simulations", 0)
        )
        total_nodes += stats.get("nodes", 0)
        root_visits += stats.get("root_visits", 0)
        depths.append(stats.get("maximum_depth", stats.get("max_depth", 0)))
        values.append(float(result.root_value))
        entropies.append(
            -sum(float(p) * math.log(float(p)) for p in result.policy if p > 0)
        )
        neural_entropies.append(
            -sum(float(p) * math.log(float(p)) for p in result.neural_policy if p > 0)
        )
        history.append(
            {
                "time": state.game.current_time,
                "action": result.selected_action.to_dict(),
                "legal_actions": len(actions),
                'decision': decision_explanation(env, state, result, config.get('decision_log_top_k', 8)),
            }
        )
        transition_start = perf_counter()
        previous = state
        state = env.step(state, result.selected_action)
        history[-1]['reward'] = env.transition_reward(previous, state)
        transition_seconds += perf_counter() - transition_start
    outcome = env.result(state).to_dict()
    truncated = is_truncated(outcome)
    score = episode_reward(outcome)
    z = score - initial_potential
    samples = [] if truncated else [
        TrainingSample(*entry, score - potential)
        for entry, potential in zip(trajectory, potentials)
    ]
    immediate_retreats = sum(
        a['action']['type'] == 'DEPLOY' and b['action']['type'] == 'RETREAT'
        and a['action']['operator_id'] == b['action']['operator_id']
        and abs(a['time'] - b['time']) < 1e-9
        for a, b in zip(history, history[1:])
    )
    metrics = {
        **outcome,
        "seed": seed,
        "stage": stage,
        "iteration": iteration,
        "episode": episode_index,
        "return": z,
        "reward_version": REWARD_VERSION,
        "truncated": truncated,
        "reward_trainable": not truncated,
        "battle_trainable": bool(training and battle_training and not truncated),
        "squad_trainable": bool(training and squad_training and squad_coordinator and not truncated),
        "battle_samples": len(samples),
        "win": outcome['termination'] == 'battle_end' and outcome['simulator_result'] == 'WIN',
        "progress_reward": env.reward_potential(state) - initial_potential,
        "terminal_reward": env.terminal_reward(state),
        "immediate_retreats": immediate_retreats,
        "decisions": len(history),
        "decision_nodes": len(history),
        "simulation_ticks": state.game.clock.tick - initial_tick,
        "ticks_per_decision": (state.game.clock.tick - initial_tick) / max(1, len(history)),
        "clear_time": state.game.current_time if outcome['termination'] == 'battle_end' and outcome['simulator_result'] == 'WIN' else None,
        "wall_time": perf_counter() - started,
        "game_time": state.game.current_time,
        "forced_action_fraction": forced / max(1, len(history)),
        "mean_value": sum(values) / max(1, len(values)),
        "policy_entropy": sum(neural_entropies) / max(1, len(neural_entropies)),
        "search_policy_entropy": sum(entropies) / max(1, len(entropies)),
        "average_legal_actions": sum(counts) / max(1, len(counts)),
        "maximum_legal_actions": max(counts, default=0),
        "average_decision_time": sum(latencies) / max(1, len(latencies)),
        "maximum_depth": max(depths, default=0),
        "actual_simulations": actual_simulations,
        "mcts_nodes": total_nodes,
        "average_mcts_nodes": total_nodes / max(1, len(history)),
        "root_visits": root_visits,
        "neural_evaluations": getattr(evaluator, "evaluations", 0) - evaluations_before,
        "search_seconds": sum(latencies),
        "sample_encode_seconds": encode_seconds,
        "transition_seconds": transition_seconds,
        "history": history,
        "interrupted": False,
        "selected_squad": list(state.game.squad),
        **action_metrics(history),
    }
    if squad_coordinator:
        metrics.update(squad_coordinator.finish(metrics, training and squad_training))
    return SelfPlayEpisode(samples, metrics, True)
