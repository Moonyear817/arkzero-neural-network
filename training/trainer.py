"""Small single-agent AlphaZero loop; independent of all desktop modules."""

import json
import logging
import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch

from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader, SKILL_ENGINE_VERSION
from arknights_sim.environment import ArknightsEnv
from arknights_sim.environment.rewards import REWARD_VERSION
from network import NeuralEvaluator, PolicyValueNetwork

from .config import resolve_device, scheduled_value, validate_config
from .events import TrainingControl, TrainingEvent
from .replay_buffer import (
    ReplayBuffer,
    atomic_torch_save,
    collate_samples,
    masked_policy_loss,
)
from .self_play import play_episode
from .stage_sampler import StageSampler
from .schema import OBSERVATION_VERSION, EVENT_GRAPH_VERSION
from .decision_logging import ACTION_CATEGORIES
from .replay_storage import ReplayStorage

logger = logging.getLogger(__name__)


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def _mean(rows, key):
    return sum(row[key] for row in rows) / max(1, len(rows))


def summarize_updates(battle_steps, episodes):
    """Expose the active learner's losses and separately retain both networks."""
    squad_steps=[r for r in episodes if 'squad_selection_loss' in r]
    fields={'policy_loss':'squad_selection_policy_loss',
            'value_loss':'squad_selection_value_loss','total_loss':'squad_selection_loss',
            'gradient_norm':'squad_selection_gradient_norm',
            'training_step_time':'squad_selection_step_time'}
    squad={key:_mean(squad_steps,source) for key,source in fields.items()}
    battle={key:_mean(battle_steps,key) for key in battle_steps[0]} if battle_steps else {}
    active=battle if battle_steps else squad
    return {
        **active,
        **{f'squad_{key}':value for key,value in squad.items()},
        **{f'battle_{key}':value for key,value in battle.items()},
        'loss_network':'battle' if battle_steps else 'squad' if squad_steps else 'none',
        'battle_optimizer_steps':len(battle_steps),
        'squad_optimizer_steps':len(squad_steps),
        'total_optimizer_steps':len(battle_steps)+len(squad_steps),
    }


def summarize_episodes(rows):
    clears = [r['clear_time'] for r in rows if r.get('clear_time') is not None]
    return {
        "episodes": len(rows),
        "success_rate": _mean(rows, "success"),
        "win_rate": sum(r.get('win', r['success']) for r in rows) / max(1, len(rows)),
        "truncated_fraction": sum(r.get('truncated', False) for r in rows) / max(1, len(rows)),
        "average_immediate_retreats": sum(r.get('immediate_retreats', 0) for r in rows) / max(1, len(rows)),
        "average_return": _mean(rows, "return"),
        "average_leaks": _mean(rows, "leaks"),
        "average_kills": _mean(rows, "kills"),
        "average_decisions": _mean(rows, "decisions"),
        "average_decision_nodes": sum(r.get('decision_nodes', r['decisions']) for r in rows) / max(1, len(rows)),
        "average_simulation_ticks": sum(r.get('simulation_ticks', 0) for r in rows) / max(1, len(rows)),
        "ticks_per_decision": sum(r.get('simulation_ticks', 0) for r in rows) / max(1, sum(r['decisions'] for r in rows)),
        "average_clear_time": sum(clears) / len(clears) if clears else None,
        **{f'{key.lower()}_percentage': 100 * sum(r.get('action_counts', {}).get(key, 0) for r in rows) / max(1, sum(r['decisions'] for r in rows)) for key in ACTION_CATEGORIES},
        "average_episode_time": _mean(rows, "wall_time"),
        "average_decision_time": _mean(rows, "average_decision_time"),
        "forced_action_fraction": _mean(rows, "forced_action_fraction"),
        "mean_value": _mean(rows, "mean_value"),
        "average_mcts_nodes": _mean(rows, "average_mcts_nodes"),
        "natural_terminal_fraction": sum(r["termination"] == "battle_end" for r in rows)
        / max(1, len(rows)),
        "decision_limit_fraction": sum(
            r["termination"] == "decision_limit" for r in rows
        )
        / max(1, len(rows)),
        "time_limit_fraction": sum(r["termination"] == "time_limit" for r in rows)
        / max(1, len(rows)),
    }


def promotion_decision(candidate, best, promote_on_equal=True):
    """Fixed-seed lexicographic score; ties are explicit, never called better."""
    if candidate.get('truncated_fraction', 0) > 0:
        return False, 'incomplete_evaluation'
    if best is None:
        return True, "first_evaluated_candidate"
    fields = ("success_rate", "win_rate", "average_return", "average_leaks", "average_kills")
    a = tuple(candidate.get(k, candidate['success_rate']) * (-1 if k == "average_leaks" else 1) for k in fields)
    b = tuple(best.get(k, best['success_rate']) * (-1 if k == "average_leaks" else 1) for k in fields)
    if a > b:
        return True, "better_fixed_seed_score"
    if a == b and promote_on_equal:
        return True, "tie_accepted"
    return False, "tie_retained" if a == b else "lower_fixed_seed_score"


class AlphaZeroTrainer:
    """train(callback=None, control=None) emits immutable, UI-neutral events.

    Pending iterations are checkpointed after completed episodes/optimizer steps.
    A stopped incomplete episode is discarded and replayed with the same seed on
    resume; completed data and metrics are never added twice. Atomic checkpoints
    include current/best networks, optimizer, replay references, RNG and progress.
    """

    def __init__(self, config=None, env_factory=None):
        self.config = validate_config(config or {})
        from accounts.readiness import require_training_ready
        require_training_ready(self.config)
        torch.set_num_threads(self.config["num_threads"])
        self.device = resolve_device(self.config["device"])
        torch.manual_seed(self.config["seed"])
        self.rng = random.Random(self.config["seed"])
        self.model = PolicyValueNetwork(future_events_enabled=self.config['future_events_enabled']).to(self.device)
        self.best_model = deepcopy(self.model).eval()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.buffer = ReplayBuffer(
            self.config["replay_buffer_size"], self.config["seed"]
        )
        self.iteration = 0
        self.best_iteration = 0
        self.best_metrics = None
        self.pending = None
        self.total_generated_states = 0
        self.history = []
        self._callback = lambda event: None
        self._control = None
        self._env_factory = env_factory
        self._cached_env = None
        self.migration_report = None
        self.stage_sampler = self._make_stage_sampler()
        self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.output_dir = Path(self.config["output_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.replay_storage = ReplayStorage(self.config)
        self.replay_recovery_notice = None
        self.squad_coordinator = self.best_squad_coordinator = None
        self._init_squad_coordinator()
        if self.config.get("warm_start") and not self.config.get("resume_from"):
            initial = torch.load(self.config["warm_start"], map_location="cpu", weights_only=True)
            self.model.load_state_dict(initial["model_state_dict"])
            self.migration_report = deepcopy(getattr(self.model, 'migration_report', None))
            logger.warning('Checkpoint weight migration: %s', json.dumps(self.migration_report, ensure_ascii=False))
            if initial.get('reward_version') != REWARD_VERSION:
                # Old terminal +/-1 values have a different meaning. Preserve
                # features/policy, but relearn the remaining-return head.
                torch.nn.init.zeros_(self.model.value_head[-2].weight)
                torch.nn.init.zeros_(self.model.value_head[-2].bias)
            self.best_model.load_state_dict(self.model.state_dict())
            if self.squad_coordinator and initial.get('squad_selector'):
                data = {k: v for k, v in initial['squad_selector'].items() if k != 'optimizer'}
                self.squad_coordinator.load_state_dict(data)
                self.squad_coordinator.selector.future_events_enabled = self.config['future_events_enabled']
                logger.warning('Squad checkpoint weight migration: %s', json.dumps(
                    getattr(self.squad_coordinator.selector, 'migration_report', None), ensure_ascii=False))
                if initial.get('reward_version') != REWARD_VERSION:
                    torch.nn.init.zeros_(self.squad_coordinator.selector.baseline[-2].weight)
                    torch.nn.init.zeros_(self.squad_coordinator.selector.baseline[-2].bias)
                self.best_squad_coordinator.selector.load_state_dict(self.squad_coordinator.selector.state_dict())
        if self.config.get("resume_from"):
            self.load_checkpoint(self.config["resume_from"])

    def _make_stage_sampler(self):
        return StageSampler(self.config['stage'], self.config.get('stage_pool'),
                            curriculum=self.config['curriculum'],
                            threshold=self.config['curriculum_success_rate'],
                            evaluations=self.config['curriculum_evaluations'])

    def _training_phase(self, iteration):
        if not self.squad_coordinator:
            return 'battle'
        if self.config['joint_training_mode'] == 'joint':
            return 'joint'
        return 'battle' if iteration % 2 else 'squad'

    def _init_squad_coordinator(self):
        self.squad_coordinator = self.best_squad_coordinator = None
        if self.config.get("auto_squad"):
            from training.squad_selection import SquadCoordinator
            self.squad_coordinator = SquadCoordinator(self._env().data_dir,
                self.config.get("squad_pool"), self.config.get("squad_size", 6),
                progression=self.config['operator_progression'], skill_overrides=self.config['skill_overrides'])
            self.squad_coordinator.selector.future_events_enabled = self.config['future_events_enabled']
            self.best_squad_coordinator = SquadCoordinator(self._env().data_dir,
                self.squad_coordinator.keys, self.squad_coordinator.size,
                selector=deepcopy(self.squad_coordinator.selector),
                progression=self.config['operator_progression'], skill_overrides=self.config['skill_overrides'])
            self.best_squad_coordinator.selector.future_events_enabled = self.config['future_events_enabled']

    def _emit(self, kind, **payload):
        self._callback(TrainingEvent(kind, payload))

    def _append_jsonl(self, name, payload):
        with (self.output_dir / name).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
            )
            stream.flush()

    def _env(self):
        if self._env_factory is not None:
            return self._env_factory()
        if self._cached_env is None:
            env = ArknightsEnv(
                data_dir=self.config["data_dir"],
                horizon=self.config["horizon"],
                max_decisions=self.config["max_decisions"],
                trace=False,
            )
            loader = OperatorLoader(
                env.data_dir / "character_table.json",
                env.data_dir / "range_table.json",
                SkillLoader(env.data_dir / "skill_table.json"),
            )
            from arknights_sim.data.progression import Progression, load_progressed
            from dataclasses import replace
            request = Progression.from_dict(self.config['operator_progression'])
            env.squad = tuple(load_progressed(loader, key, replace(request,
                skill_index=self.config['skill_overrides'].get(key, request.skill_index))) for key in self.config["squad"])
            initial = env.reset(stage_id=self.config["stage"], seed=self.config["seed"])
            if not self.config.get("stage_pool"):
                env.stage = initial.game.stage
            self._cached_env = env
        return self._cached_env

    def _episode_config(self, iteration):
        config = deepcopy(self.config)
        config["temperature"] = scheduled_value(config, "temperature", iteration)
        config["prior_uniform_mix"] = scheduled_value(
            config, "prior_uniform_mix", iteration
        )
        return config

    def _evaluate_model(self, model, seeds=None, control=None, squad_coordinator=None):
        model.eval()
        evaluator = NeuralEvaluator(model, device=self.device)
        seeds = (
            list(seeds)
            if seeds is not None
            else [
                self.config["evaluation_seed"] + i
                for i in range(self.config["evaluation_episodes"])
            ]
        )
        rows = []
        evaluation_config = self._episode_config(self.iteration)
        evaluation_config["prior_uniform_mix"] = 0.0
        evaluation_config["temperature"] = 0.0
        evaluation_config["dirichlet_epsilon"] = 0.0
        for i, seed in enumerate(seeds):
            episode = play_episode(
                self._env(),
                evaluator,
                evaluation_config,
                seed=seed,
                stage=(self.config.get("stage_pool") or [self.config["stage"]])[i % len(self.config.get("stage_pool") or [self.config["stage"]])],
                training=False,
                control=control,
                iteration=self.iteration,
                episode_index=i + 1,
                squad_coordinator=squad_coordinator or (self.best_squad_coordinator if model is self.best_model else self.squad_coordinator),
            )
            if not episode.complete:
                return None
            rows.append(episode.metrics)
        return {
            **summarize_episodes(rows),
            "seeds": seeds,
            "episode_results": rows,
            "mcts_simulations": self.config["mcts_simulations"],
            "noise": False,
            "temperature": 0.0,
        }

    def evaluate(self, checkpoint=None, seeds=None):
        """Evaluation uses fixed seeds, no root noise and zero temperature."""
        model = self.model
        coordinator = None
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = PolicyValueNetwork(future_events_enabled=payload.get('network_metadata', {}).get('future_events_enabled', True)).to(self.device)
            model.load_state_dict(payload["model_state_dict"])
            model.reward_version = payload.get('reward_version', 1)
            if payload.get("squad_selector"):
                from training.squad_selection import SquadCoordinator
                data = payload["squad_selector"]
                stored_config = payload.get('config', {})
                coordinator = SquadCoordinator(self._env().data_dir, data["pool"], data["size"],
                    trainable=False, progression=stored_config.get('operator_progression'),
                    skill_overrides=stored_config.get('skill_overrides'))
                coordinator.load_state_dict(data)
            elif self.squad_coordinator:
                raise ValueError("This checkpoint does not contain a squad-selection network")
        result = self._evaluate_model(model, seeds, squad_coordinator=coordinator)
        self._emit(
            "evaluation_finished",
            **{k: v for k, v in result.items() if k != "episode_results"},
        )
        return result

    def _training_step(self):
        started = perf_counter()
        self.model.train()
        batch = collate_samples(
            self.buffer.sample(self.config["batch_size"]), self.device
        )
        before = [p.detach().clone() for p in self.model.parameters()]
        logits, values = self.model(
            batch["state_features"], batch["action_features"], batch["mask"]
        )
        policy_loss = masked_policy_loss(logits, batch["policy"], batch["mask"])
        value_loss = torch.nn.functional.mse_loss(values.reshape(-1), batch["z"])
        # AdamW performs decoupled decay. This diagnostic is NOT added again to
        # the gradient objective, avoiding double application of regularization.
        regularization = (
            0.5
            * self.config["weight_decay"]
            * sum(p.detach().square().sum() for p in self.model.parameters())
        )
        objective = policy_loss + self.config["value_loss_weight"] * value_loss
        if not torch.isfinite(objective):
            raise FloatingPointError("Training loss is nonfinite")
        self.optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["gradient_clip"],
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        delta = sum(
            (old - p.detach()).square().sum()
            for old, p in zip(before, self.model.parameters())
        ).sqrt()
        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "total_loss": float(objective.detach().cpu()),
            "regularization_diagnostic": float(regularization.cpu()),
            "loss_including_regularization_diagnostic": float(
                (objective.detach() + regularization).cpu()
            ),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "parameter_delta": float(delta.cpu()),
            "training_step_time": perf_counter() - started,
        }

    def _evaluation_summary(self, value):
        return {k: v for k, v in value.items() if k != "episode_results"}

    def _handle_requests(self):
        if not hasattr(self._control, "take_requests"):
            return
        for command in self._control.take_requests():
            if command == "save_checkpoint":
                self.save_checkpoint()
            elif command == "evaluate":
                result = self._evaluate_model(self.model, control=self._control)
                if result is not None:
                    self._emit(
                        "evaluation_finished", **self._evaluation_summary(result)
                    )

    def train(self, callback=None, control=None):
        self._callback = callback or (lambda event: None)
        self._control = (
            control if control is not None else TrainingControl(self._callback)
        )
        started = perf_counter()
        if self.replay_recovery_notice:
            self._emit("replay_data_missing", message=self.replay_recovery_notice)
            self.replay_recovery_notice = None
        self._emit(
            "training_started",
            device=str(self.device),
            iteration=self.iteration,
            replay_buffer=len(self.buffer),
        )
        if self.best_metrics is None and self._control.checkpoint():
            baseline = self._evaluate_model(self.best_model, control=self._control)
            if baseline is not None:
                self.best_metrics = self._evaluation_summary(baseline)
                self._append_jsonl(
                    "evaluations.jsonl",
                    {"iteration": 0, "kind": "initial_network", **baseline},
                )
                self._emit("evaluation_finished", iteration=0, **self.best_metrics)
                self._save_best()
        while self.iteration < self.config["iterations"] and self._control.checkpoint():
            if self.pending is None:
                self.pending = {
                    "iteration": self.iteration + 1,
                    "episodes": [],
                    "losses": [],
                    "wall_time": 0.0,
                }
            pending = self.pending
            iteration = pending["iteration"]
            phase = self._training_phase(iteration)
            episode_target = pending.get("recovery_episode_target", self.config["episodes_per_iteration"])
            self._emit(
                "iteration_started",
                iteration=iteration,
                episodes=episode_target,
            )
            cycle_started = perf_counter()
            evaluator = NeuralEvaluator(self.model, device=self.device)
            for index in range(
                len(pending["episodes"]), episode_target
            ):
                if not self._control.checkpoint():
                    break
                if pending.get('episode_in_progress') is None:
                    pending['episode_in_progress'] = {
                        'seed': self.config["seed"] + iteration * 100000 + index,
                        'stage': self.stage_sampler.sample(self.rng),
                    }
                seed = pending['episode_in_progress']['seed']
                stage = pending['episode_in_progress']['stage']
                episode = play_episode(
                    self._env(),
                    evaluator,
                    self._episode_config(iteration),
                    seed=seed,
                    stage=stage,
                    training=True,
                    control=self._control,
                    iteration=iteration,
                    episode_index=index + 1,
                    squad_coordinator=self.squad_coordinator,
                    squad_training=phase in ('squad', 'joint'),
                    battle_training=phase in ('battle', 'joint'),
                )
                if not episode.complete:
                    break
                pending['episode_in_progress'] = None
                if phase in ('battle', 'joint'):
                    self.buffer.add_episode(episode.samples)
                    self.total_generated_states += len(episode.samples)
                pending["episodes"].append(episode.metrics)
                self._emit(
                    "episode_finished",
                    **{k: v for k, v in episode.metrics.items() if k != "history"},
                    replay_buffer=len(self.buffer),
                    generated_states=self.total_generated_states,
                )
            if self._control.stop_requested:
                pending["wall_time"] += perf_counter() - cycle_started
                break
            for _ in range(
                len(pending["losses"]), self.config["training_steps_per_iteration"]
            ):
                if phase == 'squad' or not len(self.buffer) or not any(r.get('reward_trainable', False) for r in pending['episodes']):
                    break
                if not self._control.checkpoint():
                    break
                pending["losses"].append(self._training_step())
            if self._control.stop_requested:
                pending["wall_time"] += perf_counter() - cycle_started
                break
            evaluation = self._evaluate_model(self.model, control=self._control)
            if evaluation is None:
                pending["wall_time"] += perf_counter() - cycle_started
                break
            candidate_metrics = self._evaluation_summary(evaluation)
            curriculum_advanced = self.stage_sampler.observe(evaluation['episode_results'])
            promoted, reason = promotion_decision(
                candidate_metrics, self.best_metrics, self.config["promote_on_equal"]
            )
            if promoted:
                self.best_model.load_state_dict(self.model.state_dict())
                self.best_model.eval()
                self.best_metrics = deepcopy(candidate_metrics)
                self.best_iteration = iteration
                if self.squad_coordinator:
                    self.best_squad_coordinator.selector.load_state_dict(self.squad_coordinator.selector.state_dict())
            self.iteration = iteration
            rows = pending["episodes"]
            losses = pending["losses"]
            metrics = {
                "iteration": iteration,
                "reward_version": REWARD_VERSION,
            "skill_engine_version": SKILL_ENGINE_VERSION,
                "training_phase": phase,
                "optimizer_steps": len(losses),
                "curriculum_advanced": curriculum_advanced,
                "active_stages": list(self.stage_sampler.active_stages),
                "selfplay_average_return": _mean(rows, 'return'),
                "selfplay_reward_min": min(r['return'] for r in rows),
                "selfplay_reward_max": max(r['return'] for r in rows),
                "selfplay_trainable_episodes": sum(r['reward_trainable'] for r in rows),
                "selfplay_immediate_retreats": sum(r['immediate_retreats'] for r in rows),
                "evaluation_win_rate": candidate_metrics['win_rate'],
                "evaluation_truncated_fraction": candidate_metrics['truncated_fraction'],
                "evaluation_average_immediate_retreats": candidate_metrics['average_immediate_retreats'],
                "device": str(self.device),
                "episodes": len(rows),
                "generated_states": self.total_generated_states,
                "replay_buffer": len(self.buffer),
                "states_generated": sum(row["decisions"] for row in rows),
                "replay_size": len(self.buffer),
                "replay_buffer_capacity": self.buffer.capacity,
                "mcts_simulations": self.config["mcts_simulations"],
                "success_rate": candidate_metrics["success_rate"],
                "selfplay_success_rate": _mean(rows, "success"),
                "selfplay_win_rate": _mean(rows, "win"),
                "evaluation_success_rate": candidate_metrics["success_rate"],
                "average_leaks": candidate_metrics["average_leaks"],
                "average_kills": candidate_metrics["average_kills"],
                "evaluation_average_leaks": candidate_metrics["average_leaks"],
                "evaluation_average_kills": candidate_metrics["average_kills"],
                "evaluation_average_decisions": candidate_metrics["average_decisions"],
                "evaluation_natural_terminal_fraction": candidate_metrics[
                    "natural_terminal_fraction"
                ],
                "average_decisions": _mean(rows, "decisions"),
                "average_decision_nodes": _mean(rows, "decision_nodes"),
                "simulation_ticks": sum(r['simulation_ticks'] for r in rows),
                "decision_nodes": sum(r['decision_nodes'] for r in rows),
                "ticks_per_decision": sum(r['simulation_ticks'] for r in rows) / max(1, sum(r['decision_nodes'] for r in rows)),
                "average_clear_time": summarize_episodes(rows)['average_clear_time'],
                "evaluation_average_clear_time": candidate_metrics['average_clear_time'],
                "action_counts": {key: sum(r['action_counts'][key] for r in rows) for key in ACTION_CATEGORIES},
                **{f'{key.lower()}_percentage': 100 * sum(r['action_counts'][key] for r in rows) / max(1, sum(r['decisions'] for r in rows)) for key in ACTION_CATEGORIES},
                "planner_enabled": self.config['planner_enabled'],
                "future_events_enabled": self.config['future_events_enabled'],
                "top_k_actions": self.config['top_k_actions'],
                "episode_duration": _mean(rows, "wall_time"),
                "average_decision_time": _mean(rows, "average_decision_time"),
                "forced_action_fraction": _mean(rows, "forced_action_fraction"),
                "mean_value": _mean(rows, "mean_value"),
                "policy_entropy": _mean(rows, "policy_entropy"),
                "search_policy_entropy": _mean(rows, "search_policy_entropy"),
                "actual_simulations": sum(r["actual_simulations"] for r in rows),
                "mcts_nodes": sum(r["mcts_nodes"] for r in rows),
                "average_nodes_per_decision": sum(r["mcts_nodes"] for r in rows)
                / max(1, sum(r["decisions"] for r in rows)),
                "natural_terminal_fraction": sum(
                    r["termination"] == "battle_end" for r in rows
                )
                / len(rows),
                "decision_limit_fraction": sum(
                    r["termination"] == "decision_limit" for r in rows
                )
                / len(rows),
                "time_limit_fraction": sum(
                    r["termination"] == "time_limit" for r in rows
                )
                / len(rows),
                "promoted": promoted,
                "promotion_reason": reason,
                "best_iteration": self.best_iteration,
                "temperature": self._episode_config(iteration)["temperature"],
                "prior_uniform_mix": self._episode_config(iteration)[
                    "prior_uniform_mix"
                ],
                "iteration_wall_time": pending["wall_time"]
                + perf_counter()
                - cycle_started,
                "training_wall_time": perf_counter() - started,
                **summarize_updates(losses, rows),
            }
            self.history.append(metrics)
            self._append_jsonl(
                "episodes.jsonl", {"iteration": iteration, "episodes": rows}
            )
            self._append_jsonl(
                "evaluations.jsonl",
                {"iteration": iteration, "kind": "candidate", **evaluation},
            )
            self._append_jsonl("training_metrics.jsonl", metrics)
            self.pending = None
            if iteration % self.config["checkpoint_interval"] == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"iteration_{iteration:06d}.pt"
                )
            self.save_checkpoint()
            if promoted:
                self._save_best()
            self._emit("evaluation_finished", iteration=iteration, **candidate_metrics)
            self._emit("training_metrics", **metrics)
            self._handle_requests()
        if self.pending is not None or self._control.stop_requested:
            self.save_checkpoint()
        summary = {
            "iterations_completed": self.iteration,
            "stopped": self._control.stop_requested,
            "replay_buffer": len(self.buffer),
            "generated_states": self.total_generated_states,
            "best_iteration": self.best_iteration,
            "best_metrics": self.best_metrics,
            "wall_time": perf_counter() - started,
            "history": self.history,
        }
        self._emit(
            "training_finished", **{k: v for k, v in summary.items() if k != "history"}
        )
        return summary

    def _checkpoint_payload(self, path=None):
        rng = {"python": self.rng.getstate(), "torch": torch.get_rng_state()}
        if str(self.device) == "mps" and hasattr(torch.mps, "get_rng_state"):
            rng["mps"] = torch.mps.get_rng_state().cpu()
        return {
            "format_version": 2,
            "reward_version": REWARD_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "event_graph_version": EVENT_GRAPH_VERSION,
            "skill_engine_version": SKILL_ENGINE_VERSION,
            "curriculum_state": self.stage_sampler.state_dict(),
            "model_state_dict": _cpu_tree(self.model.state_dict()),
            "best_model_state_dict": _cpu_tree(self.best_model.state_dict()),
            "optimizer_state_dict": _cpu_tree(self.optimizer.state_dict()),
            "config": deepcopy(self.config),
            "iteration": self.iteration,
            "best_iteration": self.best_iteration,
            "best_metrics": deepcopy(self.best_metrics),
            "replay_buffer": self.replay_storage.save(self.buffer, path or self.checkpoint_dir / "latest.pt"),
            "rng_state": rng,
            "pending_iteration": deepcopy(self.pending),
            "history": deepcopy(self.history),
            "generated_states": self.total_generated_states,
            "network_metadata": self.model.architecture_config,
            "weight_migration": deepcopy(self.migration_report),
            "regularization": "AdamW decoupled weight_decay; diagnostic L2 is not added to the gradient loss",
            "squad_selector": _cpu_tree(self.squad_coordinator.state_dict()) if self.squad_coordinator else None,
            "best_squad_selector": _cpu_tree(self.best_squad_coordinator.state_dict(False)) if self.best_squad_coordinator else None,
        }

    def save_checkpoint(self, path=None):
        path = Path(path) if path is not None else self.checkpoint_dir / "latest.pt"
        destination = atomic_torch_save(
            self._checkpoint_payload(path), path
        )
        self._emit("checkpoint_saved", path=str(destination), iteration=self.iteration)
        return destination

    def _save_best(self):
        # Inference-only best snapshot intentionally has no mismatched optimizer.
        payload = {
            "format_version": 1,
            "reward_version": REWARD_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "event_graph_version": EVENT_GRAPH_VERSION,
            "skill_engine_version": SKILL_ENGINE_VERSION,
            "model_state_dict": _cpu_tree(self.best_model.state_dict()),
            "config": deepcopy(self.config),
            "iteration": self.best_iteration,
            "evaluation": deepcopy(self.best_metrics),
            "best_metrics": deepcopy(self.best_metrics),
            "network_metadata": self.best_model.architecture_config,
            "inference_only": True,
            "squad_selector": _cpu_tree(self.best_squad_coordinator.state_dict(False)) if self.best_squad_coordinator else None,
        }
        atomic_torch_save(payload, self.checkpoint_dir / "best.pt")

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("inference_only"):
            raise ValueError(
                "best.pt is inference-only; resume from latest.pt or iteration checkpoint"
            )
        if payload.get("format_version") not in (1, 2):
            raise ValueError("Unsupported training checkpoint")
        if payload.get('reward_version') != REWARD_VERSION:
            raise ValueError('奖励规则已更新；旧版存档不能直接续训。请新建实验，通过 warm_start 导入权重，旧经验和优化器不会混用。')
        if payload.get('network_metadata',{}).get('version',1)!=self.model.architecture_config['version']:
            raise ValueError('此存档使用旧版战斗观察格式；请新建实验并通过 warm_start 迁移权重，不能混用旧回放和优化器状态。')
        if (payload.get('observation_version') != OBSERVATION_VERSION
                or payload.get('event_graph_version') != EVENT_GRAPH_VERSION):
            raise ValueError('存档的观察或事件图版本不兼容；请通过 warm_start 迁移权重，不能混用旧回放和优化器。')
        if payload.get('skill_engine_version', 1) != SKILL_ENGINE_VERSION:
            raise ValueError('技能实现已更新；旧版训练经验不能直接续训，请通过 warm_start 迁移权重。')
        # Runtime paths/device and requested total iterations may change; model,
        # data-generating and optimizer settings must remain consistent on resume.
        override = {
            k: self.config[k]
            for k in (
                "device",
                "num_threads",
                "iterations",
                "checkpoint_dir",
                "output_dir",
                "training_data_dir",
                "resume_from",
            )
        }
        self.config = validate_config({**payload["config"], **override})
        self.replay_storage = ReplayStorage(self.config)
        for model in (self.model, self.best_model):
            model.future_events_enabled = self.config['future_events_enabled']
            model.architecture_config['future_events_enabled'] = self.config['future_events_enabled']
        self.migration_report = payload.get('weight_migration')
        self.stage_sampler = self._make_stage_sampler()
        self.stage_sampler.load_state_dict(payload['curriculum_state'])
        self._cached_env = None
        self._init_squad_coordinator()
        if self.squad_coordinator:
            self.squad_coordinator.load_state_dict(payload["squad_selector"])
            self.best_squad_coordinator.load_state_dict(payload["best_squad_selector"])
        self.model.load_state_dict(payload["model_state_dict"])
        self.best_model.load_state_dict(payload["best_model_state_dict"])
        self.best_model.eval()
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)
        missing_replay = False
        if payload["format_version"] == 1:
            self.buffer.load_state_dict(payload["replay_buffer"])
        else:
            missing_replay = self.replay_storage.load(self.buffer, payload["replay_buffer"], path)
        self.iteration = payload["iteration"]
        self.best_iteration = payload["best_iteration"]
        self.best_metrics = payload["best_metrics"]
        self.pending = payload["pending_iteration"]
        self.replay_recovery_notice = None
        if missing_replay:
            self.replay_recovery_notice = (
                "历史训练数据已清理或未随模型一起移动；模型参数和优化器状态已恢复，"
                "将重新收集训练样本。后续训练轨迹可能与保留旧样本时不同。"
            )
            logger.warning(self.replay_recovery_notice)
            if self.pending is not None and self._training_phase(self.pending["iteration"]) in ("battle", "joint"):
                # Preserve completed updates/metrics. Collect replacement games
                # with new episode indices before the remaining optimizer steps.
                self.pending["recovery_episode_target"] = (
                    len(self.pending["episodes"]) + self.config["episodes_per_iteration"]
                )
        self.history = payload["history"]
        self.total_generated_states = payload["generated_states"]
        self.rng.setstate(payload["rng_state"]["python"])
        torch.set_rng_state(payload["rng_state"]["torch"])
        if str(self.device) == "mps" and "mps" in payload["rng_state"]:
            torch.mps.set_rng_state(payload["rng_state"]["mps"])
        return self
