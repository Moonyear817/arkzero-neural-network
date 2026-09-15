"""Profile one real neural self-play episode and one replay-sampled update.

This diagnostic never runs the full trainer loop or saves a checkpoint. Timings
include cProfile overhead; cumulative function times overlap and must not be
summed. It does not supply a scripted strategy, rollout heuristic, or labels
other than the environment's terminal result.
"""

import argparse
import cProfile
import io
import json
import pstats
import tempfile
import traceback
from pathlib import Path
from time import perf_counter

from common import ROOT

from arknights_sim.core.state import GameState
from arknights_sim.environment import ArknightsEnv
from mcts import PUCTSearch
from network import ActionEncoder, NeuralEvaluator, PolicyValueNetwork, StateEncoder
from training.config import load_config
from training.replay_buffer import ReplayBuffer
from training.self_play import play_episode
from training.trainer import AlphaZeroTrainer


def _function_summary(stats, function):
    code = function.__code__
    wanted = (str(Path(code.co_filename).resolve()), code.co_firstlineno, code.co_name)
    for (filename, line, name), value in stats.stats.items():
        if (str(Path(filename).resolve()), line, name) == wanted:
            primitive_calls, total_calls, exclusive, cumulative, _ = value
            return {
                "function": f"{function.__module__}.{function.__qualname__}",
                "source": "cProfile",
                "recorded": True,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "exclusive_seconds": exclusive,
                "cumulative_seconds": cumulative,
            }
    return {
        "function": f"{function.__module__}.{function.__qualname__}",
        "source": "cProfile",
        "recorded": False,
        "primitive_calls": 0,
        "total_calls": 0,
        "exclusive_seconds": None,
        "cumulative_seconds": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alphazero_001.yaml")
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="outputs/alphazero_profile")
    parser.add_argument(
        "--concurrent-workload-note",
        default="Not specified; this script does not isolate the machine",
    )
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / args.config)
    config.update(
        device="cpu",
        num_threads=1,
        mcts_simulations=args.simulations,
        seed=args.seed,
        iterations=1,
        episodes_per_iteration=1,
        training_steps_per_iteration=1,
        resume_from=None,
    )
    report = {
        "status": "running",
        "description": "One random-initialized CPU neural PUCT training episode and one actual replay-sampled optimizer update",
        "timing_note": "cProfile overhead is included; cumulative function times overlap and are not additive",
        "concurrent_workload_note": args.concurrent_workload_note,
        "checkpoints_saved": False,
        "config": dict(config),
    }
    profiler = cProfile.Profile()
    failure = None
    # Trainer construction normally creates its output folders. This experiment
    # directs them to a temporary location and never calls trainer.train/save.
    with tempfile.TemporaryDirectory(prefix="ark_zero_profile_") as temporary:
        config.update(
            checkpoint_dir=str(Path(temporary) / "checkpoints"),
            output_dir=str(Path(temporary) / "training_logs"),
        )
        trainer = AlphaZeroTrainer(config)
        env = trainer._env()
        evaluator = NeuralEvaluator(trainer.model, device="cpu")
        started = perf_counter()
        profiler.enable()
        try:
            episode = play_episode(
                env,
                evaluator,
                trainer._episode_config(1),
                seed=args.seed,
                stage=config["stage"],
                training=True,
                iteration=1,
                episode_index=1,
            )
            episode_finished = perf_counter()
            report["episode"] = episode.metrics
            report["episode_complete"] = episode.complete
            report["episode_wall_seconds"] = episode_finished - started
            if not episode.complete or not episode.samples:
                raise RuntimeError(
                    "Episode produced no complete replay samples; update skipped"
                )
            trainer.buffer.add_episode(episode.samples)
            update_started = perf_counter()
            update_metrics = trainer._training_step()
            report["update_wall_seconds"] = perf_counter() - update_started
            report["update_metrics"] = update_metrics
            report["replay_samples"] = len(trainer.buffer)
            report["actual_batch_size"] = min(config["batch_size"], len(trainer.buffer))
            report["network_parameters"] = trainer.model.parameter_count
            report["status"] = "completed"
        except Exception:  # noqa: BLE001 -- persist partial profile and exit nonzero
            failure = traceback.format_exc()
            report["status"] = "failed"
            report["error"] = failure
        finally:
            profiler.disable()
            report["total_profiled_wall_seconds"] = perf_counter() - started

    profiler.dump_stats(str(output / "profile.prof"))
    text_output = io.StringIO()
    stats = pstats.Stats(profiler, stream=text_output)
    stats.strip_dirs().sort_stats("cumulative").print_stats(60)
    # Use unstripped filenames for exact method matching below.
    raw_stats = pstats.Stats(profiler)
    methods = {
        "state_encoding": StateEncoder.encode,
        "action_encoding": ActionEncoder.encode,
        "neural_evaluation": NeuralEvaluator.evaluate,
        "model_forward": PolicyValueNetwork.forward,
        "puct_search": PUCTSearch.search,
        "environment_step": ArknightsEnv.step,
        "state_clone": GameState.clone,
        "stable_state_hash": GameState.stable_hash,
        "replay_sample": ReplayBuffer.sample,
        "training_batch": AlphaZeroTrainer._training_step,
    }
    report["functions"] = {
        name: _function_summary(raw_stats, function)
        for name, function in methods.items()
    }
    if not report["functions"]["puct_search"]["recorded"] and "episode" in report:
        # Some Python/PyTorch combinations omit enclosing frames from cProfile
        # even when their nested calls are present. Retain this limitation and
        # use the episode runner's existing independent search wall timers.
        report["functions"]["puct_search"].update(
            source="Self-play perf_counter timers; enclosing cProfile frame absent",
            primitive_calls=report["episode"]["decisions"],
            total_calls=report["episode"]["decisions"],
            cumulative_seconds=report["episode"]["search_seconds"],
        )
    report["profiler_total_seconds"] = raw_stats.total_tt
    (output / "profile.txt").write_text(text_output.getvalue(), encoding="utf-8")
    (output / "profile.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "seconds": report["total_profiled_wall_seconds"],
            },
            indent=2,
        )
    )
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
