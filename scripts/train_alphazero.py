"""Run the existing simulator through neural PUCT, self-play and AdamW training."""

import argparse
import json
import logging
import signal
from pathlib import Path
from common import ROOT


def main():
    from training.config import load_config
    from training.trainer import AlphaZeroTrainer
    from training.events import TrainingControl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/alphazero_001.yaml")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--simulations", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    options = {
        "iterations": args.iterations, "episodes_per_iteration": args.episodes,
        "mcts_simulations": args.simulations, "training_steps_per_iteration": args.steps,
        "evaluation_episodes": args.eval_episodes, "device": args.device,
    }
    config.update({key: value for key, value in options.items() if value is not None})
    if args.run_dir:
        run_dir = args.run_dir.resolve()
        config["output_dir"] = str(run_dir)
        config["checkpoint_dir"] = str(run_dir / "checkpoints")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("arknights.alphazero")
    trainer = AlphaZeroTrainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    control = TrainingControl()

    def request_stop(signum, frame):
        logger.info("Stop requested. Finishing the current safe unit of work and saving state.")
        control.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def report(event):
        kind = event.kind if hasattr(event, "kind") else event["kind"]
        payload = event.payload if hasattr(event, "payload") else event.get("payload", {})
        if kind in ("iteration_started", "episode_finished", "training_metrics", "evaluation_finished", "checkpoint_saved", "training_finished", "error"):
            logger.info("%s %s", kind, json.dumps(dict(payload), ensure_ascii=False, default=str))

    summary = trainer.train(callback=report, control=control)
    logger.info("Training summary: %s", json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

