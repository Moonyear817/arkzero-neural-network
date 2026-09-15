"""Lazy adapter to the actual Qt-free AlphaZero core; no training algorithms."""

from importlib.util import find_spec
from pathlib import Path


class TrainingService:
    def __init__(self, trainer_factory=None):
        self.trainer_factory = trainer_factory

    @property
    def available(self):
        # Looking up a top-level package spec does not import torch or training.
        return self.trainer_factory is not None or (
            find_spec("torch") is not None and find_spec("training") is not None
        )

    @property
    def unavailable_reason(self):
        return "AlphaZero core or PyTorch is unavailable in this installation."

    def build_trainer(self, config):
        """Called by TrainingWorker.run, never by a GUI button callback."""
        if self.trainer_factory is not None:
            return self.trainer_factory(config)
        from training import AlphaZeroTrainer

        return AlphaZeroTrainer(config)

    def make_control(self, callback):
        from training.events import TrainingControl

        return TrainingControl(callback)

    def evaluate(self, trainer, config, control, callback):
        """Use the core evaluator; cancellation is observed between episodes."""
        from training.events import TrainingEvent
        from training.trainer import summarize_episodes

        checkpoint = config.get("evaluation_checkpoint")
        if not checkpoint:
            checkpoint = str(Path(config["checkpoint_dir"]) / "latest.pt")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "No checkpoint is available for evaluation. Train first or select an existing checkpoint."
            )
        rows = []
        seeds = [
            config.get("evaluation_seed", 80000) + i
            for i in range(config["evaluation_episodes"])
        ]
        for seed in seeds:
            if not control.checkpoint():
                return {"stopped": True, "episodes_completed": len(rows)}
            result = trainer.evaluate(checkpoint=checkpoint, seeds=[seed])
            rows.extend(result["episode_results"])
        summary = {
            **summarize_episodes(rows),
            "seeds": seeds,
            "checkpoint": str(checkpoint),
        }
        callback(TrainingEvent("evaluation_finished", summary))
        return summary
