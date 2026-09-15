"""Checkpoint list metadata is derived from matching events, without torch I/O."""

import json

from desktop.models.training_state import TrainingEvent
from desktop.workers.training_worker import TrainingWorker


def test_sidecar_matches_iteration_when_metrics_arrive_after_save(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint fixture; intentionally not a torch file")
    worker = TrainingWorker(None, {"checkpoint_dir": str(tmp_path)})
    worker._receive(
        TrainingEvent(
            "training_metrics",
            {"iteration": 1, "policy_loss": 0.8, "best_iteration": 1},
        )
    )
    worker._receive(
        TrainingEvent("checkpoint_saved", {"iteration": 2, "path": str(checkpoint)})
    )
    path = checkpoint.with_suffix(".pt.json")
    pending = json.loads(path.read_text())
    assert pending["iteration"] == 2
    assert pending["metrics"] == {}
    assert pending["best_iteration"] is None
    worker._receive(
        TrainingEvent(
            "training_metrics",
            {
                "iteration": 2,
                "policy_loss": 0.4,
                "success_rate": 0.5,
                "best_iteration": 1,
            },
        )
    )
    final = json.loads(path.read_text())
    assert final["metrics"]["policy_loss"] == 0.4
    assert final["best_iteration"] == 1
    assert not tuple(tmp_path.glob("*.tmp"))


def test_late_old_metrics_do_not_relabel_new_latest(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"fixture")
    worker = TrainingWorker(None, {"checkpoint_dir": str(tmp_path)})
    worker._receive(
        TrainingEvent("checkpoint_saved", {"iteration": 1, "path": str(checkpoint)})
    )
    worker._receive(
        TrainingEvent("checkpoint_saved", {"iteration": 2, "path": str(checkpoint)})
    )
    worker._receive(
        TrainingEvent("training_metrics", {"iteration": 1, "policy_loss": 0.9})
    )
    metadata = json.loads(checkpoint.with_suffix(".pt.json").read_text())
    assert metadata["iteration"] == 2 and metadata["metrics"] == {}


def test_published_checkpoints_and_best_are_not_modified(tmp_path, monkeypatch):
    monkeypatch.setattr("desktop.workers.training_worker.WORKSPACE_ROOT", tmp_path)
    published = tmp_path / "checkpoints"
    published.mkdir()
    checkpoint = published / "latest.pt"
    checkpoint.write_bytes(b"public")
    worker = TrainingWorker(None, {"checkpoint_dir": str(published)})
    worker._receive(
        TrainingEvent("checkpoint_saved", {"iteration": 3, "path": str(checkpoint)})
    )
    assert not checkpoint.with_suffix(".pt.json").exists()
    best = tmp_path / "best.pt"
    best.write_bytes(b"old best")
    worker = TrainingWorker(None, {"checkpoint_dir": str(tmp_path)})
    worker._receive(
        TrainingEvent("checkpoint_saved", {"iteration": 3, "path": str(best)})
    )
    assert not best.with_suffix(".pt.json").exists()
