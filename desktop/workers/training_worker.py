"""The core trainer, tensor imports and checkpoint I/O run on this QThread."""

import json
import os
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from threading import Lock

from PySide6.QtCore import QThread, Signal

from desktop.models.training_state import TrainingEvent, normalize_event
from desktop.paths import WORKSPACE_ROOT


class TrainingWorker(QThread):
    event_received = Signal(object)
    metrics_updated = Signal(object)
    error = Signal(str)
    log = Signal(str)
    completed = Signal(object)

    def __init__(self, service, config, *, mode="train", parent=None):
        super().__init__(parent)
        self.service = service
        self.config = deepcopy(config)
        self.mode = mode
        self._control = None
        self._commands = []
        self._lock = Lock()
        self._metrics_by_iteration = {}
        self._pending_sidecars = {}

    def command(self, name):
        with self._lock:
            if self._control is None:
                self._commands.append(name)
                return
            control = self._control
        self._apply(control, name)

    @staticmethod
    def _apply(control, name):
        if name in ("pause", "resume", "stop"):
            getattr(control, name)()
        else:
            control.request(name)

    def _receive(self, event):
        event = normalize_event(event)
        if event.kind == "replay_data_missing":
            self.log.emit(event.payload["message"])
        if event.kind == "checkpoint_saved":
            self._checkpoint_metadata(event.payload)
        elif event.kind == "training_metrics":
            self._update_checkpoint_metrics(event.payload)
        self.event_received.emit(event)
        if event.kind == "training_metrics":
            self.metrics_updated.emit(event.payload)
        if event.kind in (
            "iteration_started",
            "episode_finished",
            "checkpoint_saved",
            "evaluation_finished",
            "training_finished",
        ):
            visible = {
                k: v
                for k, v in event.payload.items()
                if k
                in (
                    "iteration",
                    "episode",
                    "success",
                    "success_rate",
                    "kills",
                    "leaks",
                    "path",
                    "stopped",
                    "replay_buffer",
                )
            }
            self.log.emit(f"{event.kind}: {visible}")

    def _checkpoint_metadata(self, payload):
        if not payload.get("path") or type(payload.get("iteration")) is not int:
            return
        path = Path(payload["path"]).resolve()
        configured = Path(self.config["checkpoint_dir"]).resolve()
        published = (WORKSPACE_ROOT / "checkpoints").resolve()
        if path.parent != configured or path.parent == published or not path.is_file():
            return
        if path.name != "latest.pt" and not path.name.startswith("iteration_"):
            return
        iteration = payload["iteration"]
        metrics = self._metrics_by_iteration.get(iteration, {})
        self._write_sidecar(path, iteration, metrics)
        if not metrics:
            self._pending_sidecars[path] = iteration

    def _update_checkpoint_metrics(self, payload):
        iteration = payload.get("iteration")
        if type(iteration) is not int:
            return
        metrics = {
            key: value
            for key, value in payload.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        self._metrics_by_iteration[iteration] = metrics
        for old in sorted(self._metrics_by_iteration)[:-2]:
            del self._metrics_by_iteration[old]
        updated = []
        for path, saved_iteration in tuple(self._pending_sidecars.items()):
            if saved_iteration == iteration:
                self._write_sidecar(path, iteration, metrics)
                del self._pending_sidecars[path]
                updated.append(str(path))
        if updated:
            self.event_received.emit(
                TrainingEvent(
                    "checkpoint_metadata_updated",
                    {"iteration": iteration, "paths": updated},
                )
            )

    @staticmethod
    def _write_sidecar(path, iteration, metrics):
        """Called inside run's event callback; never loads a tensor checkpoint."""
        sidecar = path.with_suffix(path.suffix + ".json")
        payload = {
            "iteration": iteration,
            "path": str(path),
            "metrics": dict(metrics),
            "best_iteration": metrics.get("best_iteration"),
            "metadata_source": "desktop_training_events",
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=sidecar.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(sidecar)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def run(self):
        try:
            control = self.service.make_control(self._receive)
            with self._lock:
                self._control = control
                pending, self._commands = self._commands, []
            for name in pending:
                self._apply(control, name)
            if not control.checkpoint():
                self.completed.emit({"stopped": True})
                return
            self.log.emit(
                "Loading AlphaZero core and checkpoint on the background worker."
            )
            trainer = self.service.build_trainer(self.config)
            if self.mode == "evaluate":
                result = self.service.evaluate(
                    trainer, self.config, control, self._receive
                )
            else:
                result = trainer.train(callback=self._receive, control=control)
            # Return only simple detached data; no trainer/model/GameState crosses.
            self.completed.emit(
                normalize_event({"kind": "completed", "payload": result}).payload
            )
        except Exception as exc:  # noqa: BLE001 — GUI/worker exception boundary
            details = traceback.format_exc()
            self.log.emit(details)
            self.error.emit(f"{type(exc).__name__}: {exc}\n\n{details}")
