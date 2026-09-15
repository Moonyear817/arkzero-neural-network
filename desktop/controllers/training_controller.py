"""GUI-thread lifecycle and controls for the existing AlphaZeroTrainer."""

from copy import deepcopy

from PySide6.QtCore import QObject, Signal, Slot

from desktop.services.config_service import ConfigService
from desktop.services.training_service import TrainingService
from desktop.workers.training_worker import TrainingWorker


class TrainingController(QObject):
    metrics_updated = Signal(object)
    event_received = Signal(object)
    state_changed = Signal(str)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, parent=None, trainer_factory=None, config_service=None):
        super().__init__(parent)
        self.service = TrainingService(trainer_factory)
        self.config_service = config_service or ConfigService()
        self._state = "IDLE"
        self._worker = None
        self._last_config = None
        self._last_checkpoint = None
        self.last_metrics = {}

    @property
    def availability(self):
        return self.service.available

    @property
    def available(self):
        return self.availability

    @property
    def state(self):
        return self._state

    @property
    def is_busy(self):
        return self._worker is not None

    def _set_state(self, state):
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    def start(self, config):
        return self._launch(config, "train")

    def _launch(self, config, mode):
        from desktop.services.background_training import read_background_training
        if read_background_training().get("active"):
            self.error.emit("后台自主编队训练仍在运行。请先停止后台训练并等待保存完成。")
            return False
        if self.is_busy:
            self.log.emit("A training operation is already active.")
            return False
        if not self.availability:
            self.error.emit(self.service.unavailable_reason)
            return False
        try:
            config = self.config_service.prepare_runtime(config)
            self.config_service.save_session(config)
        except Exception as exc:  # noqa: BLE001 — GUI/worker exception boundary
            self.error.emit(str(exc))
            return False
        self._last_config = deepcopy(config)
        worker = TrainingWorker(self.service, config, mode=mode, parent=self)
        self._worker = worker
        worker.event_received.connect(self._on_event)
        worker.metrics_updated.connect(self._on_metrics)
        worker.error.connect(self._on_error)
        worker.log.connect(self.log)
        worker.finished.connect(self._on_finished)
        self._set_state("EVALUATING" if mode == "evaluate" else "STARTING")
        worker.start()
        return True

    @Slot(object)
    def _on_event(self, event):
        if event.kind == "training_started" and self.state == "STARTING":
            self._set_state("RUNNING")
        elif event.kind == "paused" and self.state == "PAUSING":
            self._set_state("PAUSED")
        elif event.kind == "checkpoint_saved":
            self._last_checkpoint = event.payload.get("path")
        self.event_received.emit(event)

    @Slot(object)
    def _on_metrics(self, metrics):
        self.last_metrics = dict(metrics)
        self.metrics_updated.emit(metrics)

    @Slot(str)
    def _on_error(self, message):
        self._set_state("ERROR")
        self.error.emit(message)

    @Slot()
    def _on_finished(self):
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        if self.state != "ERROR":
            self._set_state("IDLE")
        else:
            # Refresh view controls now that a failed worker has actually exited.
            self.state_changed.emit("ERROR")

    def pause(self):
        if self._worker is not None and self.state in ("STARTING", "RUNNING"):
            self._worker.command("pause")
            self._set_state("PAUSING")

    def resume(self):
        if self._worker is not None and self.state in ("PAUSING", "PAUSED"):
            self._worker.command("resume")
            self._set_state("RUNNING")

    def stop(self):
        if self._worker is not None and self.state != "STOPPING":
            self._worker.command("stop")
            self._set_state("STOPPING")
            self.log.emit(
                "Stop requested; completing the current safe unit and preserving a resumable checkpoint."
            )

    def save_checkpoint(self):
        if self._worker is not None and self.state not in (
            "EVALUATING",
            "STOPPING",
            "ERROR",
        ):
            self._worker.command("save_checkpoint")
            self.log.emit(
                "Checkpoint request pending until the next completed iteration. Resume if paused. Stop also saves completed progress."
            )

    def evaluate(self, config=None):
        if self._worker is not None:
            if self.state not in ("EVALUATING", "STOPPING", "ERROR"):
                self._worker.command("evaluate")
                self.log.emit(
                    "Evaluation request pending until the next completed iteration. Resume if paused."
                )
            return False
        config = deepcopy(config or self._last_config or self.config_service.load())
        if self._last_checkpoint:
            config["evaluation_checkpoint"] = self._last_checkpoint
        return self._launch(config, "evaluate")

    def set_evaluation_checkpoint(self, path):
        """Select a model for evaluation; training resumes only via explicit YAML."""
        self._last_checkpoint = str(path)

    def shutdown(self, timeout_ms=5000):
        """Request safe exit. False means caller must keep the worker alive."""
        if self._worker is None:
            return True
        self.stop()
        return self._worker.wait(timeout_ms)
