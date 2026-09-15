"""Main-thread lifecycle adapter; simulator state is owned by its worker."""

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from desktop.models.simulation_state import SUPPORTED_SQUAD
from desktop.workers.simulation_worker import SimulationWorker


class SimulatorController(QObject):
    state_changed = Signal(str)
    snapshot = Signal(object)
    events = Signal(object)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, parent=None, data_dir=None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.status = "IDLE"
        self.last_snapshot = None
        self._worker = None
        self._last_options = None
        self._restart_options = None
        self._fps = 20

    def is_busy(self):
        return self._worker is not None

    def start(
        self,
        stage_id="0-1",
        squad=SUPPORTED_SQUAD,
        seed=12345,
        speed=1.0,
        mode="Scripted baseline",
        fps=None,
        checkpoint=None,
        auto_squad=None,
        progression=None,
        skill_overrides=None,
    ):
        if self.is_busy():
            self.error.emit(
                "A simulation is already active; stop it before starting another"
            )
            return False
        options = {
            "stage_id": stage_id,
            "squad": tuple(squad),
            "seed": int(seed),
            "speed": float(speed),
            "mode": mode,
            "fps": self._fps if fps is None else fps,
            "checkpoint": checkpoint,
            "auto_squad": auto_squad,
            "progression": dict(progression or {}),
            "skill_overrides": dict(skill_overrides or {}),
        }
        try:
            worker = SimulationWorker(**options, data_dir=self.data_dir, parent=self)
        except (ValueError, TypeError) as exc:
            self._on_state("ERROR")
            self.error.emit(str(exc))
            return False
        self._last_options = options
        self.last_snapshot = None
        self._worker = worker
        worker.state_changed.connect(self._on_state, Qt.ConnectionType.QueuedConnection)
        worker.snapshot.connect(self._on_snapshot, Qt.ConnectionType.QueuedConnection)
        worker.events.connect(self.events.emit, Qt.ConnectionType.QueuedConnection)
        worker.error.connect(self.error.emit, Qt.ConnectionType.QueuedConnection)
        worker.log.connect(self.log.emit, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._finished, Qt.ConnectionType.QueuedConnection)
        self._on_state("LOADING")
        worker.start()
        return True

    def pause(self):
        if self._worker is not None:
            self._worker.pause()

    def resume(self):
        if self._worker is not None:
            self._worker.resume()

    def step_once(self):
        if self._worker is not None:
            self._worker.step_once()

    def stop(self):
        self._restart_options = None
        if self._worker is not None:
            self._on_state("STOPPING")
            self._worker.stop()

    def restart(self):
        if self._last_options is None:
            return self.start()
        if self._worker is None:
            return self.start(**self._last_options)
        self._restart_options = dict(self._last_options)
        self._on_state("STOPPING")
        self._worker.stop()
        return True

    def submit_action(self, payload_json):
        if self._worker is None:
            self.error.emit("Start a manual simulation first")
        else:
            self._worker.submit_action(payload_json)

    def set_speed(self, speed):
        if self._last_options is not None:
            self._last_options["speed"] = float(speed)
        if self._worker is not None:
            self._worker.set_speed(speed)

    def set_fps(self, fps):
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("Visualization FPS must be in [1, 60]")
        self._fps = fps
        if self._last_options is not None:
            self._last_options["fps"] = fps
        if self._worker is not None:
            self._worker.set_fps(fps)

    def shutdown(self, timeout_ms=0):
        """Request a safe stop and optionally wait; zero is nonblocking."""
        self._restart_options = None
        worker = self._worker
        if worker is None:
            return True
        worker.stop()
        return worker.wait(max(0, int(timeout_ms)))

    @Slot(str)
    def _on_state(self, state):
        self.status = state
        self.state_changed.emit(state)

    @Slot(object)
    def _on_snapshot(self, snapshot):
        self.last_snapshot = snapshot
        self.snapshot.emit(snapshot)

    @Slot()
    def _finished(self):
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        options, self._restart_options = self._restart_options, None
        if options is not None:
            QTimer.singleShot(0, lambda: self.start(**options))
