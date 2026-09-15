"""Bounded research tasks run off the GUI thread and report detached values."""

import threading
import traceback

from PySide6.QtCore import QThread, Signal


class ResearchCancelled(Exception):
    """Cooperative cancellation; never a worker failure."""


class EvaluationWorker(QThread):
    completed = Signal(object)
    error = Signal(str, str)
    cancelled = Signal()

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self._cancel = threading.Event()

    def stop(self):
        self._cancel.set()

    def check_cancelled(self):
        if self._cancel.is_set():
            raise ResearchCancelled()

    def run(self):
        try:
            self.check_cancelled()
            result = self.operation(self.check_cancelled)
            self.check_cancelled()
            self.completed.emit(result)
        except ResearchCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 -- worker boundary preserves every traceback
            self.error.emit(str(exc), traceback.format_exc())
