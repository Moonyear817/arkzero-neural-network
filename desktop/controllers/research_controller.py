"""Small shared lifecycle for one cancellable research computation at a time."""

import logging

from PySide6.QtCore import QObject, Signal, Slot

from desktop.workers.evaluation_worker import EvaluationWorker


class ResearchController(QObject):
    error = Signal(str, str)
    status_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._receiver = None
        self.status = "IDLE"

    def is_busy(self):
        return self._worker is not None

    def _status(self, value):
        self.status = value
        self.status_changed.emit(value)

    def _start(self, operation, receiver):
        if self.is_busy():
            return False
        self._receiver = receiver
        worker = EvaluationWorker(operation, self)
        self._worker = worker
        worker.completed.connect(self._completed)
        worker.error.connect(self._failed)
        worker.cancelled.connect(self._cancelled)
        worker.finished.connect(self._finished)
        self._status("RUNNING")
        worker.start()
        return True

    @Slot(object)
    def _completed(self, result):
        if self._receiver is not None:
            self._receiver(result)
        self._status("READY")

    @Slot(str, str)
    def _failed(self, message, detail):
        logging.getLogger("arknights.desktop").error(
            "%s\n%s", message, detail, extra={"category": "ERROR"}
        )
        self._status("ERROR")
        self.error.emit(message, detail)

    @Slot()
    def _cancelled(self):
        self._status("STOPPED")

    @Slot()
    def _finished(self):
        worker = self._worker
        self._worker = None
        self._receiver = None
        if worker is not None:
            worker.operation = None
            worker.deleteLater()
        self.status_changed.emit(self.status)

    def stop(self):
        if self._worker is not None and self.status != "STOPPING":
            self._status("STOPPING")
            self._worker.stop()

    def shutdown(self, timeout_ms=5000):
        self.stop()
        return self._worker is None or self._worker.wait(timeout_ms)
