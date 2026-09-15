"""Process telemetry and a one-time asynchronous compute-device probe."""

import importlib.util
import traceback
from types import MappingProxyType

import psutil
from PySide6.QtCore import QObject, QThread, QTimer, Signal


class DeviceProbe(QThread):
    result = Signal(object)
    error = Signal(str)

    def run(self):
        try:
            if importlib.util.find_spec("torch") is None:
                self.result.emit(MappingProxyType({"mps": "Unavailable", "training_available": False}))
                return
            import torch
            self.result.emit(MappingProxyType({
                "mps": "Available" if torch.backends.mps.is_available() else "Unavailable",
                "training_available": importlib.util.find_spec("training.trainer") is not None,
            }))
        except Exception:
            self.error.emit(traceback.format_exc())


class SystemController(QObject):
    metrics_updated = Signal(object)
    device_detected = Signal(object)
    error = Signal(str)

    def __init__(self, parent=None, interval=1000):
        super().__init__(parent)
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.timer = QTimer(self)
        self.timer.setInterval(interval)
        self.timer.timeout.connect(self.sample)
        self.probe = DeviceProbe(self)
        self.probe.result.connect(self.device_detected)
        self.probe.error.connect(self.error)

    def start(self):
        self.timer.start()
        self.probe.start()
        self.sample()

    def sample(self):
        self.metrics_updated.emit(MappingProxyType({
            "cpu_percent": self.process.cpu_percent(),
            "rss_mb": self.process.memory_info().rss / 1024 ** 2,
        }))

    def shutdown(self):
        self.timer.stop()
        return not self.probe.isRunning()
