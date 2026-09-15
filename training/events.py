"""Qt-free training events and cooperative safe-boundary controls."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Condition
from types import MappingProxyType


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(f"Event payload must be plain data: {type(value).__name__}")


@dataclass(frozen=True)
class TrainingEvent:
    kind: str
    payload: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "payload", _freeze(self.payload))


class TrainingControl:
    """Pause/stop requests are observed between decisions and optimizer steps.

    checkpoint() is called by the computation thread. Saving/evaluation requests
    are consumed only by the trainer at iteration boundaries. No forced thread
    termination occurs. Stop wakes a paused computation and preserves completed
    episodes; a partially generated episode is never added to replay.
    """

    def __init__(self, callback=None):
        self._condition = Condition()
        self._paused = False
        self._stop = False
        self._requests = deque()
        self.callback = callback or (lambda event: None)

    @property
    def stop_requested(self):
        with self._condition:
            return self._stop

    def pause(self):
        with self._condition:
            self._paused = True

    def resume(self):
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()

    def request(self, command):
        if command not in ("save_checkpoint", "evaluate"):
            raise ValueError(f"Unknown control request: {command}")
        with self._condition:
            self._requests.append(command)

    def take_requests(self):
        with self._condition:
            requests = tuple(self._requests)
            self._requests.clear()
            return requests

    def checkpoint(self):
        with self._condition:
            announced = False
            while self._paused and not self._stop:
                if not announced:
                    self.callback(TrainingEvent("paused"))
                    announced = True
                self._condition.wait()
            return not self._stop
