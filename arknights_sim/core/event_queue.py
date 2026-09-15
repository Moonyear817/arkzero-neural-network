from dataclasses import dataclass, field
import heapq
import math

from .event_priority import event_order_key


@dataclass(order=True, frozen=True)
class Event:
    # Keep the public positional constructor; only this explicit key compares.
    sort_key: tuple = field(init=False, repr=False)
    time: float = field(compare=False)
    sequence: int = field(compare=False)
    kind: str = field(compare=False)
    payload: tuple = field(compare=False, default=())

    def __post_init__(self):
        priority, semantic_key = event_order_key(self.kind, self.payload)
        # Sequence breaks ties only between semantically identical events.
        object.__setattr__(
            self, "sort_key", (self.time, priority, semantic_key, self.sequence)
        )


class EventQueue:
    def __init__(self):
        self.heap = []
        self.sequence = 0

    def push(self, time, kind, *payload):
        if not math.isfinite(time) or time < 0:
            raise ValueError("Event time must be finite and nonnegative")
        heapq.heappush(self.heap, Event(round(time, 9), self.sequence, kind, payload))
        self.sequence += 1

    def pop(self):
        return heapq.heappop(self.heap)

    def peek(self):
        return self.heap[0].time if self.heap else float("inf")
