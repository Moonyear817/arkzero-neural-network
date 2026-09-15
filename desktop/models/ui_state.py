"""Small presentation state, independent from live core objects."""

from dataclasses import dataclass


@dataclass
class UIState:
    simulator: str = "READY"
    training: str = "IDLE"
    device: str = "CPU"
    mps: str = "Checking"
    model: str = "None"
    stage: str = "0-1"

