from dataclasses import dataclass


@dataclass
class SimulationClock:
    time: float = 0
    tick: int = 0
    dt: float = 1 / 60

    def next_tick(self):
        return round((self.tick + 1) * self.dt, 9)
