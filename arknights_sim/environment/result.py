from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    kills: int
    leaks: int
    life: int
    total_enemies: int
    terminal: bool
    termination: str | None
    simulator_result: str | None
    initial_life: int = 1
    escorts_total: int = 0
    escorts_saved: int = 0
    escorts_alive: int = 0
    escorts_dead: int = 0
    protection_success: bool = True

    def to_dict(self):
        return asdict(self)
