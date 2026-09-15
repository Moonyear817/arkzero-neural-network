from dataclasses import dataclass


@dataclass(frozen=True)
class AttackEvent:
    attacker: str
    target: str
    start: float
    hit: float
    next_attack: float
    generation: int
    target_generation: int
    skill_effect: object | None = None
    hit_index: int = 0
    sp_eligible: bool = True
    attack_scale: float = 1
    attack_revision: int = 0
    talent_procs: tuple[str, ...] = ()
    damage_type: str | None = None
    fixed_atk: float | None = None
    is_dot: bool = False
    is_reflection: bool = False
    target_index: int = 0
