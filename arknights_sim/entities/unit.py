from dataclasses import dataclass, field
from arknights_sim.skills.skill import SkillState
from arknights_sim.skills.modifier import Modifier


@dataclass
class Unit:
    id: str
    data: object
    hp: float
    position: tuple[float, float]
    alive: bool = True
    escaped: bool = False
    blocked_by: str | None = None
    blocked_enemies: list[str] = field(default_factory=list)
    route_index: int = 0
    node: int = 0
    wait_until: float = 0
    ready_at: float = 0
    direction: int = 0
    deployed_at: float = 0
    paid_cost: int = 0
    skill: SkillState = field(default_factory=SkillState)
    modifiers: list[Modifier] = field(default_factory=list)
    generation: int = 0
    owner_id: str | None = None
    summon_card_id: str | None = None
    hidden: bool = False
    self_damage_at: float = 0
    enemy_skill_ready: dict[int, float] = field(default_factory=dict)
    burn_started: float = 0
    burn_until: float = 0
    burn_base: float = 0
    burn_growth: float = 0
    burn_ramp: float = 1
    burn_generation: int = 0
    cold_until: float = 0
    frozen_until: float = 0
    silence_until: float = 0
    tremble_until: float = 0
    attack_revision: int = 0
    maximum_hp: float | None = None
    talent_state: dict = field(default_factory=dict)
    dying_until: float = 0
    last_attack_end: float = 0

    @property
    def max_hp(self):
        return self.data.hp if self.maximum_hp is None else self.maximum_hp
