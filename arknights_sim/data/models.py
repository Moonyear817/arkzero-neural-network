"""Immutable normalized inputs. Coordinates are (col, row), origin bottom-left."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TileData:
    key: str
    height: str
    buildable: str
    passable: str
    blackboard: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class MapData:
    rows: tuple[tuple[TileData, ...], ...]  # bottom to top

    @property
    def width(self):
        return len(self.rows[0])

    @property
    def height(self):
        return len(self.rows)

    def tile(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError("Out of map")
        return self.rows[y][x]

    def ascii(self):
        def symbol(t):
            if t.key == "tile_start":
                return "R"
            if t.key == "tile_end":
                return "B"
            if t.buildable == "RANGED":
                return "H"
            if t.buildable == "NONE":
                return "#"
            return "."

        return "\n".join(" ".join(map(symbol, row)) for row in reversed(self.rows))


@dataclass(frozen=True)
class WaypointData:
    kind: str
    position: tuple[float, float]
    time: float = 0


@dataclass(frozen=True)
class RouteData:
    start: tuple[float, float]
    end: tuple[float, float]
    waypoints: tuple[WaypointData, ...] = ()
    motion: str = "WALK"
    diagonal: bool = True
    random_range: tuple[float, float] = (0, 0)


@dataclass(frozen=True)
class SpawnData:
    time: float
    enemy_id: str
    route_index: int
    wave: int = 0
    fragment: int = 0
    action_index: int = -1  # Stable graph index; -1 is a legacy absolute spawn.


@dataclass(frozen=True)
class StageActionData:
    """One parallel action group from the original level definition."""
    index: int
    event_id: str
    wave_id: int
    fragment_id: int
    action_id: int
    event_type: str
    key: str
    count: int
    pre_delay: float
    interval: float
    route_id: int = -1
    block_fragment: bool = False
    dont_block_wave: bool = False


@dataclass(frozen=True)
class FragmentData:
    pre_delay: float
    actions: tuple[StageActionData, ...]


@dataclass(frozen=True)
class WaveData:
    pre_delay: float
    post_delay: float
    fragment_delays: tuple[float, ...]
    spawns: tuple[SpawnData, ...]
    fragments: tuple[FragmentData, ...] = ()
    max_wait: float = -1
    event_scheduled: bool = False


@dataclass(frozen=True)
class EnemySkillData:
    kind: str
    initial: float
    cooldown: float
    half_hp: bool = False
    parameters: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class EnemyData:
    id: str
    name: str
    hp: float
    atk: float
    defense: float
    resistance: float
    speed: float
    interval: float
    life_cost: int = 1
    block_weight: int = 1
    damage_type: str = "PHYSICAL"
    attack_range: float = 0
    invisible: bool = False
    splash: bool = False
    weight: int = 1
    faction: str = "HOSTILE"
    motion: str = "WALK"
    death_life_cost: int = 0
    escort_attack_range: float = 0
    low_hp_attack_bonus: float = 0
    self_damage_per_second: float = 0
    prioritize_escorts: bool = False
    ranged_damage_scale: float = 1
    projectile_delay: float = 0
    projectile_radius: float = 0
    max_targets: int = 1
    lifesteal: float = 0
    skills: tuple[EnemySkillData, ...] = ()
    half_hp_defense_bonus: float = 0
    half_hp_resistance_bonus: float = 0
    freeze_immune: bool = False
    silence_immune: bool = False
    tremble_immune: bool = False
    silenceable_effects: tuple[str, ...] = ()
    special_hp: bool = False


@dataclass(frozen=True)
class SkillData:
    id: str
    cost: float
    initial_sp: float
    duration: float
    attack_multiplier: float = 1
    defense_multiplier: float = 1
    attack_speed_multiplier: float = 1
    recovery: str = "AUTO_RECOVERY"
    trigger: str = "MANUAL_TRIGGER"
    effect_target: str = "SELF"
    hp_recovery_per_sec: float = 0
    mode: str = "TIMED"
    max_charges: int = 1
    attack_scale: float = 1
    damage_type: str | None = None
    max_targets: int = 1
    hits: int = 1
    heal_ratio: float = 0
    heal_flat: float = 0
    lifesteal: float = 0
    block_delta: int = 0
    block_override: int | None = None
    stop_attack: bool = False
    prohibit_healing: bool = False
    regen_ratio: float = 0
    target_atk_multiplier: float = 1
    target_speed_multiplier: float = 1
    debuff_duration: float = 0
    physical_dodge: float = 0
    kill_refill: bool = False
    ranged_penalty_removed: bool = False
    range_override: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True)
class TalentData:
    """Resolved, progression-specific source plus an explicitly reviewed handler.

    An empty handler is deliberately unsupported, never an inert success.
    Source descriptions are evidence, not executable instructions.
    """
    index: int
    prefab: str
    name: str
    description: str
    parameters: tuple[tuple[str, float | str], ...] = ()
    attack_range: tuple[tuple[int, int], ...] = ()
    handler: str = ""
    source: str = "character_table"
    limitations: tuple[str, ...] = ()

    def get(self, key, default=0):
        return next((v for k, v in self.parameters if k == key), default)


@dataclass(frozen=True)
class OperatorData:
    id: str
    name: str
    hp: float
    atk: float
    defense: float
    resistance: float
    interval: float
    block_count: int
    cost: int
    redeploy: float
    attack_range: tuple[tuple[int, int], ...]
    position_type: str = "MELEE"
    damage_type: str = "PHYSICAL"
    skill: SkillData | None = None
    profession: str = ""
    subprofession: str = ""
    rarity: int = 0
    simulation_notes: tuple[str, ...] = ()
    elite: int = 0
    level: int = 1
    potential: int = 0  # upgrades; in-game potential = this value + 1
    skill_index: int = 0
    skill_level: int = 1  # 8..10 = mastery 1..3
    selected_skill_id: str | None = None
    summons: tuple["SummonData", ...] = ()
    healable: bool = True
    trust: float = 0
    can_attack_air: bool = False
    normal_hits: int = 1
    target_mode: str = "SINGLE"
    ranged_attack_scale: float = 1
    module_id: str = ''
    module_level: int = 0
    talents: tuple[TalentData, ...] = ()
    factions: tuple[str, ...] = ()
    base_attack_speed: float = 100


@dataclass(frozen=True)
class SummonData:
    """An explicitly supported summon card, separate from a deployed instance.

    Lifecycle fields are handler rules, not inferred from description text.
    No generic claim that other owners share Deepcolor's rules.
    """
    owner_id: str
    unit: OperatorData
    inventory: int
    max_active: int
    slot_weight: int = 1
    reset_on_owner_deploy: bool = True
    remove_with_owner: bool = True
    retreat_refund_ratio: float = 0
    return_on_retreat: bool = False

    @property
    def id(self):
        return f"{self.owner_id}::{self.unit.id}"


@dataclass(frozen=True)
class StageData:
    id: str
    map: MapData
    routes: tuple[RouteData, ...]
    waves: tuple[WaveData, ...]
    enemies: tuple[EnemyData, ...]
    initial_cost: float
    max_cost: float
    cost_interval: float
    move_multiplier: float
    life: int
    character_limit: int
    assumptions: tuple[str, ...] = ()
    devices: tuple = ()
    difficulty: str = "NORMAL"
    dormant_branches: tuple[str, ...] = ()

    @property
    def spawns(self):
        """Spawn inventory; times are lower bounds for conditional event stages."""
        return tuple(
            sorted((s for w in self.waves for s in w.spawns), key=lambda s: s.time)
        )

    @property
    def hostile_count(self):
        hostile = {e.id for e in self.enemies if e.faction == "HOSTILE"}
        return sum(s.enemy_id in hostile for s in self.spawns)

    @property
    def escort_count(self):
        return len(self.spawns) - self.hostile_count
