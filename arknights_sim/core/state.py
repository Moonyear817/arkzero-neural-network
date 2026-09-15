import copy, hashlib, json, random
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from arknights_sim.data.models import (
    TileData, MapData, WaypointData, RouteData, SpawnData, WaveData,
    EnemyData, SkillData, OperatorData, StageData, SummonData, StageActionData, FragmentData, EnemySkillData, TalentData,
)
from .clock import SimulationClock
from .event_queue import EventQueue
from arknights_sim.mechanics.definitions import DeviceData


_IMMUTABLE_DATA_TYPES = frozenset((
    TileData, MapData, WaypointData, RouteData, SpawnData, WaveData,
    EnemyData, SkillData, OperatorData, StageData, DeviceData, SummonData, StageActionData, FragmentData, EnemySkillData, TalentData,
))
_IMMUTABLE_ATOMS = frozenset((str, bytes, int, float, bool, type(None)))
# Cache only *validated deeply immutable* object graphs, keyed by identity.
# A strong reference prevents id reuse; the bound prevents unlimited retention
# when callers load many distinct stages. Mutable or malformed inputs are never
# cached. No battle state, snapshot or hash output is cached.
_IMMUTABLE_GRAPHS = OrderedDict()
_IMMUTABLE_GRAPH_LIMIT = 128


def _immutable_data_references(root):
    cached = _IMMUTABLE_GRAPHS.get(id(root))
    if cached is not None and cached[0] is root:
        return cached[1]
    references = {}

    def visit(value):
        if type(value) in _IMMUTABLE_ATOMS:
            return True
        if id(value) in references:
            return True
        if type(value) is tuple:
            references[id(value)] = value
            return all(visit(item) for item in value)
        if type(value) in _IMMUTABLE_DATA_TYPES:
            if not value.__dataclass_params__.frozen:
                return False
            references[id(value)] = value
            return all(visit(getattr(value, f.name)) for f in fields(value))
        return False

    if not visit(root):
        return {}
    _IMMUTABLE_GRAPHS[id(root)] = (root, references)
    if len(_IMMUTABLE_GRAPHS) > _IMMUTABLE_GRAPH_LIMIT:
        _IMMUTABLE_GRAPHS.popitem(last=False)
    return references


def canonical(x):
    if is_dataclass(x):
        # asdict first recursively copied the whole graph, then this function
        # walked it a second time. Build fresh output directly in one traversal.
        return {f.name: canonical(getattr(x, f.name)) for f in fields(x)}
    if isinstance(x, dict):
        return {str(k): canonical(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [canonical(v) for v in x]
    return x


class GameState:
    def __init__(self, stage, squad, seed, dt):
        self.stage = stage
        self.squad = {o.id: o for o in squad}
        self.clock = SimulationClock(dt=dt)
        self.queue = EventQueue()
        self.rng = random.Random(seed)
        self.enemies = {}
        self.operators = {}
        self.summons = {}
        self.summon_cards = {card.id: card for op in squad for card in op.summons}
        self.summon_inventory = {key: card.inventory for key, card in self.summon_cards.items()}
        self.summon_ready_at = {key: 0.0 for key in self.summon_cards}
        self.summon_sequence = 0
        self.dp = stage.initial_cost
        self.life = stage.life
        self.spawned = 0
        self.killed = 0
        self.escaped = 0
        self.escorts_saved = 0
        self.escorts_dead = 0
        self.deploy_counts = {}
        self.redeploy_at = {}
        self.done = False
        self.result = None
        self.events_processed = 0
        from arknights_sim.mechanics.runtime import initialize
        initialize(self)
        from .stage_events import initialize as initialize_events
        initialize_events(self)

    @property
    def current_time(self):
        return self.clock.time

    @property
    def allies(self):
        return {**self.operators, **self.summons} if self.summons else self.operators

    @property
    def occupied_deployment_slots(self):
        return sum(o.alive for o in self.operators.values()) + sum(
            self.summon_cards[u.summon_card_id].slot_weight
            for u in self.summons.values() if u.alive
        )

    def clone(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        # Registries, units, clock, queue, modifiers lists, skills and RNG remain
        # independently copied. Only proven immutable normalized inputs share
        # identity, including data referenced from units and spawn payloads.
        for data in (self.stage, *self.squad.values()):
            for identity, immutable in _immutable_data_references(data).items():
                # Respect earlier copies when this state belongs to a larger
                # object graph whose setup objects have already been visited.
                memo.setdefault(identity, immutable)
        result.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return result

    def snapshot(self):
        return canonical(
            {
                **{k: v for k, v in self.__dict__.items() if k not in ["rng", "queue"]},
                "rng": self.rng.getstate(),
                "queue": sorted(self.queue.heap),
                "event_sequence": self.queue.sequence,
            }
        )

    def stable_hash(self):
        return hashlib.sha256(
            json.dumps(
                self.snapshot(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def step(self, seconds):
        from .simulator import Simulator

        sim = Simulator.from_state(self)
        sim.step(seconds)
        return self
