from dataclasses import dataclass,field

@dataclass(frozen=True)
class DeviceData:
    id: str
    kind: str
    position: tuple[int,int] | None = None
    direction: int = 0
    count: int = 0
    cost: float = 5
    cooldown: float = 5
    charge: float = 0
    duration: float = 0
    skill_level: int = 1
    hp: float = 100
    defense: float = 0
    resistance: float = 0
    block_count: int = 3

@dataclass
class DeviceState:
    data: DeviceData
    position: tuple[int,int]
    hp: float = 100
    sp: float = 0
    active_until: float = 0
    alive: bool = True
    id: str = ''
    generation: int = 0
    deployed_at: float = 0
    blocked_enemies: list = field(default_factory=list)
    blocked_by: str | None = None
    modifiers: list = field(default_factory=list)

KINDS={'trap_001_crate':'crate','trap_005_sensor':'sensor','trap_121_gractrl':'gravity','trap_013_blower':'wind','trap_020_roadblock':'roadblock',
       'trap_022_frosts_friend':'friendly_frost','trap_023_ore_friend':'friendly_altar'}
DIRECTIONS=('RIGHT','UP','LEFT','DOWN')


def parse_devices(raw):
    predefined=raw.get('predefines') or {}
    if predefined.get('characterInsts') or predefined.get('characterCards'):
        raise NotImplementedError('Predefined operators')
    result=[]
    for bucket in ('tokenInsts','tokenCards'):
        for i,record in enumerate(predefined.get(bucket) or []):
            key=record['inst']['characterKey']
            if key not in KINDS:raise NotImplementedError('Device mechanics: '+key)
            if record.get('overrideSkillBlackboard') or record.get('overrideTalents'):
                raise NotImplementedError('Device parameter overrides')
            kind=KINDS[key]
            if bucket=='tokenCards' and kind!='crate':raise NotImplementedError('Deployable device: '+key)
            pos=record.get('position')
            result.append(DeviceData(key if bucket=='tokenCards' else f'{key}:{i}',kind,
                (pos['col'],pos['row']) if pos else None,DIRECTIONS.index(record.get('direction','RIGHT')),
                int(record.get('initialCnt',0)),charge=15 if kind in ('sensor','friendly_frost','friendly_altar') else 0,
                duration=20 if kind=='sensor' else 0,skill_level=record.get('mainSkillLvl',1),
                hp=8000 if kind=='roadblock' else 100,
                defense=200 if kind=='roadblock' else 0,
                resistance=20 if kind=='roadblock' else 0,
                block_count=0 if kind=='roadblock' else 3))
    return tuple(result)
