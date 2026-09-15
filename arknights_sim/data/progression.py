"""UI-independent operator progression requests and per-character legal caps.

Potential counts upgrades (0..5), corresponding to in-game potential 1..6.
A uniform target above a character's promotion ceiling uses its final phase cap.
Unsupported effects remain explicit in OperatorData.simulation_notes.
"""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Progression:
    elite: int = 0
    level: int = 1
    potential: int = 0
    skill_level: int = 1
    skill_index: int = 0

    def __post_init__(self):
        for name, low, high in (("elite", 0, 2), ("level", 1, 90),
                                ("potential", 0, 5), ("skill_level", 1, 10),
                                ("skill_index", 0, 2)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid progression {name}: {value!r}")

    @classmethod
    def from_dict(cls, values=None):
        return cls(**(values or {}))

    def to_dict(self):
        return asdict(self)


def unlocked(condition, elite, level):
    phase = int(condition['phase'].split('_')[-1])
    return (elite, level) >= (phase, condition['level'])


def skill_cap(character, skill, elite, level):
    rank = 1
    for entry in character.get('allSkillLvlup', ()):
        if not unlocked(entry['unlockCond'], elite, level):
            break
        rank += 1
    rank = min(7, rank)
    if rank == 7:
        for entry in skill.get('levelUpCostCond') or ():
            if not unlocked(entry['unlockCond'], elite, level):
                break
            rank += 1
    return min(rank, 10)


def resolve_progression(character, request):
    elite = min(request.elite, len(character['phases']) - 1)
    maximum = character['phases'][elite]['maxLevel']
    level = maximum if elite < request.elite else min(request.level, maximum)
    notes = []
    if (elite, level) != (request.elite, request.level):
        notes.append(f'养成上限调整：精英{request.elite} Lv{request.level} → 精英{elite} Lv{level}')
    potential = min(request.potential, len(character.get('potentialRanks') or ()))
    if potential != request.potential:
        notes.append(f'潜能提升上限调整：{request.potential} → {potential}')
    skills = character['skills']
    available = [i for i, entry in enumerate(skills) if unlocked(entry['unlockCond'], elite, level)]
    index = request.skill_index if request.skill_index in available else (available[0] if available else 0)
    if index != request.skill_index or not available:
        notes.append('所选携带技能不存在或未解锁：使用首个已解锁技能' if available else '此阶段无可用技能')
    rank = min(request.skill_level, skill_cap(character, skills[index], elite, level)) if available else 1
    if available and rank != request.skill_level:
        notes.append(f'技能等级上限调整：{request.skill_level} → {rank}（8/9/10 为专一/二/三）')
    return Progression(elite, level, potential, rank, index), tuple(notes)


def load_progressed(loader, key, request):
    key = loader.resolve(key)
    effective, notes = resolve_progression(loader.chars[key], request)
    operator = loader.load(key, **effective.to_dict())
    return replace(operator, simulation_notes=notes + operator.simulation_notes)
