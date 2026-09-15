import json
from .models import EnemyData, EnemySkillData


def defined_merge(target, source):
    for k, v in (source or {}).items():
        if isinstance(v, dict) and "m_defined" in v:
            if v["m_defined"]:
                target[k] = v["m_value"]
        elif isinstance(v, dict):
            defined_merge(target.setdefault(k, {}), v)
        elif v is not None:
            target[k] = v
    return target


class EnemyLoader:
    def __init__(self, path):
        self.entries = {e["Key"]: e["Value"] for e in json.load(open(path))["enemies"]}

    def load(self, ref):
        if not ref["useDb"]:
            raise NotImplementedError("Inline enemy data")
        d = self.resolve(ref)
        a = d["attributes"]
        key = ref['id']
        if key == 'enemy_3001_upeopl':
            return EnemyData(key, d['name'], a['maxHp'], 0, a['def'],
                             a.get('magicResistance', 0), a['moveSpeed'], 0,
                             life_cost=0, block_weight=0, damage_type='NONE',
                             faction='ESCORT', death_life_cost=1)
        if key == 'enemy_3002_ftrtal':
            skills=[]
            for skill in d.get('skills') or []:
                name=skill['prefabKey']
                if name not in {'DragonFire','DragonFire[Half]','DanceFire','DanceFire[Half]'}:
                    raise NotImplementedError('Escort Talulah skill: '+name)
                skills.append(EnemySkillData(name.split('[')[0],skill['initCooldown'],skill['cooldown'],
                    '[Half]' in name,tuple((b['key'],b['value']) for b in skill.get('blackboard') or [])))
            return EnemyData(key,d['name'],a['maxHp'],a['atk'],a['def'],a['magicResistance'],a['moveSpeed'],a['baseAttackTime'],
                life_cost=0,block_weight=0,damage_type='ARTS',attack_range=d['rangeRadius'],weight=a['massLevel'],
                faction='ESCORT',death_life_cost=2,skills=tuple(skills),half_hp_defense_bonus=.5,half_hp_resistance_bonus=.5)
        reviewed = ref['id'] in {'enemy_1009_lurker','enemy_1019_jshoot','enemy_1023_jmage','enemy_1024_mortar'}
        chapter_ranged = key in {'enemy_1109_uabone','enemy_1110_uamord'}
        vanguard = key in {'enemy_1113_empace','enemy_1113_empace_2'}
        rager = key in {'enemy_1062_rager','enemy_1062_rager_2','enemy_1063_rageth','enemy_1063_rageth_2'}
        drone = key in {'enemy_1112_emppnt','enemy_1112_emppnt_2'}
        centurion = key == 'enemy_1114_rgrdmn'
        board = {b['key']:b['value'] for b in d.get('talentBlackboard') or []}
        permitted = {'atkup.atk'} if vanguard else {'periodic_damage.damage'} if rager else {'vampire.heal_scale','vampire.attack@max_target'} if centurion else set()
        if (
            (d.get("motion") != "WALK" and not (drone and d.get('motion')=='FLY'))
            or (d.get("applyWay") != "MELEE" and not (reviewed or chapter_ranged or vanguard or rager or drone or centurion))
            or d.get("skills")
            or set(board) - permitted
        ):
            raise NotImplementedError(f"Enemy mechanics unsupported: {ref['id']}")
        return EnemyData(
            ref["id"], d["name"], a["maxHp"], a["atk"], a["def"],
            a["magicResistance"], a["moveSpeed"],
            a["baseAttackTime"] * 100 / a["attackSpeed"],
            d.get("lifePointReduce", 1),
            damage_type='ARTS' if key in {'enemy_1023_jmage','enemy_1110_uamord','enemy_1114_rgrdmn'} else 'PHYSICAL',
            attack_range=float(d.get('rangeRadius') or 0) if reviewed or chapter_ranged or vanguard or rager or drone or centurion else 0,
            invisible=ref['id'] in {'enemy_1009_lurker','enemy_1019_jshoot','enemy_1023_jmage'},
            splash=ref['id']=='enemy_1024_mortar',weight=int(a.get('massLevel',1)),
            escort_attack_range=1.2 if key == 'enemy_1107_uoffcr' else 0,
            low_hp_attack_bonus=board.get('atkup.atk',0),
            self_damage_per_second=board.get('periodic_damage.damage',0),
            motion=d['motion'],prioritize_escorts=vanguard or drone or key=='enemy_1110_uamord',
            ranged_damage_scale=.5 if vanguard else 1,
            projectile_delay=3 if drone else 0,projectile_radius=1.2 if drone else 0,
            max_targets=int(board.get('vampire.attack@max_target',1)),lifesteal=board.get('vampire.heal_scale',0),
            freeze_immune=bool(a.get('frozenImmune',False)),
            silence_immune=bool(a.get('silenceImmune',False)),
            tremble_immune=bool(a.get('disarmedCombatImmune',False)),
        )

    def resolve(self, ref):
        """Read inherited stats and per-stage overrides, including complex foes.

        Recognition does not relax the battle engine's mechanics checks.
        """
        if not ref.get("useDb", True):
            d = defined_merge({}, ref.get("enemyData"))
            defined_merge(d, ref.get("overwrittenData"))
            if not d.get("attributes"):
                raise ValueError(f"Missing inline enemy attributes: {ref['id']}")
            return d
        levels = sorted(self.entries[ref["id"]], key=lambda x: x["level"])
        if ref["level"] not in [v["level"] for v in levels]:
            raise ValueError("Unknown enemy level")
        d = {}
        for v in levels:
            if v["level"] <= ref["level"]:
                defined_merge(d, v["enemyData"])
        defined_merge(d, ref.get("overwrittenData"))
        return d
