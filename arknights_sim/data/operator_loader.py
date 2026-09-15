"""Roster identification and explicitly scoped battle inputs from pinned tables."""
import json
from pathlib import Path

from .models import OperatorData
from .progression import skill_cap, unlocked


class OperatorLoader:
    def __init__(self, character_path, range_path, skill_loader=None):
        self.character_path = Path(character_path)
        directory = Path(character_path).parent
        self.chars = json.loads(Path(character_path).read_text(encoding="utf-8"))
        patch = directory / "char_patch_table.json"
        if patch.exists():
            self.chars.update(json.loads(patch.read_text(encoding="utf-8"))["patchChars"])
        self.ranges = json.loads(Path(range_path).read_text(encoding="utf-8"))
        self.skills = skill_loader
        favor = directory / "favor_table.json"
        self.favor = json.loads(favor.read_text()) if favor.exists() else None
        module_path=directory/'battle_equip_table.json'
        self.modules=json.loads(module_path.read_text()) if module_path.exists() else {}
        path = directory / "operator_catalog.json"
        self.catalog = {}
        if path.exists():
            self.catalog = {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))["operators"] if r["id"] in self.chars}
        self.aliases = {}
        for key in self.available_ids():
            c = self.chars[key]
            record = self.catalog.get(key, {})
            aliases = record.get("aliases", [c["name"], c.get("appellation", "")])
            for alias in [key, *aliases]:
                if alias:
                    self.aliases.setdefault(alias.strip().casefold(), set()).add(key)

    def available_ids(self):
        return tuple(sorted(self.catalog or {k: c for k, c in self.chars.items() if k.startswith("char_") and not c.get("isNotObtainable")}))

    def resolve(self, query):
        if query in self.available_ids():
            return query
        matches = self.aliases.get(query.strip().casefold(), set())
        if len(matches) != 1:
            raise ValueError(f"{'Ambiguous' if matches else 'Unknown'} operator: {query}")
        return next(iter(matches))

    def search(self, query=""):
        query = query.strip().casefold()
        return tuple(key for key in self.available_ids() if not query or any(query in alias and key in keys for alias, keys in self.aliases.items()))

    def load(self, key, *, elite=0, level=1, skill_index=0, skill_level=1, potential=0, trust=0, module_id='', module_level=0):
        key = self.resolve(key)
        c = self.chars[key]
        if type(elite) is not int or not 0 <= elite < len(c["phases"]):
            raise ValueError("Invalid elite phase")
        p = c["phases"][elite]
        if type(level) is not int or not 1 <= level <= p["maxLevel"]:
            raise ValueError("Invalid operator level")
        frames = p["attributesKeyFrames"]
        lo, hi = frames[0], frames[-1]
        ratio = (level - lo["level"]) / max(1, hi["level"] - lo["level"])
        a = dict(lo["data"])
        for field in ("maxHp", "atk", "def", "magicResistance"):
            value = lo["data"][field] + ratio * (hi["data"][field] - lo["data"][field])
            a[field] = int(value + 0.5) if field != "magicResistance" else value
        from .trust import trust_bonuses
        if trust and self.favor is None:
            raise ValueError("Trust conversion requires favor_table.json")
        for field, bonus in trust_bonuses(c, self.favor, trust).items():
            a[field] += bonus
        notes = []
        ranks = c.get("potentialRanks") or []
        if type(potential) is not int or not 0 <= potential <= min(5, len(ranks)):
            raise ValueError("Invalid potential upgrades")
        if type(skill_level) is not int or not 1 <= skill_level <= 10:
            raise ValueError("Invalid skill level")
        fields = {"MAX_HP": "maxHp", "ATK": "atk", "DEF": "def",
                  "MAGIC_RESISTANCE": "magicResistance", "COST": "cost",
                  "RESPAWN_TIME": "respawnTime", "ATTACK_SPEED": "attackSpeed",
                  "BLOCK_CNT": "blockCnt", "BASE_ATTACK_TIME": "baseAttackTime"}
        for rank in ranks[:potential]:
            modifiers = ((rank.get("buff") or {}).get("attributes") or {}).get("attributeModifiers") or []
            supported = bool(modifiers) and rank["type"] == "BUFF"
            for modifier in modifiers:
                field = fields.get(modifier["attributeType"])
                if (field and modifier["formulaItem"] == "ADDITION"
                        and not modifier.get("loadFromBlackboard")
                        and not modifier.get("fetchBaseValueFromSourceEntity")):
                    a[field] += modifier["value"]
                else:
                    supported = False
            if not supported:
                notes.append("潜能效果未实现：" + rank["description"])
        from .module_loader import module_data
        if module_id:
            info=json.loads(self.character_path.with_name('uniequip_table.json').read_text())['equipDict'].get(module_id)
            if not info or info['charId']!=key or not unlocked({'phase':info['unlockEvolvePhase'],'level':info['unlockLevel']},elite,level):
                raise ValueError('Module does not belong to operator or is locked')
        bonuses,module_traits,module_talents=module_data(self.modules,module_id,module_level,elite,level,potential)
        for source,target in {'max_hp':'maxHp','atk':'atk','def':'def','magic_resistance':'magicResistance',
                              'attack_speed':'attackSpeed','cost':'cost','respawn_time':'respawnTime'}.items():
            a[target]+=bonuses.get(source,0)
        from .talent_loader import load_talents
        talents = load_talents(key, c, self.ranges, elite, level, potential,
                               module_talents, module_traits, module_id)
        for talent in talents:
            if not talent.handler:
                notes.append('天赋/模组效果未实现：' + talent.name + ' [' + talent.prefab + ']')
            elif talent.limitations:
                notes.append('天赋接入限制：' + talent.name + ' — ' + '; '.join(talent.limitations))
        if module_id:notes.append('模组基础属性已应用；已接入效果仍待实机时序校准：'+module_id)
        if key not in ("char_500_noirc", "char_208_melan") or elite or level != 1:
            notes.append("基础近似模拟：未校准全部特性、天赋、模组、召唤物和特殊攻击；未实现效果不计入属性")
        skill = None
        if c["skills"]:
            if type(skill_index) is not int or not 0 <= skill_index < len(c["skills"]):
                raise ValueError("Invalid skill index")
            selected = c["skills"][skill_index]
            if unlocked(selected["unlockCond"], elite, level) and skill_level > skill_cap(c, selected, elite, level):
                raise ValueError("Skill rank/mastery not unlocked")
            unlock = selected["unlockCond"]
            phase = int(unlock["phase"].split("_")[-1])
            if elite < phase or (elite == phase and level < unlock["level"]):
                notes.append("所选技能尚未解锁")
            elif self.skills:
                try:
                    skill = self.skills.load(selected["skillId"], skill_level)
                except NotImplementedError:
                    notes.append("技能资料已收录，当前技能效果未实现，战斗中禁用")
        profession, branch = c["profession"], c["subProfessionId"]
        kind = "PHYSICAL"
        if profession == "CASTER" or branch in {"artsfghter", "slower", "summoner", "underminer", "ritualist", "incantationmedic"}:
            kind = "ARTS"
        if profession == "MEDIC" and branch != "incantationmedic":
            kind = "HEAL"
        if branch in {"bard", "craftsman", "librator"}:
            kind = "NONE"
            notes.append("通常不攻击；特殊支援或技能攻击尚未实现")
        from .summon_loader import load_summons
        summons = load_summons(self, key, elite, level, potential)
        if summons:
            notes.append("触手基础作战/一技能已接入；二技能、模组和生命周期实机校准未完成")
        return OperatorData(
            key, self.catalog.get(key, {}).get("name", c["name"]),
            a["maxHp"], a["atk"], a["def"], a["magicResistance"],
            a["baseAttackTime"] * 100 / a["attackSpeed"], a["blockCnt"],
            a["cost"], a["respawnTime"],
            tuple((v["col"], v["row"]) for v in self.ranges[p["rangeId"]]["grids"]),
            c["position"], kind, skill=skill, profession=profession,
            subprofession=branch, rarity=int(c["rarity"].split("_")[-1]),
            simulation_notes=tuple(notes),
            elite=elite, level=level, potential=potential, trust=trust,
            skill_index=skill_index, skill_level=skill_level,
            selected_skill_id=c["skills"][skill_index]["skillId"] if c["skills"] else None,
            summons=summons,
            can_attack_air=profession in {'SNIPER','CASTER','SUPPORT'} or branch == 'lord',
            normal_hits=2 if branch=='sword' else 1,
            target_mode='BLOCK_COUNT' if branch in {'centurion','crusher'} else 'ALL' if branch=='reaper' else 'SINGLE',
            ranged_attack_scale=.8 if branch=='lord' else 1,
            module_id=module_id,module_level=module_level,
            talents=talents,
            factions=tuple(v for v in (c.get('nationId'), c.get('groupId'), c.get('teamId')) if v),
            healable=branch not in {'musha','reaper'},
            base_attack_speed=a['attackSpeed'],
        )
