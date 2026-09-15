import json
import math
from pathlib import Path
from .models import SkillData


# Audited self-only finite-duration buffs. A matching blackboard alone is NOT
# enough: summons, multi-hit, targeting and ally auras can share those keys.
SIMPLE_MANUAL_IDS = frozenset({
    "skchr_acdrop_1", "skchr_astesi_1", "skchr_bdhkgt_1", "skchr_flower_2",
    "skchr_frncat_2", "skchr_haak_1", "skchr_headb2_1", "skchr_hsguma_1",
    "skchr_ifrit_1", "skchr_lunacu_1", "skchr_ncdeer_1", "skchr_robrta_1",
    "skchr_shining_1", "skchr_slent2_1", "skchr_udflow_1", "skchr_vigna_1",
    "skchr_wscoot_1", "skcom_magic_rage[1]", "skcom_magic_rage[2]",
    "skcom_magic_rage[3]", "skcom_heal_rage[3]",
})
SKILL_ENGINE_VERSION = 4


class SkillLoader:
    def __init__(self, path):
        self.data = json.load(open(path))
        ranges = Path(path).with_name('range_table.json')
        self.ranges = json.loads(ranges.read_text()) if ranges.exists() else {}

    def load(self, key, level=1):
        if key not in self.data:
            raise NotImplementedError("Skill data missing")
        if type(level) is not int or not 1 <= level <= len(self.data[key]["levels"]):
            raise ValueError("Skill level")
        from .guard_skills import load_guard_skill
        guard = load_guard_skill(key, self.data[key]['levels'][level-1], self.ranges)
        if guard is not None:
            return guard
        if key not in SIMPLE_MANUAL_IDS | {"skchr_deepcl_1"} and not any(key.startswith(prefix) for prefix in (
            "skcom_atk_up[", "skcom_def_up[", "skcom_quickattack[", "skcom_heal_up[",
        )):
            raise NotImplementedError("Skill effect not implemented")
        s = self.data[key]["levels"][level - 1]
        sp = s["spData"]
        bb = {v["key"]: v["value"] for v in s["blackboard"]}
        if s["skillType"] != "MANUAL" or sp["spType"] != "INCREASE_WITH_TIME":
            raise NotImplementedError("Skill recovery/trigger")
        if (not math.isfinite(s["duration"]) or s["duration"] <= 0
                or not math.isfinite(sp["spCost"]) or sp["spCost"] <= 0):
            raise NotImplementedError("Instant, infinite or zero-cost skill needs a dedicated handler")
        allowed = {"atk", "def", "attack_speed"}
        if key == "skchr_deepcl_1":
            allowed.add("hp_recovery_per_sec")
        if set(bb) - allowed or s.get("rangeId") is not None:
            raise NotImplementedError("Skill modifiers")
        return SkillData(
            key,
            sp["spCost"],
            sp["initSp"],
            s["duration"],
            attack_multiplier=1 + bb.get("atk", 0),
            defense_multiplier=1 + bb.get("def", 0),
            attack_speed_multiplier=1 + bb.get("attack_speed", 0) / 100,
            effect_target="OWNER_SUMMONS" if key == "skchr_deepcl_1" else "SELF",
            hp_recovery_per_sec=bb.get("hp_recovery_per_sec", 0),
        )
