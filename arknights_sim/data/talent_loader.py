"""Explicit guard talent recipes. Names/text never drive combat behavior.

Unrecognized parameters fail closed at the talent level. The original resolved
record is retained for capability reporting and research previews.
"""
from .models import TalentData
from .module_loader import candidate


# (operator id, prefab key): (handler, complete set of accepted source keys).
# A subset is permitted across elite phases; new unreviewed keys are not.
RECIPES = {
    ("char_208_melan", "1"): ("static", {"atk"}),
    ("char_281_popka", "1"): ("static", {"atk", "max_hp"}),
    ("char_140_whitew", "1"): ("silence", {"duration", "damage_scale"}),
    ("char_283_midn", "1"): ("attack_crit", {"prob", "atk_scale"}),
    ("char_264_f12yin", "1"): ("mountain_crit", {"prob", "atk_scale", "atk", "duration"}),
    ("char_264_f12yin", "2"): ("defense_dodge", {"def", "prob"}),
    ("char_1026_gvial2", "1"): ("block_stats", {"atk", "def", "atk_add", "def_add"}),
    ("char_1026_gvial2", "2"): ("healing_received", {"heal_scale_1", "heal_scale_2", "hp_ratio"}),
    ("char_127_estell", "1"): ("nearby_death_heal", {"hp_ratio"}),
    ("char_274_astesi", "1"): ("time_attack_speed", {"max_stack_cnt", "interval", "attack_speed"}),
    ("char_421_crow", "1"): ("kill_attack_speed", {"max_stack_cnt", "attack_speed"}),
    ("char_415_flint", "1"): ("unblocked_damage", {"damage_scale"}),
    ("char_350_surtr", "1"): ("arts_penetration", {"magic_resist_penetrate_fixed"}),
    ("char_350_surtr", "2"): ("lethal_survival", {"surtr_t_2[withdraw].interval"}),
    ("char_4145_ulpia", "2"): ("pre_damage_heal", {"hp_ratio", "value1", "value2"}),
    ("char_4145_ulpia", "1"): ("abyssal_kill_stats", {"max_stack_cnt", "max_hp", "atk", "ulpia_t_1[abyssal].max_hp", "ulpia_t_1[abyssal].atk", "ulpia_t_1[abyssal].max_stack_cnt"}),
    ("char_337_utage", "1"): ("berserk", {"min_attack_speed", "min_hp_ratio"}),
    ("char_475_akafyu", "1"): ("berserk", {"min_attack_speed", "min_hp_ratio"}),
    ("char_1030_noirc2", "1"): ("berserk", {"min_attack_speed", "min_hp_ratio", "min_def"}),
    ("char_1030_noirc2", "2"): ("consecutive_hits", {"atk", "hit_num", "attack@clear_attackcount_time"}),
    ("char_4121_zuole", "1"): ("berserk", {"min_attack_speed", "min_hp_ratio", "min_sp_recovery_per_sec"}),
    ("char_4121_zuole", "2"): ("attack_sp_chance", {"prob_1", "prob_2", "hp_ratio", "sp"}),
    ("char_4121_zuole", "10"): ("low_hp_shelter", {"hp_ratio", "damage_resistance"}),
    ("char_265_sophia", "1"): ("high_block_aura", {"attack_speed", "def", "sophia_t_1_less.attack_speed", "sophia_t_1_less.def"}),
    ("char_286_cast3", "1"): ("deploy_melee_aura", {"duration", "atk", "def"}),
    ("char_308_swire", "1"): ("nearby_melee_attack", {"atk"}),
    ("char_4106_bryota", "1"): ("conditional_defense_aura", {"bryota_t_ally.def", "bryota_t_self.def"}),
    ("char_271_spikes", "1"): ("drone_priority", {"atk_scale"}),
    ("char_4125_rdoc", "1"): ("hp_def_penetration", {"max_hp", "def_penetrate_fixed"}),
    ("char_4011_lessng", "1"): ("other_enemy_protection", {"damage_resistance"}),
    ("char_4011_lessng", "2"): ("hurt_attack_buff", {"atk", "add_atk_duration"}),
    ("char_4010_etlchi", "1"): ("hp_steal_dot", {"dot_duration", "attack@steal_hp", "attack@steal_hp_max", "magic_value", "interval"}),
    ("char_4010_etlchi", "2"): ("emergency_heal", {"hp_ratio", "damage_resistance", "etlchi_t_2[heal].hp_ratio"}),
    ("char_4067_lolxh", "1"): ("dying", {"move_speed", "sp", "interval"}),
    ("char_4116_blkkgt", "1"): ("damage_crit_tremble", {"prob", "atk_scale", "not_combat"}),
    ("char_4116_blkkgt", "2"): ("tremble_penetration", {"def_penetrate"}),
    ("char_4064_mlynar", "1"): ("nearby_enemies_scale", {"atk_scale_base", "cnt", "atk_scale_up", "damage_resistance"}),
    ("char_4064_mlynar", "2"): ("kazimierz_reflection", {"taunt_level", "atk_scale"}),
    ("char_459_tachak", "1"): ("range_redeploy", {"def", "respawn_time", "ability_range_forward_extend"}),
    ("char_1032_excu2", "1"): ("extra_attack", {"prob", "prob_add"}),
}


def load_talents(key, character, ranges, elite, level, potential, module_talents=(), module_traits=(), module_id=""):
    chosen = []
    for index, group in enumerate(character.get("talents") or ()):
        row = candidate(group.get("candidates"), elite, level, potential)
        if row:
            chosen.append((index, row, "character_table"))
    for row in module_talents:
        index = row["talentIndex"]
        # A hidden module prefab can *add* an effect while reusing talentIndex.
        # Zuo Le's protection must not replace his berserk talent.
        match = next((i for i, (n, old, _) in enumerate(chosen)
                      if n == index and old["prefabKey"] == row["prefabKey"]), None)
        if match is None:
            chosen.append((index, row, module_id))
        else:
            chosen[match] = (index, row, module_id)
    result = []
    for index, row, source in chosen:
        board = tuple((b["key"], b["valueStr"] if b.get("valueStr") is not None else b["value"])
                      for b in row.get("blackboard") or ())
        prefab = row["prefabKey"]
        handler, accepted = RECIPES.get((key, prefab), ("", set()))
        limitations = []
        if prefab == "-1" and not board:
            handler = "no_effect_marker"
        unknown = set(k for k, _ in board) - accepted
        if unknown and handler != "no_effect_marker":
            handler = ""
            limitations.append("Unreviewed parameters: " + ", ".join(sorted(unknown)))
        if not handler:
            limitations.append("Talent runtime is not implemented")
        if handler == "extra_attack":
            limitations.append("Ammo skill integration requires a supported ammo skill")
        rid = row.get("rangeId")
        cells = tuple((v["col"], v["row"]) for v in ranges[rid]["grids"]) if rid else ()
        result.append(TalentData(index, prefab, row.get("name") or prefab,
                                 row.get("upgradeDescription") or row.get("description") or "",
                                 board, cells, handler, source, tuple(limitations)))
    for index, row in enumerate(module_traits):
        board = tuple((b["key"], b["value"]) for b in row.get("blackboard") or ())
        modules = {
            "uniequip_002_gvial2": ("module_blocked_attack", {"atk_scale"}),
            "uniequip_002_f12yin": ("module_healthy_speed", {"attack_speed"}),
            "uniequip_002_surtr": ("module_unblocked_speed", {"attack_speed"}),
            "uniequip_002_ulpia": ("module_healing", {"heal_scale"}),
            "uniequip_002_blkkgt": ("module_skill_damage", {"damage_scale", "value"}),
            # Protection is supplied by the module's separate hidden talent.
            "uniequip_002_zuole": ("no_effect_marker", {"value"}),
        }
        handler, accepted = modules.get(module_id, ("", set()))
        if set(k for k, _ in board) - accepted:
            handler = ""
        result.append(TalentData(-100-index, "trait", "模组特性",
                                 row.get("overrideDescripton") or row.get("additionalDescription") or "",
                                 board, handler=handler, source=module_id,
                                 limitations=() if handler else ("Module trait runtime is not implemented",)))
    return tuple(result)
