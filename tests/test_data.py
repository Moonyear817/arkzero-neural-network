import json
from dataclasses import replace
import pytest
from conftest import DATA
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.data.enemy_loader import defined_merge
from arknights_sim.data.skill_loader import SkillLoader
from arknights_sim.map.route import compile_route, pathfind


def test_stage_parsing(real_stage):
    assert (real_stage.map.width, real_stage.map.height) == (9, 6)
    assert len(real_stage.spawns) == 11
    assert real_stage.initial_cost == 10 and real_stage.move_multiplier == 0.5


def test_tiles_coordinate_flip(real_stage):
    assert real_stage.map.tile(8, 2).key == "tile_start"
    assert real_stage.map.tile(0, 3).key == "tile_end"
    assert real_stage.map.tile(2, 3).buildable == "RANGED"
    assert real_stage.map.ascii().splitlines()[2] == "B . H . . . . . #"


def test_route_parsing(real_stage):
    r = real_stage.routes[9]
    assert r.start == (8, 2) and r.end == (0, 3)
    assert r.waypoints[0].position == (7, 2)
    assert (r.waypoints[1].kind, r.waypoints[1].time) == ("WAIT_FOR_SECONDS", 3)


def test_wave_parsing_hand_calculated(real_stage):
    assert [x.time for x in real_stage.spawns] == [
        5,
        16,
        18,
        29,
        29.7,
        34,
        45,
        45.7,
        52,
        52.7,
        63.7,
    ]
    assert [x.route_index for x in real_stage.spawns] == [
        2,
        3,
        3,
        4,
        4,
        5,
        6,
        6,
        7,
        7,
        9,
    ]


def test_enemy_override(real_stage):
    e = next(x for x in real_stage.enemies if x.id == "enemy_1002_nsabr")
    assert e.defense == 30 and e.hp > 0
    assert defined_merge({"a": 12}, {"a": {"m_defined": False, "m_value": 0}}) == {
        "a": 12
    }
    assert defined_merge({"a": 12}, {"a": {"m_defined": True, "m_value": 0}}) == {
        "a": 0
    }


def test_operator_and_skill(squad):
    n, m = squad
    assert n.hp == 1219 and n.block_count == 3 and n.skill is None
    assert m.atk == 396 and m.attack_range == ((0, 0), (1, 0))
    assert (
        m.skill.cost == 50
        and m.skill.duration == 20
        and m.skill.attack_multiplier == 1.1
    )


def test_paths_avoid_walls(real_stage):
    for i in {s.route_index for s in real_stage.spawns}:
        nodes = compile_route(real_stage.map, real_stage.routes[i])
        for w in nodes:
            if w.kind != "WAIT_FOR_SECONDS":
                assert real_stage.map.tile(*map(round, w.position)).passable == "ALL"
        assert nodes[-1].position == (0, 3)


@pytest.mark.parametrize(
    "change",
    ["index", "conditional", "rune", "predefined", "unknown_action"],
)
def test_reject_unsupported(change):
    d = json.load(open(DATA / "level_main_00-01.json"))
    if change == "index":
        d["mapData"]["map"][0][0] = -1
    if change == "conditional":
        d["waves"][0]["fragments"][1]["actions"][1]["hiddenGroup"] = "external_trigger"
    if change == "rune":
        d["runes"][0]["difficultyMask"] = "ALL"
        d["runes"][0]["key"] = "unimplemented_test_rune"
    if change == "predefined":
        d["predefines"]["characterInsts"] = [{}]
    if change == "unknown_action":
        d["waves"][0]["fragments"][0]["actions"][0]["actionType"] = "UNKNOWN"
    with pytest.raises((ValueError, NotImplementedError)):
        StageLoader(DATA / "enemy_database.json").parse(d)


def test_unknown_skill_rejected():
    with pytest.raises(NotImplementedError):
        SkillLoader(DATA / "skill_table.json").load("unknown")


def test_independent_prts_observation(real_stage):
    from conftest import ROOT

    observation = json.loads((ROOT / "research/prts_map_observation.json").read_text())
    expected = [
        (g["enemy"], g["time"] + i * g["interval"])
        for g in observation["groups"]
        for i in range(g["count"])
    ]
    assert [(s.enemy_id, s.time) for s in real_stage.spawns] == expected
    enemy = next(e for e in real_stage.enemies if e.id == "enemy_1002_nsabr")
    assert all(getattr(enemy, k) == v for k, v in observation["soldier"].items())
