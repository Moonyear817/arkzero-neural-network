from pathlib import Path
from dataclasses import replace
import pytest
from arknights_sim.data.models import *
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/real"


@pytest.fixture
def real_stage():
    return StageLoader(DATA / "enemy_database.json").load(
        DATA / "level_main_00-01.json"
    )


@pytest.fixture
def squad():
    loader = OperatorLoader(
        DATA / "character_table.json",
        DATA / "range_table.json",
        SkillLoader(DATA / "skill_table.json"),
    )
    return [loader.load("char_500_noirc"), loader.load("char_208_melan")]


@pytest.fixture
def simple():
    tile = TileData("tile_road", "LOWLAND", "MELEE", "ALL")
    grid = MapData((tuple(tile for _ in range(8)),))
    enemy = EnemyData("e", "test", 100, 20, 0, 0, 1, 1)
    route = RouteData((0, 0), (7, 0), diagonal=False)
    wave = WaveData(0, 0, (0,), (SpawnData(0, "e", 0),))
    return StageData("test", grid, (route,), (wave,), (enemy,), 99, 99, 1, 1, 20, 8)


@pytest.fixture
def operator():
    return OperatorData(
        "op",
        "test",
        1000,
        30,
        10,
        0,
        1,
        1,
        10,
        5,
        ((0, 0), (1, 0)),
        skill=SkillData("test", 2, 0, 2, 2),
    )
