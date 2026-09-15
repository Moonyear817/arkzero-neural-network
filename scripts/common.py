from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader

DATA = ROOT / "data/real"


def stage(path="level_main_00-01"):
    p = Path(path)
    if not p.exists():
        p = DATA / (path + ".json" if not path.endswith(".json") else path)
    return StageLoader(DATA / "enemy_database.json").load(p)


def operators():
    l = OperatorLoader(
        DATA / "character_table.json",
        DATA / "range_table.json",
        SkillLoader(DATA / "skill_table.json"),
    )
    return [l.load("char_500_noirc"), l.load("char_208_melan")]
