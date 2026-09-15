"""Audited summon handlers. Raw token presence never enables an unknown runtime."""
from .models import OperatorData, SummonData
from .progression import unlocked

DEEPCOLOR = "char_110_deepcl"
TENTACLE = "token_10001_deepcl_tentac"
SUPPORTED_SUMMON_PAIRS = frozenset({(DEEPCOLOR, TENTACLE)})


def _talents(raw, elite, level, potential):
    for talent in raw.get("talents") or ():
        eligible = [c for c in talent["candidates"]
                    if unlocked(c["unlockCondition"], elite, level)
                    and c.get("requiredPotentialRank", 0) <= potential]
        if eligible:
            yield max(eligible, key=lambda c: (
                int(c["unlockCondition"]["phase"].split("_")[-1]),
                c["unlockCondition"]["level"], c.get("requiredPotentialRank", 0)))


def load_summons(loader, owner_id, elite, level, potential):
    if owner_id != DEEPCOLOR:
        return ()
    raw = loader.chars[TENTACLE]
    phase = raw["phases"][elite]
    lo, hi = phase["attributesKeyFrames"][0], phase["attributesKeyFrames"][-1]
    ratio = (level - lo["level"]) / max(1, hi["level"] - lo["level"])
    stats = dict(lo["data"])
    for field in ("maxHp", "atk", "def", "magicResistance"):
        value = lo["data"][field] + ratio * (hi["data"][field] - lo["data"][field])
        stats[field] = int(value + .5) if field != "magicResistance" else value
    owner_talents = tuple(_talents(loader.chars[owner_id], elite, level, potential))
    inventory = next(int(b["value"]) for c in owner_talents
                     if c.get("tokenKey") == TENTACLE
                     for b in c["blackboard"] if b["key"] == "cnt")
    max_active = stats["maxDeployCount"] + sum(
        b["value"] for c in _talents(raw, elite, level, 0)
        for b in c["blackboard"] if b["key"] == "max_deploy_count")
    unit = OperatorData(
        id=TENTACLE, name=raw["name"], hp=stats["maxHp"], atk=stats["atk"],
        defense=stats["def"], resistance=stats["magicResistance"],
        interval=stats["baseAttackTime"] * 100 / stats["attackSpeed"],
        block_count=stats["blockCnt"], cost=stats["cost"], redeploy=stats["respawnTime"],
        attack_range=tuple((v["col"], v["row"]) for v in loader.ranges[phase["rangeId"]]["grids"]),
        position_type=raw["position"], damage_type="PHYSICAL", healable=False,
        elite=elite, level=level,
        simulation_notes=("无模组触手；生命周期规则待实机校准；继承核心阻挡/前摇假设",),
    )
    return (SummonData(owner_id, unit, inventory, int(max_active)),)
