"""Compare loaded data/trace to manually observed independent PRTS.Map values."""

import json
from common import stage, ROOT
from arknights_sim import Simulator

s = stage()
o = json.loads((ROOT / "research/prts_map_observation.json").read_text())
expected = [
    (g["enemy"], round(g["time"] + i * g["interval"], 9))
    for g in o["groups"]
    for i in range(g["count"])
]
actual = [(x.enemy_id, round(x.time, 9)) for x in s.spawns]
checks = {
    "size": list((s.map.width, s.map.height)) == o["size"],
    "spawn_sequence_and_times": actual == expected,
    "map_spawn": s.map.tile(*o["spawn"]).key == "tile_start",
    "map_goal": s.map.tile(*o["goal"]).key == "tile_end",
    "options": all(getattr(s, k) == v for k, v in o["options"].items()),
}
e = next(e for e in s.enemies if e.id == "enemy_1002_nsabr")
checks["soldier_attributes"] = all(getattr(e, k) == v for k, v in o["soldier"].items())
sim = Simulator(s, trace=True)
sim.run(180)
waits = [e for e in sim.trace.events if e["event"] == "WAIT"]
checks["wait_duration"] = len(waits) == 1 and waits[0]["duration"] == o["wait_seconds"]
checks["all_enemies_arrive"] = sim.state.spawned == sim.state.escaped == 11
result = {
    "source": o["source"],
    "checks": checks,
    "passed": all(checks.values()),
    "continuous_trajectory_verified": False,
    "in_game_attack_timing_verified": False,
}
(ROOT / "research/stage_validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(1)
