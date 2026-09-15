import argparse, json
from common import stage, operators, ROOT
from arknights_sim import Simulator

p = argparse.ArgumentParser()
p.add_argument("--no-operators", action="store_true")
p.add_argument("--trace", default="research/battle_trace.txt")
a = p.parse_args()
sim = Simulator(stage(), operators(), trace=True)
if not a.no_operators:
    sim.run_until(4)
    sim.deploy("char_500_noirc", (3, 2), 0)
    sim.run_until(17)
    sim.deploy("char_208_melan", (1, 2), 0)
    sim.run_until(67)
    if not sim.state.done:
        sim.activate_skill("char_208_melan")
sim.run(300)
(ROOT / a.trace).write_text(sim.trace.text() + "\n")
s = sim.state
print(
    json.dumps(
        {
            "stage": s.stage.id,
            "done": s.done,
            "result": s.result,
            "time": s.current_time,
            "spawned": s.spawned,
            "killed": s.killed,
            "escaped": s.escaped,
            "life": s.life,
            "hash": s.stable_hash(),
        },
        indent=2,
    )
)
