import json, time, platform
from common import stage, operators, ROOT
from arknights_sim import Simulator

s = stage()
ops = operators()
samples = []
events = 0
seconds = 0
for _ in range(10):
    sim = Simulator(s, ops)
    start = time.perf_counter()
    sim.run(180)
    samples.append(time.perf_counter() - start)
    events += sim.state.events_processed
    seconds += sim.current_time
sim = Simulator(s, ops)
sim.run_until(20)
start = time.perf_counter()
for _ in range(200):
    sim.state.clone()
clone_time = time.perf_counter() - start
result = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "runs": 10,
    "trace": False,
    "scenario": "0-1, no operators; run until battle terminal",
    "simulated_seconds_per_run": seconds / 10,
    "mean_wall_seconds": sum(samples) / 10,
    "simulation_seconds_per_real_second": seconds / sum(samples),
    "events_per_second": events / sum(samples),
    "event_definition": "fixed ticks plus queued spawn/hit/skill events",
    "state_clones_per_second": 200 / clone_time,
}
(ROOT / "research/benchmark.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
