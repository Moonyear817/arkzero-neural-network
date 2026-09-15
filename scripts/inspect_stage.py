import argparse
from common import stage

p = argparse.ArgumentParser()
p.add_argument("stage", nargs="?", default="level_main_00-01")
a = p.parse_args()
s = stage(a.stage)
print(
    f"Stage: {s.id}\nSize: {s.map.width}x{s.map.height}\nEnemies: {len(s.spawns)}\nRoutes: {len(s.routes)} ({len({x.route_index for x in s.spawns})} used)"
)
print(s.map.ascii())
print("R=spawn B=goal H=high ground #=unavailable; origin bottom-left")
for w in s.waves:
    for x in sorted(w.spawns, key=lambda x: x.time):
        print(
            f"Wave {x.wave + 1} fragment {x.fragment + 1}: {x.enemy_id} spawn={x.time:.3f} route={x.route_index}"
        )
for i in sorted({x.route_index for x in s.spawns}):
    r = s.routes[i]
    print(
        f"Route {i}: {r.start} -> "
        + " -> ".join(str((w.kind, w.position, w.time)) for w in r.waypoints)
        + f" -> {r.end}; jitter={r.random_range}"
    )
print("Assumptions:", *s.assumptions, sep="\n")
