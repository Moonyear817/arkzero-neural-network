"""Independent deterministic grid path planner; steering approximation RULE-002."""

import heapq, math
from arknights_sim.data.models import WaypointData


def pathfind(grid, start, end, diagonal=True):
    s = tuple(round(v) for v in start)
    g = tuple(round(v) for v in end)

    def ok(p):
        x, y = p
        return (
            0 <= x < grid.width
            and 0 <= y < grid.height
            and grid.tile(x, y).passable == "ALL"
        )

    if not ok(s) or not ok(g):
        raise ValueError(f"Unwalkable endpoint {s} -> {g}")
    queue = [(0, s)]
    dist = {s: 0}
    parent = {}
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)] + (
        [(1, 1), (1, -1), (-1, 1), (-1, -1)] if diagonal else []
    )
    while queue:
        cost, p = heapq.heappop(queue)
        if cost > dist[p]:
            continue
        if p == g:
            break
        for dx, dy in directions:
            q = (p[0] + dx, p[1] + dy)
            if not ok(q) or (
                dx and dy and (not ok((p[0] + dx, p[1])) or not ok((p[0], p[1] + dy)))
            ):
                continue
            v = cost + math.hypot(dx, dy)
            if v < dist.get(q, float("inf")):
                dist[q] = v
                parent[q] = p
                heapq.heappush(queue, (v, q))
    if g not in dist:
        raise ValueError("No path")
    points = [g]
    while points[-1] != s:
        points.append(parent[points[-1]])
    points.reverse()
    return (
        [tuple(map(float, p)) for p in points[1:-1]] + [end]
        if s != g
        else ([end] if start != end else [])
    )


WAIT_KINDS = frozenset({"WAIT_FOR_SECONDS", "WAIT_CURRENT_WAVE_TIME",
                        "WAIT_CURRENT_FRAGMENT_TIME"})
ROUTE_KINDS = WAIT_KINDS | {"MOVE", "DISAPPEAR", "APPEAR_AT_POS"}


def compile_route(grid, route):
    result = []
    pos = route.start
    for w in (*route.waypoints, WaypointData("GOAL", route.end)):
        if w.kind in WAIT_KINDS or w.kind == "DISAPPEAR":
            result.append(w)
            continue
        if w.kind == "APPEAR_AT_POS":
            result.append(w)
            pos = w.position
            continue
        if route.motion == "FLY":
            result.append(w)
            pos = w.position
            continue
        pts = pathfind(grid, pos, w.position, route.diagonal)
        result.extend(WaypointData("NODE", p) for p in pts[:-1])
        result.append(WaypointData(w.kind, w.position))
        pos = w.position
    return tuple(result)
