"""Engine golden fixtures are regression contracts, not official-game truth."""

from dataclasses import replace
from itertools import permutations, combinations
import json
from pathlib import Path

import pytest

from arknights_sim import Simulator
from arknights_sim.combat.attack import AttackEvent
from arknights_sim.combat.blocking import try_block
from arknights_sim.core.event_priority import EVENT_PRIORITY, SIMULATION_PHASES
from arknights_sim.core.event_queue import EventQueue
from arknights_sim.data.models import MapData, RouteData, SpawnData, WaypointData


GOLDEN = Path(__file__).resolve().parents[1] / "golden_tests"


def fixture(name):
    data = json.loads((GOLDEN / f"{name}.json").read_text())
    assert data["status"] == "ASSUMED"
    return data


def hit(attacker="op", target="enemy_0000"):
    return AttackEvent(attacker, target, 0, 1, 2, 1, 1)


def drain(queue):
    result = []
    while queue.heap:
        event = queue.pop()
        result.append((event.kind, event.payload))
    return result


def test_same_time_kind_priority_is_independent_of_push_order():
    golden = fixture("same_frame_event_priority")
    events = [
        ("HIT", hit()),
        ("SPAWN", SpawnData(1, "e", 0)),
        ("SKILL_END", "op", 1),
        ("MODIFIER_BOUNDARY", "op"),
        ("AREA_HIT", "drone", (1, 0), 100, 1.2),
        ("SELF_DAMAGE", "enemy_0000"),
        ("BURN_TICK", "enemy_0000", 0, 1),
        ("DEVICE_PULSE", "altar"),
        ("TALENT_EVENT", "SURVIVAL_END", "op", 1),
    ]
    # Every pair in both insertion orders proves the total ordering without a
    # factorial test cost as the event vocabulary grows. Also drain full sets.
    orderings = [events,list(reversed(events))]
    orderings += [list(p) for pair in combinations(events,2) for p in permutations(pair)]
    for ordering in orderings:
        queue = EventQueue()
        for event in ordering:
            queue.push(1, *event)
        present={e[0] for e in ordering}
        assert [kind for kind, _ in drain(queue)] == [k for k in golden["queued_order"] if k in present]
    assert list(SIMULATION_PHASES) == golden["phase_order"]
    assert sorted(EVENT_PRIORITY, key=EVENT_PRIORITY.get) == golden["queued_order"]


@pytest.mark.parametrize(
    "events",
    [
        [("HIT", hit("b")), ("HIT", hit("a"))],
        [("SPAWN", SpawnData(1, "b", 0)), ("SPAWN", SpawnData(1, "a", 0))],
        [("SKILL_END", "b", 1), ("SKILL_END", "a", 1)],
        [("MODIFIER_BOUNDARY", "b"), ("MODIFIER_BOUNDARY", "a")],
    ],
)
def test_same_kind_semantic_ties_ignore_push_order(events):
    outputs = []
    for ordering in [events, list(reversed(events))]:
        queue = EventQueue()
        for event in ordering:
            queue.push(1, *event)
        outputs.append(drain(queue))
    assert outputs[0] == outputs[1]


def test_event_time_precedes_kind_priority():
    queue = EventQueue()
    queue.push(2, "MODIFIER_BOUNDARY", "op")
    queue.push(1, "HIT", hit())
    assert queue.peek() == 1
    assert queue.pop().kind == "HIT"
    assert queue.pop().kind == "MODIFIER_BOUNDARY"
    assert queue.peek() == float("inf")


@pytest.mark.parametrize("time", [-1, float("nan"), float("inf")])
def test_invalid_event_time_does_not_mutate_queue(time):
    queue = EventQueue()
    with pytest.raises(ValueError):
        queue.push(time, "MODIFIER_BOUNDARY", "op")
    assert queue.sequence == 0 and queue.heap == []


def test_new_event_kind_requires_explicit_priority():
    queue = EventQueue()
    with pytest.raises(ValueError, match="explicit priority"):
        queue.push(0, "UNDECLARED_KIND")
    assert queue.sequence == 0 and queue.heap == []


def test_same_time_lethal_hits_resolve_identically_when_inserted_in_reverse(simple, operator):
    outputs = []
    for reverse in [False, True]:
        sim = Simulator(simple, [replace(operator, hp=20, defense=0)], trace=True)
        op = sim.deploy("op", (0, 0))
        sim.run_until(0)
        enemy = sim.state.enemies["enemy_0000"]
        enemy.hp = 30
        op.ready_at = enemy.ready_at = 10
        sim.state.queue = EventQueue()
        attacks = [
            AttackEvent(op.id, enemy.id, 0, 0.5, 10, op.generation, enemy.generation),
            AttackEvent(enemy.id, op.id, 0, 0.5, 10, enemy.generation, op.generation),
        ]
        for attack in reversed(attacks) if reverse else attacks:
            sim.state.queue.push(0.5, "HIT", attack)
        sim.run_until(0.5)
        # "enemy_0000" sorts before "op" by the explicit assumed HIT tie rule.
        assert not op.alive and enemy.alive and enemy.hp == 30
        outputs.append((sim.trace.events, sim.stable_hash()))
    assert outputs[0] == outputs[1]


def test_golden_continuous_turning(simple):
    golden = fixture("continuous_turning")
    tile = simple.map.tile(0, 0)
    route = RouteData(
        tuple(golden["start"]),
        tuple(golden["end"]),
        tuple(WaypointData("MOVE", tuple(p)) for p in golden["waypoints"]),
        diagonal=False,
    )
    stage = replace(simple, map=MapData(((tile,) * 3,) * 3), routes=(route,))
    sim = Simulator(stage)
    for time, position in golden["samples"]:
        sim.run_until(time)
        assert sim.state.enemies["enemy_0000"].position == pytest.approx(position)


def test_golden_blocking_radius(simple, operator):
    golden = fixture("blocking_radius")
    for enemy_x, expected in golden["samples"]:
        sim = Simulator(simple, [operator])
        op = sim.deploy("op", tuple(golden["operator_position"]))
        sim.run_until(0)
        enemy = sim.state.enemies["enemy_0000"]
        enemy.position = (enemy_x, 0)
        blocker = try_block(enemy, sim.state.operators, sim.state.enemies, 0)
        assert (blocker is op) is expected
        assert (enemy.blocked_by == op.id) is expected


def test_golden_attack_windup(simple, operator):
    golden = fixture("attack_windup")
    sim = Simulator(simple, [operator], trace=True, windup=golden["configured_windup"])
    sim.deploy("op", (0, 0))
    sim.run_until(golden["before_time"])
    assert sim.state.enemies["enemy_0000"].hp == golden["initial_enemy_hp"]
    sim.run_until(golden["after_time"])
    assert sim.state.enemies["enemy_0000"].hp == golden["enemy_hp_after_hit"]
    attack = next(e for e in sim.trace.events if e["event"] == "ATTACK_START")
    damage = next(e for e in sim.trace.events if e["event"] == "DAMAGE")
    assert damage["time"] - attack["time"] == pytest.approx(golden["configured_windup"])
