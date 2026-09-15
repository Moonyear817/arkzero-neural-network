from dataclasses import replace
import pytest
from arknights_sim import Simulator
from arknights_sim.data.models import WaypointData, SpawnData
from arknights_sim.combat.damage import DamageCalculator as D
from arknights_sim.combat.targeting import OperatorTargetSelector, range_cells
from arknights_sim.combat.healing import heal
from arknights_sim.skills.modifier import Modifier, modified


@pytest.mark.parametrize(
    "atk,defense,expected", [(100, 20, 80), (100, 200, 5), (0, 200, 0), (100, -20, 120)]
)
def test_physical(atk, defense, expected):
    assert D.physical_damage(atk, defense) == expected


@pytest.mark.parametrize(
    "res,expected", [(0, 100), (50, 50), (100, 5), (150, 5), (-20, 120)]
)
def test_arts(res, expected):
    assert D.arts_damage(100, res) == expected


def test_true():
    assert D.calculate("TRUE", 100, 999, 999) == 100


def test_spawn_exact(simple):
    w = replace(simple.waves[0], spawns=(SpawnData(0.125, "e", 0),))
    sim = Simulator(replace(simple, waves=(w,)), trace=True)
    sim.run_until(1)
    event = next(e for e in sim.trace.events if e["event"] == "ENEMY_SPAWN")
    assert event["time"] == 0.125
    assert sim.state.enemies["enemy_0000"].position[0] == pytest.approx(0.875)


def test_movement_goal(simple):
    sim = Simulator(simple, trace=True)
    sim.run_until(2)
    assert sim.state.enemies["enemy_0000"].position == pytest.approx((2, 0))
    sim.run(10)
    assert sim.state.escaped == 1 and sim.state.life == 19 and sim.state.done
    assert next(e for e in sim.trace.events if e["event"] == "ESCAPE")[
        "time"
    ] == pytest.approx(7)


def test_waypoint_wait(simple):
    r = replace(
        simple.routes[0],
        waypoints=(
            WaypointData("MOVE", (2, 0)),
            WaypointData("WAIT_FOR_SECONDS", (0, 0), 2),
        ),
    )
    sim = Simulator(replace(simple, routes=(r,)), trace=True)
    sim.run_until(3)
    assert sim.state.enemies["enemy_0000"].position == pytest.approx((2, 0))
    sim.run_until(5)
    assert sim.state.enemies["enemy_0000"].position[0] == pytest.approx(3)
    sim.run(12)
    assert sim.state.escaped == 1
    wait = next(e for e in sim.trace.events if e["event"] == "WAIT")
    moving = next(e for e in sim.trace.events if e["event"] == "MOVING")
    assert moving["time"] - wait["time"] == pytest.approx(2)


def test_block_and_attacks(simple, operator):
    sim = Simulator(simple, [operator], trace=True)
    op = sim.deploy("op", (2, 0), 2)
    sim.run_until(3)
    assert op.blocked_enemies == ["enemy_0000"]
    e = sim.state.enemies["enemy_0000"]
    assert e.blocked_by == "op" and e.position[0] <= 2
    assert op.hp < 1000 and e.hp < 100
    pos = e.position
    sim.run_until(3.1)
    assert e.position == pos


def test_block_count(simple, operator):
    enemy = replace(simple.enemies[0], hp=10000)
    wave = replace(
        simple.waves[0], spawns=tuple(SpawnData(0, "e", 0) for _ in range(3))
    )
    sim = Simulator(replace(simple, enemies=(enemy,), waves=(wave,)), [operator])
    op = sim.deploy("op", (2, 0), 2)
    sim.run_until(3)
    assert len(op.blocked_enemies) == 1
    assert sum(e.blocked_by is not None for e in sim.state.enemies.values()) == 1
    assert sum(e.position[0] > 2 for e in sim.state.enemies.values()) == 2


def test_enemy_death_releases(simple, operator):
    sim = Simulator(simple, [replace(operator, atk=1000)], trace=True)
    op = sim.deploy("op", (1, 0))
    sim.run(20)
    assert sim.state.killed == 1 and not op.blocked_enemies
    assert not sim.state.enemies["enemy_0000"].alive


def test_operator_death_releases(simple, operator):
    enemy = replace(simple.enemies[0], atk=5000, hp=10000)
    sim = Simulator(replace(simple, enemies=(enemy,)), [operator], trace=True)
    op = sim.deploy("op", (1, 0))
    sim.run_until(2)
    assert not op.alive and op.hp == 0 and not op.blocked_enemies
    assert sim.state.enemies["enemy_0000"].blocked_by is None
    assert sim.state.redeploy_at["op"] > 5


def test_targeting_priorities_and_tie(simple, operator):
    w = replace(simple.waves[0], spawns=(SpawnData(0, "e", 0), SpawnData(0, "e", 0)))
    sim = Simulator(replace(simple, waves=(w,)), [operator])
    op = sim.deploy("op", (0, 0))
    sim.run_until(0)
    a, b = sim.state.enemies.values()
    a.position = b.position = (0.2, 0)
    assert OperatorTargetSelector.select(op, [b, a], lambda e: 1).id == a.id
    b.blocked_by = "op"
    assert OperatorTargetSelector.select(op, [a, b], lambda e: 1).id == b.id
    b.blocked_by = None
    b.position = (0.8, 0)
    assert (
        OperatorTargetSelector.select(op, [a, b], lambda e: 2 - e.position[0]).id
        == b.id
    )


def test_attack_windup(simple, operator):
    sim = Simulator(simple, [operator], windup=0.2, trace=True)
    sim.deploy("op", (0, 0))
    sim.run_until(0.1)
    assert sim.state.enemies["enemy_0000"].hp == 100
    sim.run_until(0.3)
    assert sim.state.enemies["enemy_0000"].hp == 70
    start = next(e for e in sim.trace.events if e["event"] == "ATTACK_START")
    hit = next(e for e in sim.trace.events if e["event"] == "DAMAGE")
    assert hit["time"] - start["time"] == pytest.approx(0.2)


def test_deploy_dp_retreat_redeploy(simple, operator):
    sim = Simulator(simple, [operator])
    op = sim.deploy("op", (3, 0))
    assert sim.state.dp == 89
    sim.retreat("op")
    assert sim.state.dp == 94
    with pytest.raises(ValueError):
        sim.deploy("op", (3, 0))
    sim.run_until(5)
    assert sim.state.dp == pytest.approx(99)
    op = sim.deploy("op", (3, 0))
    assert op.paid_cost == 15 and sim.state.dp == pytest.approx(84) and op.hp == 1000


@pytest.mark.parametrize(
    "reason", ["cost", "occupied", "bad_direction", "out_of_map", "unknown", "limit"]
)
def test_invalid_deploy(simple, operator, reason):
    second = replace(operator, id="other")
    s = replace(simple, initial_cost=0) if reason == "cost" else simple
    if reason == "limit":
        s = replace(s, character_limit=0)
    sim = Simulator(s, [operator, second])
    if reason == "occupied":
        sim.deploy("other", (3, 0))
    before = sim.stable_hash()
    with pytest.raises(ValueError):
        sim.deploy(
            "unknown" if reason == "unknown" else "op",
            (100, 0) if reason == "out_of_map" else (3, 0),
            5 if reason == "bad_direction" else 0,
        )
    assert sim.stable_hash() == before


def test_skill_activation_expiration(simple, operator):
    e = replace(simple.enemies[0], hp=10000, atk=0, speed=0.1)
    sim = Simulator(replace(simple, enemies=(e,)), [operator], trace=True)
    op = sim.deploy("op", (7, 0))
    with pytest.raises(ValueError):
        sim.activate_skill("op")
    sim.run_until(2)
    assert op.skill.sp == pytest.approx(2)
    sim.activate_skill("op")
    assert modified(30, "ATK", op.modifiers, 2) == 60
    sim.run_until(3)
    assert op.skill.sp == 0
    with pytest.raises(ValueError):
        sim.activate_skill("op")
    sim.run_until(4)
    assert modified(30, "ATK", op.modifiers, 4) == 30 and op.skill.sp == 0
    sim.run_until(5)
    assert op.skill.sp == pytest.approx(1)
    assert any(e["event"] == "SKILL_END" for e in sim.trace.events)


def test_clone_hash_rng_queue(simple, operator):
    sim = Simulator(simple, [operator])
    sim.deploy("op", (3, 0))
    sim.run_until(2)
    clone = sim.clone()
    assert clone.stable_hash() == sim.stable_hash()
    before = sim.stable_hash()
    clone.step(1)
    assert sim.stable_hash() == before and clone.stable_hash() != before
    state = sim.state.clone()
    state.rng.random()
    assert state.stable_hash() != sim.stable_hash()
    snapshot = sim.snapshot()
    snapshot["dp"] = 0
    assert sim.state.dp > 0
    clone = sim.state.clone()
    clone.step(1)
    assert clone.stable_hash() != sim.stable_hash()


def test_partition_determinism(simple, operator):
    a = Simulator(simple, [operator])
    b = a.clone()
    a.deploy("op", (2, 0))
    b.deploy("op", (2, 0))
    a.run_until(3)
    for _ in range(30):
        b.step(0.1)
    assert a.stable_hash() == b.stable_hash()


def test_real_determinism(real_stage, squad):
    a = Simulator(real_stage, squad, seed=12345, trace=True)
    b = Simulator(real_stage, squad, seed=12345, trace=True)
    a.run(180)
    b.run(180)
    assert a.trace.events == b.trace.events and a.stable_hash() == b.stable_hash()
    c = Simulator(real_stage, squad, seed=12346)
    c.run(180)
    assert c.stable_hash() != a.stable_hash()
    assert a.state.spawned == a.state.escaped == 11


def test_full_real_battle(real_stage, squad):
    sim = Simulator(real_stage, squad, trace=True)
    sim.run_until(4)
    sim.deploy("char_500_noirc", (3, 2))
    sim.run_until(17)
    sim.deploy("char_208_melan", (1, 2))
    sim.run_until(67)
    sim.activate_skill("char_208_melan")
    sim.run(180)
    assert sim.state.done and sim.state.killed == 11 and sim.state.escaped == 0
    assert sim.state.life == 20 and sim.state.result == "WIN"
    assert {"BLOCK", "ATTACK_START", "DAMAGE", "DEATH", "SKILL_START", "SKILL_END"} <= {
        e["event"] for e in sim.trace.events
    }


def test_modifiers_and_healing(simple, operator):
    sim = Simulator(simple, [operator])
    op = sim.deploy("op", (2, 0))
    op.hp = 1
    heal(op, 2000)
    assert op.hp == 1000
    mods = [
        Modifier("s", "op", "DEF_ADD", 10, 0, 2),
        Modifier("t", "op", "DEF_MULTIPLY", 2, 0, 2),
    ]
    assert modified(20, "DEF", mods, 1) == 60 and modified(20, "DEF", mods, 2) == 20
    op.direction = 1
    assert range_cells(op) == [(2, 0), (2, 1)]


def test_loss_terminal(simple):
    sim = Simulator(replace(simple, life=1))
    sim.run(20)
    assert sim.state.result == "LOSS"
    h = sim.stable_hash()
    sim.run(100)
    assert sim.stable_hash() == h


def test_grid_time_rejection(simple):
    sim = Simulator(simple)
    with pytest.raises(ValueError):
        sim.step(0.001)
    with pytest.raises(ValueError):
        sim.run_until(-1)


def test_retreat_cancels_pending_hit(simple, operator):
    sim = Simulator(simple, [operator], windup=0.5)
    sim.deploy("op", (0, 0))
    sim.run_until(0.1)
    sim.retreat("op")
    sim.run_until(0.7)
    assert sim.state.enemies["enemy_0000"].hp == 100


def test_redeploy_generation_cancels_old_target_hit(simple, operator):
    op = replace(operator, redeploy=0, atk=0)
    sim = Simulator(simple, [op], windup=0.5)
    sim.deploy("op", (0, 0))
    sim.run_until(0.1)
    sim.retreat("op")
    new = sim.deploy("op", (0, 0))
    sim.run_until(0.6)
    assert new.hp == 1000


def test_generic_modifier_expiry_and_block_release(simple, operator):
    enemy = replace(simple.enemies[0], hp=10000, atk=0)
    sim = Simulator(replace(simple, enemies=(enemy,)), [operator])
    op = sim.deploy("op", (2, 0))
    sim.run_until(2)
    assert len(op.blocked_enemies) == 1
    sim.add_modifier(Modifier("test", "op", "BLOCK_COUNT_ADD", -1, 2, 4))
    sim.run_until(3)
    assert not op.blocked_enemies and sim.state.enemies["enemy_0000"].blocked_by is None


def test_move_speed_modifier(simple):
    sim = Simulator(simple)
    sim.run_until(0)
    sim.add_modifier(Modifier("slow", "enemy_0000", "MOVE_SPEED_MULTIPLY", 0.5, 0, 2))
    sim.run_until(4)
    assert sim.state.enemies["enemy_0000"].position[0] == pytest.approx(3)


def test_pending_event_hash(simple, operator):
    a = Simulator(simple, [operator])
    a.deploy("op", (0, 0))
    a.run_until(0.1)
    b = a.clone()
    assert b.state.queue.heap and b.stable_hash() == a.stable_hash()
    b.state.queue.pop()
    assert a.stable_hash() != b.stable_hash()


def test_trace_disabled_does_not_change_state(simple, operator):
    a = Simulator(simple, [operator], trace=True)
    b = Simulator(simple, [operator], trace=False)
    a.deploy("op", (2, 0))
    b.deploy("op", (2, 0))
    a.run(20)
    b.run(20)
    assert a.stable_hash() == b.stable_hash() and b.trace.events == []


def test_real_wait_matches_prts(real_stage):
    sim = Simulator(real_stage, trace=True)
    sim.run(180)
    waits = [e for e in sim.trace.events if e["event"] == "WAIT"]
    assert len(waits) == 1 and waits[0]["duration"] == 3
    moving = next(
        e
        for e in sim.trace.events
        if e["event"] == "MOVING" and e["unit"] == waits[0]["unit"]
    )
    assert moving["time"] - waits[0]["time"] == pytest.approx(3)
