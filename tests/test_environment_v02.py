from dataclasses import FrozenInstanceError, replace
import pytest
from arknights_sim.environment import ArknightsEnv, Action, ActionType, Direction, WAIT


def test_action_equality_hash_serialization():
    a = Action(ActionType.DEPLOY, "op", (2, 0), Direction.LEFT)
    b = Action.from_dict(a.to_dict())
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    assert "LEFT" in str(a) and "(2, 0)" in str(a)
    with pytest.raises(FrozenInstanceError):
        a.tile = (1, 0)


@pytest.mark.parametrize(
    "args",
    [
        {"type": "DEPLOY", "operator_id": "op", "tile": (1, 0)},
        {"type": "WAIT", "operator_id": "op"},
        {"type": "RETREAT"},
        {"type": "DEPLOY", "operator_id": "op", "tile": (1.5, 0), "direction": "UP"},
        {"type": "DEPLOY", "operator_id": "op", "tile": (1, 0), "direction": "INVALID"},
    ],
)
def test_bad_action_shape(args):
    with pytest.raises(ValueError):
        Action(**args)


def test_deployment_all_tiles_directions(simple, operator):
    env = ArknightsEnv(simple, [operator])
    s = env.reset()
    deploy = [a for a in env.legal_actions(s) if a.type == ActionType.DEPLOY]
    assert len(deploy) == 8 * 4
    for action in deploy:
        child = env.step(s, action)
        assert child.game.operators["op"].position == action.tile
    assert s.game.operators == {}


def test_tile_and_occupied_filter(simple, operator):
    rows = list(simple.map.rows[0])
    rows[0] = replace(rows[0], buildable="NONE")
    rows[1] = replace(rows[1], buildable="RANGED")
    stage = replace(simple, map=replace(simple.map, rows=(tuple(rows),)))
    env = ArknightsEnv(stage, [operator, replace(operator, id="second")])
    s = env.reset()
    assert all(
        a.tile not in [(0, 0), (1, 0)]
        for a in env.legal_actions(s)
        if a.type == ActionType.DEPLOY
    )
    s = env.step(s, Action("DEPLOY", "op", (2, 0), "RIGHT"))
    assert all(
        a.tile != (2, 0) and a.operator_id != "op"
        for a in env.legal_actions(s)
        if a.type == ActionType.DEPLOY
    )
    assert Action("RETREAT", "op") in env.legal_actions(s)


def test_dp_cooldown_and_wait_threshold(simple, operator):
    stage = replace(simple, initial_cost=7)
    env = ArknightsEnv(stage, [operator])
    s = env.reset()
    assert env.legal_actions(s) == (WAIT,)
    # WAIT exposes tile events too; availability occurs exactly at its threshold.
    while not any(a.type == ActionType.DEPLOY for a in env.legal_actions(s)):
        s = env.step(s, WAIT)
    assert s.game.current_time == pytest.approx(3)
    assert "DEPLOY_AVAILABLE" in s.last_decision.reasons
    s = env.step(s, Action("DEPLOY", "op", (4, 0), "LEFT"))
    s = env.step(s, Action("RETREAT", "op"))
    assert not any(a.type == ActionType.DEPLOY for a in env.legal_actions(s))


def test_skill_filtering(simple, operator):
    env = ArknightsEnv(simple, [operator])
    s = env.reset()
    assert not any(a.type == ActionType.ACTIVATE_SKILL for a in env.legal_actions(s))
    s = env.step(s, Action("DEPLOY", "op", (7, 0), "RIGHT"))
    assert not any(a.type == ActionType.ACTIVATE_SKILL for a in env.legal_actions(s))
    s = env.advance_to_time(s, 2)
    assert Action("ACTIVATE_SKILL", "op") in env.legal_actions(s)
    s = env.step(s, Action("ACTIVATE_SKILL", "op"))
    assert not any(a.type == ActionType.ACTIVATE_SKILL for a in env.legal_actions(s))
    auto = replace(operator, skill=replace(operator.skill, trigger="AUTO_TRIGGER"))
    e = ArknightsEnv(simple, [auto])
    s = e.step(e.reset(), Action("DEPLOY", "op", (7, 0), "RIGHT"))
    s = e.advance_to_time(s, 2)
    assert not any(a.type == ActionType.ACTIVATE_SKILL for a in e.legal_actions(s))


def test_dead_retreat_filtered(simple, operator):
    env = ArknightsEnv(simple, [operator])
    s = env.step(env.reset(), Action("DEPLOY", "op", (0, 0), "RIGHT"))
    s.game.operators["op"].alive = False
    assert Action("RETREAT", "op") not in env.legal_actions(s)


def test_wait_next_decision_not_frame(simple):
    env = ArknightsEnv(simple, [])
    s = env.reset()
    c = env.advance_to_next_decision_event(s)
    assert c.game.current_time >= 0.5
    assert "KEY_TILE_CHANGED" in c.last_decision.reasons
    assert s.game.current_time == 0


def test_wait_empty_interval_jumps_spawn(simple):
    wave = replace(simple.waves[0], spawns=(replace(simple.spawns[0], time=20),))
    env = ArknightsEnv(replace(simple, waves=(wave,)), [])
    s = env.reset()
    c = env.step(s, WAIT)
    assert c.game.current_time == 20 and "ENEMY_SPAWN" in c.last_decision.reasons


def test_wait_horizon_and_decision_limits(simple, operator):
    enemy = replace(simple.enemies[0], speed=0)
    env = ArknightsEnv(replace(simple, enemies=(enemy,)), [], horizon=10)
    s = env.step(env.reset(), WAIT)
    assert env.is_terminal(s) and env.result(s).termination == "time_limit"
    assert not env.result(s).success and env.legal_actions(s) == ()
    env = ArknightsEnv(simple, [operator], max_decisions=1)
    s = env.step(env.reset(), Action("DEPLOY", "op", (0, 0), "RIGHT"))
    assert env.result(s).termination == "decision_limit"


def test_clone_key_and_trace_isolation(simple, operator):
    env = ArknightsEnv(simple, [operator], trace=True)
    s = env.reset()
    key = env.state_key(s)
    c = env.clone_state(s)
    assert env.state_key(c) == key
    c.trace[0]["event"] = "modified"
    assert s.trace[0]["event"] == "BATTLE_START"
    child = env.step(s, Action("DEPLOY", "op", (2, 0), "LEFT"))
    assert env.state_key(s) == key and env.state_key(child) != key
    child.game.rng.random()
    assert s.game.rng.getstate() != child.game.rng.getstate()


def test_pure_deterministic_transition(simple, operator):
    env = ArknightsEnv(simple, [operator], trace=True)
    a = env.reset(seed=1)
    b = env.reset(seed=1)
    actions = [Action("DEPLOY", "op", (2, 0), "LEFT"), WAIT, WAIT, WAIT]
    for action in actions:
        a = env.step(a, action)
        b = env.step(b, action)
    assert (
        env.state_key(a) == env.state_key(b)
        and a.trace == b.trace
        and a.last_decision == b.last_decision
    )


def test_illegal_transition_atomic(simple, operator):
    env = ArknightsEnv(simple, [operator])
    s = env.reset()
    key = env.state_key(s)
    with pytest.raises(ValueError):
        env.step(s, Action("DEPLOY", "op", (99, 0), "UP"))
    assert env.state_key(s) == key


def test_environment_strict_success_not_core_win(simple):
    env = ArknightsEnv(simple, [])
    s = env.reset()
    while not env.is_terminal(s):
        s = env.step(s, WAIT)
    assert (
        s.game.result == "WIN"
        and not env.result(s).success
        and env.result(s).leaks == 1
    )


@pytest.mark.parametrize("operator_id", [["op"], {}, 1, True, ""])
def test_operator_id_is_hashable_string(operator_id):
    with pytest.raises(ValueError):
        Action("RETREAT", operator_id)
