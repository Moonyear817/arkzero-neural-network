"""Isolation tests for profile-guided immutable-input sharing.

Timing results live in research/v02/clone_profile.json; tests assert correctness
instead of fragile minimum throughput on a shared development machine.
"""

import copy
from dataclasses import FrozenInstanceError, asdict, is_dataclass, replace

import pytest

from arknights_sim import Simulator
from arknights_sim.core.state import GameState
from arknights_sim.skills.modifier import Modifier


def battle(simple, operator):
    sim = Simulator(simple, [operator])
    sim.deploy("op", (2, 0))
    sim.run_until(2)
    sim.activate_skill("op")
    sim.add_modifier(Modifier("extra", "op", "DEF_ADD", 1, 2, 5))
    return sim.state


def test_clone_shares_proven_immutable_inputs_only(simple, operator):
    parent = battle(simple, operator)
    child = parent.clone()
    assert child.stable_hash() == parent.stable_hash()
    assert child.stage is parent.stage
    assert child.squad is not parent.squad
    assert child.squad["op"] is parent.squad["op"]
    assert child.operators["op"].data is parent.operators["op"].data
    assert child.enemies["enemy_0000"].data is parent.enemies["enemy_0000"].data
    for owner in [parent.stage, parent.stage.map, parent.squad["op"], operator.skill]:
        assert is_dataclass(owner) and owner.__dataclass_params__.frozen
        with pytest.raises(FrozenInstanceError):
            owner.unexpected_mutation = 1
    assert type(child.stage.map.rows) is tuple
    assert type(child.squad["op"].attack_range) is tuple


def test_clone_isolates_every_mutable_battle_component(simple, operator):
    parent = battle(simple, operator)
    parent.extension = {"nested": [1]}
    before = parent.snapshot()
    child = parent.clone()
    assert child.clock is not parent.clock
    assert child.queue is not parent.queue
    assert child.queue.heap is not parent.queue.heap
    assert child.rng is not parent.rng
    assert child.operators is not parent.operators
    assert child.enemies is not parent.enemies
    for key in ["op"]:
        assert child.operators[key] is not parent.operators[key]
        assert child.operators[key].skill is not parent.operators[key].skill
        assert child.operators[key].modifiers is not parent.operators[key].modifiers
        assert child.operators[key].blocked_enemies is not parent.operators[key].blocked_enemies
    assert child.enemies["enemy_0000"] is not parent.enemies["enemy_0000"]
    child.operators["op"].hp -= 1
    child.operators["op"].position = (7, 0)
    child.operators["op"].blocked_enemies.clear()
    child.operators["op"].modifiers.append(Modifier("child", "op", "ATK_ADD", 1, 2, 9))
    child.operators["op"].skill.sp = 99
    child.operators["op"].skill.active_until = 99
    child.enemies["enemy_0000"].blocked_by = None
    child.enemies["enemy_0000"].hp = 1
    child.enemies["enemy_0000"].position = (6, 0)
    child.queue.pop()
    child.queue.push(4, "MODIFIER_BOUNDARY", "child")
    child.clock.time = 3
    child.clock.tick += 1
    child.rng.random()
    child.deploy_counts["op"] += 1
    child.redeploy_at["op"] = 300
    child.dp = 0
    child.life = 1
    child.killed += 1
    child.escaped += 1
    child.spawned += 1
    child.extension["nested"].append(2)
    del child.squad["op"]
    assert parent.snapshot() == before


def test_snapshot_returns_isolated_mutable_output_even_for_shared_inputs(simple, operator):
    state = battle(simple, operator)
    before = state.stable_hash()
    snapshot = state.snapshot()
    snapshot["stage"]["map"]["rows"][0][0]["key"] = "changed"
    snapshot["stage"]["waves"][0]["spawns"].clear()
    snapshot["squad"]["op"]["attack_range"][0][0] = 100
    snapshot["operators"]["op"]["skill"]["sp"] = 100
    snapshot["operators"]["op"]["blocked_enemies"].clear()
    snapshot["operators"]["op"]["modifiers"][0]["value"] = 999
    snapshot["queue"].clear()
    snapshot["rng"].clear()
    assert state.stable_hash() == before


@pytest.mark.parametrize("mutable_location", ["stage", "operator"])
def test_frozen_dataclass_containing_mutable_data_is_not_shared(simple, operator, mutable_location):
    # Python annotations are not runtime validators. A frozen dataclass alone
    # does not make this list immutable, so fallback deepcopy must protect it.
    if mutable_location == "stage":
        simple = replace(simple, assumptions=["mutable"])
    else:
        operator = replace(operator, attack_range=[(0, 0), (1, 0)])
    parent = GameState(simple, [operator], seed=12345, dt=1 / 60)
    child = parent.clone()
    if mutable_location == "stage":
        assert child.stage is not parent.stage
        child.stage.assumptions.append("child")
        assert parent.stage.assumptions == ["mutable"]
    else:
        assert child.squad["op"] is not parent.squad["op"]
        child.squad["op"].attack_range.append((2, 0))
        assert parent.squad["op"].attack_range == [(0, 0), (1, 0)]


def test_deepcopy_preserves_mutable_aliases_and_cycles(simple, operator):
    parent = Simulator(simple, [operator]).state
    values = [1]
    parent.extension = {"a": values, "b": values, "self": parent}
    child = copy.deepcopy(parent)
    assert child.extension["a"] is child.extension["b"]
    assert child.extension["a"] is not values
    assert child.extension["self"] is child


def test_deepcopy_preserves_setup_alias_in_enclosing_object_graph(simple, operator):
    state = GameState(simple, [operator], seed=12345, dt=1 / 60)
    copied_operator, copied_state = copy.deepcopy([operator, state])
    assert copied_state.squad["op"] is copied_operator
    assert copied_operator == operator
    assert copied_state.squad is not state.squad
    assert copied_state.stable_hash() == state.stable_hash()


def test_canonical_field_traversal_preserves_previous_snapshot_contract(simple, operator):
    state = battle(simple, operator)

    def old_canonical(value):
        if is_dataclass(value):
            return old_canonical(asdict(value))
        if isinstance(value, dict):
            return {str(k): old_canonical(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [old_canonical(v) for v in value]
        return value

    expected = old_canonical({
        **{k: v for k, v in state.__dict__.items() if k not in ["rng", "queue"]},
        "rng": state.rng.getstate(),
        "queue": sorted(state.queue.heap),
        "event_sequence": state.queue.sequence,
    })
    assert state.snapshot() == expected


def test_cloning_after_input_replacement_uses_new_input(simple, operator):
    parent = Simulator(simple, [operator]).state
    parent.clone()  # Populate the immutable identity cache.
    parent.stage = replace(parent.stage, initial_cost=13)
    parent.squad["op"] = replace(operator, hp=1234)
    child = parent.clone()
    assert child.stage.initial_cost == 13 and child.stage is parent.stage
    assert child.squad["op"].hp == 1234
    assert child.stable_hash() == parent.stable_hash()
