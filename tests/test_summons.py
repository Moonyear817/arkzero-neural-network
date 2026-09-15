"""Summon runtime contracts; these do not certify unverified game rules."""
from dataclasses import replace
from pathlib import Path

import pytest

from arknights_sim import Simulator
from arknights_sim.data.models import MapData, SpawnData, WaveData
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from arknights_sim.entities.unit import Unit
from arknights_sim.environment import ArknightsEnv, Action
from arknights_sim.environment.env import EnvState
from arknights_sim.skills.modifier import modified
from arknights_sim.summons.runtime import available_cards

DATA = Path(__file__).resolve().parents[1] / "data/real"


@pytest.fixture(scope="module")
def loader():
    return OperatorLoader(DATA / "character_table.json", DATA / "range_table.json",
                          SkillLoader(DATA / "skill_table.json"))


@pytest.fixture
def arena(simple):
    # Future wave prevents an empty unit-test arena from immediately ending.
    return replace(simple,
                   map=MapData((tuple(replace(t, buildable="ALL") for t in simple.map.rows[0]),)),
                   waves=(WaveData(0, 0, (0,), (SpawnData(1000, "e", 0),)),))


@pytest.fixture
def sim(loader, arena):
    return Simulator(arena, [loader.load("深海色")], trace=True)


def setup_summon(sim, tile=(2, 0)):
    owner = next(iter(sim.state.squad))
    sim.deploy(owner, (7, 0))
    card = next(iter(sim.state.summon_cards))
    return sim.state.operators[owner], sim.deploy_summon(card, tile)


@pytest.mark.parametrize("elite,level,hp,atk,defense,count", [
    (0, 1, 975, 218, 128, 2), (0, 45, 1170, 295, 220, 2),
    (1, 1, 1275, 308, 236, 3), (1, 60, 1480, 348, 275, 3),
    (2, 1, 1605, 393, 280, 4), (2, 70, 2016, 462, 335, 4),
])
def test_real_growth_and_no_owner_potential_stat_leak(loader, elite, level, hp, atk, defense, count):
    owner = loader.load("深海色", elite=elite, level=level)
    full = loader.load("深海色", elite=elite, level=level, potential=5)
    card = owner.summons[0]
    assert (card.unit.hp, card.unit.atk, card.unit.defense) == (hp, atk, defense)
    assert (card.inventory, card.max_active, card.slot_weight) == (count, count, 1)
    assert (card.unit.cost, card.unit.redeploy, card.unit.interval, card.unit.block_count) == (5, 10, 1.25, 1)
    assert card == full.summons[0]
    assert card.unit.skill is None and not card.unit.healable


def test_other_summoner_not_enabled_by_shared_token_fields(loader):
    assert not loader.load("令", elite=2, skill_index=2).summons
    assert not loader.load("麦哲伦", elite=2, skill_index=2).summons
    with pytest.raises(NotImplementedError):
        loader.skills.load("skchr_deepcl_2")


def test_summon_action_hash_roundtrip():
    action = Action("DEPLOY_SUMMON", "owner::token", (3, 4), "LEFT")
    assert Action.from_dict(action.to_dict()) == action
    assert len({action, Action.from_dict(action.to_dict())}) == 1
    assert "LEFT" in str(action)
    with pytest.raises(ValueError):
        Action("DEPLOY_SUMMON", "owner::token", (3, 4))
    with pytest.raises(ValueError):
        Action("RETREAT_SUMMON", "summon_000001", (3, 4))


def test_owner_must_be_present_and_invalid_action_is_atomic(sim):
    card = next(iter(sim.state.summon_cards))
    before = sim.stable_hash()
    with pytest.raises(ValueError):
        sim.deploy_summon(card, (1, 0))
    assert sim.stable_hash() == before and not available_cards(sim.state)
    owner, token = setup_summon(sim)
    sim.state.summon_ready_at[card] = 0
    for tile, direction in [((7, 0), 0), ((2, 0), 0), ((-1, 0), 0), ((1, 0), 4), ((1.5, 0), 0)]:
        before = sim.stable_hash()
        with pytest.raises(ValueError):
            sim.deploy_summon(card, tile, direction)
        assert sim.stable_hash() == before


def test_shared_cooldown_fixed_cost_inventory_and_unique_ids(sim):
    owner, first = setup_summon(sim)
    card = first.summon_card_id
    assert sim.state.summon_inventory[card] == 1 and not available_cards(sim.state)
    sim.run_until(10 - 1 / 60)
    assert not available_cards(sim.state)
    sim.run_until(10)
    before = sim.state.dp
    second = sim.deploy_summon(card, (3, 0))
    assert before - sim.state.dp == 5
    assert first.id != second.id and first.owner_id == second.owner_id
    assert first.id not in sim.state.squad and first.id not in sim.state.operators
    assert sim.state.summon_inventory[card] == 0
    sim.run_until(20)
    assert not available_cards(sim.state)


def test_slots_and_bidirectional_occupancy(loader, arena):
    owner = loader.load("深海色")
    other = replace(loader.load("黑角"), cost=0)
    sim = Simulator(replace(arena, character_limit=2), [owner, other])
    _, token = setup_summon(sim)
    assert sim.state.occupied_deployment_slots == 2
    with pytest.raises(ValueError):sim.deploy(other.id, (3, 0))
    sim.state.stage = replace(arena, character_limit=8)
    with pytest.raises(ValueError):sim.deploy(other.id, token.position)
    env = ArknightsEnv()
    legal = env.legal_actions(EnvState(sim.state))
    assert not any(a.type == "DEPLOY" and a.tile == token.position for a in legal)
    sim.retreat_summon(token.id)
    sim.deploy(other.id, token.position)


def test_legal_summon_dp_cooldown_tile_occupancy_and_slots(loader, arena):
    ground = replace(arena.map.rows[0][0], buildable="MELEE")
    high = replace(ground, buildable="RANGED")
    no = replace(ground, buildable="NONE")
    stage = replace(arena, map=MapData(((no, high, ground, ground, ground, ground, ground, high),)))
    env = ArknightsEnv(stage=stage, squad=[loader.load("深海色")])
    state = env.reset()
    assert not any(a.type == "DEPLOY_SUMMON" for a in env.legal_actions(state))
    state = env.step(state, Action("DEPLOY", "char_110_deepcl", (7, 0), "LEFT"))
    card = next(iter(state.game.summon_cards))
    actions = [a for a in env.legal_actions(state) if a.type == "DEPLOY_SUMMON"]
    assert len(actions) == 5 * 4
    assert {a.tile for a in actions} == {(i, 0) for i in range(2, 7)}
    for action in actions:
        child = env.step(state, action)
        assert len(child.game.summons) == 1
    state.game.dp = 4.99
    assert not any(a.type == "DEPLOY_SUMMON" for a in env.legal_actions(state))
    state.game.dp = 5
    assert Action("DEPLOY_SUMMON", card, (2, 0), "LEFT") in env.legal_actions(state)
    child = env.step(state, Action("DEPLOY_SUMMON", card, (2, 0), "LEFT"))
    assert not any(a.type == "DEPLOY_SUMMON" for a in env.legal_actions(child))
    child.game.stage = replace(child.game.stage, character_limit=1)
    child.game.summon_ready_at[card] = 0
    child.game.dp = 99
    assert not any(a.type == "DEPLOY_SUMMON" for a in env.legal_actions(child))


def test_wait_stops_at_summon_dp_and_cooldown(sim):
    owner, token = setup_summon(sim)
    env = ArknightsEnv(horizon=120)
    sim.state.dp = 99
    state = EnvState(sim.state, horizon=120)
    child = env.step(state, Action("WAIT"))
    assert child.game.current_time == pytest.approx(10)
    assert "SUMMON_DEPLOY_AVAILABLE" in child.last_decision.reasons
    state.game.summon_ready_at[token.summon_card_id] = 0
    state.game.dp = 2
    child = env.step(state, Action("WAIT"))
    assert child.game.current_time == pytest.approx(3)
    assert "SUMMON_DEPLOY_AVAILABLE" in child.last_decision.reasons


def test_retreat_consumes_card_without_refund_or_operator_cooldown(sim):
    _, token = setup_summon(sim)
    before = sim.state.dp
    sim.retreat_summon(token.id)
    assert not token.alive and sim.state.dp == before
    assert sim.state.summon_inventory[token.summon_card_id] == 1
    assert sim.state.summon_ready_at[token.summon_card_id] == 10
    assert token.id not in sim.state.redeploy_at
    with pytest.raises(ValueError):sim.retreat_summon(token.id)


def test_summon_blocks_attacks_takes_damage_and_releases_on_death(sim):
    _, token = setup_summon(sim)
    data = replace(sim.state.stage.enemies[0], hp=10000, atk=400, speed=0)
    enemy = Unit("probe", data, data.hp, token.position)
    sim.state.enemies[enemy.id] = enemy
    sim.run_until(.25)
    assert token.alive and enemy.blocked_by == token.id
    # First exchange proves the summon can attack. A later fatal hit proves
    # cleanup, without requiring an already-dead unit's queued hit to land.
    enemy.data = replace(enemy.data, atk=2000)
    sim.run_until(1.5)
    assert not token.alive and not enemy.blocked_by and token.blocked_enemies == []
    assert sim.state.killed == 0 and sim.state.escaped == 0
    assert any(r["event"] == "BLOCK" and r["operator"] == token.id for r in sim.trace.events)
    assert any(r["event"] == "DAMAGE" and r["attacker"] == token.id for r in sim.trace.events)
    assert any(r["event"] == "DAMAGE" and r["target"] == token.id for r in sim.trace.events)


def test_fatal_same_frame_hit_cancels_dead_summon_attack(sim):
    _, token = setup_summon(sim)
    data = replace(sim.state.stage.enemies[0], hp=10000, atk=2000, speed=0)
    enemy = Unit("probe", data, data.hp, token.position)
    sim.state.enemies[enemy.id] = enemy
    sim.run_until(.25)
    assert not token.alive
    assert not any(r['event'] == 'DAMAGE' and r['attacker'] == token.id for r in sim.trace.events)


def test_summon_kill_credits_enemy_and_releases_block(sim):
    _, token = setup_summon(sim)
    data = replace(sim.state.stage.enemies[0], atk=0, speed=0)
    enemy = Unit("probe", data, data.hp, token.position)
    sim.state.enemies[enemy.id] = enemy
    sim.run_until(.25)
    assert sim.state.killed == 1 and not enemy.alive and token.blocked_enemies == []


@pytest.mark.parametrize("death", [False, True])
def test_owner_removal_cleans_tokens_and_redeployment_replenishes(sim, death):
    owner, token = setup_summon(sim)
    data = replace(sim.state.stage.enemies[0], hp=10000, atk=0, speed=0)
    enemy = Unit("probe", data, data.hp, token.position)
    sim.state.enemies[enemy.id] = enemy
    sim.run_until(1 / 60)
    assert enemy.blocked_by == token.id
    if death:
        killer = Unit("killer", replace(data, atk=10000), data.hp, owner.position)
        sim.state.enemies[killer.id] = killer
        sim._attack(killer, owner)
        sim.run_until(.25)
    else:
        sim.retreat(owner.id)
    assert not owner.alive and not token.alive and enemy.blocked_by is None
    assert not available_cards(sim.state)
    sim.run_until(sim.state.redeploy_at[owner.id])
    sim.deploy(owner.id, (7, 0))
    assert sim.state.summon_inventory[token.summon_card_id] == 2
    assert token.summon_card_id in available_cards(sim.state)


def test_medic_cannot_heal_tentacle_but_regeneration_can(sim):
    from arknights_sim.combat.healing import heal
    from arknights_sim.combat.targeting import HealingTargetSelector
    owner, token = setup_summon(sim)
    token.hp = 100
    healer = Unit("medic", replace(owner.data, attack_range=((0, 0), (1, 0))), 100, (1, 0))
    assert HealingTargetSelector.select(healer, [token]) is None
    heal(token, 100)
    assert token.hp == 100
    owner.skill.sp = owner.data.skill.cost
    sim.activate_skill(owner.id)
    sim.run_until(1)
    assert token.hp == pytest.approx(130)
    assert any(r["event"] == "HP_RECOVERY" for r in sim.trace.events)
    assert not any(r["event"] == "HEAL" for r in sim.trace.events)


def test_s1_owner_link_new_tokens_expiry_and_damage(sim):
    owner, first = setup_summon(sim)
    owner.skill.sp = owner.data.skill.cost
    sim.activate_skill(owner.id)
    assert modified(owner.data.atk, "ATK", owner.modifiers, 0) == owner.data.atk
    assert modified(first.data.atk, "ATK", first.modifiers, 0) == pytest.approx(218 * 1.15)
    assert modified(first.data.defense, "DEF", first.modifiers, 0) == pytest.approx(128 * 1.15)
    sim.run_until(10)
    second = sim.deploy_summon(first.summon_card_id, (3, 0))
    assert modified(second.data.atk, "ATK", second.modifiers, 10) == pytest.approx(218 * 1.15)
    data = replace(sim.state.stage.enemies[0], hp=100000, atk=0, speed=0)
    enemy = Unit("probe", data, data.hp, first.position)
    sim.state.enemies[enemy.id] = enemy
    sim.run_until(10.25)
    hits = [r for r in sim.trace.events if r["event"] == "DAMAGE" and r["attacker"] == first.id]
    assert hits[0]["amount"] == pytest.approx(218 * 1.15)
    sim.run_until(30)
    for unit in (first, second):
        assert modified(unit.data.atk, "ATK", unit.modifiers, 30) == 218
        unit.hp = 100
    sim.run_until(31)
    assert first.hp == 100 and second.hp == 100
    assert owner.skill.sp == pytest.approx(1)


def test_all_ten_s1_ranks_read_real_recovery(loader):
    for rank in range(1, 11):
        skill = loader.skills.load("skchr_deepcl_1", rank)
        raw = loader.skills.data[skill.id]["levels"][rank - 1]
        bb = {b["key"]: b["value"] for b in raw["blackboard"]}
        assert skill.effect_target == "OWNER_SUMMONS"
        assert skill.hp_recovery_per_sec == bb["hp_recovery_per_sec"]
        assert skill.cost == raw["spData"]["spCost"]


def test_clone_isolates_all_summon_state_and_queued_effects(sim):
    owner, token = setup_summon(sim)
    owner.skill.sp = owner.data.skill.cost
    sim.activate_skill(owner.id)
    before = sim.stable_hash()
    child = Simulator.from_state(sim.state.clone(), trace=True)
    assert child.stable_hash() == before
    assert child.state.summon_cards[token.summon_card_id] is sim.state.summon_cards[token.summon_card_id]
    child.state.summons[token.id].hp = 1
    child.state.summons[token.id].modifiers.clear()
    child.state.summon_inventory[token.summon_card_id] = 100
    child.state.summon_ready_at[token.summon_card_id] = 0
    child.state.rng.random()
    child.run_until(1)
    child.retreat_summon(token.id)
    assert sim.stable_hash() == before and token.alive


def test_snapshot_uses_summon_kind_and_action_labels(sim):
    from desktop.models.simulation_state import snapshot_from_state, event_from_record
    _, token = setup_summon(sim)
    snapshot = snapshot_from_state(ArknightsEnv(), EnvState(sim.state))
    assert next(u for u in snapshot.units if u.id == token.id).kind == "summon"
    assert any("触手" in a.label for a in snapshot.legal_actions)
    assert event_from_record(next(r for r in sim.trace.events if r["event"] == "SUMMON_DEPLOY")).category == "Deploy"


def test_neural_encoder_represents_summons_separately(sim):
    from network import StateEncoder, ActionEncoder
    import torch
    _, token = setup_summon(sim)
    state = EnvState(sim.state)
    encoded = StateEncoder()(state)
    assert encoded['summon_mask'].sum() >= 2  # card and alive instance
    assert encoded['operator_mask'].sum() == 1
    assert torch.isfinite(encoded['summons']).all()
    actions = ArknightsEnv().legal_actions(state)
    assert any(a.type == 'RETREAT_SUMMON' and a.operator_id == token.id for a in actions)
    assert torch.isfinite(ActionEncoder()(state, actions)).all()
