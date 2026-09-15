"""Consumable summon cards and independent allied instances.

Deepcolor lifecycle choices still require in-game verification; see
research/mechanics/summons.md. Other token families are not enabled implicitly.
"""
import math
from arknights_sim.entities.unit import Unit
from arknights_sim.skills.skill import SkillState
from arknights_sim.skills.runtime import apply_timed_skill
from arknights_sim.combat.blocking import release
from arknights_sim.mechanics import runtime as mechanics


def owner_deployed(game, owner):
    for card in owner.data.summons:
        if card.reset_on_owner_deploy:
            game.summon_inventory[card.id] = card.inventory
            game.summon_ready_at[card.id] = game.current_time


def available_cards(game):
    if game.done:
        return ()
    result = []
    for key, card in sorted(game.summon_cards.items()):
        owner = game.operators.get(card.owner_id)
        if (owner and owner.alive and game.summon_inventory[key] > 0
                and game.current_time + 1e-9 >= game.summon_ready_at[key]
                and game.dp + 1e-9 >= card.unit.cost
                and game.occupied_deployment_slots + card.slot_weight <= game.stage.character_limit
                and sum(u.alive and u.summon_card_id == key for u in game.summons.values()) < card.max_active):
            result.append(key)
    return tuple(result)


def valid_tile(game, card, tile):
    if len(tile) != 2 or any(type(v) is not int for v in tile):
        return False
    x, y = tile
    if not (0 <= x < game.stage.map.width and 0 <= y < game.stage.map.height):
        return False
    return (game.stage.map.tile(x, y).buildable in (card.unit.position_type, "ALL")
            and tile not in mechanics.solid_positions(game)
            and not any(u.alive and u.position == tile for u in game.allies.values()))


def deploy(sim, card_id, tile, direction=0):
    game = sim.state
    card = game.summon_cards.get(card_id)
    if (card is None or card_id not in available_cards(game)
            or type(direction) is not int or direction not in range(4)
            or not valid_tile(game, card, tuple(tile))):
        raise ValueError("Summon unavailable or illegal deployment")
    game.summon_sequence += 1
    key = f"summon_{game.summon_sequence:06d}"
    data = card.unit
    unit = Unit(key, data, data.hp, tuple(tile), direction=direction,
                deployed_at=game.current_time, paid_cost=data.cost, generation=1,
                skill=SkillState(data.skill.initial_sp if data.skill else 0),
                owner_id=card.owner_id, summon_card_id=card_id)
    game.dp -= data.cost
    game.summon_inventory[card_id] -= 1
    game.summon_ready_at[card_id] = game.current_time + data.redeploy
    game.summons[key] = unit
    owner = game.operators[card.owner_id]
    if (owner.data.skill and owner.data.skill.effect_target == "OWNER_SUMMONS"
            and owner.skill.active_until > game.current_time):
        apply_timed_skill(unit, owner, game.current_time)
    mechanics.gravity_switches(game)
    sim._log("SUMMON_DEPLOY", unit=key, owner=card.owner_id, card=card_id,
             position=tile, direction=direction, cost=data.cost,
             remaining=game.summon_inventory[card_id])
    return unit


def remove(sim, unit, reason):
    game = sim.state
    if not unit.alive:
        return
    unit.alive = False
    for eid in list(unit.blocked_enemies):
        release(game.enemies[eid], game.allies)
    sim._log("SUMMON_REMOVE", unit=unit.id, owner=unit.owner_id, reason=reason)


def retreat(sim, key):
    game = sim.state
    unit = game.summons.get(key)
    if game.done or unit is None or not unit.alive:
        raise ValueError("Summon unavailable")
    card = game.summon_cards[unit.summon_card_id]
    refund = math.floor(unit.paid_cost * card.retreat_refund_ratio)
    game.dp = min(game.stage.max_cost, game.dp + refund)
    if card.return_on_retreat:
        game.summon_inventory[card.id] += 1
    remove(sim, unit, "RETREAT")
    mechanics.gravity_switches(game)
    sim._log("SUMMON_RETREAT", unit=key, refund=refund)


def owner_removed(sim, owner):
    for unit in sim.state.summons.values():
        if (unit.alive and unit.owner_id == owner.id
                and sim.state.summon_cards[unit.summon_card_id].remove_with_owner):
            remove(sim, unit, "OWNER_REMOVED")
