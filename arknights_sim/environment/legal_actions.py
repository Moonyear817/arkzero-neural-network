"""Complete legal action enumeration for supported core mechanics, no pruning."""

from .action import Action, ActionType, Direction, WAIT
from arknights_sim.deployment.cost import deployment_cost


def skill_ready(op, time):
    skill = op.data.skill
    return bool(
        op.alive
        and skill
        and skill.trigger == "MANUAL_TRIGGER"
        and (not op.skill.is_active(skill,time) or skill.mode=='TOGGLE')
        and op.skill.sp + 1e-8 >= skill.cost
    )


def deployable_operators(game):
    if game.occupied_deployment_slots >= game.stage.character_limit:
        return ()
    return tuple(
        key
        for key, data in sorted(game.squad.items())
        if not (key in game.operators and game.operators[key].alive)
        and game.current_time + 1e-9 >= game.redeploy_at.get(key, 0)
        and game.dp + 1e-9 >= deployment_cost(data.cost, game.deploy_counts.get(key, 0))
    )


def legal_actions(state):
    if state.terminal:
        return ()
    game = state.game
    result = [WAIT]
    active = [
        op for op in sorted(game.operators.values(), key=lambda o: o.id) if op.alive
    ]
    occupied = {op.position for op in game.allies.values() if op.alive}
    from arknights_sim.mechanics.runtime import solid_positions,can_place
    occupied |= solid_positions(game)
    for op in active:
        if skill_ready(op, game.current_time):
            result.append(Action(ActionType.ACTIVATE_SKILL, op.id))
    for key in deployable_operators(game):
        data = game.squad[key]
        for y, row in enumerate(game.stage.map.rows):
            for x, tile in enumerate(row):
                if (x, y) not in occupied and tile.buildable in (
                    data.position_type,
                    "ALL",
                ):
                    result.extend(
                        Action(ActionType.DEPLOY, key, (x, y), direction)
                        for direction in Direction
                    )
    result.extend(Action(ActionType.RETREAT, op.id) for op in active)
    from arknights_sim.summons.runtime import available_cards, valid_tile
    for key in available_cards(game):
        card = game.summon_cards[key]
        for y, row in enumerate(game.stage.map.rows):
            for x, tile in enumerate(row):
                if valid_tile(game, card, (x, y)):
                    result.extend(Action(ActionType.DEPLOY_SUMMON, key, (x, y), direction)
                                  for direction in Direction)
    for unit in sorted(game.summons.values(), key=lambda u: u.id):
        if unit.alive:
            result.append(Action(ActionType.RETREAT_SUMMON, unit.id))
            if skill_ready(unit, game.current_time):
                result.append(Action(ActionType.ACTIVATE_SKILL, unit.id))
    for key,d in sorted(game.devices.items()):
        if not d.alive:continue
        if d.data.kind=='sensor' and d.sp+1e-8>=d.data.charge and game.current_time>=d.active_until:
            result.append(Action(ActionType.ACTIVATE_DEVICE,key))
        if d.data.kind=='crate':result.append(Action(ActionType.REMOVE_DEVICE,key))
    for data in game.stage.devices:
        if data.position is None and game.device_inventory[data.id]>0 and game.dp>=data.cost and game.current_time>=game.device_ready[data.id]:
            for y,row in enumerate(game.stage.map.rows):
                for x,tile in enumerate(row):
                    if can_place(game,(x,y)):result.append(Action(ActionType.PLACE_DEVICE,data.id,(x,y)))
    # With no alternatives, next-event WAIT already advances to the first useful
    # boundary. Timed holds would only duplicate that forced policy target.
    # With a real choice, holds allow deliberately skipping intermediate events
    # while DP, SP, cooldowns or enemy positions improve. They never stop attacks.
    if len(result) > 1 and (active or game.summons or game.dp < game.stage.max_cost
                           or any(t > game.current_time for t in game.redeploy_at.values())):
        result.extend(Action(ActionType.WAIT, wait_seconds=seconds)
                      for seconds in (0.5, 1, 2, 5))
    return tuple(result)
