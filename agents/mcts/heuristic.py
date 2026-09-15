"""General stage-independent evaluation. No operator identities or solutions."""


def evaluate_state(env, state):
    """Terminal rewards are exact; unfinished battles use bounded progress.

    Strict success means eliminating every enemy without a leak. A leak can
    never be repaired, so it has the failure value even before battle end.
    The heuristic is deliberately below +1 until success is established.
    """
    if env.is_terminal(state):
        return 1.0 if env.result(state).success else -1.0
    game = state.game
    if game.escaped:
        return -1.0
    total = max(1, len(game.stage.spawns))
    kill_progress = game.killed / total
    life_fraction = game.life / max(1, game.stage.life)
    living = [unit for unit in game.operators.values() if unit.alive]
    health = sum(unit.hp / max(1, unit.data.hp) for unit in living)
    health /= max(1, len(game.squad))
    # Empty defenses receive no health bonus. Kills dominate transient health.
    value = -0.5 + 1.25 * kill_progress + 0.1 * life_fraction + 0.1 * health
    return max(-1.0, min(0.95, value))


heuristic = evaluate_state
