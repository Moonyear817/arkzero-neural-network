"""Seeded rollout policies, kept separate from the UCT tree policy.

The optional tactical policy uses only public map routes and unit attributes.
It never drops legal actions: every candidate has positive sampling weight.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

from arknights_sim.map.route import compile_route

from .heuristic import evaluate_state


def action_kind(action):
    kind = action.type
    return kind.value if hasattr(kind, "value") else str(kind)


class RandomRolloutPolicy:
    def select_action(self, env, state, actions, rng):
        return rng.choice(actions)

    def order_actions(self, env, state, actions, rng):
        ordered = list(actions)
        rng.shuffle(ordered)
        return ordered


@lru_cache(maxsize=32)
def _route_cells(stage):
    """Return weighted traversed tiles, including all routes used by spawns."""
    counts = {}
    for spawn in stage.spawns:
        counts[spawn.route_index] = counts.get(spawn.route_index, 0) + 1
    total = max(1, len(stage.spawns))
    cells = {}
    for index, count in counts.items():
        route = stage.routes[index]
        path = compile_route(stage.map, route)
        positions = {route.start, *(node.position for node in path)}
        for position in positions:
            cells[position] = cells.get(position, 0.0) + count / total
    return tuple(cells.items())


def _turns(direction):
    value = direction.value if hasattr(direction, "value") else direction
    if isinstance(value, str):
        return {"RIGHT": 0, "UP": 1, "LEFT": 2, "DOWN": 3}[value]
    return int(value)


def _range(tile, offsets, direction):
    result = set()
    for x, y in offsets:
        for _ in range(_turns(direction)):
            x, y = -y, x
        result.add((tile[0] + x, tile[1] + y))
    return result


class TacticalRolloutPolicy(RandomRolloutPolicy):
    """Lightweight stochastic defense construction, applicable to any stage.

    Deployment weights prefer route coverage, skill use, and keeping deployed
    units fighting. WAIT and RETREAT stay possible. This is rollout guidance,
    not an action-space mask, neural prior, or a stage-specific deployment plan.
    """

    def weights(self, env, state, actions):
        game = state.game
        routes = dict(_route_cells(game.stage))
        deployed = [unit for unit in game.operators.values() if unit.alive]
        occupied_coverage = set()
        for unit in deployed:
            occupied_coverage.update(
                _range(unit.position, unit.data.attack_range, unit.direction)
            )
        deployments = []
        values = []
        for action in actions:
            kind = action_kind(action)
            if kind == "DEPLOY":
                data = game.squad[action.operator_id]
                coverage = _range(action.tile, data.attack_range, action.direction)
                route_cover = sum(routes.get(cell, 0.0) for cell in coverage)
                intercept = routes.get(action.tile, 0.0) * min(3, data.block_count)
                support = sum(cell in occupied_coverage for cell in coverage)
                value = 0.015 + 2.0 * intercept + route_cover + 0.3 * support
                # Quadratic weighting makes useful route interception likely,
                # while keeping every tile and facing in the candidate set.
                value *= value
                deployments.append(len(values))
            elif kind == "ACTIVATE_SKILL":
                value = 4.0
            elif kind == "WAIT":
                value = 1.0
            else:  # Retreat may be useful but random instant churn rarely is.
                value = 0.002
            values.append(value)
        if deployments:
            total = sum(values[index] for index in deployments)
            # Normalize the deployment category: tile count must not dictate
            # how often we deploy versus allowing current defenses to fight.
            for index in deployments:
                values[index] = 5.0 * values[index] / total
        return values

    def select_action(self, env, state, actions, rng):
        return rng.choices(actions, weights=self.weights(env, state, actions), k=1)[0]

    def order_actions(self, env, state, actions, rng):
        # Weighted random permutation; every action is expanded eventually.
        weighted = zip(actions, self.weights(env, state, actions))
        scores = [(-math.log(max(rng.random(), 1e-15)) / weight, action)
                  for action, weight in weighted]
        scores.sort(key=lambda item: item[0], reverse=True)
        return [action for _, action in scores]


@dataclass
class RolloutResult:
    value: float
    state: object
    actions: tuple
    terminal: bool
    success: bool
    legal_counts: tuple[int, ...]
    states: tuple


def rollout(env, state, policy, rng, limit, evaluator=evaluate_state):
    actions_taken = []
    legal_counts = []
    states = []
    for _ in range(limit):
        if env.is_terminal(state):
            break
        actions = env.legal_actions(state)
        legal_counts.append(len(actions))
        if not actions:
            break
        action = policy.select_action(env, state, actions, rng)
        states.append(state)
        state = env.step(state, action)
        actions_taken.append(action)
        # A strict zero-leak solution is now impossible. This is an exact
        # objective bound, not a claim that the simulator battle has ended.
        if getattr(getattr(state, "game", None), "escaped", 0):
            break
    terminal = env.is_terminal(state)
    return RolloutResult(
        evaluator(env, state), state, tuple(actions_taken), terminal,
        terminal and env.result(state).success, tuple(legal_counts), tuple(states),
    )
