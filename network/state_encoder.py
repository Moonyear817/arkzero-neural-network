"""Stage-agnostic, deterministic CPU feature encoding of the supported core.

Coordinates preserve the simulator's bottom-left origin. Feature scaling uses
physical constants, never stage IDs or a strategy. Masks distinguish padding
from real zero-valued entities. Operator metadata carries profession; feature slot 25 distinguishes healing
(+1), passive support (-1), and damage (0) without changing weight dimensions.
"""

import math
from functools import lru_cache

import torch

from arknights_sim.deployment.cost import deployment_cost
from arknights_sim.environment.legal_actions import skill_ready
from arknights_sim.map.route import compile_route
from arknights_sim.skills.modifier import modified

MAP_CHANNELS = 20
OPERATOR_FEATURES = 48
ENEMY_FEATURES = 40
GLOBAL_FEATURES = 24
FUTURE_FEATURES = 12
FUTURE_WINDOWS = (5.0, 10.0, 20.0)
MAP_FEATURE_NAMES = (
    "lowland",
    "highland",
    "melee_tile",
    "ranged_tile",
    "all_tile",
    "nondeployable",
    "spawn",
    "goal",
    "route_density",
    "walkable",
    "operator_count",
    "operator_hp",
    "operator_atk",
    "operator_block",
    "enemy_count",
    "enemy_hp",
    "blocked_enemies",
    "ready_skills",
    "x",
    "y",
)
OPERATOR_FEATURE_NAMES = (
    "deployed",
    "alive",
    "current_or_redeploy_hp_ratio",
    "max_hp",
    "atk",
    "defense",
    "resistance",
    "interval",
    "block_capacity",
    "base_dp",
    "current_dp",
    "redeploy_remaining",
    "current_or_redeploy_sp_ratio",
    "skill_ready",
    "skill_active",
    "skill_remaining",
    "x",
    "y",
    "right",
    "up",
    "left",
    "down",
    "melee",
    "ranged",
    "all_position",
    "support_role",
    "physical",
    "arts",
    "true_damage",
    "range_size",
    "range_min_x",
    "range_max_x",
    "range_min_y",
    "range_max_y",
    "redeploy_duration",
    "deploy_count",
    "attack_cooldown",
    "blocked_count",
    "has_skill",
    "skill_cost",
    "skill_duration",
    "skill_atk_multiplier",
    "skill_def_multiplier",
    "skill_as_multiplier",
    "manual_skill",
    "auto_sp",
    "deployed_duration",
    "paid_dp",
)
ENEMY_FEATURE_NAMES = (
    "hp_ratio",
    "max_hp",
    "atk",
    "defense",
    "resistance",
    "speed",
    "interval",
    "x",
    "y",
    "route_index",
    "route_progress",
    "remaining_distance",
    "wait_remaining",
    "blocked",
    "attack_cooldown",
    "block_weight",
    "life_cost",
    "physical",
    "arts",
    "true_damage",
    "goal_x",
    "goal_y",
    "next_x",
    "next_y",
    "blocker_x",
    "blocker_y",
    "blocker_hp",
    "moving",
    "waiting",
    "alive",
    "route_nodes_remaining",
    "modifier_count",
    "weight", "invisible", "revealed", "attack_range", "splash", "gravity_speed", "direction_x", "direction_y",
)
GLOBAL_FEATURE_NAMES = (
    "time",
    "dp",
    "life_ratio",
    "kills_ratio",
    "leaks_ratio",
    "spawned_ratio",
    "total_enemies",
    "active_enemies",
    "deployed_operators",
    "squad_size",
    "available_dp_ratio",
    "max_dp",
    "dp_recovery",
    "life",
    "character_limit",
    "decision_progress",
    "remaining_time",
    "width",
    "height",
    "pending_spawns",
    "next_spawn_delay",
    "pending_hits",
    "terminal",
    "move_multiplier",
)
FUTURE_FEATURE_NAMES = (
    "spawn_count",
    "total_hp",
    "total_atk",
    "mean_defense",
    "mean_speed",
    "mean_interval",
    "earliest_delay",
    "mean_delay",
    "distinct_routes",
    "mean_spawn_x",
    "mean_spawn_y",
    "total_life_cost",
)


def _xy(position, stage):
    return (
        position[0] / max(1, stage.map.width - 1),
        position[1] / max(1, stage.map.height - 1),
    )


def _damage(kind):
    return [float(kind == value) for value in ("PHYSICAL", "ARTS", "TRUE")]


@lru_cache(maxsize=32)
def _paths(stage):
    # Inputs and cached results are immutable. No state or mutable tensor cached.
    used = {spawn.route_index for spawn in stage.spawns}
    return tuple(
        compile_route(stage.map, route) if i in used else ()
        for i, route in enumerate(stage.routes)
    )


def operator_features(game, key):
    card = game.summon_cards.get(key)
    u = game.allies.get(key)
    d = game.squad[key] if key in game.squad else (card.unit if card else u.data)
    live = bool(u and u.alive)
    now = game.current_time
    skill = d.skill
    sp = u.skill.sp if live else (skill.initial_sp if skill else 0)
    active = bool(live and u.skill.is_active(u.data.skill,now))
    modifiers = u.modifiers if live else ()
    atk = modified(d.atk, "ATK", modifiers, now)
    defense = modified(d.defense, "DEF", modifiers, now)
    aspd = modified(100, "ASPD", modifiers, now)
    block = modified(d.block_count, "BLOCK_COUNT", modifiers, now)
    xs = [p[0] for p in d.attack_range] or [0]
    ys = [p[1] for p in d.attack_range] or [0]
    x, y = _xy(u.position, game.stage) if live else (0, 0)
    v = [
        float(live),
        float(live),
        u.hp / max(d.hp, 1) if live else 1.0,
        d.hp / 10000,
        atk / 2000,
        defense / 1000,
        d.resistance / 100,
        d.interval * 100 / max(aspd, 1e-6) / 10,
        block / 5,
        d.cost / 100,
        (d.cost if card or key in game.summons else deployment_cost(d.cost, game.deploy_counts.get(key, 0))) / 100,
        max(0, (game.summon_ready_at.get(key, 0) if card else game.redeploy_at.get(key, 0)) - now) / 120,
        sp / max(skill.cost, 1) if skill else 0,
        float(skill_ready(u, now)) if live else 0,
        float(active),
        max(0, u.skill.active_until - now) / 120 if live else 0,
        x,
        y,
        *[float(live and u.direction == i) for i in range(4)],
        *[float(d.position_type == p) for p in ("MELEE", "RANGED", "ALL")],
        1.0 if d.damage_type == "HEAL" else (-1.0 if d.damage_type == "NONE" else 0.0),
        *_damage(d.damage_type),
        len(d.attack_range) / 25,
        min(xs) / 10,
        max(xs) / 10,
        min(ys) / 10,
        max(ys) / 10,
        d.redeploy / 120,
        game.deploy_counts.get(key, 0) / 10,
        max(0, u.ready_at - now) / 10 if live else 0,
        len(u.blocked_enemies) / 5 if live else 0,
        float(skill is not None),
        skill.cost / 100 if skill else 0,
        skill.duration / 120 if skill else 0,
        skill.attack_multiplier / 5 if skill else 0,
        skill.defense_multiplier / 5 if skill else 0,
        skill.attack_speed_multiplier / 5 if skill else 0,
        float(skill is not None and skill.trigger == "MANUAL_TRIGGER"),
        float(skill is not None and skill.recovery == "AUTO_RECOVERY"),
        max(0, now - u.deployed_at) / 300 if live else 0,
        u.paid_cost / 100 if live else 0,
    ]
    assert len(v) == OPERATOR_FEATURES
    return v


def _enemy_features(game, enemy, paths):
    d, now, stage = enemy.data, game.current_time, game.stage
    from arknights_sim.mechanics.runtime import unit_path
    path = unit_path(game,enemy,paths)
    route = stage.routes[enemy.route_index]
    remaining = 0.0
    pos = enemy.position
    for waypoint in path[enemy.node :]:
        if waypoint.kind != "WAIT_FOR_SECONDS":
            remaining += math.dist(pos, waypoint.position)
            pos = waypoint.position
    total_distance = 0.0
    previous = route.start
    for waypoint in path:
        if waypoint.kind != "WAIT_FOR_SECONDS":
            total_distance += math.dist(previous, waypoint.position)
            previous = waypoint.position
    progress = max(0.0, min(1.0, 1 - remaining / max(total_distance, 1e-6)))
    blocker = game.allies.get(enemy.blocked_by)
    next_pos = path[enemy.node].position if enemy.node < len(path) else route.end
    v = [
        enemy.hp / max(d.hp, 1),
        d.hp / 10000,
        modified(d.atk, "ATK", enemy.modifiers, now) / 2000,
        modified(d.defense, "DEF", enemy.modifiers, now) / 1000,
        d.resistance / 100,
        modified(d.speed, "MOVE_SPEED", enemy.modifiers, now)
        * stage.move_multiplier
        / 5,
        d.interval * 100 / max(modified(100, "ASPD", enemy.modifiers, now), 1e-6) / 10,
        *_xy(enemy.position, stage),
        enemy.route_index / max(1, len(stage.routes) - 1),
        progress,
        remaining / max(1, stage.map.width + stage.map.height),
        max(0, enemy.wait_until - now) / 120,
        float(blocker is not None),
        max(0, enemy.ready_at - now) / 10,
        d.block_weight / 5,
        d.life_cost / 10,
        *_damage(d.damage_type),
        *_xy(route.end, stage),
        *_xy(next_pos, stage),
        *(_xy(blocker.position, stage) if blocker else (0, 0)),
        blocker.hp / max(blocker.data.hp, 1) if blocker else 0,
        float(not blocker and enemy.wait_until <= now),
        float(enemy.wait_until > now),
        float(enemy.alive),
        (len(path) - enemy.node) / max(1, len(path)),
        sum(m.start_time <= now and (m.end_time is None or now < m.end_time) for m in enemy.modifiers) / 10,
    ]
    from arknights_sim.mechanics.runtime import revealed,gravity_speed
    vector=(next_pos[0]-enemy.position[0],next_pos[1]-enemy.position[1])
    norm=max(1e-6,math.hypot(*vector))
    v.extend([d.weight/5,float(d.invisible),float(revealed(game,enemy)),d.attack_range/10,float(d.splash),gravity_speed(game,enemy,vector)/3,vector[0]/norm,vector[1]/norm])
    assert len(v) == ENEMY_FEATURES
    return v


def map_features(game):
    stage = game.stage
    image = torch.zeros(MAP_CHANNELS, stage.map.height, stage.map.width)
    for y, row in enumerate(stage.map.rows):
        for x, tile in enumerate(row):
            image[:10, y, x] = torch.tensor(
                [
                    tile.height == "LOWLAND",
                    tile.height == "HIGHLAND",
                    tile.buildable == "MELEE",
                    tile.buildable == "RANGED",
                    tile.buildable == "ALL",
                    tile.buildable == "NONE",
                    tile.key == "tile_start",
                    tile.key == "tile_end",
                    0,
                    tile.passable == "ALL",
                ],
                dtype=torch.float32,
            )
            image[18:20, y, x] = torch.tensor(_xy((x, y), stage))
    paths = _paths(stage)
    if stage.devices:
        from arknights_sim.mechanics.runtime import paths_for
        paths=paths_for(game)
    used = {spawn.route_index for spawn in stage.spawns}
    for index in sorted(used):
        points = {stage.routes[index].start, *(w.position for w in paths[index])}
        for pos in points:
            x, y = map(round, pos)
            if 0 <= x < stage.map.width and 0 <= y < stage.map.height:
                image[8, y, x] += 1 / max(1, len(used))
    for unit in game.allies.values():
        if not unit.alive:
            continue
        x, y = map(round, unit.position)
        if 0 <= x < stage.map.width and 0 <= y < stage.map.height:
            image[10:14, y, x] += torch.tensor(
                [
                    1,
                    unit.hp / max(unit.data.hp, 1),
                    modified(unit.data.atk, "ATK", unit.modifiers, game.current_time)
                    / 2000,
                    modified(
                        unit.data.block_count,
                        "BLOCK_COUNT",
                        unit.modifiers,
                        game.current_time,
                    )
                    / 5,
                ]
            )
            image[17, y, x] += float(skill_ready(unit, game.current_time))
    for unit in game.enemies.values():
        if unit.alive and not unit.escaped:
            x, y = map(round, unit.position)
            if 0 <= x < stage.map.width and 0 <= y < stage.map.height:
                image[14:17, y, x] += torch.tensor(
                    [0.1, unit.hp / 10000, float(unit.blocked_by is not None) / 5]
                )
    return image


class StateEncoder:
    MAP_CHANNELS = MAP_CHANNELS
    OPERATOR_FEATURES = OPERATOR_FEATURES
    ENEMY_FEATURES = ENEMY_FEATURES
    GLOBAL_FEATURES = GLOBAL_FEATURES
    FUTURE_FEATURES = FUTURE_FEATURES

    def __init__(self, max_operators=12, max_enemies=64):
        if max_operators < 1 or max_enemies < 1:
            raise ValueError("Entity caps must be positive")
        self.max_operators, self.max_enemies = max_operators, max_enemies

    def encode(self, state):
        game = state.game
        stage, now = game.stage, game.current_time
        keys = tuple(sorted(game.squad))
        enemies = sorted(
            (e for e in game.enemies.values() if e.alive and not e.escaped),
            key=lambda e: e.id,
        )
        if len(keys) > self.max_operators or len(enemies) > self.max_enemies:
            raise ValueError(
                f"Entity capacity exceeded: operators {len(keys)}/{self.max_operators}, "
                f"active enemies {len(enemies)}/{self.max_enemies}"
            )
        operators = torch.zeros(self.max_operators, OPERATOR_FEATURES)
        enemy_tensor = torch.zeros(self.max_enemies, ENEMY_FEATURES)
        op_mask = torch.arange(self.max_operators) < len(keys)
        enemy_mask = torch.arange(self.max_enemies) < len(enemies)
        for i, key in enumerate(keys):
            operators[i] = torch.tensor(operator_features(game, key))
        paths = _paths(stage)
        from arknights_sim.core.stage_events import future_event_state
        graph=future_event_state(game)
        origins={uid:row for row in graph['events'] for uid in row.get('alive_enemy_ids',())}
        blockers=set(graph['progression'].get('blocking_enemy_ids',()))
        enemy_events=torch.zeros(self.max_enemies,5)
        for i, enemy in enumerate(enemies):
            origin=origins.get(enemy.id,{})
            enemy_events[i]=torch.tensor([float(enemy.id in blockers),float(enemy.id in blockers and len(blockers)==1),origin.get('wave_id',0)/100,origin.get('fragment_id',0)/100,origin.get('action_id',0)/100])
            enemy_tensor[i] = torch.tensor(_enemy_features(game, enemy, paths))
        pending = sorted(
            (e for e in game.queue.heap if e.kind == "SPAWN"), key=lambda e: e.sort_key
        )
        total = max(1, len(stage.spawns))
        global_values = [
            now / 300,
            game.dp / 100,
            game.life / max(1, stage.life),
            game.killed / total,
            game.escaped / total,
            game.spawned / total,
            len(stage.spawns) / 100,
            len(enemies) / 64,
            sum(u.alive for u in game.operators.values()) / 8,
            len(keys) / 8,
            game.dp / max(1, stage.max_cost),
            stage.max_cost / 100,
            1 / stage.cost_interval / 5,
            game.life / 20,
            stage.character_limit / 8,
            state.decision_count / max(1, state.max_decisions),
            max(0, state.horizon - now) / 300,
            stage.map.width / 20,
            stage.map.height / 20,
            len(pending) / 100,
            (pending[0].time - now) / 120 if pending else 0,
            sum(e.kind == "HIT" for e in game.queue.heap) / 100,
            float(state.terminal),
            stage.move_multiplier / 5,
        ]
        assert len(global_values) == GLOBAL_FEATURES
        enemy_data = {e.id: e for e in stage.enemies}
        future = torch.zeros(len(FUTURE_WINDOWS), FUTURE_FEATURES)
        for i, window in enumerate(FUTURE_WINDOWS):
            events = [e for e in pending if 0 <= e.time - now <= window + 1e-9]
            if not events:
                continue
            data = [enemy_data[e.payload[0].enemy_id] for e in events]
            routes = [stage.routes[e.payload[0].route_index] for e in events]
            n = len(events)
            future[i] = torch.tensor(
                [
                    n / 100,
                    sum(d.hp for d in data) / 100000,
                    sum(d.atk for d in data) / 20000,
                    sum(d.defense for d in data) / n / 1000,
                    sum(d.speed for d in data) / n / 5,
                    sum(d.interval for d in data) / n / 10,
                    (events[0].time - now) / 120,
                    sum(e.time - now for e in events) / n / 120,
                    len({e.payload[0].route_index for e in events})
                    / max(1, len(stage.routes)),
                    sum(_xy(r.start, stage)[0] for r in routes) / n,
                    sum(_xy(r.start, stage)[1] for r in routes) / n,
                    sum(d.life_cost for d in data) / 100,
                ]
            )
        from .mechanics_encoder import mechanics_features
        from .future_event_encoder import encode_event_graph
        summon_keys = tuple(sorted(game.summon_cards)) + tuple(sorted(k for k,u in game.summons.items() if u.alive))
        summons = torch.zeros(max(1,len(summon_keys)),56)
        for i,key in enumerate(summon_keys):
            unit=game.summons.get(key)
            card=game.summon_cards[key] if key in game.summon_cards else game.summon_cards[unit.summon_card_id]
            owner=game.operators.get(card.owner_id)
            summons[i]=torch.tensor(operator_features(game,key)+[
                float(unit is None),float(unit is not None),game.summon_inventory[card.id]/10,
                card.max_active/10,sum(u.alive and u.summon_card_id==card.id for u in game.summons.values())/10,
                float(bool(owner and owner.alive)),card.slot_weight/8,float(card.return_on_retreat)])
        return {
            **encode_event_graph(game),
            "summons": summons,
            "summon_mask": torch.arange(len(summons))<len(summon_keys),
            "mechanics": mechanics_features(game),
            "map": map_features(game),
            "map_mask": torch.ones(stage.map.height, stage.map.width, dtype=torch.bool),
            "operators": operators,
            "operator_mask": op_mask,
            "enemies": enemy_tensor,
            "enemy_events": enemy_events,
            "enemy_mask": enemy_mask,
            "global": torch.tensor(global_values),
            "future": future,
        }

    __call__ = encode


def collate_states(states):
    """Pad maps and entities without changing their physical coordinate origin."""
    if not states:
        raise ValueError("Cannot collate an empty batch")
    batch = len(states)
    height = max(s["map"].shape[-2] for s in states)
    width = max(s["map"].shape[-1] for s in states)
    op_count = max(s["operators"].shape[0] for s in states)
    enemy_count = max(s["enemies"].shape[0] for s in states)
    from .mechanics_encoder import MECHANICS_CHANNELS
    from .future_event_encoder import EVENT_FEATURES, PROGRESSION_FEATURES
    event_count=max(s.get('events',torch.zeros(1,EVENT_FEATURES)).shape[0] for s in states)
    summon_count=max(s.get('summons',torch.zeros(1,56)).shape[0] for s in states)
    result = {
        "events": states[0]["map"].new_zeros(batch,event_count,EVENT_FEATURES),
        "event_mask": torch.zeros(batch,event_count,dtype=torch.bool),
        "event_relations": torch.zeros(batch,event_count,event_count,dtype=torch.long),
        "progression": torch.stack([s.get('progression',torch.zeros(PROGRESSION_FEATURES)) for s in states]),
        "summons": states[0]["map"].new_zeros(batch,summon_count,56),
        "summon_mask": torch.zeros(batch,summon_count,dtype=torch.bool),
        "mechanics": states[0]["map"].new_zeros(batch, MECHANICS_CHANNELS, height, width),
        "map": states[0]["map"].new_zeros(batch, MAP_CHANNELS, height, width),
        "map_mask": states[0]["map_mask"].new_zeros(batch, height, width),
        "operators": states[0]["operators"].new_zeros(
            batch, op_count, OPERATOR_FEATURES
        ),
        "operator_mask": states[0]["operator_mask"].new_zeros(batch, op_count),
        "enemies": states[0]["enemies"].new_zeros(batch, enemy_count, ENEMY_FEATURES),
        "enemy_mask": states[0]["enemy_mask"].new_zeros(batch, enemy_count),
        "enemy_events": states[0]["map"].new_zeros(batch,enemy_count,5),
        "global": torch.stack([s["global"] for s in states]),
        "future": torch.stack([s["future"] for s in states]),
    }
    for i, state in enumerate(states):
        if 'enemy_events' in state:result['enemy_events'][i,:len(state['enemy_events'])]=state['enemy_events']
        if 'events' in state:
            n=state['events'].shape[0]
            result['events'][i,:n]=state['events'];result['event_mask'][i,:n]=state['event_mask']
            result['event_relations'][i,:n,:n]=state['event_relations']
        if 'summons' in state:
            n=state['summons'].shape[0]
            result['summons'][i,:n]=state['summons'];result['summon_mask'][i,:n]=state['summon_mask']
        h, w = state["map"].shape[-2:]
        result["map"][i, :, :h, :w] = state["map"]
        if "mechanics" in state:result["mechanics"][i, :, :h, :w] = state["mechanics"]
        result["map_mask"][i, :h, :w] = state["map_mask"]
        for entity, mask in (("operators", "operator_mask"), ("enemies", "enemy_mask")):
            count = state[entity].shape[0]
            result[entity][i, :count] = state[entity]
            result[mask][i, :count] = state[mask]
    return result
