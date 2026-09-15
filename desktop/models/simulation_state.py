"""Frozen presentation records. No live simulator objects cross the UI boundary."""

import json
from dataclasses import dataclass, replace

from arknights_sim.map.route import compile_route

SUPPORTED_SQUAD = ("char_500_noirc", "char_208_melan")


@dataclass(frozen=True)
class TileSnapshot:
    key: str
    height: str
    buildable: str
    passable: str


@dataclass(frozen=True)
class UnitSnapshot:
    id: str
    name: str
    kind: str
    position: tuple[float, float]
    hp: float
    max_hp: float
    direction: int = 0
    blocked_by: str | None = None
    skill_ready: bool = False
    skill_active: bool = False


@dataclass(frozen=True)
class ActionOption:
    label: str
    payload_json: str


@dataclass(frozen=True)
class BattleEvent:
    time: float
    kind: str
    category: str
    detail: str

    @property
    def text(self):
        return f"{self.time:08.3f}  {self.kind}  {self.detail}"


@dataclass(frozen=True)
class SimulationSnapshot:
    stage_id: str
    time: float
    dp: float
    life: int
    kills: int
    leaks: int
    total_enemies: int
    terminal: bool
    result: str
    tiles: tuple[tuple[TileSnapshot, ...], ...]
    routes: tuple[tuple[tuple[float, float], ...], ...]
    units: tuple[UnitSnapshot, ...] = ()
    legal_actions: tuple[ActionOption, ...] = ()
    decision_count: int = 0
    mechanics_status: str = ''
    event_progression: tuple = ()
    upcoming_events: tuple = ()

    @property
    def width(self):
        return len(self.tiles[0]) if self.tiles else 0

    @property
    def height(self):
        return len(self.tiles)


def snapshot_from_stage(stage):
    """Create a map-only DTO in a worker, using the core's route compiler."""
    used = sorted({spawn.route_index for spawn in stage.spawns})
    return SimulationSnapshot(
        stage_id=stage.id,
        time=0.0,
        dp=stage.initial_cost,
        life=stage.life,
        kills=0,
        leaks=0,
        total_enemies=len(stage.spawns),
        terminal=False,
        result="READY",
        tiles=tuple(
            tuple(
                TileSnapshot(tile.key, tile.height, tile.buildable, tile.passable)
                for tile in row
            )
            for row in stage.map.rows
        ),
        routes=tuple(
            (
                stage.routes[index].start,
                *(
                    waypoint.position
                    for waypoint in compile_route(stage.map, stage.routes[index])
                    if waypoint.kind != "WAIT_FOR_SECONDS"
                ),
            )
            for index in used
        ),
    )


def snapshot_from_state(env, state, stage_snapshot=None):
    from arknights_sim.environment.legal_actions import skill_ready

    game = state.game
    base = stage_snapshot or snapshot_from_stage(game.stage)
    units = tuple(
        UnitSnapshot(
            unit.id,
            unit.data.name,
            kind,
            tuple(unit.position),
            float(unit.hp),
            float(unit.data.hp),
            unit.direction,
            unit.blocked_by,
            bool(kind in ("operator", "summon") and skill_ready(unit, game.current_time)),
            bool(kind in ("operator", "summon") and (unit.skill.active_until > game.current_time
                 or any(m.start_time <= game.current_time < m.end_time for m in unit.modifiers))),
        )
        for kind, group in (("operator", game.operators), ("summon", game.summons), ("enemy", game.enemies))
        for unit in sorted(group.values(), key=lambda item: item.id)
        if unit.alive and not unit.escaped
    )
    names={'crate':'障碍物','sensor':'侦测器','wind':'源石流发生装置','gravity':'重力控制'}
    units += tuple(UnitSnapshot(k,names[d.data.kind],'device',d.position,d.hp,100,d.data.direction,skill_ready=d.data.charge>0 and d.sp>=d.data.charge,skill_active=game.current_time<d.active_until) for k,d in game.devices.items() if d.alive and d.data.kind!='gravity')
    def action_label(action):
        if not action.operator_id:return str(action)
        if action.operator_id in game.squad:return str(action).replace(action.operator_id,game.squad[action.operator_id].name)
        if action.operator_id in game.summon_cards:
            card=game.summon_cards[action.operator_id]
            return str(action).replace(action.operator_id, f'{game.squad[card.owner_id].name} / {card.unit.name}')
        if action.operator_id in game.summons:
            unit=game.summons[action.operator_id]
            return str(action).replace(action.operator_id, f'{unit.data.name} #{unit.id.rsplit("_",1)[-1]}')
        d=game.devices.get(action.operator_id)
        data=d.data if d else next(v for v in game.stage.devices if v.id==action.operator_id)
        return str(action).replace(action.operator_id,names[data.kind])
    legal = tuple(
        ActionOption(
            action_label(action),
            json.dumps(action.to_dict(), separators=(",", ":")),
        )
        for action in env.legal_actions(state)
    )
    outcome = env.result(state)
    event_state = env.future_event_state(state) if hasattr(env, 'future_event_state') else {}
    progression = event_state.get('progression', {})
    event_progression = tuple((key, progression[key]) for key in (
        'current_wave', 'current_fragment', 'current_event_id', 'blocking',
        'blocking_enemy_count', 'remaining_actions', 'completion_progress', 'next_trigger',
    ) if key in progression)
    enemy_names = {enemy.id: enemy.name for enemy in game.stage.enemies}
    upcoming = tuple(
        (enemy_names.get(row.get('enemy_id'), row.get('enemy_id', '')),
         row.get('route_id', -1), row.get('remaining_count', 0),
         row.get('estimated_relative_time') if row.get('time_known') else None)
        for row in event_state.get('events', [])
        if row.get('enemy_id') and row.get('remaining_count', 0) > 0
    )[:3]
    info=[]
    if game.gravity is not None:info.append('重力方向：'+('右','上','左','下')[game.gravity])
    for key,d in game.devices.items():
        if d.alive and d.data.kind=='sensor':info.append('侦测器：'+(f'反隐剩余 {d.active_until-game.current_time:.1f}秒' if game.current_time<d.active_until else f'充能 {d.sp:.1f}/{d.data.charge:g}'))
    for key,count in game.device_inventory.items():info.append(f'障碍物剩余：{count}')
    for key,card in game.summon_cards.items():
        info.append(f'{card.unit.name}剩余：{game.summon_inventory[key]} · 冷却 {max(0,game.summon_ready_at[key]-game.current_time):.1f}s')
    if game.stage.devices:
        from arknights_sim.mechanics.runtime import paths_for
        paths=paths_for(game)
        base=replace(base,routes=tuple((game.stage.routes[i].start,*(w.position for w in paths[i] if w.kind!='WAIT_FOR_SECONDS')) for i in sorted({s.route_index for s in game.stage.spawns})))
    result = (
        "SUCCESS"
        if outcome.success
        else f"FAILED ({outcome.termination})"
        if outcome.terminal
        else "RUNNING"
    )
    return replace(
        base,
        time=game.current_time,
        dp=float(game.dp),
        life=game.life,
        kills=game.killed,
        leaks=game.escaped,
        terminal=outcome.terminal,
        result=result,
        units=units,
        legal_actions=legal,
        decision_count=state.decision_count,
        mechanics_status=' · '.join(info),
        event_progression=event_progression,
        upcoming_events=upcoming,
    )


def event_from_record(record):
    kind = str(record["event"])
    if kind in ("SPAWN", "ENEMY_SPAWN"):
        category = "Spawn"
    elif kind in ("MOVING", "WAYPOINT", "WAIT", "BLOCK", "UNBLOCK"):
        category = "Movement"
    elif kind in ("DEPLOY", "RETREAT", "SUMMON_DEPLOY", "SUMMON_RETREAT", "SUMMON_REMOVE"):
        category = "Deploy"
    elif "SKILL" in kind:
        category = "Skill"
    elif kind in ("HIT", "DAMAGE", "HEAL", "HP_RECOVERY", "SPLASH_DAMAGE"):
        category = "Damage"
    elif kind in ("DEATH", "KILL", "ESCAPE", "OPERATOR_DEATH", "ENEMY_DEATH"):
        category = "Death"
    elif "ATTACK" in kind:
        category = "Attack"
    else:
        category = "Other"
    detail = " ".join(
        f"{key}={value}"
        for key, value in record.items()
        if key not in ("time", "event")
    )
    return BattleEvent(float(record["time"]), kind, category, detail)
