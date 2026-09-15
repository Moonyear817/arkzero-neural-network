"""Authoritative deterministic stage progression and its read-only observation.

Action definitions are immutable. Every dynamic counter, gate, start time and
spawn provenance is stored on GameState and therefore cloned with search nodes.
A conditional ETA is reported as unknown; a lower bound is never an exact ETA.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import math
from arknights_sim.data.event_graph import action_definitions, event_dependencies
from arknights_sim.data.models import SpawnData

EVENT_SCHEMA_VERSION = 1
_EPS = 1e-8


@dataclass
class ActionProgress:
    started_at: float | None = None
    spawned: int = 0
    unit_ids: list[str] = field(default_factory=list)


@dataclass
class StageEventRuntime:
    wave_index: int = 0
    fragment_index: int = -1
    phase: str = 'wave_delay'
    deadline: float | None = None
    gate_started_at: float | None = None
    wave_started_at: dict[int, float] = field(default_factory=dict)
    fragment_started_at: dict[tuple[int, int], float] = field(default_factory=dict)
    actions: dict[int, ActionProgress] = field(default_factory=dict)
    revision: int = 0


def initialize(game):
    # Empty fragments in a real parsed wave are supported too. Legacy StageData
    # has no fragment definitions and continues its exact absolute queue path.
    if any(w.event_scheduled or w.fragments for w in game.stage.waves):
        game.stage_events = StageEventRuntime(
            deadline=game.stage.waves[0].pre_delay,
            actions={a.index: ActionProgress() for a in action_definitions(game.stage)},
        )
    else:
        game.stage_events = None


def next_boundary(game):
    runtime = game.stage_events
    if runtime is None or runtime.phase == 'complete' or runtime.deadline is None:
        return math.inf
    # Already-due gates waiting for enemies must not create a zero-time loop.
    if runtime.deadline <= game.current_time + _EPS and runtime.phase in ('actions', 'wave_gate'):
        return math.inf
    return runtime.deadline


def _alive(game, action):
    runtime = game.stage_events
    if runtime is None:
        return ()
    return tuple(uid for uid in runtime.actions[action.index].unit_ids
                 if uid in game.enemies and game.enemies[uid].alive
                 and game.enemies[uid].data.faction == 'HOSTILE')


def _wave_blockers(game):
    runtime = game.stage_events
    return tuple(uid for a in action_definitions(game.stage)
                 if a.wave_id <= runtime.wave_index and a.event_type == 'SPAWN'
                 and not a.dont_block_wave for uid in _alive(game, a))


def blockers(game):
    runtime = game.stage_events
    if runtime is None or runtime.phase == 'complete':
        return ()
    if runtime.phase == 'wave_gate':
        return _wave_blockers(game)
    if runtime.phase == 'actions':
        fragment = game.stage.waves[runtime.wave_index].fragments[runtime.fragment_index]
        return tuple(uid for a in fragment.actions if a.block_fragment for uid in _alive(game, a))
    return ()


def register_spawn(game, spawn, unit_id):
    runtime = game.stage_events
    if runtime is not None and spawn.action_index in runtime.actions:
        action = runtime.actions[spawn.action_index]
        action.spawned += 1
        action.unit_ids.append(unit_id)
        runtime.revision += 1


def settle(sim):
    """Resolve only ready stage transitions; combat damage priorities stay intact."""
    game, now = sim.state, sim.current_time
    runtime = game.stage_events
    if runtime is None or runtime.phase == 'complete':
        return False
    changed = False
    while runtime.phase != 'complete':
        wave = game.stage.waves[runtime.wave_index]
        due = runtime.deadline is not None and now + _EPS >= runtime.deadline
        if runtime.phase == 'wave_delay':
            if not due:
                break
            runtime.wave_started_at[runtime.wave_index] = now
            runtime.fragment_index = 0
            sim._log('WAVE_START', wave=runtime.wave_index)
            if wave.fragments:
                runtime.phase = 'fragment_delay'
                runtime.deadline = now + wave.fragments[0].pre_delay
            else:
                runtime.phase = 'wave_gate'
                runtime.gate_started_at = now
                runtime.deadline = now + wave.max_wait if wave.max_wait >= 0 else None
        elif runtime.phase == 'fragment_delay':
            if not due:
                break
            fragment = wave.fragments[runtime.fragment_index]
            runtime.fragment_started_at[(runtime.wave_index, runtime.fragment_index)] = now
            sim._log('FRAGMENT_START', wave=runtime.wave_index, fragment=runtime.fragment_index)
            duration = 0.0
            for action in fragment.actions:
                runtime.actions[action.index].started_at = now
                duration = max(duration, action.pre_delay + max(0, action.count - 1) * action.interval)
                if action.event_type == 'SPAWN':
                    for i in range(action.count):
                        spawn = SpawnData(now + action.pre_delay + i * action.interval,
                                          action.key, action.route_id, action.wave_id,
                                          action.fragment_id, action.index)
                        game.queue.push(spawn.time, 'SPAWN', spawn)
            runtime.phase = 'actions'
            runtime.deadline = now + duration
            # First drain newly enqueued same-time spawns before checking alive
            # blockers or allowing the next fragment to activate.
            if game.queue.peek() <= now + _EPS:
                runtime.revision += 1
                return True
        elif runtime.phase == 'actions':
            if not due:
                break
            fragment = wave.fragments[runtime.fragment_index]
            if any(a.event_type == 'SPAWN' and runtime.actions[a.index].spawned < a.count
                   for a in fragment.actions):
                break
            if blockers(game):
                break
            sim._log('FRAGMENT_END', wave=runtime.wave_index, fragment=runtime.fragment_index)
            runtime.fragment_index += 1
            if runtime.fragment_index < len(wave.fragments):
                runtime.phase = 'fragment_delay'
                runtime.deadline = now + wave.fragments[runtime.fragment_index].pre_delay
            else:
                # Keep the last valid fragment index while awaiting wave clear.
                runtime.fragment_index = len(wave.fragments) - 1
                runtime.phase = 'wave_gate'
                runtime.gate_started_at = now
                runtime.deadline = now + wave.max_wait if wave.max_wait >= 0 else None
        elif runtime.phase == 'wave_gate':
            if _wave_blockers(game) and not due:
                break
            sim._log('WAVE_GATE_OPEN', wave=runtime.wave_index,
                     reason='timeout' if _wave_blockers(game) else 'clear')
            runtime.phase = 'post_delay'
            runtime.deadline = now + wave.post_delay
        elif runtime.phase == 'post_delay':
            if not due:
                break
            sim._log('WAVE_END', wave=runtime.wave_index)
            runtime.wave_index += 1
            runtime.fragment_index = -1
            runtime.gate_started_at = None
            if runtime.wave_index == len(game.stage.waves):
                runtime.phase = 'complete'
                runtime.deadline = None
            else:
                runtime.phase = 'wave_delay'
                runtime.deadline = now + game.stage.waves[runtime.wave_index].pre_delay
        else:
            raise ValueError(f'Unknown event phase: {runtime.phase}')
        runtime.revision += 1
        changed = True
    return changed


def scheduling_complete(game):
    return game.stage_events is None or game.stage_events.phase == 'complete'


@lru_cache(maxsize=64)
def _static_rows(stage):
    """Cached immutable descriptions; callers receive fresh output dictionaries."""
    actions = action_definitions(stage)
    if not actions:
        # Legacy synthetic stages remain useful to all real training interfaces.
        return tuple((i, s) for i, s in enumerate(stage.spawns))
    return tuple((a.index, a) for a in actions)


def _fragment_estimates(game):
    """Earliest starts and timing certainty, accounting for current unresolved gates."""
    runtime = game.stage_events
    if runtime is None:
        return {}
    out, cursor, known = {}, 0.0, True
    for wi, wave in enumerate(game.stage.waves):
        # A real activation removes uncertainty inherited from previous gates.
        if wi in runtime.wave_started_at:
            cursor, known = runtime.wave_started_at[wi], True
        else:
            cursor += wave.pre_delay
            if wi == runtime.wave_index and runtime.phase == 'wave_delay':
                cursor, known = runtime.deadline, True
        for fi, fragment in enumerate(wave.fragments):
            key = (wi, fi)
            if key in runtime.fragment_started_at:
                cursor, known = runtime.fragment_started_at[key], True
            else:
                cursor += fragment.pre_delay
                if (wi == runtime.wave_index and fi == runtime.fragment_index
                        and runtime.phase == 'fragment_delay'):
                    cursor, known = runtime.deadline, True
            out[key] = (cursor, known)
            duration = max((a.pre_delay + max(0, a.count - 1) * a.interval
                            for a in fragment.actions), default=0)
            cursor += duration
            if any(a.block_fragment and (runtime.actions[a.index].started_at is None
                    or runtime.actions[a.index].spawned < a.count or _alive(game, a))
                   for a in fragment.actions):
                known = False
                cursor = max(cursor, game.current_time)
        # A wave can also wait on nonblocking-fragment enemies from earlier
        # fragments/waves. Exact death times cannot be read from the stage file.
        unresolved = any(a.event_type == 'SPAWN' and not a.dont_block_wave
                         and a.wave_id <= wi and (runtime.actions[a.index].started_at is None
                         or runtime.actions[a.index].spawned < a.count or _alive(game, a))
                         for a in action_definitions(game.stage))
        if wi == runtime.wave_index and runtime.phase == 'post_delay':
            cursor, known = runtime.deadline, True
        else:
            if unresolved and wave.max_wait != 0:
                known = False
                cursor = max(cursor, game.current_time)
            cursor += wave.post_delay
        if wi >= runtime.wave_index and not known:
            # Lower bounds for pending work cannot be in the past.
            cursor = max(cursor, game.current_time)
    return out


def _details(stage, enemy_id, route_id):
    enemy = next((e for e in stage.enemies if e.id == enemy_id), None)
    stats = ({name: getattr(enemy, name) for name in ('hp', 'atk', 'defense', 'resistance',
              'speed', 'life_cost', 'weight', 'block_weight', 'interval', 'attack_range',
              'invisible', 'splash', 'damage_type')} if enemy else {})
    route = stage.routes[route_id] if 0 <= route_id < len(stage.routes) else None
    geometry = ({'start': list(route.start), 'end': list(route.end),
                 'waypoints': [{'kind': p.kind, 'position': list(p.position), 'time': p.time}
                               for p in route.waypoints]} if route else {})
    return stats, geometry


def future_event_state(game):
    """Complete public event graph, with dynamic status and conditional timing."""
    runtime, now, stage = game.stage_events, game.current_time, game.stage
    edges = event_dependencies(stage)
    parents = {}
    for source, target, relation in edges:
        parents.setdefault(target, []).append((source, relation))
    estimates = _fragment_estimates(game)
    events = []
    if runtime is None:
        # Compare values because callers may deepcopy states and queue payloads.
        pending = [e.payload[0] for e in game.queue.heap if e.kind == 'SPAWN']
        for i, spawn in _static_rows(stage):
            remain = int(spawn in pending)
            stats, geometry = _details(stage, spawn.enemy_id, spawn.route_index)
            events.append(dict(index=i, event_id=f'legacy_spawn_{i}', type='SPAWN', event_type='SPAWN',
                wave_id=spawn.wave, fragment_id=spawn.fragment, action_id=i, enemy_id=spawn.enemy_id,
                route_id=spawn.route_index, spawn_count=1, remaining_count=remain, spawned_count=1-remain,
                alive_count=0, alive_enemy_ids=[], pre_delay=spawn.time, interval=0, trigger_condition='absolute_time',
                dependencies=[], status='pending' if remain else 'complete', blocking=False,
                estimated_relative_time=max(0, spawn.time-now), time_known=True,
                earliest_relative_time=max(0, spawn.time-now), wave_pre_delay=0, wave_post_delay=0,
                wave_max_wait=-1, fragment_pre_delay=0, enemy_stats=stats, route_geometry=geometry))
    else:
        for _, action in _static_rows(stage):
            progress = runtime.actions[action.index]
            start, known = estimates[(action.wave_id, action.fragment_id)]
            alive = len(_alive(game, action))
            remaining = action.count-progress.spawned if action.event_type == 'SPAWN' else 0
            last_time = start + action.pre_delay + max(0, action.count-1) * action.interval
            finished = (progress.started_at is not None and now + _EPS >= last_time
                        and remaining == 0 and (not action.block_fragment or alive == 0))
            status = 'complete' if finished else 'active' if progress.started_at is not None else 'pending'
            first_remaining = start + action.pre_delay + progress.spawned * action.interval
            if action.event_type == 'SPAWN':
                relative = max(0, first_remaining-now) if remaining else 0.0
            else:
                # Passive UI groups have no enemy count, but their delayed
                # occurrence still determines the next fragment's timing.
                occurrence = start + action.pre_delay
                if progress.started_at is not None and occurrence <= now + _EPS and action.interval > 0:
                    occurrence += min(max(0, action.count-1),
                                      math.floor((now-occurrence)/action.interval + _EPS)+1) * action.interval
                relative = max(0, occurrence-now) if not finished else 0.0
            deps = parents.get(action.index, [])
            conditional = [r for _, r in deps if r in ('fragment_clear', 'wave_clear', 'wave_clear_or_timeout')]
            trigger = conditional[0] if conditional else 'fragment_complete' if deps else 'stage_start'
            stats, geometry = _details(stage, action.key if action.event_type == 'SPAWN' else '', action.route_id)
            events.append(dict(index=action.index, event_id=action.event_id, type=action.event_type,
                event_type=action.event_type, wave_id=action.wave_id, fragment_id=action.fragment_id,
                action_id=action.action_id, enemy_id=action.key if action.event_type == 'SPAWN' else '',
                route_id=action.route_id, spawn_count=action.count if action.event_type == 'SPAWN' else 0,
                remaining_count=remaining, spawned_count=progress.spawned, alive_count=alive,
                alive_enemy_ids=list(_alive(game, action)),
                pre_delay=action.pre_delay, interval=action.interval, trigger_condition=trigger,
                dependencies=sorted({s for s, _ in deps}), status=status,
                blocking=bool(alive and (action.block_fragment or not action.dont_block_wave)),
                block_fragment=action.block_fragment, dont_block_wave=action.dont_block_wave,
                estimated_relative_time=relative if known else None, time_known=known,
                earliest_relative_time=relative, wave_pre_delay=stage.waves[action.wave_id].pre_delay,
                wave_post_delay=stage.waves[action.wave_id].post_delay,
                wave_max_wait=stage.waves[action.wave_id].max_wait,
                fragment_pre_delay=stage.waves[action.wave_id].fragments[action.fragment_id].pre_delay,
                enemy_stats=stats, route_geometry=geometry))
    pending = [e for e in events if e['status'] != 'complete']
    remaining = len(stage.spawns)-game.spawned
    blocking_ids = blockers(game) if runtime else ()
    phase = runtime.phase if runtime else 'absolute_timeline'
    if runtime:
        current_wave_rows = [e for e in events if e['wave_id'] == runtime.wave_index]
        current = next((e for e in current_wave_rows if e['status'] == 'active'), None)
        if current is None:
            current = next((e for e in current_wave_rows if e['alive_count'] and e['blocking']), None)
        if current is None:
            candidates = [e for e in current_wave_rows if e['fragment_id'] == runtime.fragment_index]
            current = candidates[-1] if candidates else None
    else:
        current = pending[0] if pending else None
    active_wave = stage.waves[runtime.wave_index] if runtime and runtime.wave_index < len(stage.waves) else None
    timed_boundary = min(next_boundary(game), min((event.time for event in game.queue.heap
                        if event.kind == 'SPAWN'), default=math.inf))
    progression = dict(
        current_wave=runtime.wave_index if runtime and phase != 'complete' else (current['wave_id'] if current else -1),
        current_fragment=runtime.fragment_index if runtime else (current['fragment_id'] if current else -1),
        current_event_id=(current['event_id'] if current else f'wave_{runtime.wave_index}_{phase}'
                          if runtime and phase != 'complete' else None),
        phase=phase, blocking=bool(blocking_ids), blocking_enemy_count=len(blocking_ids),
        blocking_enemy_ids=list(blocking_ids), remaining_actions=len(pending),
        remaining_enemies=max(0, len(stage.spawns)-game.killed-game.escaped),
        unspawned_enemies=max(0, remaining), completion_progress=(sum(e['status']=='complete' for e in events)/max(1,len(events))),
        wave_can_end=bool(runtime and (phase == 'post_delay' or (phase == 'wave_gate' and not _wave_blockers(game)))),
        next_trigger=('battle_complete' if game.done else
                      ('remaining_enemies_resolved' if phase=='complete' else
                       'enemy_clear_or_timeout' if phase=='wave_gate' and runtime.deadline is not None else
                       'enemy_clear' if phase=='wave_gate' or blocking_ids else 'time')),
        revision=runtime.revision if runtime else game.spawned,
        next_boundary_relative_time=(max(0, timed_boundary-now) if math.isfinite(timed_boundary) else None),
        wave_timeout_remaining=(max(0, runtime.deadline-now) if runtime and phase=='wave_gate'
                                and runtime.deadline is not None else None),
        last_blocker=len(blocking_ids)==1,
        current_wave_pre_delay=active_wave.pre_delay if active_wave else 0,
        current_wave_post_delay=active_wave.post_delay if active_wave else 0,
        current_wave_max_wait=active_wave.max_wait if active_wave else -1,
    )
    return {'schema_version': EVENT_SCHEMA_VERSION, 'events': events,
            'dependencies': list(edges), 'progression': progression}


def route_wait_deadline(game, checkpoint, arrival):
    """Interpret route waits without turning elapsed wave time into a new delay.

    ROUTE-WAIT-001 ASSUMED: CURRENT uses the active scheduler clock when the
    checkpoint is reached. Freeze the deadline on entry so later fragment
    changes cannot extend an already-entered wait. Client parity is unverified.
    """
    if checkpoint.kind == 'WAIT_FOR_SECONDS':
        return arrival + checkpoint.time
    runtime = game.stage_events
    if runtime is None:
        raise ValueError('Relative route wait requires an event-scheduled stage')
    if checkpoint.kind == 'WAIT_CURRENT_WAVE_TIME':
        clocks = runtime.wave_started_at
    elif checkpoint.kind == 'WAIT_CURRENT_FRAGMENT_TIME':
        clocks = runtime.fragment_started_at
    else:
        raise ValueError('Unknown route wait: ' + checkpoint.kind)
    # Completion advances wave_index beyond the final wave, but living units
    # still retain access to the most recently activated clock.
    if not clocks:
        raise ValueError('Relative route wait has no activated scheduler clock')
    return max(arrival, clocks[max(clocks)] + checkpoint.time)
