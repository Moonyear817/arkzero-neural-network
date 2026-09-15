"""Normalize supported level actions without inventing external trigger behavior.

The existing single-wave timing is preserved. Multi-wave clear/timeout and
blocking SPAWN groups are explicit simulator rules pending client timing
validation; branches, hidden/random actions and blocking UI controls fail closed.
"""
from functools import lru_cache
import math
from .models import FragmentData, SpawnData, StageActionData, WaveData

PASSIVE_ACTIONS = frozenset(('STORY', 'DISPLAY_ENEMY_INFO', 'PREVIEW_CURSOR', 'EMPTY'))


def _nonnegative(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be finite and nonnegative')
    return value


def parse_event_waves(raw_waves, routes):
    waves, cursor, index = [], 0.0, 0
    for wi, wave in enumerate(raw_waves):
        pre = _nonnegative(wave.get('preDelay', 0), 'Wave preDelay')
        post = wave.get('postDelay', 0)
        if post < 0:
            raise NotImplementedError('External/advanced negative wave postDelay')
        _nonnegative(post, 'Wave postDelay')
        timeout = wave.get('maxTimeWaitingForNextWave', -1)
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or (timeout < 0 and timeout != -1):
            raise ValueError('Invalid wave wait timeout')
        cursor += pre
        fragments, inventory = [], []
        for fi, fragment in enumerate(wave.get('fragments') or []):
            delay = _nonnegative(fragment.get('preDelay', 0), 'Fragment preDelay')
            cursor += delay
            actions, duration = [], 0.0
            for ai, action in enumerate(fragment.get('actions') or []):
                kind = action.get('actionType')
                if kind not in PASSIVE_ACTIONS and kind != 'SPAWN':
                    raise NotImplementedError(f'Unsupported stage action: {kind}')
                if (action.get('hiddenGroup') or action.get('randomSpawnGroupKey')
                        or action.get('randomSpawnGroupPackKey')
                        or action.get('randomType', 'ALWAYS') != 'ALWAYS'):
                    raise NotImplementedError('Conditional external/random spawn')
                if not action.get('managedByScheduler', True):
                    raise NotImplementedError('Externally managed stage action')
                if action.get('forceBlockWaveInBranch'):
                    raise NotImplementedError('Branch/special counted spawn')
                if action.get('isUnharmfulAndAlwaysCountAsKilled') and action.get('key') not in {'enemy_3001_upeopl','enemy_3002_ftrtal'}:
                    raise NotImplementedError('Special counted spawn: ' + action.get('key', ''))
                block = bool(action.get('blockFragment', False))
                if block and kind != 'SPAWN':
                    raise NotImplementedError('Blocking external/UI stage action')
                count = action.get('count', 0)
                if type(count) is not int or count < 0:
                    raise ValueError('Spawn count must be a nonnegative integer')
                action_pre = _nonnegative(action.get('preDelay', 0), 'Action preDelay')
                interval = _nonnegative(action.get('interval', 0), 'Action interval')
                route = action.get('routeIndex', -1)
                if kind == 'SPAWN':
                    if not isinstance(route, int) or not 0 <= route < len(routes):
                        raise ValueError('Unknown route')
                    if routes[route].motion not in ('WALK', 'FLY'):
                        raise NotImplementedError(routes[route].motion)
                definition = StageActionData(index, f'wave_{wi}_fragment_{fi}_action_{ai}',
                                             wi, fi, ai, kind, action.get('key', ''),
                                             count, action_pre, interval, route, block,
                                             bool(action.get('dontBlockWave', False)))
                actions.append(definition)
                duration = max(duration, action_pre + max(0, count - 1) * interval)
                if kind == 'SPAWN':
                    inventory.extend(SpawnData(cursor + action_pre + i * interval,
                                               definition.key, route, wi, fi, index)
                                     for i in range(count))
                index += 1
            fragments.append(FragmentData(delay, tuple(actions)))
            cursor += duration
        waves.append(WaveData(pre, post, tuple(f.pre_delay for f in fragments),
                              tuple(inventory), tuple(fragments), timeout, True))
        cursor += post
    return tuple(waves)


@lru_cache(maxsize=64)
def action_definitions(stage):
    return tuple(a for w in stage.waves for f in w.fragments for a in f.actions)


@lru_cache(maxsize=64)
def event_dependencies(stage):
    """Directed semantic dependencies between action groups, not chronological IDs."""
    edges = set()
    previous_wave_actions = []
    previous_fragment = []
    previous_wave = None
    for wave in stage.waves:
        wave_actions = [a for f in wave.fragments for a in f.actions]
        first = True
        for fragment in wave.fragments:
            if not fragment.actions:
                continue
            if first:
                sources = previous_fragment
                for target in fragment.actions:
                    for source in sources:
                        relation = 'fragment_clear' if source.block_fragment else 'wave_actions_complete'
                        edges.add((source.index, target.index, relation))
                    for source in previous_wave_actions:
                        if (source.event_type == 'SPAWN' and not source.dont_block_wave
                                and previous_wave is not None and previous_wave.max_wait != 0):
                            # A hard final-fragment gate cannot be bypassed by
                            # the later wave timeout. Do not replace it with a
                            # weaker OR condition on the same source/target.
                            if source in previous_fragment and source.block_fragment:
                                continue
                            relation = 'wave_clear_or_timeout' if previous_wave.max_wait > 0 else 'wave_clear'
                            edges.add((source.index, target.index, relation))
                first = False
            else:
                for target in fragment.actions:
                    for source in previous_fragment:
                        relation = 'fragment_clear' if source.block_fragment else 'action_complete'
                        edges.add((source.index, target.index, relation))
            previous_fragment = list(fragment.actions)
        previous_wave_actions.extend(wave_actions)
        previous_wave = wave
    return tuple(sorted(edges))
