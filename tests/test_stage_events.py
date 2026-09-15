"""Stage-event causality contracts; these are not claims of client-frame parity."""
import json
from dataclasses import replace
import pytest
from conftest import DATA
from arknights_sim import Simulator
from arknights_sim.data.event_graph import parse_event_waves
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.environment import Action, ArknightsEnv, WAIT
from arknights_sim.core.stage_events import future_event_state


def spawn(count=1, delay=0, interval=1, *, block=False, nonblocking_wave=False):
    return dict(actionType='SPAWN', key='e', count=count, preDelay=delay,
                interval=interval, routeIndex=0, blockFragment=block,
                dontBlockWave=nonblocking_wave, managedByScheduler=True)


def fragment(*actions, delay=0):
    return dict(preDelay=delay, actions=list(actions))


def wave(*fragments, pre=0, post=0, timeout=-1):
    return dict(preDelay=pre, postDelay=post, maxTimeWaitingForNextWave=timeout,
                fragments=list(fragments))


def event_stage(simple, *waves):
    return replace(simple, waves=parse_event_waves(waves, simple.routes),
                   enemies=(replace(simple.enemies[0], speed=0),))


def spawn_times(sim):
    return [e['time'] for e in sim.trace.events if e['event'] == 'ENEMY_SPAWN']


def test_original_prts_spawn_timestamps_unchanged(real_stage):
    sim = Simulator(real_stage, trace=True)
    sim.run()
    assert spawn_times(sim) == [5, 16, 18, 29, 29.7, 34, 45, 45.7, 52, 52.7, 63.7]
    assert sim.current_time == pytest.approx(85.65)
    assert sim.state.stage_events.phase == 'complete'


def test_parser_retains_parallel_groups_and_graph_ids(simple):
    stage = event_stage(simple, wave(fragment(spawn(3, 1, 2), spawn(2, 2, 1)),
                                     fragment(spawn(), delay=3)))
    first, second = stage.waves[0].fragments
    assert first.actions[0].event_id == 'wave_0_fragment_0_action_0'
    assert first.actions[0].count == 3
    assert second.actions[0].index == 2
    # Different groups in a fragment are concurrent, not serial actions.
    assert [s.time for s in stage.spawns] == [1, 2, 3, 3, 5, 8]
    sim = Simulator(stage)
    sim.run_until(0)
    graph = future_event_state(sim.state)
    assert graph['dependencies'] == [(0, 2, 'action_complete'), (1, 2, 'action_complete')]
    assert graph['events'][2]['dependencies'] == [0, 1]
    assert graph['events'][2]['estimated_relative_time'] == 8


def test_last_enemy_death_releases_next_wave_and_updates_future(simple, operator):
    stage = event_stage(simple, wave(fragment(spawn()), post=1),
                        wave(fragment(spawn(delay=0.5), delay=1), pre=2))
    sim = Simulator(stage, [operator], trace=True)
    sim.deploy('op', (0, 0))
    sim.run_until(0)
    graph = future_event_state(sim.state)
    assert graph['events'][1]['estimated_relative_time'] is None
    assert not graph['events'][1]['time_known']
    assert (0, 1, 'wave_clear') in graph['dependencies']
    assert graph['progression']['blocking_enemy_count'] == 1
    assert graph['progression']['current_event_id'].startswith('wave_0_')
    assert graph['progression']['remaining_enemies'] == 2
    assert graph['progression']['unspawned_enemies'] == 1
    sim.run_until(3)
    assert sim.state.spawned == 1
    sim.run_until(3.25)  # Fourth normal attack kills the first enemy.
    assert sim.state.killed == 1
    assert sim.state.stage_events.phase == 'post_delay'
    graph = future_event_state(sim.state)
    assert graph['events'][1]['time_known']
    death_time = next(e['time'] for e in sim.trace.events if e['event'] == 'DEATH')
    assert graph['events'][1]['estimated_relative_time'] == pytest.approx(death_time + 4.5 - sim.current_time)
    sim.run_until(8)
    assert spawn_times(sim) == pytest.approx([0, death_time + 4.5])


def test_holding_last_enemy_delays_wave(simple, operator):
    stage = event_stage(simple, wave(fragment(spawn())), wave(fragment(spawn())))
    idle = Simulator(stage, trace=True)
    idle.run_until(20)
    assert idle.state.spawned == 1 and idle.state.stage_events.wave_index == 0
    assert future_event_state(idle.state)['progression']['blocking_enemy_count'] == 1
    fight = Simulator(stage, [operator], trace=True)
    fight.deploy('op', (0, 0))
    fight.run_until(4)
    assert fight.state.spawned == 2
    death_time = next(e['time'] for e in fight.trace.events if e['event'] == 'DEATH')
    assert spawn_times(fight) == pytest.approx([0, death_time])


def test_block_fragment_waits_for_its_spawned_units(simple, operator):
    stage = event_stage(simple, wave(fragment(spawn(block=True)),
                                     fragment(spawn(delay=1), delay=2)))
    sim = Simulator(stage, [operator], trace=True)
    sim.deploy('op', (0, 0))
    sim.run_until(0)
    graph = future_event_state(sim.state)
    assert graph['dependencies'] == [(0, 1, 'fragment_clear')]
    assert graph['events'][1]['estimated_relative_time'] is None
    assert graph['progression']['current_fragment'] == 0
    sim.run_until(3.25)
    assert sim.state.stage_events.fragment_index == 1
    sim.run_until(7)
    death_time = next(e['time'] for e in sim.trace.events if e['event'] == 'DEATH')
    assert spawn_times(sim) == pytest.approx([0, death_time + 3])


def test_nonblocking_wave_enemy_does_not_prevent_progression(simple):
    stage = event_stage(simple, wave(fragment(spawn(nonblocking_wave=True))),
                        wave(fragment(spawn(delay=1))))
    sim = Simulator(stage, trace=True)
    sim.run_until(1)
    assert sim.state.spawned == 2
    assert sim.state.enemies['enemy_0000'].alive
    assert sim.state.stage_events.wave_index == 1
    # The first enemy still participates in battle completion and life loss.
    assert not sim.state.done
    assert future_event_state(sim.state)['progression']['blocking_enemy_ids'] == ['enemy_0001']


def test_wave_timeout_is_after_dispatch_and_before_postdelay(simple):
    stage = event_stage(simple, wave(fragment(spawn(delay=2)), timeout=3, post=1),
                        wave(fragment(spawn()), pre=1))
    sim = Simulator(stage, trace=True)
    sim.run_until(6)
    assert sim.state.spawned == 1
    assert sim.state.enemies['enemy_0000'].alive
    sim.run_until(7)
    assert spawn_times(sim) == [2, 7]
    opened = [e for e in sim.trace.events if e['event'] == 'WAVE_GATE_OPEN']
    assert opened[0]['time'] == 5 and opened[0]['reason'] == 'timeout'


def test_empty_parsed_wave_retains_delays(simple):
    stage = event_stage(simple, wave(pre=2, post=1), wave(fragment(spawn()), pre=3))
    sim = Simulator(stage, trace=True)
    sim.run_until(5)
    assert sim.state.spawned == 0 and not sim.state.done
    sim.run_until(6)
    assert spawn_times(sim) == [6]


@pytest.mark.parametrize('change', ['negative_post', 'external', 'random', 'blocking_story', 'predefined'])
def test_unknown_stage_controls_are_not_fabricated(simple, change):
    data = wave(fragment(spawn()))
    action = data['fragments'][0]['actions'][0]
    if change == 'negative_post': data['postDelay'] = -1
    if change == 'external': action['managedByScheduler'] = False
    if change == 'random': action['randomSpawnGroupKey'] = 'g1'
    if change == 'blocking_story': action.update(actionType='STORY', blockFragment=True)
    if change == 'predefined': action['actionType'] = 'ACTIVATE_PREDEFINED'
    with pytest.raises(NotImplementedError):
        parse_event_waves([data], simple.routes)


def test_stage_loader_accepts_supported_multiple_waves_and_block_spawn():
    raw = json.loads((DATA/'level_main_00-01.json').read_text())
    raw['waves'] *= 2
    raw['waves'][0]['fragments'][1]['actions'][1]['blockFragment'] = True
    stage = StageLoader(DATA/'enemy_database.json').parse(raw)
    assert len(stage.waves) == 2 and len(stage.spawns) == 22
    assert stage.waves[0].fragments[1].actions[1].block_fragment
    assert any('official boundary timing unverified' in a for a in stage.assumptions)


def test_wait_duration_serialization_mask_and_resource_recovery(simple, operator):
    assert 'wait_seconds' not in WAIT.to_dict()
    hold = Action('WAIT', wait_seconds=2)
    assert Action.from_dict(hold.to_dict()) == hold
    stage = event_stage(simple, wave(fragment(spawn(delay=100))))
    stage = replace(stage, initial_cost=20)
    env = ArknightsEnv(stage, [operator, replace(operator, id='second')])
    state = env.reset()
    state = env.step(state, Action('DEPLOY', 'op', (1, 0), 'LEFT'))
    state = env.step(state, Action('DEPLOY', 'second', (2, 0), 'LEFT'))
    state = env.step(state, Action('RETREAT', 'second'))
    before = env.state_key(state)
    assert hold in env.legal_actions(state)
    child = env.step(state, hold)
    assert child.game.current_time == 2
    assert child.game.dp == pytest.approx(state.game.dp + 2)
    assert child.game.operators['op'].skill.sp == pytest.approx(2)
    assert child.game.redeploy_at['second'] - child.game.current_time == 3
    assert child.last_decision.reasons == ('PLANNED_WAIT_END',)
    assert env.state_key(state) == before
    poor = ArknightsEnv(replace(stage, initial_cost=0), [operator])
    assert poor.legal_actions(poor.reset()) == (WAIT,)


@pytest.mark.parametrize('kwargs', [dict(wait_seconds=0), dict(wait_seconds=0.1),
                                   dict(wait_seconds=True), dict(wait_seconds=float('nan'))])
def test_invalid_wait_durations(kwargs):
    with pytest.raises(ValueError): Action('WAIT', **kwargs)


def test_event_clone_fastforward_and_action_sequence_determinism(simple, operator):
    stage = event_stage(simple, wave(fragment(spawn(block=True)), fragment(spawn(), delay=2)),
                        wave(fragment(spawn())))
    stage = replace(stage, initial_cost=30)
    env = ArknightsEnv(stage, [operator], trace=True)
    a, b = env.reset(seed=17), env.reset(seed=17)
    initial = env.state_key(a)
    prediction = env.get_next_decision_event(a)
    assert env.state_key(a) == initial
    assert prediction == env.fast_forward(a).last_decision
    for action in [Action('DEPLOY','op',(0,0),'RIGHT'), Action('WAIT',wait_seconds=2), WAIT,
                   Action('WAIT',wait_seconds=5)]:
        assert action in env.legal_actions(a)
        a, b = env.step(a, action), env.step(b, action)
        assert a.game.snapshot() == b.game.snapshot()
        assert env.future_event_state(a) == env.future_event_state(b)
        assert env.get_result(a) == env.result(b)
    c = env.clone_state(a)
    c.game.stage_events.actions[0].unit_ids.append('child_only')
    assert 'child_only' not in a.game.stage_events.actions[0].unit_ids
    assert c.game.stage is a.game.stage
    x, y = env.fast_forward(a, 2), env.fast_forward(a, 2)
    assert x.game.snapshot() == y.game.snapshot()


def test_event_observation_is_not_a_mutable_view(simple):
    env = ArknightsEnv(event_stage(simple,wave(fragment(spawn())),wave(fragment(spawn()))), [])
    state = env.reset()
    before = env.state_key(state)
    graph = env.future_event_state(state)
    graph['events'][0]['enemy_stats']['hp'] = -1
    graph['events'][0]['route_geometry']['start'][0] = -99
    graph['progression']['blocking_enemy_ids'].append('fake')
    assert env.state_key(state) == before


def test_blocked_future_lower_bound_is_relative_to_now(simple):
    stage = event_stage(simple, wave(fragment(spawn(block=True)), fragment(spawn(), delay=5)))
    sim = Simulator(stage)
    sim.run_until(10)
    graph = future_event_state(sim.state)
    assert graph['events'][1]['estimated_relative_time'] is None
    assert graph['events'][1]['earliest_relative_time'] == 5
    assert graph['progression']['next_boundary_relative_time'] is None
    assert graph['events'][0]['alive_enemy_ids'] == ['enemy_0000']


def test_wave_dependencies_distinguish_hard_clear_timeout_and_nonblocking(simple):
    for timeout, expected in [(-1, 'wave_clear'), (2, 'wave_clear_or_timeout'), (0, None)]:
        stage = event_stage(simple, wave(fragment(spawn()), timeout=timeout), wave(fragment(spawn(delay=1))))
        env = ArknightsEnv(stage, [])
        graph = env.future_event_state(env.reset())
        edges = [relation for source, target, relation in graph['dependencies'] if (source,target)==(0,1)]
        if expected:
            assert expected in edges
            assert graph['progression']['next_trigger'] == ('enemy_clear' if timeout == -1 else 'enemy_clear_or_timeout')
        else:
            assert not any('clear' in relation for relation in edges)
    # Fragment blocking remains mandatory even if its units don't block the
    # wave and the wave timeout is zero. The graph must preserve this condition.
    stage = event_stage(simple, wave(fragment(spawn(block=True, nonblocking_wave=True)), timeout=0),
                        wave(fragment(spawn())))
    env = ArknightsEnv(stage, [])
    initial = env.reset()
    graph = env.future_event_state(initial)
    assert graph['dependencies'] == [(0, 1, 'fragment_clear')]
    assert graph['events'][1]['trigger_condition'] == 'fragment_clear'
    delayed = env.fast_forward(initial, 10)
    assert delayed.game.spawned == 1


def test_passive_future_action_retains_delay_in_observation(simple):
    passive = dict(actionType='STORY', key='ignored_ui', count=1, preDelay=8, interval=0)
    stage = event_stage(simple, wave(fragment(spawn(delay=1)), fragment(passive, delay=2),
                                     fragment(spawn(), delay=3)))
    env = ArknightsEnv(stage, [])
    graph = env.future_event_state(env.reset())
    assert graph['events'][1]['spawn_count'] == 0
    assert graph['events'][1]['estimated_relative_time'] == 11
    assert graph['events'][2]['estimated_relative_time'] == 14


def test_fast_forward_stops_at_battle_end_or_horizon_without_mutating_parent(simple, operator):
    stage = event_stage(simple, wave(fragment(spawn())))
    env = ArknightsEnv(stage, [replace(operator, atk=1000)], trace=True)
    state = env.step(env.reset(seed=9), Action('DEPLOY', 'op', (0,0), 'RIGHT'))
    before = env.state_key(state)
    a, b = env.fast_forward(state, 5), env.fast_forward(state, 5)
    assert a.game.done and a.game.current_time < 5
    assert a.game.snapshot() == b.game.snapshot()
    assert a.trace == b.trace
    assert a.last_decision.reasons == ('BATTLE_END',)
    assert env.future_event_state(a)['progression']['next_trigger'] == 'battle_complete'
    assert env.state_key(state) == before
    with pytest.raises(ValueError): env.fast_forward(a, 1)
    limited = ArknightsEnv(stage, [], horizon=1)
    original = limited.reset()
    child = limited.fast_forward(original, 5)
    assert child.game.current_time == 1 and not child.game.done
    assert child.last_decision.reasons == ('TIME_LIMIT',)
    assert limited.get_next_decision_event(child).reasons == ('TIME_LIMIT',)
    assert original.game.current_time == 0
