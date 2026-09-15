"""Counterfactual plans in the real engine, not claims about untrained policy skill."""
from dataclasses import replace

import pytest

from arknights_sim.data.event_graph import parse_event_waves
from arknights_sim.data.models import SkillData
from arknights_sim.environment import Action, ArknightsEnv


def spawn(key, delay=0, route=0):
    return dict(actionType='SPAWN', key=key, count=1, preDelay=delay,
                interval=0, routeIndex=route)


def wave(*actions):
    return dict(preDelay=0, postDelay=0, maxTimeWaitingForNextWave=-1,
                fragments=[dict(preDelay=0, actions=list(actions))])


def arena(simple, waves, *, initial_cost=10):
    enemies = (replace(simple.enemies[0], id='weak', name='Weak', hp=100, atk=0, speed=0),
               replace(simple.enemies[0], id='elite', name='Elite', hp=2000, atk=100, speed=0))
    return replace(simple, id='planning_fixture', initial_cost=initial_cost,
                   waves=parse_event_waves(waves, simple.routes), enemies=enemies)


def hold(env, state, seconds):
    return env.step(state, Action('WAIT', wait_seconds=seconds))


def test_case1_delay_final_kill_delays_next_wave_and_recovers_resources(simple, operator):
    stage = arena(simple, [wave(spawn('weak')), wave(spawn('elite'))])
    killer = replace(operator, id='killer', cost=1, atk=200, skill=None)
    battery = replace(operator, id='battery', cost=0, atk=0, block_count=0,
                      skill=SkillData('reserve', 10, 0, 3, 2))
    env = ArknightsEnv(stage, [killer, battery], trace=True, horizon=30)
    initial = env.step(env.reset(), Action('DEPLOY', 'battery', (7, 0), 'RIGHT'))
    immediate = env.step(initial, Action('DEPLOY', 'killer', (0, 0), 'RIGHT'))
    immediate = hold(env, immediate, 1)
    delayed = hold(env, initial, 5)
    assert delayed.game.spawned == 1 and delayed.game.killed == 0
    future = env.future_event_state(delayed)
    boss = next(row for row in future['events'] if row['enemy_id'] == 'elite')
    assert boss['remaining_count'] == 1 and not boss['time_known']
    assert future['progression']['blocking_enemy_count'] == 1
    assert delayed.game.dp > immediate.game.dp
    assert delayed.game.operators['battery'].skill.sp > immediate.game.operators['battery'].skill.sp
    delayed = env.step(delayed, Action('DEPLOY', 'killer', (0, 0), 'RIGHT'))
    delayed = hold(env, delayed, 1)
    assert delayed.game.stage_events.wave_started_at[1] - immediate.game.stage_events.wave_started_at[1] == pytest.approx(5)
    # WAIT itself did not disable an operator's auto-attacks: the delayed branch
    # delayed the damage dealer's deployment, an actual legal game operation.


def test_case2_preserve_ready_skill_for_known_future_elite(simple, operator):
    stage = arena(simple, [wave(spawn('weak'), spawn('elite', delay=5))])
    unit = replace(operator, cost=0, atk=0, block_count=0,
                   skill=SkillData('burst', 10, 10, 2, 3))
    env = ArknightsEnv(stage, [unit], horizon=30)
    initial = env.step(env.reset(), Action('DEPLOY', unit.id, (7, 0), 'RIGHT'))
    future = next(r for r in env.future_event_state(initial)['events'] if r['enemy_id'] == 'elite')
    assert future['estimated_relative_time'] == pytest.approx(5)
    spent = env.step(initial, Action('ACTIVATE_SKILL', unit.id))
    spent, saved = hold(env, spent, 5), hold(env, initial, 5)
    assert Action('ACTIVATE_SKILL', unit.id) not in env.legal_actions(spent)
    assert Action('ACTIVATE_SKILL', unit.id) in env.legal_actions(saved)
    assert saved.game.spawned == spent.game.spawned == 2


def test_case3_early_deployment_can_charge_before_future_threat(simple, operator):
    stage = arena(simple, [wave(spawn('elite', delay=5))])
    unit = replace(operator, cost=0, atk=0, skill=SkillData('burst', 4, 0, 2, 3))
    env = ArknightsEnv(stage, [unit], horizon=30)
    initial = env.reset()
    assert initial.game.spawned == 0
    future = env.future_event_state(initial)['events'][0]
    assert future['enemy_id'] == 'elite' and future['route_id'] == 0
    early = env.step(initial, Action('DEPLOY', unit.id, (7, 0), 'RIGHT'))
    early = hold(env, early, 5)
    late = hold(env, initial, 5)
    late = env.step(late, Action('DEPLOY', unit.id, (7, 0), 'RIGHT'))
    assert Action('ACTIVATE_SKILL', unit.id) in env.legal_actions(early)
    assert Action('ACTIVATE_SKILL', unit.id) not in env.legal_actions(late)


def test_case4_early_retreat_makes_redeployment_ready_for_future_event(simple, operator):
    stage = arena(simple, [wave(spawn('elite', delay=12))], initial_cost=99)
    unit = replace(operator, cost=0, redeploy=10, atk=0)
    env = ArknightsEnv(stage, [unit], horizon=30)
    initial = env.step(env.reset(), Action('DEPLOY', unit.id, (7, 0), 'RIGHT'))
    early = env.step(initial, Action('RETREAT', unit.id))
    early = env.fast_forward(early, 12)
    late = hold(env, initial, 5)
    late = env.step(late, Action('RETREAT', unit.id))
    late = env.fast_forward(late, 7)
    assert early.game.current_time == late.game.current_time == pytest.approx(12)
    assert any(a.type == 'DEPLOY' for a in env.legal_actions(early))
    assert not any(a.type == 'DEPLOY' for a in env.legal_actions(late))


def test_case5_wait_for_affordable_operator_before_event(simple, operator):
    stage = arena(simple, [wave(spawn('elite', delay=5))], initial_cost=5)
    weak = replace(operator, id='weak_op', cost=5, atk=10)
    suitable = replace(operator, id='suitable_op', cost=10, atk=200)
    env = ArknightsEnv(stage, [weak, suitable], horizon=30)
    initial = env.reset()
    assert any(a.type == 'DEPLOY' and a.operator_id == weak.id for a in env.legal_actions(initial))
    assert not any(a.type == 'DEPLOY' and a.operator_id == suitable.id for a in env.legal_actions(initial))
    spend = env.step(initial, Action('DEPLOY', weak.id, (7, 0), 'RIGHT'))
    spend, save = hold(env, spend, 5), hold(env, initial, 5)
    assert any(a.type == 'DEPLOY' and a.operator_id == suitable.id for a in env.legal_actions(save))
    assert not any(a.type == 'DEPLOY' and a.operator_id == suitable.id for a in env.legal_actions(spend))
