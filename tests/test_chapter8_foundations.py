"""Foundation contracts, not a certification of chapter-eight battle fidelity."""
from dataclasses import replace
from copy import deepcopy
import json
import pytest
from conftest import DATA, ROOT
from arknights_sim import Simulator
from arknights_sim.data.models import WaypointData
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from arknights_sim.data.trust import trust_bonuses
from arknights_sim.map.route import compile_route
from arknights_sim.core.stage_events import route_wait_deadline
from accounts.battle_inputs import load_account_operator
from accounts.snapshot import load_snapshot
from test_stage_events import event_stage, wave, fragment, spawn


@pytest.fixture(scope='module')
def loader():
    return OperatorLoader(DATA/'character_table.json', DATA/'range_table.json', SkillLoader(DATA/'skill_table.json'))


@pytest.mark.parametrize('kind', ['WAIT_CURRENT_WAVE_TIME', 'WAIT_CURRENT_FRAGMENT_TIME'])
def test_relative_wait_uses_activation_clock_and_expired_wait_does_not_restart(simple, kind):
    stage = event_stage(simple, wave(fragment(spawn()), pre=5))
    sim = Simulator(stage)
    sim.run_until(7)
    checkpoint = WaypointData(kind, (99,99), 10)
    assert route_wait_deadline(sim.state, checkpoint, 7) == 15
    assert route_wait_deadline(sim.state, checkpoint, 18) == 18
    path = compile_route(stage.map, replace(stage.routes[0], waypoints=(checkpoint,)))
    assert path[0] == checkpoint  # wait position is not a movement destination


def test_wave_and_fragment_clocks_are_distinct(simple):
    stage = event_stage(simple, wave(fragment(spawn()), fragment(spawn(), delay=3), pre=5))
    sim = Simulator(stage); sim.run_until(8)
    assert route_wait_deadline(sim.state, WaypointData('WAIT_CURRENT_WAVE_TIME',(0,0),10),8)==15
    assert route_wait_deadline(sim.state, WaypointData('WAIT_CURRENT_FRAGMENT_TIME',(0,0),10),8)==18


def test_relative_wait_movement_clone_and_replay(simple):
    checkpoint = WaypointData('WAIT_CURRENT_WAVE_TIME', (99,99), 10)
    stage = event_stage(simple, wave(fragment(spawn(delay=2)), pre=5))
    stage = replace(stage, routes=(replace(stage.routes[0], waypoints=(checkpoint,)),),
                    enemies=(replace(stage.enemies[0], speed=1),))
    sim = Simulator(stage, trace=True); sim.run_until(9)
    assert sim.state.enemies['enemy_0000'].position == (0,0)
    assert sim.state.enemies['enemy_0000'].wait_until == 15
    parent = sim.state.stable_hash(); clone=sim.clone()
    clone.run_until(16)
    assert sim.state.stable_hash()==parent
    fresh = Simulator(stage, trace=True); fresh.run_until(16)
    assert fresh.state.stable_hash()==clone.state.stable_hash()
    assert fresh.trace.events==clone.trace.events
    assert clone.state.enemies['enemy_0000'].position[0]==pytest.approx(1)


@pytest.mark.parametrize('bad', [-1,201,True,float('nan')])
def test_trust_rejects_invalid_percent(loader,bad):
    with pytest.raises(ValueError):loader.load('char_4145_ulpia',trust=bad)


def test_trust_uses_battle_phase_not_raw_percent_and_caps(loader):
    raw=loader.chars['char_4145_ulpia']
    assert trust_bonuses(raw,loader.favor,50)=={'maxHp':250,'atk':40}
    assert trust_bonuses(raw,loader.favor,100)=={'maxHp':500,'atk':80}
    assert trust_bonuses(raw,loader.favor,200)=={'maxHp':500,'atk':80}
    base=loader.load('char_4145_ulpia'); full=loader.load('char_4145_ulpia',trust=200)
    assert (full.hp-base.hp,full.atk-base.atk)==(500,80)
    assert full.trust==200


def test_confirmed_account_binds_actual_ulpianus_and_rejects_tampering(loader):
    path=ROOT/'data/accounts/snapshots/82d3e5dbea0c91b94275c195f5aac466f19115a81e86af65d1b0204ad5e49408.json'
    snapshot=load_snapshot(path,loader)
    row=next(r for r in snapshot['operators'] if r['operator_id']=='char_4145_ulpia')
    op=load_account_operator(snapshot,loader,row['operator_id'])
    assert (op.elite,op.level,op.potential,op.trust)==(row['elite'],row['level'],0,200)
    assert op.selected_skill_id==row['selected_skill_id']
    broken=deepcopy(snapshot);broken['operators'][0]['level']=90
    with pytest.raises(ValueError):load_account_operator(broken,loader,row['operator_id'])
    with pytest.raises(ValueError):load_account_operator(snapshot,loader,'char_002_amiya')
    with pytest.raises(ValueError):load_account_operator(snapshot,loader,row['operator_id'],'not_a_skill')


def test_roadblock_stats_priority_destruction_and_no_kill_credit(simple, operator):
    from arknights_sim.mechanics.definitions import DeviceData
    from arknights_sim.mechanics import runtime
    stage=replace(simple,devices=(DeviceData('obstacle','roadblock',(1,0),hp=100,defense=20,resistance=20,block_count=0),))
    # Keep a live enemy far away to prevent terminal completion during the test.
    stage=replace(stage,enemies=(replace(stage.enemies[0],speed=0),),
                  routes=(replace(stage.routes[0],start=(6,0)),))
    sim=Simulator(stage,[replace(operator,atk=200)],trace=True)
    sim.deploy('op',(0,0));sim.run_until(.25)
    assert not sim.state.devices['obstacle'].alive
    assert sim.state.killed==0
    assert (1,0) not in runtime.solid_positions(sim.state)
    assert any(e['event']=='DEVICE_DESTROY' for e in sim.trace.events)
    with pytest.raises(ValueError):runtime.remove(sim,'obstacle')


def test_roadblock_is_not_selected_over_normal_enemy(simple, operator):
    from arknights_sim.mechanics.definitions import DeviceData
    stage=replace(simple,map=replace(simple.map,rows=simple.map.rows*2),
                  devices=(DeviceData('obstacle','roadblock',(1,0),hp=8000),))
    sim=Simulator(stage,[operator],trace=True);sim.deploy('op',(0,0));sim.run_until(.25)
    assert sim.state.devices['obstacle'].hp==8000
    attacks=[e for e in sim.trace.events if e['event']=='ATTACK_START' and e['attacker']=='op']
    assert attacks[0]['target']=='enemy_0000'


def test_real_r81_loads_and_roadblock_fields_match_source():
    from arknights_sim.data.map_catalog import MapCatalog
    from arknights_sim.data.stage_loader import StageLoader
    catalog=MapCatalog(DATA)
    stage=StageLoader(catalog.enemy_path).load(catalog.path(catalog.resolve('R8-1')))
    assert len(stage.spawns)==56
    sim=Simulator(stage,trace=True)
    assert len(sim.state.devices)==5
    assert all((d.hp,d.data.defense,d.data.resistance,d.data.block_count)==(8000,200,20,0) for d in sim.state.devices.values())
    sim.run()
    assert sim.state.done and sim.state.result=='LOSS'  # no fabricated clear
    assert sim.state.escaped==5
