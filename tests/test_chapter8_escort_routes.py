"""Escort objectives and route visibility; client-frame parity remains separate."""
import json
from dataclasses import replace
import pytest
from arknights_sim import Simulator
from arknights_sim.data.models import EnemyData,SpawnData,WaypointData
from arknights_sim.data.map_catalog import MapCatalog
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.environment import ArknightsEnv
from arknights_sim.core.stage_events import blockers
from conftest import DATA
from test_stage_events import event_stage,wave,fragment,spawn


def escort_stage(simple):
    escort=EnemyData('civilian','Civilian',50,0,0,0,1,0,life_cost=0,block_weight=0,damage_type='NONE',faction='ESCORT',death_life_cost=1)
    return replace(simple,enemies=(replace(simple.enemies[0],speed=0),escort),
        waves=(replace(simple.waves[0],spawns=(SpawnData(0,'civilian',0),SpawnData(20,'e',0))),))


def test_escort_cannot_be_friendly_target_or_blocked_and_arrival_is_not_leak(simple,operator):
    stage=escort_stage(simple);sim=Simulator(stage,[operator],trace=True);sim.deploy('op',(2,0));sim.run_until(10)
    escort=sim.state.enemies['enemy_0000']
    assert not escort.alive and escort.hp==50 and escort.blocked_by is None
    assert sim.state.escorts_saved==1 and sim.state.escaped==0 and sim.state.killed==0
    assert sim.state.life==stage.life


def test_escort_death_loses_life_but_no_kill_credit_and_no_three_star(simple):
    stage=escort_stage(simple)
    env=ArknightsEnv(stage,[]);state=env.reset();sim=Simulator.from_state(state.game,trace=True)
    sim.run_until(0);escort=sim.state.enemies['enemy_0000'];sim._remove_enemy(escort)
    assert sim.state.life==stage.life-1 and sim.state.killed==0 and sim.state.escorts_dead==1
    sim.state.done=True;sim.state.result='WIN';sim.state.killed=stage.hostile_count
    result=env.result(state)
    assert not result.success and not result.protection_success
    assert result.escorts_dead==1 and result.total_enemies==1 and result.leaks==0


def test_civilian_flag_does_not_hold_wave_clear(simple):
    data=spawn();data['key']='civilian';data['isUnharmfulAndAlwaysCountAsKilled']=True
    # Parser accepts only the audited real id, then use that matching data.
    data['key']='enemy_3001_upeopl'
    base=escort_stage(simple)
    civilian=replace(base.enemies[1],id=data['key'],speed=0)
    stage=event_stage(simple,wave(fragment(data)),wave(fragment(spawn(delay=1))))
    stage=replace(stage,enemies=(*stage.enemies,civilian))
    sim=Simulator(stage);sim.run_until(1)
    assert sim.state.spawned==2 and sim.state.enemies['enemy_0000'].alive
    assert 'enemy_0000' not in blockers(sim.state)


def test_teleport_hides_without_death_and_reappears_at_destination(simple,operator):
    route=replace(simple.routes[0],waypoints=(WaypointData('DISAPPEAR',(99,99)),WaypointData('WAIT_FOR_SECONDS',(99,99),2),WaypointData('APPEAR_AT_POS',(5,0))))
    stage=replace(simple,routes=(route,))
    sim=Simulator(stage,[operator],trace=True);sim.deploy('op',(0,0));sim.run_until(1)
    enemy=sim.state.enemies['enemy_0000']
    assert enemy.hidden and enemy.alive and enemy.blocked_by is None and enemy.hp==100
    clone=sim.clone();sim.run_until(2.5);clone.run_until(2.5)
    assert not enemy.hidden and enemy.position[0]==pytest.approx(5.5)
    assert clone.state.stable_hash()==sim.state.stable_hash()


def test_flying_crosses_impassable_tiles_and_cannot_be_melee_blocked(simple,operator):
    from arknights_sim.data.models import MapData
    tiles=list(simple.map.rows[0]);tiles[3]=replace(tiles[3],passable='FLY_ONLY')
    stage=replace(simple,map=MapData((tuple(tiles),)),routes=(replace(simple.routes[0],motion='FLY'),),enemies=(replace(simple.enemies[0],motion='FLY'),))
    sim=Simulator(stage,[operator],trace=True);sim.deploy('op',(2,0));sim.run_until(4)
    e=sim.state.enemies['enemy_0000']
    assert e.hp==100 and e.blocked_by is None and e.position[0]==pytest.approx(4)


def test_real_r82_has_protection_count_and_can_finish():
    catalog=MapCatalog(DATA);stage=StageLoader(catalog.enemy_path).load(catalog.path(catalog.resolve('R8-2')))
    assert stage.escort_count>0 and stage.hostile_count+stage.escort_count==len(stage.spawns)
    sim=Simulator(stage,trace=True);sim.run()
    assert sim.state.done
    assert sim.state.killed==0


def test_projectile_survives_source_death_and_damages_old_position(simple,operator):
    enemy=replace(simple.enemies[0],speed=0,attack_range=5,atk=100,projectile_delay=3,projectile_radius=1.2)
    stage=replace(simple,enemies=(enemy,))
    sim=Simulator(stage,[replace(operator,atk=0)],trace=True);u=sim.deploy('op',(2,0));sim.run_until(.5)
    assert u.hp==1000
    # Keep the stage unfinished while testing a projectile already in flight.
    source=sim.state.enemies['enemy_0000'];source.alive=False
    sim.state.queue.push(10,'SPAWN',SpawnData(10,'e',0))
    sim.state.spawned=0
    sim.run_until(3.5)
    assert u.hp==910


def test_rager_periodic_damage_uses_spawn_clock_and_counts_death(simple):
    enemy=replace(simple.enemies[0],speed=0,hp=100,self_damage_per_second=60)
    stage=replace(simple,enemies=(enemy,),waves=(replace(simple.waves[0],spawns=(SpawnData(.5,'e',0),)),))
    sim=Simulator(stage,trace=True);sim.run_until(2.5)
    hits=[e['time'] for e in sim.trace.events if e['event']=='SELF_DAMAGE']
    assert hits==[1.5,2.5] and sim.state.killed==1 and sim.state.done
