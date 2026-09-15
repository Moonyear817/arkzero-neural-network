from dataclasses import replace
from pathlib import Path
import pytest,torch
from arknights_sim.data.models import *
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.core.simulator import Simulator
from arknights_sim.environment import ArknightsEnv,Action
from arknights_sim.mechanics.definitions import DeviceData
from arknights_sim.mechanics import runtime as m

ROOT=Path(__file__).resolve().parents[1]


def stage(devices=()):
    tile=TileData('tile_road','LOWLAND','MELEE','ALL')
    grid=MapData(tuple(tuple(tile for _ in range(5)) for _ in range(3)))
    enemy=EnemyData('dog','Dog',1000,100,0,0,1,1)
    return StageData('test',grid,(RouteData((4,1),(0,1),diagonal=False),),(WaveData(0,0,(0,),(SpawnData(5,'dog',0),)),),(enemy,),50,99,1,1,20,8,devices=devices)


def op(key='op',block=1):
    return OperatorData(key,key,1000,100,0,0,1,block,5,10,((0,0),(1,0)))


def test_crate_reroutes_preserves_parent_and_blocks_full_closure():
    s=stage((DeviceData('crate','crate',count=3),));env=ArknightsEnv(stage=s,squad=())
    original=env.reset();child=env.step(original,Action('PLACE_DEVICE','crate',(2,1)))
    assert original.game.device_inventory['crate']==3 and child.game.device_inventory['crate']==2
    assert (2,1) not in [w.position for w in m.paths_for(child.game)[0]]
    assert child.game.dp==45 and child.game.stable_hash()!=original.game.stable_hash()
    sim=Simulator.from_state(child.game);sim.run_until(5)
    m.place(sim,'crate',(2,0));sim.run_until(10)
    assert not m.can_place(sim.state,(2,2))
    assert not any(a.tile==(2,2) for a in env.legal_actions(child) if a.type=='PLACE_DEVICE')


def test_detector_ready_active_expires_and_reveals_invisible():
    s=stage((DeviceData('sensor','sensor',(2,2),charge=15,duration=20),))
    s=replace(s,enemies=(replace(s.enemies[0],invisible=True,speed=.01),))
    sim=Simulator(s);sim.run_until(5);enemy=next(iter(sim.state.enemies.values()))
    assert not m.revealed(sim.state,enemy)
    with pytest.raises(ValueError):m.activate(sim,'sensor')
    sim.run_until(15);m.activate(sim,'sensor');assert m.revealed(sim.state,enemy)
    sim.run_until(35);assert not m.revealed(sim.state,enemy)
    assert sim.state.devices['sensor'].sp==0
    sim.run_until(50);assert sim.state.devices['sensor'].sp==15


def test_gravity_pressure_priority_release_and_weight_speed():
    s=stage((DeviceData('gravity','gravity',(0,0),direction=1),))
    rows=[list(r) for r in s.map.rows]
    rows[0][1]=replace(rows[0][1],key='tile_grvtybtn',blackboard=(('source_direction',2),('activate_block_cnt',1)))
    rows[0][3]=replace(rows[0][3],key='tile_grvtybtn',blackboard=(('source_direction',0),('activate_block_cnt',1)))
    s=replace(s,map=MapData(tuple(tuple(r) for r in rows)))
    sim=Simulator(s,(op('a'),op('b')));a=sim.deploy('a',(1,0),3)
    assert sim.state.gravity==3 and m.operator_as(sim.state,a)==25
    sim.deploy('b',(3,0),1);assert sim.state.gravity==3
    sim.retreat('a');assert sim.state.gravity==1
    sim.run_until(5);e=next(iter(sim.state.enemies.values()));e.data=replace(e.data,weight=3)
    assert m.gravity_speed(sim.state,e,(0,1))==2.4
    assert m.gravity_speed(sim.state,e,(0,-1))==.36


def test_wind_depends_on_direction_and_skill_level():
    s=stage((DeviceData('wind','wind',(0,1),direction=0),));sim=Simulator(s,(op(),))
    a=sim.deploy('op',(2,1),0);assert m.operator_attack(sim.state,a)==1.3
    a.direction=2;assert m.operator_attack(sim.state,a)==.7
    sim.run_until(5);e=next(iter(sim.state.enemies.values()));e.position=(2,1)
    assert m.speed_factor(sim.state,e,(-1,0))==.5
    assert m.speed_factor(sim.state,e,(1,0))==1.8


def test_3_7_actual_map_challenge_and_neural_device_actions():
    loader=StageLoader(ROOT/'data/maps/levels/enemydata/enemy_database.json')
    raw=ROOT/'data/maps/levels/obt/main/level_main_03-07.json'
    normal=loader.load(raw);hard=loader.load(raw,'FOUR_STAR')
    assert len(normal.spawns)==61 and hard.life==1
    assert next(d.count for d in normal.devices if d.kind=='crate')==5
    assert next(d.count for d in hard.devices if d.kind=='crate')==3
    assert hard.enemies[0].hp==pytest.approx(normal.enemies[0].hp*1.2)
    env=ArknightsEnv(stage=hard,squad=());state=env.reset()
    from network import StateEncoder,ActionEncoder,PolicyValueNetwork
    torch.set_num_threads(1);net=PolicyValueNetwork()
    actions=env.legal_actions(state);encoded=StateEncoder()(state);features=ActionEncoder()(state,actions)
    logits,value=net(encoded,features);assert torch.isfinite(logits).all()
    assert any(a.type=='PLACE_DEVICE' for a in actions)
    target=next(i for i,a in enumerate(actions) if a.type=='PLACE_DEVICE')
    loss=-logits.log_softmax(0)[target]+value.square();loss.backward()
    assert net.mechanics_input.weight.grad.abs().sum()>0
    child=env.step(state,actions[target]);assert not torch.equal(encoded['mechanics'],StateEncoder()(child)['mechanics'])


def test_ranged_invisibility_and_splash_damage():
    from arknights_sim.entities.unit import Unit
    from arknights_sim.combat.targeting import EnemyTargetSelector
    s=stage();sim=Simulator(s,(op('a'),op('b')))
    a=sim.deploy('a',(1,0));b=sim.deploy('b',(2,0))
    e=Unit('mortar',replace(s.enemies[0],attack_range=7,splash=True),1000,(4,2));sim.state.enemies[e.id]=e
    assert EnemyTargetSelector.select(e,sim.state.operators).id=='b'
    sim._attack(e,b);sim.run_until(.2)
    assert a.hp==900 and b.hp==900


def test_legacy_inference_weights_migrate_but_replay_resume_is_rejected():
    from network import PolicyValueNetwork
    from training.trainer import AlphaZeroTrainer
    net=PolicyValueNetwork();old=torch.load(ROOT/'outputs/overnight_joint/checkpoints/best.pt',weights_only=True)
    net.load_state_dict(old['model_state_dict'])
    assert torch.count_nonzero(net.mechanics_input.weight)==0
    with pytest.raises(ValueError,match='旧版'):
        AlphaZeroTrainer(dict(resume_from=str(ROOT/'outputs/overnight_joint/checkpoints/latest.pt'),checkpoint_dir='/tmp/ark-mechanics-resume-check',output_dir='/tmp/ark-mechanics-resume-check',data_dir=str(ROOT/'data/real')))
