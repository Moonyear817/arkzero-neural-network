"""Source-level selection and observable combat effects, not just descriptions."""
from dataclasses import replace
import json
import random
import pytest
from arknights_sim import Simulator
from arknights_sim.combat.attack import AttackEvent
from arknights_sim.combat.runtime import deal_damage, resolve_hit, attack_power
from arknights_sim.core.event_queue import EventQueue
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from arknights_sim.entities.unit import Unit
from arknights_sim.skills.modifier import modified
from arknights_sim.talents import runtime as tr
from conftest import DATA


@pytest.fixture(scope='module')
def loader():
    return OperatorLoader(DATA/'character_table.json',DATA/'range_table.json',SkillLoader(DATA/'skill_table.json'))


def source(loader, name, handler=None, **kw):
    op=loader.load(name,elite=2,**kw)
    return tuple(t for t in op.talents if handler is None or t.handler==handler)


def params(t, **values):
    return replace(t,parameters=tuple({**dict(t.parameters),**values}.items()))


def setup(simple, operator, talents, *, seed=7, **kw):
    enemy=replace(simple.enemies[0],hp=100000,atk=0,speed=0)
    op=replace(operator,**{'skill':None,'talents':talents,**kw})
    sim=Simulator(replace(simple,enemies=(enemy,)),[op],seed=seed,trace=True)
    u=sim.deploy('op',(0,0));sim.run_until(0)
    e=sim.state.enemies['enemy_0000']
    return sim,u,e


def hit(sim,u,e,**kw):
    a=AttackEvent(u.id,e.id,sim.current_time,sim.current_time,sim.current_time,u.generation,e.generation,**kw)
    resolve_hit(sim,a)


def test_lappland_source_is_talent_and_progression_not_skill(loader):
    assert not loader.load('拉普兰德').talents
    first=loader.load('拉普兰德',elite=1,potential=0).talents[0]
    elite=loader.load('拉普兰德',elite=2,potential=2).talents[0]
    potential=loader.load('拉普兰德',elite=2,potential=4).talents[0]
    assert (first.get('duration'),elite.get('duration'),potential.get('duration'))==(1,5,6)
    assert elite.name=='精神摧毁' and elite.handler=='silence'
    assert loader.load('拉普兰德',elite=2,skill_index=1).talents==loader.load('拉普兰德',elite=2,skill_index=0).talents


def test_silence_refresh_expire_and_immune_effect_specific(simple,operator,loader):
    sim,u,e=setup(simple,operator,source(loader,'拉普兰德'))
    e.data=replace(e.data,silenceable_effects=('lifesteal',))
    hit(sim,u,e)
    assert e.silence_until==5 and not tr.effect_enabled(e,'lifesteal',0)
    assert tr.effect_enabled(e,'self_damage',0)  # no blanket ability deletion
    sim.state.clock.time=2;hit(sim,u,e)
    assert e.silence_until==7 and tr.effect_enabled(e,'lifesteal',7)
    e.silence_until=0;e.data=replace(e.data,silence_immune=True);hit(sim,u,e)
    assert e.silence_until==0
    assert any(r['event']=='STATUS_IMMUNE' for r in sim.trace.events)


def test_silence_changes_lifesteal_and_restores_it(simple,operator,loader):
    sim,u,e=setup(simple,operator,source(loader,'拉普兰德'))
    e.data=replace(e.data,lifesteal=1,atk=100,silenceable_effects=('lifesteal',))
    e.hp=500;hit(sim,u,e);before=e.hp
    hit(sim,e,u);assert e.hp==before
    sim.state.clock.time=6;hit(sim,e,u);assert e.hp==before+90


def test_talent_and_skill_atk_share_additive_percent_bucket(simple,operator,loader):
    t=loader.load('玫兰莎',elite=1,level=55,potential=5).talents
    skill=replace(operator.skill,attack_multiplier=1.5,initial_sp=2)
    sim,u,e=setup(simple,operator,t,skill=skill)
    sim.activate_skill('op')
    assert attack_power(sim,u)==pytest.approx(30*(1+.08+.5))
    assert u.data.atk==30


def test_hp_talent_preserves_ratio_and_caps_healing(simple,operator,loader):
    t=loader.load('泡普卡',elite=1,level=55,potential=5).talents
    sim,u,e=setup(simple,operator,t)
    assert u.hp==u.max_hp==1080
    from arknights_sim.skills.runtime import heal_self
    u.hp=1000;heal_self(sim,u,10000)
    assert u.hp==1080 and u.data.hp==1000


def test_attack_rng_once_for_multitarget_multihit_replay_and_clone(simple,operator,loader):
    t=source(loader,'山','mountain_crit')
    sim,u,e=setup(simple,operator,t,seed=17,normal_hits=2)
    second=Unit('enemy_extra',e.data,e.data.hp,(0,0));sim.state.enemies[second.id]=second
    parent=sim.stable_hash();clone=sim.clone()
    clone._attack(clone.state.operators['op'],clone.state.enemies[e.id],extra_targets=(clone.state.enemies[second.id],))
    rolls=[x for x in clone.trace.events if x['event']=='TALENT_ROLL']
    assert len(rolls)==1 and rolls[0]['roll']==random.Random(17).random()
    assert len([x for x in clone.state.queue.heap if x.kind=='HIT'])==4
    assert sim.stable_hash()==parent
    other=sim.clone();other._attack(other.state.operators['op'],other.state.enemies[e.id],extra_targets=(other.state.enemies[second.id],))
    clone.run_until(.5);other.run_until(.5)
    assert clone.stable_hash()==other.stable_hash() and clone.trace.events==other.trace.events


def test_damage_level_crit_rolls_each_hit_and_tremble_applies_before_penetration(simple,operator,loader):
    ts=source(loader,'锏');ts=tuple(params(t,prob=1) if t.handler=='damage_crit_tremble' else t for t in ts)
    sim,u,e=setup(simple,operator,ts)
    e.data=replace(e.data,defense=40);e.blocked_by='op';e.attack_revision=3
    before=e.hp;hit(sim,u,e)
    assert before-e.hp==pytest.approx(30*1.6-40*.75)
    assert e.tremble_until==5 and e.attack_revision==4
    hit(sim,u,e)
    assert len([r for r in sim.trace.events if r['event']=='TALENT_ROLL'])==2
    assert len([r for r in sim.trace.events if r['event']=='TREMBLE'])==2


def test_tremble_cancels_blocked_hit_but_not_unblocked_ranged(simple,operator,loader):
    sim,u,e=setup(simple,operator,())
    e.data=replace(e.data,atk=100,attack_range=3)
    e.blocked_by='op';sim._attack(e,u)
    tr.apply_tremble(sim,e,3);sim.run_until(.5)
    assert u.hp==1000
    e.blocked_by=None;e.ready_at=0;sim._attack(e,u)
    sim.run_until(1)
    assert u.hp<1000


def test_probability_sequence_is_seeded_and_not_global_rng(simple,operator,loader):
    sim,u,e=setup(simple,operator,())
    t=loader.load('月见夜',elite=1,potential=5).talents[0]
    expected=random.Random(7)
    outcomes=[tr.chance(sim,u,t,.2,'ATTACK') for _ in range(1000)]
    assert outcomes==[expected.random()<.2 for _ in range(1000)]
    assert 160<sum(outcomes)<240
    random.seed(999)
    assert sim.clone().state.rng.random()==sim.state.rng.random()


def test_ulpianus_module_potential_and_pre_damage_heal_even_dodge(simple,operator,loader):
    data=loader.load('乌尔比安',elite=2,level=90,potential=0,trust=200,module_id='uniequip_002_ulpia',module_level=3)
    heal=next(t for t in data.talents if t.handler=='pre_damage_heal')
    assert (heal.get('value1'),heal.get('value2'),heal.get('hp_ratio'))==(130,175,.6)
    assert next(t for t in data.talents if t.handler=='abyssal_kill_stats').get('max_stack_cnt')==9
    sim,u,e=setup(simple,operator,data.talents)
    u.hp=590;deal_damage(sim,u,700,'TRUE',e)
    assert u.alive and u.hp==pytest.approx(100)  # 590 + 175*1.2 - 700
    dodge=params(source(loader,'山','defense_dodge')[0],prob=1)
    u.data=replace(u.data,talents=u.data.talents+(dodge,));u.hp=500
    deal_damage(sim,u,10000,'PHYSICAL',e)
    assert u.hp==pytest.approx(710)


def test_ulpianus_shared_kill_cap_and_retreat_cleanup(simple,operator,loader):
    ts=source(loader,'乌尔比安','abyssal_kill_stats')
    sim,u,e=setup(simple,operator,ts)
    ally=replace(operator,id='ally',cost=0,skill=None,factions=('abyssal',))
    sim.state.squad['ally']=ally;v=sim.deploy('ally',(2,0))
    for i in range(12):
        victim=Unit('victim'+str(i),e.data,1,(0,0));sim.state.enemies[victim.id]=victim
        sim._remove_enemy(victim,u)
    assert attack_power(sim,u)==30+9*30 and u.max_hp==1000+9*120
    assert attack_power(sim,v)==30+9*15 and v.max_hp==1000+9*60
    sim.retreat('op')
    assert attack_power(sim,v)==30 and v.max_hp==1000
    sim.state.clock.time=10;new=sim.deploy('op',(0,0))
    assert new.talent_state=={} and new.max_hp==1000


def test_emergency_heal_catches_lethal_once_and_redeploy_resets(simple,operator,loader):
    ts=source(loader,'隐德来希','emergency_heal')
    sim,u,e=setup(simple,operator,ts)
    deal_damage(sim,u,2000,'PHYSICAL',e)
    assert u.hp==501 and u.alive and u.talent_state['emergency_used']
    deal_damage(sim,u,100,'PHYSICAL',e)
    assert u.hp==411
    deal_damage(sim,u,2000,'TRUE',e)
    assert not u.alive
    sim.state.clock.time=10;new=sim.deploy('op',(0,0))
    assert not new.talent_state.get('emergency_used')


def test_surtr_lethal_window_forces_removal_without_retreat_refund(simple,operator,loader):
    sim,u,e=setup(simple,operator,source(loader,'史尔特尔','lethal_survival'))
    deal_damage(sim,u,5000,'TRUE',e);dp=sim.state.dp
    assert u.hp==1 and u.alive
    deal_damage(sim,u,5000,'TRUE',e);assert u.hp==1
    sim.run_until(8)
    assert not u.alive and sim.state.redeploy_at['op']==13
    assert sim.state.dp==min(99,dp+8)


def test_steal_before_damage_dot_refresh_and_nonrecursive(simple,operator,loader):
    sim,u,e=setup(simple,operator,source(loader,'隐德来希','hp_steal_dot'))
    e.data=replace(e.data,hp=1000,resistance=50);e.hp=1000
    hit(sim,u,e)
    assert e.max_hp==925 and e.hp==895 and u.max_hp==1075
    u.ready_at=100;e.ready_at=100
    sim.run_until(1)
    assert e.hp==pytest.approx(795)  # 200 arts, 50 RES
    assert u.max_hp==1075  # dot does not reapply steal
    hit(sim,u,e);u.ready_at=100;sim.run_until(2)
    ticks=[r for r in sim.trace.events if r['event']=='DAMAGE' and r['amount']==100]
    assert len(ticks)==2
    assert len([r for r in sim.trace.events if r['event']=='MAX_HP_STOLEN'])==2


def test_nearby_death_heal_not_on_escape_or_escort(simple,operator,loader):
    sim,u,e=setup(simple,operator,source(loader,'艾丝黛尔'))
    u.hp=400;sim._remove_enemy(e)
    assert u.hp==520
    escort=Unit('escort',replace(e.data,faction='ESCORT'),1,(0,0));sim.state.enemies['escort']=escort
    sim._remove_enemy(escort);assert u.hp==520


def test_zuole_hidden_module_keeps_both_original_talents_and_sp_chance(simple,operator,loader):
    ts=source(loader,'左乐',level=90,potential=4,trust=200,module_id='uniequip_002_zuole',module_level=1)
    assert {'berserk','attack_sp_chance','low_hp_shelter'} <= {t.handler for t in ts}
    sim,u,e=setup(simple,operator,ts,skill=replace(operator.skill,cost=100,recovery='AUTO_RECOVERY'))
    u.hp=200;tr.refresh(sim)
    assert modified(100,'ASPD',u.modifiers,0)==150 and tr.natural_sp_rate(u)==3
    deal_damage(sim,u,100,'PHYSICAL',e);assert u.hp==125
    expected=random.Random(7).random()<.75
    tr.on_attack(sim,u);assert u.skill.sp==int(expected)


def test_reaper_heal_has_per_attack_target_cap_and_unhealable_trait(simple,operator,loader):
    sim,u,e=setup(simple,operator,(),subprofession='reaper',elite=2,block_count=2,healable=False)
    u.hp=500
    for target_index in range(3):hit(sim,u,e,target_index=target_index)
    assert u.hp==640
    from arknights_sim.skills.runtime import heal_self
    heal_self(sim,u,100,as_healing=True)
    assert u.hp==640


def test_gavial_module_replaces_matching_talent_and_updates_for_blocking(simple,operator,loader):
    ts=source(loader,'百炼嘉维尔',level=90,potential=1,trust=200,module_id='uniequip_002_gvial2',module_level=2)
    assert len([t for t in ts if t.handler=='block_stats'])==1
    sim,u,e=setup(simple,operator,ts)
    u.blocked_enemies=['a','b'];tr.refresh(sim)
    assert attack_power(sim,u)==pytest.approx(30*1.21)
    u.blocked_enemies.clear();tr.refresh(sim)
    assert attack_power(sim,u)==pytest.approx(30*1.13)


def test_talent_state_clone_isolation_and_hash_covers_timers(simple,operator,loader):
    sim,u,e=setup(simple,operator,source(loader,'羽毛笔'))
    before=sim.stable_hash();child=sim.clone()
    child.state.operators['op'].talent_state['kills:1']=4
    child.state.enemies[e.id].silence_until=10
    assert child.stable_hash()!=before and sim.stable_hash()==before
    assert u.talent_state=={} and e.silence_until==0
