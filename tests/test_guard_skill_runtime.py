"""Combat-level effects, resource causality and deterministic multi-hit contracts."""
from dataclasses import replace
import pytest
from arknights_sim import Simulator
from arknights_sim.data.models import SkillData
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from arknights_sim.environment import ArknightsEnv, Action
from conftest import DATA

@pytest.fixture(scope='module')
def loader():
    return OperatorLoader(DATA/'character_table.json',DATA/'range_table.json',SkillLoader(DATA/'skill_table.json'))


def fight(simple,operator,skill,**kwargs):
    stage=replace(simple,enemies=(replace(simple.enemies[0],hp=10000,atk=0,speed=0),))
    op=replace(operator,skill=skill,**kwargs)
    sim=Simulator(stage,[op],trace=True);sim.deploy(op.id,(0,0))
    return sim


def outgoing(sim):
    return [e['amount'] for e in sim.trace.events if e['event']=='DAMAGE' and e['attacker']=='op']


def test_attack_recovery_does_not_charge_by_wait_and_consumes_on_next_attack(simple,operator):
    skill=SkillData('next',2,0,0,recovery='ATTACK_RECOVERY',trigger='AUTO_TRIGGER',mode='NEXT_ATTACK',attack_scale=2)
    sim=fight(simple,operator,skill);sim.run_until(2.5)
    assert outgoing(sim)==[30,30,60]
    assert sim.state.operators['op'].skill.sp==0
    idle=fight(simple,operator,skill,attack_range=())
    idle.state.enemies.clear()  # spawn still pending
    idle.state.operators['op'].position=(7,0)
    idle.run_until(10)
    assert idle.state.operators['op'].skill.sp==0


def test_defense_recovery_charges_only_damaging_attacks_and_manual_mask(simple,operator,loader):
    skill=replace(loader.skills.load('skchr_estell_2',7),cost=2,initial_sp=0)
    sim=fight(simple,operator,skill)
    sim.run_until(2)
    assert sim.state.operators['op'].skill.sp==0
    sim.state.enemies['enemy_0000'].data=replace(sim.state.enemies['enemy_0000'].data,atk=100)
    sim.run_until(4)
    assert sim.state.operators['op'].skill.sp==2
    sim.activate_skill('op')
    assert sim.state.operators['op'].skill.sp==0
    sim.run_until(6)
    assert sim.state.operators['op'].skill.sp==0


def test_double_hit_applies_defense_per_hit_and_fresh_replay(simple,operator,loader):
    skill=replace(loader.skills.load('skchr_crow_1',7),cost=1,initial_sp=1)
    def run():
        sim=fight(simple,operator,skill)
        sim.run_until(0)
        sim.state.enemies['enemy_0000'].data=replace(sim.state.enemies['enemy_0000'].data,defense=10)
        return sim
    sim=run();parent=sim.state.stable_hash();clone=sim.clone();clone.run_until(.5)
    assert outgoing(clone)==pytest.approx([30*1.35-10]*2)
    assert sim.state.stable_hash()==parent
    replay=run();replay.run_until(.5)
    assert clone.state.stable_hash()==replay.state.stable_hash()
    assert clone.trace.events==replay.trace.events


def test_instant_heal_uses_max_hp_caps_and_has_no_active_duration(simple,operator,loader):
    skill=loader.skills.load('skcom_heal_self[2]',1)
    sim=fight(simple,operator,skill);u=sim.state.operators['op'];u.hp=950;u.skill.sp=skill.cost
    sim.activate_skill('op')
    assert u.hp==1000 and u.skill.sp==0 and not u.skill.active


def test_damage_conversion_and_lifesteal(simple,operator,loader):
    skill=replace(loader.skills.load('skchr_midn_1'),initial_sp=80)
    sim=fight(simple,operator,skill);sim.run_until(0);e=sim.state.enemies['enemy_0000'];e.data=replace(e.data,defense=1000,resistance=20)
    sim.activate_skill('op');sim.run_until(.5)
    assert outgoing(sim)==pytest.approx([30*1.05*.8])
    skill=loader.skills.load('skchr_gvial2_1',7)
    sim=fight(simple,operator,skill);u=sim.state.operators['op'];u.hp=500;u.skill.sp=skill.cost
    sim.activate_skill('op');sim.run_until(.5)
    assert u.hp==pytest.approx(500+30*1.6*.35)


def test_skill_debuff_refreshes_without_multiplying_itself(simple,operator,loader):
    skill=replace(loader.skills.load('skchr_frncat_1'),cost=1,initial_sp=1)
    sim=fight(simple,operator,skill);sim.run_until(.5)
    from arknights_sim.skills.modifier import modified
    e=sim.state.enemies['enemy_0000']
    assert modified(100,'ATK',e.modifiers,.5)==85
    sim.state.operators['op'].skill.sp=1;sim.run_until(1.5)
    assert modified(100,'ATK',e.modifiers,1.5)==85
    assert modified(100,'ATK',e.modifiers,7)==100


def test_stop_attack_zero_block_and_regeneration(simple,operator,loader):
    skill=loader.skills.load('skchr_utage_1')
    sim=fight(simple,operator,skill);u=sim.state.operators['op'];u.hp=500;u.skill.sp=skill.cost
    sim.activate_skill('op');sim.run_until(1)
    assert outgoing(sim)==[] and u.blocked_enemies==[]
    assert u.hp==pytest.approx(560)


def test_automatic_skills_are_not_manual_agent_actions(simple,operator,loader):
    op=replace(operator,skill=loader.skills.load('skchr_huang_1'))
    env=ArknightsEnv(simple,[op]);state=env.step(env.reset(),Action('DEPLOY','op',(0,0),'RIGHT'))
    state.game.operators['op'].skill.sp=op.skill.cost
    assert Action('ACTIVATE_SKILL','op') not in env.legal_actions(state)


def test_auto_timed_skill_starts_without_manual_action(simple,operator,loader):
    skill=replace(loader.skills.load('skchr_whitew_2',9),cost=1,initial_sp=1)
    sim=fight(simple,operator,skill);sim.run_until(.5)
    assert sim.state.operators['op'].skill.active
    assert outgoing(sim)==pytest.approx([30*2.1])


def test_reaper_and_centurion_attack_multiple_without_cloning_targets(simple,operator):
    from arknights_sim.data.models import SpawnData
    stage=replace(simple,waves=(replace(simple.waves[0],spawns=(SpawnData(0,'e',0),SpawnData(0,'e',0),SpawnData(0,'e',0))),),
                  enemies=(replace(simple.enemies[0],hp=10000,atk=0,speed=0),))
    sim=Simulator(stage,[replace(operator,skill=None,target_mode='BLOCK_COUNT',block_count=2)],trace=True)
    sim.deploy('op',(0,0));sim.run_until(.5)
    damages=[e for e in sim.trace.events if e['event']=='DAMAGE' and e['attacker']=='op']
    assert len(damages)==2 and len({e['target'] for e in damages})==2


def test_toggle_changes_range_recovers_cooldown_and_reverses_cleanly(simple,operator,loader):
    from arknights_sim.skills.modifier import modified
    from arknights_sim.combat.targeting import range_cells
    skill=loader.skills.load('skchr_f12yin_2',10)
    sim=fight(simple,operator,skill);u=sim.state.operators['op'];u.skill.sp=skill.cost;u.hp=500
    sim.activate_skill('op')
    assert u.skill.active and modified(30,'ATK',u.modifiers,0)==54
    assert range_cells(u)==[(0,0)]
    sim.run_until(5)
    assert u.skill.sp==skill.cost and u.hp==pytest.approx(850)
    sim.activate_skill('op')
    assert not u.skill.active and modified(30,'ATK',u.modifiers,5)==30
    assert range_cells(u)==[(0,0),(1,0)]
    assert sim.state.stable_hash()


def test_infinite_auto_buff_is_finite_serializable_and_does_not_repeat(simple,operator,loader):
    skill=replace(loader.skills.load('skchr_whitew_1',7),cost=1,initial_sp=1)
    sim=fight(simple,operator,skill);sim.run_until(5)
    u=sim.state.operators['op']
    assert u.skill.is_active(skill,5) and u.skill.sp==0
    assert len([e for e in sim.trace.events if e['event']=='SKILL_START'])==1
    assert sim.state.stable_hash()==sim.clone().state.stable_hash()


def test_new_registered_skill_recipes_cover_all_source_ranks(loader):
    from arknights_sim.data.guard_skills import load_guard_skill
    keys=[]
    for key,raw in loader.skills.data.items():
        if load_guard_skill(key,raw['levels'][0],loader.skills.ranges) is not None:
            keys.append(key)
            for row in raw['levels']:
                assert load_guard_skill(key,row,loader.skills.ranges) is not None
    assert len(keys)>=35
