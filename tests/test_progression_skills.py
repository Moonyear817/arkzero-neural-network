from dataclasses import replace
from pathlib import Path
import pytest
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader, SIMPLE_MANUAL_IDS
from arknights_sim.data.progression import Progression, load_progressed
from arknights_sim.data.skill_description import describe_level
from arknights_sim.environment import ArknightsEnv, Action

DATA=Path(__file__).resolve().parents[1]/'data/real'

@pytest.fixture(scope='module')
def loader():
    return OperatorLoader(DATA/'character_table.json', DATA/'range_table.json', SkillLoader(DATA/'skill_table.json'))


def test_potential_is_per_operator_and_changes_real_stats(loader):
    base=loader.load('玫兰莎');full=loader.load('玫兰莎',potential=5)
    assert full.atk==base.atk+25
    assert full.cost==base.cost-2 and full.redeploy==base.redeploy-10
    black=loader.load('黑角',potential=5)
    assert black.hp==loader.load('黑角').hp+90
    assert any('潜能效果未实现' in n for n in loader.load('能天使',potential=5).simulation_notes)
    with pytest.raises(ValueError):loader.load('黑角',potential=6)


def test_every_character_caps_uniform_targets(loader):
    for key in loader.available_ids():
        op=load_progressed(loader,key,Progression(2,90,5,10,2))
        raw=loader.chars[key]
        assert op.elite==len(raw['phases'])-1
        assert op.level==raw['phases'][-1]['maxLevel']
        assert op.hp>0 and op.interval>0
    melon=load_progressed(loader,'玫兰莎',Progression(2,90,5,10,2))
    assert (melon.elite,melon.level,melon.skill_level,melon.skill_index)==(1,55,7,0)
    black=load_progressed(loader,'黑角',Progression(2,1))
    assert (black.elite,black.level)==(0,30)


def test_skill_unlock_rank_and_mastery(loader):
    with pytest.raises(ValueError):loader.load('玫兰莎',skill_level=7)
    assert load_progressed(loader,'玫兰莎',Progression(skill_level=10)).skill_level==4
    h=loader.load('星熊',elite=2,level=90,skill_level=10)
    raw=loader.skills.data[h.selected_skill_id]['levels'][9]
    assert h.skill.duration==raw['duration']
    assert h.skill.attack_multiplier==1+next(x['value'] for x in raw['blackboard'] if x['key']=='atk')
    assert loader.load('银灰',elite=2,skill_index=2).skill is None


def test_description_retains_special_rules_and_negative_placeholders():
    assert describe_level({'description':'速度<@x>-{-attack_speed}</>；伤害{atk:0%}', 'blackboard':[{'key':'attack_speed','value':-30},{'key':'atk','value':1.2}]})=='速度-30；伤害120%'


def test_added_skills_have_full_valid_ranks(loader):
    for key in SIMPLE_MANUAL_IDS:
        for rank,raw in enumerate(loader.skills.data[key]['levels'],1):
            skill=loader.skills.load(key,rank)
            assert skill.duration==raw['duration'] and skill.cost==raw['spData']['spCost']
    for key in ('skchr_hsguma_3','skchr_acdrop_2','skchr_mgllan_3','skchr_ghost2_2'):
        with pytest.raises(NotImplementedError):loader.skills.load(key)


def test_new_skill_damage_expiry_and_clone(loader,simple):
    from arknights_sim import Simulator
    op=replace(loader.load('星熊',elite=2,skill_level=10),cost=0)
    stage=replace(simple,enemies=tuple(replace(e,hp=100000,atk=0) for e in simple.enemies))
    sim=Simulator(stage,[op],trace=True);unit=sim.deploy(op.id,(2,0));unit.skill.sp=op.skill.cost
    sim.activate_skill(op.id)
    from arknights_sim.skills.modifier import modified
    assert modified(op.atk,'ATK',unit.modifiers,sim.current_time)==op.atk*op.skill.attack_multiplier
    assert modified(op.defense,'DEF',unit.modifiers,sim.current_time)==op.defense*op.skill.defense_multiplier
    clone=sim.clone();clone.state.operators[op.id].skill.sp=99
    assert unit.skill.sp==0
    assert modified(op.atk,'ATK',unit.modifiers,op.skill.duration)==op.atk


def test_neural_puct_and_training_can_use_skill(loader,simple):
    import torch
    from network import StateEncoder,ActionEncoder,PolicyValueNetwork,NeuralEvaluator
    from mcts import PUCTSearch,PUCTConfig
    from training.replay_buffer import TrainingSample,ReplayBuffer
    torch.set_num_threads(1)
    op=replace(loader.load('星熊',elite=2,skill_level=10),cost=0)
    env=ArknightsEnv(stage=simple,squad=[op],trace=True)
    state=env.reset();state=env.step(state,Action('DEPLOY',op.id,(2,0),'RIGHT'))
    state.game.operators[op.id].skill.sp=op.skill.cost
    legal=env.legal_actions(state);activate=Action('ACTIVATE_SKILL',op.id)
    assert activate in legal
    model=PolicyValueNetwork();result=PUCTSearch(NeuralEvaluator(model),PUCTConfig(mcts_simulations=8,max_depth=2,seed=1)).search(env,state)
    assert activate in result.actions
    assert result.policy[result.actions.index(activate)]>=0
    features=StateEncoder()(state);actions=ActionEncoder()(state,legal)
    logits,value=model(features,actions)
    loss=-logits.log_softmax(0)[legal.index(activate)]+value.square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    target=torch.zeros(len(legal));target[legal.index(activate)]=1
    sample=TrainingSample(features,tuple(a.to_dict() for a in legal),actions,target,0)
    buffer=ReplayBuffer();buffer.add_episode([sample]);assert any(a['type']=='ACTIVATE_SKILL' for a in buffer.sample(1)[0].action_dicts)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
    for _ in range(15):
        optimizer.zero_grad()
        learned_logits,_=model(features,actions)
        objective=-learned_logits.log_softmax(0)[legal.index(activate)]
        objective.backward();optimizer.step()
    # Synthetic policy training probes the action plumbing, not game competence.
    evaluator=NeuralEvaluator(model)
    result=PUCTSearch(evaluator,PUCTConfig(mcts_simulations=8,max_depth=1,seed=1,temperature=0,prior_uniform_mix=0,dirichlet_epsilon=0)).search(env,state)
    assert result.selected_action==activate
    before=env.state_key(state);child=env.step(state,result.selected_action)
    assert env.state_key(state)==before
    assert any(r['event']=='SKILL_START' for r in child.trace)
    assert activate not in env.legal_actions(child)


def test_preview_and_ui_configuration(qtbot,loader,tmp_path):
    from desktop.widgets.progression_dialog import ProgressionDialog
    dialog=ProgressionDialog(DATA,['char_500_noirc','char_208_melan'])
    qtbot.addWidget(dialog);dialog.preset.setCurrentIndex(3);dialog.potential.setCurrentIndex(5);dialog.skill_level.setCurrentIndex(9)
    assert dialog.request()==Progression(2,90,5,10)
    dialog.refresh_preview();qtbot.waitUntil(lambda:dialog._worker is None,timeout=10000)
    assert 'E0 Lv30' in dialog.table.item(0,2).text()
    assert 'E1 Lv55' in dialog.table.item(1,2).text()


def test_simulation_worker_uses_progression(loader):
    from desktop.workers.simulation_worker import SimulationWorker
    worker=SimulationWorker(stage_id='0-1',squad=['char_208_melan'],seed=1,mode='Manual control',data_dir=DATA,progression=Progression(1,55,5,7).to_dict())
    worker._load();op=worker._state.game.squad['char_208_melan']
    assert op==load_progressed(loader,'玫兰莎',Progression(1,55,5,7))


def test_training_config_and_pool_progression(loader,tmp_path):
    from training import AlphaZeroTrainer
    from training.squad_selection import SquadCoordinator
    config={'data_dir':str(DATA),'checkpoint_dir':str(tmp_path/'ckpt'),'output_dir':str(tmp_path/'out'),'device':'cpu','operator_progression':Progression(1,55,5,7).to_dict()}
    trainer=AlphaZeroTrainer(config)
    assert trainer._env().squad[1].potential==5
    path=trainer.save_checkpoint()
    restored=AlphaZeroTrainer({**config,'resume_from':str(path)})
    assert restored._env().squad==trainer._env().squad
    pool=SquadCoordinator(DATA,['char_208_melan'],1,trainable=False,progression=config['operator_progression'])
    assert pool.operators[0].level==55


def test_summons_keep_owner_skill_and_null_token_slots(loader):
    from arknights_sim.data.summon_catalog import summon_records
    rows=summon_records(loader)
    mag=[r for r in rows if r['owner_id']==loader.resolve('麦哲伦')]
    assert len(mag)==3
    assert {r['token_id'] for r in mag}=={x['override_token'] for x in mag[0]['owner_skill_bindings']}
    deep=next(r for r in rows if r['owner_id']==loader.resolve('深海色'))
    assert deep['token_skills'][0]['skillId'] is None
    assert deep['phases'][0]['attributesKeyFrames'][0]['data']['cost']==5
    assert deep['status']=='PARTIALLY_IMPLEMENTED'
    assert all(r['status']=='UNSUPPORTED_RUNTIME' for r in rows if r is not deep)


def test_all_skills_and_summons_visible_in_data_service():
    from desktop.services.research_service import load_game_data
    snapshot=load_game_data(DATA)
    skills=[r for r in snapshot.records if r.category=='Skills']
    summons=[r for r in snapshot.records if r.category=='Summons']
    assert len(skills)==893 and len(summons)==74
    ling=next(r for r in skills if r.id=='skchr_ling_3')
    assert 'UNSUPPORTED' in dict(ling.fields)['Skill details']


def test_progression_episode_replay(loader,tmp_path):
    from agents.episode import run_episode
    from agents.random_agent import RandomAgent
    from arknights_sim.environment.replay import save_episode,replay_solution
    # Exact, uncapped request preserves setup in a new simulator instance.
    op=load_progressed(loader,'玫兰莎',Progression(2,90,5,10,2))
    env=ArknightsEnv(data_dir=DATA,squad=[op],trace=True)
    initial=env.reset(seed=8)
    episode=run_episode(env,RandomAgent(8),initial,seed=8)
    path=tmp_path/'progression.json'
    document=save_episode(path,env,initial,episode,seed=8)
    assert document['operator_progression'][op.id]['potential']==5
    result=replay_solution(path)
    assert result['result']==document['result']


def test_summon_source_selection_respects_skill_elite_and_potential(loader):
    from arknights_sim.data.summon_catalog import resolve_summon_sources
    first=resolve_summon_sources(loader,'麦哲伦',Progression())
    third=resolve_summon_sources(loader,'麦哲伦',Progression(2,90,5,10,2))
    assert first['selected_tokens'][0]['token_id']=='token_10005_mgllan_drone1'
    assert third['selected_tokens'][0]['token_id']=='token_10005_mgllan_drone3'
    assert first['source_inventory_counts']==[3]
    assert third['source_inventory_counts']==[5]
    assert any(c['requiredPotentialRank']==4 for c in third['selected_talents'])
    assert third['runtime_status']=='UNSUPPORTED_RUNTIME'


def test_new_buff_changes_actual_damage_and_expires(loader,simple):
    from arknights_sim import Simulator
    from arknights_sim.entities.unit import Unit
    op=replace(loader.load('星熊',elite=2,skill_level=10),cost=0)
    enemy=replace(simple.enemies[0],hp=100000,speed=0,atk=0)
    sim=Simulator(replace(simple,enemies=(enemy,)),[op],trace=True)
    unit=sim.deploy(op.id,(2,0));target=Unit('probe',enemy,enemy.hp,(2,0))
    sim.state.enemies[target.id]=target
    unit.skill.sp=op.skill.cost;sim.activate_skill(op.id);sim._attack(unit,target)
    sim.run_until(0.2)
    hits=[r for r in sim.trace.events if r['event']=='DAMAGE' and r['attacker']==op.id]
    assert hits[0]['amount']==pytest.approx(op.atk*op.skill.attack_multiplier)
    sim.run_until(op.skill.duration+2)
    later=[r for r in sim.trace.events if r['event']=='DAMAGE' and r['attacker']==op.id and r['time']>op.skill.duration]
    assert later and all(r['amount']==pytest.approx(op.atk) for r in later)
