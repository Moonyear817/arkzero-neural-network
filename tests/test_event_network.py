from dataclasses import replace
from pathlib import Path
import pytest
import torch

from arknights_sim.data.event_graph import parse_event_waves
from arknights_sim.environment import ArknightsEnv
from arknights_sim.environment.action import Action,ActionType,Direction
from network import PolicyValueNetwork,StateEncoder,ActionEncoder,NeuralEvaluator,collate_states
from network.future_event_encoder import FutureEventEncoder,EVENT_FEATURES,PROGRESSION_FEATURES,EVENT_FEATURE_NAMES
from network.squad_selector import SquadSelector,stage_features,candidate_features


@pytest.fixture(autouse=True)
def one_thread():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def graph_stage(simple,count=3):
    fragments=[]
    for i in range(count):
        fragments.append(dict(preDelay=0,actions=[dict(actionType='SPAWN',key='e',count=1,preDelay=0,interval=0,routeIndex=0,blockFragment=i==0)]))
    waves=parse_event_waves([dict(preDelay=0,postDelay=0,fragments=fragments)],simple.routes)
    return replace(simple,waves=waves)


def test_full_graph_keeps_conditionally_blocked_future_and_dependencies(simple,operator):
    env=ArknightsEnv(graph_stage(simple,131),squad=[operator]);state=env.advance_to_time(env.reset(),.1)
    encoded=StateEncoder()(state)
    assert encoded['events'].shape==(131,EVENT_FEATURES)
    assert encoded['event_mask'].all()
    assert encoded['progression'].shape==(PROGRESSION_FEATURES,)
    assert encoded['event_relations'][1,0]==2
    assert encoded['event_relations'][0,1]==5
    assert encoded['events'][1,EVENT_FEATURE_NAMES.index('time_known')]==0
    assert encoded['events'][1,EVENT_FEATURE_NAMES.index('can_trigger')]==0
    assert encoded['enemy_events'][0,0]==1
    assert encoded['enemy_events'][0,1]==1
    assert encoded['events'][1,EVENT_FEATURE_NAMES.index('remaining_count')]>0
    assert encoded['events'][0,EVENT_FEATURE_NAMES.index('block_fragment')]==1
    assert all(torch.isfinite(v).all() for v in encoded.values())


def test_dependency_attention_changes_context_and_ignores_padding():
    torch.manual_seed(12);model=FutureEventEncoder().eval()
    events=torch.randn(3,EVENT_FEATURES);mask=torch.ones(3,dtype=torch.bool)
    relation=torch.zeros(3,3,dtype=torch.long)
    with torch.no_grad():
        first=model(events,mask,relation)
        relation[2,0]=2;relation[0,2]=5
        second=model(events,mask,relation)
        padded=torch.cat((events,torch.full((4,EVENT_FEATURES),9999.)))
        rel=torch.full((7,7),9,dtype=torch.long);rel[:3,:3]=relation
        third=model(padded,torch.arange(7)<3,rel)
    assert not torch.allclose(first,second)
    assert torch.allclose(second,third,atol=2e-6,rtol=2e-5)
    empty=model(torch.zeros(1,EVENT_FEATURES),torch.zeros(1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long))
    assert torch.equal(empty,torch.zeros(256))


def test_hierarchy_neutral_type_mass_despite_many_deployments(simple,operator):
    env=ArknightsEnv(simple,squad=[operator]);state=env.reset();model=PolicyValueNetwork().eval()
    actions=env.legal_actions(state);features=ActionEncoder()(state,actions)
    with torch.no_grad():logits,value=model(StateEncoder()(state),features)
    policy=logits.softmax(0)
    types={a.type for a in actions}
    for kind in types:
        mass=sum(policy[i] for i,a in enumerate(actions) if a.type==kind)
        assert float(mass)==pytest.approx(1/len(types),abs=1e-6)
    assert len([a for a in actions if a.type==ActionType.DEPLOY])>20
    assert -1<=value<=1
    # Strict masking removes an entire prefix, without leaving missing mass.
    mask=torch.tensor([a.type==ActionType.WAIT for a in actions])
    with torch.no_grad():masked,_=model(StateEncoder()(state),features,mask)
    assert masked.softmax(0)[~mask].sum()==0
    assert float(masked.softmax(0).sum())==pytest.approx(1.)


def test_event_context_changes_embedding_and_ablation_removes_it(simple,operator):
    env=ArknightsEnv(graph_stage(simple),squad=[operator]);state=env.reset()
    encoded=StateEncoder()(state);changed={k:v.clone() for k,v in encoded.items()}
    changed['events'][-1,EVENT_FEATURE_NAMES.index('hp')]+=10
    model=PolicyValueNetwork().eval()
    with torch.no_grad():
        first=model.encode_state(encoded);second=model.encode_state(changed)
        assert not torch.allclose(first,second)
        model.future_events_enabled=False
        assert torch.equal(model.encode_state(encoded),model.encode_state(changed))


def test_predict_topk_preserves_legal_action_identity_and_state(simple,operator):
    env=ArknightsEnv(simple,squad=[operator]);state=env.reset();key=env.state_key(state)
    evaluator=NeuralEvaluator(PolicyValueNetwork())
    result=evaluator.predict(env,state)
    assert result['legal_actions']==env.legal_actions(state)
    assert result['state_embedding'].shape==(256,)
    ranked=evaluator.top_k_actions(env,state,8)
    assert len(ranked)==8 and len({a for a,p in ranked})==8
    assert all(a in result['legal_actions'] and p>0 for a,p in ranked)
    assert all(a[1]>=b[1] for a,b in zip(ranked,ranked[1:]))
    assert evaluator.top_k_actions(env,state,8)==ranked
    assert env.state_key(state)==key


def test_checkpoint_legacy_preserves_spatial_and_meta_rng(tmp_path):
    root=Path(__file__).resolve().parents[1]
    old=torch.load(root/'outputs/mechanics_verification/best.pt',map_location='cpu',weights_only=True)['model_state_dict']
    torch.manual_seed(15)
    with torch.device('meta'):model=PolicyValueNetwork()
    before=torch.get_rng_state().clone()
    model.load_state_dict(old,assign=True)
    assert torch.equal(before,torch.get_rng_state())
    assert all(not p.is_meta and torch.isfinite(p).all() for p in model.parameters())
    for key in ('map_input.weight','residual_blocks.0.conv1.weight','residual_blocks.1.conv2.weight'):
        assert torch.equal(model.state_dict()[key],old[key])
    assert model.migration_report['loaded_parameters']==1114210
    assert model.migration_report['new_parameters']>0
    path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
    restored=PolicyValueNetwork();restored.load_state_dict(torch.load(path,weights_only=True))
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    malformed=dict(restored.state_dict());malformed['map_input.weight']=torch.zeros(2,2)
    with pytest.raises(ValueError,match='shape mismatch'):restored.load_state_dict(malformed,strict=False)


def test_squad_context_contains_whole_graph_and_learns(simple,operator):
    stage=graph_stage(simple);env=ArknightsEnv(stage,squad=[operator])
    context,actual=stage_features(env,stage.id)
    assert context['events'].shape[0]==3
    candidates=candidate_features(actual,[operator,replace(operator,id='op2',atk=100)])
    selector=SquadSelector()
    chosen,logp,entropy,value=selector.choose(context,candidates,1,seed=4)
    optimizer=torch.optim.Adam(selector.parameters(),lr=.001)
    loss=-logp+(value-1).square()-.02*entropy;loss.backward();optimizer.step();optimizer.zero_grad()
    chosen,logp,entropy,value=selector.choose(context,candidates,1,seed=4)
    loss=-logp+(value-1).square()-.02*entropy;loss.backward()
    assert len(chosen)==1 and torch.isfinite(loss)
    assert selector.future_events.input.weight.grad.abs().sum()>0


def test_predict_encodes_and_runs_backbone_once(simple,operator,monkeypatch):
    env=ArknightsEnv(simple,squad=[operator]);state=env.reset()
    evaluator=NeuralEvaluator(PolicyValueNetwork())
    counts={'state':0,'backbone':0}
    encode=evaluator.state_encoder.encode;backbone=evaluator.model.encode_state
    def state_once(value):
        counts['state']+=1
        return encode(value)
    def backbone_once(value):
        counts['backbone']+=1
        return backbone(value)
    monkeypatch.setattr(evaluator.state_encoder,'encode',state_once)
    monkeypatch.setattr(evaluator.model,'encode_state',backbone_once)
    result=evaluator.predict(env,state)
    assert counts=={'state':1,'backbone':1}
    assert result['state_embedding'].shape==(256,)


def test_two_device_entities_have_distinguishable_semantics(simple,operator):
    from network.hierarchical_policy import HierarchicalPolicy
    torch.manual_seed(33)
    head=HierarchicalPolicy()
    latent=torch.randn(1,256)
    features=torch.zeros(1,2,103)
    features[0,:,96]=5;features[0,:,97]=torch.tensor([1.,2.])
    features[0,:,98:101]=-1
    features[0,0,52:54]=torch.tensor([0.,0.]);features[0,1,52:54]=torch.tensor([1.,1.])
    output=head(latent,features,torch.ones(1,2,dtype=torch.bool),torch.zeros(1,2))
    assert not torch.isclose(output[0,0],output[0,1])
    assert float(output.exp().sum().detach())==pytest.approx(1.)


def test_new_event_modules_receive_gradients_from_real_choice_targets(simple,operator):
    torch.manual_seed(91)
    squad=[operator,replace(operator,id='op2',atk=100)]
    env=ArknightsEnv(graph_stage(simple),squad=squad)
    state=env.step(env.reset(),Action(ActionType.DEPLOY,'op',(1,0),Direction.RIGHT))
    state=env.advance_to_time(state,2)
    model=PolicyValueNetwork();actions=env.legal_actions(state)
    encoded=StateEncoder()(state);features=ActionEncoder()(state,actions)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    # Added blocks intentionally start as identities and the value's last
    # layer starts at zero. Their upstream weights become trainable after the
    # first update; an all-WAIT batch would not exercise conditional heads.
    for _ in range(2):
        optimizer.zero_grad()
        logits,value=model(encoded,features)
        target=torch.linspace(1,2,len(actions));target/=target.sum()
        loss=-(target*logits.log_softmax(0)).sum()+(value-.5).square()
        loss.backward();optimizer.step()
    for name in ('future_events','progression_encoder','context_projection','context_gate','hierarchical_policy'):
        parameters=getattr(model,name).parameters()
        assert sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)>0,name
    assert model.residual_blocks[2].conv1.weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
