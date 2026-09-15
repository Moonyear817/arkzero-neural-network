"""Train squad choices from actual battle outcomes; no scripted composition labels."""
from pathlib import Path
from time import perf_counter
import torch
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from network.squad_selector import SquadSelector,candidate_features,stage_features
from arknights_sim.environment.rewards import REWARD_VERSION, episode_reward, is_truncated
from .schema import OBSERVATION_VERSION, EVENT_GRAPH_VERSION


def eligible_ids(loader):
    # The battle curriculum excludes branches whose defining mechanic is absent.
    # This is a simulator capability mask, not a tactical preference or ranking.
    basic={'pioneer','fearless','protector','fastshot','corecaster','physician','fighter'}
    result=[]
    for key in loader.available_ids():
        raw=loader.chars[key]
        if raw['subProfessionId'] not in basic:continue
        active=[c for t in raw.get('talents') or [] for c in t.get('candidates') or [] if c['unlockCondition']['phase']=='PHASE_0' and c['unlockCondition']['level']<=1 and c.get('requiredPotentialRank',0)==0 and c.get('description')]
        if active:continue
        result.append(key)
    return tuple(result)


class SquadCoordinator:
    def __init__(self,data_dir,pool=None,size=6,selector=None,trainable=True,progression=None,skill_overrides=None):
        data=Path(data_dir)
        self.loader=OperatorLoader(data/'character_table.json',data/'range_table.json',SkillLoader(data/'skill_table.json'))
        self.keys=tuple(pool) if pool is not None else eligible_ids(self.loader)
        if len(set(self.keys))!=len(self.keys):raise ValueError('Duplicate roster candidates')
        from dataclasses import replace
        from arknights_sim.data.progression import Progression, load_progressed
        request=Progression.from_dict(progression)
        self.operators=tuple(load_progressed(self.loader,k,replace(request,skill_index=(skill_overrides or {}).get(k,request.skill_index))) for k in self.keys)
        self.size=size
        if not 1<=size<=min(12,len(self.keys)):raise ValueError('Invalid neural squad size')
        self.selector=selector or SquadSelector()
        self.optimizer=torch.optim.Adam(self.selector.parameters(),lr=0.0003) if trainable else None
        self._cache={}
        self.trace=None

    def choose(self,env,stage_id,seed,training,*,learn=True):
        learning = training and learn
        if learning and self.optimizer is None:raise ValueError('Inference-only squad selector cannot train')
        context,stage=stage_features(env,stage_id,seed)
        if stage not in self._cache:self._cache[stage]=candidate_features(stage,self.operators)
        with torch.set_grad_enabled(learning):
            indices,logp,entropy,value=self.selector.choose(context,self._cache[stage],self.size,greedy=not training,seed=seed)
        self.trace=(logp,entropy,value) if learning else None
        return tuple(self.operators[i] for i in indices)

    def finish(self,metrics,training):
        if not training or self.trace is None or is_truncated(metrics):
            self.trace=None
            return {}
        started=perf_counter()
        logp,entropy,value=self.trace;self.trace=None
        reward=episode_reward(metrics)
        advantage=reward-value.detach()
        policy_loss=-advantage*logp
        value_loss=(value-reward).square()
        loss=policy_loss+0.5*value_loss-0.02*entropy
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite squad-selection loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient_norm=torch.nn.utils.clip_grad_norm_(self.selector.parameters(),5,error_if_nonfinite=True);self.optimizer.step()
        return dict(squad_selection_loss=float(loss.detach()),squad_selection_reward=reward,
                    squad_selection_entropy=float(entropy.detach()),
                    squad_selection_policy_loss=float(policy_loss.detach()),
                    squad_selection_value_loss=float(value_loss.detach()),
                    squad_selection_gradient_norm=float(gradient_norm.detach()),
                    squad_selection_step_time=perf_counter()-started)

    def state_dict(self,include_optimizer=True):
        result=dict(version=1,reward_version=REWARD_VERSION,
                    observation_version=OBSERVATION_VERSION,event_graph_version=EVENT_GRAPH_VERSION,
                    future_events_enabled=getattr(self.selector,'future_events_enabled',True),
                    pool=list(self.keys),size=self.size,model=self.selector.state_dict())
        if include_optimizer and self.optimizer is not None:result['optimizer']=self.optimizer.state_dict()
        return result

    def load_state_dict(self,state):
        if state['version']!=1 or tuple(state['pool'])!=self.keys or state['size']!=self.size:raise ValueError('Squad selector configuration mismatch')
        self.selector.load_state_dict(state['model'])
        if 'optimizer' in state and self.optimizer is not None:
            if state.get('reward_version') != REWARD_VERSION:raise ValueError('Squad reward version mismatch')
            if (state.get('observation_version') != OBSERVATION_VERSION or state.get('event_graph_version') != EVENT_GRAPH_VERSION):
                raise ValueError('Squad observation/event graph version mismatch; warm-start weights only')
            self.optimizer.load_state_dict(state['optimizer'])
        if 'future_events_enabled' in state:
            self.selector.future_events_enabled=state['future_events_enabled']
