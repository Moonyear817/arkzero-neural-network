"""Learned, stage-conditioned squad selection without a fixed operator ranking."""
import torch
from torch import nn
from arknights_sim.core.state import GameState
from network.state_encoder import StateEncoder,operator_features

PROFESSIONS=('PIONEER','WARRIOR','TANK','SNIPER','CASTER','MEDIC','SUPPORT','SPECIAL')
CANDIDATE_FEATURES=56
STAGE_FEATURES=380


def candidate_features(stage,operators):
    game=GameState(stage,operators,12345,1/60)
    return torch.tensor([operator_features(game,op.id)+[float(op.profession==p) for p in PROFESSIONS] for op in operators],dtype=torch.float32)


def stage_features(env,stage_id,seed=12345):
    state=env.reset(stage_id=stage_id,squad=(),seed=seed)
    encoded=StateEncoder().encode(state)
    from .mechanics_encoder import mechanics_features
    context=torch.cat((torch.nn.functional.adaptive_avg_pool2d(encoded['map'],(4,4)).flatten(),encoded['global'],encoded['future'].flatten(),torch.nn.functional.adaptive_avg_pool2d(mechanics_features(state.game),(4,4)).flatten()))
    return dict(legacy=context,**{k:encoded[k] for k in ('events','event_mask','event_relations','progression')}),state.game.stage


class SquadSelector(nn.Module):
    def __init__(self):
        super().__init__()
        from .future_event_encoder import FutureEventEncoder,PROGRESSION_FEATURES
        self.future_events_enabled=True
        self.future_events=FutureEventEncoder()
        self.event_context=nn.Sequential(nn.Linear(256+PROGRESSION_FEATURES,256),nn.ReLU(),nn.Linear(256,STAGE_FEATURES))
        self.mechanics_context=nn.Linear(256,STAGE_FEATURES,bias=False)
        self.scorer=nn.Sequential(nn.Linear(STAGE_FEATURES+2*CANDIDATE_FEATURES+1,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.baseline=nn.Sequential(nn.Linear(STAGE_FEATURES,64),nn.ReLU(),nn.Linear(64,1),nn.Tanh())
        nn.init.zeros_(self.baseline[-2].weight)
        nn.init.zeros_(self.baseline[-2].bias)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        from .checkpoint_compat import migrate_weights
        return super().load_state_dict(migrate_weights(self,state_dict),strict=strict,assign=assign)

    def choose(self,context,candidates,size,*,greedy=False,seed=12345):
        if isinstance(context,dict):
            graph=context;context=graph['legacy']
        else:graph=None
        if len(context)>STAGE_FEATURES:context=context[:STAGE_FEATURES]+self.mechanics_context(context[STAGE_FEATURES:])
        if graph is not None and self.future_events_enabled:
            event=self.future_events(graph['events'],graph['event_mask'],graph['event_relations'])
            context=context+self.event_context(torch.cat((event,graph['progression'])))
        if not 1<=size<=min(12,len(candidates)):raise ValueError('Invalid squad size')
        selected=[];logs=[];entropies=[]
        generator=torch.Generator(device='cpu').manual_seed(seed)
        for step in range(size):
            team=candidates[selected].mean(0) if selected else torch.zeros(CANDIDATE_FEATURES)
            features=torch.cat((context.expand(len(candidates),-1),candidates,team.expand(len(candidates),-1),torch.full((len(candidates),1),step/size)),dim=1)
            scores=self.scorer(features).squeeze(-1)
            mask=torch.zeros_like(scores,dtype=torch.bool)
            if selected:mask[selected]=True
            logits=scores.masked_fill(mask,-torch.inf)
            distribution=torch.distributions.Categorical(logits=logits)
            choice=int(logits.argmax()) if greedy else int(torch.multinomial(distribution.probs,1,generator=generator))
            selected.append(choice);logs.append(distribution.log_prob(torch.tensor(choice)));entropies.append(distribution.entropy())
        return selected,torch.stack(logs).mean(),torch.stack(entropies).mean(),self.baseline(context).squeeze(-1)
