"""Conditional legal-prefix policy without action-count bias.

Metadata is used only to group identities. No unit, stage or event ID enters a
learned embedding. Every conditional softmax contains legal children only.
"""
import torch
from torch import nn


def _conditional(scores, keys, parent_keys):
    _, groups = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    _, parents = torch.unique(parent_keys, dim=0, sorted=True, return_inverse=True)
    count=int(groups.max())+1
    sums=scores.new_zeros(count).scatter_add(0,groups,scores)
    sizes=scores.new_zeros(count).scatter_add(0,groups,torch.ones_like(scores))
    means=sums/sizes
    parent=torch.zeros(count,dtype=torch.long,device=scores.device).scatter(0,groups,parents)
    nparents=int(parents.max())+1
    maximum=scores.new_full((nparents,),-torch.inf).scatter_reduce(0,parent,means,reduce='amax',include_self=True)
    mass=scores.new_zeros(nparents).scatter_add(0,parent,torch.exp(means-maximum[parent]))
    return (means-maximum[parent]-mass[parent].log())[groups]


class HierarchicalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_head=nn.Linear(256,9)
        self.type_embedding=nn.Embedding(9,32)
        self.operator_encoder=nn.Linear(48,128)
        self.operator_query=nn.Linear(288,128)
        self.tile_encoder=nn.Sequential(nn.Linear(31,128),nn.ReLU(),nn.Linear(128,128))
        self.tile_query=nn.Linear(416,128)
        self.direction_head=nn.Linear(416,4)
        self.skill_head=nn.Linear(288,128)
        self.wait_head=nn.Linear(256,5)
        # Start types and waits neutral; learn from outcomes instead of a
        # random fixed preference for a category with few legal children.
        nn.init.zeros_(self.type_head.weight);nn.init.zeros_(self.type_head.bias)
        nn.init.zeros_(self.wait_head.weight);nn.init.zeros_(self.wait_head.bias)

    def forward(self,latent,features,mask,legacy_scores):
        outputs=[]
        for b in range(features.shape[0]):
            f=features[b,mask[b]];old=legacy_scores[b,mask[b]]
            if f.shape[-1]<103:raise ValueError('Hierarchical policy requires observation/action version 3')
            metadata=f[:,96:102]
            kinds=metadata[:,0].long()
            if bool(((kinds<0)|(kinds>=9)).any()):raise ValueError('Unknown action type')
            context=latent[b].expand(len(f),-1)
            typed=torch.cat((context,self.type_embedding(kinds)),dim=-1)
            entity_features=f[:,4:52].clone()
            devices=(kinds>=4)&(kinds<=6)
            entity_features[devices,:4]=f[devices,83:87]
            placed_devices=(kinds==5)|(kinds==6)
            entity_features[placed_devices,4:6]=f[placed_devices,52:54]
            op=torch.relu(self.operator_encoder(entity_features))
            op_score=(op*self.operator_query(typed)).sum(-1)/128**.5
            skill_score=(op*self.skill_head(typed)).sum(-1)/128**.5
            op_score=torch.where(kinds==1,skill_score,op_score)
            tile_input=torch.cat((f[:,52:54],f[:,58:78],f[:,87:96]),dim=-1)
            tile=self.tile_encoder(tile_input)
            condition=torch.cat((typed,op),dim=-1)
            tile_score=(tile*self.tile_query(condition)).sum(-1)/128**.5
            direction=metadata[:,4].long().clamp(0,3)
            direction_score=self.direction_head(condition).gather(1,direction[:,None]).squeeze(1)+old
            type_score=self.type_head(latent[b])[kinds]
            waits=metadata[:,5].long().clamp(0,4)
            wait_score=self.wait_head(latent[b])[waits]
            zeros=torch.zeros(len(f),1,device=f.device,dtype=f.dtype)
            values=_conditional(type_score,metadata[:,:1],zeros)
            values=values+_conditional(op_score,metadata[:,:2],metadata[:,:1])
            values=values+_conditional(tile_score,metadata[:,:4],metadata[:,:2])
            values=values+_conditional(direction_score,metadata[:,:5],metadata[:,:4])
            values=values+_conditional(wait_score,metadata[:,:6],metadata[:,:5])
            output=features.new_full((features.shape[1],),-torch.inf)
            outputs.append(output.masked_scatter(mask[b],values))
        return torch.stack(outputs)
