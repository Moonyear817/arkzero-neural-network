"""Semantic future event graph tensors and dependency-aware attention.

All action groups are retained, including conditionally blocked future spawns.
IDs address graph nodes; they are never learned vocabulary entries.
"""
import math
from functools import lru_cache
import torch
from torch import nn

EVENT_GRAPH_VERSION = 1
EVENT_FEATURE_NAMES = (
    'spawn','wait','other','wave','fragment','action','spawn_count','spawned_count',
    'remaining_count','alive_count','pre_delay','interval','pending','active','complete',
    'blocking','time_known','relative_time','earliest_time','dependency_count',
    'hp','atk','defense','resistance','speed','life_cost','weight','block_weight',
    'route_start_x','route_start_y','route_end_x','route_end_y','route_length',
    'route_waypoints','waits_for_clear','waits_for_time','threat','completion',
    'route_index','source_count','dependent_count','current_wave','current_fragment',
    'can_trigger','remaining_hp','remaining_atk','time_lower_bound','has_route',
    'block_fragment','dont_block_wave',
) + tuple(f'route_cell_{i}' for i in range(16)) + ('wave_pre_delay','wave_post_delay','wave_max_wait','fragment_pre_delay')
EVENT_FEATURES = len(EVENT_FEATURE_NAMES)
PROGRESSION_FEATURE_NAMES = (
    'wave','fragment','blocking','blocking_enemies','remaining_actions','remaining_enemies',
    'completion','wave_can_end','waits_for_clear','waits_for_time','revision',
    'active_groups','pending_groups','completed_groups','known_times','conditional_groups',
    'next_delay','remaining_hp','remaining_threat','dp','life_ratio','deployed_slots',
    'ready_skills','redeploying','wave_timeout_remaining','wave_timeout_known','last_blocker',
    'next_boundary','next_boundary_known','wave_pre_delay','wave_post_delay','wave_max_wait',
)
PROGRESSION_FEATURES = len(PROGRESSION_FEATURE_NAMES)


@lru_cache(maxsize=256)
def _route_cells(stage,route_id):
    if route_id is None or not 0<=route_id<len(stage.routes):return [0.0]*16
    from arknights_sim.map.route import compile_route
    route=stage.routes[route_id]
    points=[route.start,route.end]+[point.position for point in compile_route(stage.map,route)]
    cells=[0.0]*16
    for x,y in points:
        col=min(3,max(0,int(4*x/max(1,stage.map.width))))
        row=min(3,max(0,int(4*y/max(1,stage.map.height))))
        cells[row*4+col]+=1/max(1,len(points))
    return cells


def encode_event_graph(game):
    from arknights_sim.core.stage_events import future_event_state
    graph = future_event_state(game)
    rows = graph.get('events', [])
    count = max(1, len(rows))
    features = torch.zeros(count, EVENT_FEATURES)
    mask = torch.arange(count) < len(rows)
    # 0 unrelated, 1/2/3 time/clear/clear-or-timeout dependency; reverse4/5/6, fragment7,wave8,self9.
    relations = torch.zeros(count, count, dtype=torch.long)
    if rows:
        waves=torch.tensor([r.get('wave_id',0) for r in rows])
        fragments=torch.tensor([r.get('fragment_id',0) for r in rows])
        same_wave=waves[:,None]==waves[None,:]
        same_fragment=fragments[:,None]==fragments[None,:]
        relations=torch.where(same_wave,torch.where(same_fragment,7,8),0)
        relations.fill_diagonal_(9)
    dependencies = graph.get('dependencies', [])
    p = graph.get('progression', {})
    width, height = max(1, game.stage.map.width-1), max(1, game.stage.map.height-1)
    incoming, outgoing = [0]*count, [0]*count
    edges = []
    for edge in dependencies:
        source, target = int(edge[0]), int(edge[1])
        if not (0 <= source < len(rows) and 0 <= target < len(rows)):
            raise ValueError('Event dependency index is outside graph')
        incoming[target] += 1; outgoing[source] += 1; edges.append((source,target,str(edge[2]) if len(edge)>2 else 'time'))
    for i, row in enumerate(rows):
        stats = row.get('enemy_stats', {})
        geo = row.get('route_geometry', {}) or {}
        start, end = geo.get('start',(0,0)), geo.get('end',(0,0))
        points = geo.get('waypoints', ())
        points = [point.get('position',(0,0)) if isinstance(point,dict) else point for point in points]
        path = [start, *points, end]
        length = sum(math.dist(a,b) for a,b in zip(path,path[1:]) if len(a)==2 and len(b)==2)
        kind = str(row.get('event_type',row.get('type',''))).upper()
        trigger = str(row.get('trigger_condition','')).upper()
        status = row.get('status','pending')
        n = max(0,int(row.get('spawn_count',0))); left = max(0,int(row.get('remaining_count',n)))
        relative = row.get('estimated_relative_time')
        known = bool(row.get('time_known',relative is not None))
        threat = stats.get('hp',0)/10000 + stats.get('atk',0)/2000 + stats.get('life_cost',0)/10
        clear = any(s in trigger for s in ('CLEAR','DEATH','KILL','ENEMY','BLOCK'))
        timed = any(s in trigger for s in ('TIME','DELAY','INTERVAL'))
        values = [float(kind=='SPAWN'),float('WAIT' in kind),float(kind!='SPAWN' and 'WAIT' not in kind),
            row.get('wave_id',0)/100,row.get('fragment_id',0)/100,row.get('action_id',0)/100,
            n/100,row.get('spawned_count',0)/100,left/100,row.get('alive_count',0)/100,
            row.get('pre_delay',0)/120,row.get('interval',0)/120,
            float(status=='pending'),float(status=='active'),float(status=='complete'),
            float(row.get('blocking',False)),float(known),max(0,relative or 0)/300,
            max(0,row.get('earliest_relative_time',0) or 0)/300,incoming[i]/100,
            stats.get('hp',0)/10000,stats.get('atk',0)/2000,stats.get('defense',0)/1000,
            stats.get('resistance',0)/100,stats.get('speed',0)/5,stats.get('life_cost',0)/10,
            stats.get('weight',0)/5,stats.get('block_weight',0)/5,
            start[0]/width,start[1]/height,end[0]/width,end[1]/height,length/(width+height),len(points)/100,
            float(clear),float(timed),threat,min(1,row.get('spawned_count',0)/max(1,n)),
            max(0,row.get('route_id',0) or 0)/max(1,len(game.stage.routes)),incoming[i]/100,outgoing[i]/100,
            float(row.get('wave_id')==p.get('current_wave')),float(row.get('fragment_id')==p.get('current_fragment') and row.get('wave_id')==p.get('current_wave')),
            float(status!='complete' and known and relative is not None and relative<=0 and not row.get('blocking',False) and all(rows[j].get('status')=='complete' for j in row.get('dependencies',()))),
            left*stats.get('hp',0)/100000,left*stats.get('atk',0)/20000,
            float(not known),float(bool(geo)),float(row.get('block_fragment',False)),float(row.get('dont_block_wave',False)),*(_route_cells(game.stage,row.get('route_id')) if kind=='SPAWN' else [0.0]*16),
            row.get('wave_pre_delay',0)/120,row.get('wave_post_delay',0)/120,row.get('wave_max_wait',-1)/120,row.get('fragment_pre_delay',0)/120]
        features[i] = torch.tensor(values,dtype=torch.float32)
    for source,target,relation in edges:
        kind=3 if 'timeout' in relation else 2 if 'clear' in relation else 1
        relations[target,source]=kind;relations[source,target]=kind+3
    now=game.current_time
    trigger=str(p.get('next_trigger','')).upper()
    delays=[row.get('estimated_relative_time') for row in rows if row.get('status')!='complete' and row.get('estimated_relative_time') is not None]
    values=[p.get('current_wave',0)/100,p.get('current_fragment',0)/100,float(p.get('blocking',False)),
        p.get('blocking_enemy_count',0)/100,p.get('remaining_actions',0)/100,p.get('remaining_enemies',0)/100,
        p.get('completion_progress',0),float(p.get('wave_can_end',False)),
        float(any(s in trigger for s in ('CLEAR','DEATH','KILL','ENEMY','BLOCK'))),float(any(s in trigger for s in ('TIME','DELAY','INTERVAL'))),
        p.get('revision',0)/1000,sum(r.get('status')=='active' for r in rows)/100,
        sum(r.get('status')=='pending' for r in rows)/100,sum(r.get('status')=='complete' for r in rows)/100,
        sum(bool(r.get('time_known',r.get('estimated_relative_time') is not None)) for r in rows)/max(1,len(rows)),
        sum(r.get('estimated_relative_time') is None for r in rows)/max(1,len(rows)),
        max(0,min(delays))/300 if delays else 0,float(features[:,44].sum()),float((features[:,36]*features[:,8]).sum()),
        game.dp/max(1,game.stage.max_cost),game.life/max(1,game.stage.life),game.occupied_deployment_slots/max(1,game.stage.character_limit),
        sum(bool(u.alive and u.data.skill and u.skill.sp>=u.data.skill.cost) for u in game.allies.values())/12,
        sum(t>now for t in game.redeploy_at.values())/12,
        (p.get('wave_timeout_remaining') or 0)/120,float(p.get('wave_timeout_remaining') is not None),
        float(p.get('last_blocker',p.get('blocking_enemy_count',0)==1)),
        (p.get('next_boundary_relative_time') or 0)/120,float(p.get('next_boundary_relative_time') is not None),
        p.get('current_wave_pre_delay',0)/120,p.get('current_wave_post_delay',0)/120,p.get('current_wave_max_wait',-1)/120]
    return dict(events=features,event_mask=mask,event_relations=relations,progression=torch.tensor(values,dtype=torch.float32))


class FutureEventEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(EVENT_FEATURES,128)
        self.relations=nn.Embedding(10,4)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(128,4,512,dropout=0,activation='gelu',batch_first=True,norm_first=True) for _ in range(2)])
        self.norm=nn.LayerNorm(128)

    def forward(self,events,mask,relations):
        single=events.ndim==2
        if single:events,mask,relations=events[None],mask[None],relations[None]
        valid=mask.bool();safe=valid.clone();safe[:,0]=True
        x=self.input(events.masked_fill(~valid[...,None],0)).masked_fill(~valid[...,None],0)
        bias=self.relations(relations.clamp(0,9)).permute(0,3,1,2).reshape(-1,events.shape[1],events.shape[1])
        padding=torch.zeros_like(safe,dtype=x.dtype).masked_fill(~safe,-torch.inf)
        for layer in self.layers:
            # Keep additive dependency biases in all modes. Some PyTorch fused
            # encoder paths reinterpret a floating mask as a Boolean mask.
            normalized=layer.norm1(x)
            attention=layer.self_attn(normalized,normalized,normalized,attn_mask=bias,key_padding_mask=padding,need_weights=False)[0]
            x=x+layer.dropout1(attention)
            normalized=layer.norm2(x)
            x=x+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(normalized)))))
            x=x.masked_fill(~valid[...,None],0)
        x=self.norm(x).masked_fill(~valid[...,None],0)
        mean=x.sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        maximum=x.masked_fill(~valid[...,None],-torch.inf).amax(1)
        maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
        result=torch.cat((mean,maximum),-1)
        return result[0] if single else result
