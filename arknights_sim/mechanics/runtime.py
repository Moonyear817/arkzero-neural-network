"""PRTS definitions: crates, detector, airflow and directional gravity.

Gravity debris/boss effects are deliberately rejected by stage capability checks.
"""
import math
from dataclasses import replace
from functools import lru_cache
from arknights_sim.data.models import MapData,WaypointData,RouteData
from arknights_sim.map.route import compile_route, WAIT_KINDS
from .definitions import DeviceState

VECTORS=((1,0),(0,1),(-1,0),(0,-1))


def initialize(game):
    game.devices={d.id:DeviceState(d,d.position,hp=d.hp,id=d.id) for d in game.stage.devices if d.position is not None}
    game.device_inventory={d.id:d.count for d in game.stage.devices if d.position is None}
    game.device_ready={d.id:0.0 for d in game.stage.devices if d.position is None}
    game.device_sequence=0
    game.gravity=next((d.direction for d in game.stage.devices if d.kind=='gravity'),None)
    game.gravity_pressed=[]
    game.enemy_paths={}
    game.mechanic_revision=0


def solid_positions(game):
    return frozenset(v.position for v in game.devices.values() if v.alive and v.data.kind in ('crate','sensor','roadblock','friendly_frost','friendly_altar'))

@lru_cache(maxsize=256)
def blocked_grid(grid,positions):
    return MapData(tuple(tuple(replace(t,passable='FLY_ONLY',buildable='NONE') if (x,y) in positions else t for x,t in enumerate(row)) for y,row in enumerate(grid.rows)))


def paths_for(game):
    grid=blocked_grid(game.stage.map,solid_positions(game))
    escort_grid=blocked_grid(game.stage.map,escort_obstacles(game))
    return compiled_paths(game.stage,grid,escort_grid)


def escort_obstacles(game):
    # Roadblocks bind escorts with their passive skill. They do not make the
    # escort's authored checkpoint on that tile an invalid path endpoint.
    return frozenset(d.position for d in game.devices.values() if d.alive and d.data.kind in ('crate','sensor','friendly_frost','friendly_altar'))

@lru_cache(maxsize=128)
def compiled_paths(stage,grid,escort_grid=None):
    used={s.route_index for s in stage.spawns}
    escort_ids={e.id for e in stage.enemies if e.faction=='ESCORT'}
    escort_routes={s.route_index for s in stage.spawns if s.enemy_id in escort_ids}
    hostile_routes={s.route_index for s in stage.spawns if s.enemy_id not in escort_ids}
    if escort_routes & hostile_routes:raise NotImplementedError('Shared hostile/escort route requires separate path state')
    return tuple(compile_route(escort_grid if i in escort_routes and escort_grid is not None else grid,r)
                 if i in used else () for i,r in enumerate(stage.routes))


def unit_path(game,enemy,default):
    return game.enemy_paths.get(enemy.id,default[enemy.route_index])


def remaining_route(game,enemy,default):
    path=unit_path(game,enemy,default)
    # Preserve pending checkpoints, waits and destination, never restart passed checkpoints.
    pending=tuple(w for w in path[enemy.node:] if w.kind in WAIT_KINDS | {'MOVE','DISAPPEAR','APPEAR_AT_POS'})
    return RouteData(enemy.position,game.stage.routes[enemy.route_index].end,pending,
                     motion=game.stage.routes[enemy.route_index].motion,
                     diagonal=game.stage.routes[enemy.route_index].diagonal)


def can_place(game,position):
    if position in solid_positions(game) or any(o.alive and o.position==position for o in game.allies.values()):return False
    tile=game.stage.map.tile(*position)
    if tile.buildable not in ('MELEE','ALL'):return False
    # Occupied enemy cells are excluded until collision / obstacle destruction is calibrated.
    if any(e.alive and math.dist(e.position,position)<0.71 for e in game.enemies.values()):return False
    try:
        grid=blocked_grid(game.stage.map,solid_positions(game)|{position})
        escort_grid=blocked_grid(game.stage.map,escort_obstacles(game)|{position})
        compiled_paths(game.stage,grid,escort_grid)
        old=paths_for(game)
        for e in game.enemies.values():
            if e.alive:compile_route(escort_grid if e.data.faction=='ESCORT' else grid,remaining_route(game,e,old))
    except ValueError:return False
    return True


def reroute(game,old):
    grid=blocked_grid(game.stage.map,solid_positions(game))
    escort_grid=blocked_grid(game.stage.map,escort_obstacles(game))
    for e in game.enemies.values():
        if e.alive:
            game.enemy_paths[e.id]=compile_route(escort_grid if e.data.faction=='ESCORT' else grid,remaining_route(game,e,old));e.node=0
    game.mechanic_revision+=1


def place(sim,key,position):
    g=sim.state;data=next((d for d in g.stage.devices if d.id==key and d.position is None),None)
    if g.done or not data or g.device_inventory[key]<=0 or g.current_time<g.device_ready[key] or g.dp<data.cost or not can_place(g,position):raise ValueError('Illegal device deployment')
    old=paths_for(g);g.device_sequence+=1;identifier=f'{key}@{g.device_sequence}'
    g.devices[identifier]=DeviceState(data,position,hp=data.hp,id=identifier,deployed_at=g.current_time);g.device_inventory[key]-=1;g.dp-=data.cost;g.device_ready[key]=g.current_time+data.cooldown
    reroute(g,old);sim.paths=dict(enumerate(paths_for(g)));sim._log('DEVICE_DEPLOY',device=identifier,position=position)


def remove(sim,key):
    g=sim.state;d=g.devices.get(key)
    if g.done or not d or not d.alive or d.data.kind!='crate':raise ValueError('Device cannot be removed')
    old=paths_for(g);d.alive=False
    from arknights_sim.combat.blocking import release
    for eid in list(d.blocked_enemies):release(g.enemies[eid],{**g.allies,**g.devices})
    # Token count is consumable: retreat does not restore a used card.
    reroute(g,old);sim.paths=dict(enumerate(paths_for(g)));sim._log('DEVICE_REMOVE',device=key)


def activate(sim,key):
    g=sim.state;d=g.devices.get(key)
    if g.done or not d or not d.alive or d.data.kind!='sensor' or d.sp+1e-8<d.data.charge or g.current_time<d.active_until:raise ValueError('Device is not ready')
    d.sp=0;d.active_until=g.current_time+d.data.duration;g.mechanic_revision+=1
    sim._log('DEVICE_ACTIVATE',device=key,active_until=d.active_until)


def revealed(game,enemy):
    if enemy.hidden:return False
    if not enemy.data.invisible or enemy.blocked_by:return True
    for d in game.devices.values():
        if d.alive and d.data.kind=='sensor' and game.current_time<d.active_until:
            dx,dy=enemy.position[0]-d.position[0],enemy.position[1]-d.position[1]
            if any(abs(dx-x)<=.5 and abs(dy-y)<=.5 for x in range(-3,4) for y in range(-3,4) if abs(x)+abs(y)<=3):return True
    return False


def alignment(direction,vector):
    dx,dy=VECTORS[direction];projection=dx*vector[0]+dy*vector[1]
    return 1 if projection>1e-8 else -1 if projection< -1e-8 else 0


def gravity_speed(game,enemy,vector):
    if game.gravity is None:return 1
    a=alignment(game.gravity,vector);w=min(4,max(0,enemy.data.weight))
    return (1.08,1.2,1.32,2.4,2.64)[w] if a>0 else (.9,.66,.6,.36,.3)[w] if a<0 else 1


def in_wind(d,position):
    vx,vy=VECTORS[d.data.direction];dx=position[0]-d.position[0];dy=position[1]-d.position[1]
    return abs(dx*vy-dy*vx)<=.5 and -.5<=dx*vx+dy*vy<=3.5


def speed_factor(game,enemy,vector):
    factor=gravity_speed(game,enemy,vector)
    for d in game.devices.values():
        if d.alive and d.data.kind=='wind' and in_wind(d,enemy.position) and revealed(game,enemy):
            a=alignment(d.data.direction,vector)
            factor*=((1.8 if d.data.skill_level==1 else 1.5) if a>0 else .5 if a<0 else 1)
    return factor


def operator_as(game,op):
    return 0 if game.gravity is None else 25*alignment(game.gravity,VECTORS[op.direction])


def operator_attack(game,op):
    increases=0;decreases=1
    for d in game.devices.values():
        if d.alive and d.data.kind=='wind' and in_wind(d,op.position):
            a=alignment(d.data.direction,VECTORS[op.direction])
            if a>0:increases+=.3
            elif a<0:decreases*=.7
    return (1+increases)*decreases


def advance(game,old,time):
    for d in game.devices.values():
        if d.alive and d.data.charge:d.sp=min(d.data.charge,d.sp+max(0,time-max(old,d.active_until)))


def gravity_switches(game):
    """Occupied pressure plates: earliest still-pressed plate owns direction."""
    if game.gravity is None:return
    # IPA tile_grvtybtn polling component: _defaultInterval = 0.1 seconds.
    if abs(game.current_time/.1-round(game.current_time/.1))>1e-6:return
    from arknights_sim.skills.modifier import modified
    pressed=set()
    for y,row in enumerate(game.stage.map.rows):
        for x,tile in enumerate(row):
            if tile.key!='tile_grvtybtn':continue
            board=dict(tile.blackboard);minimum=board.get('activate_block_cnt',1)
            units=any(o.alive and o.position==(x,y) and modified(o.data.block_count,'BLOCK_COUNT',o.modifiers,game.current_time)>=minimum for o in game.allies.values())
            enemies=any(e.alive and e.data.weight>=3 and abs(e.position[0]-x)<=.5 and abs(e.position[1]-y)<=.5 for e in game.enemies.values())
            if units or enemies:pressed.add((x,y))
    previous=list(game.gravity_pressed)
    ordered=[p for p in previous if p in pressed]+sorted(pressed-set(previous))
    game.gravity_pressed=ordered
    if ordered:
        tile=game.stage.map.tile(*ordered[0]);source=int(dict(tile.blackboard)['source_direction'])
        # Source enum UP, RIGHT, DOWN, LEFT -> simulator RIGHT, UP, LEFT, DOWN.
        direction=(1,0,3,2)[source]
        if game.gravity!=direction:game.gravity=direction;game.mechanic_revision+=1


def roadblock_target(game, operator):
    """Enemy-faction obstacle with taunt -1: normal enemies take precedence."""
    from arknights_sim.combat.targeting import in_range
    return min((d for d in game.devices.values() if d.alive and d.data.kind=='roadblock'
                and in_range(operator,d)), key=lambda d:d.id, default=None)


def destroy_roadblock(sim, key):
    game=sim.state;device=game.devices[key]
    if device.data.kind != 'roadblock' or device.hp > 0:
        raise ValueError('Roadblock destruction requires lethal damage')
    old=paths_for(game)
    device.alive=False
    reroute(game,old)
    sim.paths=dict(enumerate(paths_for(game)))
    sim._log('DEVICE_DESTROY',device=key)


def roadblock_binds(game, unit):
    # Passive sktok_roadblock: escort targets only, not ordinary hostile mobs.
    return unit.data.id in {'enemy_3001_upeopl','enemy_3002_ftrtal','enemy_3002_ftrtal_s'} and any(
        d.alive and d.data.kind=='roadblock' and math.dist(d.position,unit.position)<=0.6
        for d in game.devices.values())
