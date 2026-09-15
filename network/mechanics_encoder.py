"""Spatial mechanics observation. Values describe rules/state, never good actions."""
import torch
from arknights_sim.mechanics import runtime as mechanics

MECHANICS_CHANNELS=16
MECHANICS_FEATURE_NAMES=('solid_device','detector','device_charge','detector_active','crate_inventory','crate_cooldown','gravity_right','gravity_up','gravity_left','gravity_down','gravity_plate','plate_pressed','wind_x','wind_y','hidden_enemy','terrain')


def mechanics_features(game):
    grid=game.stage.map
    result=torch.zeros(MECHANICS_CHANNELS,grid.height,grid.width)
    for p in mechanics.solid_positions(game):result[0,p[1],p[0]]=1
    for d in game.devices.values():
        if not d.alive:continue
        x,y=d.position
        if d.data.kind=='sensor':
            result[1,y,x]=1;result[2,y,x]=d.sp/max(1,d.data.charge);result[3,y,x]=max(0,d.active_until-game.current_time)/max(1,d.data.duration)
        if d.data.kind=='wind':
            for ty in range(grid.height):
                for tx in range(grid.width):
                    if mechanics.in_wind(d,(tx,ty)):result[12:14,ty,tx]+=torch.tensor(mechanics.VECTORS[d.data.direction])
    for d in game.stage.devices:
        if d.position is None:
            result[4].fill_(game.device_inventory[d.id]/10)
            result[5].fill_(max(0,game.device_ready[d.id]-game.current_time)/10)
    if game.gravity is not None:result[6+game.gravity].fill_(1)
    for y,row in enumerate(grid.rows):
        for x,tile in enumerate(row):
            if tile.key=='tile_grvtybtn':
                result[10,y,x]=(dict(tile.blackboard).get('source_direction',0)+1)/4
                result[11,y,x]=float((x,y) in game.gravity_pressed)
            result[15,y,x]={'tile_healing':1,'tile_defup':.5,'tile_grass':.25,'tile_defbreak':-.5,'tile_poison':-1}.get(tile.key,0)
    for e in game.enemies.values():
        if e.alive and e.data.invisible:
            x,y=map(round,e.position)
            if 0<=x<grid.width and 0<=y<grid.height:result[14,y,x]+=float(not mechanics.revealed(game,e))
    return result
