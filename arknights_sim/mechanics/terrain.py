"""Simple persistent terrain effects use each tile's source parameters."""
from arknights_sim.skills.modifier import modified


def board(game,unit):
    x,y=map(round,unit.position)
    if not (0<=x<game.stage.map.width and 0<=y<game.stage.map.height):return '',{}
    t=game.stage.map.tile(x,y)
    return t.key,dict(t.blackboard)


def defense(game,unit,value):
    key,b=board(game,unit)
    if key=='tile_defup':return value+b.get('def',0)
    if key=='tile_defbreak':return value*.5
    return value


def concealed(game,unit):
    return board(game,unit)[0]=='tile_grass' and not unit.blocked_enemies


def advance(sim,dt):
    game=sim.state
    for unit in list(game.allies.values()):
        if not unit.alive:continue
        key,b=board(game,unit)
        if key=='tile_healing':unit.hp=min(unit.max_hp,unit.hp+dt*unit.max_hp*b.get('HP_RECOVERY_PER_SEC_BY_MAX_HP_RATIO',0))
        # IPA shows poison is a retained periodic buff; continuous damage would
        # be incorrect. It remains unavailable until its lifetime is implemented.
