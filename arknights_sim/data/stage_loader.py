import json
from dataclasses import replace
from pathlib import Path
from .models import (
    MapData,
    TileData,
    RouteData,
    WaypointData,
    StageData,
    WaveData,
    SpawnData,
)
from .enemy_loader import EnemyLoader
from arknights_sim.map.route import WAIT_KINDS, ROUTE_KINDS


def point(p, offset=None):
    o = offset or {}
    return (p["col"] + o.get("x", 0), p["row"] + o.get("y", 0))


class StageLoader:
    """Normalize supported battle rules and explicit stage event dependencies."""

    def __init__(self, enemy_path):
        self.enemy_loader = EnemyLoader(enemy_path)

    def load(self, path, difficulty="NORMAL"):
        return self.parse(json.loads(Path(path).read_text()), Path(path).stem, difficulty)

    def parse(self, d, name="stage", difficulty="NORMAL"):
        if difficulty not in ('NORMAL','FOUR_STAR'):raise NotImplementedError('Difficulty '+difficulty)
        from arknights_sim.mechanics.definitions import parse_devices
        devices=parse_devices(d)
        active_runes=[r for r in (d.get('runes') or []) if r['difficultyMask'] in ('ALL',difficulty)]
        hp_scale=atk_scale=def_scale=1
        life_override=None
        for r in active_runes:
            b={v['key']:v.get('valueStr') if v.get('valueStr') is not None else v['value'] for v in r['blackboard']}
            if r.get('professionMask',1023)!=1023 or r.get('buildableMask','ALL')!='ALL':raise NotImplementedError('Restricted rune mask')
            if r['key']=='gbuff_lifepoint':life_override=int(b['value'])
            elif r['key']=='ebuff_attribute' and set(b)<= {'atk','def','max_hp'}:
                atk_scale*=b.get('atk',1);def_scale*=b.get('def',1);hp_scale*=b.get('max_hp',1)
            elif r['key']=='cbuff_token_initial_cnt':
                if b['char']!='trap_001_crate':raise NotImplementedError('Rune token '+b['char'])
                devices=tuple(replace(v,count=max(0,v.count+int(b['value']))) if v.id==b['char'] else v for v in devices)
            else:raise NotImplementedError('Active rune: '+r['key'])
        assumptions = [
            "RULE-001 fragment timing (0-1 PRTS.Map matched)",
            "RULE-002 steering and spawn jitter",
            "RULE-003 tutorial pause excluded",
        ]
        dormant_branches = ()
        if d.get('branches'):
            branches=d['branches']
            # The copied Black Snake branch occurs in several levels without
            # Black Snake, flame devices, or a branch activation action. Retain
            # its name, but do not execute a definition without a trigger.
            actions=[a for w in d['waves'] for f in w['fragments'] for a in f['actions']]
            can_trigger=any(a['actionType'] not in {'SPAWN','STORY','PREVIEW_CURSOR','DISPLAY_ENEMY_INFO','EMPTY'}
                            or a.get('key')=='enemy_1515_bsnake' for a in actions)
            branch_actions=[a for b in branches.values() for phase in b.get('phases',[]) for a in phase.get('actions',[])]
            valid=(set(branches)=={'bsnake_flame'} and branch_actions and
                   all(a['actionType']=='ACTIVATE_PREDEFINED' and a['key'].startswith('trap_021_flame#') for a in branch_actions))
            if can_trigger or not valid:raise NotImplementedError('Active or unreviewed branches')
            dormant_branches=tuple(sorted(branches))
            assumptions.append('BRANCH-001 dormant bsnake_flame definition has no supported triggering actor/action')
        for k in ["globalBuffs", "tilesDisallowToLocate", "enemies"]:
            if d.get(k):
                raise NotImplementedError(k)
        m = d["mapData"]
        if m.get("blockEdges"):
            raise NotImplementedError("blockEdges")
        tiles = tuple(
            TileData(
                t["tileKey"], t["heightType"], t["buildableType"], t["passableMask"],
                tuple((v['key'],v['value']) for v in t.get('blackboard') or [] if v.get('value') is not None)
            )
            for t in m["tiles"]
        )
        if not m["map"] or not m["map"][0] or len({len(r) for r in m["map"]}) != 1:
            raise ValueError("Invalid map dimensions")
        if any(i < 0 or i >= len(tiles) for r in m["map"] for i in r):
            raise ValueError("Invalid tile index")
        grid = MapData(
            tuple(tuple(tiles[i] for i in row) for row in reversed(m["map"]))
        )
        routes = []
        for r in d["routes"]:
            if r is None:
                raise NotImplementedError("Null route entries require route-index preservation")
            wp = []
            for w in r.get("checkpoints") or []:
                if w["type"] not in ROUTE_KINDS:
                    raise NotImplementedError(w["type"])
                if w.get("randomizeReachOffset") or w.get("reachDistance", 0):
                    raise NotImplementedError("Checkpoint reach geometry")
                if w["time"] < 0:
                    raise ValueError("Negative wait")
                wp.append(
                    WaypointData(
                        w["type"], point(w["position"], w.get("reachOffset")), w["time"]
                    )
                )
            routes.append(
                RouteData(
                    point(r["startPosition"], r.get("spawnOffset")),
                    point(r["endPosition"]),
                    tuple(wp),
                    r["motionMode"],
                    r["allowDiagonalMove"],
                    (r["spawnRandomRange"]["x"], r["spawnRandomRange"]["y"]),
                )
            )
            hidden = False
            for checkpoint in wp:
                if checkpoint.kind == "DISAPPEAR":
                    if hidden: raise ValueError("Nested route disappearance")
                    hidden = True
                elif checkpoint.kind == "APPEAR_AT_POS":
                    if not hidden: raise ValueError("Route appearance without disappearance")
                    hidden = False
                elif checkpoint.kind == "MOVE" and hidden:
                    raise NotImplementedError("Movement while disappeared")
            if hidden: raise NotImplementedError("Route ends while disappeared")
        if any(w.kind.startswith("WAIT_CURRENT_") for r in routes for w in r.waypoints):
            assumptions.append("ROUTE-WAIT-001 current active wave/fragment clock at checkpoint entry; client parity unverified")
        from .event_graph import parse_event_waves
        waves = parse_event_waves(d["waves"], routes)
        if len(waves) > 1 or any(a.block_fragment for w in waves for f in w.fragments for a in f.actions):
            assumptions.append("EVENT-GATE-001 clear/timeout scheduling implemented; official boundary timing unverified")
        used = {s.enemy_id for w in waves for s in w.spawns}
        enemies = tuple(
            self.enemy_loader.load(r) for r in d["enemyDbRefs"] if r["id"] in used
        )
        enemies=tuple(replace(e,hp=e.hp*hp_scale,atk=e.atk*atk_scale,defense=e.defense*def_scale) for e in enemies)
        if used != {e.id for e in enemies}:
            raise ValueError("Missing enemy reference")
        o = d["options"]
        if o["costIncreaseTime"] <= 0 or o["moveMultiplier"] <= 0:
            raise NotImplementedError("Non-positive time/movement")
        return StageData(
            name,
            grid,
            tuple(routes),
            tuple(waves),
            enemies,
            o["initialCost"],
            o["maxCost"],
            o["costIncreaseTime"],
            o["moveMultiplier"],
            o["maxLifePoint"] if life_override is None else life_override,
            o["characterLimit"],
            tuple(assumptions),
            devices,difficulty,dormant_branches,
        )
