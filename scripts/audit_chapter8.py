#!/usr/bin/env python3
"""Reproducible, independent-component audit; never opens training gates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accounts.readiness import chapter_eight_records
from arknights_sim.data.map_catalog import MapCatalog
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.data.event_graph import parse_event_waves
from arknights_sim.data.models import RouteData
from arknights_sim.mechanics.definitions import parse_devices
from arknights_sim.map.route import ROUTE_KINDS


def audit(directory):
    catalog = MapCatalog(directory)
    loader = StageLoader(catalog.enemy_path)
    stages=[]
    for record in chapter_eight_records(catalog):
        path=catalog.path(record); raw=json.loads(path.read_text())
        blockers=[]
        def check(component, fn):
            try:
                return fn()
            except (NotImplementedError, ValueError, KeyError) as exc:
                blockers.append({'component':component,'reason':str(exc)})
                return None
        loaded=check('full_stage',lambda:loader.load(path))
        if loaded is not None:
            from arknights_sim import Simulator
            constructed=check('runtime_paths',lambda:Simulator(loaded))
        else:constructed=None
        # Each component is inspected even when the full loader fails early.
        check('predefined_units_and_devices',lambda:parse_devices(raw))
        enemies=[]
        for ref in raw['enemyDbRefs']:
            data=check('enemy:'+ref['id'],lambda ref=ref:loader.enemy_loader.load(ref))
            resolved=loader.enemy_loader.resolve(ref)
            enemies.append({'id':ref['id'],'basic_loader_supported':data is not None,
                            'skills':resolved.get('skills'), 'talents':resolved.get('talentBlackboard')})
        route_kinds=sorted({w['type'] for r in raw['routes'] if r for w in r.get('checkpoints') or []})
        for kind in sorted(set(route_kinds)-ROUTE_KINDS):
            blockers.append({'component':'route','reason':kind})
        routes=tuple(RouteData((0,0),(0,0),motion=r['motionMode']) if r else RouteData((0,0),(0,0)) for r in raw['routes'])
        waves=check('event_scheduler',lambda:parse_event_waves(raw['waves'],routes))
        groups=[]
        for wi,wave in enumerate(raw['waves']):
            for fi,fragment in enumerate(wave['fragments']):
                for ai,action in enumerate(fragment['actions']):
                    groups.append({'wave':wi,'fragment':fi,'action':ai,**action})
        for rune in raw.get('runes') or []:
            if rune['difficultyMask'] in ('ALL','NORMAL') and rune['key'] not in {'gbuff_lifepoint','ebuff_attribute','cbuff_token_initial_cnt'}:
                blockers.append({'component':'normal_rune','reason':rune['key']})
        stages.append({'code':record['code'],'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'loader_supported':loaded is not None,'runtime_constructed':constructed is not None,'formal_training_ready':False,
            'wave_count':len(raw['waves']),'spawn_group_count':sum(a['actionType']=='SPAWN' for a in groups),
            'fragment_clear_groups':[a for a in groups if a.get('blockFragment')],
            'waves':[{'index':i,'pre_delay':w['preDelay'],'post_delay':w['postDelay'],
                      'clear_timeout':w['maxTimeWaitingForNextWave'],'fragments':w['fragments']} for i,w in enumerate(raw['waves'])],
            'event_parser_supported':waves is not None,'route_checkpoint_kinds':route_kinds,
            'enemies':enemies,'blockers':blockers})
    return {'scope':'chapter8_ordinary','stages':stages,
            'note':'Loader support is not combat fidelity. Gates and relative route clock semantics remain ASSUMED until client comparison.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/real');parser.add_argument('--output',default='outputs/guards_chapter8/scenarios')
    args=parser.parse_args();report=audit(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'capability_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 第八章普通关卡逐项检查','', '可解析不等于战斗规则完整；未通过机制验收的关卡不进入正式训练。','',
           '| 关卡 | 波数 | 出怪组 | 清场阻塞组 | 调度可解析 | 完整加载 | 当前缺失 |', '|---|---:|---:|---:|---|---|---|']
    for r in report['stages']:
        reasons='；'.join(dict.fromkeys(b['reason'] for b in r['blockers']))
        lines.append(f"| {r['code']} | {r['wave_count']} | {r['spawn_group_count']} | {len(r['fragment_clear_groups'])} | {r['event_parser_supported']} | {r['loader_supported']} | {reasons} |")
    lines+=['','各组原始延时、清场条件、敌人技能和来源文件校验值见 capability_audit.json。',
            '不能把所有出怪组统一改为清场触发；blockFragment、波次清场与组内延时必须分别执行。']
    (out/'capability_audit.md').write_text('\n'.join(lines)+'\n')
    print(f"Audited {len(report['stages'])} ordinary stages; {sum(r['loader_supported'] for r in report['stages'])} fully loadable.")

if __name__=='__main__':main()
