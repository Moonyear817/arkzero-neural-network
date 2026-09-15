"""Validate every mirrored map and build the lightweight offline UI index."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arknights_sim.data.enemy_loader import EnemyLoader
from arknights_sim.data.map_catalog import recognize
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.map.route import compile_route
from import_prts_maps import save_json


def main():
    directory=ROOT/'data/maps'
    manifest=json.loads((directory/'download_manifest.json').read_text())
    if not manifest.get('complete'):raise SystemExit('Complete the map download before indexing')
    index=json.loads((directory/'site_index.json').read_text())
    names={}
    for match in re.finditer(r"JSON\.parse\(('(?:\\.|[^'\\])*')\)",(directory/'site_bundle.js').read_text()):
        try:value=json.loads(ast.literal_eval(match.group(1)))
        except (ValueError,SyntaxError):continue
        if isinstance(value,dict) and isinstance(value.get('main_00-01'),dict) and 'zh_CN' in value['main_00-01']:
            names=value;break
    enemy_path=directory/'levels/enemydata/enemy_database.json'
    enemies=EnemyLoader(enemy_path)
    loader=StageLoader(enemy_path)
    by_path={}
    failures={}
    for position,path in enumerate(sorted({v['data_path'] for v in index.values()}),1):
        raw=json.loads((directory/'levels'/path).read_text())
        try:
            data=recognize(raw,enemies)
            grid=data['map']
            supported=False
            reason=''
            try:
                stage=loader.parse(raw,Path(path).stem)
                for ri in {s.route_index for s in stage.spawns}:compile_route(stage.map,stage.routes[ri])
                supported=True
            except (NotImplementedError,ValueError,KeyError,TypeError,IndexError,ZeroDivisionError) as exc:
                reason=str(exc)
            spawns=[g for g in data['groups'] if g['action'].get('actionType')=='SPAWN']
            errors=[dict(id=e['id'],error=e['error']) for e in data['enemies'] if e['error']]
            by_path[path]=dict(width=grid.width,height=grid.height,waves=len(raw.get('waves') or []),routes=len(data['routes']),spawn_groups=len(spawns),declared_spawn_count=sum(g['action'].get('count',0) for g in spawns),enemy_types=len(data['enemies']),enemy_errors=errors,simulation_supported=supported,support_reason=reason,sha256=manifest['files'][path]['sha256'])
        except Exception as exc:
            failures[path]=str(exc)
        if position%250==0:print(f'Indexed {position}; recognition errors {len(failures)}',flush=True)
    if failures:
        save_json(directory/'recognition_errors.json',failures)
        raise SystemExit(f'Map recognition failed for {len(failures)} files')
    records={}
    for key,entry in index.items():
        record=dict(by_path[entry['data_path']],**entry,id=key,name=names.get(key,{}).get('zh_CN',{}).get('name',entry['code']),description=names.get(key,{}).get('zh_CN',{}).get('description',''),source_url='https://map.ark-nights.com/map/'+key.replace('#','%23'))
        if key in ('main_03-07','main_03-07#f#'):
            try:
                stage=loader.load(directory/'levels'/entry['data_path'],entry['difficulty'])
                from arknights_sim.core.simulator import Simulator
                Simulator(stage)
                record.update(simulation_supported=True,support_reason='',fidelity='verification',mechanics=['障碍物改道','侦测器开关','隐匿','远程攻击','炮弹溅射','突袭词条'])
            except (NotImplementedError,ValueError,KeyError,TypeError,IndexError) as exc:
                record.update(simulation_supported=False,support_reason=str(exc))
        elif entry['difficulty']!='NORMAL':
            record.update(simulation_supported=False,support_reason='此难度的符文/词条尚未实现；保留原始定义，不套用普通难度战斗')
        elif record['simulation_supported'] and key not in ('main_00-01','main_00-08','main_00-09'):
            record.update(simulation_supported=False,support_reason='基础结构可解析；此关卡的地形和事件机制尚未完成战斗验收')
        records[key]=record
    summary=dict(schema_version=1,indexed_at=datetime.now(timezone.utc).isoformat(),source='https://map.ark-nights.com',variants=len(records),unique_maps=len(by_path),enemy_database_entries=len(enemies.entries),runnable_variants=sum(r['simulation_supported'] for r in records.values()),unresolved_enemy_references=sum(len(r['enemy_errors']) for r in by_path.values()),download_manifest_sha256=hashlib.sha256((directory/'download_manifest.json').read_bytes()).hexdigest(),support_reasons=dict(Counter(r['support_reason'] for r in records.values() if not r['simulation_supported'])))
    save_json(directory/'catalog.json',dict(summary,stages=records))
    save_json(directory/'recognition_report.json',summary)
    save_json(directory/'recognition_errors.json',{})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
