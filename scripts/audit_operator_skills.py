"""Exhaustive roster skill coverage; data presence is not effect implementation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from arknights_sim.data.skill_description import describe_level


def audit(directory):
    skills=SkillLoader(directory/'skill_table.json')
    loader=OperatorLoader(directory/'character_table.json',directory/'range_table.json',skills)
    records=[]
    from arknights_sim.data.summon_catalog import summon_records
    summons=summon_records(loader)
    for key in loader.available_ids():
        char=loader.chars[key]
        for index,ref in enumerate(char['skills']):
            skill_id=ref['skillId'];entry=skills.data.get(skill_id)
            levels=[]
            for rank,raw in enumerate(entry['levels'] if entry else (),1):
                try:
                    skill=skills.load(skill_id,rank);status='IMPLEMENTED_SIMPLIFIED'
                    reason=('Owner-linked summon ATK/DEF and HP regeneration; no modules; lifecycle/timing unverified.' if skill.effect_target=='OWNER_SUMMONS' else 'Existing timed self-stat modifier engine; official timing/stacking unverified.')
                except NotImplementedError as exc:
                    status='UNSUPPORTED';reason=str(exc)
                levels.append(dict(rank=rank,name=raw['name'],description=describe_level(raw),raw_description=raw['description'],blackboard=raw['blackboard'],sp=raw['spData'],duration=raw['duration'],range_id=raw.get('rangeId'),status=status,reason=reason))
            records.append(dict(operator_id=key,operator_name=loader.catalog.get(key,{}).get('name',char['name']),skill_index=index,skill_id=skill_id,unlock=ref['unlockCond'],mastery_unlock=ref.get('levelUpCostCond',[]),levels=levels))
    unique={r['skill_id'] for r in records}
    supported={r['skill_id'] for r in records if r['levels'] and all(x['status']=='IMPLEMENTED_SIMPLIFIED' for x in r['levels'])}
    return dict(summon_relations=len(summons),summon_owners=len({r["owner_id"] for r in summons}),summons=summons,operators=len(loader.available_ids()),operators_without_skills=sum(not loader.chars[k]['skills'] for k in loader.available_ids()),operator_skill_slots=len(records),unique_skills=len(unique),implemented_simplified_unique=len(supported),unsupported_unique=len(unique-supported),missing_skill_data=sorted({r['skill_id'] for r in records if not r['levels']}),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (directory/'character_table.json',directory/'char_patch_table.json',directory/'skill_table.json')},records=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='research/skills/coverage.json');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];result=audit(root/'data/real');out=root/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','summons')},ensure_ascii=False,indent=2))
