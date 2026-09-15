import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from accounts.snapshot import load_snapshot
from accounts.readiness import curriculum_audit
from accounts.battle_inputs import load_account_operator
parser=argparse.ArgumentParser()
parser.add_argument('--snapshot',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1];data=root/'data/real'
loader=OperatorLoader(data/'character_table.json',data/'range_table.json',SkillLoader(data/'skill_table.json'))
snapshot=load_snapshot(args.snapshot,loader)
report=curriculum_audit(data,snapshot,loader)
for record,row in zip(report['operators'],snapshot['operators']):
 raw=loader.chars[row['operator_id']]
 record['source_trait']=raw.get('trait');record['source_talents']=raw.get('talents');record['equipped_module']=row['module']
 for skill in record['skills']:
  try:
   op=load_account_operator(snapshot,loader,row['operator_id'],skill['id'])
   skill['account_input_bound']=True;skill['simulation_notes']=op.simulation_notes
  except (ValueError,NotImplementedError) as exc:
   skill['account_input_bound']=False;skill['binding_error']=str(exc)
 report['snapshot_id']=snapshot['snapshot_id']
out=root/'outputs/guards_chapter8';(out/'confirmed_capability_matrix.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Guards:',len(report['operators']),'skill inputs bound:',sum(s['account_input_bound'] for r in report['operators'] for s in r['skills']))
