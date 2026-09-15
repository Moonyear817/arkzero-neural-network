"""Report exact readiness; does not start training or alter map capability flags."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from accounts.readiness import curriculum_audit
from accounts.snapshot import load_snapshot
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot');p.add_argument('--output');a=p.parse_args()
    d=Path(__file__).resolve().parents[1]/'data/real'
    loader=OperatorLoader(d/'character_table.json',d/'range_table.json',SkillLoader(d/'skill_table.json'))
    snapshot=load_snapshot(a.snapshot,loader) if a.snapshot else None
    report=curriculum_audit(d,snapshot,loader)
    text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if a.output:
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
    else:print(text)

if __name__=='__main__':main()
