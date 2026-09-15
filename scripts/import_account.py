"""Recognize a screenshot or confirm a manually reviewed account draft."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from accounts.ocr import recognize
from accounts.snapshot import confirm_snapshot
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    o=s.add_parser('recognize');o.add_argument('image');o.add_argument('--output',required=True)
    c=s.add_parser('confirm');c.add_argument('draft');c.add_argument('--confirmed-by',required=True)
    c.add_argument('--directory',default='data/accounts/snapshots')
    a=p.parse_args();d=Path(__file__).resolve().parents[1]/'data/real'
    loader=OperatorLoader(d/'character_table.json',d/'range_table.json',SkillLoader(d/'skill_table.json'))
    if a.command=='recognize':
        result=recognize(a.image,loader);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(out)
    else:print(confirm_snapshot(json.loads(Path(a.draft).read_text()),loader,a.directory,confirmed_by=a.confirmed_by))

if __name__=='__main__':main()
