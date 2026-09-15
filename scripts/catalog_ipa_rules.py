"""Index decoded prefab rules without executing their serialized action programs."""
import collections,json
from pathlib import Path

root=Path(__file__).resolve().parents[1];source=root/'research/mechanics/ipa';rows=[];types=collections.Counter()

def visit(value):
    if isinstance(value,dict):
        if '$type' in value:types[value['$type']]+=1
        for k,v in value.items():
            if k=='SerializedState' and isinstance(v,str):
                try:visit(json.loads(v))
                except ValueError:pass
            else:visit(v)
    elif isinstance(value,list):
        for v in value:visit(v)

for path in source.glob('*.ab.json'):
    data=json.loads(path.read_text());lookup={v['path_id']:v for v in data}
    for r in data:
        visit(r.get('data'))
        if r['type']!='GameObject':continue
        d=r['data'];components=[]
        for c in d.get('m_Component',[]):
            obj=lookup.get(c.get('component',{}).get('m_PathID'))
            if obj and obj['type']=='MonoBehaviour':components.append(obj)
        if components:rows.append(dict(name=d['m_Name'],source=path.name,path_id=r['path_id'],components=components))
out=root/'data/mechanics';out.mkdir(exist_ok=True)
(out/'ipa_rule_catalog.json').write_text(json.dumps(dict(version='IPA 2.7.71',objects=rows,action_node_types=dict(types)),ensure_ascii=False,indent=2))
print('Rule objects',len(rows),'Action node types',len(types))
