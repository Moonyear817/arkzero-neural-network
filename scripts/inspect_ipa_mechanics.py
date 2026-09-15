"""Read a user-provided IPA; preserve originals and export source parameter trees."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import UnityPy
from UnityPy.helpers import CompressionHelper
from UnityPy.enums.BundleFile import CompressionFlags
from lz4ak_read import decompress_lz4ak


def main():
    parser=argparse.ArgumentParser();parser.add_argument('ipa',type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'research/mechanics/ipa';out.mkdir(parents=True,exist_ok=True)
    CompressionHelper.DECOMPRESSION_MAP[CompressionFlags.LZHAM]=decompress_lz4ak
    records=[]
    with zipfile.ZipFile(args.ipa) as z:
        selected=[n for n in z.namelist() if n.endswith('.ab') and any(k in n for k in ('[uc]tiles.ab','[uc]envsystems.ab','[uc]dynamicabilities.ab','[uc]globalbuffs.ab','[uc]projectiles.ab','[uc]skills.ab','battle/[pack]common.ab','enm_pfb_base_0.ab','config/common.ab'))]
        for n in selected:
            raw=z.read(n);name=Path(n).name;(out/name).write_bytes(raw)
            record=dict(source=n,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            try:
                env=UnityPy.load(raw);objects=[]
                for obj in env.objects:
                    row=dict(type=obj.type.name,path_id=obj.path_id)
                    try:row['data']=obj.read_typetree()
                    except Exception as exc:row['error']=str(exc)
                    objects.append(row)
                (out/(name+'.json')).write_text(json.dumps(objects,ensure_ascii=False,default=str),encoding='utf-8')
                record.update(objects=len(objects),parsed=sum('data' in o for o in objects))
            except Exception as exc:record['error']=str(exc)
            records.append(record);print(record,flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(ipa=str(args.ipa),resources=records),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
