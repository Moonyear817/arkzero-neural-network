"""Mirror every map listed by PRTS.Map, retaining variants and source checksums."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SITE = 'https://map.ark-nights.com'


def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={'User-Agent':'ArknightsZero-LocalMaps/0.3 (personal research)'}),timeout=40) as response:
                return response.read()
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def extract_index(script):
    for match in re.finditer(r"JSON\.parse\(('(?:\\.|[^'\\])*')\)", script):
        try:
            data = json.loads(ast.literal_eval(match.group(1)))
        except (ValueError, SyntaxError):
            continue
        if isinstance(data,dict) and data and all(isinstance(v,dict) and 'data_path' in v for v in data.values()):
            return data
    raise ValueError('PRTS.Map index not found; website format may have changed')


def safe_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.suffix != '.json':
        raise ValueError(f'Invalid map path: {value}')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not 1 <= args.workers <= 12: parser.error('workers must be 1–12')
    directory=ROOT/'data/maps'
    directory.mkdir(parents=True,exist_ok=True)
    index_path=directory/'site_index.json'
    if args.refresh or not index_path.exists():
        html=fetch(SITE+'/map/main_00-01')
        (directory/'site.html').write_bytes(html)
        scripts=re.findall(r'<script[^>]+src="([^"]+)',html.decode())
        bundle=next(s for s in scripts if 'bundle.' in s and '.esm.js' in s)
        raw=fetch(urljoin(SITE,bundle))
        (directory/'site_bundle.js').write_bytes(raw)
        save_json(index_path,extract_index(raw.decode()))
    index=json.loads(index_path.read_text())
    paths=sorted({str(safe_path(v['data_path'])) for v in index.values()})
    paths.append('enemydata/enemy_database.json')
    manifest_path=directory/'download_manifest.json'
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous=old.get('files',{})
    records={}
    errors={}

    def download(path):
        destination=directory/'levels'/path
        url=SITE+'/data/levels/'+path
        if not args.refresh and destination.exists():
            raw=destination.read_bytes()
            digest=hashlib.sha256(raw).hexdigest()
            if previous.get(path,{}).get('sha256') == digest:
                return path, previous[path]
        raw=fetch(url)
        data=json.loads(raw)
        if path != 'enemydata/enemy_database.json' and not all(k in data for k in ('mapData','waves','routes','enemyDbRefs')):
            raise ValueError(f'Map data incomplete: {path}')
        if path == 'enemydata/enemy_database.json' and not isinstance(data.get('enemies'),list):
            raise ValueError('Enemy database invalid')
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.json.tmp')
        temporary.write_bytes(raw)
        temporary.replace(destination)
        return path,dict(url=url,path=str(destination.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

    def checkpoint():
        save_json(manifest_path,dict(source=SITE,downloaded_at=datetime.now(timezone.utc).isoformat(),index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),variants=len(index),unique_maps=len(paths)-1,complete=len(records)==len(paths) and not errors,files=dict(sorted(records.items())),errors=errors))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(download,path):path for path in paths}
        for future in as_completed(futures):
            path=futures[future]
            try:
                _,record=future.result();records[path]=record
            except Exception as exc:
                errors[path]=str(exc)
                print(f'FAILED {path}: {exc}',flush=True)
            if (len(records)+len(errors)) % 100 == 0:
                checkpoint()
                print(f'Downloaded {len(records)}/{len(paths)}; errors {len(errors)}',flush=True)
    checkpoint()
    print(json.dumps(dict(variants=len(index),maps=len(paths)-1,downloaded=len(records),errors=errors),ensure_ascii=False),flush=True)
    if errors: raise SystemExit(1)


if __name__=='__main__':main()
