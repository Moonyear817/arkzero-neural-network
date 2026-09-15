"""Import the live PRTS roster and complete revision text, with resumable caching.

Wiki prose remains source data, never executable simulation logic. Battle inputs
are joined to the project's pinned GameData by the wiki's explicit character ID.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LIST_URL = 'https://prts.wiki/w/' + quote('干员一览')


class RosterParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'data-zh' in attrs:
            self.current = {k[5:]: v for k, v in attrs.items() if k.startswith('data-')}
            self.current['trait'] = ''
            self.rows.append(self.current)

    def handle_data(self, data):
        if self.current is not None:
            self.current['trait'] += data

    def handle_endtag(self, tag):
        if tag == 'div':
            self.current = None


def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'ArknightsZero-LocalCatalog/0.3 (personal research)'}), timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'data/real'
    cache = ROOT / 'data/prts'
    cache.mkdir(parents=True, exist_ok=True)
    html_path = cache / 'operator_list.html'
    if args.refresh or not html_path.exists():
        html_path.write_bytes(fetch(LIST_URL))
    roster = RosterParser()
    roster.feed(html_path.read_text(encoding='utf-8'))
    if not roster.rows or len({r['zh'] for r in roster.rows}) != len(roster.rows):
        raise ValueError('Empty or duplicated PRTS roster')
    pages_path = cache / 'pages.json'
    pages = json.loads(pages_path.read_text()) if pages_path.exists() and not args.refresh else {}
    missing = [r['zh'] for r in roster.rows if r['zh'] not in pages]
    for start in range(0, len(missing), 20):
        names = missing[start:start + 20]
        url = 'https://prts.wiki/api.php?' + urlencode(dict(action='query', prop='revisions', rvprop='ids|timestamp|content', rvslots='main', titles='|'.join(names), format='json', formatversion=2))
        response = json.loads(fetch(url))
        if 'error' in response or 'warnings' in response:
            raise ValueError(response)
        found = {p['title']: p for p in response['query']['pages']}
        normalized = {v['from']: v['to'] for v in response['query'].get('normalized', [])}
        for name in names:
            page = found[normalized.get(name, name)]
            if 'revisions' not in page:
                raise ValueError(f'Missing wiki page: {name}')
            pages[name] = page
        write_json(pages_path, pages)
        print(f'PRTS pages cached: {len(pages)}/{len(roster.rows)}', flush=True)
        time.sleep(0.3)
    characters = json.loads((directory / 'character_table.json').read_text())
    patch_path = directory / 'char_patch_table.json'
    if not patch_path.exists():
        manifest = json.loads((ROOT / 'data/manifest.json').read_text())
        source = next(v['url'] for v in manifest if v['path'].endswith('/character_table.json'))
        patch_url = source.replace('/character_table.json', '/char_patch_table.json')
        raw = fetch(patch_url)
        json.loads(raw)
        patch_path.write_bytes(raw)
        manifest.append(dict(path='data/real/char_patch_table.json', url=patch_url, sha256=hashlib.sha256(raw).hexdigest()))
        write_json(ROOT / 'data/manifest.json', manifest)
    characters.update(json.loads(patch_path.read_text()).get('patchChars', {}))
    records, unmatched = [], []
    for row in roster.rows:
        page = pages[row['zh']]
        revision = page['revisions'][0]
        source_text = revision['slots']['main']['content']
        match = re.search(r'^\|干员id\s*=\s*([^\s|}]+)', source_text, re.M)
        key = match.group(1) if match else None
        if key not in characters:
            unmatched.append({'name': row['zh'], 'id': key})
        # Preserve every source field, every wiki section and revision provenance.
        records.append(dict(id=key, name=row['zh'], aliases=list(dict.fromkeys([row['zh'], row.get('en', ''), row.get('ja', ''), row.get('id', ''), key or ''])), wiki=row, source_url='https://prts.wiki/w/' + quote(page['title']), revision_id=revision['revid'], revision_timestamp=revision['timestamp'], wikitext=source_text, game_data_available=key in characters))
    result = dict(schema_version=1, imported_at=datetime.now(timezone.utc).isoformat(), source_url=LIST_URL, list_sha256=hashlib.sha256(html_path.read_bytes()).hexdigest(), count=len(records), unmatched=unmatched, operators=records)
    write_json(directory / 'operator_catalog.json', result)
    print(json.dumps({'operators': len(records), 'unmatched': unmatched}, ensure_ascii=False), flush=True)
    if unmatched:
        raise SystemExit('Incomplete join: see operator_catalog.json unmatched')


if __name__ == '__main__':
    main()
