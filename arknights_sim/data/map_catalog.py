"""Offline PRTS.Map recognition, independent of battle-mechanics support."""
import json
from pathlib import Path, PurePosixPath

from .enemy_loader import EnemyLoader
from .models import MapData, TileData
from .stage_loader import StageLoader


def decode_map(raw):
    source = raw['mapData']
    tiles = [TileData(t['tileKey'], t['heightType'], t['buildableType'], t['passableMask']) for t in source['tiles']]
    rows = source['map']
    if not rows or not rows[0] or len({len(r) for r in rows}) != 1:
        raise ValueError('Invalid map dimensions')
    if any(type(i) is not int or i < 0 or i >= len(tiles) for row in rows for i in row):
        raise ValueError('Invalid tile reference')
    return MapData(tuple(tuple(tiles[i] for i in row) for row in reversed(rows)))


def recognize(raw, enemy_loader):
    grid = decode_map(raw)
    groups = []
    for wi, wave in enumerate(raw.get('waves') or []):
        for fi, fragment in enumerate(wave.get('fragments') or []):
            for ai, action in enumerate(fragment.get('actions') or []):
                groups.append(dict(wave=wi,fragment=fi,order=ai,wave_pre_delay=wave.get('preDelay',0),wave_post_delay=wave.get('postDelay',0),fragment_pre_delay=fragment.get('preDelay',0),action=action))
    enemies = []
    for ref in raw.get('enemyDbRefs') or []:
        try:
            data = enemy_loader.resolve(ref)
            enemies.append(dict(id=ref['id'],level=ref.get('level'),resolved=data,reference=ref,error=None))
        except (KeyError,ValueError,TypeError) as exc:
            enemies.append(dict(id=ref.get('id',''),level=ref.get('level'),resolved=None,reference=ref,error=str(exc)))
    return dict(map=grid,groups=groups,enemies=enemies,routes=raw.get('routes') or [],
                runes=raw.get('runes') or [],branches=raw.get('branches') or [],
                predefines=raw.get('predefines') or {},options=raw.get('options') or {},
                global_buffs=raw.get('globalBuffs') or [])


class MapCatalog:
    def __init__(self, data_dir):
        self.directory = Path(data_dir).parent / 'maps'
        path = self.directory / 'catalog.json'
        self.metadata = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        self.records = self.metadata.get('stages', {})
        self._enemies = None

    def resolve(self, query):
        if query in self.records:
            return self.records[query]
        matches = [r for r in self.records.values() if (r['code'] or '').casefold() == query.casefold() and r['difficulty'] == 'NORMAL']
        if len(matches) != 1:
            raise ValueError(f"{'Ambiguous' if matches else 'Unknown'} map: {query}")
        return matches[0]

    def path(self, record):
        relative = PurePosixPath(record['data_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid map path')
        return self.directory / 'levels' / relative

    @property
    def enemy_path(self):
        return self.directory / 'levels/enemydata/enemy_database.json'

    def inspect(self, query):
        record = self.resolve(query)
        raw = json.loads(self.path(record).read_text(encoding='utf-8'))
        if self._enemies is None:
            self._enemies = EnemyLoader(self.enemy_path)
        return record, recognize(raw, self._enemies)

    def load_stage(self, query):
        record = self.resolve(query)
        if not record['simulation_supported']:
            raise NotImplementedError(f"此地图已收录，可查看布局、出怪和敌人资料；战斗尚不支持：{record['support_reason']}")
        return StageLoader(self.enemy_path).load(self.path(record),record['difficulty'])
