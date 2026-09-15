"""Trust percent -> battle phase -> attribute bonuses from pinned GameData."""
import math


def trust_bonuses(character, favor_table, trust):
    if type(trust) not in (int, float) or not math.isfinite(trust) or not 0 <= trust <= 200:
        raise ValueError('Trust must be a percentage between 0 and 200')
    frames = character.get('favorKeyFrames') or []
    if not frames or trust == 0:
        return {}
    rows = favor_table['favorFrames']
    phase = max((r['data']['battlePhase'] for r in rows if r['data']['percent'] <= trust), default=0)
    ordered = sorted(frames, key=lambda f: f['level'])
    lo = max((f for f in ordered if f['level'] <= phase), key=lambda f:f['level'], default=ordered[0])
    hi = next((f for f in ordered if f['level'] >= phase), ordered[-1])
    ratio = max(0, min(1, (phase-lo['level']) / max(1, hi['level']-lo['level'])))
    result = {}
    for key, base in lo['data'].items():
        if isinstance(base, bool):
            if base or hi['data'].get(key):
                raise NotImplementedError('Trust flag: ' + key)
            continue
        value = base + ratio * (hi['data'].get(key, base)-base)
        if value:
            if key not in {'maxHp', 'atk', 'def', 'magicResistance'}:
                raise NotImplementedError('Trust attribute: ' + key)
            result[key] = int(value+0.5) if key != 'magicResistance' else value
    return result
