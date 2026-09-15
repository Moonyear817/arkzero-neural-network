"""Read-only skill descriptions, preserving unresolved placeholders explicitly."""
import re


def describe_level(data):
    blackboard = {entry['key'].lower(): entry['value'] for entry in data.get('blackboard', ())}
    def substitute(match):
        expression = match.group(1)
        key, _, format_spec = expression.partition(':')
        sign = -1 if key.startswith('-') else 1
        key = key.lstrip('-').lower()
        if key not in blackboard:
            return match.group(0)
        value = sign * blackboard[key]
        return f'{value * 100:g}%' if '%' in format_spec else f'{value:g}'
    text = re.sub(r'\{([^{}]+)\}', substitute, data.get('description') or '')
    return re.sub(r'<[^>]+>', '', text).replace('\\n', '\n')


def skill_details(loader, key):
    """All ranks, including unimplemented/locked skills; never a behavior parser."""
    entry = loader.data.get(key)
    if not entry:
        return 'UNKNOWN: missing skill data'
    lines = []
    for rank, level in enumerate(entry['levels'], 1):
        try:
            loader.load(key, rank)
            status = 'IMPLEMENTED_SIMPLIFIED — 已实现简化效果；非官方逐帧验证'
        except NotImplementedError as exc:
            status = 'UNSUPPORTED — ' + str(exc)
        sp = level['spData']
        lines.append(f"{'Rank '+str(rank) if rank<=7 else '专精 '+str(rank-7)} · {level['name']}\n{describe_level(level)}\n{status}\n{level['skillType']} / {sp['spType']} · SP {sp['initSp']} / {sp['spCost']} · {level['duration']:g}s")
    return '\n\n'.join(lines)
