import math


def range_cells(op):
    out = []
    data=op.data.skill
    cells=data.range_override if data and op.skill.active and data.range_override is not None else op.data.attack_range
    for x, y in cells:
        for _ in range(op.direction):
            x, y = -y, x
        out.append((op.position[0] + x, op.position[1] + y))
    return out


def in_range(op, enemy):
    if getattr(enemy, 'hidden', False): return False
    if getattr(enemy.data, 'motion', 'WALK') == 'FLY' and not op.data.can_attack_air: return False
    return enemy.blocked_by == op.id or any(
        abs(enemy.position[0] - x) <= 0.5 and abs(enemy.position[1] - y) <= 0.5
        for x, y in range_cells(op)
    )


class OperatorTargetSelector:
    @staticmethod
    def select(op, enemies, remaining):
        candidates = [
            e for e in enemies if e.alive and not e.escaped and e.data.faction == 'HOSTILE' and in_range(op, e)
        ]
        return min(
            candidates,
            key=lambda e: (e.blocked_by != op.id, remaining(e), e.id),
            default=None,
        )


class EnemyTargetSelector:
    @staticmethod
    def select(enemy, operators):
        op = operators.get(enemy.blocked_by)
        if op and op.alive:return op
        def reachable(o):
            radius = enemy.data.attack_range
            if getattr(o.data, 'faction', None) == 'ESCORT':
                radius = max(radius, enemy.data.escort_attack_range)
            return o.alive and not getattr(o,'hidden',False) and radius>0 and math.dist(o.position,enemy.position)<=radius
        return max((o for o in operators.values() if reachable(o)),key=lambda o:(
            sum(t.get('taunt_level') for t in getattr(o.data,'talents',()) if t.handler=='kazimierz_reflection'),
            enemy.data.prioritize_escorts and getattr(o.data,'faction',None)=='ESCORT',o.deployed_at,o.id),default=None)


class HealingTargetSelector:
    @staticmethod
    def select(op, allies):
        return min(
            (u for u in allies if u.alive and u.data.healable
             and not (u.skill.active and u.data.skill and u.data.skill.prohibit_healing)
             and u.hp < u.max_hp and in_range(op, u)),
            key=lambda u: (u.hp / u.max_hp, u.id), default=None,
        )
