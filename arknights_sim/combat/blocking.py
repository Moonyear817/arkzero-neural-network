import math


def release(enemy, operators):
    if enemy.blocked_by in operators:
        op = operators[enemy.blocked_by]
        if enemy.id in op.blocked_enemies:
            op.blocked_enemies.remove(enemy.id)
    enemy.blocked_by = None


def try_block(enemy, operators, enemies, time):
    if enemy.blocked_by or enemy.hidden or enemy.data.faction != "HOSTILE" or enemy.data.motion == "FLY":
        return None
    for op in sorted(operators.values(), key=lambda o: (o.deployed_at, o.id)):
        used = sum(enemies[e].data.block_weight for e in op.blocked_enemies)
        from arknights_sim.skills.modifier import modified

        capacity = modified(op.data.block_count, "BLOCK_COUNT", op.modifiers, time)
        if (
            op.alive
            and used + enemy.data.block_weight <= capacity
            and math.dist(enemy.position, op.position) <= 0.5
        ):
            enemy.blocked_by = op.id
            op.blocked_enemies.append(enemy.id)
            return op
    return None
