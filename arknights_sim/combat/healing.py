def heal(unit, amount):
    if amount < 0:
        raise ValueError("Negative healing")
    if unit.alive and getattr(unit.data, "healable", True):
        unit.hp = min(unit.data.hp, unit.hp + amount)
