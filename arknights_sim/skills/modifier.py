from dataclasses import dataclass


@dataclass(frozen=True)
class Modifier:
    source: str
    target: str
    kind: str
    value: float
    start_time: float
    end_time: float | None
    priority: int = 0


def modified(base, kind, modifiers, time):
    """Base-relative bonuses share a bucket; final multipliers remain distinct.

    PERCENT is used for source effects described as ATK/DEF/HP +x%. The legacy
    ADD and MULTIPLY contracts remain available to callers. More exotic
    stacking groups still require their own reviewed handlers.
    """
    ms = sorted(
        (m for m in modifiers if m.start_time <= time and (m.end_time is None or time < m.end_time)),
        key=lambda m: (m.priority, m.source),
    )
    value = base + sum(m.value for m in ms if m.kind == kind + "_ADD")
    value *= 1 + sum(m.value for m in ms if m.kind == kind + "_PERCENT")
    for m in ms:
        if m.kind == kind + "_MULTIPLY":
            value *= m.value
    return value
