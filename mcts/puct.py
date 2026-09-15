"""PUCT selection and visit-count policies, with a caller-owned random source."""

import math


def action_group(action):
    """Use the policy's actual action type; generic toy actions stay one group."""
    kind = getattr(action, 'type', None)
    return getattr(kind, 'value', kind)


def puct_score(edge, parent_visits: int, c_puct: float) -> float:
    # The virtual first parent visit lets the prior guide the very first choice.
    return edge.Q + c_puct * edge.P * math.sqrt(max(1, parent_visits)) / (1 + edge.N)


def select_edge(node, c_puct: float, rng):
    if not node.edges:
        raise ValueError("Cannot select an action from an unexpanded node")
    actions = node.active_actions or tuple(node.edges)
    candidates = [node.edges[action] for action in actions]
    # Preserve the network's mass for each action type when representing it
    # with only a few leaves. Otherwise one of hundreds of DEPLOY leaves loses
    # its type's probability mass while a handful of WAIT leaves keeps theirs.
    active_mass = {}
    for edge in candidates:
        group = action_group(edge.action)
        active_mass[group] = active_mass.get(group, 0.0) + edge.P
    full_mass = node.group_priors
    if not full_mass:
        full_mass = {}
        for edge in node.edges.values():
            group = action_group(edge.action)
            full_mass[group] = full_mass.get(group, 0.0) + edge.P
    represented_mass = math.fsum(full_mass.get(group, 0.0) for group in active_mass)
    scored = []
    for edge in candidates:
        group = action_group(edge.action)
        conditional = (full_mass.get(group, 0.0) * edge.P / active_mass[group] / represented_mass
                       if active_mass[group] > 0 and represented_mass > 0 else 0.0)
        score = edge.Q + c_puct * conditional * math.sqrt(max(1, node.N)) / (1 + edge.N)
        scored.append((score, edge))
    best = max(score for score, _ in scored)
    return rng.choice([edge for score, edge in scored if score == best])


def visit_policy(visit_counts, temperature: float, rng) -> tuple[float, ...]:
    """Normalize N**(1/T); at T=0 use one seeded winner among maximum visits."""
    counts = tuple(visit_counts)
    if not counts or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError(
            "Visit counts must be a nonempty sequence of nonnegative integers"
        )
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    maximum = max(counts)
    if maximum == 0:
        if len(counts) == 1:
            return (1.0,)
        raise ValueError("A multi-action visit policy needs at least one simulation")
    if temperature == 0:
        winner = rng.choice([i for i, count in enumerate(counts) if count == maximum])
        return tuple(float(i == winner) for i in range(len(counts)))
    # Divide log counts by the maximum before exp to avoid overflow at low T.
    weights = [
        math.exp((math.log(n) - math.log(maximum)) / temperature) if n else 0.0
        for n in counts
    ]
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)


def sample_policy(actions, policy, rng):
    return rng.choices(tuple(actions), weights=policy, k=1)[0]
