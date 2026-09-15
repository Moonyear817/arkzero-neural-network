"""JSON-ready explanations shared by self-play and the desktop neural agent."""

from collections import Counter


ACTION_CATEGORIES = ('WAIT', 'DEPLOY', 'SKILL', 'RETREAT', 'DEVICE')


def action_category(action):
    kind = action.get('type', '')
    if kind in ('DEPLOY', 'DEPLOY_SUMMON'):
        return 'DEPLOY'
    if kind in ('RETREAT', 'RETREAT_SUMMON'):
        return 'RETREAT'
    if kind == 'ACTIVATE_SKILL':
        return 'SKILL'
    if kind == 'WAIT':
        return 'WAIT'
    return 'DEVICE'


def action_metrics(history):
    counts = Counter(action_category(row['action']) for row in history)
    result = {'action_counts': {key: counts[key] for key in ACTION_CATEGORIES}}
    result['action_percentages'] = {
        key: 100.0 * counts[key] / max(1, len(history)) for key in ACTION_CATEGORIES
    }
    for key in ACTION_CATEGORIES:
        result[f'{key.lower()}_percentage'] = result['action_percentages'][key]
    return result


def decision_explanation(env, state, result, top_k=8):
    graph = env.future_event_state(state) if hasattr(env, 'future_event_state') else {}
    progression = graph.get('progression', {})
    upcoming = []
    for event in graph.get('events', []):
        if event.get('remaining_count', 0) <= 0:
            continue
        upcoming.append({key: event.get(key) for key in (
            'event_id', 'event_type', 'wave_id', 'fragment_id', 'enemy_id',
            'route_id', 'remaining_count', 'status', 'trigger_condition',
            'dependencies', 'estimated_relative_time',
        )})
        if len(upcoming) == top_k:
            break
    order = sorted(range(len(result.actions)), key=lambda i: -result.neural_policy[i])[:top_k]
    type_policy = {}
    for action, probability in zip(result.actions, result.neural_policy):
        key = action.type.value
        type_policy[key] = type_policy.get(key, 0.0) + probability
    return {
        'time': state.game.current_time,
        'current_wave': progression.get('current_wave'),
        'current_fragment': progression.get('current_fragment'),
        'current_event_id': progression.get('current_event_id'),
        'progression': progression,
        'upcoming': upcoming,
        'policy': [dict(action=result.actions[i].to_dict(),
                        probability=result.neural_policy[i],
                        search_probability=result.policy[i],
                        visits=result.visit_counts[i]) for i in order],
        'action_type_policy': dict(sorted(type_policy.items(), key=lambda item: -item[1])),
        'search_candidates': [action.to_dict() for action in result.root.active_actions],
        'value': result.root_value,
        'selected_action': result.selected_action.to_dict(),
        'decision_reasons': list(state.last_decision.reasons),
    }
