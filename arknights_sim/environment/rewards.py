"""Shared battle/squad objective, versioned independently from network shapes.

Undiscounted progress differences telescope: loops cannot mint reward. The
network predicts remaining return, not the score already earned in the past.
"""

REWARD_VERSION = 2
KILL_WEIGHT = 0.2
LIFE_WEIGHT = 0.2
PERFECT_BONUS = 0.6
WIN_BONUS = 0.2
LOSS_BONUS = -0.6
TRUNCATION_BOUND = -0.8


def progress(metrics):
    total = max(1, metrics['total_enemies'])
    initial_life = max(1, metrics['initial_life'])
    killed = min(1.0, max(0.0, metrics['kills'] / total))
    lost = min(1.0, max(0.0, (initial_life - metrics['life']) / initial_life))
    return KILL_WEIGHT * killed - LIFE_WEIGHT * lost


def is_truncated(metrics):
    return metrics['termination'] in ('decision_limit', 'time_limit')


def terminal_bonus(metrics):
    if is_truncated(metrics):
        # Search-only pessimistic bound. Adding previous progress makes the
        # complete score -0.8, so timing out cannot improve on a natural loss.
        # Never use this artificial boundary as a supervised outcome.
        return TRUNCATION_BOUND - progress(metrics)
    if metrics['termination'] != 'battle_end':
        raise ValueError('A terminal bonus requires a completed battle')
    if metrics['success']:
        return PERFECT_BONUS
    return WIN_BONUS if metrics['simulator_result'] == 'WIN' else LOSS_BONUS


def episode_reward(metrics):
    return terminal_bonus(metrics) + progress(metrics)


def state_progress(state):
    game = state.game
    return progress(dict(total_enemies=game.stage.hostile_count, kills=game.killed,
                         initial_life=game.stage.life, life=game.life))
