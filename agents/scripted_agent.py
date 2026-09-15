"""V0.1 historical regression only. Search must never import this module.

The external scheduler deliberately uses advance_to_time: historical action times
are not environment decision events and are never taught to a search policy.
"""

from dataclasses import dataclass

from arknights_sim.environment import Action, ActionType, Direction

from .base import BaseAgent


@dataclass(frozen=True)
class ScheduledAction:
    time: float
    action: Action


class ScriptedAgent(BaseAgent):
    """Reproduce the existing Noir Corne + Melantha battle as a regression."""

    schedule = (
        ScheduledAction(4.0, Action(ActionType.DEPLOY, "char_500_noirc", (3, 2), Direction.RIGHT)),
        ScheduledAction(17.0, Action(ActionType.DEPLOY, "char_208_melan", (1, 2), Direction.RIGHT)),
        ScheduledAction(67.0, Action(ActionType.ACTIVATE_SKILL, "char_208_melan")),
    )

    def __init__(self):
        self.index = 0

    def reset(self):
        self.index = 0

    def prepare_state(self, env, state):
        if self.index >= len(self.schedule) or env.is_terminal(state):
            return state
        scheduled = self.schedule[self.index]
        if state.game.current_time > scheduled.time + 1e-8:
            raise ValueError("Historical regression action time has already passed")
        return env.advance_to_time(state, scheduled.time)

    def select_action(self, env, state):
        actions = env.legal_actions(state)
        if self.index >= len(self.schedule):
            action = Action(ActionType.WAIT)
        else:
            scheduled = self.schedule[self.index]
            if abs(state.game.current_time - scheduled.time) > 1e-8:
                raise ValueError("Call ScriptedAgent.prepare_state before select_action")
            action = scheduled.action
        if action not in actions:
            raise ValueError(f"Historical regression action is no longer legal: {action}")
        if self.index < len(self.schedule):
            self.index += 1
        return action
