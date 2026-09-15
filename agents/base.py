"""Minimal agent interface, independent of simulator implementation details."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arknights_sim.environment import Action, ArknightsEnv, EnvState


class BaseAgent(ABC):
    """Choose one legal action; randomness belongs to the agent, not GameState."""

    def reset(self):
        """Restore episode-local state before an independent evaluation."""

    def prepare_state(self, env: "ArknightsEnv", state: "EnvState") -> "EnvState":
        """Optional external scheduling hook, used only by regression baselines."""
        return state

    @abstractmethod
    def select_action(self, env: "ArknightsEnv", state: "EnvState") -> "Action":
        """Return an action contained in env.legal_actions(state)."""
