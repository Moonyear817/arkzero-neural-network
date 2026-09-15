"""Shared, strategy-free episode runner and decision records."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from arknights_sim.environment import Action


@dataclass(frozen=True)
class DecisionRecord:
    decision: int
    time: float
    action: Action
    legal_action_count: int
    next_state_key: str
    external_advance_to: float | None = None
    next_decision_event: dict | None = None

    def to_dict(self):
        record = {
            "decision": self.decision,
            "time": self.time,
            "action": self.action.to_dict(),
            "legal_action_count": self.legal_action_count,
            "next_state_key": self.next_state_key,
        }
        if self.external_advance_to is not None:
            record["external_advance_to"] = self.external_advance_to
        if self.next_decision_event is not None:
            record["next_decision_event"] = self.next_decision_event
        return record


@dataclass(frozen=True)
class Episode:
    final_state: Any
    history: tuple[DecisionRecord, ...]
    wall_time: float

    @property
    def decisions(self):
        return len(self.history)


def run_episode(
    env,
    agent,
    state=None,
    *,
    stage_id="0-1",
    squad=None,
    seed=12345,
    max_decisions=1000,
    reset_agent=True,
    record_keys=True,
):
    """Run until terminal; a decision limit is an error, never a false success.

    ``prepare_state`` provides explicit external scheduling for regression agents.
    Normal agents use the identity hook and advance time solely via WAIT. Each
    record includes its actual execution time, including externally scheduled
    baseline actions. MCTS solution replay uses only its ordinary event actions.
    """
    if state is None:
        state = env.reset(stage_id=stage_id, squad=squad, seed=seed)
    if reset_agent and hasattr(agent, "reset"):
        agent.reset()
    started = perf_counter()
    history = []
    while not env.is_terminal(state):
        if len(history) >= max_decisions:
            raise RuntimeError(
                f"Episode exceeded {max_decisions} decisions without terminating"
            )
        before_prepare_time = state.game.current_time
        if hasattr(agent, "prepare_state"):
            state = agent.prepare_state(env, state)
        external_advance_to = (
            state.game.current_time
            if state.game.current_time != before_prepare_time
            else None
        )
        if env.is_terminal(state):
            break
        legal = env.legal_actions(state)
        action = agent.select_action(env, state)
        if action not in legal:
            raise ValueError(f"Agent selected an illegal action: {action}")
        current_time = state.game.current_time
        state = env.step(state, action)
        history.append(
            DecisionRecord(
                len(history),
                current_time,
                action,
                len(legal),
                env.state_key(state) if record_keys else "",
                external_advance_to,
                state.last_decision.to_dict()
                if hasattr(state, "last_decision")
                else None,
            )
        )
    return Episode(state, tuple(history), perf_counter() - started)
