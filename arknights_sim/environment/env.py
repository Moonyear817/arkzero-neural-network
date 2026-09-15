"""Pure transitions around V0.1. The simulator core has no dependency on agents."""

import copy, hashlib, json, math
from dataclasses import dataclass
from pathlib import Path
from arknights_sim.core.simulator import Simulator
from arknights_sim.core.state import GameState
from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from .action import Action, ActionType, WAIT
from .legal_actions import legal_actions
from .decision_events import DecisionEvent, signature, changed
from .result import EpisodeResult


@dataclass
class EnvState:
    game: GameState
    decision_count: int = 0
    max_decisions: int = 512
    horizon: float = 300
    last_decision: DecisionEvent = DecisionEvent(0, ("RESET",))
    trace: tuple[dict, ...] = ()
    trace_enabled: bool = False

    @property
    def terminal(self):
        return (
            self.game.done
            or self.decision_count >= self.max_decisions
            or self.game.current_time >= self.horizon - 1e-9
        )

    def clone(self):
        # Trace records are exposed to callers, so isolate them as well.
        return EnvState(
            self.game.clone(),
            self.decision_count,
            self.max_decisions,
            self.horizon,
            self.last_decision,
            copy.deepcopy(self.trace),
            self.trace_enabled,
        )


class ArknightsEnv:
    def __init__(
        self,
        stage=None,
        squad=None,
        *,
        data_dir=None,
        dt=1 / 60,
        windup=0.2,
        horizon=300,
        max_decisions=512,
        trace=False,
    ):
        if not math.isfinite(horizon) or horizon <= 0 or max_decisions <= 0:
            raise ValueError("Episode limits must be positive")
        if (
            dt <= 0
            or not math.isfinite(dt)
            or abs(horizon / dt - round(horizon / dt)) > 1e-6
        ):
            raise ValueError("Horizon must align with dt")
        self.stage = stage
        self.squad = tuple(squad) if squad is not None else None
        self.data_dir = (
            Path(data_dir)
            if data_dir
            else Path(__file__).resolve().parents[2] / "data/real"
        )
        self.dt = dt
        self.windup = windup
        self.horizon = horizon
        self.max_decisions = max_decisions
        self.trace = trace
        self._stage_cache = {}

    def reset(self, stage_id="0-1", squad=None, seed=12345):
        stage = self.stage
        if stage is None:
            stage = self._stage_cache.get(stage_id)
        if stage is None:
            from arknights_sim.data.map_catalog import MapCatalog
            catalog = MapCatalog(self.data_dir)
            if stage_id in catalog.records:
                stage = catalog.load_stage(stage_id)
        if stage is None:
            filename = "level_main_00-01.json" if stage_id == "0-1" else stage_id
            if not filename.endswith(".json"):
                filename += ".json"
            stage = StageLoader(self.data_dir / "enemy_database.json").load(
                self.data_dir / filename
            )
        self._stage_cache[stage_id] = stage
        team = tuple(squad) if squad is not None else self.squad
        if team is None:
            loader = OperatorLoader(
                self.data_dir / "character_table.json",
                self.data_dir / "range_table.json",
                SkillLoader(self.data_dir / "skill_table.json"),
            )
            team = tuple(
                loader.load(key) for key in ("char_500_noirc", "char_208_melan")
            )
        sim = Simulator(
            stage, team, seed=seed, dt=self.dt, windup=self.windup, trace=self.trace
        )
        # Reset exposes a settled t=0 boundary, including immediate spawns.
        sim.run_until(0)
        return EnvState(
            sim.state,
            max_decisions=self.max_decisions,
            horizon=self.horizon,
            trace=tuple(sim.trace.events),
            trace_enabled=self.trace,
        )

    def legal_actions(self, state):
        return legal_actions(state)

    def clone_state(self, state):
        return state.clone()

    def state_key(self, state):
        payload = (
            state.game.stable_hash(),
            state.decision_count,
            state.max_decisions,
            state.horizon,
        )
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode()
        ).hexdigest()

    def is_terminal(self, state):
        return state.terminal

    def reward_potential(self, state):
        from .rewards import state_progress
        return state_progress(state)

    def transition_reward(self, before, after):
        return self.reward_potential(after) - self.reward_potential(before)

    def terminal_reward(self, state):
        from .rewards import terminal_bonus
        return terminal_bonus(self.result(state).to_dict())

    def result(self, state):
        g = state.game
        total = g.stage.hostile_count
        reason = (
            "battle_end"
            if g.done
            else "decision_limit"
            if state.decision_count >= state.max_decisions
            else "time_limit"
            if g.current_time >= state.horizon - 1e-9
            else None
        )
        return EpisodeResult(
            bool(g.done and g.result == "WIN" and g.escaped == 0 and g.killed == total and g.escorts_dead == 0),
            g.killed,
            g.escaped,
            g.life,
            total,
            state.terminal,
            reason,
            g.result,
            g.stage.life,
            g.stage.escort_count, g.escorts_saved,
            sum(e.alive and e.data.faction == 'ESCORT' for e in g.enemies.values()),
            g.escorts_dead, g.escorts_dead == 0,
        )

    def _simulator(self, state):
        return Simulator.from_state(state.game, trace=state.trace_enabled)

    def _record(self, state, sim, reasons):
        state.last_decision = DecisionEvent(state.game.current_time, tuple(reasons))
        if state.trace_enabled:
            state.trace += tuple(sim.trace.events)
        return state

    def step(self, state, action):
        if not isinstance(action, Action) or action not in self.legal_actions(state):
            raise ValueError(f"Illegal action: {action}")
        child = self.clone_state(state)
        sim = self._simulator(child)
        if action.type == ActionType.WAIT:
            if action.wait_seconds is None:
                self._wait(child, sim)
            else:
                self._hold(child, sim, action.wait_seconds)
        else:
            if action.type == ActionType.DEPLOY:
                sim.deploy(action.operator_id, action.tile, action.direction.index)
            elif action.type == ActionType.DEPLOY_SUMMON:
                sim.deploy_summon(action.operator_id, action.tile, action.direction.index)
            elif action.type == ActionType.RETREAT_SUMMON:
                sim.retreat_summon(action.operator_id)
            elif action.type == ActionType.RETREAT:
                sim.retreat(action.operator_id)
            elif action.type == ActionType.ACTIVATE_SKILL:
                sim.activate_skill(action.operator_id)
            elif action.type == ActionType.PLACE_DEVICE:
                from arknights_sim.mechanics.runtime import place
                place(sim,action.operator_id,action.tile)
            elif action.type == ActionType.ACTIVATE_DEVICE:
                from arknights_sim.mechanics.runtime import activate
                activate(sim,action.operator_id)
            elif action.type == ActionType.REMOVE_DEVICE:
                from arknights_sim.mechanics.runtime import remove
                remove(sim,action.operator_id)
            self._record(child, sim, (action.type.value,))
        child.decision_count += 1
        return child

    def _wait(self, state, sim):
        before = signature(state.game)
        reasons = ()

        def meaningful(game):
            nonlocal reasons
            reasons = changed(before, signature(game))
            return bool(reasons)

        sim.run_until(state.horizon, stop_when=meaningful)
        if not reasons:
            reasons = ("TIME_LIMIT",) if not state.game.done else ("BATTLE_END",)
        self._record(state, sim, reasons)

    def _hold(self, state, sim, seconds):
        # Public boundaries stay on the physics grid; never round a requested
        # hold down to zero or bypass normal combat/skill/event execution.
        ticks = max(1, math.ceil(seconds / state.game.clock.dt - 1e-9))
        target = min(state.horizon, state.game.current_time + ticks * state.game.clock.dt)
        sim.run_until(target)
        reason = "BATTLE_END" if state.game.done else "TIME_LIMIT" if state.game.current_time >= state.horizon - 1e-9 else "PLANNED_WAIT_END"
        self._record(state, sim, (reason,))

    def future_event_state(self, state):
        from arknights_sim.core.stage_events import future_event_state
        return future_event_state(state.game)

    def get_result(self, state):
        return self.result(state)

    def get_next_decision_event(self, state):
        """Predict the next boundary on a clone without consuming the caller's RNG."""
        if state.terminal:
            reason = "BATTLE_END" if state.game.done else "DECISION_LIMIT" if state.decision_count >= state.max_decisions else "TIME_LIMIT"
            return DecisionEvent(state.game.current_time, (reason,))
        return self.advance_to_next_decision_event(state).last_decision

    def fast_forward(self, state, seconds=None):
        """Pure planner transition: next event, or a bounded strategic hold."""
        if seconds is None:
            return self.advance_to_next_decision_event(state)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Fast-forward duration must be finite and positive")
        if state.terminal:
            raise ValueError("Episode ended")
        child = self.clone_state(state)
        self._hold(child, self._simulator(child), seconds)
        child.decision_count += 1
        return child

    def advance_to_next_decision_event(self, state):
        return self.step(state, WAIT)

    def advance_to_time(self, state, time):
        """External scheduled input (regression/replay only); never a search action."""
        if state.terminal:
            raise ValueError("Episode ended")
        child = self.clone_state(state)
        sim = self._simulator(child)
        sim.run_until(min(time, state.horizon))
        return self._record(child, sim, ("EXTERNAL_SCHEDULE",))
