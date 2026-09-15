"""Cooperative simulator ownership in a QThread; signals carry frozen DTOs."""

import json
import logging
import math
import traceback
from collections import deque
from pathlib import Path
from threading import Condition
from time import perf_counter

from PySide6.QtCore import QThread, Signal

from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.progression import Progression, load_progressed
from arknights_sim.data.skill_loader import SkillLoader
from arknights_sim.environment import Action, ArknightsEnv
from desktop.models.simulation_state import (
    event_from_record,
    snapshot_from_stage,
    snapshot_from_state,
)

logger = logging.getLogger("arknights.desktop")
MODES = ("Manual control", "Automatic control", "Hybrid control")
AUTO_MODES = ("Automatic control", "Hybrid control", "Random agent")


class SimulationWorker(QThread):
    state_changed = Signal(str)
    snapshot = Signal(object)
    events = Signal(object)
    error = Signal(str)
    log = Signal(str)

    def __init__(
        self,
        *,
        stage_id,
        squad,
        seed,
        speed=1.0,
        fps=20,
        mode="Scripted baseline",
        checkpoint=None,
        auto_squad=None,
        progression=None,
        skill_overrides=None,
        data_dir=None,
        parent=None,
    ):
        super().__init__(parent)
        if mode not in MODES + ("Scripted baseline", "Random agent", "No deployments"):
            raise ValueError("Unknown simulator mode")
        self.stage_id = stage_id
        self.squad = tuple(squad)
        self.seed = seed
        self.mode = mode
        self.checkpoint = checkpoint
        self.auto_squad = mode == "Automatic control" if auto_squad is None else auto_squad
        self.progression = Progression.from_dict(progression)
        self.skill_overrides = dict(skill_overrides or {})
        for index in self.skill_overrides.values():
            Progression(skill_index=index)
        self.data_dir = data_dir
        self._condition = Condition()
        self._paused = mode in ("Manual control", "Hybrid control")
        self._stopping = False
        self._steps = 0
        self._commands = deque()
        self._speed = 1.0
        self._fps = 20
        self.set_speed(speed)
        self.set_fps(fps)
        self._status = None
        self._last_emit = 0.0
        self._trace_cursor = 0
        self._env = self._state = self._stage_snapshot = None
        self._agent = None
        self._pending_wait = None
        self._display_target = 0.0

    def pause(self):
        with self._condition:
            self._paused = True
            self._condition.notify_all()

    def resume(self):
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def step_once(self):
        with self._condition:
            self._paused = True
            self._steps += 1
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()

    def submit_action(self, payload_json):
        with self._condition:
            if self.mode == "Hybrid control":
                self._paused = True
            self._commands.append(str(payload_json))
            self._condition.notify_all()

    def set_speed(self, speed):
        speed = float(speed)
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Playback speed must be finite and nonnegative (0 = MAX)")
        with self._condition:
            self._speed = speed
            self._condition.notify_all()

    def set_fps(self, fps):
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("Visualization FPS must be in [1, 60]")
        with self._condition:
            self._fps = fps
            self._condition.notify_all()

    def _set_status(self, status):
        if self._status != status:
            self._status = status
            self.state_changed.emit(status)

    def _message(self, message):
        logger.info(message, extra={"category": "SIM"})
        self.log.emit(message)

    def _load(self):
        self._env = ArknightsEnv(data_dir=self.data_dir, trace=True)
        data = Path(self._env.data_dir)
        loader = OperatorLoader(
            data / "character_table.json",
            data / "range_table.json",
            SkillLoader(data / "skill_table.json"),
        )
        from dataclasses import replace
        def build(key):
            request = replace(self.progression, skill_index=self.skill_overrides.get(key, self.progression.skill_index))
            return load_progressed(loader, key, request)
        self._env.squad = tuple(build(key) for key in self.squad)
        if self.mode in ("Automatic control", "Hybrid control"):
            from agents.neural_battle_agent import NeuralBattleAgent
            self._agent = NeuralBattleAgent(self.checkpoint or "", data, self.seed)
            if self.auto_squad and self._agent.coordinator is not None:
                self._agent.coordinator.operators = tuple(build(key) for key in self._agent.coordinator.keys)
                self._agent.coordinator._cache.clear()
            if self.auto_squad:
                self._env.squad = self._agent.choose_squad(self._env, self.stage_id, self.seed)
                self._message("模型自主编队：" + "、".join(op.name for op in self._env.squad))
        self._message("干员养成设置：" + json.dumps(self.progression.to_dict(), ensure_ascii=False))
        for op in self._env.squad:
            self._message(f"{op.name}：精英{op.elite} Lv{op.level} · 潜能提升{op.potential}次 · S{op.skill_index + 1} Lv{op.skill_level} · HP {op.hp:g} ATK {op.atk:g} DEF {op.defense:g} DP {op.cost:g}")
            for note in op.simulation_notes:
                self._message(f"{op.name}：{note}")
        self._state = self._env.reset(stage_id=self.stage_id, seed=self.seed)
        self._stage_snapshot = snapshot_from_stage(self._state.game.stage)
        if self.mode == "Scripted baseline":
            from agents.scripted_agent import ScriptedAgent

            if set(self.squad) != {"char_500_noirc", "char_208_melan"}:
                raise ValueError(
                    "The historical baseline requires both Noir Corne and Melantha"
                )
            self._agent = ScriptedAgent()
        elif self.mode == "Random agent":
            from agents.random_agent import RandomAgent

            self._agent = RandomAgent(self.seed)
        self._message(f"Stage {self.stage_id} loaded · {self.mode} · seed {self.seed}")

    def _publish(self, force=False):
        now = perf_counter()
        if not force and now - self._last_emit < 1 / self._fps:
            return
        self._last_emit = now
        fresh = tuple(
            event_from_record(record)
            for record in self._state.trace[self._trace_cursor :]
        )
        self._trace_cursor = len(self._state.trace)
        if fresh:
            self.events.emit(fresh)
            for event in fresh:
                logger.info(event.text, extra={"category": "SIM"})
        self.snapshot.emit(
            snapshot_from_state(self._env, self._state, self._stage_snapshot)
        )

    def _scheduled_time(self):
        if self.mode == "Scripted baseline" and self._agent.index < len(
            self._agent.schedule
        ):
            return self._agent.schedule[self._agent.index].time
        return math.inf

    def _scheduled_action_if_due(self):
        if (
            not self._env.is_terminal(self._state)
            and abs(self._state.game.current_time - self._scheduled_time()) < 1e-8
        ):
            # Existing baseline owns the schedule and action legality. Playback
            # merely reaches its exact external action boundary before calling it.
            self._state = self._agent.prepare_state(self._env, self._state)
            self._state = self._env.step(
                self._state, self._agent.select_action(self._env, self._state)
            )

    def _choose_action(self):
        action = self._agent.select_action(self._env, self._state)
        explanation = getattr(self._agent, 'last_decision_explanation', None)
        if explanation:
            self._message('AI 决策：' + json.dumps(explanation, ensure_ascii=False, separators=(',', ':')))
        return action

    def _advance_decision(self):
        if self.mode in AUTO_MODES:
            if self._pending_wait is not None:
                self._state, self._pending_wait = self._pending_wait, None
            else:
                self._state = self._env.step(
                    self._state, self._choose_action()
                )
        else:
            candidate = self._env.advance_to_next_decision_event(self._state)
            scheduled = self._scheduled_time()
            if scheduled <= candidate.game.current_time + 1e-9:
                self._state = self._env.advance_to_time(self._state, scheduled)
                self._scheduled_action_if_due()
            else:
                self._state = candidate
        self._display_target = self._state.game.current_time

    def _advance_visual(self, speed, fps):
        if self.mode in AUTO_MODES and self._pending_wait is None:
            action = self._choose_action()
            candidate = self._env.step(self._state, action)
            if candidate.game.current_time <= self._state.game.current_time + 1e-9:
                self._state = candidate
                return
            self._pending_wait = candidate
        now = self._state.game.current_time
        self._display_target = max(now, self._display_target) + (
            0.5 if speed == 0 else speed / fps
        )
        dt = self._state.game.clock.dt
        target = round(math.floor((self._display_target + 1e-9) / dt) * dt, 9)
        target = min(target, self._state.horizon, self._scheduled_time())
        if self._pending_wait is not None:
            target = min(target, self._pending_wait.game.current_time)
        if target > now + 1e-9:
            self._state = self._env.advance_to_time(self._state, target)
        if (
            self._pending_wait is not None
            and target >= self._pending_wait.game.current_time - 1e-9
        ):
            self._state, self._pending_wait = self._pending_wait, None
            self._display_target = self._state.game.current_time
        self._scheduled_action_if_due()

    def _apply_command(self, payload):
        if self.mode not in ("Manual control", "Hybrid control"):
            raise ValueError("Manual actions are available in Manual control mode")
        action = Action.from_dict(json.loads(payload))
        # Recheck against the current worker-owned state, not an old UI snapshot.
        self._state = self._env.step(self._state, action)
        self._pending_wait = None
        self._display_target = self._state.game.current_time

    def run(self):
        try:
            self._set_status("LOADING")
            self._load()
            self._publish(force=True)
            while not self._env.is_terminal(self._state):
                cycle_started = perf_counter()
                with self._condition:
                    while (
                        self._paused
                        and not self._stopping
                        and not self._steps
                        and not self._commands
                    ):
                        if self._status != "PAUSED":
                            self._set_status("PAUSED")
                            self._publish(force=True)
                        self._condition.wait()
                    if self._stopping:
                        break
                    payload = self._commands.popleft() if self._commands else None
                    step = self._steps > 0 and payload is None
                    if step:
                        self._steps -= 1
                    paused, speed, fps = self._paused, self._speed, self._fps
                self._set_status("PAUSED" if paused else "RUNNING")
                if payload is not None:
                    try:
                        self._apply_command(payload)
                    except (ValueError, TypeError, KeyError) as exc:
                        self._message(f"Action rejected: {exc}")
                        self.error.emit(str(exc))
                elif step:
                    self._advance_decision()
                elif not paused:
                    self._advance_visual(speed, fps)
                self._publish(force=step or payload is not None)
                if not paused and speed != 0:
                    remaining = max(0, 1 / fps - (perf_counter() - cycle_started))
                    with self._condition:
                        if not self._stopping and not self._paused:
                            self._condition.wait(timeout=remaining)
            self._publish(force=True)
            if self._stopping:
                self._set_status("STOPPED")
                self._message("Simulation stopped safely")
            else:
                self._set_status("FINISHED")
                result = self._env.result(self._state)
                self._message(
                    f"Battle finished: {'SUCCESS' if result.success else 'FAILED'} · kills {result.kills}/{result.total_enemies} · leaks {result.leaks}"
                )
        except Exception:  # noqa: BLE001 -- transport errors instead of crashing Qt
            details = traceback.format_exc()
            logger.error(details, extra={"category": "ERROR"})
            self._set_status("ERROR")
            self.error.emit(details)
            self.log.emit(details)
