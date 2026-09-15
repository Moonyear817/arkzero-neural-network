"""Real simulator integration, immutable UI records and cooperative controls."""

import json
from dataclasses import FrozenInstanceError

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from agents.episode import run_episode
from agents.scripted_agent import ScriptedAgent
from arknights_sim.environment import ArknightsEnv
from desktop.controllers.simulator_controller import SimulatorController
from desktop.i18n import get_language, set_language, tr
from desktop.models.simulation_state import (
    BattleEvent,
    event_from_record,
    snapshot_from_stage,
    snapshot_from_state,
)
from desktop.services.logging_service import LoggingService
from desktop.views.simulator_view import SimulatorView
from desktop.widgets.battle_timeline import BattleTimeline


@pytest.fixture
def controller(qtbot):
    control = SimulatorController()
    yield control
    assert control.shutdown(10000)
    qtbot.waitUntil(lambda: not control.is_busy(), timeout=10000)


def test_stage_and_battle_snapshots_are_frozen_detached(real_stage, squad):
    env = ArknightsEnv(real_stage, squad)
    state = env.reset()
    stage = snapshot_from_stage(real_stage)
    snapshot = snapshot_from_state(env, state, stage)
    assert snapshot.width == real_stage.map.width
    assert snapshot.height == real_stage.map.height
    assert snapshot.routes and snapshot.tiles
    assert snapshot.total_enemies == 11
    with pytest.raises(FrozenInstanceError):
        snapshot.dp = -1
    with pytest.raises(FrozenInstanceError):
        snapshot.tiles[0][0].key = "changed"
    old_dp = snapshot.dp
    state.game.dp = 999
    assert snapshot.dp == old_dp
    assert isinstance(snapshot.legal_actions, tuple)
    assert json.loads(snapshot.legal_actions[0].payload_json)["type"] == "WAIT"


def test_scripted_visual_worker_preserves_regression_result_and_trace(
    controller, qtbot
):
    events, snapshots, errors = [], [], []
    controller.events.connect(lambda batch: events.extend(batch))
    controller.snapshot.connect(snapshots.append)
    controller.error.connect(errors.append)
    assert controller.start(speed=0, mode="Scripted baseline")
    qtbot.waitUntil(lambda: not controller.is_busy(), timeout=20000)
    assert not errors
    final = snapshots[-1]
    assert (final.result, final.kills, final.leaks, final.life) == (
        "SUCCESS",
        11,
        0,
        20,
    )
    assert final.terminal
    # Exact trace comparison against the original baseline within the current
    # simulator, not a claim that assumed mechanics match the real client.
    env = ArknightsEnv(trace=True)
    original = run_episode(env, ScriptedAgent())
    assert tuple(events) == tuple(
        event_from_record(record) for record in original.final_state.trace
    )
    assert len(snapshots) < 100


def test_pause_step_resume_and_stop_are_cooperative(controller, qtbot):
    controller.start(mode="No deployments", speed=1)
    qtbot.waitUntil(lambda: controller.status == "RUNNING", timeout=10000)
    qtbot.waitUntil(lambda: controller.last_snapshot.time > 0, timeout=2000)
    controller.pause()
    qtbot.waitUntil(lambda: controller.status == "PAUSED")
    qtbot.wait(40)
    paused = controller.last_snapshot.time
    qtbot.wait(80)
    assert controller.last_snapshot.time == paused
    controller.step_once()
    qtbot.waitUntil(lambda: controller.last_snapshot.time > paused)
    assert controller.last_snapshot.time - paused > 0.5
    assert controller.status == "PAUSED"
    controller.resume()
    qtbot.waitUntil(lambda: controller.status == "RUNNING")
    controller.stop()
    qtbot.waitUntil(lambda: not controller.is_busy())
    assert controller.status == "STOPPED"


def test_manual_mode_uses_legal_action_options(controller, qtbot):
    controller.start(mode="Manual control", speed=1)
    qtbot.waitUntil(lambda: controller.status == "PAUSED", timeout=10000)
    assert controller.last_snapshot.time == 0
    controller.step_once()
    # Fragment activation is now a decision before the first affordable unit.
    qtbot.waitUntil(lambda: controller.last_snapshot.time > 0)
    controller.step_once()
    qtbot.waitUntil(
        lambda: any(
            "DEPLOY" in option.label
            for option in controller.last_snapshot.legal_actions
        )
    )
    option = next(
        option
        for option in controller.last_snapshot.legal_actions
        if "DEPLOY" in option.label
    )
    operator_id = json.loads(option.payload_json)["operator_id"]
    controller.submit_action(option.payload_json)
    qtbot.waitUntil(
        lambda: any(unit.id == operator_id for unit in controller.last_snapshot.units)
    )
    assert controller.status == "PAUSED"
    assert any(
        "RETREAT" in option.label for option in controller.last_snapshot.legal_actions
    )


def test_stale_illegal_action_is_rejected_without_crashing_worker(controller, qtbot):
    errors = []
    controller.error.connect(errors.append)
    controller.start(mode="Manual control")
    qtbot.waitUntil(lambda: controller.status == "PAUSED", timeout=10000)
    before = controller.last_snapshot
    controller.submit_action('{"type":"RETREAT","operator_id":"char_208_melan"}')
    qtbot.waitUntil(lambda: bool(errors))
    assert "Illegal action" in errors[-1]
    assert controller.is_busy() and controller.status == "PAUSED"
    assert controller.last_snapshot == before


def test_background_error_is_signaled_and_traceback_saved(
    controller, qtbot, tmp_path, monkeypatch
):
    def broken_reset(*args, **kwargs):
        raise RuntimeError("deliberate worker failure")

    monkeypatch.setattr(ArknightsEnv, "reset", broken_reset)
    logs = LoggingService(tmp_path)
    errors = []
    controller.error.connect(errors.append)
    try:
        controller.start(speed=0)
        qtbot.waitUntil(lambda: not controller.is_busy(), timeout=10000)
        assert controller.status == "ERROR"
        assert errors and "deliberate worker failure" in errors[-1]
        assert "Traceback" in (tmp_path / "errors.log").read_text()
    finally:
        logs.close()


def test_restart_replaces_worker_safely(controller, qtbot):
    snapshots = []
    controller.snapshot.connect(snapshots.append)
    controller.start(mode="Manual control")
    qtbot.waitUntil(lambda: controller.status == "PAUSED", timeout=10000)
    controller.step_once()
    qtbot.waitUntil(lambda: controller.last_snapshot.time > 0)
    count = len(snapshots)
    controller.restart()
    qtbot.waitUntil(
        lambda: (
            len(snapshots) > count
            and snapshots[-1].time == 0
            and controller.status == "PAUSED"
        ),
        timeout=10000,
    )
    assert controller.is_busy()


def test_random_visual_episode_ends_normally(controller, qtbot):
    errors = []
    controller.error.connect(errors.append)
    controller.start(mode="Random agent", speed=0)
    qtbot.waitUntil(lambda: not controller.is_busy(), timeout=20000)
    assert not errors
    assert controller.last_snapshot.terminal
    assert controller.last_snapshot.kills + controller.last_snapshot.leaks == 11


def test_view_controls_map_and_timeline_smoke(controller, qtbot):
    view = SimulatorView(controller)
    qtbot.addWidget(view)
    view.resize(1000, 780)
    view.show()
    view.mode.setCurrentIndex(view.mode.findData("Manual control"))
    qtbot.mouseClick(view.run_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: controller.status == "PAUSED", timeout=10000)
    assert view.play_button.isEnabled()
    assert not view.pause_button.isEnabled()
    assert view.map._snapshot is not None
    qtbot.mouseClick(view.step_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: controller.last_snapshot.time > 0)
    assert dict(controller.last_snapshot.event_progression)['current_fragment'] == 1
    first_time = controller.last_snapshot.time
    qtbot.mouseClick(view.step_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: controller.last_snapshot.time > first_time)
    assert view.action_combo.count() > 1
    view.action_filter.setText("DEPLOY")
    assert view.action_combo.count() > 0
    assert tr("BATTLE_START") in view.timeline.console.toPlainText()


def test_timeline_filters_search_and_keeps_bounded_records(qtbot):
    timeline = BattleTimeline(max_lines=3)
    qtbot.addWidget(timeline)
    timeline.append_events(
        tuple(
            BattleEvent(
                i,
                "SPAWN" if i % 2 == 0 else "DAMAGE",
                "Spawn" if i % 2 == 0 else "Damage",
                f"enemy={i}",
            )
            for i in range(6)
        )
    )
    assert len(timeline._events) == 3
    timeline.category.setCurrentIndex(timeline.category.findData("Damage"))
    assert tr("SPAWN") not in timeline.console.toPlainText()
    timeline.search.setText("enemy=5")
    assert timeline.console.toPlainText().count("enemy=") == 1
    timeline.auto_scroll.setChecked(False)
    timeline.clear()
    assert not timeline.console.toPlainText()


def test_language_switch_updates_simulator_without_changing_control_ids(
    controller, qtbot
):
    previous_language = get_language()
    view = SimulatorView(controller)
    qtbot.addWidget(view)
    try:
        set_language("en")
        view.mode.setCurrentIndex(view.mode.findData("Manual control"))
        view._run()
        qtbot.waitUntil(lambda: controller.status == "PAUSED", timeout=10000)
        view.timeline.category.setCurrentIndex(view.timeline.category.findData("Other"))
        set_language("zh_CN")
        assert view.run_button.text() == "开始模拟"
        assert view.mode.currentData() == "Manual control"
        assert view.mode.currentText() == "手动控制"
        assert view.timeline.category.currentData() == "Other"
        assert "战斗开始" in view.timeline.console.toPlainText()
        assert "时间" in view.metrics.text()
        assert view.status_label.text() == "已暂停"
        assert view.action_combo.currentText() == "等待"
        assert controller.status == "PAUSED" and controller.last_snapshot.time == 0
        set_language("en")
        assert view.run_button.text() == "RUN"
        assert view.mode.currentText() == "Manual control"
        assert "BATTLE_START" in view.timeline.console.toPlainText()
        assert view.action_combo.currentText() == "WAIT"
    finally:
        set_language(previous_language)
