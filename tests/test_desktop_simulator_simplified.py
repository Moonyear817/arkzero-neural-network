"""Simplified presentation preserves controls, model identity and device actions."""

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Qt, Signal

from arknights_sim.data.stage_loader import StageLoader
from arknights_sim.environment import Action, ArknightsEnv
from desktop.i18n import get_language, set_language, tr
from desktop.models.simulation_state import snapshot_from_state
from desktop.views.simulator_view import SimulatorView


class Controller(QObject):
    snapshot = Signal(object)
    events = Signal(object)
    state_changed = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.status = "IDLE"
        self.last_snapshot = None
        self.data_dir = None
        self.calls = []

    def is_busy(self):
        return self.status in ("LOADING", "RUNNING", "PAUSED", "STOPPING")

    def update(self, status):
        self.status = status
        self.state_changed.emit(status)

    def start(self, *args, **kwargs):
        self.calls.append(("start", args, kwargs))
        self.update("LOADING")
        return True

    def pause(self):
        self.calls.append(("pause",))
        self.update("PAUSED")

    def resume(self):
        self.calls.append(("resume",))
        self.update("RUNNING")

    def stop(self):
        self.calls.append(("stop",))
        self.update("STOPPED")

    def restart(self):
        self.calls.append(("restart",))

    def step_once(self):
        self.calls.append(("step",))

    def submit_action(self, payload):
        self.calls.append(("action", payload))

    def set_speed(self, speed):
        self.calls.append(("speed", speed))


@pytest.fixture
def view(qtbot):
    previous = get_language()
    set_language("en")
    controller = Controller()
    widget = SimulatorView(controller)
    qtbot.addWidget(widget)
    widget.resize(1100, 800)
    widget.show()
    yield widget
    set_language(previous)


def test_map_dominates_initial_view_with_secondary_controls_folded(view):
    assert view.map.isVisible()
    assert not view.timeline_section.toggle.isChecked()
    assert not view.timeline.isVisible()
    assert not view.more_settings.toggle.isChecked()
    assert not view.seed.isVisible()
    assert not view.restart_button.isVisible()
    assert not view.manual_actions.isVisible()
    assert not view.run_button.isVisible()
    assert not view.play_button.isVisible()
    assert not view.pause_button.isVisible()
    assert view.battle_button.text() == "Start battle"
    assert view.mode.count() == 3
    assert view.map.height() > view.height() / 2


def test_single_primary_button_dispatches_start_pause_resume_and_end(view, qtbot):
    qtbot.mouseClick(view.battle_button, Qt.MouseButton.LeftButton)
    assert view.controller.calls[-1][0] == "start"
    assert not view.battle_button.isEnabled()
    view.controller.update("RUNNING")
    assert view.battle_button.text() == "Pause"
    assert view.stop_button.isVisible()
    assert not view.step_button.isVisible()
    qtbot.mouseClick(view.battle_button, Qt.MouseButton.LeftButton)
    assert view.controller.calls[-1] == ("pause",)
    assert view.battle_button.text() == "Resume battle"
    assert view.step_button.isVisible()
    qtbot.mouseClick(view.step_button, Qt.MouseButton.LeftButton)
    assert view.controller.calls[-1] == ("step",)
    qtbot.mouseClick(view.battle_button, Qt.MouseButton.LeftButton)
    assert view.controller.calls[-1] == ("resume",)
    qtbot.mouseClick(view.stop_button, Qt.MouseButton.LeftButton)
    assert view.controller.calls[-1] == ("stop",)
    assert view.battle_button.text() == "Start battle"
    assert not view.stop_button.isVisible()


def test_model_handoff_preserves_mode_squad_and_rejects_during_battle(view, tmp_path):
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    first.touch()
    second.touch()
    original_squad = view.selected_squad
    assert view.set_model(str(first))
    assert view.mode.currentData() == "Manual control"
    assert view.selected_squad == original_squad
    assert view.model_summary.toolTip() == str(first.resolve())
    assert "not used" in view.model_summary.text()
    assert view.model_row.isVisible()
    view.mode.setCurrentIndex(view.mode.findData("Hybrid control"))
    assert not view.auto_squad.isChecked()
    assert view._run()
    args = view.controller.calls[-1][2]
    assert args["checkpoint"] == str(first.resolve())
    assert args["mode"] == "Hybrid control" and args["auto_squad"] is False
    errors = []
    view.controller.error.connect(errors.append)
    assert not view.set_model(str(second))
    assert errors
    assert view.model_path.text() == str(first.resolve())
    assert view.model_summary.toolTip() == str(first.resolve())
    assert not view.set_squad(("char_208_melan",))
    assert view.selected_squad == original_squad


def test_automatic_squad_source_is_visible_and_never_silently_switches_mode(
    view, tmp_path
):
    model = tmp_path / "joint.pt"
    model.touch()
    view.set_model(model)
    view.mode.setCurrentIndex(view.mode.findData("Automatic control"))
    assert "model" in view.squad_summary.text()
    assert not view.pick_squad.isVisible()
    assert view.set_squad(("char_208_melan",))
    assert view.mode.currentData() == "Automatic control"
    assert view.auto_squad.isChecked()
    view._run()
    assert view.controller.calls[-1][2]["auto_squad"] is True


def test_diagnostic_modes_are_in_more_settings_and_incompatible_baseline_rejected(view):
    assert view.experiment_mode.findData("Random agent") >= 0
    assert view.mode.findData("Random agent") == -1
    view.experiment_mode.setCurrentIndex(
        view.experiment_mode.findData("Scripted baseline")
    )
    view.set_squad(("char_208_melan",))
    errors = []
    view.controller.error.connect(errors.append)
    assert view._run() is False
    assert errors and not view.controller.calls
    assert view.experiment_mode.currentData() == "Scripted baseline"
    assert view.mode.currentData() == "Manual control"


def test_real_3_7_obstacle_actions_survive_collapsed_timeline(view):
    root = Path(__file__).resolve().parents[1]
    stage = StageLoader(root / "data/maps/levels/enemydata/enemy_database.json").load(
        root / "data/maps/levels/obt/main/level_main_03-07.json"
    )
    env = ArknightsEnv(stage=stage, squad=())
    state = env.reset()
    view._active_mode = "Manual control"
    view.controller.update("PAUSED")
    snapshot = snapshot_from_state(env, state)
    view.controller.last_snapshot = snapshot
    view.controller.snapshot.emit(snapshot)
    assert view.manual_actions.isVisible()
    assert view.action_combo.count() == len(env.legal_actions(state))
    device = next(
        index
        for index in range(view.action_combo.count())
        if json.loads(view.action_combo.itemData(index))["type"] == "PLACE_DEVICE"
    )
    payload = view.action_combo.itemData(device)
    view.action_combo.setCurrentIndex(device)
    view._apply_action()
    assert view.controller.calls[-1] == ("action", payload)
    assert Action.from_dict(json.loads(payload)) in env.legal_actions(state)
    assert "Obstacles remaining" in view.metrics.text()
    assert not view.timeline.isVisible()


def test_simplified_controls_translate_live_without_changing_business_values(view):
    view.mode.setCurrentIndex(view.mode.findData("Hybrid control"))
    set_language("zh_CN")
    assert view.battle_button.text() == "开始对战"
    assert view.more_settings.toggle.text() == "更多设置"
    assert view.timeline_section.toggle.text() == "战斗时间轴"
    assert view.mode.currentData() == "Hybrid control"
    view.controller.update("PAUSED")
    assert view.battle_button.text() == "继续"
    set_language("en")
    assert view.battle_button.text() == "Resume battle"
    assert view.timeline_section.toggle.text() == tr("Battle timeline")
