"""Simplified training remains an honest adapter for foreground/background work."""

from copy import deepcopy

import pytest
import yaml
from PySide6.QtCore import QObject, Signal

from desktop.i18n import get_language, set_language, tr
from desktop.services.config_service import ConfigService
from desktop.views.dashboard_view import DashboardView
from desktop.views.training_view import TrainingView


class Controller(QObject):
    state_changed = Signal(str)
    metrics_updated = Signal(object)
    event_received = Signal(object)
    log = Signal(str)
    error = Signal(str)
    availability = True

    def __init__(self, service):
        super().__init__()
        self.config_service = service
        self.state = "IDLE"
        self.is_busy = False
        self.calls = []

    def set_state(self, state):
        self.state = state
        self.is_busy = state not in ("IDLE", "ERROR")
        self.state_changed.emit(state)

    def start(self, config):
        self.calls.append(("start", deepcopy(config)))
        self.set_state("RUNNING")
        return True

    def pause(self):
        self.calls.append(("pause",))
        self.set_state("PAUSED")

    def resume(self):
        self.calls.append(("resume",))
        self.set_state("RUNNING")

    def stop(self):
        self.calls.append(("stop",))
        self.set_state("STOPPING")

    def save_checkpoint(self):
        self.calls.append(("save",))

    def evaluate(self, config=None):
        self.calls.append(("evaluate", config))


@pytest.fixture
def setup_view(qtbot, tmp_path, monkeypatch):
    language = get_language()
    set_language("en")
    background = {}
    monkeypatch.setattr(
        "desktop.services.background_training.read_background_training",
        lambda: deepcopy(background),
    )
    service = ConfigService(tmp_path)
    service.config_dir.mkdir(parents=True)
    service.default_path.write_text(yaml.safe_dump(service.DEFAULTS))
    controller = Controller(service)
    view = TrainingView(controller)
    qtbot.addWidget(view)
    yield view, controller, service, background
    view.background_timer.stop()
    set_language(language)


def test_four_choices_and_optional_sections_start_collapsed(setup_view):
    view, _, _, _ = setup_view
    assert view.fields["stage"].currentData() == "0-1"
    assert view.squad_mode.currentData() == "fixed"
    assert view.resume_selector.currentData() == "fresh"
    assert view.size_selector.currentData() == "quick"
    assert view.advanced.body.isHidden()
    assert view.more_actions.body.isHidden()
    assert view.more_metrics.body.isHidden()
    assert view.pause_button.isHidden() and view.resume_button.isHidden()
    assert view.stop_button.isHidden()
    assert view.start_button.isEnabled()


def test_single_visible_primary_button_tracks_real_foreground_state(setup_view):
    view, controller, _, _ = setup_view
    view.start_button.click()
    assert controller.calls[-1][0] == "start"
    assert view.start_button.text() == "PAUSE"
    assert not view.stop_button.isHidden()
    assert view.config_group.isHidden() and view.advanced.isHidden()
    assert not view.task_summary.isHidden()
    assert "0-1" in view.task_summary.text()
    assert "Fixed squad" in view.task_summary.text()
    assert "Start from scratch" in view.task_summary.text()
    view.start_button.click()
    assert controller.calls[-1][0] == "pause"
    assert view.start_button.text() == "RESUME"
    view.start_button.click()
    assert controller.calls[-1][0] == "resume"
    view.stop_button.click()
    assert controller.calls[-1][0] == "stop"
    assert not view.start_button.isEnabled()
    assert not view.stop_button.isEnabled()
    assert view.pause_button.isHidden() and view.resume_button.isHidden()
    assert not view.task_notice.isHidden()
    assert "Stopping safely" in view.task_notice.text()
    controller.set_state("IDLE")
    assert not view.config_group.isHidden() and not view.advanced.isHidden()
    assert view.task_summary.isHidden() and view.task_notice.isHidden()


def test_custom_yaml_is_not_replaced_by_presets_on_load_or_start(setup_view):
    view, controller, service, _ = setup_view
    config = service.load()
    config.update(
        iterations=7,
        episodes_per_iteration=9,
        mcts_simulations=27,
        training_steps_per_iteration=11,
        evaluation_episodes=4,
        learning_rate=0.0023,
        stage="local-stage",
        stage_pool=["0-1", "local-stage"],
        squad=["char_002_amiya", "char_103_angel"],
        auto_squad=True,
    )
    path = service.save_session(config, service.config_dir / "custom.yaml")
    view.load_config(path)
    assert view.size_selector.currentData() == "custom"
    assert view.collect_config() == {**config, "device": "auto"}
    view.start_button.click()
    actual = controller.calls[-1][1]
    for key in config:
        assert actual[key] == config[key]
    assert view.auto_squad.isChecked()
    assert view.selected_squad == tuple(config["squad"])


def test_preset_only_changes_budget_after_explicit_selection(setup_view):
    view, _, _, _ = setup_view
    view.fields["learning_rate"].setValue(0.0023)
    squad = tuple(view.selected_squad)
    view.size_selector.setCurrentIndex(view.size_selector.findData("small"))
    config = view.collect_config()
    for key, value in view.PRESETS["small"].items():
        assert config[key] == value
    assert config["learning_rate"] == 0.0023
    assert tuple(config["squad"]) == squad
    view.fields["mcts_simulations"].setValue(41)
    assert view.size_selector.currentData() == "custom"


def test_resume_is_explicit_and_core_still_validates_checkpoint(setup_view, tmp_path):
    view, controller, _, _ = setup_view
    view.resume_selector.setCurrentIndex(view.resume_selector.findData("resume"))
    with pytest.raises(ValueError, match="Choose"):
        view.collect_config()
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"invalid-format is intentionally left for Core validation")
    view.resume_path.setText(str(checkpoint))
    assert view.collect_config()["resume_from"] == str(checkpoint)
    assert "Core validates" in view.resume_note.text()
    assert controller.calls == []  # no UI tensor deserialization or compatibility claim
    view.resume_selector.setCurrentIndex(view.resume_selector.findData("fresh"))
    assert "resume_from" not in view.collect_config()


def test_custom_warm_start_remains_explicit_until_user_changes_it(setup_view, tmp_path):
    view, _, service, _ = setup_view
    config = service.load()
    config["warm_start"] = str(tmp_path / "weights.pt")
    path = service.save_session(config)
    view.load_config(path)
    assert view.resume_selector.currentData() == "configured"
    assert view.collect_config()["warm_start"] == config["warm_start"]
    assert not view.resume_note.isHidden()
    view.resume_selector.setCurrentIndex(view.resume_selector.findData("fresh"))
    assert "warm_start" not in view.collect_config()
    assert view.resume_note.isHidden()


def test_background_task_shares_state_and_stop_without_fake_pause(
    setup_view, monkeypatch
):
    view, controller, _, background = setup_view
    stops = []
    monkeypatch.setattr(
        "desktop.services.background_training.request_background_stop",
        lambda: stops.append(True),
    )
    background.update(
        active=True,
        status="RUNNING",
        iteration=12,
        metrics={"evaluation_success_rate": 0.5, "training_wall_time": 321},
    )
    view._background_status()
    assert view.task_source.text() == "Background training"
    assert view.state_label.text() == "RUNNING"
    assert not view.start_button.isEnabled()
    assert not view.pause_button.isEnabled()
    assert not view.resume_button.isEnabled()
    assert view.metrics["iteration"].text() == "12"
    assert view.metrics["success_rate"].text() == "50.0%"
    assert "unavailable" in view.background_notice.text()
    view.stop_button.click()
    assert stops == [True]
    assert controller.calls == []
    assert not view.stop_button.isEnabled()
    assert view.config_group.isHidden() and view.advanced.isHidden()
    assert "UNKNOWN" in view.task_summary.text()
    background.update(active=False, status="STOPPED")
    view._background_status()
    assert view.task_source.text() == "Last background task"
    assert view.state_label.text() == "STOPPED"
    assert view.start_button.isEnabled()
    assert not view.config_group.isHidden()


def test_embedded_dashboard_shows_one_overview_and_keeps_details(qtbot, setup_view):
    view, _, _, _ = setup_view
    dashboard = DashboardView(refresh_ms=10000)
    qtbot.addWidget(dashboard)
    view.set_dashboard(dashboard)
    assert view.summary_widget.isHidden() and view.more_metrics.isHidden()
    assert dashboard.PRIMARY_CARDS == (
        "iteration",
        "success_rate",
        "training_wall_time",
    )
    assert dashboard.more_metrics.body.isHidden()
    dashboard.update_metrics(
        {"iteration": 4, "evaluation_success_rate": 0.5, "policy_loss": 0.3}
    )
    dashboard.flush()
    dashboard.update_metrics({"iteration": 4, "evaluation_success_rate": 0.5})
    dashboard.flush()
    assert dashboard.charts["success_rate"].series.count() == 1
    assert dashboard.cards["policy_loss"].value_label.text() == "0.3000"
    dashboard.timer.stop()


def test_language_switch_preserves_four_choice_values(setup_view):
    view, _, _, _ = setup_view
    original = view.collect_config()
    set_language("zh_CN")
    assert view.squad_mode.currentText() == "固定编队"
    assert view.resume_selector.currentText() == "从头训练"
    assert view.size_selector.currentText() == "快速验证"
    assert view.advanced.toggle.text() == "高级配置与 YAML"
    assert view.stop_button.text() == "停止并保存"
    assert view.collect_config() == original
    set_language("en")
    assert view.start_button.text() == tr("START TRAINING")


def test_new_foreground_task_clears_background_scores_and_curves(qtbot, setup_view):
    from desktop.models.training_state import TrainingEvent

    view, _, _, background = setup_view
    dashboard = DashboardView(refresh_ms=10000)
    qtbot.addWidget(dashboard)
    view.set_dashboard(dashboard)
    background.update(
        active=False,
        status="STOPPED",
        pid=1,
        iteration=22,
        metrics={"evaluation_success_rate": 0.75, "training_wall_time": 90},
    )
    view._background_status()
    dashboard.flush()
    assert dashboard.cards["success_rate"].value_label.text() == "75.0%"
    assert dashboard.charts["success_rate"].series.count() == 1
    dashboard.set_status(training="STARTING")
    view.start_button.click()
    dashboard.flush()
    assert dashboard.cards["training"].value_label.text() == "STARTING"
    assert dashboard.cards["success_rate"].value_label.text() == "—"
    assert dashboard.charts["success_rate"].series.count() == 0
    assert view.metrics["success_rate"].text() == "—"
    view._event(
        TrainingEvent("training_metrics", {"iteration": 1, "success_rate": 0.25})
    )
    dashboard.flush()
    set_language("zh_CN")
    assert view.metrics["success_rate"].text() == "25.0%"
    assert dashboard.cards["success_rate"].value_label.text() == "25.0%"
    assert dashboard.charts["success_rate"].series.count() == 1
    dashboard.timer.stop()


def test_embedded_overview_keeps_episode_details_accessible(qtbot, setup_view):
    from desktop.models.training_state import TrainingEvent

    view, _, _, _ = setup_view
    dashboard = DashboardView(refresh_ms=10000)
    qtbot.addWidget(dashboard)
    view.set_dashboard(dashboard)
    view._event(
        TrainingEvent(
            "episode_finished",
            {
                "episode": 1,
                "kills": 4,
                "leaks": 7,
                "success": False,
                "game_time": 65.5,
                "generated_states": 90,
            },
        )
    )
    dashboard.flush()
    assert dashboard.cards["game_time"].value_label.text() == "65.5000"
    assert dashboard.cards["generated_states"].value_label.text() == "90"
    assert (
        dashboard.cards["episode_result"].value_label.text()
        == "Failure · 4 kills / 7 leaks"
    )
    dashboard.more_metrics.set_expanded(True)
    assert not dashboard.more_metrics.body.isHidden()
    set_language("zh_CN")
    assert (
        dashboard.cards["episode_result"].value_label.text()
        == "未通关 · 击杀 4 / 漏怪 7"
    )
    dashboard.timer.stop()


def test_running_task_keeps_lifecycle_and_save_messages_visible(setup_view):
    from desktop.models.training_state import TrainingEvent

    view, controller, _, _ = setup_view
    view.start_button.click()
    controller.set_state("PAUSING")
    assert view.advanced.isHidden()
    assert not view.task_notice.isHidden()
    assert "next safe" in view.task_notice.text()
    controller.set_state("PAUSED")
    assert view.task_notice.isHidden()
    view._save_checkpoint()
    assert "pending" in view.task_notice.text()
    assert not view.task_notice.isHidden()
    set_language("zh_CN")
    assert "已排队" in view.task_notice.text()
    assert "固定编队" in view.task_summary.text()
    view._event(TrainingEvent("checkpoint_saved", {"path": "/tmp/latest.pt"}))
    assert "latest.pt" in view.task_notice.text()
    assert "已排队" not in view.task_notice.text()
