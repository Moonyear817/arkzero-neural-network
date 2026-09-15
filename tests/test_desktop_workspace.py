"""Cross-page behavior of the simplified workspace."""

import pytest

pytest.importorskip("PySide6")

from desktop.i18n import get_language, set_language
from desktop.main_window import MainWindow
from desktop.services.settings_service import SettingsService


@pytest.fixture
def workspace(qtbot, tmp_path):
    language = get_language()
    window = MainWindow(settings_service=SettingsService(tmp_path), start_services=False)
    yield window
    window.begin_shutdown()
    qtbot.waitUntil(lambda: window._shutdown_done, timeout=10000)
    window.deleteLater()
    set_language(language)


def test_navigation_and_translation_preserve_page_identity(workspace):
    workspace.navigate("logs")
    set_language("en")
    assert [workspace.sidebar.item(i).text() for i in range(3)] == ["Training", "Battle", "Models"]
    assert workspace.research_tabs.currentWidget() is workspace.logs_view
    set_language("zh_CN")
    assert [workspace.sidebar.item(i).text() for i in range(3)] == ["训练", "对战", "模型"]
    assert workspace.utility_sidebar.item(1).text() == "设置"
    assert workspace.research_tabs.currentWidget() is workspace.logs_view
    workspace.navigate("training")
    assert workspace.dashboard.parent() is not None
    assert workspace.stack.currentWidget() is workspace.training_view


def test_explicit_model_handoff_targets_battle_only(workspace, tmp_path):
    checkpoint = tmp_path / "chosen.pt"
    checkpoint.write_bytes(b"deferred worker loading")
    before_squad = workspace.simulator_view.selected_squad
    workspace.models_view.use_in_simulator.emit(str(checkpoint))
    assert workspace.stack.currentWidget() is workspace.simulator_view
    assert workspace.simulator_view.model_path.text() == str(checkpoint)
    assert workspace.simulator_view.selected_squad == before_squad
    assert workspace.simulator_view.mode.currentData() == "Automatic control"
    assert workspace.simulator_view.experiment_mode.currentData() == ""
    assert workspace.simulator._worker is None
    assert workspace.training._last_checkpoint is None


def test_model_handoff_does_not_change_an_active_battle(workspace, tmp_path, monkeypatch):
    checkpoint = tmp_path / "chosen.pt"
    checkpoint.write_bytes(b"deferred worker loading")
    old_path = workspace.simulator_view.model_path.text()
    errors = []
    workspace.error_raised.connect(errors.append)
    with monkeypatch.context() as patch:
        patch.setattr(workspace.simulator, "is_busy", lambda: True)
        workspace.models_view.use_in_simulator.emit(str(checkpoint))
    assert errors
    assert workspace.simulator_view.model_path.text() == old_path
    assert workspace.stack.currentWidget() is workspace.training_view


def test_status_diagnostics_navigates_to_logs(workspace):
    workspace.research_status.diagnostics_requested.emit()
    assert workspace.stack.currentWidget() is workspace.research_view
    assert workspace.research_tabs.currentWidget() is workspace.logs_view
    workspace.on_system_metrics({"cpu_percent": 12.5, "rss_mb": 256})
    assert "256" in workspace.research_status.system_button.text()
    assert "12.5" in workspace.research_status.system_button.toolTip()


def test_shutdown_waits_for_training_cleanup_and_does_not_rescan(workspace, monkeypatch):
    from types import SimpleNamespace
    calls = []
    with monkeypatch.context() as patch:
        patch.setattr(workspace, "_closing", True)
        patch.setattr(workspace.training, "_worker", object())
        patch.setattr(workspace.training, "shutdown", lambda timeout_ms: True)
        patch.setattr(workspace.models, "refresh", lambda: calls.append("refresh"))
        workspace.on_training_event(SimpleNamespace(kind="checkpoint_saved", payload={"path": "latest.pt"}))
        workspace._finish_close()
        assert not workspace._shutdown_done
        assert not calls


def test_display_fps_setting_reaches_simulator_immediately(workspace):
    workspace.apply_display_settings({**workspace.settings, "visualization_fps": 12})
    assert workspace.simulator._fps == 12


def test_neural_battle_loading_gates_training_immediately(workspace, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(workspace.simulator, "is_busy", lambda: True)
        patch.setattr(workspace.simulator, "_last_options", {"mode": "Automatic control"})
        workspace.on_simulator_state("LOADING")
        assert not workspace.training_view.start_button.isEnabled()
