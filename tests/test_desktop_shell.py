"""Presentation lifecycle, bounded telemetry, and configuration persistence."""

import json
import pytest

pytest.importorskip("PySide6")

from desktop.services.settings_service import SettingsService
from desktop.services.logging_service import LoggingService
from desktop.views.dashboard_view import DashboardView
from desktop.i18n import get_language, set_language


@pytest.fixture(autouse=True)
def restore_language():
    previous = get_language()
    yield
    set_language(previous)


def test_settings_roundtrip_and_validation(tmp_path):
    service = SettingsService(tmp_path)
    original = service.load()
    changed = {**original, "theme": "Dark", "refresh_ms": 500, "visualization_fps": 12}
    service.save(changed)
    assert service.load() == changed
    with pytest.raises(ValueError):
        service.save({**changed, "refresh_ms": 0})
    assert service.load() == changed


def test_logging_disk_categories_and_bounded_ui(tmp_path):
    service = LoggingService(tmp_path)
    try:
        service.log("TRAIN", "Iteration completed")
        service.log("ERROR", "Traceback: deliberate probe", error=True)
        entries = service.drain()
        assert [x.category for x in entries] == ["TRAIN", "ERROR"]
        assert "Iteration completed" in (tmp_path / "training.log").read_text()
        assert "Traceback" in (tmp_path / "errors.log").read_text()
        assert "Iteration completed" not in (tmp_path / "simulator.log").read_text()
    finally:
        service.close()


def test_dashboard_throttles_metrics_and_keeps_unknown_blank(qtbot):
    dashboard = DashboardView(refresh_ms=10000)
    qtbot.addWidget(dashboard)
    for iteration in range(10):
        dashboard.update_metrics({"iteration": iteration, "policy_loss": iteration / 10})
    assert dashboard.charts["policy_loss"].series.count() == 0
    dashboard.flush()
    assert dashboard.charts["policy_loss"].series.count() == 1
    assert dashboard.cards["policy_loss"].value_label.text() == "0.9000"
    assert dashboard.cards["success_rate"].value_label.text() == "—"
    dashboard.timer.stop()


def test_main_window_smoke_and_safe_shutdown(qtbot, tmp_path):
    from desktop.main_window import MainWindow
    settings = SettingsService(tmp_path)
    window = MainWindow(settings_service=settings, start_services=False)
    qtbot.addWidget(window)
    assert window.stack.count() == 5
    assert window.sidebar.count() == 3
    assert window.utility_sidebar.count() == 2
    assert window.windowTitle().startswith("Arknights Zero")
    window.navigate("training")
    assert window.stack.currentWidget() is window.training_view
    window.navigate("mcts")
    assert window.stack.currentWidget() is window.research_view
    assert window.research_tabs.currentWidget() is window.mcts_view
    assert window.sidebar.currentRow() == -1
    assert window.utility_sidebar.currentRow() == 0
    window.navigate("settings")
    assert window.stack.currentWidget() is window.settings_view
    window.sidebar.setCurrentRow(1)
    assert window.stack.currentWidget() is window.simulator_view
    window.begin_shutdown()
    qtbot.waitUntil(lambda: window._shutdown_done, timeout=10000)
    assert all(not c.is_busy() for c in (window.simulator, window.models, window.data, window.mcts))
