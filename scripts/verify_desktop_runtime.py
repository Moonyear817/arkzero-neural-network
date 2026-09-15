#!/usr/bin/env python3
"""Desktop acceptance using Qt button actions and real core engines.

The report records whether this used the native or offscreen Qt platform.
"""

from pathlib import Path
import argparse
import json
import os
import platform
import sys
import time

STARTED = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workspace", type=Path, help="Isolated writable acceptance workspace")
    args = parser.parse_args()
    if args.workspace:
        os.environ["ARKNIGHTS_ZERO_WORKSPACE"] = str(args.workspace.resolve())
    import psutil
    from PySide6 import __version__ as qt_version
    from PySide6.QtCore import Qt, QTimer, QEventLoop
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from desktop.main_window import MainWindow
    from desktop.paths import WORKSPACE_ROOT, RESOURCE_ROOT
    import shutil
    model_source = RESOURCE_ROOT / "outputs/mechanics_verification/best.pt"
    if not model_source.is_file():
        model_source = RESOURCE_ROOT / "checkpoints/best.pt"
    if args.workspace:
        destination = WORKSPACE_ROOT / "checkpoints/best.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(model_source, destination)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    from desktop.i18n import set_language
    set_language("en")
    errors = []
    window.error_raised.connect(errors.append)
    window.show()
    output = args.output or WORKSPACE_ROOT / "research/desktop/simplification/runtime"
    output.mkdir(parents=True, exist_ok=True)
    report = {"architecture": platform.machine(), "pyside6": qt_version,
              "qt_platform": app.platformName()}

    def until(predicate, timeout=60):
        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(20)
        def check():
            if predicate() or errors:
                loop.quit()
        poll.timeout.connect(check)
        deadline = QTimer()
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll.start()
        deadline.start(timeout * 1000)
        if not predicate():
            loop.exec()
        poll.stop()
        deadline.stop()
        app.processEvents()
        if errors:
            raise RuntimeError("\n".join(errors))
        if not predicate():
            raise TimeoutError(f"UI timeout: probe={window.system.probe.isRunning()}, data={window.data.status}, models={window.models.status}, sim={window.simulator.status}, train={window.training.state}")

    def click(button):
        assert button.isEnabled(), button.text()
        assert button.isVisible(), button.text()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        app.processEvents()

    def capture(name):
        QTest.qWait(100)
        assert window.grab().save(str(output / (name + ".png")))

    try:
        until(lambda: window.isVisible())
        report["first_window_seconds"] = time.perf_counter() - STARTED
        until(lambda: not window.system.probe.isRunning() and not window.models.is_busy() and not window.data.is_busy())
        report["services_ready_seconds"] = time.perf_counter() - STARTED
        report["mps"] = window.ui_state.mps
        report["data_records"] = len(window.data.snapshot.records)
        process = psutil.Process()
        process.cpu_percent()
        QTest.qWait(5000)
        report["idle_cpu_percent_5s"] = process.cpu_percent()
        report["idle_rss_mb"] = process.memory_info().rss / 1024 ** 2
        capture("dashboard_idle")

        window.navigate("battle")
        view = window.simulator_view
        view.more_settings.set_expanded(True)
        view.experiment_mode.setCurrentIndex(view.experiment_mode.findData("Scripted baseline"))
        view.more_settings.set_expanded(False)
        view.speed.setCurrentText("10x")
        click(view.battle_button)
        until(lambda: window.simulator.last_snapshot is not None and window.simulator.last_snapshot.time >= 22)
        click(view.battle_button)
        until(lambda: window.simulator.status == "PAUSED")
        before = window.simulator.last_snapshot.time
        view.timeline_section.set_expanded(True)
        capture("simulator_running")
        click(view.step_button)
        until(lambda: window.simulator.last_snapshot.time > before)
        report["simulator_step_advanced"] = True
        view.speed.setCurrentText("MAX")
        click(view.battle_button)
        until(lambda: not window.simulator.is_busy())
        result = window.simulator.last_snapshot
        assert result.terminal and result.kills == 11 and result.leaks == 0
        report["simulator"] = {"stage": result.stage_id, "kills": result.kills,
            "leaks": result.leaks, "time": result.time, "result": result.result,
            "timeline_lines": view.timeline.console.document().blockCount()}
        assert report["simulator"]["timeline_lines"] > 10
        capture("simulator_complete")

        window.navigate("models")
        canonical = str(WORKSPACE_ROOT / "checkpoints/best.pt") if args.workspace else str(model_source)
        if canonical not in {x.path for x in window.models.models}:
            window.models.checkpoint_dir = Path(canonical).parent
            window.models.refresh()
            until(lambda: not window.models.is_busy())
        model = next(x for x in window.models.models if x.path == canonical)
        # Listing is sorted by a presentation table; locate using the displayed name/path.
        window.models.select(model.path)
        window.models.load_model()
        until(lambda: not window.models.is_busy())
        assert window.models.loaded
        report["model_loaded"] = {"name": window.models.loaded.name, "iteration": window.models.loaded.iteration}
        capture("models")

        window.navigate("mcts")
        window.mcts.search(simulations=16)
        until(lambda: not window.mcts.is_busy())
        report["mcts"] = {"actions": len(window.mcts.result.rows),
            "network_value": window.mcts.result.network_value, "nodes": window.mcts.result.node_count}
        capture("mcts")

        window.navigate("training")
        training = window.training_view
        for key, value in {"iterations": 2, "episodes_per_iteration": 1, "mcts_simulations": 4,
                "training_steps_per_iteration": 2, "batch_size": 8, "replay_buffer_size": 1024,
                "evaluation_episodes": 1}.items():
            training.fields[key].setValue(value)
        training.fields["device"].setCurrentText("CPU")
        metrics, states = [], []
        window.training.metrics_updated.connect(lambda item: metrics.append(dict(item)))
        window.training.state_changed.connect(states.append)
        click(training.start_button)
        until(lambda: window.training.state == "RUNNING")
        click(training.start_button)
        until(lambda: window.training.state == "PAUSED")
        report["pause_confirmed"] = True
        click(training.start_button)
        until(lambda: len(metrics) >= 1, timeout=180)
        capture("training_live")
        # Navigation remains interactive during background work.
        window.navigate("logs")
        assert window.research_tabs.currentWidget() is window.logs_view
        window.navigate("training")
        click(training.stop_button)
        until(lambda: not window.training.is_busy)
        checkpoint = window.training._last_checkpoint
        assert checkpoint and Path(checkpoint).is_file()
        report["training"] = {"states": states, "iterations_with_metrics": len(metrics),
            "last_metrics": metrics[-1], "checkpoint": checkpoint,
            "stopped_safely": window.training.state == "IDLE", "navigation_while_running": True}
        window.models.refresh()
        until(lambda: not window.models.is_busy())
        report["new_checkpoint_listed"] = any(m.path == checkpoint for m in window.models.models)
        assert report["new_checkpoint_listed"]
        window.navigate("settings")
        language = window.settings_view.fields["language"]
        language.setCurrentIndex(language.findData("en"))
        language.setCurrentIndex(language.findData("zh_CN"))
        assert window.utility_sidebar.item(1).text() == "设置"
        assert window.training_view.start_button.text() == "开始训练"
        assert window.settings_service.load()["language"] == "zh_CN"
        capture("settings_chinese")
        language.setCurrentIndex(language.findData("en"))
        assert window.utility_sidebar.item(1).text() == "Settings"
        capture("settings_english")
        language.setCurrentIndex(language.findData("zh_CN"))
        window.navigate("training")
        window.dashboard.flush()
        capture("dashboard_chinese")
        window.navigate("mcts")
        capture("mcts_chinese")
        report["language_switch_and_persistence"] = True
        report["final_language"] = "zh_CN"
        report["rss_after_work_mb"] = process.memory_info().rss / 1024 ** 2
        report["errors"] = errors
    finally:
        window.begin_shutdown()
        until(lambda: window._shutdown_done)
        (output / "acceptance.json").write_text(json.dumps(report, indent=2, default=dict) + "\n")


if __name__ == "__main__":
    main()
