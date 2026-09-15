"""One workspace coordinating presentation adapters, with cooperative shutdown."""

import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QListWidget, QStackedWidget, QMessageBox, QApplication, QTabWidget)

from desktop.paths import WORKSPACE_ROOT, RESOURCE_ROOT, ensure_workspace
from desktop.models.ui_state import UIState
from desktop.services.settings_service import SettingsService
from desktop.services.config_service import ConfigService
from desktop.services.logging_service import LoggingService
from desktop.controllers.system_controller import SystemController
from desktop.controllers.simulator_controller import SimulatorController
from desktop.controllers.training_controller import TrainingController
from desktop.controllers.model_controller import ModelController
from desktop.controllers.data_controller import DataController
from desktop.controllers.mcts_controller import MCTSController
from desktop.views.dashboard_view import DashboardView
from desktop.views.simulator_view import SimulatorView
from desktop.views.training_view import TrainingView
from desktop.views.mcts_view import MCTSView
from desktop.views.models_view import ModelsView
from desktop.views.game_data_view import GameDataView
from desktop.views.logs_view import LogsView
from desktop.views.settings_view import SettingsView
from desktop.widgets.status_bar import ResearchStatusBar
from desktop.i18n import tr, bind_text, set_language, language_manager, register_translations

register_translations({
    "Dashboard": "仪表盘", "MCTS": "MCTS", "Models": "模型", "Game Data": "游戏数据", "Logs": "日志", "Settings": "设置",
    "Battle": "对战", "Research tools": "研究工具", "Search analysis": "搜索分析",
    "Logs and diagnostics": "日志与诊断", "Desktop 0.6 · Focused workspace": "桌面版 0.6 · 简洁工作区",
    "Finish the current battle before changing its model.": "请先结束当前对战，再更换本局模型。",
    "Another task is using the compute engine.": "另一个任务正在使用计算引擎。",
    "Close the application?": "关闭应用？",
    "Background training will continue after this window closes. Use Stop and save on the Training page to end it.": "关闭窗口后，后台训练仍会继续。如需结束，请在训练页点击“停止并保存”。",
    "Close window": "关闭窗口",
    "Refresh models": "刷新模型", "AlphaZero Research\nEnvironment": "AlphaZero 研究环境",
    "Desktop 0.5 · Mechanics Verification": "桌面版 0.5 · 战斗机制验证", "Background task failed": "后台任务失败",
    "Unknown error": "未知错误", "Training is currently running": "训练正在运行",
    "Stop training and quit? Completed work will be saved at a safe boundary.": "停止训练并退出？已完成的工作将在安全边界保存。",
    "Cancel": "取消", "Stop and Quit": "停止并退出", "Stopping workers safely…": "正在安全停止后台任务…",
})


class MainWindow(QMainWindow):
    error_raised = Signal(str)
    PAGE_KEYS = ("training", "battle", "models", "research", "settings")
    PAGE_NAMES = ("Training", "Battle", "Models", "Research tools", "Settings")
    RESEARCH_KEYS = ("mcts", "data", "logs")
    RESEARCH_NAMES = ("Search analysis", "Game Data", "Logs and diagnostics")

    def __init__(self, parent=None, settings_service=None, start_services=True):
        super().__init__(parent)
        ensure_workspace()
        self.setWindowTitle("Arknights Zero — AlphaZero Research Environment")
        self.resize(1160, 820)
        self.setMinimumSize(920, 620)
        self.settings_service = settings_service or SettingsService()
        self.settings = self.settings_service.load()
        set_language(self.settings.get("language", "zh_CN"))
        self.log_service = LoggingService(self.settings["log_dir"])
        self.ui_state = UIState()
        self.system_metrics = {}
        self._closing = False
        self._shutdown_done = False
        self._simulation_started = None
        self._dialogs = []
        self.simulator = SimulatorController(data_dir=self.settings["data_dir"])
        self.training = TrainingController(config_service=ConfigService(root=WORKSPACE_ROOT))
        self.models = ModelController(self.settings["checkpoint_dir"], data_dir=self.settings["data_dir"])
        self.data = DataController(data_dir=self.settings["data_dir"])
        self.mcts = MCTSController(data_dir=self.settings["data_dir"])
        self.system = SystemController(self)
        self.controllers = (self.simulator, self.training, self.models, self.data, self.mcts)
        self._build_ui()
        self._connect_services()
        language_manager.language_changed.connect(self.retranslate)
        self.apply_display_settings(self.settings)
        self.dashboard.set_status(simulator="READY", training="IDLE", stage="0-1", model="None",
            engine="Checking", device="Checking")
        self.log_timer = QTimer(self)
        self.log_timer.setInterval(250)
        self.log_timer.timeout.connect(self.flush_logs)
        self.log_timer.start()
        self.guard_timer = QTimer(self)
        self.guard_timer.setInterval(250)
        self.guard_timer.timeout.connect(self.update_compute_controls)
        self.guard_timer.start()
        self.close_timer = QTimer(self)
        self.close_timer.setInterval(100)
        self.close_timer.timeout.connect(self._finish_close)
        self.log_service.log("APP", "Arknights Zero desktop started")
        if start_services:
            QTimer.singleShot(0, self.start_services)

    def _build_ui(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        left = QVBoxLayout()
        title = QLabel("Arknights Zero")
        title.setObjectName("appTitle")
        left.addWidget(title)
        subtitle = QLabel("AlphaZero Research\nEnvironment")
        bind_text(subtitle, "AlphaZero Research\nEnvironment")
        left.addWidget(subtitle)
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.addItems([tr(name) for name in self.PAGE_NAMES[:3]])
        self.sidebar.setFixedWidth(165)
        left.addWidget(self.sidebar, 1)
        self.utility_sidebar = QListWidget()
        self.utility_sidebar.setObjectName("utilitySidebar")
        self.utility_sidebar.addItems([tr(name) for name in self.PAGE_NAMES[3:]])
        self.utility_sidebar.setFixedWidth(165)
        self.utility_sidebar.setFixedHeight(104)
        left.addWidget(self.utility_sidebar)
        version = QLabel()
        version.setWordWrap(True)
        version.setMaximumWidth(165)
        bind_text(version, "Desktop 0.6 · Focused workspace")
        left.addWidget(version)
        layout.addLayout(left)
        self.stack = QStackedWidget()
        self.dashboard = DashboardView(refresh_ms=self.settings["refresh_ms"])
        self.simulator_view = SimulatorView(self.simulator)
        self.training_view = TrainingView(self.training, config_service=ConfigService(root=WORKSPACE_ROOT))
        self.training_view.set_dashboard(self.dashboard)
        if hasattr(self.training_view, "apply_defaults"):
            self.training_view.apply_defaults(self.settings)
        self.simulator.set_fps(self.settings["visualization_fps"])
        self.mcts_view = MCTSView(self.mcts)
        self.models_view = ModelsView(self.models, autoload=False)
        for button in (self.training_view.start_button, self.simulator_view.battle_button,
                       self.models_view.use_button):
            button.setProperty("primary", True)
        self.data_view = GameDataView(self.data, autoload=False)
        self.logs_view = LogsView(maximum=self.settings["max_log_lines"])
        self.settings_view = SettingsView(self.settings_service, self.settings)
        self.research_view = QWidget()
        research_layout = QVBoxLayout(self.research_view)
        research_title = bind_text(QLabel(), "Research tools")
        research_title.setObjectName("pageTitle")
        research_layout.addWidget(research_title)
        self.research_tabs = QTabWidget()
        for view, name in zip((self.mcts_view, self.data_view, self.logs_view), self.RESEARCH_NAMES):
            self.research_tabs.addTab(view, tr(name))
        research_layout.addWidget(self.research_tabs, 1)
        for view in (self.training_view, self.simulator_view, self.models_view,
                     self.research_view, self.settings_view):
            self.stack.addWidget(view)
        self.sidebar.currentRowChanged.connect(lambda row: self.navigate(self.PAGE_KEYS[row]) if row >= 0 else None)
        self.utility_sidebar.currentRowChanged.connect(lambda row: self.navigate(self.PAGE_KEYS[row + 3]) if row >= 0 else None)
        self.navigate("training")
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(container)
        self.research_status = ResearchStatusBar()
        self.research_status.diagnostics_requested.connect(lambda: self.navigate("logs"))
        self.setStatusBar(self.research_status)

    def navigate(self, page):
        """Stable page identities shared by navigation and contextual shortcuts."""
        page = {"simulator": "battle", "dashboard": "training"}.get(page, page)
        if page in self.RESEARCH_KEYS:
            self.research_tabs.setCurrentIndex(self.RESEARCH_KEYS.index(page))
            page = "research"
        index = self.PAGE_KEYS.index(page)
        for navigation, selected in ((self.sidebar, index if index < 3 else -1),
                                     (self.utility_sidebar, index - 3 if index >= 3 else -1)):
            previous = navigation.blockSignals(True)
            navigation.setCurrentRow(selected)
            navigation.blockSignals(previous)
        self.stack.setCurrentIndex(index)

    def use_model_in_battle(self, path):
        """A model choice is explicit per battle; never replace a running state."""
        if self.simulator.is_busy():
            self.show_error(tr("Finish the current battle before changing its model."))
            return
        from desktop.services.background_training import read_background_training
        if self.training.is_busy or self.models.is_busy() or self.mcts.is_busy() or read_background_training().get("active"):
            self.show_error(tr("Another task is using the compute engine."))
            return
        if self.simulator_view.set_model(path) is not False:
            # This command explicitly asks to use a model, so prepare an AI
            # battle while retaining the user's saved manual roster.
            self.simulator_view.experiment_mode.setCurrentIndex(0)
            self.simulator_view.mode.setCurrentIndex(
                self.simulator_view.mode.findData("Automatic control")
            )
            self.navigate("battle")

    def _connect_services(self):
        for controller in self.controllers:
            controller.error.connect(self.show_error)
        self.system.error.connect(self.show_error)
        self.system.metrics_updated.connect(self.on_system_metrics)
        self.system.device_detected.connect(self.on_device_detected)
        self.training.state_changed.connect(self.on_training_state)
        self.training.metrics_updated.connect(self.dashboard.update_metrics)
        self.training.event_received.connect(self.on_training_event)
        self.training.log.connect(lambda text: self.log_service.log("TRAIN", text))
        self.simulator.state_changed.connect(self.on_simulator_state)
        self.simulator.snapshot.connect(self.on_simulator_snapshot)
        self.simulator.log.connect(lambda text: self.log_service.log("SIM", text))
        self.models.model_loaded.connect(self.on_model_loaded)
        self.models_view.use_in_simulator.connect(self.use_model_in_battle)
        self.models.status_changed.connect(lambda text: self.log_service.log("APP", "Models: " + text))
        self.mcts.status_changed.connect(lambda text: self.log_service.log("MCTS", text))
        self.models.status_changed.connect(self.update_compute_controls)
        self.mcts.status_changed.connect(self.update_compute_controls)
        self.settings_view.settings_saved.connect(self.apply_display_settings)

    def start_services(self):
        if self._closing:
            return
        self.system.start()
        self.models.refresh()
        self.data.load()

    def retranslate(self, *_):
        for index, name in enumerate(self.PAGE_NAMES[:3]):
            self.sidebar.item(index).setText(tr(name))
        for index, name in enumerate(self.PAGE_NAMES[3:]):
            self.utility_sidebar.item(index).setText(tr(name))
        for index, name in enumerate(self.RESEARCH_NAMES):
            self.research_tabs.setTabText(index, tr(name))

    def on_device_detected(self, result):
        self.ui_state.mps = result["mps"]
        self.dashboard.set_status(engine="AVAILABLE" if result["training_available"] else "NOT AVAILABLE",
                                  device=f"CPU · MPS {result['mps']}")
        self.on_system_metrics(self.system_metrics)

    def on_system_metrics(self, metrics):
        self.system_metrics = dict(metrics)
        self.dashboard.update_metrics(metrics)
        self.research_status.update_status(self.ui_state, self.system_metrics)

    def on_training_state(self, state):
        self.ui_state.training = state
        self.dashboard.set_status(training=state)
        self.log_service.log("TRAIN", f"State: {state}")
        self.research_status.update_status(self.ui_state, self.system_metrics)
        self.update_compute_controls()

    def on_training_event(self, event):
        kind = getattr(event, "kind", "")
        payload = dict(getattr(event, "payload", {}))
        if kind == "training_started":
            self.ui_state.device = str(payload.get("device", "CPU")).upper()
            self.dashboard.set_status(device=self.ui_state.device)
        elif kind in ("checkpoint_saved", "checkpoint_metadata_updated"):
            if not self._closing:
                self.models.refresh()
            if payload.get("path"):
                self.dashboard.set_status(model=Path(payload["path"]).name)
        if "iteration" in payload:
            self.dashboard.update_metrics({"iteration": payload["iteration"]})

    def on_simulator_state(self, state):
        self.ui_state.simulator = state
        self.dashboard.set_status(simulator=state)
        self.research_status.update_status(self.ui_state, self.system_metrics)
        self.update_compute_controls()
        if state in ("RUNNING", "STARTING"):
            self._simulation_started = time.perf_counter()

    def on_simulator_snapshot(self, snapshot):
        self.ui_state.stage = snapshot.stage_id
        values = {}
        if self._simulation_started is not None:
            elapsed = time.perf_counter() - self._simulation_started
            if elapsed > 0:
                values["simulator_speed"] = f"{snapshot.time / elapsed:.1f}×"
        self.dashboard.update_metrics(values)

    def on_model_loaded(self, model):
        self.ui_state.model = model.name
        self.mcts.set_model(model.path)
        self.training.set_evaluation_checkpoint(model.path)
        self.log_service.log("APP", f"Loaded model {model.name}")

    def update_compute_controls(self):
        from desktop.services.background_training import read_background_training
        background = read_background_training()
        training_busy = self.training.is_busy or background.get("active", False)
        if not self.training.is_busy:
            self.ui_state.training = background.get("status", "RUNNING") if background.get("active") else self.training.state
            self.research_status.update_status(self.ui_state, self.system_metrics)
        neural_sim_busy = self.simulator.is_busy() and (self.simulator._last_options or {}).get("mode") in ("Automatic control", "Hybrid control")
        research_busy = self.models.is_busy() or self.mcts.is_busy()
        self.mcts_view.setEnabled(not training_busy and not neural_sim_busy and not self.models.is_busy())
        self.models_view.set_compute_available(not training_busy and not self.simulator.is_busy() and not self.mcts.is_busy())
        self.simulator_view.set_compute_available(not training_busy and not research_busy)
        if hasattr(self.training_view, "set_compute_available"):
            self.training_view.set_compute_available(not research_busy and not neural_sim_busy and not background.get("active", False))

    def apply_display_settings(self, values):
        set_language(values.get("language", "zh_CN"))
        self.dashboard.timer.setInterval(values["refresh_ms"])
        self.logs_view.set_maximum(values["max_log_lines"])
        self.simulator_view.timeline.set_max_lines(values["max_log_lines"])
        self.simulator.set_fps(values["visualization_fps"])
        scheme = {"System": Qt.ColorScheme.Unknown, "Dark": Qt.ColorScheme.Dark,
                  "Light": Qt.ColorScheme.Light}[values["theme"]]
        QApplication.instance().styleHints().setColorScheme(scheme)
        style = RESOURCE_ROOT / "desktop/resources/styles/base.qss"
        if style.exists():
            self.setStyleSheet(style.read_text())

    def flush_logs(self):
        self.logs_view.add_entries(self.log_service.drain())

    def show_error(self, *parts):
        message = "\n".join(str(p) for p in parts if p)
        self.log_service.log("ERROR", message, error=True)
        self.error_raised.emit(message)
        dialog = QMessageBox(QMessageBox.Icon.Critical, tr("Background task failed"),
                             message.splitlines()[-1] if message else tr("Unknown error"),
                             QMessageBox.StandardButton.Ok, self)
        dialog.setDetailedText(message)
        self._dialogs.append(dialog)
        dialog.finished.connect(lambda: self._dialogs.remove(dialog) if dialog in self._dialogs else None)
        dialog.open()

    def closeEvent(self, event):
        if self._shutdown_done:
            event.accept()
            return
        event.ignore()
        if self._closing:
            return
        if self.training.is_busy:
            dialog = QMessageBox(QMessageBox.Icon.Question, tr("Training is currently running"),
                tr("Stop training and quit? Completed work will be saved at a safe boundary."), parent=self)
            cancel = dialog.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
            stop = dialog.addButton(tr("Stop and Quit"), QMessageBox.ButtonRole.AcceptRole)
            dialog.setDefaultButton(cancel)
            dialog.exec()
            if dialog.clickedButton() is not stop:
                return
        else:
            from desktop.services.background_training import read_background_training
            if read_background_training().get("active"):
                dialog = QMessageBox(QMessageBox.Icon.Question, tr("Close the application?"),
                    tr("Background training will continue after this window closes. Use Stop and save on the Training page to end it."), parent=self)
                dialog.addButton(tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
                close = dialog.addButton(tr("Close window"), QMessageBox.ButtonRole.AcceptRole)
                dialog.exec()
                if dialog.clickedButton() is not close:
                    return
        self.begin_shutdown()

    def begin_shutdown(self):
        """Also used by the runtime harness; never forcibly terminate a worker."""
        self._closing = True
        self.stack.setEnabled(False)
        self.sidebar.setEnabled(False)
        self.utility_sidebar.setEnabled(False)
        self.guard_timer.stop()
        self.system.timer.stop()
        for controller in self.controllers:
            if hasattr(controller, "stop"):
                controller.stop()
        self.statusBar().showMessage(tr("Stopping workers safely…"))
        self.close_timer.start()

    def _finish_close(self):
        ready = True
        for controller in self.controllers:
            if not controller.shutdown(timeout_ms=0):
                ready = False
        if not self.system.shutdown():
            ready = False
        if not ready:
            return
        # QThread.wait can finish before its queued cleanup signal is delivered.
        if self.training.is_busy or any(c.is_busy() for c in (self.simulator,self.models,self.data,self.mcts)):
            return
        self.close_timer.stop()
        self.log_timer.stop()
        self.dashboard.timer.stop()
        self.log_service.log("APP", "All workers stopped safely")
        self.flush_logs()
        self.log_service.close()
        self._shutdown_done = True
        self.close()
