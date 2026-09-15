"""Compact task status with inspectable system diagnostics."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QStatusBar, QLabel, QToolButton, QMenu
from desktop.i18n import tr, language_manager, register_translations

register_translations({
    "Starting services…": "正在启动服务…", "Available": "可用", "Unavailable": "不可用",
    "Training: {training}   ·   Battle: {battle}": "训练：{training}   ·   对战：{battle}",
    "System status · {rss:.0f} MB": "系统状态 · {rss:.0f} MB",
    "Process CPU: {cpu:.1f}%": "进程 CPU：{cpu:.1f}%",
    "Process memory: {rss:.0f} MB": "进程内存：{rss:.0f} MB",
    "MPS: {mps}": "MPS：{mps}",
    "Training device: {device}": "训练设备：{device}",
    "Open logs": "查看日志",
})


class ResearchStatusBar(QStatusBar):
    diagnostics_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel(tr("Starting services…"))
        self.addWidget(self.label, 1)
        self.system_button = QToolButton()
        self.system_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.system_button)
        self.system_actions = [menu.addAction("") for _ in range(4)]
        for action in self.system_actions:
            action.setEnabled(False)
        menu.addSeparator()
        self.logs_action = menu.addAction(tr("Open logs"))
        self.logs_action.triggered.connect(self.diagnostics_requested)
        self.system_button.setMenu(menu)
        self.addPermanentWidget(self.system_button)
        self._last = None
        language_manager.language_changed.connect(self.retranslate)
        self.retranslate()

    def retranslate(self, *_):
        self.logs_action.setText(tr("Open logs"))
        if self._last:
            self.update_status(*self._last)
        else:
            self.label.setText(tr("Starting services…"))
            self.system_button.setText(tr("Starting services…"))

    def update_status(self, ui_state, metrics):
        self._last = (ui_state, metrics)
        self.label.setText(tr("Training: {training}   ·   Battle: {battle}").format(
            training=tr(ui_state.training), battle=tr(ui_state.simulator)))
        rss = metrics.get("rss_mb", 0)
        self.system_button.setText(tr("System status · {rss:.0f} MB").format(rss=rss))
        details = (
            tr("Process CPU: {cpu:.1f}%").format(cpu=metrics.get("cpu_percent", 0)),
            tr("Process memory: {rss:.0f} MB").format(rss=rss),
            tr("MPS: {mps}").format(mps=tr(ui_state.mps)),
            tr("Training device: {device}").format(device=ui_state.device),
        )
        for action, text in zip(self.system_actions, details):
            action.setText(text)
        self.system_button.setToolTip("\n".join(details))
