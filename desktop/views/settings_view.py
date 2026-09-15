"""Persistent desktop preferences with stable values and live language switching."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from desktop.i18n import bind_combo, bind_text, register_translations, set_language, tr
from desktop.widgets.disclosure import Disclosure

register_translations(
    {
        "Settings": "设置",
        "Language": "界面语言",
        "Default device": "默认计算设备",
        "Theme": "外观",
        "Default config": "默认训练配置",
        "GameData directory": "游戏数据目录",
        "Checkpoint directory": "模型检查点目录",
        "Log directory": "日志目录",
        "Metrics refresh · ms": "指标刷新间隔（毫秒）",
        "Maximum visible log lines": "界面日志最大行数",
        "Map repaint FPS · not simulation FPS": "地图显示帧率（不影响模拟步长）",
        "Archive a numbered checkpoint after each completed iteration": "每轮完成后额外归档一个编号检查点",
        "Automatic checkpoints": "自动归档",
        "Save settings": "保存设置",
        "Settings saved": "设置已保存",
        "Invalid settings": "设置无效",
        "Language saved": "界面语言已保存",
        "Computation, saving and directories": "计算、保存与目录",
        "Display and refresh": "显示与刷新",
        "Language changes immediately and is saved automatically. Other settings use Save settings.": "语言切换立即生效并自动保存，其他修改请点击“保存设置”。",
        "Theme and display limits apply immediately. Paths and defaults apply when the application restarts.\nThe latest resumable checkpoint, safe-stop and final checkpoints are always preserved.": "外观和显示限制立即生效，路径及默认配置在重启应用后生效。\n始终保留最新可恢复检查点，以及安全停止和最终检查点。",
    }
)


class SettingsView(QWidget):
    settings_saved = Signal(object)

    def __init__(self, service, values, parent=None):
        super().__init__(parent)
        self.service = service
        self.values = {**service.defaults, **values}
        layout = QVBoxLayout(self)
        title = bind_text(QLabel(), "Settings")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        form = QFormLayout()
        self.computation_section = Disclosure("Computation, saving and directories")
        computation_form = QFormLayout()
        self.computation_section.add_layout(computation_form)
        self.display_section = Disclosure("Display and refresh")
        display_form = QFormLayout()
        self.display_section.add_layout(display_form)
        self.fields = {}
        language = QComboBox()
        language.setObjectName("settings_language")
        language.addItem("简体中文", "zh_CN")
        language.addItem("English", "en")
        language.setCurrentIndex(language.findData(self.values["language"]))
        self.fields["language"] = language
        form.addRow(bind_text(QLabel(), "Language"), language)
        for key, label in (("device", "Default device"), ("theme", "Theme")):
            widget = QComboBox()
            options = (
                ["Auto", "CPU", "MPS"]
                if key == "device"
                else ["System", "Light", "Dark"]
            )
            for item in options:
                widget.addItem(item, item)
            widget.setCurrentIndex(widget.findData(self.values[key]))
            bind_combo(widget, {item: item for item in options})
            self.fields[key] = widget
            target_form = form if key == "theme" else computation_form
            target_form.addRow(bind_text(QLabel(), label), widget)
        for key, label in (
            ("config", "Default config"),
            ("data_dir", "GameData directory"),
            ("checkpoint_dir", "Checkpoint directory"),
            ("log_dir", "Log directory"),
        ):
            widget = QLineEdit(self.values[key])
            self.fields[key] = widget
            computation_form.addRow(bind_text(QLabel(), label), widget)
        for key, label, low, high in (
            ("refresh_ms", "Metrics refresh · ms", 250, 10000),
            ("max_log_lines", "Maximum visible log lines", 100, 100000),
            ("visualization_fps", "Map repaint FPS · not simulation FPS", 1, 60),
        ):
            widget = QSpinBox()
            widget.setRange(low, high)
            widget.setValue(self.values[key])
            self.fields[key] = widget
            display_form.addRow(bind_text(QLabel(), label), widget)
        check = bind_text(
            QCheckBox(), "Archive a numbered checkpoint after each completed iteration"
        )
        check.setChecked(self.values["auto_save_checkpoint"])
        self.fields["auto_save_checkpoint"] = check
        computation_form.addRow(bind_text(QLabel(), "Automatic checkpoints"), check)
        layout.addLayout(form)
        language_note = bind_text(
            QLabel(),
            "Language changes immediately and is saved automatically. Other settings use Save settings.",
        )
        language_note.setWordWrap(True)
        layout.addWidget(language_note)
        layout.addWidget(self.computation_section)
        layout.addWidget(self.display_section)
        note = bind_text(
            QLabel(),
            "Theme and display limits apply immediately. Paths and defaults apply when the application restarts.\nThe latest resumable checkpoint, safe-stop and final checkpoints are always preserved.",
        )
        note.setWordWrap(True)
        self.computation_section.add_widget(note)
        self.save_button = bind_text(QPushButton(), "Save settings")
        self.save_button.clicked.connect(self.save)
        layout.addWidget(self.save_button)
        layout.addStretch()
        language.currentIndexChanged.connect(self._language_changed)

    def _language_changed(self):
        language = self.fields["language"].currentData()
        try:
            self.values = self.service.save({**self.values, "language": language})
            set_language(language)
            self.settings_saved.emit(self.values)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))

    def save(self):
        values = dict(self.values)
        for key, widget in self.fields.items():
            if isinstance(widget, QComboBox):
                values[key] = widget.currentData()
            elif isinstance(widget, QLineEdit):
                values[key] = widget.text().strip()
            elif isinstance(widget, QSpinBox):
                values[key] = widget.value()
            else:
                values[key] = widget.isChecked()
        try:
            self.values = self.service.save(values)
            set_language(self.values["language"])
            self.settings_saved.emit(self.values)
            bind_text(self.save_button, "Settings saved")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, tr("Invalid settings"), str(exc))
