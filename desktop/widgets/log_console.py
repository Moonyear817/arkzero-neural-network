from collections import deque
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QPushButton, QPlainTextEdit, QFileDialog, QApplication
from desktop.i18n import bind_text, bind_combo, tr, register_translations

register_translations({"ALL": "全部", "APP": "应用", "SIM": "模拟", "TRAIN": "训练", "ERROR": "错误",
    "Search logs": "搜索日志", "Copy selected": "复制选中内容", "Save visible log…": "保存当前日志…",
    "Save log": "保存日志", "Cannot save log": "无法保存日志"})


class LogConsole(QWidget):
    def __init__(self, maximum=10000, parent=None):
        super().__init__(parent)
        self.entries = deque(maxlen=maximum)
        layout = QVBoxLayout(self)
        tools = QHBoxLayout()
        self.category = QComboBox()
        categories = ["ALL", "APP", "SIM", "TRAIN", "MCTS", "ERROR"]
        for category in categories:
            self.category.addItem(category, category)
        bind_combo(self.category, {key: key for key in categories})
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search logs")
        bind_text(self.search, "Search logs", "setPlaceholderText")
        copy = QPushButton("Copy selected")
        save = QPushButton("Save visible log…")
        bind_text(copy, "Copy selected")
        bind_text(save, "Save visible log…")
        for widget in (self.category, self.search, copy, save):
            tools.addWidget(widget)
        layout.addLayout(tools)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(maximum)
        layout.addWidget(self.text)
        self.category.currentTextChanged.connect(self.refresh)
        self.search.textChanged.connect(self.refresh)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.text.textCursor().selectedText()))
        save.clicked.connect(self.save_visible)

    def matches(self, entry):
        return (self.category.currentData() in ("ALL", entry.category)
                and self.search.text().casefold() in entry.text.casefold())

    def add_entries(self, entries):
        self.entries.extend(entries)
        visible = [f"[{r.category}] {r.text}" for r in entries if self.matches(r)]
        if visible:
            self.text.appendPlainText("\n".join(visible))

    def set_maximum(self, maximum):
        self.entries = deque(self.entries, maxlen=maximum)
        self.text.setMaximumBlockCount(maximum)
        self.refresh()

    def refresh(self):
        self.text.setPlainText("\n".join(f"[{r.category}] {r.text}" for r in self.entries if self.matches(r)))

    def save_visible(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("Save log"), "arknights-zero.log", "Log (*.log)")
        if path:
            try:
                from pathlib import Path
                Path(path).write_text(self.text.toPlainText(), encoding="utf-8")
            except OSError as exc:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, tr("Cannot save log"), str(exc))
