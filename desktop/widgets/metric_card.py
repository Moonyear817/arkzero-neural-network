from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from desktop.i18n import bind_text, tr


class MetricCard(QFrame):
    def __init__(self, title, value="—", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        label = QLabel(title)
        bind_text(label, title)
        label.setObjectName("metricLabel")
        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("metricValue")
        layout.addWidget(label)
        layout.addWidget(self.value_label)

    def set_value(self, value):
        bind_text(self.value_label, str(value))
