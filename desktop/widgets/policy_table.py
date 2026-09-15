from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from desktop.i18n import language_manager, tr
from desktop.views.research_translations import action_text


class NumericItem(QTableWidgetItem):
    def __init__(self, value, precision=4):
        super().__init__(
            tr("Unavailable")
            if value is None
            else (str(value) if isinstance(value, int) else f"{value:.{precision}f}")
        )
        self.value = value
        self.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            return (self.value if self.value is not None else float("-inf")) < (
                other.value if other.value is not None else float("-inf")
            )
        return super().__lt__(other)


class PolicyTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(
            ["Action", "Neural P", "MCTS π", "Visits N", "Q Value"]
        )
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(False)
        self.setMinimumWidth(560)
        for column in range(1, 5):
            self.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Fixed
            )
            self.setColumnWidth(column, 78)
        self._rows = ()
        language_manager.language_changed.connect(self._retranslate)
        self._retranslate()

    def set_rows(self, rows):
        self._rows = tuple(rows)
        self.setSortingEnabled(False)
        self.setRowCount(len(rows))
        for index, row in enumerate(rows):
            self.setItem(index, 0, QTableWidgetItem(action_text(row.action)))
            for column, value in enumerate(
                (row.neural_probability, row.mcts_probability, row.visits, row.q_value),
                1,
            ):
                self.setItem(index, column, NumericItem(value))
        self.setSortingEnabled(True)
        self.sortItems(3, Qt.SortOrder.DescendingOrder)

    def _retranslate(self, *_):
        self.setHorizontalHeaderLabels(
            [
                tr(key)
                for key in (
                    "Action",
                    "Neural P",
                    "MCTS π",
                    "Visits N",
                    "Q Value",
                )
            ]
        )
        self.set_rows(self._rows)
