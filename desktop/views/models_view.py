from datetime import UTC, datetime
from math import isfinite

from PySide6.QtCore import QItemSelectionModel, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.i18n import bind_text, language_manager, register_translations, tr
from desktop.widgets.disclosure import Disclosure
from desktop.widgets.policy_table import NumericItem

from .research_translations import button, field_value, label

register_translations(
    {
        "Use this model in battle": "用此模型对战",
        "Run evaluation": "运行评估",
        "Latest evaluation": "最近评估",
        "More": "更多",
        "Model details": "模型详情",
        "Select a model to use in battle or evaluate.": "选择一个模型，用于对战或运行评估。",
        "{count} models": "{count} 个模型",
        "Success {rate}": "通关率 {rate}",
        "Score {score}": "评分 {score}",
        "Evaluation simulations": "评估模拟次数",
    }
)


class ModelsView(QWidget):
    use_in_simulator = Signal(str)

    def __init__(self, controller, parent=None, *, autoload=True):
        super().__init__(parent)
        self.controller = controller
        self._detail_kind = None
        self._detail_data = None
        self._compute_available = True
        self._model_rows = ()
        layout = QVBoxLayout(self)
        title = label("Models")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        toolbar = QHBoxLayout()
        self.use_button = button("Use this model in battle")
        self.refresh_button = button("Refresh")
        self.load_button = button("Load model")
        self.evaluate_button = button("Run evaluation")
        self.compare_button = button("Compare selected 2")
        self.stop_button = button("Stop")
        self.simulations = QSpinBox()
        self.simulations.setRange(1, 1000)
        self.simulations.setValue(16)
        for control_button in (
            self.use_button,
            self.evaluate_button,
            self.stop_button,
        ):
            toolbar.addWidget(control_button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.status = label("Select a model to use in battle or evaluate.")
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 4)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 3)
        self.more = Disclosure("More")
        advanced_toolbar = QHBoxLayout()
        for control_button in (
            self.load_button,
            self.compare_button,
            self.refresh_button,
        ):
            advanced_toolbar.addWidget(control_button)
        advanced_toolbar.addWidget(label("Evaluation simulations"))
        advanced_toolbar.addWidget(self.simulations)
        advanced_toolbar.addStretch()
        self.more.add_layout(advanced_toolbar)
        layout.addWidget(self.more)
        self.details_section = Disclosure("Model details")
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(1000)
        bind_text(
            self.details,
            "Model details, evaluation results and basic comparison appear here.",
            setter="setPlaceholderText",
        )
        self.details_section.add_widget(self.details)
        layout.addWidget(self.details_section)
        self.refresh_button.clicked.connect(controller.refresh)
        self.use_button.clicked.connect(self._use_in_simulator)
        self.load_button.clicked.connect(self._load)
        self.evaluate_button.clicked.connect(self._evaluate)
        self.compare_button.clicked.connect(self._compare)
        self.stop_button.clicked.connect(controller.stop)
        self.table.itemSelectionChanged.connect(self._selection)
        controller.models_changed.connect(self._models)
        controller.model_loaded.connect(self._loaded)
        controller.evaluation_ready.connect(self._evaluated)
        controller.comparison_ready.connect(self._comparison)
        controller.status_changed.connect(self._status)
        language_manager.language_changed.connect(self._retranslate)
        self._status("IDLE")
        self._retranslate()
        if autoload:
            QTimer.singleShot(0, controller.refresh)

    def _paths(self):
        return tuple(
            self.table.item(index.row(), 0).data(Qt.ItemDataRole.UserRole)
            for index in self.table.selectionModel().selectedRows()
            if self.table.item(index.row(), 0) is not None
        )

    def _selection(self):
        paths = self._paths()
        if len(paths) == 1:
            info = self.controller.select(paths[0])
            self._detail_kind, self._detail_data = "selection", info
            self._render_details()
        elif self._detail_kind == "selection":
            self._detail_kind, self._detail_data = None, None
            self.details.clear()
        self._buttons()

    def set_compute_available(self, available):
        """Gate model computation/switching while keeping metadata browsable."""
        self._compute_available = bool(available)
        self._buttons()

    def _buttons(self):
        busy = self.controller.is_busy()
        count = len(self._paths())
        can_use = self._compute_available and not busy and count == 1
        self.refresh_button.setEnabled(not busy)
        self.use_button.setEnabled(can_use)
        self.load_button.setEnabled(can_use)
        self.evaluate_button.setEnabled(can_use)
        self.compare_button.setEnabled(not busy and count == 2)
        self.stop_button.setEnabled(busy)
        self.stop_button.setVisible(busy)

    def _computation_path(self):
        paths = self._paths()
        if (
            self._compute_available
            and not self.controller.is_busy()
            and len(paths) == 1
        ):
            return paths[0]
        return None

    def _use_in_simulator(self):
        path = self._computation_path()
        if path is not None:
            self.use_in_simulator.emit(path)

    def _load(self):
        path = self._computation_path()
        if path is not None:
            self.controller.load_model(path)

    def _evaluate(self):
        path = self._computation_path()
        if path is not None:
            self.controller.evaluate(path, simulations=self.simulations.value())

    def _status(self, status):
        self._buttons()
        if status in ("RUNNING", "STOPPING", "ERROR", "STOPPED"):
            self.status.set_template(
                "{state}{suffix}",
                state=status,
                suffix=" · processing in background" if status == "RUNNING" else "",
            )

    def _models(self, models):
        self._model_rows = tuple(models)
        self._populate_table()
        self.status.set_template("{count} models", count=len(self._model_rows))
        self._selection()

    def _populate_table(self):
        selected = set(self._paths())
        blocker = QSignalBlocker(self.table)
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(self._model_rows))
        for row, model in enumerate(self._model_rows):
            name = QTableWidgetItem(model.name)
            name.setData(Qt.ItemDataRole.UserRole, model.path)
            name.setToolTip(model.path)
            self.table.setItem(row, 0, name)
            self.table.setItem(row, 1, QTableWidgetItem(field_value(model.iteration)))
            self.table.setItem(row, 2, self._evaluation_item(model))
            self.table.setItem(
                row,
                3,
                QTableWidgetItem(
                    " ".join(
                        tr(label)
                        for active, label in (
                            (model.best, "BEST"),
                            (model.latest, "LATEST"),
                        )
                        if active
                    )
                ),
            )
        self.table.setSortingEnabled(True)
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) in selected:
                self.table.selectionModel().select(
                    self.table.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
        blocker.unblock()

    @staticmethod
    def _evaluation_item(model):
        for key in ("success_rate", "evaluation_score"):
            try:
                value = float(model.metric(key))
            except (TypeError, ValueError):
                continue
            if not isfinite(value):
                continue
            item = NumericItem(value)
            item.setText(
                tr("Success {rate}").format(rate=f"{value:.1%}")
                if key == "success_rate"
                else tr("Score {score}").format(score=f"{value:g}")
            )
            return item
        item = NumericItem(None)
        item.setText(tr("UNKNOWN"))
        return item

    def _loaded(self, model):
        self.status.set_template(
            "Loaded {name} · iteration {iteration} · {device}",
            name=model.name,
            iteration=model.iteration,
            device=model.device,
        )
        self._detail_kind, self._detail_data = "loaded", model
        self._render_details()
        self.details_section.set_expanded(True)

    def _evaluated(self, result):
        self.status.set_template(
            "Evaluation complete · seeds {seeds} · {simulations} simulations",
            seeds=result.seeds,
            simulations=result.simulations,
        )
        self._detail_kind, self._detail_data = "evaluated", result
        self._render_details()
        self.details_section.set_expanded(True)

    def _compare(self):
        paths = self._paths()
        if not self.controller.is_busy() and len(paths) == 2:
            self.controller.compare(paths)

    def _comparison(self, models):
        self._detail_kind, self._detail_data = "comparison", models
        self._render_details()
        self.details_section.set_expanded(True)

    def _render_details(self):
        data = self._detail_data
        if data is None:
            return
        if self._detail_kind == "selection":
            lines = [
                data.path,
                f"{tr('Iteration')}: {data.iteration}",
                f"{tr('Created')}: "
                + datetime.fromtimestamp(data.created_time, tz=UTC)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M"),
                f"{tr('Size (MB)')}: {data.size / 1024**2:.2f}",
                data.metadata_note,
            ]
            fields = {
                key: data.metric(key)
                for key in (
                    "evaluation_score",
                    "success_rate",
                    "policy_loss",
                    "value_loss",
                )
            }
            fields.update(dict(data.metrics))
            lines.extend(f"{tr(k)}: {field_value(v)}" for k, v in fields.items())
        elif self._detail_kind == "loaded":
            lines = [
                data.path,
                f"{tr('Parameters')}: {data.parameter_count:,}",
                f"{tr('Initial-state value')}: {data.value:+.4f}",
                tr(
                    "Value is an uncalibrated signed estimate, not a success probability."
                ),
            ]
        elif self._detail_kind == "evaluated":
            lines = [f"{tr(k)}: {field_value(v)}" for k, v in data.metrics]
            lines.extend(
                [
                    "",
                    f"{tr('Saved')}: {data.saved_path}",
                    tr("Evaluation used no root noise and temperature 0."),
                ]
            )
        else:
            fields = [
                "success_rate",
                "average_leaks",
                "average_decisions",
                "value_estimate",
                "policy_divergence",
                "inference_latency",
                "policy_loss",
                "value_loss",
            ]
            lines = [
                f"A: {data[0].path}",
                f"B: {data[1].path}",
                "",
                tr("Saved evaluation/metadata comparison:"),
            ]
            lines.extend(
                f"{tr(field)}:  A {field_value(data[0].metric(field))}    B {field_value(data[1].metric(field))}"
                for field in fields
            )
            lines.extend(
                [
                    "",
                    tr(
                        "UNKNOWN means no measured metadata is available; no inference is fabricated."
                    ),
                ]
            )
        self.details.setPlainText("\n".join(lines))

    def _retranslate(self, *_):
        self.table.setHorizontalHeaderLabels(
            [
                tr(text)
                for text in (
                    "Model",
                    "Iteration",
                    "Latest evaluation",
                    "Flags",
                )
            ]
        )
        self._render_details()
        self._populate_table()
