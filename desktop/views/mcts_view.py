from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.i18n import bind_combo, language_manager, tr
from desktop.widgets.policy_table import PolicyTable

from .research_translations import action_text, button, label


class MCTSView(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.snapshot = None
        self.node_id = 0
        outer = QVBoxLayout(self)
        title = label("MCTS")
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        top = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("Neural PUCT", "puct")
        self.mode.addItem("Plain UCT", "uct")
        bind_combo(self.mode, {"puct": "Neural PUCT", "uct": "Plain UCT"})
        self.stage = QComboBox()
        self.stage.addItem("0-1")
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(12345)
        self.simulations = QSpinBox()
        self.simulations.setRange(1, 1000)
        self.simulations.setValue(16)
        self.run_button = button("Run search")
        self.stop_button = button("Stop")
        for widget in (
            self.mode,
            self.stage,
            label("Seed"),
            self.seed,
            label("Simulations"),
            self.simulations,
            self.run_button,
            self.stop_button,
        ):
            top.addWidget(widget)
        top.addStretch()
        outer.addLayout(top)
        self.model_label = label()
        outer.addWidget(self.model_label)
        self.value_label = label("Network value: unavailable")
        self.value_label.setObjectName("metricValue")
        outer.addWidget(self.value_label)
        self.status = label(
            "Search begins at the first meaningful deployment decision. No preset action sequence."
        )
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        splitter = QSplitter()
        self.policy_table = PolicyTable()
        splitter.addWidget(self.policy_table)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        controls = QHBoxLayout()
        self.back_button = button("Parent")
        self.root_button = button("Root")
        controls.addWidget(self.back_button)
        controls.addWidget(self.root_button)
        right_layout.addLayout(controls)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Action / Node", "N", "Q"])
        right_layout.addWidget(self.tree)
        right_layout.addWidget(
            label(
                "Double-click a child to inspect its next level.\nTop 20 children; at most 2,000 nodes retained."
            )
        )
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)
        outer.addWidget(
            label(
                "Value is a signed estimate (Good / Neutral / Poor), not a calibrated win probability."
            )
        )
        self.run_button.clicked.connect(self._run)
        self.stop_button.clicked.connect(controller.stop)
        self.back_button.clicked.connect(self._back)
        self.root_button.clicked.connect(lambda: self._show_node(0))
        self.tree.itemDoubleClicked.connect(self._child)
        controller.result_ready.connect(self._result)
        controller.model_changed.connect(lambda _: self._status(controller.status))
        controller.status_changed.connect(self._status)
        language_manager.language_changed.connect(self._retranslate)
        self._status("IDLE")
        self._retranslate()

    def _run(self):
        self.controller.search(
            self.stage.currentText(),
            self.seed.value(),
            self.simulations.value(),
            self.mode.currentData(),
        )
        self._status("RUNNING")

    def _status(self, status):
        busy = self.controller.is_busy()
        self.run_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.model_label.set_template(
            "Model: {path}",
            path=self.controller.checkpoint or "None selected — choose one on Models",
        )
        if status in ("RUNNING", "STOPPING", "ERROR", "STOPPED"):
            self.status.set_template(status)

    def _result(self, snapshot):
        self.snapshot = snapshot
        self.policy_table.set_rows(snapshot.rows)
        v = snapshot.network_value
        tendency = (
            "Good"
            if v is not None and v > 0.2
            else "Poor"
            if v is not None and v < -0.2
            else "Neutral"
        )
        if v is None:
            self.value_label.set_template("Network value: unavailable (plain UCT)")
        else:
            self.value_label.set_template(
                "Network value: {value} · {tendency} · {terminal}",
                value=f"{v:+.4f}",
                tendency=tendency,
                terminal="Terminal" if snapshot.terminal else "Non-terminal",
            )
        algorithm = tr(snapshot.algorithm)
        if algorithm.startswith("Neural PUCT · "):
            algorithm = algorithm.replace("Neural PUCT", tr("Neural PUCT")).replace(
                "evaluation, no root noise", tr("evaluation, no root noise")
            )
        self.status.set_template(
            "{algorithm}\nTime {time}s · {simulations} simulations · {nodes} nodes · {wall}s\nSelected: {action}",
            algorithm=algorithm,
            time=f"{snapshot.game_time:.3f}",
            simulations=snapshot.simulations,
            nodes=snapshot.node_count,
            wall=f"{snapshot.wall_time:.3f}",
            action=action_text(snapshot.selected_action),
        )
        self._show_node(0)

    def _show_node(self, node_id):
        if self.snapshot is None or not 0 <= node_id < len(self.snapshot.tree):
            return
        self.node_id = node_id
        node = self.snapshot.tree[node_id]
        self.tree.clear()
        root = QTreeWidgetItem(
            [action_text(node.action), str(node.visits), f"{node.q_value:+.4f}"]
        )
        root.setData(0, Qt.ItemDataRole.UserRole, node.id)
        self.tree.addTopLevelItem(root)
        for index in node.children:
            child = self.snapshot.tree[index]
            row = QTreeWidgetItem(
                [action_text(child.action), str(child.visits), f"{child.q_value:+.4f}"]
            )
            row.setData(0, Qt.ItemDataRole.UserRole, child.id)
            root.addChild(row)
        root.setExpanded(True)
        self.tree.resizeColumnToContents(0)
        self.back_button.setEnabled(node.parent_id is not None)

    def _child(self, item, _column):
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if self.snapshot and self.snapshot.tree[index].children:
            self._show_node(index)

    def _back(self):
        if self.snapshot:
            parent = self.snapshot.tree[self.node_id].parent_id
            if parent is not None:
                self._show_node(parent)

    def _retranslate(self, *_):
        self.tree.setHeaderLabels([tr("Action / Node"), "N", "Q"])
        if self.snapshot is not None:
            selected = self.node_id
            self._result(self.snapshot)
            self._show_node(selected)
        self._status(self.controller.status)
