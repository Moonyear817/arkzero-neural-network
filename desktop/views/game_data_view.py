from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QHBoxLayout,
    QHeaderView,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.i18n import bind_combo, language_manager, register_translations, tr
from desktop.widgets.stage_map import StageMap

from .research_translations import button, field_value, label

register_translations({
    "{count} records · {directory}": "{count} 条资料 · {directory}",
    "Branch": "分支", "Rarity": "星级", "All skills": "全部技能",
    "Support": "模拟支持范围", "PRTS": "PRTS 来源",
    "Mechanics": "战斗机制", "Summons": "召唤物",
})


class GameDataView(QWidget):
    def __init__(self, controller, parent=None, *, autoload=True):
        super().__init__(parent)
        self.controller = controller
        self.records = ()
        self.map_catalog = None
        outer = QVBoxLayout(self)
        title = label("Game Data")
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        top = QHBoxLayout()
        self.category = QComboBox()
        for category in ("Stages", "Operators", "Enemies", "Skills", "Summons", "Mechanics"):
            self.category.addItem(category, category)
        bind_combo(
            self.category,
            {key: key for key in ("Stages", "Operators", "Enemies", "Skills", "Summons", "Mechanics")},
        )
        self.refresh_button = button("Refresh data")
        self.status = label("Reading supported GameData…")
        top.addWidget(self.category)
        top.addWidget(self.refresh_button)
        top.addWidget(self.status, 1)
        outer.addLayout(top)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索干员中文名、英文名、编号或职业")
        self.search.textChanged.connect(self._filter)
        outer.addWidget(self.search)
        splitter = QSplitter()
        self.entries = QListWidget()
        splitter.addWidget(self.entries)
        right = QWidget()
        details = QVBoxLayout(right)
        self.fields = QTableWidget(0, 2)
        self.fields.setHorizontalHeaderLabels(["Field", "Value"])
        self.fields.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.fields.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        details.addWidget(self.fields)
        self.detail_mode = QComboBox()
        self.detail_mode.addItem("干员资料与技能说明", "Operator details")
        self.detail_mode.addItem("PRTS 页面原文（完整存档）", "Full wiki text")
        self.detail_mode.addItem("原始游戏数据", "GameData")
        self.detail_mode.addItem("原始技能数据", "All skills")
        self.detail_mode.currentIndexChanged.connect(lambda: self._select(self.entries.currentItem(), None))
        details.addWidget(self.detail_mode)
        self.source_text = QPlainTextEdit()
        self.source_text.setReadOnly(True)
        self.source_text.setPlaceholderText("完整 PRTS 资料与游戏数据")
        details.addWidget(self.source_text, 1)
        self.stage_map = StageMap()
        details.addWidget(self.stage_map, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter, 1)
        outer.addWidget(
            label(
                "PRTS 全员资料已收录；基础近似模拟不代表完整还原技能、天赋和模组。"
            )
        )
        self.category.currentTextChanged.connect(self._filter)
        self.entries.currentItemChanged.connect(self._select)
        self.refresh_button.clicked.connect(controller.load)
        controller.loaded.connect(self._loaded)
        controller.status_changed.connect(self._status_changed)
        language_manager.language_changed.connect(self._retranslate)
        self._retranslate()
        if autoload:
            QTimer.singleShot(0, controller.load)

    def _status_changed(self, status):
        self.refresh_button.setEnabled(not self.controller.is_busy())
        if status == "ERROR":
            self.status.set_template("Data load failed; see Logs")

    def _loaded(self, snapshot):
        self.records = snapshot.records
        from arknights_sim.data.map_catalog import MapCatalog
        self.map_catalog = MapCatalog(snapshot.directory)
        self.status.set_template(
            "{count} records · {directory}",
            count=len(self.records),
            directory=snapshot.directory,
        )
        self._filter()

    def _filter(self, *_):
        self.entries.clear()
        for record in self.records:
            query = self.search.text().strip().casefold()
            searchable = f"{record.name} {record.id} " + " ".join(v for k, v in record.fields if k not in ("Full wiki text", "GameData", "All skills", "Operator details"))
            if record.category == self.category.currentData() and query in searchable.casefold():
                item = QListWidgetItem(f"{record.name}  ·  {record.id}")
                item.setData(Qt.ItemDataRole.UserRole, record)
                self.entries.addItem(item)
        if self.entries.count():
            self.entries.setCurrentRow(0)
        else:
            self.fields.setRowCount(0)
            self.source_text.clear()
            self.stage_map.hide()

    def _select(self, current, _previous):
        if current is None:
            return
        record = current.data(Qt.ItemDataRole.UserRole)
        long_fields = {"Wiki details", "Full wiki text", "GameData", "All skills", "Operator details", "Skill details", "Summon details"}
        fields = [(k, v) for k, v in record.fields if k not in long_fields]
        self.source_text.setVisible(record.category == "Operators")
        self.detail_mode.setVisible(record.category == "Operators")
        self.source_text.setPlainText(dict(record.fields).get(self.detail_mode.currentData(), ""))
        if record.category == 'Summons':
            self.source_text.show()
            self.source_text.setPlainText(dict(record.fields).get('Summon details', 'UNKNOWN'))
        if record.category == 'Skills':
            self.source_text.show()
            self.source_text.setPlainText(dict(record.fields).get('Skill details', 'UNKNOWN'))
        if record.category=='Mechanics':
            self.source_text.show()
            self.source_text.setPlainText(record.name+'\n\n'+'\n\n'.join(f'{k}：{v}' for k,v in record.fields))
        self.fields.setRowCount(len(fields))
        for row, (key, value) in enumerate(fields):
            self.fields.setItem(row, 0, QTableWidgetItem(tr(key)))
            self.fields.setItem(row, 1, QTableWidgetItem(field_value(value)))
        self.stage_map.setVisible(record.stage_map is not None)
        if record.stage_map is not None:
            self.stage_map.set_snapshot(record.stage_map)
        if self.map_catalog is not None and record.id in self.map_catalog.records:
            from desktop.widgets.map_picker import preview, describe
            try:
                entry, data = self.map_catalog.inspect(record.id)
                self.stage_map.show()
                self.stage_map.set_snapshot(preview(data, record.id))
                self.source_text.show()
                self.source_text.setPlainText(describe(entry, data))
            except Exception as exc:
                self.source_text.show()
                self.source_text.setPlainText("地图读取失败：" + str(exc))

    def _retranslate(self, *_):
        self.fields.setHorizontalHeaderLabels([tr("Field"), tr("Value")])
        self._select(self.entries.currentItem(), None)
