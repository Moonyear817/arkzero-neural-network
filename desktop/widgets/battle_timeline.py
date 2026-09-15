"""Bounded, searchable battle trace presentation with optional auto-scroll."""

from collections import deque

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.i18n import (
    bind_combo,
    bind_text,
    language_manager,
    register_translations,
    tr,
)

register_translations(
    {
        "Search battle events": "搜索战斗事件",
        "Battle event filter": "战斗事件筛选",
        "Battle timeline": "战斗时间轴",
        "Auto-scroll": "自动滚动",
        "ALL": "全部",
        "Spawn": "出生",
        "Movement": "移动与阻挡",
        "Deploy": "部署与撤退",
        "Attack": "攻击",
        "Damage": "伤害",
        "Skill": "技能",
        "Death": "死亡与漏怪",
        "Other": "其他",
        "BATTLE_START": "战斗开始",
        "BATTLE_END": "战斗结束",
        "SPAWN": "敌人出生",
        "MOVING": "开始移动",
        "WAYPOINT": "到达路径点",
        "WAIT": "等待",
        "BLOCK": "建立阻挡",
        "UNBLOCK": "解除阻挡",
        "DEPLOY": "部署",
        "RETREAT": "撤退",
        "ATTACK_START": "攻击开始",
        "HIT": "命中",
        "DAMAGE": "伤害",
        "HEAL": "治疗",
        "DEATH": "死亡",
        "ESCAPE": "漏怪",
        "SKILL_START": "技能开启",
        "SKILL_END": "技能结束",
        "MODIFIER_BOUNDARY": "属性效果切换",
    }
)


class BattleTimeline(QWidget):
    def __init__(self, parent=None, max_lines=10000):
        super().__init__(parent)
        self._events = deque(maxlen=max_lines)
        self.search = QLineEdit()
        bind_text(self.search, "Search battle events", "setPlaceholderText")
        bind_text(self.search, "Search battle events", "setAccessibleName")
        self.category = QComboBox()
        categories = [
            "ALL",
            "Spawn",
            "Movement",
            "Deploy",
            "Attack",
            "Damage",
            "Skill",
            "Death",
            "Other",
        ]
        for category in categories:
            self.category.addItem(category, category)
        bind_combo(self.category, {category: category for category in categories})
        bind_text(self.category, "Battle event filter", "setAccessibleName")
        self.auto_scroll = QCheckBox("Auto-scroll")
        bind_text(self.auto_scroll, "Auto-scroll")
        self.auto_scroll.setChecked(True)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.search, 1)
        toolbar.addWidget(self.category)
        toolbar.addWidget(self.auto_scroll)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(max_lines)
        self.console.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        bind_text(self.console, "Battle timeline", "setAccessibleName")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.console)
        self.search.textChanged.connect(self._refresh)
        self.category.currentTextChanged.connect(self._refresh)
        language_manager.language_changed.connect(self._refresh)

    def _event_text(self, event):
        return f"{event.time:08.3f}  {tr(event.kind)}  {event.detail}"

    def _matches(self, event):
        return (
            self.category.currentData() == "ALL"
            or self.category.currentData() == event.category
        ) and (
            self.search.text().casefold() in event.text.casefold()
            or self.search.text().casefold() in self._event_text(event).casefold()
        )

    def append_events(self, events):
        self._events.extend(events)
        self._refresh()

    def _refresh(self, *_):
        scroll = self.console.verticalScrollBar()
        previous = scroll.value()
        self.console.setPlainText(
            "\n".join(
                self._event_text(event)
                for event in self._events
                if self._matches(event)
            )
        )
        scroll.setValue(scroll.maximum() if self.auto_scroll.isChecked() else previous)

    def clear(self):
        self._events.clear()
        self.console.clear()

    def set_max_lines(self, limit):
        if type(limit) is not int or limit < 1:
            raise ValueError("Timeline line limit must be positive")
        self._events = deque(self._events, maxlen=limit)
        self.console.setMaximumBlockCount(limit)
        self._refresh()
