"""Throttled dashboard; consumes metrics, never runs the engine."""

import math
from typing import ClassVar

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from desktop.i18n import register_translations
from desktop.widgets.disclosure import Disclosure
from desktop.widgets.metric_card import MetricCard
from desktop.widgets.training_chart import TrainingChart

register_translations(
    {
        "More metrics and charts": "更多指标和图表",
        "Research overview": "研究概览",
        "Live engine metrics. Unmeasured values remain blank.": "实时引擎指标。未测量的数值保留为空。",
        "Simulator": "模拟器",
        "Training": "训练",
        "Training engine": "训练引擎",
        "Device": "计算设备",
        "Current model": "当前模型",
        "Current stage": "当前关卡",
        "Iteration": "迭代",
        "Episode": "对局",
        "Last episode result": "最近一局结果",
        "Game time · last episode": "最近一局游戏时间",
        "Generated states": "已生成样本数",
        "Evaluation success rate": "评估通关率",
        "Policy loss": "策略损失",
        "Value loss": "价值损失",
        "Total loss": "总损失",
        "Replay buffer": "回放缓冲区",
        "MCTS simulations": "MCTS 模拟次数",
        "Decision latency": "决策耗时",
        "Simulation / wall time": "模拟时间 / 实际时间",
        "Process memory": "进程内存",
        "Training wall time": "训练实际耗时",
        "Replay buffer size": "回放缓冲区大小",
        "Episode duration · seconds": "对局耗时 · 秒",
        "Decision latency · seconds": "决策耗时 · 秒",
        "waiting for metrics": "等待指标",
        "Iteration / sample": "迭代 / 采样",
        "READY": "就绪",
        "RUNNING": "运行中",
        "IDLE": "空闲",
        "PAUSED": "已暂停",
        "ERROR": "错误",
        "AVAILABLE": "可用",
        "NOT AVAILABLE": "不可用",
        "Checking": "检测中",
        "None": "无",
        "CPU · MPS Available": "CPU · MPS 可用",
        "CPU · MPS Unavailable": "CPU · MPS 不可用",
    }
)


class DashboardView(QWidget):
    CARD_LABELS: ClassVar[dict] = {
        "simulator": "Simulator",
        "training": "Training",
        "engine": "Training engine",
        "device": "Device",
        "model": "Current model",
        "stage": "Current stage",
        "iteration": "Iteration",
        "episode": "Episode",
        "success_rate": "Evaluation success rate",
        "policy_loss": "Policy loss",
        "value_loss": "Value loss",
        "total_loss": "Total loss",
        "replay_buffer": "Replay buffer",
        "mcts_simulations": "MCTS simulations",
        "average_decision_time": "Decision latency",
        "simulator_speed": "Simulation / wall time",
        "rss_mb": "Process memory",
        "training_wall_time": "Training wall time",
        "episode_result": "Last episode result",
        "game_time": "Game time · last episode",
        "generated_states": "Generated states",
    }
    CHARTS: ClassVar[dict] = {
        "success_rate": "Evaluation success rate",
        "policy_loss": "Policy loss",
        "value_loss": "Value loss",
        "total_loss": "Total loss",
        "replay_buffer": "Replay buffer size",
        "episode_duration": "Episode duration · seconds",
        "average_decision_time": "Decision latency · seconds",
    }

    PRIMARY_CARDS = ("iteration", "success_rate", "training_wall_time")

    def __init__(self, parent=None, refresh_ms=1000):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.primary_grid = QGridLayout()
        self.cards = {}
        for index, key in enumerate(self.PRIMARY_CARDS):
            card = MetricCard(self.CARD_LABELS[key])
            self.primary_grid.addWidget(card, 0, index)
            self.cards[key] = card
        layout.addLayout(self.primary_grid)
        self.charts = {key: TrainingChart(label) for key, label in self.CHARTS.items()}
        layout.addWidget(self.charts["success_rate"])
        self.more_metrics = Disclosure("More metrics and charts")
        grid = QGridLayout()
        others = [
            (key, label)
            for key, label in self.CARD_LABELS.items()
            if key not in self.PRIMARY_CARDS
        ]
        for index, (key, label) in enumerate(others):
            card = MetricCard(label)
            grid.addWidget(card, index // 3, index % 3)
            self.cards[key] = card
        self.more_metrics.add_layout(grid)
        charts = QGridLayout()
        for index, key in enumerate(k for k in self.CHARTS if k != "success_rate"):
            charts.addWidget(self.charts[key], index // 2, index % 2)
        self.more_metrics.add_layout(charts)
        layout.addWidget(self.more_metrics)
        self.pending = {}
        self.chart_pending = {}
        self.current_iteration = 0
        self._last_chart_points = {}
        self.timer = QTimer(self)
        self.timer.setInterval(refresh_ms)
        self.timer.timeout.connect(self.flush)
        self.timer.start()

    def update_metrics(self, values):
        values = dict(values)
        for source, target in (
            ("evaluation_success_rate", "success_rate"),
            ("replay_size", "replay_buffer"),
        ):
            if source in values:
                values[target] = values[source]
        self.current_iteration = values.get("iteration", self.current_iteration)
        self.pending.update(values)
        for key in self.CHARTS:
            value = values.get(key)
            if isinstance(value, (float, int)) and math.isfinite(value):
                self.chart_pending[key] = (self.current_iteration, value)

    def set_status(self, **values):
        self.pending.update(values)

    def reset_training_metrics(self):
        """A new task starts with no scores or history borrowed from an old task."""
        preserved = {
            "simulator", "training", "engine", "device", "rss_mb", "simulator_speed"
        }
        for key, card in self.cards.items():
            if key not in preserved:
                card.set_value("—")
                self.pending.pop(key, None)
        self.chart_pending.clear()
        self._last_chart_points.clear()
        self.current_iteration = 0
        for chart in self.charts.values():
            chart.series.clear()

    def flush(self):
        for key, value in self.pending.items():
            if key not in self.cards:
                continue
            if value is None:
                value = "—"
            elif key == "success_rate":
                value = f"{value:.1%}" if isinstance(value, (int, float)) else value
            elif key in ("average_decision_time", "training_wall_time") and isinstance(
                value, (int, float)
            ):
                value = (
                    f"{value * 1000:.1f} ms"
                    if key == "average_decision_time"
                    else f"{value:.1f} s"
                )
            elif key == "rss_mb" and isinstance(value, (int, float)):
                value = f"{value:.0f} MB"
            elif isinstance(value, float):
                value = f"{value:.4f}"
            self.cards[key].set_value(value)
        for key, (x, y) in self.chart_pending.items():
            if self._last_chart_points.get(key) != (x, y):
                self.charts[key].add_value(x, y)
                self._last_chart_points[key] = (x, y)
        self.pending.clear()
        self.chart_pending.clear()
