"""Configuration and controls; all computation is delegated to the controller."""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import ClassVar

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
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
from desktop.services.config_service import ConfigService
from desktop.widgets.disclosure import Disclosure
from desktop.widgets.training_chart import TrainingChart

def _episode_result_source(payload):
    if payload.get("truncated") or payload.get("termination") in ("time_limit", "decision_limit"):
        return "Truncated · excluded from training"
    if payload.get("win") or payload.get("simulator_result") == "WIN":
        return "Clear with leaks · {kills} kills / {leaks} leaks"
    return "Failure · {kills} kills / {leaks} leaks"


register_translations(
    {
        "Training": "训练",
        "AlphaZeroTrainer · AVAILABLE": "AlphaZero 训练引擎 · 可用",
        "Training Engine: NOT AVAILABLE": "训练引擎：不可用",
        "Training configuration": "训练配置",
        "Metrics refresh at iteration/episode boundaries. Values are uncalibrated estimates.": "指标在每轮或每局结束后刷新，Value 为未经概率校准的估计值。",
        "Load config": "加载配置",
        "Save session config": "保存本次配置",
        "Stage": "关卡",
        "Squad": "干员队伍",
        "Device": "计算设备",
        "Noir Corne": "黑角",
        "Melantha": "玫兰莎",
        "Iterations": "总训练轮数",
        "MCTS simulations": "MCTS 模拟次数",
        "Episodes / iteration": "每轮对局数",
        "Batch size": "批量大小",
        "Replay buffer capacity": "经验池容量",
        "Updates / iteration": "每轮参数更新次数",
        "Evaluation episodes": "评估对局数",
        "Seed": "随机种子",
        "Learning rate": "学习率",
        "c_puct": "探索系数 c_puct",
        "Dirichlet alpha": "Dirichlet α",
        "Dirichlet epsilon": "Dirichlet ε",
        "Temperature": "采样温度",
        "START TRAINING": "开始训练",
        "PAUSE": "暂停",
        "RESUME": "继续",
        "STOP": "停止",
        "SAVE CHECKPOINT": "保存检查点",
        "RUN EVALUATION": "运行评估",
        "Live training metrics": "实时训练指标",
        "Iteration": "训练轮次",
        "Episode": "对局进度",
        "Current stage": "当前关卡",
        "Last episode result": "最近一局结果",
        "Game time · last episode": "最近一局游戏时间",
        "Generated states": "已生成样本数",
        "Replay buffer": "经验池样本数",
        "Network checkpoint": "网络检查点",
        "Policy loss": "策略损失",
        "Value loss": "价值损失",
        "Total loss": "总损失",
        "Evaluation success rate": "评估通关率",
        "Loaded {name}. Changes save to gui_session.yaml.": "已加载 {name}，修改将保存到 gui_session.yaml。",
        "Saved {path}": "已保存：{path}",
        "Running": "进行中",
        "Zero-leak clear": "零漏怪通关",
        "Clear with leaks · {kills} kills / {leaks} leaks": "通关（有漏怪）· 击杀 {kills} / 漏怪 {leaks}",
        "Truncated · excluded from training": "达到训练上限 · 本局不参与学习",
        "Reward verification · autonomous squad": "奖励机制验证 · 自主编队",
        "Failure · {kills} kills / {leaks} leaks": "未通关 · 击杀 {kills} / 漏怪 {leaks}",
        "Desktop defaults loaded. Each training run saves to its own dated folder; published models are preserved.": "已加载默认设置。每次训练保存到独立的日期目录，保留已有模型。",
        "Evaluation pending until the next completed iteration. Resume if paused.": "评估已排队，将在下一轮完成后执行；已暂停时请先继续。",
        "Evaluation request pending until the next completed iteration. Resume if paused.": "评估请求已排队，将在下一轮完成后执行；已暂停时请先继续。",
        "Checkpoint request pending until the next completed iteration. Resume if paused; STOP also saves completed progress.": "检查点保存已排队，将在下一轮完成后执行；已暂停时请先继续。停止也会保存已完成的进度。",
        "Checkpoint request pending until the next completed iteration. Resume if paused. Stop also saves completed progress.": "检查点保存已排队，将在下一轮完成后执行；已暂停时请先继续。停止也会保存已完成的进度。",
        "Pausing at the next safe decision/update boundary.": "将在下一个安全的决策或参数更新边界暂停。",
        "Stopping safely. Completed progress is saved; an unfinished episode is discarded.": "正在安全停止：保存已完成进度，丢弃尚未结束对局的样本。",
    }
)


register_translations(
    {
        "Choose a stage, squad, starting point and experiment size.": "选择关卡、编队方式、训练起点和实验规模。",
        "Local stage library…": "本地地图库…",
        "AI selects squad": "AI 自主编队",
        "Fixed squad": "固定编队",
        "Starting point": "训练起点",
        "Start from scratch": "从头训练",
        "Continue a checkpoint": "继续已有训练",
        "Loaded YAML starting point": "使用已加载 YAML 的训练起点",
        "Choose a resumable checkpoint": "选择可恢复训练的检查点",
        "Browse…": "选择文件…",
        "Experiment size": "实验规模",
        "Quick check": "快速验证",
        "Small experiment": "小型实验",
        "Longer experiment": "较长实验",
        "Custom configuration": "自定义配置",
        "Advanced configuration and YAML": "高级配置与 YAML",
        "More actions": "更多操作",
        "More training metrics": "更多训练指标",
        "Training wall time": "训练耗时",
        "Stop and save": "停止并保存",
        "Training in this app": "本应用中的训练",
        "Last background task": "上次后台任务",
        "Background training": "后台训练",
        "Stage: {stage} · Squad: {squad} · Starting point: {starting}": "关卡：{stage} · 编队：{squad} · 起点：{starting}",
        "Fixed squad · {count} operators": "固定编队 · {count} 人",
        "Checkpoint saved: {name}": "已保存检查点：{name}",
        "Evaluation completed.": "评估已完成。",
        "Background task active": "后台任务进行中",
        "Choose operators… ({count} selected)": "选择干员…（已选 {count} 人）",
        "{iterations} iterations · {episodes} episodes/iteration · {simulations} simulations · {steps} updates/iteration": "{iterations} 轮 · 每轮 {episodes} 局 · {simulations} 次模拟 · 每轮 {steps} 次更新",
        "Resume requires a compatible full training checkpoint. Core validates it; best.pt is inference-only.": "继续训练需要兼容的完整训练检查点，最终由引擎校验；best.pt 仅用于推理，不能恢复训练。",
        "Saved model and training settings are restored by Core. Iterations is the total target, not extra iterations.": "引擎将恢复已保存的模型与训练设置；轮数表示总目标，不是额外增加的轮数。",
        "Loaded YAML uses warm-start weights. Selecting a starting point explicitly replaces that setting.": "已加载的 YAML 使用预训练权重初始化。明确选择训练起点后将替换该设置。",
        "Choose a resumable checkpoint before continuing.": "请先选择可恢复训练的检查点。",
        "The selected checkpoint file does not exist.": "所选检查点文件不存在。",
        "Loaded configuration uses a stage pool: {stages}": "已加载配置使用多个关卡：{stages}",
        "This background task supports stop and save. Pause and resume are unavailable.": "此后台任务支持停止并保存，暂不支持暂停和继续。",
        "Last background task · {status} · {count} iterations completed": "上次后台任务 · {status} · 已完成 {count} 轮",
        "COMPLETE": "已完成",
        "STOPPED": "已停止并保存",
        "FAILED": "失败",
        "INTERRUPTED": "进程已中断",
    }
)


class TrainingView(QWidget):
    PRESETS: ClassVar[dict] = {
        "quick": {
            "iterations": 1,
            "episodes_per_iteration": 2,
            "mcts_simulations": 16,
            "training_steps_per_iteration": 2,
            "evaluation_episodes": 1,
        },
        "small": {
            "iterations": 5,
            "episodes_per_iteration": 5,
            "mcts_simulations": 32,
            "training_steps_per_iteration": 10,
            "evaluation_episodes": 3,
        },
        "extended": {
            "iterations": 20,
            "episodes_per_iteration": 5,
            "mcts_simulations": 64,
            "training_steps_per_iteration": 20,
            "evaluation_episodes": 5,
        },
    }

    def __init__(self, controller, parent=None, config_service=None):
        super().__init__(parent)
        self.controller = controller
        self.config_service = (
            config_service or controller.config_service or ConfigService()
        )
        self._compute_available = True
        self._loaded, self._settings, self._background = {}, {}, {}
        self._managed_paths = True
        self._loading = False
        self._foreground_seen = False
        self._dashboard = None
        self._background_stop_requested = False
        self._metric_task = None
        self._task_notice_kind = None
        self.fields, self.metrics = {}, {}
        self.root_layout = QVBoxLayout(self)
        title = bind_text(QLabel(), "Training")
        title.setObjectName("pageTitle")
        self.root_layout.addWidget(title)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        self.content_layout = QVBoxLayout(body)
        self.root_layout.addWidget(scroll)
        layout = self.content_layout
        self.task_source = QLabel()
        self.state_label = bind_text(QLabel(), "IDLE")
        self.state_label.setObjectName("training_state")
        status_row = QHBoxLayout()
        status_row.addWidget(self.task_source)
        status_row.addWidget(self.state_label)
        status_row.addStretch()
        layout.addLayout(status_row)
        self.background_notice = QLabel()
        self.background_notice.setWordWrap(True)
        self.background_notice.hide()
        layout.addWidget(self.background_notice)
        self.task_summary = QLabel()
        self.task_summary.setWordWrap(True)
        self.task_summary.hide()
        layout.addWidget(self.task_summary)
        self.task_notice = QLabel()
        self.task_notice.setWordWrap(True)
        self.task_notice.hide()
        layout.addWidget(self.task_notice)
        self.notice = bind_text(
            QLabel(), "Choose a stage, squad, starting point and experiment size."
        )
        self.notice.setWordWrap(True)

        self.config_group = bind_text(QGroupBox(), "Training configuration", "setTitle")
        form = QFormLayout(self.config_group)
        stage_row = QHBoxLayout()
        stage = QComboBox()
        stage.addItem("0-1", "0-1")
        self.fields["stage"] = stage
        stage_row.addWidget(stage, 1)
        self.choose_map = bind_text(QPushButton(), "Local stage library…")
        self.choose_map.clicked.connect(self._choose_training_map)
        stage_row.addWidget(self.choose_map)
        form.addRow(bind_text(QLabel(), "Stage"), stage_row)
        self.stage_pool_notice = QLabel()
        self.stage_pool_notice.setWordWrap(True)
        form.addRow(self.stage_pool_notice)

        squad_row = QHBoxLayout()
        self.squad_mode = QComboBox()
        self.squad_mode.addItem("AI selects squad", "auto")
        self.squad_mode.addItem("Fixed squad", "fixed")
        bind_combo(
            self.squad_mode, {"auto": "AI selects squad", "fixed": "Fixed squad"}
        )
        squad_row.addWidget(self.squad_mode)
        self.selected_squad = ("char_500_noirc", "char_208_melan")
        self.pick_squad = QPushButton()
        self.pick_squad.clicked.connect(self._pick_squad)
        squad_row.addWidget(self.pick_squad, 1)
        self.auto_squad = QCheckBox(self)
        self.auto_squad.hide()
        self.squad_mode.currentIndexChanged.connect(self._squad_mode_changed)
        self.auto_squad.toggled.connect(self._auto_squad_changed)
        form.addRow(bind_text(QLabel(), "Squad"), squad_row)

        start_row = QHBoxLayout()
        self.resume_selector = QComboBox()
        self.resume_selector.addItem("Start from scratch", "fresh")
        self.resume_selector.addItem("Continue a checkpoint", "resume")
        bind_combo(
            self.resume_selector,
            {"fresh": "Start from scratch", "resume": "Continue a checkpoint"},
        )
        start_row.addWidget(self.resume_selector)
        self.resume_path = QLineEdit()
        bind_text(
            self.resume_path, "Choose a resumable checkpoint", "setPlaceholderText"
        )
        start_row.addWidget(self.resume_path, 1)
        self.resume_browse = bind_text(QPushButton(), "Browse…")
        self.resume_browse.clicked.connect(self._choose_resume_checkpoint)
        start_row.addWidget(self.resume_browse)
        self.resume_selector.currentIndexChanged.connect(self._resume_changed)
        form.addRow(bind_text(QLabel(), "Starting point"), start_row)
        self.resume_note = QLabel()
        self.resume_note.setWordWrap(True)
        form.addRow(self.resume_note)
        self.size_selector = QComboBox()
        for key, name in (
            ("quick", "Quick check"),
            ("small", "Small experiment"),
            ("extended", "Longer experiment"),
            ("custom", "Custom configuration"),
        ):
            self.size_selector.addItem(name, key)
        bind_combo(
            self.size_selector,
            {
                "quick": "Quick check",
                "small": "Small experiment",
                "extended": "Longer experiment",
                "custom": "Custom configuration",
            },
        )
        self.size_selector.currentIndexChanged.connect(self._size_changed)
        form.addRow(bind_text(QLabel(), "Experiment size"), self.size_selector)
        self.size_summary = QLabel()
        self.size_summary.setWordWrap(True)
        form.addRow(self.size_summary)
        layout.addWidget(self.config_group)

        controls = QHBoxLayout()
        self.start_button = bind_text(QPushButton(), "START TRAINING")
        self.start_button.setObjectName("training_start")
        self.start_button.clicked.connect(self._primary_action)
        self.stop_button = bind_text(QPushButton(), "Stop and save")
        self.stop_button.setObjectName("training_stop")
        self.stop_button.clicked.connect(self._stop_task)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch()
        layout.addLayout(controls)
        # Compatibility handles remain callable, but only the primary action is visible.
        self.pause_button = QPushButton(self)
        self.pause_button.clicked.connect(controller.pause)
        self.pause_button.hide()
        self.resume_button = QPushButton(self)
        self.resume_button.clicked.connect(controller.resume)
        self.resume_button.hide()
        self.background_stop = QPushButton(self)
        self.background_stop.clicked.connect(self._stop_task)
        self.background_stop.hide()

        self.summary_widget = QWidget()
        summary = QVBoxLayout(self.summary_widget)
        summary_row = QGridLayout()
        for index, (key, label) in enumerate(
            (
                ("iteration", "Iteration"),
                ("success_rate", "Evaluation success rate"),
                ("training_wall_time", "Training wall time"),
            )
        ):
            summary_row.addWidget(bind_text(QLabel(), label), 0, index)
            self.metrics[key] = QLabel("—")
            summary_row.addWidget(self.metrics[key], 1, index)
        summary.addLayout(summary_row)
        self.summary_chart = TrainingChart("Evaluation success rate")
        summary.addWidget(self.summary_chart)
        layout.addWidget(self.summary_widget)
        self.dashboard_slot = QVBoxLayout()
        layout.addLayout(self.dashboard_slot)

        self.advanced = Disclosure("Advanced configuration and YAML")
        self.advanced.add_widget(self.notice)
        from desktop.widgets import progression_dialog  # registers shared translations
        self.progression_button = bind_text(QPushButton(), 'Operator progression')
        self.progression_button.clicked.connect(self._edit_progression)
        self.advanced.add_widget(self.progression_button)
        from desktop.widgets import account_dialog
        self.account_button = bind_text(QPushButton(), 'Account review…')
        self.account_button.clicked.connect(self._edit_account)
        self.advanced.add_widget(self.account_button)
        self.engine_label = bind_text(
            QLabel(),
            "AlphaZeroTrainer · AVAILABLE"
            if controller.availability
            else "Training Engine: NOT AVAILABLE",
        )
        self.advanced.add_widget(self.engine_label)
        file_bar = QHBoxLayout()
        self.config_selector = QComboBox()
        self.config_selector.setObjectName("training_config")
        for path in self.config_service.list_configs():
            self.config_selector.addItem(tr("Reward verification · autonomous squad") if path.name == "reward_verification.yaml" else path.name, str(path))
        file_bar.addWidget(self.config_selector, 1)
        load = bind_text(QPushButton(), "Load config")
        load.clicked.connect(
            lambda: self.load_config(self.config_selector.currentData())
        )
        file_bar.addWidget(load)
        save = bind_text(QPushButton(), "Save session config")
        save.clicked.connect(self.save_config)
        file_bar.addWidget(save)
        self.advanced.add_layout(file_bar)
        columns = QHBoxLayout()
        first, second = QFormLayout(), QFormLayout()
        columns.addLayout(first)
        columns.addLayout(second)
        self.advanced.add_layout(columns)
        device = QComboBox()
        for label, key in (("Auto", "auto"), ("CPU", "cpu"), ("MPS", "mps")):
            device.addItem(label, key)
        bind_combo(device, {"auto": "Auto", "cpu": "CPU", "mps": "MPS"})
        self.fields["device"] = device
        first.addRow(bind_text(QLabel(), "Device"), device)
        for key, label, maximum, target in (
            ("iterations", "Iterations", 100000, first),
            ("mcts_simulations", "MCTS simulations", 10000, first),
            ("episodes_per_iteration", "Episodes / iteration", 10000, first),
            ("batch_size", "Batch size", 4096, first),
            ("replay_buffer_size", "Replay buffer capacity", 1000000, first),
            ("training_steps_per_iteration", "Updates / iteration", 100000, second),
            ("evaluation_episodes", "Evaluation episodes", 10000, second),
            ("seed", "Seed", 2147483647, second),
        ):
            field = QSpinBox()
            field.setRange(0 if key == "seed" else 1, maximum)
            field.valueChanged.connect(self._advanced_changed)
            self.fields[key] = field
            target.addRow(bind_text(QLabel(), label), field)
        for key, label, maximum, decimals in (
            ("learning_rate", "Learning rate", 1.0, 6),
            ("c_puct", "c_puct", 100.0, 3),
            ("dirichlet_alpha", "Dirichlet alpha", 10.0, 3),
            ("dirichlet_epsilon", "Dirichlet epsilon", 1.0, 3),
            ("temperature", "Temperature", 10.0, 3),
        ):
            field = QDoubleSpinBox()
            field.setDecimals(decimals)
            field.setRange(
                0.000001
                if key == "learning_rate"
                else 0.001
                if key == "dirichlet_alpha"
                else 0,
                maximum,
            )
            field.setSingleStep(0.0001 if key == "learning_rate" else 0.05)
            self.fields[key] = field
            second.addRow(bind_text(QLabel(), label), field)
        for key, field in self.fields.items():
            field.setObjectName(key)
        layout.addWidget(self.advanced)

        self.more_actions = Disclosure("More actions")
        more = QHBoxLayout()
        self.save_checkpoint_button = bind_text(QPushButton(), "SAVE CHECKPOINT")
        self.save_checkpoint_button.setObjectName("training_save_checkpoint")
        self.save_checkpoint_button.clicked.connect(self._save_checkpoint)
        self.evaluate_button = bind_text(QPushButton(), "RUN EVALUATION")
        self.evaluate_button.setObjectName("training_evaluate")
        self.evaluate_button.clicked.connect(self._evaluate)
        more.addWidget(self.save_checkpoint_button)
        more.addWidget(self.evaluate_button)
        more.addStretch()
        self.more_actions.add_layout(more)
        layout.addWidget(self.more_actions)
        self.more_metrics = Disclosure("More training metrics")
        live = QGridLayout()
        for index, (key, label) in enumerate(
            (
                ("episode", "Episode"),
                ("stage", "Current stage"),
                ("episode_result", "Last episode result"),
                ("game_time", "Game time · last episode"),
                ("generated_states", "Generated states"),
                ("replay_buffer", "Replay buffer"),
                ("model", "Network checkpoint"),
                ("mcts_simulations", "MCTS simulations"),
                ("device", "Device"),
                ("policy_loss", "Policy loss"),
                ("value_loss", "Value loss"),
                ("total_loss", "Total loss"),
            )
        ):
            row, col = divmod(index, 2)
            value = QLabel("—")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.metrics[key] = value
            live.addWidget(bind_text(QLabel(), label), row, col * 2)
            live.addWidget(value, row, col * 2 + 1)
        self.more_metrics.add_layout(live)
        layout.addWidget(self.more_metrics)
        layout.addStretch()
        controller.state_changed.connect(self._refresh_controls)
        controller.metrics_updated.connect(self._metrics_updated)
        controller.event_received.connect(self._event)
        controller.log.connect(self._message)
        self.load_config()
        self._refresh_controls(controller.state)
        self.background_timer = QTimer(self)
        self.background_timer.setInterval(2000)
        self.background_timer.timeout.connect(self._background_status)
        self.background_timer.start()
        self._background_status()
        language_manager.language_changed.connect(self._update_task_summary)

    def set_dashboard(self, widget):
        if self._dashboard is widget:
            return
        if self._dashboard is not None:
            self.dashboard_slot.removeWidget(self._dashboard)
        self._dashboard = widget
        self.dashboard_slot.addWidget(widget)
        self.summary_widget.hide()
        self.more_metrics.hide()
        if self._background:
            self._background_status()

    def _reset_task_metrics(self, task):
        self._metric_task = task
        for value in self.metrics.values():
            bind_text(value, "—")
        self.summary_chart.series.clear()
        self._last_summary_point = None
        if self._dashboard is not None:
            self._dashboard.reset_training_metrics()

    def _squad_mode_changed(self):
        self.auto_squad.setChecked(self.squad_mode.currentData() == "auto")
        self.pick_squad.setEnabled(self.squad_mode.currentData() == "fixed")

    def _auto_squad_changed(self, value):
        index = self.squad_mode.findData("auto" if value else "fixed")
        if self.squad_mode.currentIndex() != index:
            self.squad_mode.setCurrentIndex(index)
        self.pick_squad.setEnabled(not value)

    def _resume_changed(self):
        if not self._loading and self.resume_selector.currentData() != "configured":
            self._loaded.pop("warm_start", None)
        resume = self.resume_selector.currentData() == "resume"
        self.resume_path.setVisible(resume)
        self.resume_browse.setVisible(resume)
        self.resume_note.setVisible(resume or bool(self._loaded.get("warm_start")))
        bind_text(
            self.resume_note,
            "Resume requires a compatible full training checkpoint. Core validates it; best.pt is inference-only.",
        )
        if not resume and self._loaded.get("warm_start"):
            bind_text(
                self.resume_note,
                "Loaded YAML uses warm-start weights. Selecting a starting point explicitly replaces that setting.",
            )

    def _update_task_summary(self, _language=None):
        if self._background.get("active"):
            background = self._background
            stage = background.get("last_episode", {}).get("stage", tr("UNKNOWN"))
            names = background.get("selected_names") or background.get("selected_squad")
            squad = ", ".join(names) if names else tr("UNKNOWN")
            starting = tr("UNKNOWN")
        else:
            stages = self._loaded.get("stage_pool") or [
                self.fields["stage"].currentData()
            ]
            stage = ", ".join(stages)
            squad = (
                tr("AI selects squad")
                if self.auto_squad.isChecked()
                else tr("Fixed squad · {count} operators").format(
                    count=len(self.selected_squad)
                )
            )
            starting = self.resume_selector.currentText()
            if self.resume_selector.currentData() == "resume":
                starting += " · " + Path(self.resume_path.text()).name
        self.task_summary.setText(
            tr("Stage: {stage} · Squad: {squad} · Starting point: {starting}").format(
                stage=stage, squad=squad, starting=starting
            )
        )

    def _task_message(self, message, kind="request", **values):
        self._task_notice_kind = kind
        bind_text(self.task_notice, message, **values)
        self.task_notice.show()
        bind_text(self.notice, message, **values)

    def _choose_resume_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Continue a checkpoint"),
            str(
                self._settings.get(
                    "checkpoint_dir", self.config_service.root / "checkpoints"
                )
            ),
            "PyTorch checkpoint (*.pt *.pth)",
        )
        if path:
            self.resume_path.setText(path)
            self.resume_selector.setCurrentIndex(
                self.resume_selector.findData("resume")
            )

    def _size_changed(self):
        if self._loading:
            return
        preset = self.PRESETS.get(self.size_selector.currentData())
        if preset:
            self._loading = True
            for key, value in preset.items():
                self.fields[key].setValue(value)
            self._loading = False
        self._update_size_summary()
        if self.size_selector.currentData() == "custom":
            self.advanced.set_expanded(True)

    def _advanced_changed(self):
        if self._loading:
            return
        self.size_selector.setCurrentIndex(self.size_selector.findData("custom"))
        self._update_size_summary()

    def _update_size_summary(self):
        bind_text(
            self.size_summary,
            "{iterations} iterations · {episodes} episodes/iteration · {simulations} simulations · {steps} updates/iteration",
            iterations=self.fields["iterations"].value(),
            episodes=self.fields["episodes_per_iteration"].value(),
            simulations=self.fields["mcts_simulations"].value(),
            steps=self.fields["training_steps_per_iteration"].value(),
        )

    def _primary_action(self):
        if self._background.get("active"):
            return
        if self.controller.state in ("PAUSING", "PAUSED"):
            self.controller.resume()
        elif self.controller.state in ("STARTING", "RUNNING"):
            self.controller.pause()
        elif not self.controller.is_busy:
            self._start()

    def _stop_task(self):
        if self._background.get("active"):
            from desktop.services.background_training import request_background_stop

            request_background_stop()
            self._background_stop_requested = True
            self._refresh_controls(self.controller.state)
            self._task_message(
                "Stopping safely. Completed progress is saved; an unfinished episode is discarded.",
                kind="lifecycle",
            )
        else:
            self.controller.stop()

    def load_config(self, path=None):
        try:
            self._loading = True
            config = self.config_service.load(path)
            self._loaded = config
            self._managed_paths = (
                path is None
                or Path(path).resolve() == self.config_service.default_path.resolve()
            ) and not config.get("resume_from")
            for key, widget in self.fields.items():
                value = config[key]
                if isinstance(widget, QComboBox):
                    stable_value = str(value).lower() if key == "device" else str(value)
                    if key == "stage" and widget.findData(stable_value) < 0:
                        widget.addItem(stable_value, stable_value)
                    widget.setCurrentIndex(widget.findData(stable_value))
                else:
                    widget.setValue(value)
            self.set_squad(config["squad"])
            self.auto_squad.setChecked(bool(config.get("auto_squad", False)))
            self._auto_squad_changed(self.auto_squad.isChecked())
            resume_path = config.get("resume_from") or ""
            self.resume_path.setText(str(resume_path))
            configured_index = self.resume_selector.findData("configured")
            if configured_index >= 0:
                self.resume_selector.removeItem(configured_index)
            if config.get("warm_start") and not resume_path:
                self.resume_selector.addItem("Loaded YAML starting point", "configured")
            bind_combo(
                self.resume_selector,
                {
                    "fresh": "Start from scratch",
                    "resume": "Continue a checkpoint",
                    "configured": "Loaded YAML starting point",
                },
            )
            self.resume_selector.setCurrentIndex(
                self.resume_selector.findData(
                    "resume"
                    if resume_path
                    else "configured"
                    if config.get("warm_start")
                    else "fresh"
                )
            )
            self._resume_changed()
            explicit_custom = (
                path is not None
                and Path(path).resolve() != self.config_service.default_path.resolve()
            )
            preset_key = "custom"
            if not explicit_custom:
                preset_key = next(
                    (
                        key
                        for key, values in self.PRESETS.items()
                        if all(config.get(k) == v for k, v in values.items())
                    ),
                    "custom",
                )
            self.size_selector.setCurrentIndex(self.size_selector.findData(preset_key))
            self._update_size_summary()
            pool = config.get("stage_pool") or []
            self.stage_pool_notice.setVisible(bool(pool))
            bind_text(
                self.stage_pool_notice,
                "Loaded configuration uses a stage pool: {stages}",
                stages=", ".join(pool),
            )
            bind_text(
                self.notice,
                "Loaded {name}. Changes save to gui_session.yaml.",
                name=Path(path or self.config_service.default_path).name,
            )
        except Exception as exc:  # noqa: BLE001 — GUI/worker exception boundary
            bind_text(self.notice, str(exc))
            self.controller.error.emit(str(exc))
        finally:
            self._loading = False

    def _edit_account(self):
        if self.controller.is_busy:
            return
        from desktop.widgets.account_dialog import AccountDialog
        dialog = AccountDialog(self)
        account_path = self._loaded.get('account_snapshot') or self._loaded.get('account_draft')
        if account_path:
            path = Path(account_path)
            if not path.is_absolute():path = self.config_service.root / path
            if path.exists():dialog.apply_draft(json.loads(path.read_text()))
        def save_reference(path):
            self._loaded['account_snapshot'] = path
            self._loaded.pop('account_draft', None)
            self._advanced_changed()
        dialog.snapshot_saved.connect(save_reference)
        dialog.exec()

    def _edit_progression(self):
        if self.controller.is_busy:
            return
        from arknights_sim.data.progression import Progression
        from desktop.widgets.progression_dialog import ProgressionDialog
        dialog = ProgressionDialog(self._loaded.get('data_dir'), self.selected_squad,
            Progression.from_dict(self._loaded.get('operator_progression')),
            self._loaded.get('skill_overrides'), self)
        if dialog.exec():
            self._loaded['operator_progression'] = dialog.request().to_dict()
            self._loaded['skill_overrides'] = dialog.overrides()
            self._advanced_changed()

    def collect_config(self):
        config = dict(self._loaded)
        for key, widget in self.fields.items():
            config[key] = (
                widget.currentData()
                if isinstance(widget, QComboBox)
                else widget.value()
            )
        config["squad"] = list(self.selected_squad)
        config["auto_squad"] = self.auto_squad.isChecked()
        if self.resume_selector.currentData() == "resume":
            path = self.resume_path.text().strip()
            if not path:
                raise ValueError(tr("Choose a resumable checkpoint before continuing."))
            checkpoint = Path(path).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = self.config_service.root / checkpoint
            if not checkpoint.is_file():
                raise ValueError(tr("The selected checkpoint file does not exist."))
            config["resume_from"] = str(checkpoint.resolve())
            config.pop("warm_start", None)
        else:
            config.pop("resume_from", None)
            if self.resume_selector.currentData() == "fresh":
                config.pop("warm_start", None)
        return self.config_service.validate(config)

    def _pick_squad(self):
        from desktop.widgets.squad_picker import SquadPicker

        try:
            picker = SquadPicker(
                self._loaded.get("data_dir"), self.selected_squad, self
            )
            if picker.exec():
                self.set_squad(picker.squad())
                self.auto_squad.setChecked(False)
        except Exception as exc:  # noqa: BLE001 — GUI boundary
            self.controller.error.emit(str(exc))

    def set_squad(self, squad):
        self.selected_squad = tuple(squad)
        bind_text(
            self.pick_squad, "Choose operators… ({count} selected)", count=len(squad)
        )

    def _choose_training_map(self):
        from desktop.paths import DATA_DIR
        from desktop.widgets.map_picker import MapPicker

        picker = MapPicker(self._loaded.get("data_dir") or DATA_DIR, self)
        if picker.exec() and picker.selected_id:
            key = picker.selected_id
            record = picker.catalog.records[key]
            stage = self.fields["stage"]
            if stage.findData(key) < 0:
                stage.addItem(f"{record['code']} · {record['name']}", key)
            stage.setCurrentIndex(stage.findData(key))
            self._loaded["stage_pool"] = []
            self.stage_pool_notice.hide()

    def _background_status(self):
        from desktop.services.background_training import read_background_training

        self._background = read_background_training()
        state = self._background
        if not state.get("active"):
            self._background_stop_requested = False
        use_background = (
            bool(state)
            and (state.get("active") or not self._foreground_seen)
            and not self.controller.is_busy
        )
        self.background_notice.setVisible(use_background)
        if use_background:
            task_key = (
                "background",
                state.get("pid"),
                state.get("started_at"),
                state.get("output_dir"),
            )
            if self._metric_task != task_key:
                self._reset_task_metrics(task_key)
            if state.get("active"):
                bind_text(
                    self.background_notice,
                    "This background task supports stop and save. Pause and resume are unavailable.",
                )
            else:
                bind_text(
                    self.background_notice,
                    "Last background task · {status} · {count} iterations completed",
                    status=tr(state.get("status", "UNKNOWN")),
                    count=state.get("iteration", 0),
                )
            values = dict(state.get("metrics", {}))
            values.setdefault("iteration", state.get("iteration", 0))
            self._metrics_updated(values)
            if self._dashboard is not None:
                self._dashboard.update_metrics(values)
                self._dashboard.set_status(training=state.get("status", "UNKNOWN"))
        self._refresh_controls(self.controller.state)

    def apply_defaults(self, settings):
        """Apply saved desktop defaults on startup; explicit custom YAML paths win."""
        self._settings = dict(settings)
        self.load_config(settings.get("config"))
        device = settings.get("device", "Auto")
        self.fields["device"].setCurrentIndex(
            self.fields["device"].findData(device.lower())
        )
        if settings.get("data_dir"):
            self._loaded["data_dir"] = settings["data_dir"]
        published_best = (
            Path(
                settings.get("checkpoint_dir", self.config_service.root / "checkpoints")
            )
            / "best.pt"
        )
        if not self._loaded.get("evaluation_checkpoint") and published_best.is_file():
            self._loaded["evaluation_checkpoint"] = str(published_best)
        if self._managed_paths:
            bind_text(
                self.notice,
                "Desktop defaults loaded. Each training run saves to its own dated folder; published models are preserved.",
            )

    def _run_config(self):
        config = self.collect_config()
        if self._managed_paths:
            session = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
            checkpoint_root = Path(
                self._settings.get(
                    "checkpoint_dir", self.config_service.root / "checkpoints"
                )
            )
            config["checkpoint_dir"] = str(checkpoint_root / "desktop" / session)
            config["output_dir"] = str(
                self.config_service.root / "outputs/desktop_training" / session
            )
        if not self._settings.get("auto_save_checkpoint", True):
            # The core always retains latest and safe-stop progress. This setting
            # controls archival per-iteration copies only, never safe-stop saves.
            config["checkpoint_interval"] = config["iterations"] + 1
        return config

    def save_config(self):
        try:
            path = self.config_service.save_session(self.collect_config())
            bind_text(self.notice, "Saved {path}", path=str(path))
            return path
        except Exception as exc:  # noqa: BLE001 — GUI/worker exception boundary
            self.controller.error.emit(str(exc))
            return None

    def _start(self):
        try:
            config = self._run_config()
            if self.controller.start(config):
                self._foreground_seen = True
                self._reset_task_metrics(("foreground", config.get("output_dir")))
                self.background_notice.hide()
                self._loaded = config
                bind_text(self.metrics["stage"], config["stage"])
                bind_text(
                    self.metrics["mcts_simulations"], str(config["mcts_simulations"])
                )
        except Exception as exc:  # noqa: BLE001 — GUI/worker exception boundary
            self.controller.error.emit(str(exc))

    def _evaluate(self):
        try:
            self.controller.evaluate(self.collect_config())
            if self.controller.is_busy and self.controller.state != "EVALUATING":
                self._task_message(
                    "Evaluation pending until the next completed iteration. Resume if paused.",
                )
        except Exception as exc:  # noqa: BLE001 — GUI/worker exception boundary
            self.controller.error.emit(str(exc))

    def _save_checkpoint(self):
        self.controller.save_checkpoint()
        self._task_message(
            "Checkpoint request pending until the next completed iteration. Resume if paused; STOP also saves completed progress.",
        )

    def set_compute_available(self, available):
        self._compute_available = available
        self._refresh_controls(self.controller.state)

    def _refresh_controls(self, state):
        background_active = bool(self._background.get("active"))
        busy = self.controller.is_busy
        available = self.controller.availability
        self.pause_button.setEnabled(
            busy and state in ("STARTING", "RUNNING") and not background_active
        )
        self.resume_button.setEnabled(
            busy and state in ("PAUSING", "PAUSED") and not background_active
        )
        self.pause_button.hide()
        self.resume_button.hide()
        self.background_stop.hide()
        self.config_group.setEnabled(not busy and not background_active)
        self.advanced.setEnabled(not busy and not background_active)
        self.config_group.setVisible(not busy and not background_active)
        self.advanced.setVisible(not busy and not background_active)
        self.task_summary.setVisible(busy or background_active)
        self._update_task_summary()
        self.stop_button.setVisible(busy or background_active)
        self.stop_button.setEnabled(
            (busy and state != "STOPPING")
            or (
                background_active
                and self._background.get("status") != "STOPPING"
                and not self._background_stop_requested
            )
        )
        if background_active:
            bind_text(self.task_source, "Background training")
            bind_text(self.state_label, self._background.get("status", "UNKNOWN"))
            bind_text(self.start_button, "Background task active")
            self.start_button.setEnabled(False)
        elif busy:
            bind_text(self.task_source, "Training in this app")
            bind_text(self.state_label, state)
            if state in ("PAUSING", "PAUSED"):
                bind_text(self.start_button, "RESUME")
                self.start_button.setEnabled(True)
            elif state in ("STARTING", "RUNNING"):
                bind_text(self.start_button, "PAUSE")
                self.start_button.setEnabled(True)
            else:
                bind_text(self.start_button, state)
                self.start_button.setEnabled(False)
        else:
            show_previous = (
                bool(self._background) and not self._foreground_seen and state == "IDLE"
            )
            bind_text(
                self.task_source,
                "Last background task" if show_previous else "Training in this app",
            )
            bind_text(
                self.state_label,
                self._background.get("status", "UNKNOWN") if show_previous else state,
            )
            bind_text(self.start_button, "START TRAINING")
            self.start_button.setEnabled(available and self._compute_available)
        self.save_checkpoint_button.setEnabled(
            busy and state in ("RUNNING", "PAUSING", "PAUSED") and not background_active
        )
        self.evaluate_button.setEnabled(
            available
            and not background_active
            and state not in ("STOPPING", "EVALUATING", "STARTING", "ERROR")
            and (busy or self._compute_available)
        )
        if state == "PAUSING":
            self._task_message(
                "Pausing at the next safe decision/update boundary.", kind="lifecycle"
            )
        elif state == "STOPPING":
            self._task_message(
                "Stopping safely. Completed progress is saved; an unfinished episode is discarded.",
                kind="lifecycle",
            )
        elif self._task_notice_kind == "lifecycle" or not (busy or background_active):
            self.task_notice.hide()
            self._task_notice_kind = None

    def _message(self, message):
        if "pending" in message:
            self._task_message(message)

    def _metrics_updated(self, metrics):
        metrics = dict(metrics)
        if "evaluation_success_rate" in metrics:
            metrics["success_rate"] = metrics["evaluation_success_rate"]
        for key, value in metrics.items():
            if key in self.metrics:
                bind_text(
                    self.metrics[key],
                    f"{value:.6f}" if isinstance(value, float) else str(value),
                )
        if "training_wall_time" in metrics:
            bind_text(
                self.metrics["training_wall_time"],
                f"{metrics['training_wall_time']:.1f} s",
            )
        if "success_rate" in metrics:
            bind_text(self.metrics["success_rate"], f"{metrics['success_rate']:.1%}")
            point = (
                metrics.get("iteration", self._loaded.get("iteration", 0)),
                metrics["success_rate"],
            )
            if (
                self._dashboard is None
                and getattr(self, "_last_summary_point", None) != point
            ):
                self.summary_chart.add_value(*point)
                self._last_summary_point = point

    def _event(self, event):
        payload = event.payload
        if event.kind == "training_started":
            self._foreground_seen = True
            self._reset_task_metrics(("foreground", self._loaded.get("output_dir")))
            self.background_notice.hide()
        self._metrics_updated(payload)
        if event.kind == "iteration_started":
            bind_text(self.metrics["episode"], f"0 / {payload.get('episodes', '?')}")
            bind_text(self.metrics["episode_result"], "Running")
        elif event.kind == "episode_finished":
            bind_text(
                self.metrics["episode"],
                f"{payload.get('episode', '?')} / {self._loaded.get('episodes_per_iteration', '?')}",
            )
            if payload.get("success"):
                bind_text(self.metrics["episode_result"], "Zero-leak clear")
            else:
                bind_text(
                    self.metrics["episode_result"],
                    _episode_result_source(payload),
                    kills=payload.get("kills", "?"),
                    leaks=payload.get("leaks", "?"),
                )
        elif event.kind == "checkpoint_saved":
            bind_text(self.metrics["model"], Path(payload["path"]).name)
            self._task_message(
                "Checkpoint saved: {name}", name=Path(payload["path"]).name
            )
        elif event.kind == "evaluation_finished":
            self._task_message("Evaluation completed.")
        if self._dashboard is not None:
            self._dashboard.update_metrics(payload)
            if event.kind == "episode_finished":
                value = self._dashboard.cards["episode_result"].value_label
                if payload.get("success"):
                    bind_text(value, "Zero-leak clear")
                else:
                    bind_text(
                        value,
                        _episode_result_source(payload),
                        kills=payload.get("kills", "?"),
                        leaks=payload.get("leaks", "?"),
                    )
