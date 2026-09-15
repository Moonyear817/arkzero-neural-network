"""Simulator controls and visual playback, with no battle rules in callbacks."""

import re
import json
import os
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
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
from desktop.widgets.battle_timeline import BattleTimeline
from desktop.widgets.disclosure import Disclosure
from desktop.widgets.stage_map import StageMap
from desktop.widgets.progression_dialog import ProgressionDialog
from arknights_sim.data.progression import Progression
from desktop.paths import WORKSPACE_ROOT

register_translations(
    {
        "Simulator stage": "模拟器关卡",
        "Wave {wave} · Fragment {fragment} · Decisions {decisions}": "波次 {wave} · 片段 {fragment} · 决策 {decisions}",
        "Waiting for clearance: {count}": "等待清场：剩余 {count} 个推进阻挡敌人",
        "Upcoming: {groups}": "接下来：{groups}",
        "{name} ×{count}, route {route}, {timing}": "{name} ×{count}，路线 {route}，{timing}",
        "after trigger": "满足条件后出现",
        "in {seconds:.1f}s": "{seconds:.1f} 秒后",
        "Untrained reward verification model": "奖励机制验证模型（尚未训练）",
        "Event planning verification model (smoke trained)": "事件规划验证模型（仅完成短训练）",
        "Simulator seed": "模拟器随机种子",
        "Simulator control mode": "模拟器控制方式",
        "Stage": "关卡",
        "Seed": "随机种子",
        "Control": "控制方式",
        "Squad": "干员队伍",
        "Noir Corne · 黑角": "黑角",
        "Melantha · 玫兰莎": "玫兰莎",
        "RUN": "开始模拟",
        "Play": "继续播放",
        "Pause": "暂停",
        "Step": "单步",
        "Restart": "重新开始",
        "Stop": "停止",
        "Scripted baseline": "历史回归基线",
        "Manual control": "手动控制",
        "Automatic control": "自动控制",
        "Hybrid control": "人机协同",
        "Random agent": "随机智能体",
        "No deployments": "不部署干员",
        "Playback": "播放速度",
        "MAX": "最快",
        "Visualization playback speed": "可视化播放速度",
        "Legal simulator action": "模拟器合法动作",
        "Apply action": "执行动作",
        "Filter legal actions; click a map tile to filter deployments": "筛选合法动作；点击地图格子可筛选对应部署动作",
        "Historical baseline reproduces the existing regression. Manual control starts paused; Step advances to the next decision event.": "历史基线用于复现已有回归测试。手动模式从暂停开始；单步会推进到下一个决策事件。",
        "Time —    DP —    Life —    Kills —    Leaks —": "时间 —    费用 —    生命 —    击杀 —    漏怪 —",
        "Time {time:.2f}s    DP {dp:.1f}    Life {life}    Kills {kills}/{total}    Leaks {leaks}": "时间 {time:.2f}秒    费用 {dp:.1f}    生命 {life}    击杀 {kills}/{total}    漏怪 {leaks}",
        "{count} legal actions": "{count} 个合法动作",
        "IDLE": "空闲",
        "LOADING": "加载中",
        "RUNNING": "运行中",
        "PAUSED": "已暂停",
        "STOPPING": "正在停止",
        "STOPPED": "已停止",
        "FINISHED": "已结束",
        "ERROR": "发生错误",
        "SUCCESS": "通关成功",
        "FAILED ({reason})": "未通关（{reason}）",
        "battle_end": "战斗结束",
        "decision_limit": "达到决策次数上限",
        "time_limit": "达到时间上限",
        "DEPLOY": "部署",
        "RETREAT": "撤退",
        "ACTIVATE_SKILL": "开启技能",
        "DEPLOY_SUMMON": "部署召唤物",
        "RETREAT_SUMMON": "撤退召唤物",
        "PLACE_DEVICE": "放置装置",
        "ACTIVATE_DEVICE": "启动机关",
        "REMOVE_DEVICE": "撤除装置",
        "WAIT": "等待",
        "RIGHT": "向右",
        "UP": "向上",
        "LEFT": "向左",
        "DOWN": "向下",
    }
)


def _label(source):
    label = QLabel()
    bind_text(label, source)
    return label


def _result_text(result):
    if result.startswith("FAILED ("):
        return tr("FAILED ({reason})").format(reason=tr(result[8:-1]))
    return tr(result)


def _action_text(label):
    for source in (
        "PLACE_DEVICE",
        "ACTIVATE_DEVICE",
        "REMOVE_DEVICE",
        "ACTIVATE_SKILL",
        "DEPLOY_SUMMON",
        "RETREAT_SUMMON",
        "DEPLOY",
        "RETREAT",
        "WAIT",
        "RIGHT",
        "UP",
        "LEFT",
        "DOWN",
    ):
        label = label.replace(source, tr(source))
    for original, source in (
        ("障碍物", "Obstacle"),
        ("侦测器", "Detector"),
        ("源石流发生装置", "Originium flow device"),
        ("重力控制", "Gravity control"),
    ):
        label = label.replace(original, tr(source))
    return label


def _mechanics_text(text):
    """Translate the current DTO's legacy display strings without game logic."""
    for original, source in (
        ("右", "RIGHT"),
        ("上", "UP"),
        ("左", "LEFT"),
        ("下", "DOWN"),
    ):
        text = text.replace(
            f"重力方向：{original}",
            tr("Gravity: {direction}").format(direction=tr(source)),
        )
    text = re.sub(
        r"侦测器：反隐剩余 ([\d.]+)秒",
        lambda match: tr("Detector: reveal {seconds}s remaining").format(
            seconds=match[1]
        ),
        text,
    )
    text = re.sub(
        r"侦测器：充能 ([\d.]+)/([\d.]+)",
        lambda match: tr("Detector: charge {sp}/{cost}").format(
            sp=match[1], cost=match[2]
        ),
        text,
    )
    return re.sub(
        r"障碍物剩余：(\d+)",
        lambda match: tr("Obstacles remaining: {count}").format(count=match[1]),
        text,
    )


register_translations(
    {
        "Start battle": "开始对战",
        "Resume battle": "继续",
        "End battle": "结束",
        "AI automatic": "AI 自动",
        "Manual": "手动",
        "Cooperative": "人机协同",
        "More settings": "更多设置",
        "Battle timeline": "战斗时间轴",
        "Local maps…": "本地地图库…",
        "Choose squad…": "选择干员…",
        "Choose model…": "选择模型…",
        "Choose policy model": "选择策略模型",
        "Models (*.pt *.pth)": "模型 (*.pt *.pth)",
        "Let the model choose the squad": "由模型自主编队",
        "Diagnostic mode": "诊断模式",
        "Use selected control": "使用上方控制方式",
        "Model": "模型",
        "Selected model: {name}": "所选模型：{name}",
        "This battle: {name}": "本局模型：{name}",
        "Loading model: {name}": "正在加载模型：{name}",
        "Requested model: {name}": "请求的模型：{name}",
        "No model selected": "尚未选择模型",
        "No model used": "本局不使用模型",
        "Model not used in this mode": "当前模式不使用模型",
        "Squad source: manual · {count} operators": "编队来源：手动 · {count} 名干员",
        "Squad source: model": "编队来源：模型自主选择",
        "This battle's squad: manual · {count} operators": "本局编队：手动 · {count} 名干员",
        "This battle's squad: model": "本局编队：模型自主选择",
        "The selected roster is saved for manual or cooperative control.": "已保存手动编队，可在手动或人机协同模式使用。",
        "End the current battle before changing its model or squad.": "请先结束当前对战，再切换模型或编队。",
        "Select an existing model file (.pt or .pth).": "请选择现有的模型文件（.pt 或 .pth）。",
        "Training or model research is using compute resources. Stop it first, or choose manual control.": "训练或模型研究正在使用计算资源；请先停止它，或选择手动控制。",
        "A squad may contain at most 12 distinct operators.": "编队最多包含 12 名不重复的干员。",
        "The historical baseline requires stage 0-1 with Noir Corne and Melantha.": "历史回归基线需要关卡 0-1，以及黑角和玫兰莎两名干员。",
        "Manual: choose an action or click a tile. Step advances to the next decision event.": "手动：选择动作或点击地图格子。单步推进到下一个决策事件。",
        "Cooperative: resume to let the model act; a manual action pauses the model.": "人机协同：继续后由模型行动；执行手动动作会暂停模型。",
        "AI automatic: the selected model chooses the squad and battle actions.": "AI 自动：所选模型负责自主编队和战斗动作。",
        "Diagnostic playback uses the selected baseline or random policy.": "诊断播放使用所选回归基线或随机策略。",
        "Actions are available while a manual or cooperative battle is active.": "手动或人机协同对战开始后可执行动作。",
        "Obstacle": "障碍物",
        "Detector": "侦测器",
        "Originium flow device": "源石流发生装置",
        "Gravity control": "重力控制",
        "Gravity: {direction}": "重力方向：{direction}",
        "Detector: reveal {seconds}s remaining": "侦测器：反隐剩余 {seconds}秒",
        "Detector: charge {sp}/{cost}": "侦测器：充能 {sp}/{cost}",
        "Obstacles remaining: {count}": "障碍物剩余：{count}",
    }
)


class SimulatorView(QWidget):
    """Map-first presentation. Controls only dispatch existing engine APIs."""

    NEURAL_MODES = ("Automatic control", "Hybrid control")
    MANUAL_MODES = ("Manual control", "Hybrid control")
    BUSY_STATES = ("LOADING", "RUNNING", "PAUSED", "STOPPING")

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._options = ()
        self._active_mode = None
        self._compute_available = True
        self._battle_setup = None
        self._battle_loaded = False
        self._model_selected_explicitly = False
        self.selected_squad = ("char_500_noirc", "char_208_melan")
        self.progression = Progression()
        self.skill_overrides = {}
        self.progression_path = WORKSPACE_ROOT / "configs/simulator_progression.json"
        self._progression_error = None
        if self.progression_path.exists():
            try:
                saved = json.loads(self.progression_path.read_text())
                self.progression = Progression.from_dict(saved["progression"])
                self.skill_overrides = dict(saved.get("skill_overrides", {}))
                for index in self.skill_overrides.values(): Progression(skill_index=index)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.progression = Progression()
                self.skill_overrides = {}
                self._progression_error = str(exc)

        self.stage = QComboBox()
        self.stage.addItem("0-1", "0-1")
        bind_text(self.stage, "Simulator stage", "setAccessibleName")
        self.map_picker = bind_text(QPushButton(), "Local maps…")
        self.map_picker.clicked.connect(self._pick_map)
        self.mode = QComboBox()
        mode_labels = {
            "Automatic control": "AI automatic",
            "Manual control": "Manual control",
            "Hybrid control": "Cooperative",
        }
        for key in ("Manual control", "Automatic control", "Hybrid control"):
            self.mode.addItem(mode_labels[key], key)
        bind_combo(self.mode, mode_labels)
        bind_text(self.mode, "Simulator control mode", "setAccessibleName")
        basic = QHBoxLayout()
        basic.addWidget(_label("Stage"))
        basic.addWidget(self.stage, 1)
        basic.addWidget(self.map_picker)
        basic.addSpacing(12)
        basic.addWidget(_label("Control"))
        basic.addWidget(self.mode)

        self.model_row = QWidget()
        model_line = QHBoxLayout(self.model_row)
        model_line.setContentsMargins(0, 0, 0, 0)
        self.model_summary = QLabel()
        self.model_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.model_summary.setMinimumWidth(100)
        self.model_browse = bind_text(QPushButton(), "Choose model…")
        self.model_browse.clicked.connect(self._browse_model)
        self.auto_squad = bind_text(QCheckBox(), "Let the model choose the squad")
        model_line.addWidget(self.model_summary, 1)
        model_line.addWidget(self.model_browse)
        model_line.addWidget(self.auto_squad)
        # Read-only compatibility storage, never an unannounced mutable model.
        self.model_path = QLineEdit(self)
        self.model_path.setReadOnly(True)
        self.model_path.hide()

        candidates = (
            WORKSPACE_ROOT / "outputs/event_architecture/verification.pt",
            WORKSPACE_ROOT / "outputs/reward_verification/initial.pt",
            WORKSPACE_ROOT / "outputs/mechanics_verification/best.pt",
            WORKSPACE_ROOT / "outputs/overnight_joint/checkpoints/best.pt",
            WORKSPACE_ROOT / "checkpoints/best.pt",
        )
        self.model_path.setText(
            str(next((path for path in candidates if path.is_file()), candidates[-1]))
        )

        self.squad_row = QWidget()
        squad_line = QHBoxLayout(self.squad_row)
        squad_line.setContentsMargins(0, 0, 0, 0)
        self.squad_summary = QLabel()
        self.pick_squad = bind_text(QPushButton(), "Choose squad…")
        self.pick_squad.clicked.connect(self._pick_squad)
        squad_line.addWidget(self.squad_summary, 1)
        squad_line.addWidget(self.pick_squad)
        self.progression_button = bind_text(QPushButton(), "Operator progression")
        self.progression_button.clicked.connect(self._edit_progression)
        squad_line.addWidget(self.progression_button)

        self.battle_button = bind_text(QPushButton(), "Start battle")
        self.battle_button.setObjectName("simulator_battle")
        self.battle_button.clicked.connect(self._primary_action)
        self.stop_button = bind_text(QPushButton(), "End battle")
        self.stop_button.clicked.connect(controller.stop)
        self.step_button = bind_text(QPushButton(), "Step")
        self.step_button.clicked.connect(controller.step_once)
        self.speed = QComboBox()
        for label, value in (
            ("1x", 1.0),
            ("2x", 2.0),
            ("5x", 5.0),
            ("10x", 10.0),
            ("MAX", 0.0),
        ):
            self.speed.addItem(label, value)
        bind_combo(
            self.speed, {1.0: "1x", 2.0: "2x", 5.0: "5x", 10.0: "10x", 0.0: "MAX"}
        )
        bind_text(self.speed, "Visualization playback speed", "setAccessibleName")
        self.speed.currentIndexChanged.connect(
            lambda _: controller.set_speed(self.speed.currentData())
        )
        controls = QHBoxLayout()
        controls.addWidget(self.battle_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.step_button)
        controls.addStretch()
        controls.addWidget(_label("Playback"))
        controls.addWidget(self.speed)

        # Old programmatic integrations remain usable, with no duplicate buttons.
        self.run_button = bind_text(QPushButton(self), "RUN")
        self.play_button = bind_text(QPushButton(self), "Play")
        self.pause_button = bind_text(QPushButton(self), "Pause")
        self.run_button.setObjectName("simulator_run")
        for button in (self.run_button, self.play_button, self.pause_button):
            button.hide()
        self.run_button.clicked.connect(self._run)
        self.play_button.clicked.connect(controller.resume)
        self.pause_button.clicked.connect(controller.pause)

        self.status_label = QLabel()
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        info = QHBoxLayout()
        info.addWidget(self.metrics, 1)
        info.addWidget(self.status_label)
        self.map = StageMap()
        self.stage_map = self.map
        self.timeline = BattleTimeline()
        self.timeline.setMinimumHeight(130)
        self.timeline.setMaximumHeight(250)
        self.timeline_section = Disclosure("Battle timeline", expanded=False)
        self.timeline_section.add_widget(self.timeline)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.map)
        self.splitter.addWidget(self.timeline_section)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setChildrenCollapsible(False)

        self.manual_actions = QWidget()
        manual_layout = QVBoxLayout(self.manual_actions)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        self.action_filter = QLineEdit()
        bind_text(
            self.action_filter,
            "Filter legal actions; click a map tile to filter deployments",
            "setPlaceholderText",
        )
        self.action_combo = QComboBox()
        self.action_combo.setMinimumContentsLength(25)
        bind_text(self.action_combo, "Legal simulator action", "setAccessibleName")
        self.apply_button = bind_text(QPushButton(), "Apply action")
        self.apply_button.setObjectName("simulator_apply_action")
        self.action_count = QLabel()
        action_line = QHBoxLayout()
        action_line.addWidget(self.action_combo, 1)
        action_line.addWidget(self.apply_button)
        action_line.addWidget(self.action_count)
        manual_layout.addWidget(self.action_filter)
        manual_layout.addLayout(action_line)
        self.apply_button.clicked.connect(self._apply_action)
        self.action_filter.textChanged.connect(self._refresh_actions)
        self.map.tile_selected.connect(self._tile_selected)
        self.help_label = QLabel()
        self.help_label.setWordWrap(True)

        self.more_settings = Disclosure("More settings", expanded=False)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(12345)
        bind_text(self.seed, "Simulator seed", "setAccessibleName")
        self.experiment_mode = QComboBox()
        diagnostics = {
            "": "Use selected control",
            "Scripted baseline": "Scripted baseline",
            "Random agent": "Random agent",
            "No deployments": "No deployments",
        }
        for key, label in diagnostics.items():
            self.experiment_mode.addItem(label, key)
        bind_combo(self.experiment_mode, diagnostics)
        extra = QGridLayout()
        extra.addWidget(_label("Seed"), 0, 0)
        extra.addWidget(self.seed, 0, 1)
        extra.addWidget(_label("Diagnostic mode"), 0, 2)
        extra.addWidget(self.experiment_mode, 0, 3)
        self.restart_button = bind_text(QPushButton(), "Restart")
        self.restart_button.clicked.connect(self._restart)
        extra.addWidget(self.restart_button, 0, 4)
        self.more_settings.add_layout(extra)
        self.more_settings.add_widget(self.help_label)

        layout = QVBoxLayout(self)
        layout.addLayout(basic)
        layout.addWidget(self.model_row)
        layout.addWidget(self.squad_row)
        layout.addLayout(controls)
        layout.addLayout(info)
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self.manual_actions)
        layout.addWidget(self.more_settings)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.experiment_mode.currentIndexChanged.connect(self._mode_changed)
        self.auto_squad.toggled.connect(self._refresh_context)
        controller.snapshot.connect(self._snapshot)
        controller.events.connect(self.timeline.append_events)
        controller.state_changed.connect(self._status)
        language_manager.language_changed.connect(self._retranslate)
        self._mode_changed()
        self._retranslate()

    def _effective_mode(self):
        return self.experiment_mode.currentData() or self.mode.currentData()

    def _busy(self):
        return self.controller.is_busy() or self.controller.status in self.BUSY_STATES

    def _reject(self, source):
        self.controller.error.emit(tr(source))
        return False

    def _primary_action(self):
        if self.controller.status == "RUNNING":
            self.controller.pause()
        elif self.controller.status == "PAUSED":
            self.controller.resume()
        elif self.controller.status not in ("LOADING", "STOPPING"):
            self._run()

    def _run(self):
        if self._busy():
            return False
        mode = self._effective_mode()
        if not self._compute_available and mode != "Manual control":
            return self._reject(
                "Training or model research is using compute resources. Stop it first, or choose manual control."
            )
        squad = self.selected_squad
        if len(squad) > 12 or len(set(squad)) != len(squad):
            return self._reject("A squad may contain at most 12 distinct operators.")
        if mode == "Scripted baseline" and (
            set(squad) != {"char_500_noirc", "char_208_melan"}
            or self.stage.currentData() not in ("0-1", "level_main_00-01")
        ):
            return self._reject(
                "The historical baseline requires stage 0-1 with Noir Corne and Melantha."
            )
        checkpoint = (
            self.model_path.text().strip() if mode in self.NEURAL_MODES else None
        )
        if checkpoint is not None and not Path(checkpoint).is_file():
            return self._reject("Select an existing model file (.pt or .pth).")
        auto_squad = mode == "Automatic control" or (
            mode == "Hybrid control" and self.auto_squad.isChecked()
        )
        self._active_mode = mode
        self._battle_loaded = False
        self._battle_setup = {
            "mode": mode,
            "checkpoint": checkpoint,
            "auto_squad": auto_squad,
            "squad": tuple(squad),
        }
        self.timeline.clear()
        self.action_filter.clear()
        accepted = self.controller.start(
            self.stage.currentData(),
            squad,
            self.seed.value(),
            self.speed.currentData(),
            mode=mode,
            checkpoint=checkpoint,
            auto_squad=auto_squad,
            progression=self.progression.to_dict(),
            skill_overrides=self.skill_overrides,
        )
        if accepted is False:
            self._battle_setup = None
        self._refresh_context()
        return accepted

    def _edit_progression(self):
        if self._busy(): return
        dialog = ProgressionDialog(self.controller.data_dir, self.selected_squad,
                                   self.progression, self.skill_overrides, self)
        if self._progression_error:
            dialog.message.setText(self._progression_error)
        if dialog.exec():
            request, overrides = dialog.request(), dialog.overrides()
            try:
                self.progression_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.progression_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"progression":request.to_dict(), "skill_overrides":overrides}, indent=2)+"\n")
                os.replace(temporary, self.progression_path)
            except OSError as exc:
                self.controller.error.emit(str(exc));return
            self.progression, self.skill_overrides = request, overrides
            self._progression_error = None
            self._refresh_context()

    def set_model(self, path):
        """Explicit model-page handoff: never change mode, squad or a live run."""
        if self._busy():
            return self._reject(
                "End the current battle before changing its model or squad."
            )
        value = Path(path).expanduser()
        if value.suffix.lower() not in (".pt", ".pth") or not value.is_file():
            return self._reject("Select an existing model file (.pt or .pth).")
        self.model_path.setText(str(value.resolve()))
        self._model_selected_explicitly = True
        self._refresh_context()
        return True

    def set_squad(self, squad):
        if self._busy():
            return self._reject(
                "End the current battle before changing its model or squad."
            )
        squad = tuple(squad)
        if len(squad) > 12 or len(set(squad)) != len(squad):
            return self._reject("A squad may contain at most 12 distinct operators.")
        self.selected_squad = squad
        # Roster selection does not silently switch modes or disable model squads.
        self._refresh_context()
        return True

    def _pick_squad(self):
        if self._busy():
            return
        from desktop.widgets.squad_picker import SquadPicker

        try:
            picker = SquadPicker(self.controller.data_dir, self.selected_squad, self)
        except Exception as exc:  # noqa: BLE001 -- preserve picker failure reporting
            self.controller.error.emit(str(exc))
            return
        if picker.exec():
            self.set_squad(picker.squad())

    def _browse_model(self):
        if self._busy():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Choose policy model"),
            self.model_path.text(),
            tr("Models (*.pt *.pth)"),
        )
        if path:
            self.set_model(path)

    def _pick_map(self):
        if self._busy():
            return
        from desktop.paths import DATA_DIR
        from desktop.widgets.map_picker import MapPicker

        picker = MapPicker(self.controller.data_dir or DATA_DIR, self)
        if picker.exec() and picker.selected_id:
            key = picker.selected_id
            record = picker.catalog.records[key]
            if self.stage.findData(key) < 0:
                self.stage.addItem(f"{record['code']} · {record['name']}", key)
            self.stage.setCurrentIndex(self.stage.findData(key))

    def set_compute_available(self, available):
        self._compute_available = bool(available)
        self._update_buttons()

    def _mode_changed(self, *_):
        if not self._busy():
            blocker = QSignalBlocker(self.auto_squad)
            if self.mode.currentData() != "Hybrid control":
                self.auto_squad.setChecked(
                    self.mode.currentData() == "Automatic control"
                )
            del blocker
        self._status(self.controller.status)

    def _refresh_context(self, *_):
        busy = self.controller.status in self.BUSY_STATES
        setup = self._battle_setup if busy else None
        mode = setup["mode"] if setup else self._effective_mode()
        neural = mode in self.NEURAL_MODES
        self.model_row.setVisible(neural or self._model_selected_explicitly)
        self.model_browse.setVisible(neural)
        self.auto_squad.setVisible(mode == "Hybrid control")
        self.auto_squad.setEnabled(not busy and mode == "Hybrid control")
        automatic_squad = (
            setup["auto_squad"]
            if setup
            else mode == "Automatic control"
            or (mode == "Hybrid control" and self.auto_squad.isChecked())
        )
        count = len(setup["squad"] if setup else self.selected_squad)
        source = (
            "This battle's squad: model"
            if busy and automatic_squad
            else "Squad source: model"
            if automatic_squad
            else "This battle's squad: manual · {count} operators"
            if busy
            else "Squad source: manual · {count} operators"
        )
        self.squad_summary.setText(tr(source).format(count=count))
        self.squad_summary.setToolTip(
            tr("The selected roster is saved for manual or cooperative control.")
            if automatic_squad
            else "\n".join(setup["squad"] if setup else self.selected_squad)
        )
        self.pick_squad.setVisible(not automatic_squad)
        self.pick_squad.setEnabled(not busy)
        self.progression_button.setEnabled(not busy)
        p = self.progression
        self.progression_button.setToolTip(f'E{p.elite} Lv{p.level} · P{p.potential+1} · S{p.skill_index+1} Lv{p.skill_level}')
        checkpoint = setup["checkpoint"] if setup else self.model_path.text().strip()
        if checkpoint:
            name = Path(checkpoint).name
            if Path(checkpoint).parent.name == "reward_verification" and name == "initial.pt":
                name = tr("Untrained reward verification model")
            elif Path(checkpoint).parent.name == "event_architecture" and name == "verification.pt":
                name = tr("Event planning verification model (smoke trained)")
            source = (
                "Loading model: {name}"
                if busy and not self._battle_loaded
                else "This battle: {name}"
                if busy
                else "Selected model: {name}"
            )
            self.model_summary.setText(tr(source).format(name=name))
            if not neural:
                self.model_summary.setText(
                    self.model_summary.text()
                    + " · "
                    + tr("Model not used in this mode")
                )
            self.model_summary.setToolTip(str(Path(checkpoint).expanduser().resolve()))
        else:
            self.model_summary.setText(
                tr("No model used") if busy and not neural else tr("No model selected")
            )
            self.model_summary.setToolTip("")
        hint = (
            "Manual: choose an action or click a tile. Step advances to the next decision event."
            if mode == "Manual control"
            else "Cooperative: resume to let the model act; a manual action pauses the model."
            if mode == "Hybrid control"
            else "AI automatic: the selected model chooses the squad and battle actions."
            if mode == "Automatic control"
            else "Diagnostic playback uses the selected baseline or random policy."
        )
        bind_text(self.help_label, hint)
        self.manual_actions.setVisible(
            mode in self.MANUAL_MODES
            and self.controller.status in ("RUNNING", "PAUSED")
        )

    def _restart(self):
        if self.controller.status in ("LOADING", "STOPPING"):
            return
        self.timeline.clear()
        # Restart uses the explicitly selected setup, including any model handoff.
        if not self._busy():
            self._run()
        else:
            self._battle_loaded = False
            self.controller.restart()

    def _tile_selected(self, x, y):
        if self._active_mode in self.MANUAL_MODES and self.controller.status in (
            "RUNNING",
            "PAUSED",
        ):
            self.action_filter.setText(f"@ ({x}, {y})")

    def _apply_action(self):
        payload = self.action_combo.currentData()
        if payload and self._active_mode in self.MANUAL_MODES:
            self.controller.submit_action(payload)

    def _snapshot(self, snapshot):
        self._battle_loaded = True
        self.map.set_snapshot(snapshot)
        self.metrics.setText(
            tr(
                "Time {time:.2f}s    DP {dp:.1f}    Life {life}    Kills {kills}/{total}    Leaks {leaks}"
            ).format(
                time=snapshot.time,
                dp=snapshot.dp,
                life=snapshot.life,
                kills=snapshot.kills,
                total=snapshot.total_enemies,
                leaks=snapshot.leaks,
            )
        )
        self._options = snapshot.legal_actions
        if getattr(snapshot, "mechanics_status", ""):
            self.metrics.setText(
                self.metrics.text() + "\n" + _mechanics_text(snapshot.mechanics_status)
            )
        progress = dict(getattr(snapshot, 'event_progression', ()))
        if progress:
            stage_text = tr('Wave {wave} · Fragment {fragment} · Decisions {decisions}').format(
                wave=max(0, progress.get('current_wave', -1) + 1),
                fragment=max(0, progress.get('current_fragment', -1) + 1),
                decisions=snapshot.decision_count)
            if progress.get('blocking'):
                stage_text += ' · ' + tr('Waiting for clearance: {count}').format(count=progress.get('blocking_enemy_count', 0))
            groups = [tr('{name} ×{count}, route {route}, {timing}').format(
                name=name, count=count, route=route + 1,
                timing=tr('after trigger') if delay is None else tr('in {seconds:.1f}s').format(seconds=max(0, delay)))
                for name, route, count, delay in getattr(snapshot, 'upcoming_events', ())]
            if groups:
                stage_text += '\n' + tr('Upcoming: {groups}').format(groups='；'.join(groups))
            self.metrics.setText(self.metrics.text() + '\n' + stage_text)
        self.action_count.setText(
            tr("{count} legal actions").format(count=len(self._options))
        )
        self._refresh_actions()
        self._refresh_context()
        if snapshot.terminal:
            self.status_label.setText(_result_text(snapshot.result))

    def _refresh_actions(self, *_):
        previous = self.action_combo.currentData()
        blocker = QSignalBlocker(self.action_combo)
        self.action_combo.clear()
        query = self.action_filter.text().casefold()
        for option in self._options:
            translated = _action_text(option.label)
            if query in option.label.casefold() or query in translated.casefold():
                self.action_combo.addItem(translated, option.payload_json)
        index = self.action_combo.findData(previous)
        if index >= 0:
            self.action_combo.setCurrentIndex(index)
        del blocker
        self.apply_button.setEnabled(
            self._active_mode in self.MANUAL_MODES
            and self.controller.status in ("RUNNING", "PAUSED")
            and self.action_combo.count() > 0
        )

    def _update_buttons(self):
        status = self.controller.status
        busy = status in self.BUSY_STATES
        available = (
            self._compute_available or self._effective_mode() == "Manual control"
        )
        source = (
            "Pause"
            if status == "RUNNING"
            else "Resume battle"
            if status == "PAUSED"
            else "LOADING"
            if status == "LOADING"
            else "STOPPING"
            if status == "STOPPING"
            else "Start battle"
        )
        bind_text(self.battle_button, source)
        self.battle_button.setEnabled(
            status in ("RUNNING", "PAUSED") or (not busy and available)
        )
        self.run_button.setEnabled(not busy and available)
        self.play_button.setEnabled(status == "PAUSED")
        self.pause_button.setEnabled(status == "RUNNING")
        self.step_button.setVisible(status == "PAUSED")
        self.step_button.setEnabled(status == "PAUSED")
        self.stop_button.setVisible(busy)
        self.stop_button.setEnabled(busy and status != "STOPPING")
        self.restart_button.setEnabled(status not in ("IDLE", "LOADING", "STOPPING"))

    def _status(self, status):
        result = self.controller.last_snapshot
        self.status_label.setText(
            _result_text(result.result)
            if status == "FINISHED" and result is not None
            else tr(status)
        )
        diagnostic = (
            self._active_mode
            if status in self.BUSY_STATES
            else self.experiment_mode.currentData()
        )
        if diagnostic in ("Scripted baseline", "Random agent", "No deployments"):
            self.status_label.setText(self.status_label.text() + " · " + tr(diagnostic))
        busy = status in self.BUSY_STATES
        for control in (
            self.stage,
            self.map_picker,
            self.seed,
            self.mode,
            self.experiment_mode,
            self.model_browse,
        ):
            control.setEnabled(not busy)
        self._refresh_context()
        self._update_buttons()
        self._refresh_actions()

    def _retranslate(self, *_):
        if self.controller.last_snapshot is not None:
            self._snapshot(self.controller.last_snapshot)
        else:
            self.metrics.setText(tr("Time —    DP —    Life —    Kills —    Leaks —"))
            self.action_count.setText(tr("{count} legal actions").format(count=0))
        self._status(self.controller.status)
