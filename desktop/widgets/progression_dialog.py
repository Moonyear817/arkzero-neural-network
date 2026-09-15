"""Progression configuration only; preview parsing runs in its own worker."""
from dataclasses import replace
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QAbstractItemView
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                              QLabel, QPushButton, QSpinBox, QTableWidget,
                              QTableWidgetItem, QVBoxLayout)
from arknights_sim.data.progression import Progression, load_progressed
from desktop.i18n import bind_text, register_translations, tr
from desktop.paths import DATA_DIR

register_translations({
    'Operator progression': '干员养成', 'All operators: level': '全员等级',
    'E0 Lv1': '精零 1 级', 'E1 Lv1': '精一 1 级', 'E2 Lv1': '精二 1 级',
    'E2 Lv90': '精二 90 级（按各干员上限）', 'Custom level': '自定义等级',
    'Elite phase': '精英阶段', 'Level': '等级', 'Potential upgrades': '潜能提升',
    'No upgrades (in-game P1)': '无提升（游戏潜能 1）',
    '{n} upgrades (in-game P{p})': '提升 {n} 次（游戏潜能 {p}）',
    'Skill rank / mastery': '技能等级 / 专精', 'Mastery {n}': '专精 {n}',
    'Equipped skill': '全员携带技能', 'Operator': '干员',
    'Follow global skill': '跟随全员技能', 'Preview actual stats': '预览实际属性',
    'Apply': '应用', 'Cancel': '取消', 'Loading…': '加载中…',
    'Actual stats / limitations': '实际属性 / 限制',
    'Targets are capped per operator. Lower promotion ceilings use that operator’s maximum level. Locked/missing skills fall back to the first unlocked skill.': '按各干员实际上限应用；无法达到目标精英阶段时使用该干员最高阶段满级。不存在或未解锁的携带技能改用首个已解锁技能。',
    'Potential counts upgrades, not the in-game number. Trust is 0; unsupported talents and special skills remain disabled and are listed in preview.': '潜能按提升次数计数，对应游戏潜能 1–6。信赖为 0；未实现的天赋和特殊技能不会假装生效，详见预览。',
    'Global targets also apply to model-selected squads. Individual skill choices below apply when that operator is selected.': '全员设置同样适用于模型自主编队。下方可单独指定干员携带的技能，入队时生效。',
})


class ProgressionPreview(QThread):
    ready = Signal(object)
    error = Signal(str)

    def __init__(self, directory, keys, request, overrides, parent=None):
        super().__init__(parent)
        self.directory, self.keys, self.request = Path(directory or DATA_DIR), tuple(keys), request
        self.overrides = dict(overrides)

    def run(self):
        try:
            from arknights_sim.data.operator_loader import OperatorLoader
            from arknights_sim.data.skill_loader import SkillLoader
            loader = OperatorLoader(self.directory/'character_table.json', self.directory/'range_table.json', SkillLoader(self.directory/'skill_table.json'))
            rows = []
            for key in self.keys:
                op = load_progressed(loader, key, replace(self.request, skill_index=self.overrides.get(key, self.request.skill_index)))
                descriptions = []
                for skill in loader.chars[op.id]['skills']:
                    raw = loader.skills.data[skill['skillId']]['levels']
                    descriptions.append(raw[min(op.skill_level, len(raw))-1]['name'])
                rows.append((op, tuple(descriptions)))
            self.ready.emit(tuple(rows))
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


class ProgressionDialog(QDialog):
    def __init__(self, directory=None, keys=(), request=None, overrides=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('Operator progression'))
        self.resize(840, 650)
        self.directory, self.keys = directory, tuple(keys)
        self._overrides = dict(overrides or {})
        self._worker = None
        request = request or Progression()
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.preset = QComboBox()
        presets = [((0,1),'E0 Lv1'), ((1,1),'E1 Lv1'), ((2,1),'E2 Lv1'), ((2,90),'E2 Lv90'), (None,'Custom level')]
        for data, label in presets:
            self.preset.addItem(tr(label), data)
        form.addRow(tr('All operators: level'), self.preset)
        self.elite = QSpinBox(); self.elite.setRange(0,2); self.elite.setValue(request.elite)
        self.level = QSpinBox(); self.level.setRange(1,90); self.level.setValue(request.level)
        form.addRow(tr('Elite phase'),self.elite); form.addRow(tr('Level'),self.level)
        self.potential = QComboBox()
        for n in range(6):
            self.potential.addItem(tr('No upgrades (in-game P1)') if not n else tr('{n} upgrades (in-game P{p})').format(n=n,p=n+1), n)
        self.potential.setCurrentIndex(request.potential)
        form.addRow(tr('Potential upgrades'),self.potential)
        self.skill_level = QComboBox()
        for n in range(1,11):
            self.skill_level.addItem(f'Rank {n}' if n<=7 else tr('Mastery {n}').format(n=n-7),n)
        self.skill_level.setCurrentIndex(request.skill_level-1)
        form.addRow(tr('Skill rank / mastery'),self.skill_level)
        self.skill_index = QComboBox()
        for n in range(3): self.skill_index.addItem(f'S{n+1}',n)
        self.skill_index.setCurrentIndex(request.skill_index)
        form.addRow(tr('Equipped skill'),self.skill_index)
        layout.addLayout(form)
        for text in ('Targets are capped per operator. Lower promotion ceilings use that operator’s maximum level. Locked/missing skills fall back to the first unlocked skill.', 'Potential counts upgrades, not the in-game number. Trust is 0; unsupported talents and special skills remain disabled and are listed in preview.', 'Global targets also apply to model-selected squads. Individual skill choices below apply when that operator is selected.'):
            label=QLabel(tr(text));label.setWordWrap(True);layout.addWidget(label)
        self.table=QTableWidget(len(keys),3)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setHorizontalHeaderLabels([tr('Operator'),tr('Equipped skill'),tr('Actual stats / limitations')])
        self.table.setColumnWidth(0,150);self.table.setColumnWidth(1,160)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.choices={}
        for row,key in enumerate(keys):
            self.table.setItem(row,0,QTableWidgetItem(key))
            choice=QComboBox();choice.addItem(tr('Follow global skill'),None)
            for n in range(3):choice.addItem(f'S{n+1}',n)
            choice.setCurrentIndex(choice.findData(self._overrides.get(key)))
            self.table.setCellWidget(row,1,choice);self.choices[key]=choice
            choice.currentIndexChanged.connect(self.invalidate_preview)
        layout.addWidget(self.table,1)
        self.preview=bind_text(QPushButton(),'Preview actual stats');self.preview.clicked.connect(self.refresh_preview)
        layout.addWidget(self.preview)
        self.message=QLabel();self.message.setWordWrap(True);layout.addWidget(self.message)
        self.buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('Apply'))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('Cancel'))
        self.buttons.accepted.connect(self.accept);self.buttons.rejected.connect(self.reject);layout.addWidget(self.buttons)
        match=next((i for i,(v,_) in enumerate(presets) if v==(request.elite,request.level)),4)
        self.preset.setCurrentIndex(match)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        for control in (self.elite,self.level):control.valueChanged.connect(self._custom_changed)
        for control in (self.potential,self.skill_level,self.skill_index):control.currentIndexChanged.connect(self.invalidate_preview)

    def _preset_changed(self):
        value=self.preset.currentData()
        if value:
            for widget,n in zip((self.elite,self.level),value):
                widget.blockSignals(True);widget.setValue(n);widget.blockSignals(False)
        self.invalidate_preview()

    def _custom_changed(self):
        self.preset.blockSignals(True);self.preset.setCurrentIndex(4);self.preset.blockSignals(False)
        self.invalidate_preview()

    def request(self):
        return Progression(self.elite.value(),self.level.value(),self.potential.currentData(),self.skill_level.currentData(),self.skill_index.currentData())

    def overrides(self):
        result=dict(self._overrides)
        for key,choice in self.choices.items():
            if choice.currentData() is None:result.pop(key,None)
            else:result[key]=choice.currentData()
        return result

    def invalidate_preview(self,*_):
        for row in range(self.table.rowCount()):self.table.setItem(row,2,QTableWidgetItem('—'))

    def refresh_preview(self):
        if self._worker is not None:return
        self._preview_request=(self.request(),self.overrides())
        self._worker=ProgressionPreview(self.directory,self.keys,*self._preview_request,self)
        self._worker.ready.connect(self._ready)
        self._worker.error.connect(self.message.setText)
        self._worker.finished.connect(self._finished)
        self.message.setText(tr('Loading…'));self.preview.setEnabled(False)
        self._worker.start()

    def _ready(self,rows):
        if self._preview_request!=(self.request(),self.overrides()):return
        for row,(op,names) in enumerate(rows):
            self.table.setItem(row,0,QTableWidgetItem(op.name))
            choice=self.choices[op.id]
            for n in range(3):choice.setItemText(n+1,f'S{n+1} · {names[n]}' if n<len(names) else f'S{n+1} · —')
            text=f'E{op.elite} Lv{op.level} · P{op.potential+1} · S{op.skill_index+1} Lv{op.skill_level}\nHP {op.hp:g} ATK {op.atk:g} DEF {op.defense:g} DP {op.cost:g}\n'+'\n'.join(op.simulation_notes)
            item=QTableWidgetItem(text);item.setToolTip(text);self.table.setItem(row,2,item)
        self.table.resizeRowsToContents();self.message.clear()

    def _finished(self):
        self._worker.deleteLater();self._worker=None;self.preview.setEnabled(True)

    def done(self,result):
        if self._worker is not None:
            self.message.setText(tr('Loading…'));return
        super().done(result)
