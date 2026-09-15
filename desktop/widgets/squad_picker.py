"""Profession-organized operator cards with real, fully clickable checkboxes."""
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox,QDialog,QDialogButtonBox,QGridLayout,QHBoxLayout,QLabel,QLineEdit,QPushButton,QScrollArea,QTabBar,QVBoxLayout,QWidget)
from arknights_sim.data.operator_loader import OperatorLoader
from desktop.paths import DATA_DIR

PROFESSIONS=(('', '全部'),('PIONEER','先锋'),('WARRIOR','近卫'),('TANK','重装'),('SNIPER','狙击'),('CASTER','术师'),('MEDIC','医疗'),('SUPPORT','辅助'),('SPECIAL','特种'))


class OperatorCard(QCheckBox):
    def hitButton(self, position):
        return self.rect().contains(position)


class RosterPanel(QWidget):
    selection_changed=Signal(object)

    def __init__(self,data_dir=None,selected=(),parent=None):
        super().__init__(parent)
        directory=Path(data_dir or DATA_DIR)
        try:
            self.loader=OperatorLoader(directory/'character_table.json',directory/'range_table.json')
        except OSError:
            self.loader=None
        self.selected=set(selected)
        self.checks={}
        layout=QVBoxLayout(self)
        self.search=QLineEdit();self.search.setPlaceholderText('搜索干员中文名、英文名、日文名或编号')
        layout.addWidget(self.search)
        self.tabs=QTabBar()
        for key,name in PROFESSIONS:self.tabs.addTab(name)
        layout.addWidget(self.tabs)
        top=QHBoxLayout();self.count=QLabel();top.addWidget(self.count,1)
        clear=QPushButton('清空编队');clear.clicked.connect(lambda:self.set_squad(()));top.addWidget(clear);layout.addLayout(top)
        self.summary=QLabel();self.summary.setWordWrap(True);layout.addWidget(self.summary)
        scroll=QScrollArea();scroll.setWidgetResizable(True);body=QWidget();self.grid=QGridLayout(body)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop);scroll.setWidget(body);layout.addWidget(scroll,1)
        self.search.textChanged.connect(self._filter);self.tabs.currentChanged.connect(self._filter)
        self._filter()

    def set_squad(self,keys):
        if len(set(keys))>12:raise ValueError('队伍最多12人')
        self.selected=set(keys);self._filter();self.selection_changed.emit(self.squad())

    def squad(self):return tuple(sorted(self.selected))

    def _filter(self,*_):
        while self.grid.count():
            item=self.grid.takeAt(0)
            if item.widget():item.widget().deleteLater()
        self.checks={}
        if self.loader is None:
            self.count.setText('干员库尚不可用，请检查设置中的数据目录')
            self.summary.setText('')
            return
        profession=PROFESSIONS[self.tabs.currentIndex()][0]
        keys=[k for k in self.loader.search(self.search.text()) if not profession or self.loader.chars[k]['profession']==profession]
        keys.sort(key=lambda k:(-int(self.loader.chars[k]['rarity'].split('_')[-1]),self.loader.catalog.get(k,{}).get('name',self.loader.chars[k]['name'])))
        for index,key in enumerate(keys):
            raw=self.loader.chars[key];record=self.loader.catalog.get(key,{});wiki=record.get('wiki',{})
            name=record.get('name',raw['name']);stars=int(raw['rarity'].split('_')[-1])
            check=OperatorCard(f"{name}　{stars}★\n{wiki.get('subprofession',raw['subProfessionId'])}")
            check.setMinimumHeight(60);check.setChecked(key in self.selected)
            check.setAccessibleName(name);check.setToolTip(f"{wiki.get('en','')}\n{wiki.get('trait','')}\n{key}")
            check.toggled.connect(lambda checked,k=key:self._toggle(k,checked))
            self.checks[key]=check;self.grid.addWidget(check,index//3,index%3)
        self._count()

    def _toggle(self,key,checked):
        if checked and key not in self.selected and len(self.selected)>=12:
            self.checks[key].blockSignals(True);self.checks[key].setChecked(False);self.checks[key].blockSignals(False)
            self.count.setText('最多选择12人，请先取消一名干员');return
        if checked:self.selected.add(key)
        else:self.selected.discard(key)
        self._count();self.selection_changed.emit(self.squad())

    def _count(self):
        self.count.setText(f'已选 {len(self.selected)} / 12 人 · 当前显示 {len(self.checks)} / {len(self.loader.available_ids())} 人')
        names=[self.loader.catalog.get(k,{}).get('name',self.loader.chars[k]['name']) for k in sorted(self.selected) if k in self.loader.chars]
        self.summary.setText('当前编队：'+('、'.join(names) if names else '尚未选择干员'))


class SquadPicker(QDialog):
    def __init__(self,data_dir=None,selected=(),parent=None):
        super().__init__(parent);self.setWindowTitle('选择干员 · 按职业编队');self.resize(960,760)
        layout=QVBoxLayout(self);self.panel=RosterPanel(data_dir,selected,self);layout.addWidget(self.panel,1)
        self.loader=self.panel.loader;self.search=self.panel.search
        note=QLabel('对战养成请在“干员养成”中设置；训练使用所选训练配置。未实现的技能、天赋和模组不会视作已还原。');note.setWordWrap(True);layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('使用此编队');buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    @property
    def selected(self):return self.panel.selected
    @selected.setter
    def selected(self,value):self.panel.set_squad(value)
    def _filter(self):self.panel._filter()
    def squad(self):return self.panel.squad()
