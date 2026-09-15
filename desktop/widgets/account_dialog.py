"""Screenshot-assisted account review. OCR and validation run off the UI thread."""
import json
from copy import deepcopy
from pathlib import Path
from PySide6.QtCore import QThread,Signal,Qt,QUrl
from PySide6.QtGui import QDesktopServices,QColor
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QPushButton,QFileDialog,QLabel,
                               QTableWidget,QTableWidgetItem,QMessageBox,QPlainTextEdit)
from desktop.paths import DATA_DIR,WORKSPACE_ROOT
from desktop.i18n import tr,register_translations

register_translations({'Account guards':'账号近卫','Import screenshot':'导入截图',
 'Import draft':'导入草稿','Save draft':'保存草稿','Confirm account snapshot':'确认账号快照',
 'Account snapshot saved':'账号快照已保存','Account review failed':'账号核对失败',
 'Unknown fields stay blank. Potential is upgrades 0–5. Masteries map skill IDs to 0–3; no module: {"id":"","level":0}.':'未知字段保持空白。潜能填提升次数0–5；专精按技能ID填写0–3；未装备模组填 {"id":"","level":0}。',
 'Account review…':'账号截图与核对…','Add guard':'添加近卫','Operator ID':'干员ID',
 'Remove selected row':'删除选中行',
 'Owned':'已持有','Elite':'精英','Level':'等级','Potential upgrades':'潜能提升次数',
 'Trust %':'信赖%','Skill rank':'技能等级','Masteries':'各技能专精','Module':'装备模组'})
register_translations({'Video roster':'视频整理结果','Open full roster':'查看完整角色清单',
 'No video draft':'当前没有待核对的视频草稿','Operator':'干员',
 'Selected row details':'选中干员的原始资料与证据'})
FIELDS=('operator_id','owned','elite','level','potential','trust','skill_rank','masteries','module')
LABELS=('Operator','Owned','Elite','Level','Potential upgrades','Trust %','Skill rank','Masteries','Module')


class AccountJob(QThread):
    result=Signal(object)
    failed=Signal(str)
    def __init__(self,operation,value,parent=None):
        super().__init__(parent);self.operation=operation;self.value=value
    def run(self):
        try:
            from arknights_sim.data.operator_loader import OperatorLoader
            from arknights_sim.data.skill_loader import SkillLoader
            loader=OperatorLoader(DATA_DIR/'character_table.json',DATA_DIR/'range_table.json',SkillLoader(DATA_DIR/'skill_table.json'))
            if self.operation=='recognize':
                from accounts.ocr import recognize
                result=recognize(self.value,loader)
            else:
                from accounts.snapshot import confirm_snapshot
                result=str(confirm_snapshot(self.value,loader,WORKSPACE_ROOT/'data/accounts/snapshots',confirmed_by='desktop-user'))
            self.result.emit(result)
        except Exception as exc:self.failed.emit(str(exc))


class AccountDialog(QDialog):
    snapshot_saved=Signal(str)
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle(tr('Account guards'));self.resize(1150,650)
        self.sources=[];self.worker=None;self._pending=None;self.document={};self.review_path=None
        layout=QVBoxLayout(self);layout.addWidget(QLabel(tr('Unknown fields stay blank. Potential is upgrades 0–5. Masteries map skill IDs to 0–3; no module: {"id":"","level":0}.')))
        bar=QHBoxLayout();self.buttons=[]
        for label,callback in [('Video roster',self.import_video_draft),('Import screenshot',self.import_image),('Import draft',self.import_draft),
                               ('Add guard',lambda:self.table.insertRow(self.table.rowCount())),
                               ('Remove selected row',lambda:self.table.removeRow(self.table.currentRow())),
                               ('Save draft',self.save_draft),('Confirm account snapshot',self.confirm)]:
            b=QPushButton(tr(label));b.clicked.connect(callback);bar.addWidget(b);self.buttons.append(b)
        layout.addLayout(bar)
        self.table=QTableWidget(0,len(FIELDS));self.table.setHorizontalHeaderLabels([tr(x) for x in LABELS]);layout.addWidget(self.table)
        self.evidence=QPlainTextEdit();self.evidence.setReadOnly(True);self.evidence.setMaximumHeight(170);layout.addWidget(self.evidence)
        self.table.currentCellChanged.connect(self.show_row_evidence)
        self.full_roster=QPushButton(tr('Open full roster'));self.full_roster.clicked.connect(self.open_full_roster)
        layout.addWidget(self.full_roster)
        self.status=QLabel();layout.addWidget(self.status)
    def apply_draft(self,draft):
        # Keep conflicting screenshots as separate review rows; never overwrite
        # existing values silently. Confirmation rejects duplicate operators.
        self.document.update({key:deepcopy(value) for key,value in draft.items()
                              if key not in ('operators','sources')})
        self.sources.extend(draft.get('sources',[]))
        for record in draft.get('operators',[]):
            row=self.table.rowCount();self.table.insertRow(row)
            for col,key in enumerate(FIELDS):
                value=record.get(key)
                text='' if value is None else (record.get('name') or value) if key=='operator_id' else json.dumps(value,ensure_ascii=False)
                item=QTableWidgetItem(text)
                if col==0:
                    item.setData(Qt.ItemDataRole.UserRole,deepcopy(record))
                    item.setToolTip(str(value))
                if value is None:item.setBackground(QColor('#fff0cb'))
                self.table.setItem(row,col,item)
        self.evidence.setPlainText(json.dumps(draft.get('ocr_lines',draft.get('warnings',[])),ensure_ascii=False,indent=2))
    def draft(self):
        records=[]
        for row in range(self.table.rowCount()):
            first=self.table.item(row,0)
            original=first.data(Qt.ItemDataRole.UserRole) if first else None
            record=deepcopy(original or {})
            for col,key in enumerate(FIELDS):
                item=self.table.item(row,col);text=item.text().strip() if item else ''
                if key=='operator_id':
                    record[key]=(original['operator_id'] if original and text in
                                 (original.get('name'),original.get('operator_id')) else text or None)
                else:record[key]=None if not text else json.loads(text)
            records.append(record)
        return {**self.document,'operators':records,'sources':self.sources}
    def show_row_evidence(self,row,*_):
        first=self.table.item(row,0)
        record=first.data(Qt.ItemDataRole.UserRole) if first else None
        if not record:return
        detail={k:record[k] for k in ('name','module_name','module_type','selected_skill_id',
                'observed_modules','evidence','warnings') if k in record}
        self.evidence.setPlainText(json.dumps(detail,ensure_ascii=False,indent=2))
    def import_video_draft(self):
        pointer=WORKSPACE_ROOT/'data/accounts/drafts/pending_video.json'
        if not pointer.exists():self.status.setText(tr('No video draft'));return
        try:
            entry=json.loads(pointer.read_text())
            path=WORKSPACE_ROOT/(entry.get('confirmed_snapshot') or entry['draft_path'])
            self.review_path=WORKSPACE_ROOT/entry['review_path']
            # An explicit video import appends like screenshot imports; it never
            # silently replaces edits already made in the review table.
            self.apply_draft(json.loads(path.read_text()))
            self.status.setText(('Confirmed account snapshot / 已确认的账号快照'
                                 if entry.get('confirmed_snapshot') else 'Review required / 请核对后确认'))
        except Exception as exc:self.error(str(exc))
    def open_full_roster(self):
        if not self.review_path:
            pointer=WORKSPACE_ROOT/'data/accounts/drafts/pending_video.json'
            if pointer.exists():self.review_path=WORKSPACE_ROOT/json.loads(pointer.read_text())['review_path']
        if self.review_path and self.review_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.review_path)))
    def start_job(self,operation,value):
        if self.worker:return
        self._pending=None;self.worker=AccountJob(operation,value,self)
        for b in self.buttons:b.setEnabled(False)
        self.worker.result.connect(lambda result:setattr(self,'_pending',(operation,result)))
        self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.finished_job);self.worker.start()
        self.status.setText('Processing… / 正在处理…')
    def finished_job(self):
        self.worker.deleteLater();self.worker=None
        for b in self.buttons:b.setEnabled(True)
        if self._pending:
            operation,result=self._pending
            if operation=='recognize':self.apply_draft(result);self.status.setText('Review required / 请核对后确认')
            else:self.status.setText(tr('Account snapshot saved')+': '+result);self.snapshot_saved.emit(result)
        self._pending=None
    def error(self,message):
        self.status.setText(message);QMessageBox.warning(self,tr('Account review failed'),message)
    def import_image(self):
        path,_=QFileDialog.getOpenFileName(self,tr('Import screenshot'),'','Images (*.png *.jpg *.jpeg *.heic)')
        if path:self.start_job('recognize',path)
    def import_draft(self):
        path,_=QFileDialog.getOpenFileName(self,tr('Import draft'),'','JSON (*.json)')
        if path:
            try:self.apply_draft(json.loads(Path(path).read_text()))
            except Exception as exc:self.error(str(exc))
    def save_draft(self):
        try:
            data=self.draft();path,_=QFileDialog.getSaveFileName(self,tr('Save draft'),'account_draft.json','JSON (*.json)')
            if path:Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        except Exception as exc:self.error(str(exc))
    def confirm(self):
        try:self.start_job('confirm',self.draft())
        except Exception as exc:self.error(str(exc))
    def reject(self):
        if not self.worker:super().reject()
    def closeEvent(self,event):
        if self.worker:event.ignore()
        else:super().closeEvent(event)
