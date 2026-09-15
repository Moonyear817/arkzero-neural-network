"""Browse every offline map without pretending all mechanics can be simulated."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog,QDialogButtonBox,QHBoxLayout,QLabel,QLineEdit,QListWidget,QListWidgetItem,QPlainTextEdit,QSplitter,QVBoxLayout,QWidget
from arknights_sim.data.map_catalog import MapCatalog
from desktop.models.simulation_state import SimulationSnapshot, TileSnapshot
from desktop.widgets.stage_map import StageMap


def preview(data, stage_id):
    def point(value,offset=None):
        offset=offset or {}
        return (value['col']+offset.get('x',0),value['row']+offset.get('y',0))
    lines=[]
    for route in data['routes']:
        if not route:continue
        if not route.get('startPosition') or not route.get('endPosition'):continue
        lines.append((point(route['startPosition'],route.get('spawnOffset')),
                      *(point(w['position'],w.get('reachOffset')) for w in route.get('checkpoints') or [] if w.get('position') and w.get('type')=='MOVE'),point(route['endPosition'])))
    grid=data['map'];options=data['options']
    return SimulationSnapshot(stage_id,0,options.get('initialCost',0),options.get('maxLifePoint',0),0,0,sum(g['action'].get('count',0) for g in data['groups'] if g['action'].get('actionType')=='SPAWN'),False,'PREVIEW',tuple(tuple(TileSnapshot(t.key,t.height,t.buildable,t.passable) for t in row) for row in grid.rows),tuple(lines))


def describe(record,data):
    difficulty={'NORMAL':'普通','FOUR_STAR':'突袭','EASY':'简单','ALL':'全部'}.get(record['difficulty'],record['difficulty'])
    lines=[f"{record['code']} · {record['name']} · {difficulty}",record.get('description',''),
           f"地图 {record['width']}×{record['height']} · {record['waves']} 波 · {record['routes']} 条路线",
           '可运行当前基础战斗模拟' if record['simulation_supported'] else '已识别，可查看资料；战斗未实现：'+record['support_reason'],
           '路线连线为起点、检查点、终点示意；不是最终绕障路径。',
           '出怪顺序保留波次、片段及触发条件；存在清场、随机或脚本条件时，不虚构固定秒数。',
           '\n敌人属性：已合并数据库等级和关卡单体覆写；难度/环境词条另列，不把它们当作已应用。']
    if record.get('fidelity')=='verification':
        lines.insert(1,'机制验证版：可操作障碍物和侦测器，已接入隐匿、远程攻击、溅射与本关突袭词条。尚未完成原版逐项对照，不能作为原版等价结果。')
        lines[0]+=' · 等待验证'
    for enemy in data['enemies']:
        if enemy['error']:
            lines.append(f"{enemy['id']}：无法解析属性：{enemy['error']}");continue
        d=enemy['resolved'];a=d.get('attributes',{})
        lines.append(f"{d.get('name',enemy['id'])} [{enemy['id']}] 等级{enemy['level']}\n生命 {a.get('maxHp','未知')} / 攻击 {a.get('atk','未知')} / 防御 {a.get('def','未知')} / 法抗 {a.get('magicResistance','未知')}\n移动速度 {a.get('moveSpeed','未知')} / 攻击间隔 {a.get('baseAttackTime','未知')} / 攻速 {a.get('attackSpeed','未知')}")
    lines.append('\n出怪与事件（时间均相对所属波次/片段）')
    for group in data['groups']:
        a=group['action'];kind=a.get('actionType','')
        flags={k:a[k] for k in ('blockFragment','hiddenGroup','randomSpawnGroupKey','randomSpawnGroupPackKey','randomType') if a.get(k) and a.get(k)!='ALWAYS'}
        lines.append(f"波{group['wave']+1} / 片段{group['fragment']+1} / 事件{group['order']+1}：{kind} {a.get('key','')} ×{a.get('count',0)}\n波前延迟 {group['wave_pre_delay']}秒 / 片段前延迟 {group['fragment_pre_delay']}秒 / 事件前延迟 {a.get('preDelay',0)}秒 / 间隔 {a.get('interval',0)}秒 / 路线 {a.get('routeIndex','—')}"+(f"\n触发条件：{flags}" if flags else ''))
    if data['runes'] or data['global_buffs'] or data['branches']:
        import json
        lines+=['\n难度词条、环境效果与分支（原始条件保留）',json.dumps({'runes':data['runes'],'globalBuffs':data['global_buffs'],'branches':data['branches']},ensure_ascii=False,indent=2)]
    lines += ['\n来源：'+record['source_url']]
    return '\n\n'.join(lines)


class MapPicker(QDialog):
    def __init__(self,data_dir,parent=None):
        super().__init__(parent)
        self.catalog=MapCatalog(data_dir);self.selected_id=None
        self.setWindowTitle('本地地图库 · PRTS.Map');self.resize(1180,800)
        layout=QVBoxLayout(self)
        self.search=QLineEdit();self.search.setPlaceholderText('搜索关卡编号、名称、活动或内部 ID')
        layout.addWidget(self.search)
        layout.addWidget(QLabel(f"{len(self.catalog.records)} 个关卡/难度条目 · 地图与敌人资料均在本地"))
        split=QSplitter();self.entries=QListWidget();split.addWidget(self.entries)
        right=QWidget();column=QVBoxLayout(right);self.map=StageMap();self.details=QPlainTextEdit();self.details.setReadOnly(True)
        column.addWidget(self.map,1);column.addWidget(self.details,1);split.addWidget(right);split.setStretchFactor(1,3);layout.addWidget(split,1)
        self.buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText('选此关卡进行模拟')
        self.buttons.accepted.connect(self.accept);self.buttons.rejected.connect(self.reject);layout.addWidget(self.buttons)
        self.entries.currentItemChanged.connect(self._select);self.search.textChanged.connect(self._filter);self._filter()

    def _filter(self):
        self.entries.clear();query=self.search.text().strip().casefold()
        for key,r in self.catalog.records.items():
            if query not in f"{key} {r['code']} {r['name']} {r['zone_id']}".casefold():continue
            difficulty={'NORMAL':'普通','FOUR_STAR':'突袭','EASY':'简单','ALL':'全部'}.get(r['difficulty'],r['difficulty'])
            availability='机制验证' if r.get('fidelity')=='verification' else '可模拟' if r['simulation_supported'] else '资料预览'
            item=QListWidgetItem(f"{r['code']} · {r['name']} · {availability} · {difficulty}")
            item.setData(Qt.ItemDataRole.UserRole,key);self.entries.addItem(item)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        if self.entries.count():self.entries.setCurrentRow(0)
        else:self.details.clear();self.map.set_snapshot(None)

    def _select(self,item,_previous):
        if item is None:return
        self.selected_id=item.data(Qt.ItemDataRole.UserRole)
        try:
            record,data=self.catalog.inspect(self.selected_id)
            self.map.set_snapshot(preview(data,self.selected_id));self.details.setPlainText(describe(record,data))
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(record['simulation_supported'])
        except Exception as exc:
            self.details.setPlainText('读取失败：'+str(exc));self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
