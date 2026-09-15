import json
from pathlib import Path
import pytest
from accounts.snapshot import confirm_snapshot,load_snapshot,validate_record
from accounts.ocr import draft_from_lines
from accounts.readiness import curriculum_audit
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader

DATA=Path(__file__).resolve().parents[1]/'data/real'
@pytest.fixture(scope='module')
def loader():return OperatorLoader(DATA/'character_table.json',DATA/'range_table.json',SkillLoader(DATA/'skill_table.json'))
@pytest.fixture
def record(loader):
    key=loader.resolve('玫兰莎')
    return dict(operator_id=key,owned=True,elite=0,level=1,potential=0,trust=0,skill_rank=1,
                masteries={s['skillId']:0 for s in loader.chars[key]['skills']},module={'id':'','level':0})


def test_snapshot_versions_integrity_and_confirmation(loader,record,tmp_path):
    with pytest.raises(ValueError):confirm_snapshot({'operators':[record]},loader,tmp_path,confirmed_by='')
    p=confirm_snapshot({'operators':[record]},loader,tmp_path,confirmed_by='test reviewer')
    first=load_snapshot(p,loader);assert first['operators'][0]==record
    record['level']=2
    q=confirm_snapshot({'operators':[record]},loader,tmp_path,confirmed_by='test reviewer')
    assert p!=q and load_snapshot(p,loader)==first
    content=json.loads(p.read_text());content['operators'][0]['level']=3;p.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='integrity'):load_snapshot(p,loader)


@pytest.mark.parametrize('field,value',[('elite',None),('owned',None),('level',100),('potential',6),('trust',201),('masteries',{}),('module',None)])
def test_missing_invalid_values_never_default(loader,record,field,value):
    record[field]=value
    with pytest.raises(ValueError):validate_record(record,loader)


def test_ocr_does_not_infer_ownership_potential_or_missing_fields(loader):
    rows=[{'text':t,'confidence':.99} for t in ['玫兰莎','等级 30','精英 0','信赖 90%']]
    result=draft_from_lines(rows,loader,{'path':'synthetic'})
    r=result['operators'][0]
    assert r['level']==30 and r['trust']==90
    assert r['owned'] is None and r['potential'] is None and r['module'] is None
    with pytest.raises(ValueError,match='待确认'):validate_record(r,loader)
    rows.append({'text':'等级 10','confidence':.99})
    assert draft_from_lines(rows,loader,{})['operators'][0]['level'] is None


def test_unowned_and_non_guard(loader,record,tmp_path):
    record['owned']=False
    path=confirm_snapshot({'operators':[record]},loader,tmp_path,confirmed_by='test')
    report=curriculum_audit(DATA,load_snapshot(path,loader),loader)
    assert not report['operators'][0]['trainable']
    assert 'NOT_OWNED' in report['operators'][0]['reasons']
    record['operator_id']=loader.resolve('黑角')
    with pytest.raises(ValueError,match='guard'):validate_record(record,loader)


def test_chapter_scope_and_actual_loader_audit():
    report=curriculum_audit(DATA)
    assert report['stages'][0]['code']=='R8-1'
    assert not any(r['code'].startswith('H8') for r in report['stages'])
    assert not report['formal_training_ready']
    assert all(r['reason'] for r in report['stages'])


def test_trainer_rejects_missing_snapshot_before_writing(tmp_path):
    from training.trainer import AlphaZeroTrainer
    with pytest.raises(ValueError,match='账号快照'):
        AlphaZeroTrainer({'training_scope':'account_guards_chapter8','checkpoint_dir':str(tmp_path/'ckpt'),
                          'output_dir':str(tmp_path/'out')})
    assert not list(tmp_path.iterdir())


def test_account_dialog_preserves_unknowns_and_draft(qtbot,record):
    from desktop.widgets.account_dialog import AccountDialog
    dialog=AccountDialog();qtbot.addWidget(dialog)
    record['trust']=None
    dialog.apply_draft({'operators':[record],'sources':[{'path':'example'}]})
    assert dialog.draft()['operators'][0]==record
    assert dialog.table.item(0,5).text()==''
    assert dialog.draft()['sources']==[{'path':'example'}]


def test_guard_config_defaults_and_formal_gate(tmp_path):
    from training.config import load_config
    from training.trainer import AlphaZeroTrainer
    config=load_config(Path(__file__).resolve().parents[1]/'configs/guards_chapter8.yaml')
    assert config['mcts_simulations']==16 and config['num_threads']==1
    assert config['stage']=='R8-1' and not config['future_events_enabled']
    with pytest.raises(ValueError,match='账号快照'):AlphaZeroTrainer(config)


def test_desktop_account_config_preserves_empty_pending_squad_and_path(tmp_path):
    from desktop.services.config_service import ConfigService
    service=ConfigService(tmp_path)
    config=service.validate({'training_scope':'account_guards_chapter8','squad':[],
                             'account_snapshot':'data/accounts/snapshots/pending.json'})
    prepared=service.prepare_runtime(config)
    assert prepared['squad']==[]
    assert prepared['account_snapshot']==str(tmp_path/'data/accounts/snapshots/pending.json')


def test_video_review_preserves_skill_module_evidence_when_editing(qtbot,record):
    from desktop.widgets.account_dialog import AccountDialog
    from PySide6.QtWidgets import QTableWidgetItem
    dialog=AccountDialog();qtbot.addWidget(dialog)
    record.update(name='玫兰莎',selected_skill_id=next(iter(record['masteries'])),
                  evidence={'frame':'video/frame.png','time_seconds_approx':2.5},
                  observed_modules=[],review_status='AWAITING_USER_CONFIRMATION')
    dialog.apply_draft({'operators':[record],'status':'DRAFT_REQUIRES_HUMAN_CONFIRMATION'})
    assert dialog.table.item(0,0).text()=='玫兰莎'
    dialog.table.setItem(0,5,QTableWidgetItem('200'))
    saved=dialog.draft()
    assert saved['operators'][0]=={**record,'trust':200}
    assert saved['status']=='DRAFT_REQUIRES_HUMAN_CONFIRMATION'


def test_selected_skill_must_be_unlocked(loader,record):
    record['selected_skill_id']='skchr_surtr_3'
    with pytest.raises(ValueError,match='Selected skill'):validate_record(record,loader)
    key=loader.resolve('史尔特尔');raw=loader.chars[key]
    record.update(operator_id=key,masteries={s['skillId']:0 for s in raw['skills']},
                  selected_skill_id=raw['skills'][2]['skillId'])
    with pytest.raises(ValueError,match='locked'):validate_record(record,loader)


def test_module_source_and_favor_conversion(loader,record):
    key=loader.resolve('史尔特尔');raw=loader.chars[key]
    record.update(operator_id=key,elite=2,level=90,skill_rank=7,
                  masteries={s['skillId']:0 for s in raw['skills']},
                  module={'id':'uniequip_002_surtr','level':3},trust=200)
    validate_record(record,loader)
    record['trust']=0
    with pytest.raises(ValueError,match='trust requirement'):validate_record(record,loader)
    record['trust']=200;record['level']=1
    with pytest.raises(ValueError,match='not unlocked'):validate_record(record,loader)
    record['level']=90;record['module']['id']='uniequip_002_mlynar'
    with pytest.raises(ValueError,match='belong'):validate_record(record,loader)


def test_account_reference_cannot_silently_use_default_training():
    from accounts.readiness import require_training_ready
    for field in ('account_snapshot','account_draft'):
        with pytest.raises(ValueError,match='默认练度'):
            require_training_ready({field:'account.json','stage':'0-1'})


def test_confirmation_preserves_source_but_updates_review_status(loader,record,tmp_path):
    record.update(review_status='AWAITING_USER_CONFIRMATION',evidence={'frame':'original.png'})
    path=confirm_snapshot({'operators':[record]},loader,tmp_path,confirmed_by='test human')
    saved=load_snapshot(path,loader)['operators'][0]
    assert saved['review_status']=='CONFIRMED_BY_USER'
    assert saved['evidence']==record['evidence']
    assert record['review_status']=='AWAITING_USER_CONFIRMATION'
