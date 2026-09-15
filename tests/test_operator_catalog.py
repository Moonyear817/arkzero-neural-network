"""PRTS coverage, gameplay actions, healing semantics and model integration."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from arknights_sim import Simulator
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from arknights_sim.environment import Action, ArknightsEnv
from network import StateEncoder, ActionEncoder, NeuralEvaluator, PolicyValueNetwork
from network.state_encoder import OPERATOR_FEATURE_NAMES

DATA = Path(__file__).resolve().parents[1] / 'data/real'


@pytest.fixture(scope='module')
def loader():
    return OperatorLoader(DATA / 'character_table.json', DATA / 'range_table.json', SkillLoader(DATA / 'skill_table.json'))


def test_complete_source_and_identity(loader):
    catalog = json.loads((DATA / 'operator_catalog.json').read_text())
    assert catalog['count'] == len(catalog['operators']) == 431
    assert not catalog['unmatched']
    assert len(loader.available_ids()) == 431
    assert all(r['revision_id'] and r['wikitext'] and r['game_data_available'] for r in catalog['operators'])
    assert loader.resolve('能天使') == loader.resolve('Exusiai') == 'char_103_angel'
    assert loader.resolve('阿米娅(医疗)') == 'char_1037_amiya3'
    assert loader.resolve('阿米娅(近卫)') == 'char_1001_amiya2'
    assert loader.resolve('阿米娅') == 'char_002_amiya'
    with pytest.raises(ValueError, match='Ambiguous'):
        loader.resolve('Amiya')
    with pytest.raises(ValueError, match='Unknown'):
        loader.resolve('missing-character')


def test_every_imported_operator_deploys_encodes_retires(loader, real_stage):
    stage = replace(real_stage, initial_cost=99)
    encoder, actions = StateEncoder(), ActionEncoder()
    for key in loader.available_ids():
        op = loader.load(key)
        assert op.hp > 0 and op.interval > 0 and op.attack_range
        env = ArknightsEnv(stage=stage, squad=[op])
        initial = env.reset()
        deploy = next(a for a in env.legal_actions(initial) if a.type == 'DEPLOY')
        state = env.step(initial, deploy)
        assert state.game.operators[key].alive
        assert not initial.game.operators
        legal = env.legal_actions(state)
        assert all(torch.isfinite(v).all() for v in encoder(state).values())
        assert torch.isfinite(actions(state, legal)).all()
        state = env.step(state, Action('RETREAT', key))
        assert not state.game.operators[key].alive
        assert state.game.redeploy_at[key] > state.game.current_time


def test_level_bounds_and_skill_support(loader):
    op = loader.load('玫兰莎', elite=1, level=55)
    assert op.hp == loader.chars[op.id]['phases'][1]['attributesKeyFrames'][-1]['data']['maxHp']
    with pytest.raises(ValueError):
        loader.load('玫兰莎', elite=2)
    with pytest.raises(ValueError):
        loader.load('玫兰莎', level=999)
    assert loader.load('芬').skill is None
    assert any('技能' in n for n in loader.load('芬').simulation_notes)
    assert loader.load('黑角').simulation_notes == ()
    assert loader.load('玫兰莎').skill.attack_multiplier == pytest.approx(1.1)


def test_medic_heals_allies_and_never_damages_enemies(loader, simple):
    medic = replace(loader.load('安赛尔'), position_type='MELEE', cost=0)
    tank = replace(loader.load('黑角'), cost=0)
    sim = Simulator(simple, [medic, tank], trace=True)
    sim.deploy(medic.id, (2, 0))
    target = sim.deploy(tank.id, (3, 0))
    target.hp -= 200
    sim.run_until(0.5)
    assert target.hp == pytest.approx(tank.hp - 200 + medic.atk)
    events = sim.trace.events
    assert any(r['event'] == 'HEAL' for r in events)
    assert not any(r['event'] == 'DAMAGE' and r.get('attacker') == medic.id for r in events)


def test_new_roles_and_twelve_member_neural_policy(loader, real_stage):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        keys = ['阿米娅', '安赛尔', '浊心斯卡蒂', '能天使', '黑角', '玫兰莎', '芬', '德克萨斯', '星熊', '艾雅法拉', '银灰', '斯卡蒂']
        squad = [loader.load(k) for k in keys]
        env = ArknightsEnv(stage=replace(real_stage, initial_cost=99), squad=squad)
        state = env.reset()
        encoded = StateEncoder()(state)
        assert encoded['operator_mask'].sum() == 12
        index = OPERATOR_FEATURE_NAMES.index('support_role')
        ids = sorted(state.game.squad)
        assert encoded['operators'][ids.index(loader.resolve('安赛尔')), index] == 1
        assert encoded['operators'][ids.index(loader.resolve('浊心斯卡蒂')), index] == -1
        model = PolicyValueNetwork()
        priors, value = NeuralEvaluator(model).evaluate(env, state, env.legal_actions(state))
        assert len(priors) == len(env.legal_actions(state))
        assert sum(priors) == pytest.approx(1, abs=1e-5)
        assert -1 <= value <= 1
    finally:
        torch.set_num_threads(previous)


def test_picker_selection_survives_search(qtbot, loader):
    from PySide6.QtCore import Qt
    from desktop.widgets.squad_picker import SquadPicker
    picker = SquadPicker(DATA, ['char_103_angel'])
    qtbot.addWidget(picker)
    picker.search.setText('char_103_angel')
    assert len(picker.panel.checks) == 1
    assert picker.panel.checks['char_103_angel'].isChecked()
    picker.search.setText('安赛尔')
    picker.panel.checks[loader.resolve('安赛尔')].click()
    picker.search.clear()
    assert len(picker.panel.checks) == 431
    assert set(picker.squad()) == {'char_103_angel', loader.resolve('安赛尔')}
