import hashlib
import json
from pathlib import Path

import pytest
import torch
from arknights_sim.data.map_catalog import MapCatalog
from arknights_sim.environment import ArknightsEnv
from desktop.workers.simulation_worker import MODES, SimulationWorker
from training.squad_selection import SquadCoordinator
from training.trainer import AlphaZeroTrainer

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/real'


def test_all_map_files_present_verified_and_indexed():
    manifest=json.loads((ROOT/'data/maps/download_manifest.json').read_text())
    catalog=MapCatalog(DATA)
    assert manifest['complete'] and not manifest['errors']
    assert len(manifest['files'])==3371
    assert len(catalog.records)==4694
    assert catalog.metadata['unresolved_enemy_references']==0
    for row in manifest['files'].values():
        raw=(ROOT/row['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256']


def test_map_recognition_and_supported_environment():
    catalog=MapCatalog(DATA)
    record,data=catalog.inspect('main_00-01')
    assert (data['map'].width,data['map'].height)==(9,6)
    soldier=next(e for e in data['enemies'] if e['id']=='enemy_1002_nsabr')
    assert soldier['resolved']['attributes']['def']==30
    assert len([g for g in data['groups'] if g['action']['actionType']=='SPAWN'])>1
    assert ArknightsEnv(data_dir=DATA).reset('main_00-08').game.stage.id=='level_main_00-08'
    with pytest.raises(NotImplementedError):catalog.load_stage('main_00-01#f#')
    _,challenge=catalog.inspect('main_00-01#f#')
    assert challenge['runes']


def test_three_ui_modes_only():
    assert MODES==('Manual control','Automatic control','Hybrid control')


def test_hybrid_manual_action_invalidates_pending_ai_future():
    from arknights_sim.environment import Action
    worker=SimulationWorker(stage_id='0-1',squad=['char_500_noirc'],seed=1,mode='Hybrid control')
    worker._env=ArknightsEnv(data_dir=DATA)
    worker._state=worker._env.reset()
    worker._state=worker._env.advance_to_time(worker._state,5)
    action=next(a for a in worker._env.legal_actions(worker._state) if a.type=='DEPLOY')
    worker._pending_wait=worker._env.advance_to_time(worker._state,6)
    worker._apply_command(json.dumps(action.to_dict()))
    assert worker._pending_wait is None
    assert worker._state.game.operators[action.operator_id].alive
    worker.submit_action(json.dumps(Action('WAIT').to_dict()))
    assert worker._paused


def test_neural_squad_changes_parameters_and_is_reproducible():
    torch.set_num_threads(1)
    coordinator=SquadCoordinator(DATA,size=6)
    env=ArknightsEnv(data_dir=DATA)
    original=[p.detach().clone() for p in coordinator.selector.parameters()]
    squad=coordinator.choose(env,'0-1',444,True)
    assert len({o.id for o in squad})==6
    assert len(coordinator.keys)==77
    metrics=coordinator.finish(dict(total_enemies=11,kills=8,leaks=3,success=False,initial_life=20,life=17,termination="battle_end",simulator_result="WIN"),True)
    assert metrics['squad_selection_entropy']>0
    assert any(not torch.equal(a,b) for a,b in zip(original,coordinator.selector.parameters()))
    a=coordinator.choose(env,'0-1',444,False)
    b=coordinator.choose(env,'0-1',445,False)
    assert [o.id for o in a]==[o.id for o in b]


def test_joint_training_and_resume_pair_both_networks(tmp_path):
    config=dict(stage='main_00-01',stage_pool=['main_00-01','main_00-08'],auto_squad=True,joint_training_mode='joint',squad_size=3,iterations=1,episodes_per_iteration=1,mcts_simulations=2,max_decisions=512,max_depth=3,horizon=300,batch_size=2,training_steps_per_iteration=1,evaluation_episodes=1,replay_buffer_size=16,device='cpu',data_dir=str(DATA),checkpoint_dir=str(tmp_path/'checkpoints'),output_dir=str(tmp_path/'outputs'))
    trainer=AlphaZeroTrainer(config)
    before={k:v.detach().clone() for k,v in trainer.squad_coordinator.selector.state_dict().items()}
    trainer.train()
    checkpoint=tmp_path/'checkpoints/latest.pt'
    payload=torch.load(checkpoint,weights_only=True)
    assert payload['squad_selector']['pool']
    assert any(not torch.equal(before[k],v) for k,v in payload['squad_selector']['model'].items())
    resumed=AlphaZeroTrainer({**config,'resume_from':str(checkpoint)})
    assert resumed.iteration==1
    assert all(torch.equal(v,resumed.squad_coordinator.selector.state_dict()[k]) for k,v in trainer.squad_coordinator.selector.state_dict().items())
    assert json.loads((tmp_path/'outputs/evaluations.jsonl').read_text().splitlines()[-1])['episode_results'][0]['selected_squad']


def test_automatic_worker_uses_a_neural_agent(tmp_path,qtbot):
    from network import PolicyValueNetwork
    from desktop.controllers.simulator_controller import SimulatorController
    coordinator=SquadCoordinator(DATA,size=3)
    path=tmp_path/'model.pt'
    torch.save(dict(model_state_dict=PolicyValueNetwork().state_dict(),squad_selector=coordinator.state_dict(False)),path)
    controller=SimulatorController(data_dir=DATA)
    errors=[];controller.error.connect(errors.append)
    try:
        assert controller.start(mode='Automatic control',checkpoint=str(path),speed=0,auto_squad=True)
        qtbot.waitUntil(lambda:controller.last_snapshot is not None or bool(errors),timeout=15000)
        assert not errors
        assert controller._worker._agent.__class__.__name__=='NeuralBattleAgent'
        assert len(controller._worker._env.squad)==3
        qtbot.waitUntil(lambda:controller.last_snapshot.time>0 or bool(errors),timeout=15000)
        assert not errors
    finally:
        controller.shutdown(15000)
        qtbot.waitUntil(lambda:not controller.is_busy(),timeout=15000)
