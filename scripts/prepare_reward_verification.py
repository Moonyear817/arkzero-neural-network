"""Save an untrained, resumable reward-v2 starting point without running games."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arknights_sim.environment.rewards import REWARD_VERSION
from training import AlphaZeroTrainer
from training.config import load_config
from training.replay_buffer import atomic_torch_save


def prepare():
    config = load_config(ROOT/'configs/reward_verification.yaml')
    for key in ('output_dir', 'checkpoint_dir'):
        config[key] = str(ROOT/config[key])
    config['data_dir'] = str(ROOT/'data/real')
    destination = ROOT/'outputs/reward_verification'
    destination.mkdir(parents=True, exist_ok=True)
    trainer = AlphaZeroTrainer(config)
    assert trainer.iteration == 0 and len(trainer.buffer) == 0
    full = trainer._checkpoint_payload(destination/'ready_to_train.pt')
    full.update(verification_only=True, training_started=False)
    atomic_torch_save(full, destination/'ready_to_train.pt')
    inference = dict(format_version=1, reward_version=REWARD_VERSION,
        model_state_dict=full['model_state_dict'], network_metadata=full['network_metadata'],
        squad_selector=trainer.squad_coordinator.state_dict(False), config=config,
        iteration=0, inference_only=True, verification_only=True,
        reward_trained=False, mechanics_trained=False, training_started=False)
    atomic_torch_save(inference, destination/'initial.pt')
    status = dict(status='PAUSED_FOR_VERIFICATION', training_started=False,
        reward_version=REWARD_VERSION, network_version=full['network_metadata']['version'],
        initial_model=str(destination/'initial.pt'), ready_checkpoint=str(destination/'ready_to_train.pt'),
        active_stages=list(trainer.stage_sampler.active_stages),
        config=str(ROOT/'configs/reward_verification.yaml'),
        message='仅保存全新网络和训练设置；未开始训练，未宣称策略已改善。')
    (destination/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    files = set(ROOT.glob('*.md')) | {ROOT/'pyproject.toml'}
    for directory in ('agents', 'arknights_sim', 'mcts', 'network', 'training', 'desktop', 'scripts', 'tests', 'configs'):
        files.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.md'))
    archive = destination/'source_snapshot.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(files):
            stream.write(path, str(path.relative_to(ROOT)))
    artifacts = [archive, destination/'initial.pt', destination/'ready_to_train.pt', destination/'status.json']
    tests = ROOT/'logs/reward_v2_tests.log'
    if tests.is_file():
        artifacts.append(tests)
    manifest = dict(status='PAUSED_FOR_VERIFICATION', source_files=len(files),
        artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
        original_equivalence_verified=False, policy_improvement_verified=False)
    (destination/'saved_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    prepare()
