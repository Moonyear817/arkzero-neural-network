"""Bounded real two-phase training, full restoration and saved verification model."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from training import AlphaZeroTrainer
    from training.config import load_config
    from training.replay_buffer import atomic_torch_save
    from network import NeuralEvaluator, PolicyValueNetwork
    from network.squad_selector import SquadSelector

    destination = ROOT / 'outputs/event_architecture'
    config = load_config(ROOT / 'configs/event_architecture_verification.yaml')
    for key in ('output_dir', 'checkpoint_dir'):
        config[key] = str(ROOT / config[key])
    config['data_dir'] = str(ROOT / 'data/real')
    if (Path(config['checkpoint_dir']) / 'latest.pt').exists():
        raise FileExistsError('Verification already saved; use the trainer resume interface to continue it.')
    trainer = AlphaZeroTrainer(config)
    battle_before = deepcopy(trainer.model.state_dict())
    squad_before = deepcopy(trainer.squad_coordinator.selector.state_dict())
    phases = []

    def same(left, right):
        return left.keys() == right.keys() and all(torch.equal(v, right[k]) for k, v in left.items())

    def same_tree(left, right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(same_tree(v, right[k]) for k,v in left.items())
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(same_tree(a,b) for a,b in zip(left,right))
        return left == right

    def receive(event):
        nonlocal battle_before, squad_before
        payload = dict(event.payload)
        if event.kind == 'training_metrics':
            battle_after = trainer.model.state_dict()
            squad_after = trainer.squad_coordinator.selector.state_dict()
            battle_changed = not same(battle_before, battle_after)
            squad_changed = not same(squad_before, squad_after)
            phase = payload['training_phase']
            assert (battle_changed, squad_changed) == ((True, False) if phase == 'battle' else (False, True))
            assert payload['selfplay_trainable_episodes'] == 1
            assert payload['natural_terminal_fraction'] == 1
            assert payload['gradient_norm'] > 0
            phases.append(dict(phase=phase, battle_changed=battle_changed,
                               squad_changed=squad_changed, gradient_norm=payload['gradient_norm']))
            battle_before, squad_before = deepcopy(battle_after), deepcopy(squad_after)
        if event.kind in ('iteration_started', 'episode_finished', 'training_metrics',
                          'evaluation_finished', 'checkpoint_saved', 'training_finished', 'error'):
            print(event.kind, json.dumps(payload, ensure_ascii=False, default=str), flush=True)

    start = time.perf_counter()
    result = trainer.train(callback=receive)
    assert result['iterations_completed'] == 2 and trainer.pending is None
    assert [row['phase'] for row in phases] == ['battle', 'squad']
    assert len(trainer.buffer) > 0
    assert {'events', 'event_relations', 'progression', 'enemy_events'} <= trainer.buffer.samples[0].state_features.keys()
    resume_path = Path(config['checkpoint_dir']) / 'latest.pt'
    restored = AlphaZeroTrainer(dict(config, resume_from=str(resume_path)))
    assert restored.iteration == 2 and restored.pending is None
    assert same(trainer.model.state_dict(), restored.model.state_dict())
    assert same(trainer.squad_coordinator.selector.state_dict(), restored.squad_coordinator.selector.state_dict())
    assert len(restored.buffer) == len(trainer.buffer)
    assert same_tree(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    assert same_tree(trainer.squad_coordinator.optimizer.state_dict(), restored.squad_coordinator.optimizer.state_dict())
    assert same_tree(trainer.buffer.state_dict(), restored.buffer.state_dict())
    assert restored.rng.getstate() == trainer.rng.getstate()
    env = trainer._env()
    selected = trainer.squad_coordinator.choose(env, config['stage'], config['seed'], False)
    state = env.reset(stage_id=config['stage'], seed=config['seed'], squad=selected)
    first = NeuralEvaluator(trainer.model).predict(env, state)
    second = NeuralEvaluator(restored.model).predict(env, state)
    assert first['policy'] == second['policy'] and first['value'] == second['value']
    assert torch.equal(first['state_embedding'], second['state_embedding'])

    payload = trainer._checkpoint_payload(destination / 'ready_to_resume.pt')
    payload.update(verification_only=True, training_started=True, smoke_only=True,
                   policy_improvement_verified=False)
    atomic_torch_save(payload, destination / 'ready_to_resume.pt')
    inference = {key: payload[key] for key in (
        'format_version', 'reward_version', 'observation_version', 'event_graph_version',
        'skill_engine_version', 'model_state_dict', 'network_metadata', 'config', 'iteration')}
    inference.update(squad_selector=trainer.squad_coordinator.state_dict(False),
                     inference_only=True, verification_only=True, training_started=True,
                     smoke_only=True, policy_improvement_verified=False)
    atomic_torch_save(inference, destination / 'verification.pt')

    # Prove compatibility using actual saved old artifacts. They remain untouched.
    migrations = {}
    for label, path in (
        ('battle_v1', ROOT / 'checkpoints/best.pt'),
        ('battle_v2', ROOT / 'outputs/mechanics_verification/best.pt'),
    ):
        old = torch.load(path, map_location='cpu', weights_only=True)
        model = PolicyValueNetwork()
        model.load_state_dict(old['model_state_dict'])
        for key in ('map_input.weight', 'residual_blocks.0.conv1.weight', 'residual_blocks.1.conv2.weight'):
            assert torch.equal(model.state_dict()[key], old['model_state_dict'][key])
        migrations[label] = dict(source=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **model.migration_report)
    old_path = ROOT / 'outputs/overnight_joint/checkpoints/best.pt'
    old_squad = torch.load(old_path, map_location='cpu', weights_only=True)['squad_selector']['model']
    selector = SquadSelector()
    selector.load_state_dict(old_squad)
    migrations['squad_v1'] = dict(source=str(old_path.relative_to(ROOT)), **selector.migration_report)
    (destination / 'checkpoint_migration.json').write_text(json.dumps(migrations, ensure_ascii=False, indent=2) + '\n')
    status = dict(status='PAUSED_FOR_VERIFICATION', training_process_running=False,
        long_training_started=False, smoke_training_completed=True,
        strategy_quality_verified=False, original_equivalence_verified=False,
        iterations=trainer.iteration, replay_samples=len(trainer.buffer), phases=phases,
        restore_exact=True, optimizer_and_replay_restore_exact=True,
        wall_seconds=time.perf_counter()-start,
        verification_model=str(destination/'verification.pt'),
        resume_checkpoint=str(destination/'ready_to_resume.pt'),
        summary=result)
    (destination / 'smoke_verification.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in status.items() if k != 'summary'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
