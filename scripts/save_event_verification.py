"""Freeze the reviewed source/artifacts and record that verification is paused."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    import psutil
    destination = ROOT / 'outputs/event_architecture'
    smoke = json.loads((destination/'smoke_verification.json').read_text())
    desktop = json.loads((destination/'runtime_verification.json').read_text())
    assert smoke['restore_exact'] and smoke['optimizer_and_replay_restore_exact']
    assert desktop['simulation_stopped'] and not desktop['errors']
    names = {'train_alphazero.py', 'run_overnight_joint.py', 'verify_event_training.py'}
    running = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            args = process.info['cmdline'] or []
            if any(Path(arg).name in names for arg in args) and Path(process.cwd()).resolve() == ROOT:
                running.append(process.info['pid'])
        except (psutil.Error, OSError):
            continue
    assert not running, f'Training processes are still running: {running}'
    status = dict(status='PAUSED_FOR_VERIFICATION', training_process_running=False,
        long_training_started=False, smoke_training_completed=True,
        iterations_completed=smoke['iterations'], pending_iteration=None,
        verification_model='outputs/event_architecture/verification.pt',
        resume_checkpoint='outputs/event_architecture/ready_to_resume.pt',
        policy_improvement_verified=False, original_equivalence_verified=False,
        message='已保存事件规划架构、短训练与桌面验证；模拟停止，等待用户验证。')
    (destination/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    sources = set(ROOT.glob('*.md')) | {ROOT/'pyproject.toml'}
    for directory in ('agents','arknights_sim','mcts','network','training','desktop','scripts','tests','configs'):
        sources.update(p for p in (ROOT/directory).rglob('*')
                       if p.is_file() and p.suffix in ('.py','.yaml','.md'))
    sources.add(destination/'benchmark_probe.py')
    archive = destination/'source_snapshot.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(sources):
            stream.write(path, str(path.relative_to(ROOT)))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}
    before = {r['path']:r['sha256'] for r in json.loads((destination/'baseline_inventory.json').read_text())}
    artifacts = {str(p.relative_to(ROOT)):digest(p) for p in sorted(destination.iterdir())
                 if p.is_file() and p.name != 'saved_manifest.json'}
    for p in sorted((destination/'smoke').glob('*.jsonl')):
        artifacts[str(p.relative_to(ROOT))] = digest(p)
    manifest = dict(**status, source_files=len(sources), source_hashes=hashes,
        modified_sources=[p for p,h in hashes.items() if p in before and before[p]!=h],
        new_sources=[p for p in hashes if p not in before],
        artifacts=artifacts,
        checkpoint_sources={p:digest(ROOT/p) for p in (
            'checkpoints/best.pt', 'outputs/overnight_joint/checkpoints/best.pt',
            'outputs/mechanics_verification/best.pt', 'outputs/reward_verification/initial.pt')})
    (destination/'saved_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=status['status'], source_files=len(sources),
        modified=len(manifest['modified_sources']), new=len(manifest['new_sources']),
        training_processes=running), ensure_ascii=False))


if __name__ == '__main__':
    main()
