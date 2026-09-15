"""Verify separation against an existing model and a bounded real training run."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def same_tree(left, right):
    import torch

    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same_tree(a, b)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from arknights_sim.data.skill_loader import SKILL_ENGINE_VERSION
    from desktop.services.research_service import inspect_model
    from network import NeuralEvaluator, PolicyValueNetwork
    from training import AlphaZeroTrainer
    from training.replay_buffer import ReplayBuffer, atomic_torch_save
    from training.replay_storage import ReplayStorage, file_digest

    source = args.source.resolve()
    destination = args.checkpoint_dir.resolve()
    output = args.output.resolve()
    if (destination / "latest.pt").exists():
        raise FileExistsError("Choose an unused checkpoint directory for verification")
    output.mkdir(parents=True, exist_ok=True)
    original_hash = file_digest(source)
    original = torch.load(source, map_location="cpu", weights_only=True)
    # Storage migration preserves even old skill metadata. It must not make
    # incompatible historical replay appear valid for the current simulator.
    checkpoint = output / "legacy_separated/latest.pt"
    config = dict(
        original["config"],
        device="cpu",
        num_threads=1,
        iterations=2,
        resume_from=None,
        warm_start=str(checkpoint),
        checkpoint_dir=str(destination),
        output_dir=str(output / "smoke"),
        training_data_dir=str(ROOT / "训练数据"),
        data_dir=str(ROOT / "data/real"),
    )
    replay = ReplayBuffer()
    if original["format_version"] == 1:
        replay.load_state_dict(original["replay_buffer"])
    else:
        assert not ReplayStorage.load(replay, original["replay_buffer"], source)
    storage = ReplayStorage(dict(config, checkpoint_dir=str(checkpoint.parent)))
    payload = dict(
        original, format_version=2, replay_buffer=storage.save(replay, checkpoint)
    )
    atomic_torch_save(payload, checkpoint)
    converted = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in original.keys() - {"format_version", "replay_buffer"}:
        same_tree(original[key], converted[key])
    restored_replay = ReplayBuffer()
    assert not storage.load(restored_replay, converted["replay_buffer"], checkpoint)
    same_tree(replay.state_dict(), restored_replay.state_dict())
    original_model, converted_model = PolicyValueNetwork(), PolicyValueNetwork()
    for model, payload in ((original_model, original), (converted_model, converted)):
        model.load_state_dict(payload["model_state_dict"])
        model.future_events_enabled = original["config"]["future_events_enabled"]
    smoke = AlphaZeroTrainer(config)
    env = smoke._env()
    squad = (
        smoke.squad_coordinator.choose(env, config["stage"], config["seed"], False)
        if smoke.squad_coordinator
        else None
    )
    state = env.reset(
        stage_id=config["stage"],
        seed=config["seed"],
        **({"squad": squad} if squad else {}),
    )
    first = NeuralEvaluator(original_model).predict(env, state)
    second = NeuralEvaluator(converted_model).predict(env, state)
    assert first["policy"] == second["policy"] and first["value"] == second["value"]
    assert torch.equal(first["state_embedding"], second["state_embedding"])
    listed = inspect_model(str(checkpoint), ROOT / "data/real")
    assert listed.path == str(checkpoint)
    assert file_digest(source) == original_hash
    report = {
        "source": str(source),
        "checkpoint": str(checkpoint),
        "original_sha256": original_hash,
        "original_unchanged": True,
        "converted_samples": len(replay),
        "exact_model_and_optimizer": True,
        "exact_replay_and_rng": True,
        "identical_inference": True,
        "desktop_model_load": True,
        "old_checkpoint_bytes": source.stat().st_size,
        "model_checkpoint_bytes": checkpoint.stat().st_size,
        "training_data_dir": str(storage.directory),
        "legacy_skill_engine_version": original.get("skill_engine_version", 1),
        "current_skill_engine_version": SKILL_ENGINE_VERSION,
        "smoke_start": "warm_start_weights_only",
        "training_data_bytes": sum(
            p.stat().st_size for p in storage.directory.glob("*.pt")
        ),
    }
    print("Existing model converted and verified", flush=True)
    # Exercise current skill rules with new samples and both optimizer phases.
    battle_before = deepcopy(smoke.model.state_dict())
    squad_before = (
        deepcopy(smoke.squad_coordinator.selector.state_dict())
        if smoke.squad_coordinator
        else None
    )
    phases = []

    def receive(event):
        if event.kind == "training_metrics":
            phases.append(
                {
                    "iteration": event.payload["iteration"],
                    "phase": event.payload["training_phase"],
                    "gradient_norm": event.payload["gradient_norm"],
                }
            )
            print("Training iteration complete:", phases[-1], flush=True)

    smoke.train(callback=receive)
    assert smoke.iteration == config["iterations"] and smoke.pending is None
    assert any(
        not torch.equal(v, smoke.model.state_dict()[k])
        for k, v in battle_before.items()
    )
    if squad_before:
        assert any(
            not torch.equal(v, smoke.squad_coordinator.selector.state_dict()[k])
            for k, v in squad_before.items()
        )
    resumed = AlphaZeroTrainer(
        dict(config, resume_from=str(smoke.checkpoint_dir / "latest.pt"))
    )
    same_tree(smoke.model.state_dict(), resumed.model.state_dict())
    same_tree(smoke.optimizer.state_dict(), resumed.optimizer.state_dict())
    same_tree(smoke.buffer.state_dict(), resumed.buffer.state_dict())
    if smoke.squad_coordinator:
        same_tree(
            smoke.squad_coordinator.state_dict(), resumed.squad_coordinator.state_dict()
        )
    report.update(
        real_training_phases=phases,
        real_training_resume=True,
        final_samples=len(smoke.buffer),
        final_iteration=smoke.iteration,
        verified_current_checkpoint=str(smoke.checkpoint_dir / "latest.pt"),
        current_training_data_dir=str(smoke.replay_storage.directory),
        training_stopped=True,
        policy_improvement_verified=False,
    )
    (output / "runtime_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
