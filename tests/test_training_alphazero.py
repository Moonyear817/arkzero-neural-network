"""Training integration uses actual networks/PUCT and small simulator fixtures."""

from copy import deepcopy
from threading import Event, Thread

import pytest
import torch

from arknights_sim.environment import ArknightsEnv
from network import ActionEncoder, NeuralEvaluator, PolicyValueNetwork, StateEncoder
from training import (
    AlphaZeroTrainer,
    ReplayBuffer,
    TrainingControl,
    TrainingEvent,
    TrainingSample,
)
from training.config import default_config, load_config, validate_config
from training.replay_buffer import collate_samples, masked_policy_loss
from training.self_play import play_episode
from training.trainer import promotion_decision


def tiny_config(tmp_path, **updates):
    result = default_config()
    result.update(
        device="cpu",
        iterations=1,
        episodes_per_iteration=1,
        mcts_simulations=2,
        max_depth=3,
        batch_size=2,
        training_steps_per_iteration=2,
        evaluation_episodes=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        output_dir=str(tmp_path / "outputs"),
    )
    result.update(updates)
    return result


def env_factory(simple, operator):
    return lambda: ArknightsEnv(
        stage=simple, squad=[operator], horizon=30, max_decisions=128
    )


def sample_from(env, *, z=-1.0, limit=None):
    state = env.reset()
    actions = env.legal_actions(state)
    if limit:
        actions = actions[:limit]
    return TrainingSample(
        StateEncoder().encode(state),
        tuple(a.to_dict() for a in actions),
        ActionEncoder().encode(state, actions),
        torch.full((len(actions),), 1 / len(actions)),
        z,
    )


def test_config_defaults_and_validation():
    value = load_config("configs/alphazero_001.yaml")
    assert value["stage"] == "0-1"
    assert value["replay_buffer_size"] == 50000
    with pytest.raises(ValueError):
        validate_config({"mcts_simulations": 0})
    with pytest.raises(ValueError):
        validate_config({"prior_uniform_mix": 1.1})


def test_training_event_payload_is_immutable():
    original = {"loss": [1.0]}
    event = TrainingEvent("training_metrics", original)
    original["loss"].append(2.0)
    assert event.payload["loss"] == (1.0,)
    with pytest.raises(TypeError):
        event.payload["x"] = 1


def test_control_pause_resume_and_stop():
    control = TrainingControl()
    seen = Event()
    finished = Event()
    control.callback = lambda event: seen.set()
    control.pause()
    results = []
    thread = Thread(
        target=lambda: (results.append(control.checkpoint()), finished.set())
    )
    thread.start()
    assert seen.wait(2)
    assert not finished.is_set()
    control.resume()
    assert finished.wait(2)
    thread.join()
    assert results == [True]
    control.stop()
    assert control.checkpoint() is False
    control.request("save_checkpoint")
    assert control.take_requests() == ("save_checkpoint",)


def test_replay_cpu_isolation_capacity_and_safe_roundtrip(simple, operator, tmp_path):
    sample = sample_from(env_factory(simple, operator)())
    replay = ReplayBuffer(2, seed=1)
    replay.add_episode([sample, sample, sample])
    assert len(replay) == 2
    assert all(
        t.device.type == "cpu" and not t.requires_grad
        for t in sample.state_features.values()
    )
    path = replay.save(tmp_path / "replay.pt")
    raw = torch.load(path, weights_only=True)
    assert raw["version"] == 1
    restored = ReplayBuffer.load(path)
    assert restored.sample(1)[0].actions == sample.actions
    assert torch.equal(restored.sample(1)[0].policy, sample.policy)
    with pytest.raises(TypeError):
        replay.add_episode([object()])


def test_episode_accepts_distinct_remaining_returns(simple, operator):
    env = env_factory(simple, operator)()
    failure, success = sample_from(env), sample_from(env, z=1.0)
    replay = ReplayBuffer()
    replay.add_episode([failure, success])
    assert [s.z for s in replay.samples] == [-1.0, 1.0]
    with pytest.raises(ValueError):
        sample_from(env, z=float("nan"))


def test_dynamic_action_mask_loss_and_padding_gradient(simple, operator):
    env = env_factory(simple, operator)()
    samples = [sample_from(env, limit=1), sample_from(env, limit=3)]
    batch = collate_samples(samples)
    assert batch["mask"].tolist() == [[True, False, False], [True, True, True]]
    logits = torch.randn(2, 3, requires_grad=True)
    loss = masked_policy_loss(logits, batch["policy"], batch["mask"])
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[0, 1:].tolist() == [0, 0]
    assert batch["policy"][0, 1:].sum() == 0
    model = PolicyValueNetwork()
    logits, values = model(
        batch["state_features"], batch["action_features"], batch["mask"]
    )
    objective = (
        masked_policy_loss(logits, batch["policy"], batch["mask"])
        + (values - batch["z"]).square().mean()
    )
    objective.backward()
    assert torch.isfinite(objective)


def test_selfplay_interruption_discards_incomplete_data(simple, operator, tmp_path):
    env = env_factory(simple, operator)()
    control = TrainingControl()
    control.stop()
    result = play_episode(
        env,
        NeuralEvaluator(PolicyValueNetwork()),
        tiny_config(tmp_path),
        seed=1,
        control=control,
    )
    assert not result.complete
    assert result.samples == []


def test_actual_tiny_training_checkpoint_and_evaluation(simple, operator, tmp_path):
    trainer = AlphaZeroTrainer(tiny_config(tmp_path), env_factory(simple, operator))
    before = deepcopy(trainer.model.state_dict())
    events = []
    result = trainer.train(callback=events.append)
    assert result["iterations_completed"] == 1
    sample_count = len(trainer.buffer)
    assert sample_count > 0
    assert all(-1 <= s.z <= 1 for s in trainer.buffer.samples)
    assert result["history"][0]["parameter_delta"] > 0
    assert any(
        not torch.equal(before[k], trainer.model.state_dict()[k]) for k in before
    )
    assert {e.kind for e in events} >= {
        "training_metrics",
        "episode_finished",
        "checkpoint_saved",
        "training_finished",
    }
    payload = torch.load(tmp_path / "checkpoints/latest.pt", weights_only=True)
    assert payload["iteration"] == 1
    assert payload["pending_iteration"] is None
    assert payload["optimizer_state_dict"]["state"]
    assert all(t.device.type == "cpu" for t in payload["model_state_dict"].values())
    evaluation = trainer.evaluate(seeds=[19])
    assert evaluation["temperature"] == 0
    assert evaluation["noise"] is False
    assert len(trainer.buffer) == sample_count
    loaded = AlphaZeroTrainer(
        tiny_config(tmp_path, resume_from=str(tmp_path / "checkpoints/latest.pt")),
        env_factory(simple, operator),
    )
    assert loaded.iteration == 1
    assert len(loaded.buffer) == sample_count
    for key in trainer.model.state_dict():
        assert torch.equal(
            trainer.model.state_dict()[key], loaded.model.state_dict()[key]
        )
    env = env_factory(simple, operator)()
    state = env.reset()
    actions = env.legal_actions(state)
    assert NeuralEvaluator(trainer.model).evaluate(
        env, state, actions
    ) == NeuralEvaluator(loaded.model).evaluate(env, state, actions)


def test_cooperative_stop_resume_keeps_completed_episodes_once(
    simple, operator, tmp_path
):
    config = tiny_config(tmp_path, episodes_per_iteration=2)
    trainer = AlphaZeroTrainer(config, env_factory(simple, operator))
    control = TrainingControl()
    trainer.train(
        callback=lambda event: (
            control.stop() if event.kind == "episode_finished" else None
        ),
        control=control,
    )
    assert trainer.iteration == 0
    first_count = len(trainer.buffer)
    assert first_count > 0
    assert len(trainer.pending["episodes"]) == 1
    resumed = AlphaZeroTrainer(
        {**config, "resume_from": str(tmp_path / "checkpoints/latest.pt")},
        env_factory(simple, operator),
    )
    summary = resumed.train()
    assert summary["iterations_completed"] == 1
    assert len(resumed.buffer) == resumed.history[0]["states_generated"]
    assert len(resumed.buffer) > first_count
    assert len(resumed.history) == 1


def test_tie_promotion_is_explicit():
    value = {
        "success_rate": 0.0,
        "average_return": -1.0,
        "average_leaks": 1.0,
        "average_kills": 0.0,
    }
    assert promotion_decision(value, value, True) == (True, "tie_accepted")
    assert promotion_decision(value, value, False) == (False, "tie_retained")


def test_resume_reproduces_next_optimizer_update_exactly(simple, operator, tmp_path):
    factory = env_factory(simple, operator)
    config = tiny_config(tmp_path / "continuous", episodes_per_iteration=2)
    continuous = AlphaZeroTrainer(config, factory)
    continuous.train()
    split_config = tiny_config(tmp_path / "split", episodes_per_iteration=2)
    interrupted = AlphaZeroTrainer(split_config, factory)
    control = TrainingControl()
    original_step = interrupted._training_step

    def stop_after_first_update():
        value = original_step()
        control.stop()
        return value

    interrupted._training_step = stop_after_first_update
    interrupted.train(control=control)
    assert len(interrupted.pending["losses"]) == 1
    resumed = AlphaZeroTrainer(
        {**split_config, "resume_from": str(tmp_path / "split/checkpoints/latest.pt")},
        factory,
    )
    resumed.train()
    assert len(resumed.buffer) == len(continuous.buffer) > 0
    assert resumed.history[0]["policy_loss"] == continuous.history[0]["policy_loss"]
    assert resumed.history[0]["value_loss"] == continuous.history[0]["value_loss"]
    for key in continuous.model.state_dict():
        assert torch.equal(
            continuous.model.state_dict()[key], resumed.model.state_dict()[key]
        )


def test_completed_iteration_resume_extends_without_duplicate_rows(
    simple, operator, tmp_path
):
    config = tiny_config(tmp_path)
    trainer = AlphaZeroTrainer(config, env_factory(simple, operator))
    trainer.train()
    resumed = AlphaZeroTrainer(
        {
            **config,
            "iterations": 2,
            "resume_from": str(tmp_path / "checkpoints/latest.pt"),
        },
        env_factory(simple, operator),
    )
    resumed.train()
    assert [row["iteration"] for row in resumed.history] == [1, 2]
    assert len(resumed.buffer) == sum(r["states_generated"] for r in resumed.history)
    assert (
        len((tmp_path / "outputs/training_metrics.jsonl").read_text().splitlines()) == 2
    )
