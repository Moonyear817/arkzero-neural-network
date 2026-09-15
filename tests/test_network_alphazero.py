import pytest
import torch

from arknights_sim.environment import ArknightsEnv
from arknights_sim.environment.action import Action, ActionType, Direction
from network import (
    ActionEncoder,
    NeuralEvaluator,
    PolicyValueNetwork,
    StateEncoder,
    collate_states,
)
from network.action_encoder import ACTION_FEATURES
from network.state_encoder import (
    ENEMY_FEATURE_NAMES,
    FUTURE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    MAP_FEATURE_NAMES,
    OPERATOR_FEATURE_NAMES,
)


@pytest.fixture(autouse=True)
def single_thread_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def real_env(real_stage, squad):
    return ArknightsEnv(stage=real_stage, squad=squad)


def decision_state(env):
    state = env.reset(seed=12345)
    while not any(a.type == ActionType.DEPLOY for a in env.legal_actions(state)):
        state = env.advance_to_next_decision_event(state)
    return state


def test_encoder_shapes_finite_deterministic_and_core_isolation(real_env):
    state = decision_state(real_env)
    key = real_env.state_key(state)
    encoder = StateEncoder()
    a, b = encoder.encode(state), encoder.encode(real_env.clone_state(state))
    assert a["map"].shape == (20, 6, 9)
    assert a["operators"].shape == (12, 48)
    assert a["enemies"].shape == (64, 40)
    assert a["global"].shape == (24,)
    assert a["future"].shape == (3, 12)
    assert a["operator_mask"].sum() == 2
    assert all(torch.equal(a[name], b[name]) for name in a)
    assert all(torch.isfinite(value).all() for value in a.values())
    a["operators"].fill_(999)
    assert real_env.state_key(state) == key
    assert [
        len(v)
        for v in (
            MAP_FEATURE_NAMES,
            OPERATOR_FEATURE_NAMES,
            ENEMY_FEATURE_NAMES,
            GLOBAL_FEATURE_NAMES,
            FUTURE_FEATURE_NAMES,
        )
    ] == [20, 48, 40, 24, 12]


def test_encoder_overflow_is_explicit_not_truncation(real_env):
    state = real_env.reset()
    with pytest.raises(ValueError, match="capacity exceeded"):
        StateEncoder(max_operators=1).encode(state)
    state = real_env.advance_to_time(state, 30)
    assert sum(e.alive for e in state.game.enemies.values()) > 1
    with pytest.raises(ValueError, match="active enemies"):
        StateEncoder(max_enemies=1).encode(state)


def test_baseline_has_no_healing_or_passive_support_role(real_env):
    state = real_env.reset()
    encoded = StateEncoder().encode(state)
    index = OPERATOR_FEATURE_NAMES.index("support_role")
    assert not encoded["operators"][:, index].any()


def test_future_features_use_pending_queue_and_three_horizons(real_env):
    state = real_env.reset()
    for _ in range(100):
        if any(e.kind=='SPAWN' for e in state.game.queue.heap):break
        state=real_env.advance_to_next_decision_event(state)
    encoded = StateEncoder().encode(state)
    counts = encoded["future"][:, FUTURE_FEATURE_NAMES.index("spawn_count")]
    assert counts[0] <= counts[1] <= counts[2]
    child = real_env.clone_state(state)
    child.game.queue.heap = [e for e in child.game.queue.heap if e.kind != "SPAWN"]
    assert not StateEncoder().encode(child)["future"].any()
    assert real_env.state_key(child) != real_env.state_key(state)


def test_operator_cooldown_sp_direction_and_route_progress(real_env):
    state = decision_state(real_env)
    action = next(
        a
        for a in real_env.legal_actions(state)
        if a.type == ActionType.DEPLOY and a.direction == Direction.LEFT
    )
    state = real_env.step(state, action)
    encoded = StateEncoder().encode(state)
    row = sorted(state.game.squad).index(action.operator_id)
    assert encoded["operators"][row, OPERATOR_FEATURE_NAMES.index("left")] == 1
    assert encoded["operators"][row, OPERATOR_FEATURE_NAMES.index("deployed")] == 1
    later = real_env.advance_to_time(state, 15)
    later_encoded = StateEncoder().encode(later)
    assert later_encoded["enemy_mask"].any()
    assert later_encoded["enemies"][0, ENEMY_FEATURE_NAMES.index("route_progress")] > 0
    state = real_env.step(state, Action(ActionType.RETREAT, action.operator_id))
    encoded = StateEncoder().encode(state)
    assert (
        encoded["operators"][row, OPERATOR_FEATURE_NAMES.index("redeploy_remaining")]
        > 0
    )
    assert encoded["operators"][row, OPERATOR_FEATURE_NAMES.index("deployed")] == 0


def test_action_encoding_preserves_order_and_has_dynamic_action_count(real_env):
    state = decision_state(real_env)
    actions = real_env.legal_actions(state)
    encoder = ActionEncoder()
    features = encoder.encode(state, actions)
    assert features.shape == (len(actions), ACTION_FEATURES)
    assert torch.equal(encoder.encode(state, reversed(actions)), features.flip(0))
    assert encoder.encode(state, ()).shape == (0, ACTION_FEATURES)
    deploy = [i for i, a in enumerate(actions) if a.type == ActionType.DEPLOY]
    assert not torch.equal(features[deploy[0]], features[deploy[1]])


def test_model_forward_backward_optimizer(real_env):
    torch.manual_seed(3)
    state = decision_state(real_env)
    inputs = StateEncoder().encode(state)
    actions = ActionEncoder().encode(state, real_env.legal_actions(state))
    model = PolicyValueNetwork()
    assert 500_000 <= model.parameter_count <= 3_000_000
    logits, value = model(inputs, actions)
    assert logits.shape == (len(actions),)
    assert value.shape == () and -1 <= value <= 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.value_head[-2].weight.detach().clone()
    loss = -torch.log_softmax(logits, dim=-1)[0] + (value - 1).square()
    loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    optimizer.step()
    assert not torch.equal(before, model.value_head[-2].weight)


def test_batched_map_padding_does_not_change_outputs(real_env, simple, operator):
    torch.manual_seed(4)
    small_env = ArknightsEnv(stage=simple, squad=[operator])
    small = small_env.reset()
    large = decision_state(real_env)
    encoder = StateEncoder()
    s, l = encoder.encode(small), encoder.encode(large)
    a = ActionEncoder().encode(small, small_env.legal_actions(small))
    b = collate_states([s, l])
    model = PolicyValueNetwork().eval()
    with torch.no_grad():
        alone_logits, alone_value = model(s, a)
        batch_logits, batch_values = model(b, a.unsqueeze(0).repeat(2, 1, 1))
    assert torch.allclose(alone_logits, batch_logits[0], atol=2e-6, rtol=2e-5)
    assert torch.allclose(alone_value, batch_values[0], atol=2e-6, rtol=2e-5)
    # Even arbitrary padded payloads cannot influence valid states.
    b["map"].masked_fill_(~b["map_mask"].unsqueeze(1), 99999)
    b["operators"].masked_fill_(~b["operator_mask"].unsqueeze(-1), 99999)
    b["enemies"].masked_fill_(~b["enemy_mask"].unsqueeze(-1), 99999)
    with torch.no_grad():
        dirty_logits, dirty_values = model(b, a.unsqueeze(0).repeat(2, 1, 1))
    assert torch.equal(dirty_logits, batch_logits)
    assert torch.equal(dirty_values, batch_values)


def test_padded_actions_have_zero_softmax_and_no_nan_gradients(real_env):
    state = decision_state(real_env)
    s = collate_states([StateEncoder().encode(state)] * 2)
    features = ActionEncoder().encode(state, real_env.legal_actions(state))
    actions = features.unsqueeze(0).repeat(2, 1, 1)
    mask = torch.ones(actions.shape[:2], dtype=torch.bool)
    mask[0, 1:] = False
    model = PolicyValueNetwork()
    logits, value = model(s, actions, mask)
    assert torch.softmax(logits, dim=-1)[0, 1:].sum() == 0
    assert torch.softmax(logits, dim=-1)[0, 0] == 1
    safe_logp = torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0)
    target = mask.float() / mask.sum(-1, keepdim=True)
    loss = -(target * safe_logp).sum(-1).mean() + value.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with pytest.raises(ValueError, match="valid action"):
        model(s, actions, torch.zeros_like(mask))


def test_model_seed_reproducibility_and_action_permutation(real_env):
    state = decision_state(real_env)
    s = StateEncoder().encode(state)
    a = ActionEncoder().encode(state, real_env.legal_actions(state))
    torch.manual_seed(19)
    first = PolicyValueNetwork()
    torch.manual_seed(19)
    second = PolicyValueNetwork()
    with torch.no_grad():
        logits, value = first(s, a)
        other_logits, other_value = second(s, a)
        reversed_logits, reversed_value = first(s, a.flip(0))
    assert torch.equal(logits, other_logits) and torch.equal(value, other_value)
    assert torch.allclose(logits, reversed_logits.flip(0), atol=1e-7)
    assert torch.equal(value, reversed_value)


def test_neural_evaluator_preserves_action_order_mode_and_parent(real_env):
    state = decision_state(real_env)
    key = real_env.state_key(state)
    actions = real_env.legal_actions(state)
    model = PolicyValueNetwork().train()
    evaluator = NeuralEvaluator(model)
    priors, value = evaluator.evaluate(real_env, state, actions)
    reverse_priors, reverse_value = evaluator.evaluate(
        real_env, state, reversed(actions)
    )
    assert len(priors) == len(actions) and abs(sum(priors) - 1) < 1e-6
    assert priors == pytest.approx(tuple(reversed(reverse_priors)), abs=1e-7)
    assert value == reverse_value and -1 <= value <= 1
    assert model.training and evaluator.evaluations == 2
    assert real_env.state_key(state) == key
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate(real_env, state, ())
    terminal = real_env.clone_state(state)
    terminal.game.done = True
    with pytest.raises(ValueError, match="Terminal"):
        evaluator.evaluate(real_env, terminal, actions)
