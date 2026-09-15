"""A real search artifact must replay all decisions and damage in a fresh engine."""

import copy
import json

import pytest

from agents.episode import run_episode
from agents.mcts import MCTSConfig
from agents.mcts_agent import PlainMCTSAgent
from arknights_sim.environment import ArknightsEnv
from arknights_sim.environment.replay import replay_solution, save_episode


@pytest.fixture(scope="module")
def searched_solution(tmp_path_factory):
    path = tmp_path_factory.mktemp("real_mcts_replay") / "solution.json"
    env = ArknightsEnv(trace=True)
    initial = env.reset(seed=12345)
    agent = PlainMCTSAgent(MCTSConfig(mcts_simulations=16, seed=12345))
    episode = run_episode(env, agent, initial)
    assert env.result(episode.final_state).success
    document = save_episode(path, env, initial, episode)
    return path, document


def test_real_mcts_solution_replays_in_fresh_instance(searched_solution):
    path, document = searched_solution
    # Default replay constructs another environment and reloads data independently.
    result = replay_solution(path)
    assert result["identical"] and result["trace_identical"]
    assert result["result"]["kills"] == 11
    assert result["result"]["leaks"] == 0
    assert result["result"]["success"]
    assert result["state_key"] == document["final_state_key"]
    assert result["game_hash"] == document["final_game_hash"]
    assert result["decisions"] == len(document["history"])
    assert all("next_decision_event" in row for row in document["history"])


@pytest.mark.parametrize(
    "tamper",
    [
        "initial_key",
        "action",
        "time",
        "nan_time",
        "state_key",
        "result",
        "trace",
        "trace_hash",
        "event",
        "decisions",
        "index",
        "legal_count",
    ],
)
def test_replay_rejects_changed_artifact(searched_solution, tmp_path, tamper):
    _, original = searched_solution
    document = copy.deepcopy(original)
    first = document["history"][0]
    if tamper == "initial_key":
        document["initial_state_key"] = "wrong"
    elif tamper == "action":
        first["action"] = {
            "type": "DEPLOY",
            "operator_id": "char_208_melan",
            "tile": [999, 999],
            "direction": "UP",
        }
    elif tamper == "time":
        first["time"] += 1
    elif tamper == "nan_time":
        first["time"] = float("nan")
    elif tamper == "state_key":
        first["next_state_key"] = "wrong"
    elif tamper == "result":
        document["result"]["kills"] = 0
    elif tamper == "trace":
        document["trace"][0]["event"] = "changed"
    elif tamper == "trace_hash":
        document["trace_hash"] = "wrong"
    elif tamper == "event":
        first["next_decision_event"]["reasons"] = ["changed"]
    elif tamper == "decisions":
        document["decisions"] += 1
    elif tamper == "index":
        first["decision"] = 99
    elif tamper == "legal_count":
        first["legal_action_count"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        replay_solution(path)
