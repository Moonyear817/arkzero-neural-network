"""Agent integration regressions on the real, locally loaded 0-1 stage."""

import pytest
import subprocess
import sys
from pathlib import Path

from agents.episode import run_episode
from agents.random_agent import RandomAgent
from agents.scripted_agent import ScriptedAgent
from arknights_sim.environment import ArknightsEnv


@pytest.mark.parametrize("seed", [7, 12345])
def test_random_agent_complete_episode(seed):
    env = ArknightsEnv()
    episode = run_episode(env, RandomAgent(seed), seed=seed)
    assert env.is_terminal(episode.final_state)
    assert 0 < episode.decisions <= 1000
    assert all(record.legal_action_count > 0 for record in episode.history)
    game = episode.final_state.game
    assert game.killed + game.escaped == len(game.stage.spawns) or game.life == 0


def test_random_agent_seed_reproducible():
    env = ArknightsEnv()
    first = run_episode(env, RandomAgent(41), seed=19)
    second = run_episode(env, RandomAgent(41), seed=19)
    assert first.history == second.history
    assert env.state_key(first.final_state) == env.state_key(second.final_state)


def test_random_agent_does_not_use_simulator_rng():
    env = ArknightsEnv()
    state = env.reset()
    before = state.game.rng.getstate()
    agent = RandomAgent(7)
    assert agent.select_action(env, state) in env.legal_actions(state)
    assert state.game.rng.getstate() == before


def test_scripted_agent_historical_regression():
    env = ArknightsEnv()
    episode = run_episode(env, ScriptedAgent())
    game = episode.final_state.game
    assert env.is_terminal(episode.final_state)
    assert game.result == "WIN"
    assert game.killed == 11 and game.escaped == 0 and game.life == 20
    assert [(record.time, record.action) for record in episode.history[:3]] == [
        (scheduled.time, scheduled.action) for scheduled in ScriptedAgent.schedule
    ]


def test_scripted_agent_can_reset_for_independent_episodes():
    env = ArknightsEnv()
    agent = ScriptedAgent()
    first = run_episode(env, agent)
    second = run_episode(env, agent)
    assert first.history == second.history
    assert env.state_key(first.final_state) == env.state_key(second.final_state)


def test_episode_limit_does_not_claim_completion():
    with pytest.raises(RuntimeError, match="exceeded"):
        run_episode(ArknightsEnv(), RandomAgent(7), max_decisions=1)


def test_mcts_can_run_with_scripted_import_forbidden():
    # A fresh interpreter also catches accidental imports through agents.__init__.
    program = """
import importlib.abc
import sys
class RejectScriptedImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if 'scripted' in fullname:
            raise AssertionError('Search imported the regression baseline')
        return None
sys.meta_path.insert(0, RejectScriptedImport())
from arknights_sim.environment import ArknightsEnv
from agents.mcts_agent import PlainMCTSAgent
env = ArknightsEnv()
state = env.reset()
agent = PlainMCTSAgent(mcts_simulations=2, rollout_limit=2, max_depth=2)
assert agent.select_action(env, state) in env.legal_actions(state)
assert not any('scripted' in name for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True, timeout=30,
    )
