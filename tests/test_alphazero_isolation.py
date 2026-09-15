"""The learning path must remain usable without UI or baseline-strategy imports."""

import subprocess
import sys
from pathlib import Path


def test_neural_search_does_not_import_desktop_or_scripted_baselines():
    source = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "desktop" or fullname.startswith(("desktop.", "PySide6", "agents.scripted_agent", "agents.mcts")):
            raise AssertionError("Forbidden learning dependency: " + fullname)
sys.meta_path.insert(0, Guard())
import torch
torch.set_num_threads(1)
from training import AlphaZeroTrainer
from network import PolicyValueNetwork, NeuralEvaluator
from mcts import PUCTSearch, PUCTConfig
from arknights_sim.environment import ArknightsEnv
env=ArknightsEnv()
state=env.reset(seed=1)
while len(env.legal_actions(state)) == 1 and not env.is_terminal(state):
    state=env.advance_to_next_decision_event(state)
search=PUCTSearch(NeuralEvaluator(PolicyValueNetwork()), PUCTConfig(simulations=2))
assert search.search(env,state).selected_action in env.legal_actions(state)
'''
    result = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr

