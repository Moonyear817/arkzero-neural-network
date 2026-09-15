"""Uniform legal-action baseline with an independent, repeatable random stream."""

import random

from .base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, seed=12345):
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self):
        self.rng.seed(self.seed)

    def select_action(self, env, state):
        actions = env.legal_actions(state)
        if not actions:
            raise ValueError("Cannot select an action in a terminal state")
        return self.rng.choice(actions)
