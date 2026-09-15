"""An agent controlling the environment through UCT and simulated futures."""

import time

from .mcts.search import MCTSConfig, MCTSSearch
from .mcts.stats import ActionStatistics, SearchStatistics


class PlainMCTSAgent:
    def __init__(self, config=None, **config_options):
        if config is not None and config_options:
            raise ValueError("Pass either config or MCTS configuration keywords")
        self.config = config or MCTSConfig(**config_options)
        self.searcher = MCTSSearch(self.config)
        self.last_stats = None
        self.last_root = None
        self.stats_history = []
        self._plan = []

    def reset(self):
        self.searcher = MCTSSearch(self.config)
        self.last_stats = None
        self.last_root = None
        self.stats_history.clear()
        self._plan.clear()

    def select_action(self, env, state):
        started = time.perf_counter()
        if env.is_terminal(state):
            raise ValueError("Cannot act in a terminal state")
        legal = env.legal_actions(state)
        key = env.state_key(state)
        if self._plan:
            planned = self._plan[0]
            if planned.state_key == key and planned.action in legal:
                self._plan.pop(0)
                self.last_stats = SearchStatistics(
                    root_state_key=key, time=state.game.current_time,
                    nodes=0, selected_action=planned.action,
                    selection_reason="plan_reuse", reused_discovered_plan=True,
                    average_legal_actions=len(legal), maximum_legal_actions=len(legal),
                    actions=tuple(ActionStatistics(action, 0, 0.0, 0.0) for action in legal),
                    wall_time=time.perf_counter() - started,
                )
                self.last_root = None
                self.stats_history.append(self.last_stats)
                return planned.action
            # Divergent observed states invalidate the proof. Never execute a
            # stale plan against a changed state, even if its action is legal.
            self._plan.clear()
        result = self.searcher.search(env, state)
        self.last_stats = result.stats
        self.last_root = result.root
        self.stats_history.append(result.stats)
        if result.discovered_plan and self.config.reuse_successful_plan:
            self._plan = list(result.discovered_plan[1:])
        return result.action


MCTSAgent = PlainMCTSAgent
