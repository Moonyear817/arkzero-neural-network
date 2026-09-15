# Plain single-agent MCTS

`PlainMCTSAgent` uses classical UCT, with `Q + c * sqrt(log(N_parent) / (1 + N_child))` and the same reward perspective at every ancestor. There is no alternating sign, neural network, policy prior, PUCT, or stage-specific deployment prescription.

`MCTSConfig` controls `mcts_simulations`, `exploration_constant`, `max_depth` (tree depth), `rollout_limit` (additional decision actions), `seed`, `rollout_policy`, and `reuse_successful_plan`.

Two rollout policies are available:

- `random`: uniform sampling of the complete legal-action tuple.
- `tactical` (default): general stochastic route interception and attack-range coverage, useful skill activation, and fewer wasteful retreats. Every legal action keeps a strictly positive sampling weight. A weighted random permutation orders expansion; no action is removed.

Evaluation is isolated in `heuristic.py`. Verified terminal success is +1 and terminal failure is -1. Intermediate evaluation uses kill progress, life, and friendly health. After a leak, the strict zero-leak objective becomes impossible; the rollout may stop immediately with -1 even though the simulator itself has not ended the battle. This is an objective bound and does not redefine terminal state.

A successful rollout establishes a complete continuation from the search root to terminal success. The agent retains the shortest discovered continuation and follows it only while each observed full state key matches the simulated pre-action key and the action is still legal. An unexpected state invalidates the remaining plan and triggers a fresh search. These plans are generated solely through environment transitions, never read from a scripted agent or solution file. This optional optimization matters in single-agent planning because an already verified winning continuation should not be lost to later random searches.

Search statistics distinguish `found_solution`, `plan_reuse`, and `most_visited`. Plan reuse reports zero simulations and zero new nodes. All legal root actions appear in statistics, including unvisited actions. `average_depth` and `maximum_depth` measure tree leaf depth; `rollout_steps` separately counts rollout decisions. `average_legal_actions` samples generated tree and rollout decision states, including repeated visits, rather than distinct transposition keys. Nodes include the root; rollout states do not become tree nodes.

Each search clones its root. Environment transitions clone their inputs, so child states do not modify ancestors. Search drops observational battle traces from its private clone, while real episode traces remain intact. Trace does not participate in game transitions or state keys.

For a reproducible run, construct a fresh agent with the same MCTS seed and reset the environment with the same simulator seed. Timing measurements naturally vary; selected actions, counts, values, state keys, and discovered continuations are deterministic.
