# Neural single-player PUCT

This package wraps the existing environment without changing simulator rules.
The original UCT implementation remains available in `agents.mcts`.

```python
from mcts import PUCTConfig, PUCTSearch
from network import NeuralEvaluator, PolicyValueNetwork

search = PUCTSearch(
    NeuralEvaluator(PolicyValueNetwork(), device="cpu"),
    PUCTConfig(simulations=64, seed=12345),
)
result = search.search(env, state, training=True)
next_state = env.step(state, result.selected_action)
```

The evaluator receives `(env, state, ordered_legal_actions)` and returns priors
in exactly that order plus a scalar value in `[-1, 1]`. The search normalizes
finite nonnegative priors, mixes a configurable uniform component, and adds
Dirichlet noise only to the root of training searches. Every legal action has an
edge, including actions that have not been visited. Child states are created
lazily through `env.step`.

Only `top_k_actions` candidates (default 8) initially participate in selection.
Each legal action type first receives its highest-prior leaf, ordered by total
type probability when the budget cannot represent every type; remaining slots
use descending leaf probabilities. Ties use original legal-action order. With progressive widening,
the candidate count is `top_k_actions + floor(widening_coefficient * N **
widening_exponent)` (defaults 2 and 0.5), bounded by the legal count. Every action
can eventually become eligible as visits grow. All original edges/identities
remain in the result, with zero visit probability for unvisited candidates.
Training root noise participates in ranking so the initial shortlist also
explores. PUCT preserves each represented type's total probability and
redistributes that mass over its currently active leaves in proportion to their
priors. This prevents five WAIT leaves from crowding out a DEPLOY type with
equal probability spread over hundreds of placements. Original priors, legal
identities and visit targets stay intact. The public evaluator `top_k_actions`
continues to return strictly probability-ranked leaves for inspection.

Selection uses `Q + c_puct * P * sqrt(max(1, parent.N)) / (1 + edge.N)`.
One simulation reaches an unexpanded child, a depth bound, or a terminal state.
Nonterminal leaves use neural remaining-return values; natural terminal states
use the shared environment reward bonus. Truncated boundaries use a pessimistic
search-only bound, never a supervised battle result. Backup adds each edge's
progress reward when moving to its parent, without player sign alternation.
New forced-action nodes continue within the same simulation up to max_depth,
so waiting-only boundaries do not each consume a separate simulation. Generic
environments without reward methods retain the legacy terminal +1/−1 contract.

For training, `result.policy` is proportional to `visit_counts ** (1 / T)`.
At temperature zero, a seeded tie break chooses one maximum-visit action.
Evaluation always uses temperature zero and no root noise. `result.visit_policy`
contains raw `N / sum(N)` fractions for research displays and is distinct from
the temperature-adjusted action distribution. The raw neural root priors and
root value are retained separately from search priors and backed-up Q values.

With `skip_forced_actions=True`, a state with one legal action returns policy
`(1.0,)` after root inference. It creates no simulations, and its actual visits
remain zero. Statistics explicitly distinguish this case from a searched root.
Terminal roots return no action and bypass inference entirely.

`enabled=False` runs the same legal policy interface without simulator search.
It is an ablation/evaluation mode, not an alternative policy improvement
algorithm: copying its own priors provides no new policy learning signal.
Values can still learn completed returns. Statistics record zero simulations
and explicitly identify this mode. `NeuralEvaluator.predict` also exposes
legal policy, value and state embedding; `top_k_actions` returns ordered
`(action, probability)` candidates for future beam/MPC consumers.

Search uses a private state clone with observational trace disabled. Prediction
cache entries live for one search and are keyed by both stable state key and
the complete ordered action tuple. RNG state belongs to the search instance;
fresh searches with matching seeds and identical model/environment inputs are
reproducible. No battle schedule, scripted agent, tactical rollout, solution
reuse, or network-independent heuristic is consulted.

This algorithm inherits the simulator's existing assumptions. It does not
verify continuous turning, blocking radius, damage timing, or same-frame
priority. The environment's declared decision/time limits bound search. They
remain explicit truncations and do not become supervised win/loss labels.
