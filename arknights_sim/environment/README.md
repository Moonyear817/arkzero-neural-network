# Agent Environment contract

`ArknightsEnv` wraps the existing `Simulator` without adding an Agent dependency to the core. `reset(stage_id="0-1", squad=None, seed=12345)` returns a settled `EnvState` at time zero. The default roster is the two supported operators at loader defaults. An explicit roster contains normalized `OperatorData` objects. A supplied `StageData` overrides stage-file lookup. Custom stages can be loaded before constructing the environment.

| Method | Contract |
| --- | --- |
| `legal_actions(state)` | Stable tuple containing every supported legal action; empty at terminal |
| `step(state, action)` | Validate first, clone, apply action, return child; invalid actions raise `ValueError` without modifying parent |
| `clone_state(state)` | Isolates mutable battle objects, RNG, event queue, skills, modifiers and trace records |
| `state_key(state)` | SHA-256 of full core snapshot/hash, decision count, decision limit and horizon |
| `is_terminal(state)` | Battle end, simulation horizon, or decision count limit |
| `result(state)` | Strict success, kills, leaks, life, total enemies, termination reason and original core result |
| `advance_to_next_decision_event(state)` | Exactly the WAIT action, including one decision count |
| `advance_to_time(state, time)` | Explicit schedule bridge for historical regression/replay; not an Agent action |
| `future_event_state(state)` | Detached full action-group dependency graph, current progression, blockers and future conditions |
| `get_next_decision_event(state)` | Predict the next event on an isolated clone; never mutate the input |
| `fast_forward(state, seconds=None)` | Return a clone advanced to the next decision, or by the specified relative duration; stop at terminal/horizon |
| `get_result(state)` | Alias for the existing result contract |

## State and reproducibility

`state.game` is the original `GameState`: time/tick, DP, life, kills/leaks/spawn counts, map, squad, all enemy/operator instances, deployment counts, redeploy times, skills/modifiers, pending hit generations, queue, RNG and timing configuration. Undeployed operators are the squad entries without a live deployed instance. The stage event runtime additionally stores wave/fragment phase, activation/deadline times, dispatched counts and spawned-unit provenance. Pending future groups remain in the graph even before their fragment is activated and they enter the combat queue.

The wrapper adds `decision_count`, `max_decisions` (512 by default), `horizon` (300 seconds), `last_decision`, and optional `trace`. State keys include every transition-relevant field. Trace and last-decision descriptions are observational metadata; they do not affect transition keys. Trace records are still deeply isolated when cloning. Normalized setup data may share identity only after recursive immutability validation. Snapshots are detached JSON-compatible values, not live references.

Determinism covers the same data, configuration, seed and action sequence within the tested Python/platform environment. Search has its own seeded RNG; it does not consume the simulator RNG. It does not promise identical floating point results on all platforms or compatibility with future model versions.

## Actions and legality

`Action` is a frozen dataclass with comparable/hashable fields. `ActionType` supports DEPLOY, ACTIVATE_SKILL, RETREAT, WAIT, the existing three device actions and two summon actions. `Direction` is RIGHT/UP/LEFT/DOWN (0/1/2/3 in the core). `(x, y)` uses the existing bottom-left origin. `.to_dict()` / `.from_dict()` serialize explicit structure instead of unstable action IDs.

DEPLOY requires an available squad member, no live deployment, completed redeploy cooldown, sufficient escalating DP cost, a free deployment slot, compatible tile and no live occupant. Every compatible tile and all four directions are enumerated. ACTIVATE_SKILL requires a live deployed operator, an existing manual skill, enough SP and no active duration. RETREAT requires a live deployed operator. WAIT exists in every nonterminal state. The same core checks remain in force when applying actions.

## WAIT and decision boundaries

WAIT observes the core as it advances internally. It returns at the first settled fixed-tick boundary with a change in spawn/death/leak count, live unit state, blocking, enemy tile membership, waypoint waiting, deploy affordability, redeploy availability, skill readiness/duration, or battle end. Terminal can also occur at an exact queued-event timestamp. Multiple changes at one boundary are reported together in `last_decision.reasons`.

The 60 Hz physics model is preserved. Queue events may occur between ticks; nonterminal Agent observation is quantized to the next settled tick, with at most one tick of delay. This is a declared environment scheduling choice, not a measured game rule. HP changes, attack cooldown countdown and tiny motion inside the same tile do not individually create decisions. All tile crossings are currently treated as meaningful topology changes; busier future stages may need a more selective event definition.

A WAIT always advances simulation time. If no event can occur, it reaches the horizon and ends the episode. Instantaneous deploy/retreat/skill loops also terminate at the decision limit. Time/decision limits are unsuccessful truncations, never false clears. The core's life-remaining WIN is preserved, while environment success requires all enemies killed and zero leaks.

Architecture v3 also observes wave/fragment changes, stage blockers and operator danger thresholds. Strategic `Action('WAIT', wait_seconds=0.5/1/2/5)` holds through intervening decisions for that duration while the ordinary simulation continues. These options are legal only when there is a real non-WAIT choice; forced waiting keeps one action. Default WAIT retains its previous serialization and next-event behavior. Waiting does not stop automatic attacks or grant a new cease-fire ability.

## Event graph and supported scheduler rules

The simulator, not the neural network, owns parsing, dependencies and progression. The graph contains every supported action group, typed directed dependencies, counts/status, actual alive enemy IDs, route geometry and threat attributes. Future exact times are exposed only when known. A pending clear-dependent event has `estimated_relative_time=None`; its earliest possible time is a lower bound, not a promised spawn time.

Supported normalized rules include parallel actions within a fragment, blocking SPAWN groups, wave clear gates, `dontBlockWave`, pre/post delays and finite wave wait limits. The implemented finite wait starts when the final fragment completes; post-delay starts after clear or timeout. These multi-wave semantics have deterministic regression tests but still require official-client timing calibration. The original 0-1 spawn sequence is preserved. External/random/hidden branches, blocking story/UI controls and unsupported special actions fail explicitly. This change does not enlarge the map catalog's verified capability gate.

## Saved solution replay

`save_episode` writes the setup fingerprint, actions/times, legal-action counts, decision-event descriptions, intermediate keys, final core/environment keys, result and optional complete battle trace. `replay_solution` constructs a new environment, reloads default normalized operators from the fixed GameData and validates all these fields. A modified stored trace is also checked against its recorded digest.

Default file replay supports the reviewed loader-default squad. For a synthetic stage or custom operator attributes, pass `env_factory(document)` to reconstruct precisely that setup; mismatched data is rejected by the initial key. This format is an action replay, not an arbitrary snapshot loader. Trace-free files verify state/results but do not claim trace equality. Wall times and search statistics are observations and are not deterministic replay outputs.

Continuous turning, blocking radius, attack windup and same-frame priorities remain ASSUMED. Passing replay proves consistency of this simulator, not agreement with the official game's physical timing.
