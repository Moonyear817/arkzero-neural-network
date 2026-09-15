# Event-aware neural architecture v3

The simulator owns the rules and event dependencies. The network receives the
entire supported action-group graph and its current execution state. It learns
choices under those rules; no stage ID, enemy ID vocabulary or walkthrough is
learned. Unsupported simulator rules remain explicitly unsupported.

## Inputs and compatibility

| Tensor | Shape before batching | Meaning |
|---|---|---|
| `map` | 20 × H × W | Existing terrain, deployment, route, ally/enemy occupancy and combat channels |
| `mechanics` | 16 × H × W | Devices, charging, inventory, gravity, wind, invisibility and terrain effects |
| `operators` | 12 × 48 | Existing squad attributes, deployment, HP/SP, skills, redeploy cooldowns, direction/range |
| `enemies` | 64 × 40 | Existing active enemy attributes, position, route progress, blocker and mechanics features |
| `enemy_events` | 64 × 5 | Stage blocker/last blocker flags and wave/fragment/action provenance |
| `summons` | dynamic × 56 | Independent summon cards and active instances; attributes, inventory, owner deployment and slot rules |
| `global` | 24 | Existing resources, life, counts, deployment limits and time/decision budgets |
| `future` | 3 × 12 | Retained legacy queued-spawn summaries over 5, 10 and 20 seconds |
| `events` | dynamic E × 70 | All event groups, including blocked and distant groups; types, progress, trigger conditions, timings, stats, routes and wave/fragment wait rules |
| `event_relations` | E × E | Directed time/clear/clear-or-timeout dependencies, reverse edges, same-fragment/wave and self relations |
| `progression` | 32 | Current wave/fragment, blockers, completion, future counts, resources, timeout and next-boundary information |
| action features | A × 103 | Existing 96 semantic features plus seven hierarchy metadata columns |

Boolean masks accompany map, operator, enemy, summon and event tensors.
`collate_states` pads graphs, relations, maps and dynamic summons. It never crops
event groups or silently removes enemies. The inherited operator/enemy caps
still raise explicit errors when exceeded. Summons use their own dynamic slots.

The event schema includes group counts, pre-delay, interval, known/conditional
ETA, lower-bound ETA, enemy stats, start/end geometry, route length and a 4 × 4
compiled-route summary. Exact route geometry remains in the authoritative
simulator graph. A conditional ETA is never represented as an exact known time;
`can_trigger` requires known due time and completed direct dependencies. All
ordered feature names are declared in the encoder source files.

The final seven action columns contain type, entity identity, tile x/y,
direction, wait category and duration. Identity columns only group legal
prefixes; no numerical entity identity is fed into a learned head. Existing
legacy action ordinal slot 78 is cleared before its neural projection for the
same reason.

## Architecture

The old 96-channel map/mechanics inputs and first two residual blocks retain
exact checkpoint weights. Two additional blocks start as identity functions.
Masked spatial, operator and enemy pooling and the old 832→512→256 fusion remain.

`FutureEventEncoder` uses a 70→128 projection, two Transformer layers with four
heads and 512-wide feed-forward sublayers, then masked mean/max pooling to 256.
Learned per-head additive biases represent dependency types. Attention can also
connect unrelated groups, allowing comparison of threats in distant waves.
Padded tokens cannot influence valid tokens. Dropout is zero for reproducible
planning. The implementation explicitly preserves floating dependency biases in
both training and inference, avoiding a fused PyTorch mask reinterpretation.

Progression uses a 32→128 MLP; summon entities use 56→128→128 and masked pooling.
Their context is projected to 256 and combined with the retained state feature
through a learned gate and LayerNorm. The main value head remains 256→128→1 with
Tanh. It predicts the shared remaining-return objective, not calibrated win
probability. `future_events_enabled=False` disables event, progression and
enemy-provenance contributions for staged training and ablation.

Policy normalizes conditional distributions over **legal children only**:

- action type → entity → tile → direction;
- skill/retreat → entity;
- WAIT → next-event / 0.5 / 1 / 2 / 5 seconds;
- device and summon actions retain their existing identities and appropriate prefixes.

Grouped normalization uses one score per child, independent of its number of
descendants. Thus deployment's many tiles do not automatically outweigh WAIT.
Type and wait heads start neutral. Device entity scores include charge, inventory,
activation and location, so multiple devices can be distinguished. The existing action embedding contributes at
the final directional choice. Returned leaf log probabilities follow the exact
supplied legal-action order and remain compatible with replay and PUCT.
Padding receives negative infinity and exactly zero probability.

Squad selection retains its existing scorer, baseline and mechanics context.
An independent copy of the same future encoder plus progression projection
provides full stage event information before selecting the squad. Candidates
remain physical/skill attributes plus professions, with sampling without
replacement. It uses the same episode objective as battle training.

## Parameter counts

| Battle module | Parameters |
|---|---:|
| Map input | 17,376 |
| Mechanics input | 13,824 |
| Four residual blocks | 664,320 |
| Operator encoder | 22,784 |
| Enemy encoder | 21,760 |
| Enemy event encoder | 640 |
| Global encoder | 1,600 |
| Legacy window encoder | 2,368 |
| Legacy fusion | 557,824 |
| Future event encoder | 405,928 |
| Progression encoder | 4,224 |
| Summon encoder | 23,808 |
| Context projection | 229,888 |
| Context gate and normalization | 131,840 |
| Legacy action encoder/query/bias | 111,489 |
| Hierarchical heads | 159,794 |
| Value head | 33,025 |
| **Battle total** | **2,402,492** |

Squad: future encoder 405,928; event context 171,644; mechanics context 97,280;
scorer 71,553; baseline 24,449. **Squad total 770,854. Combined 3,173,346.**

## Interfaces

```python
encoded = StateEncoder().encode(state)
actions = env.legal_actions(state)
features = ActionEncoder().encode(state, actions)
log_probabilities, value = model(encoded, features)
embedding = model.encode_state(encoded)  # also model.state_embedding(encoded)

result = NeuralEvaluator(model).predict(env, state)
# result: policy, value, legal_actions, state_embedding
ranked = NeuralEvaluator(model).top_k_actions(env, state, 8)
# deterministic tuple of (Action, original policy probability), highest first
```

`forward_with_embedding` returns the same outputs plus the shared feature;
`predict` performs state encoding and the backbone exactly once.

Existing `evaluate(env, state, ordered_actions)` continues returning ordered
probabilities and scalar value. `forward` supports padded batches and masks.
Terminal states use exact environment results instead of network inference.

## Checkpoints

Metadata declares architecture/observation version 3 and event graph version 1.
Current checkpoints restore exactly. v1/v2 weights can warm-start v3: unchanged
weights copy exactly; old enemy32/action80 columns expand explicitly; missing
new modules initialize with the new architecture. Unknown tensors or unexpected
shapes fail explicitly, including with `strict=False`.

Loading prints loaded/skipped/new parameter counts and exposes
`model.migration_report`, including tensor lists and partial expansions. Meta
construction plus `assign=True` creates concrete new tensors under a local,
restored RNG context. Old optimizer and replay state cannot be used as v3
experience; training enforces schema/version guards. New event heads have not
learned long-term strategies merely by migrating an older model.
