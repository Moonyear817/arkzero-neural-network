# AlphaZero training core

This package calls the existing simulator through `ArknightsEnv`, the neural
encoders/model through `network`, and single-player PUCT through `mcts`. It never
imports desktop modules, PySide6, a scripted agent, or a tactical rollout policy.

```python
from training import AlphaZeroTrainer, TrainingControl, load_config

trainer = AlphaZeroTrainer(load_config("configs/alphazero_001.yaml"))
control = TrainingControl()
summary = trainer.train(callback=lambda event: print(event.kind), control=control)
```

The callback receives `TrainingEvent(kind, payload)`. Payloads are detached,
immutable mappings of simple values. Events include `training_started`,
`iteration_started`, `episode_finished`, `evaluation_finished`,
`training_metrics`, `checkpoint_saved`, and `training_finished`. An application
adapter can convert these events to Qt signals without a Qt dependency here.

`TrainingControl.pause()`, `resume()`, and `stop()` are thread-safe. Pause/stop
are observed between decisions and optimizer updates, never during an atomic
parameter update. Stop discards the incomplete episode and saves completed
episodes and update progress. A stopped pending iteration resumes with its next
unfinished episode/update. Requests from `request("save_checkpoint")` and
`request("evaluate")` run at the next completed iteration boundary; a paused
training run must resume before reaching that boundary. No process/thread is
forcibly terminated by the core.

Replay samples contain CPU state tensors, a tuple of canonical JSON identities
in the exact legal-action order, matching action features and policy targets,
and a bounded remaining-return target `z`. Reward v2 adds progress differences
(0.2 × killed fraction − 0.2 × lost-life fraction) and a terminal bonus:
0.6 for a zero-leak clear, 0.2 for an ordinary WIN, −0.6 for a natural loss.
Each state learns only future reward. Time/decision truncations are logged
separately and excluded from both battle and squad updates; their search-only
pessimistic full-score bound is −0.8. User interruptions also discard incomplete
data. Evaluation episodes never enter replay. See `STATUS_REWARD_VERIFICATION.md`.

Reward v2 is intentionally unchanged for event planning. No discount favors
early kills, and repeated waits with identical eventual kills/life/result have
identical total reward. DP, SP, cooldown, damage and device presses earn no
standalone reward. They matter through the resulting future battle state.

Replay storage is a lazily populated bounded deque, not a preallocated tensor
for 50,000 states. Batch collation pads dynamic action counts and supplies a mask.
Padded actions contribute no policy probability or gradient. Replay batches
prefer 80% meaningful-choice states when available; forced-action rows neither
contribute to nor dilute policy loss. They can still train values. The objective is
policy cross-entropy plus weighted value MSE. AdamW applies decoupled weight
decay. `regularization_diagnostic` reports half the decay coefficient times the
squared parameter norm; it is **not added again to the gradient objective**.
`total_loss` is the optimized policy + value objective, while
`loss_including_regularization_diagnostic` is explicitly a reporting diagnostic.

Self-play uses configurable temperature, root Dirichlet noise and a configurable
uniform-prior mixture. Their schedules are in YAML. Fixed-seed evaluation uses
zero temperature, no noise and no uniform mixture. The initial network is
evaluated before training; later candidates are compared to the accepted model
on the same evaluation settings. Promotion compares zero-leak success, ordinary win rate, return, fewer
leaks, then kills. Incomplete candidate evaluations cannot be promoted. `promote_on_equal: true` accepts ties explicitly as
`tie_accepted`; a tie is never presented as improvement. The current candidate generates
self-play. Auto-squad training defaults to alternating battle and squad updates;
the other network remains frozen. Both optimize the shared reward objective.
Optional curriculum unlocks configured stages after repeated fixed-seed mastery,
while evaluation always covers the same full pool for comparable model selection.

`latest.pt` and `iteration_NNNNNN.pt` are atomic, resumable checkpoints containing
current/best model state dictionaries, optimizer, validated configuration,
architecture/reward metadata, curriculum progress, external replay references, RNG, completed metrics and pending iteration
progress. Load them with `trainer.load_checkpoint(path)` or `resume_from` in the
configuration. On resume, paths/device/thread count/requested total iterations
may change; the checkpoint's data-generating and optimization settings are
preserved. `best.pt` deliberately contains only its matching inference model and
evaluation metadata, and cannot resume an optimizer that belongs to a different
candidate. All checkpoint/replay loaders use `torch.load(weights_only=True)`.

Checkpoint format 2 stores a replay manifest rather than sample tensors. Immutable
chunks of at most 256 samples live under `训练数据/<run-name>-<path-hash>/` in the
project workspace. `training_data_dir` can override the storage root; desktop
configuration resolves it against its workspace. Runs outside the project default
to the common parent of their checkpoint/output directories. New saves write only
new samples; `latest.pt` and historical checkpoints share existing chunks. Each
manifest retains its own sample ordering, bounded-buffer selection and replay RNG.
Chunk references are relative to the checkpoint, so moving the entire project
preserves them. Chunks are written atomically before publishing the checkpoint.
A failed checkpoint write leaves the preceding checkpoint usable.

After stopping training, delete a task's training-data subfolder to reclaim sample
space. If any referenced chunk is missing, resume clears that checkpoint's entire
replay and emits `replay_data_missing` with a user-facing explanation. Weights,
optimizers, completed updates and metrics are retained. A partially completed battle
iteration collects replacement episodes before attempting its remaining updates.
This is a fresh replay history, not an exact continuation of the deleted samples.
Corrupt chunks fail checksum validation explicitly. Reward, observation, event and
skill compatibility checks apply before missing-data recovery and are not bypassed.

Format 1 embedded replay remains readable when its simulation versions are
compatible. Its next save uses format 2; original files are not rewritten. Retained
old-format checkpoints still contain their original samples. Historical chunks are
not automatically deleted when replay evicts rows, because older checkpoints may
still need them; the replay capacity limits active samples, not total disk storage.
Logs in `outputs/` remain separate from sample chunks. Full training backup requires
both checkpoint and training-data directories; model inference only needs weights.
Call `_checkpoint_payload(destination)` when an internal export script writes a
checkpoint outside the normal checkpoint directory, to anchor relative references.

Metrics are saved to `training_metrics.jsonl`, completed training episodes with
action histories to `episodes.jsonl`, and every initial/candidate evaluation to
`evaluations.jsonl`. Metrics distinguish self-play from evaluation and include
actual simulations/nodes, forced actions, losses, update size, timing and terminal
reasons. The CPU tests verify exact inference and optimizer continuation after
save/resume. This does not claim universal bitwise reproducibility across
different devices or PyTorch versions.

Legacy reward checkpoints cannot resume their optimizer/replay. Explicit warm starts
preserve policy/features but reset the final value layer; incompatible squad
value baselines are reset too. Desktop inference ignores legacy value predictions
under the new reward objective. The old source checkpoints remain unchanged.

Architecture v3 stores observation version 3 and event graph version 1 in full
checkpoints and replay files. Full resumes reject old optimizer/replay semantics;
warm-start weight migrations print loaded/skipped/new parameters and keep old
checkpoints intact. An interrupted pending episode preserves its sampled stage
and seed, so resuming does not advance the curriculum sampler a second time.

The future encoder can be ablated with `future_events_enabled: false`; it is
enabled by default for both battle and squad networks. `planner_enabled: false`
is a direct-policy ablation: without search-improved or supervised labels it
does not constitute a separate policy-improvement training method. No synthetic
behavior labels are injected. Use `planner_enabled: true` for the existing
self-play RL flow. `top_k_actions`, `progressive_widening`,
`widening_coefficient` and `widening_exponent` control PUCT candidate expansion.
These settings persist across full resumes, alongside the future-encoder flag.

Each episode history now includes a `decision` explanation: current event,
wave/fragment progression, upcoming enemy groups and trigger dependencies,
top policy candidates, search visit probabilities, value and decision reasons.
`decision_log_top_k` defaults to 8. The same explanation is available on the
desktop neural agent as `last_decision_explanation`.

Episode and iteration statistics include decision-node counts, simulation
ticks, ticks per decision, WAIT/DEPLOY/SKILL/RETREAT/DEVICE counts and percentages,
and clear time (null when no battle was cleared). Summon deployment/retreat
belongs to the corresponding action family. Both battle and squad gradient
norms are available. These diagnostics describe decisions and training health;
they do not prove that an untrained network has learned the supplied mechanics.

The generic `policy_loss`, `value_loss`, `total_loss` and `gradient_norm` refer
to the active learner (`loss_network`), including squad-only iterations. Separate
`squad_*` and `battle_*` metrics keep joint updates distinguishable. Existing
`optimizer_steps` still counts battle updates; `battle_optimizer_steps`,
`squad_optimizer_steps` and `total_optimizer_steps` make both counts explicit.
An iteration with no updates has `loss_network: none`, zero update counts and
zero diagnostic losses. `policy_entropy` measures neural priors;
`search_policy_entropy` separately measures the action distribution after
search/temperature, so greedy evaluation does not hide neural uncertainty.
Episodes record explicit `battle_trainable`, `squad_trainable` and sample counts;
squad-only iterations do not construct unused battle replay tensors.
