# Simulator policy fixtures

These fixtures freeze the current research simulator's behavior. They are **not
official-game observations** and a passing test does not upgrade a mechanic to
VERIFIED. The source for all four fixtures is the V0.1/V0.2 implementation;
independent official-game measurements are still missing.

`tests/test_event_priority_v02.py` executes the fixtures during the normal pytest
suite. Each fixture records a permitted research status and links to its
mechanic note. When real measurements disagree, update the model, fixture, and
evidence together; do not redefine an official observation to match the engine.

| Fixture | Status | What the fixture locks |
| --- | --- | --- |
| continuous_turning.json | ASSUMED | Piecewise straight movement through required waypoints; instantaneous turns |
| blocking_radius.json | ASSUMED | 0.5 tile center-distance contact threshold |
| attack_windup.json | ASSUMED | Default 0.2-second hit delay and 1-second example interval |
| same_frame_event_priority.json | ASSUMED | Fixed queue and simulator phase policy |
