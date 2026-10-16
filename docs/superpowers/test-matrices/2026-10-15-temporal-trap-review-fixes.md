# Temporal Trap Review Fixes Acceptance Matrix

## Frozen Scope

- These tests refine observable contracts from the approved temporal-trap and resumable-evaluation specification.
- Planner and trainer checks use deterministic real environments or one-episode mocked training boundaries.
- Launcher checks use staged fake trainer/evaluator programs; no real training or 200-game evaluation runs.
- Production code, launchers, README, rewards, and existing test expectations are not changed by this test-author pass.

## Traceability Matrix

| ID | Requirement | Level | Scenario | Expected observable outcome | Command |
|---|---|---|---|---|---|
| TTR-01 | Two new placements form a physically possible action sequence | Planner integration/boundary | A legal first bomb is followed by an adjacent second candidate | A returned two-placement plan includes enough ticks to move and then execute the second bomb action; an impossible tick-1 second placement is never returned | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.py` |
| TTR-02 | Pair safety uses the complete simultaneous forecast | Planner safety regression | Each placement is independently survivable, but the learner cannot survive while both learner bombs are active | The unsafe pair is rejected; any returned pair must be survivable with both bombs present | same |
| TTR-03 | Escape margin counts only reachable refuges | Planner metric regression | A candidate sits in a small connected walkable region surrounded by walls and disconnected cells | `escape_margin` never exceeds the learner's connected walkable region | same |
| TTR-04 | Interception source is navigation even when action values overlap | Trainer integration | Interception and the legacy sequence teacher both numerically choose `ACTION_RIGHT` | One interception label is recorded; combat label/loss stay zero and navigation loss is positive | same |
| TTR-05 | Both target-750 and temporal-250 launchers validate every game outcome and recompute gates | Launcher process integration/error | Each staged fake evaluator emits seed/aid-only rows, rows missing each outcome field, invalid binary outcomes, or forged top-level gate rates | Each launcher exits nonzero for every malformed or aggregate-inconsistent artifact | `venv\Scripts\python.exe -m unittest -v test_phase_c_sd_target_750.py`; `venv\Scripts\python.exe -m unittest -v test_phase_c_sd_temporal_250.py` |
| TTR-06 | Unsafe/timing rejection is not duplicate rejection | Planner diagnostics regression | Compare a bomb-only baseline with one additional adjacent candidate that fails timing and combined safety | Plan is rejected, unsafe count increases, and duplicate count changes only for the additional genuine same-position candidate | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.py` |

## RED Baseline Rule

Failures are valid only when assertions expose the missing physical timing, combined safety, reachable-margin, source-attribution, strict-artifact, aggregate-recomputation, or diagnostic-classification behavior. Syntax errors, unavailable dependencies, fixture failures, and real training/evaluation are invalid RED results.
