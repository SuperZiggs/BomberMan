# Temporal Placement Regression Acceptance Matrix

## Frozen behavior packet

The planner must treat a proposed bomb as absent until its placement action, then apply ordinary bomb timing. A legal, safe plan may move before its first placement and must gain pressure from a distinct second placement. Escape margin counts only refuges reached along safe, unblocked, time-indexed paths. Existing safety, determinism, public contracts, RuleBot, rewards, and checkpoints remain intact. These tests do not train or alter run artifacts.

## Traceability

| ID | Requirement | Level | Scenario | Observable assertion | Exact command |
| --- | --- | --- | --- | --- | --- |
| TPR-01 | R1, R5 | Environment/planner integration | A delayed second placement occurs near an earlier blast and a real-tick twin has the second bomb placed only on its action tick | Before placement the proposed bomb cannot block movement, burn, or chain-react; from placement onward predicted damage and blockage match the real-tick twin | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_future_placement_matches_real_tick_timeline` |
| TPR-02 | R2, R3, R5 | Planner acceptance | Real arena has a safe first move, first placement after that move, further movement, and a distinct second placement with added enemy pressure | Non-`None` plan has two distinct positions, the correct move action, legal capacity/ticks, positive incremental pressure, stable repeat result, and no live-environment mutation | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_move_then_two_distinct_placements_gain_pressure` |
| TPR-03 | R3 | Planner acceptance | Existing pair fixtures called with a two-bomb budget | Each scenario requires a non-`None` two-placement plan and independently measured gain from bomb two, so `None` cannot pass | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_existing_pair_scenarios_require_a_pair` |
| TPR-04 | R4 | Metric boundary | A corridor refuge is spatially connected but its only route is closed or hazardous at the required intermediate tick | Escape margin excludes that refuge and counts only states reached through safe, unblocked transitions | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_escape_margin_excludes_temporally_unreachable_refuge` |
| TPR-05 | R5 | Existing integration/regression | Current counter-bomb, sudden-death, deterministic, RuleBot and trainer contracts | Existing focused suites continue to pass after implementation | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.py test_combat.py test_sudden_death.py` |
| TPR-06 | Caller-selected launcher output, R5 | Process integration/error | Stage fake trainer/evaluator and invoke the existing launcher with a fresh caller-selected output directory, retry that directory, then invoke evaluation-only recovery there | First run trains and evaluates once at the chosen path; retry cannot overwrite it; recovery evaluates its best checkpoint without training; default existing run directories are never touched | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_launcher_accepts_fresh_output_and_recovers_same_directory` |
| TPR-07 | R2, R5 future pair survival | Planner safety integration | BOMB-first, move to a distinct second placement, then escape through a corridor; real replay survives both detonations | Pair survival accepts the real legal route rather than treating the learner's old position as a ghost opponent blocker | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_future_pair_survival_ignores_old_learner_position` |
| TPR-08 | R1 delayed singleton timing | Planner pressure integration | One move precedes a single bomb; immediate and real-tick enemy refuge counts differ | Returned singleton pressure equals the real-tick refuge reduction at detonation | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_move_then_single_pressure_matches_real_placement_tick` |
| TPR-09 | R5 capacity fallback | Planner boundary | Two-bomb configuration with only one free ammo slot and a safe, above-threshold singleton | Planner returns that one-placement plan with legal move first and real-tick pressure | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_two_bomb_budget_uses_safe_singleton_when_ammo_is_one` |
| TPR-10 | R5 sudden-death closure | Environment/planner integration | Placement follows a move at step 475, when the only refuge closes; non-SD twin remains safe | Real-tick closure and planned future placement both reject the unsafe bomb; non-SD twin still admits it | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_future_placement_applies_sudden_death_closure` |
| TPR-11 | R1, R5 expiring existing bomb | Environment/planner integration | An enemy bomb with timer 1 explodes on the first move; learner reaches a future placement after all flames clear | Real replay and future placement forecast both permit the safe bomb; the expired bomb is absent | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_expired_existing_bomb_is_absent_at_future_placement` |
| TPR-12 | R5 restored counter-bomb capacity | Environment/planner safety integration | An enemy's timer-1 bomb expires during a three-move route, freeing its capacity before learner placement | Real replay rejects a new learner bomb under robust counter-bomb safety, and future placement forecast also rejects it | `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.TemporalPlacementRegression.test_expired_enemy_bomb_restores_counter_bomb_capacity` |

Run all new tests with `venv\Scripts\python.exe -m unittest -v test_temporal_trap_regressions.py`. On this host, set `TEMP` and `TMP` to `E:\Bomberman\__pycache__` in the same PowerShell session before running any command above. A valid RED baseline is an assertion failure for missing temporal behavior, never an import, fixture, or dependency failure.

The public planner returns aggregate pressure and placements, not a forecast cube. TPR-01 therefore compares the planner's current pressure observable with a real-tick environment replay through its existing `_combined_pressures` boundary. If that private boundary is removed during implementation, a separate reviewer should preserve the same before/after real-tick assertion at the replacement boundary. The explicitly unsafe adjacent-bomb fixture in the older suite must continue to reject the pair; TPR-02 and TPR-03 use independently replayed safe, legal pair scenarios for mandatory positive assertions.

## Recorded RED baseline

The five TPR tests were run individually with the exact commands above, and together with the all-new-tests command. Python compilation succeeds. All failures are behavior assertions:

| Test | Current assertion result |
| --- | --- |
| TPR-01 | Planned pair pressure `0.0`; real-tick pair pressure `0.04` after a delayed placement |
| TPR-02 | Planner returned `None` despite a replayed safe move-first, two-placement route with second-bomb refuge reduction |
| TPR-03 | Planner returned `None` for both original and mirrored positive pair arenas |
| TPR-04 | Escape margin `3`; time-indexed reachable count `1` |
| TPR-06 | Training received default `--checkpoint_dir ...phaseC-sd-temporal-250` instead of caller-selected `...phaseC-sd-temporal-250-v2` |

The combined command reports `Ran 5 tests` and `FAILED (failures=6)` because TPR-03 has two failing subtests. TPR-05 is a preservation gate for the implementer, not a new RED test.

## Independent fixture correction and current result

After the initial RED run, the test author tightened TPR-04: the intermediate crossing is now open at tick 0 and permanently closed from tick 1. Its static snapshot count remains 3, while every path arriving from the start after tick 0 has only the start reachable. This removes any dispute about waiting for a temporary closure to clear. TPR-02/03 now search real legal action timelines that match the public plan fields, rather than assuming the planner took the shortest geometric route. The pressure assertion samples the second bomb's actual detonation and flame window; sampling after the flames expire would miss a real but transient refuge reduction. These changes preserve the RED target and do not relax required behavior.

The revised five-test suite ran against the concurrently updated implementation with `TEMP=TMP=E:\Bomberman\__pycache__`: `Ran 5 tests in 4.480s`, `OK`. The exact revised tests were not rerun against the old implementation because the shared production files changed during the fixture review. The original assertion failures above remain the recorded pre-implementation RED evidence.

The same independent author revised one older launcher source-shape assertion in `test_phase_c_sd_temporal_250.py`: it now checks that the launcher's top-level parameters contain `[switch]$EvaluateOnly` and `[string]$OutputDirectory`, without requiring `EvaluateOnly` to be the only parameter. Its targeted command, `venv\Scripts\python.exe -m unittest -v test_phase_c_sd_temporal_250.PhaseCSDTemporal250LauncherAcceptance.test_launcher_declares_evaluate_only_switch`, reports `Ran 1 test`, `OK`. The revised new suite reports `Ran 5 tests`, `OK`.

## Additional independent RED cases

Each new TPR-07 through TPR-10 command above was run individually after the five-test suite passed. All four failures are behavior assertions, with the real-environment witness checks passing first:

| Test | Current assertion failure |
| --- | --- |
| TPR-07 | `_pair_survival` returned `False` although the real pair route survives and the future-position survival forecast accepts it |
| TPR-08 | Single-plan pressure `0.3913043478260869`; real placement-tick pressure `0.36` (25 baseline refuges, 16 after bomb) |
| TPR-09 | With one ammo slot and a safe singleton above threshold `0.35`, planner returned `None` |
| TPR-10 | Future placement safety returned `True` although the real closure at step 475 makes bombing unsafe |

The combined new-test command now reports `Ran 9 tests in 4.737s`, `FAILED (failures=4)`: the original five tests are GREEN, and exactly TPR-07 through TPR-10 are RED.

TPR-11 and TPR-12 were then run individually with the commands above. Both are valid RED behavior failures: TPR-11 future placement safety returned `False` although the real route reaches `(1,3)` alive with zero bombs/flames and can safely bomb; TPR-12 returned `True` although the enemy's real bomb expired, its capacity returned to one free slot, and robust real-tick counter-bomb safety is `False` (non-robust is `True`). Each reports `Ran 1 test`, `FAILED (failures=1)`.

Two older assertions in `test_temporal_trap_planner.py` were independently reviewed after the delayed placement behavior changed. The larger-pressure ranking fixture still selects `ACTION_RIGHT` and `(5,4)`; its hardcoded immediate-tick `0.5` was replaced with the real-tick result `1 - 4/9 = 0.5555555555555556`. The adjacent unsafe pair `(5,3), (4,3)` remains forbidden, while a different safe pair `(5,3), (4,1)` is now allowed and verified against a real-tick legal route and pressure gain. The unsafe and duplicate diagnostic assertions remain. Both targeted legacy commands under `test_temporal_trap_planner.TemporalPlanAcceptance` report `Ran 1 test`, `OK`.
