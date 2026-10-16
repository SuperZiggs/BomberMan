# Phase C Best Fallback Acceptance-Test Matrix

## Frozen Packet and Independence

- Scope: verify that source represents the state that produced `runs/phaseC-sd-target-750` before temporal planning/following/interception and resumable evaluation were introduced.
- Preserved contracts: environment-owned Sudden Death, 13-channel observations, static pressure/two-position attack teacher, target-aware tuning, exact 750-episode launcher parameters from `runs/phaseC-sd-best-fallback/config.json`, and one-command 200-game JSON evaluation.
- Removed contracts: temporal runtime and artifacts, delayed-bomb `placement_tick`, and resumable evaluator/launcher flags and helpers.
- Frozen evidence: `runs/phaseC-sd-best-fallback/config.json`, `best_phaseC.pt`, and `evaluation-200.json`.
- Permitted writes: this matrix and `test_phase_c_best_fallback.py` only. Production code, existing tests, launchers, run outputs, Git state, training, and evaluation are out of scope.
- Safety: the suite parses source/launcher files, instantiates a fresh environment/model, loads the frozen checkpoint, runs one model forward pass, and validates the existing JSON artifact. It never invokes the PowerShell launcher, `train(...)`, `evaluate_seeded(...)`, or any 200-game process.

## Traceability Matrix

| ID | Frozen requirement | Level | Scenario | Expected observable outcome | Executable test |
|---|---|---|---|---|---|
| FBF-01 | R1, R3 | Source/API regression | Inspect trainer runtime and planner module | No temporal planner, follower, interception imports, options, helpers, or module remain | `RemovedFeatureAcceptance.test_temporal_planner_follower_and_interception_runtime_are_removed` |
| FBF-02 | R1, R3 | Data-contract regression | Inspect `Bomb` fields and environment source | `Bomb` has only live-game fields; delayed `placement_tick` is absent | `RemovedFeatureAcceptance.test_delayed_bomb_placement_tick_is_removed` |
| FBF-03 | R1, R3 | API/CLI regression | Inspect evaluator source/signature, CLI help, and target launcher | Resume/cadence state, helpers, CLI flags, and `EvaluateOnly` recovery path are absent | `RemovedFeatureAcceptance.test_resumable_evaluation_flags_and_helpers_are_removed` |
| FBF-04 | R1, R3 | Repository boundary | Check named post-fallback runtime, launcher, test, plan, spec, and matrix paths | Temporal and resumable feature artifacts no longer exist | `RemovedFeatureAcceptance.test_temporal_and_resumable_feature_artifacts_are_removed` |
| FBF-05 | R2 | Runtime integration | Reset a Schedule-E-compatible SD environment | Both observations have shape `(13, 9, 9)` and SD configuration remains callable | `PreservedBestRunAcceptance.test_sudden_death_observation_and_checkpoint_are_13_channel_compatible` |
| FBF-06 | R2 | Teacher behavior/contract | Inspect teacher signature and exercise cumulative pressure `0.20 + 0.15` at depth two | Only the static teacher options remain and the first safe bomb is selected at threshold `0.35` | `PreservedBestRunAcceptance.test_static_two_position_pressure_teacher_is_preserved` |
| FBF-07 | R2, R4 | Configuration/launcher contract | Compare frozen config with launcher options and trainer defaults | All approved target-tuning, Schedule E, loss, seed, validation, snapshot, logging, warm-start, and output values are preserved | `PreservedBestRunAcceptance.test_frozen_config_and_effective_training_parameters` |
| FBF-08 | R4 | Launcher static integration | Inspect subprocess order and evaluator arguments without executing the launcher | One invocation trains 750 episodes, then evaluates selected best for exactly 200 seeded CPU games and targets `evaluation-200.json` | `PreservedBestRunAcceptance.test_launcher_keeps_one_command_train_then_exact_200_game_json_evaluation` |
| FBF-09 | R5 | Checkpoint compatibility | Load frozen checkpoint into the 13-channel network and run one forward pass | Checkpoint loads; output shapes are actor `(1, 6)` and critic `(1,)` | `PreservedBestRunAcceptance.test_sudden_death_observation_and_checkpoint_are_13_channel_compatible` |
| FBF-10 | R5 | Artifact integrity | Validate existing JSON count, headline metrics, and recompute rates from game records | Exactly 200 games; WR `67.5`, ActiveWR `51`, Draw `21.5`, SD death `13`, Suicide `0.5` | `PreservedBestRunAcceptance.test_frozen_evaluation_artifact_has_exact_200_game_result` |
| FBF-11 | R6 | Safety/non-goal | Run only this acceptance module | No training or evaluation function/process is called and existing run artifacts are read-only | Entire suite; command below |

## Commands

- RED baseline: `venv\Scripts\python.exe -m unittest -v test_phase_c_best_fallback.py`
- Syntax-only check: `venv\Scripts\python.exe -m py_compile test_phase_c_best_fallback.py`
- Production implementation and GREEN verification are explicitly outside this test-author handoff.

## RED Baseline Record

- Frozen pre-fallback source: Git `HEAD=0d3b4745acc46bf976924b07688dd4926c65dfc8` (`feat: add temporal plan follower training`). The required tracked files were materialized with `git show HEAD:<path>` into an isolated `%TEMP%\phase-c-best-fallback-red-<guid>` fixture; only `test_phase_c_best_fallback.py` and the three frozen run artifacts were copied from the working tree. The fixture was deleted after the run.
- Exact test command inside the isolated fixture: `E:\Bomberman\venv\Scripts\python.exe -m unittest -v test_phase_c_best_fallback.py`.
- Result: valid RED; `9` tests ran with `5` failures, `0` errors, and `4` passes in `0.164s`.
- `test_static_two_position_pressure_teacher_is_preserved`: the public teacher still exposed `diagnostics`, `temporal_trap_horizon`, `temporal_trap_max_bombs`, `interception_horizon`, and `use_temporal_planner`.
- `test_delayed_bomb_placement_tick_is_removed`: `Bomb` still had the sixth field `placement_tick`.
- `test_resumable_evaluation_flags_and_helpers_are_removed`: all eleven forbidden resume/cadence tokens and helpers remained, including `resume_partial`, `partial_every`, `initial_games`, `on_checkpoint`, and `run_seeded_evaluation_to_file`.
- `test_temporal_and_resumable_feature_artifacts_are_removed`: sixteen tracked temporal/resumable launcher, test, plan, spec, and matrix files remained.
- `test_temporal_planner_follower_and_interception_runtime_are_removed`: all eleven checked runtime tokens remained, including planner imports/types, follower helpers, interception, and temporal CLI options.
- Passing preservation evidence: exact frozen config/effective launcher values, one-command 200-game JSON evaluation contract, 13-channel checkpoint load/forward pass, and exact 200-game metrics all passed. No training or evaluation ran.
- Concurrency note: production fallback edits appeared in the working tree while the test-author baseline was being prepared. A direct working-tree rerun of the same command then passed all `9` tests in `2.803s`. This GREEN run is evidence that the current uncommitted working tree already implements the contract; it does not replace the immutable RED baseline above.
