# Phase C SD 1500 Launcher Acceptance-Test Matrix

## Scope and Independence

- Frozen source: the seven requirements supplied on 2026-10-15.
- Production evidence inspected: current `train_actor_critic.py`, `evaluate.py`, `run_train_phaseC_sd.ps1`, and `run_train_phaseC_sd_v2.ps1`.
- Test convention inspected: `test_sudden_death.py` using Python `unittest`.
- Permitted changes: this matrix and executable acceptance tests only.
- Prohibited changes: production code, launcher scripts, old run outputs, commits, and pushes.
- Long-run constraint: the 1500-episode job and real 200-game evaluation are not executed. Launcher process behavior is exercised against temporary fake training/evaluation scripts, while trainer validation wiring is checked with a stubbed evaluator.
- Residual risk: these tests cannot establish model quality or the wall-clock stability of a real 1500-episode run; they establish command construction, fail-fast orchestration, SD validation configuration, evaluation artifact handling, and gate reporting.

## Matrix

| Req | Level | Scenario | Observable outcome | Acceptance test / command |
|---|---|---|---|---|
| 1, 2 | Launcher contract | Inspect the new one-command launcher | `run_train_phaseC_sd_1500.ps1` invokes Phase C with exactly 1500 episodes, board 9, 2 agents, 500 steps, learning rate `0.0001`, wall bonus `0.15`, warm-start `runs\pressure-C-next\final_phaseC.pt`, SD probability `0.75`, and Schedule E `475/25/40` | `PhaseCSD1500LauncherAcceptance.test_launcher_training_contract_is_exact` |
| 3 | Launcher contract | Inspect validation arguments | Training receives validation every 250 episodes with 100 games, which includes the trainer's episode-zero candidate | `PhaseCSD1500LauncherAcceptance.test_launcher_validation_contract_is_exact` |
| 3 | Trainer integration | Run one one-step episode with validation interval 1 and a stub evaluator | Validation occurs at episodes 0 and 1; every candidate evaluation is forced to CPU-style seeded evaluation with SD enabled and Schedule E `475/25/40` | `SuddenDeathTrainingAcceptance.test_validation_candidates_use_enabled_schedule_e_from_episode_zero` |
| 4 | Launcher process integration | Run the launcher against temporary fake train/evaluate scripts | Training is followed by seeded evaluation of generated `best_phaseC.pt` for exactly 200 games on board 9, 2 agents, 500 steps, CPU, SD Schedule E; output is `<new run>\evaluation-200.json` | `PhaseCSD1500LauncherAcceptance.test_success_path_trains_then_evaluates_best_checkpoint` |
| 5 | Launcher negative integration | Interpreter, warm-start, or output directory is missing/present incorrectly | Launcher exits nonzero before invoking training and never overwrites the existing output directory | `PhaseCSD1500LauncherAcceptance.test_preflight_failures_stop_before_training` |
| 5 | Launcher negative integration | Fake training exits nonzero or omits `best_phaseC.pt` | Launcher exits nonzero and never invokes evaluation | `PhaseCSD1500LauncherAcceptance.test_training_failure_or_missing_best_stops_before_evaluation` |
| 5 | Launcher negative integration | Fake evaluation exits nonzero | Launcher exits nonzero after one training call and one evaluation call | `PhaseCSD1500LauncherAcceptance.test_evaluation_failure_is_process_failure` |
| 6 | Launcher gate integration | Fake evaluation writes metrics exactly on, above, and below each threshold | Output prints PASS for `active_win_rate >= 38.0`, `win_rate >= 60.5`, `draw_rate <= 23.5`, `suicide_rate <= 1.5`, and `sudden_death_death_rate <= 18.5`; any model-gate miss prints FAIL but launcher still exits zero | `PhaseCSD1500LauncherAcceptance.test_model_gates_report_boundaries_without_failing_process` |
| 7 | Test safety | Execute targeted acceptance tests | No real 1500-episode training or 200-game evaluation runs; all launcher artifacts stay in a temporary directory; existing run directories are untouched | All `PhaseCSD1500LauncherAcceptance` tests |

## Commands

- New launcher and validation acceptance tests: `venv\Scripts\python.exe -m unittest -v test_sudden_death.SuddenDeathTrainingAcceptance.test_validation_candidates_use_enabled_schedule_e_from_episode_zero test_sudden_death.PhaseCSD1500LauncherAcceptance`
- RED baseline must fail only because the new launcher and SD validation forwarding behavior are absent.

## RED Baseline Record

- Exact command: `venv\Scripts\python.exe -m unittest -v test_sudden_death.SuddenDeathTrainingAcceptance.test_validation_candidates_use_enabled_schedule_e_from_episode_zero test_sudden_death.PhaseCSD1500LauncherAcceptance`
- Result: valid RED; `8` test methods ran with `11` assertion failures and `0` errors in `1.562s`.
- Expected trainer failure: validation reaches episodes `0` and `1`, but `evaluate_seeded` receives no `sudden_death=True` or Schedule E keyword arguments (`AssertionError: None is not True`).
- Expected launcher failures: `10` launcher assertions fail with `Missing required launcher: E:\Bomberman\run_train_phaseC_sd_1500.ps1`, including the parameterized preflight and post-training failure cases.
- Safety confirmation: no real training/evaluation command ran because the launcher is absent; the trainer integration executed only one episode with one environment step and a stubbed evaluator.
