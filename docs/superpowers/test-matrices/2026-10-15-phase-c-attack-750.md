# Phase C Attack 750 Acceptance-Test Matrix

## Scope and Independence

- Frozen source: the eight approved requirements supplied on 2026-10-15.
- Production evidence inspected: current `train_actor_critic.py`, `bomberman_env.py`, `rule_based_bot.py`, `evaluate.py`, and `run_train_phaseC_sd_1500.ps1`.
- Test convention inspected: root-level Python `unittest` suites, including `test_combat.py` and `test_sudden_death.py`.
- Permitted changes: this matrix and new executable acceptance tests only.
- Prohibited changes: production code, launcher scripts, existing run outputs, commits, and pushes.
- Long-run constraint: tests must not execute real 750-episode training or real 200-game evaluation. Launcher process behavior is exercised in a temporary staged repository with fake train/evaluate entry points; trainer behavior uses bounded deterministic environments and stubs only at external or long-running boundaries.
- Residual risk: the suite cannot prove trained-model quality, CUDA stability, or the wall-clock behavior of a real 750-episode run. It proves teacher selection, pressure classification, loss weighting, public configuration, fail-fast orchestration, evaluation arguments/artifact handling, and gate reporting.

## Matrix

| Req | Level | Scenario | Observable outcome | Acceptance test / command |
|---|---|---|---|---|
| 1 | Regression unit/integration | Exercise representative RuleBasedAgent direct attack plus environment opponent kill and own-blast death | RuleBasedAgent retains its existing direct-blast action; environment kill reward remains `+1.0`, death/suicide reward remains `-1.0`, and a self-kill does not increment kills | `PhaseCAttackTeacherAcceptance.test_frozen_rulebot_and_environment_reward_contract` |
| 1, 4 | Public API/CLI contract | Inspect `train(...)` defaults and `train_actor_critic.py --help` | Existing positional/keyword callers remain valid; defaults are wall bonus `0.15`, entropy `0.01`, attack lookahead `2`, pressure gain `0.15`, late-game start `400`, multiplier `1.5`, and supervision coefficients `0.5/0.75/0.35`; every required CLI flag is exposed | `PhaseCTrainingConfigurationAcceptance.test_train_api_and_cli_expose_frozen_defaults` |
| 2 | Teacher behavior unit | A safe pressure-producing bomb position is reachable in exactly two movement steps while no current/adjacent position meets the pressure threshold | `attack_lookahead=2` returns the first safe movement action toward that candidate; `attack_lookahead=1` does not use the two-step candidate | `PhaseCAttackTeacherAcceptance.test_two_step_pressure_candidate_requires_lookahead_two` |
| 2, 3 | Teacher boundary unit | Compare proposed positions at pressure gains immediately below and at `0.15` | Below-threshold proposals are not treated as pressure combat; a gain of exactly `0.15` qualifies; direct-blast and ordinary navigation actions remain available as non-pressure guidance | `PhaseCAttackTeacherAcceptance.test_pressure_threshold_is_inclusive_and_preserves_fallback_guidance` |
| 3 | Trainer classification integration | Feed one pressure move/bomb above threshold and one below threshold through bounded training | Only qualifying proposals increment/use combat supervision; below-threshold attack guidance remains navigation rather than pressure combat | `PhaseCTrainingLossAcceptance.test_min_pressure_gain_controls_combat_labeling` |
| 5, 6 | Trainer loss integration | Record gradients of synthetic escape, combat, and navigation teacher losses on both sides of a configurable late-game boundary | Gradients use `escape_loss_coef`, `combat_loss_coef`, and `navigation_loss_coef`; `late_game_attack_multiplier` applies from the boundary onward to attack/combat supervision only, never escape, policy/value reward paths, or RuleBot action selection | `PhaseCTrainingLossAcceptance.test_configurable_loss_weights_and_late_game_boundary` |
| 4, 7 | Launcher contract | Execute the staged launcher against fake train/evaluate scripts | Training receives Phase C, 750 episodes, 9x9, 2 agents, max 500, lr `0.00005`, entropy `0.01`, all attack/loss parameters, validation `250/100`, SD probability `0.75`, Schedule E `475/25/40`, warm-start `runs\phaseC-sd-1500-e\best_phaseC.pt`, and fresh output `runs\phaseC-sd-attack-750` | `PhaseCAttack750LauncherAcceptance.test_launcher_training_contract_is_exact` |
| 7 | Launcher process integration | Fake training creates `best_phaseC.pt`, then fake evaluation writes JSON | Generated best checkpoint is evaluated once for 200 seeded CPU games on 9x9/2 agents/max 500 with Schedule E, producing `evaluation-200.json` | `PhaseCAttack750LauncherAcceptance.test_success_path_evaluates_generated_best_with_schedule_e` |
| 7 | Launcher negative integration | Missing interpreter/checkpoint, pre-existing output, failed training, missing best, or failed evaluation | Launcher exits nonzero at the first failed stage, never overwrites output, and never starts a later stage | `PhaseCAttack750LauncherAcceptance.test_fail_fast_and_no_overwrite_contract` |
| 7 | Launcher gate integration | Fake evaluation emits metrics exactly at and just beyond each existing threshold | The existing five gates report PASS at their boundaries and FAIL beyond them without changing their names or thresholds | `PhaseCAttack750LauncherAcceptance.test_existing_five_gate_boundaries_are_preserved` |
| 8 | Test safety | Run the targeted suite | No real 750-episode training, 200-game evaluation, production edit, run-output edit, commit, or push occurs; launcher artifacts remain in temporary directories | All tests in `test_phase_c_attack_750.py` |

## Planned Command

- Targeted acceptance suite: `venv\Scripts\python.exe -m unittest -v test_phase_c_attack_750.py`
- RED is valid only when failures identify absent attack lookahead/pressure configuration, absent configurable loss behavior, absent CLI/API options, or the missing `run_train_phaseC_sd_attack_750.ps1`; setup/import/syntax/dependency errors are invalid and must be corrected before handoff.

## RED Baseline Record

- Exact command: `venv\Scripts\python.exe -m unittest -v test_phase_c_attack_750.py`
- Result: valid RED; `11` test methods ran with `15` assertion failures, `0` errors, and `1` passing frozen-regression test in `2.422s`.
- Missing teacher contract (`2` failures): `get_attack_action is missing required options: ['attack_lookahead', 'min_pressure_gain']`.
- Missing train API/loss contract (`3` failures): `train is missing required options: ['attack_lookahead', 'combat_loss_coef', 'escape_loss_coef', 'late_game_attack_multiplier', 'late_game_start', 'min_pressure_gain', 'navigation_loss_coef']`.
- Missing CLI contract (`1` failure): `AssertionError: '--attack_lookahead' not found` in `train_actor_critic.py --help` output; this is the first absent required flag reached by the loop.
- Missing launcher contract (`9` failures, including subtests): `Missing required launcher: E:\Bomberman\run_train_phaseC_sd_attack_750.ps1` across the success, exact-arguments, gate, and fail-fast scenarios.
- Passing preservation check: `test_frozen_rulebot_and_environment_reward_contract` confirms the current RuleBasedAgent direct attack, environment kill/death rewards, kill accounting, and trainer kill/suicide shaping before implementation.
- Safety confirmation: no real training or evaluation ran. The missing API checks stop trainer probes before execution, the missing launcher stops every staged process test before dispatch, and the only subprocess executed was `train_actor_critic.py --help`.
- Focused existing-regression command: `venv\Scripts\python.exe -m unittest -v test_combat.CombatChecks.test_attack_teacher_routes_via_allowed_moves_to_a_firing_position test_combat.CombatChecks.test_attack_target_requires_clear_blast_and_escape test_combat.CombatChecks.test_combat_gradient_is_not_diluted_by_navigation_label_count test_combat.CombatChecks.test_rulebot_uses_real_range_and_blocked_rays test_combat.CombatChecks.test_suicide_is_penalized_and_is_not_a_kill`.
- Focused existing-regression result: `5` tests passed with `0` failures and `0` errors in `0.020s`.
