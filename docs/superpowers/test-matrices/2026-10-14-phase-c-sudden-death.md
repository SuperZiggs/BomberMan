# Phase C Sudden Death Acceptance-Test Matrix

## Scope and Independence

- Frozen source: the approved ten-item requirements packet supplied on 2026-10-14.
- Production evidence inspected: `E:\Bomberman\.worktrees\sd-red-baseline` only.
- Convention evidence inspected: baseline `test_combat.py` only.
- Production files changed: none.
- Long training is excluded; the only training execution is a two-episode smoke test.

## Matrix

| Req | Level | Scenario | Observable outcome | Acceptance test / command |
|---|---|---|---|---|
| 1 | Unit/API | Constructor exposes disabled-by-default SD plus start, interval, and warning options; invalid lower bounds are supplied | Default is disabled; start/interval reject `0`; warning rejects `-1` | `SuddenDeathEnvironmentAcceptance.test_public_configuration_and_invalid_boundaries` |
| 2 | Environment integration | A due first ring contains a moving agent, bomb, flame ownership/source metadata, power-up, and wood | Closure occurs before action; agent dies in place; every ring cell is rigid; bomb is removed and capacity restored; transient layers are cleared | `SuddenDeathEnvironmentAcceptance.test_due_closure_precedes_actions_and_clears_every_ring_layer` |
| 2 | Environment integration | `start=1`, `interval=2` across three steps | First inner ring closes on step 1, no ring closes on step 2, second inner ring closes on step 3 | `SuddenDeathEnvironmentAcceptance.test_interval_closes_exactly_one_next_inner_ring_when_due` |
| 3 | Environment/evaluation integration | Sole survivor wins only because the opponent is caught by the ring | SD death is separate; kills, self-blast deaths, kill counters, ActiveWR, and passive win stay zero; survivor receives no `+10` terminal reward | `SuddenDeathEnvironmentAcceptance.test_ring_death_has_no_bomb_credit_suicide_credit_or_terminal_bonus`; evaluator rate test |
| 4 | Observation unit | Default/disabled observations and enabled warning progression | Shape is `(13,H,W)`; channel 12 is zero when disabled/outside window; next-ring values progress `0.5 -> 1.0`; no other ring is marked | Observation tests in `SuddenDeathEnvironmentAcceptance` |
| 4 | Observation boundary | `warning_steps=0` | Warning channel remains zero before and after closure | `SuddenDeathEnvironmentAcceptance.test_zero_warning_steps_keeps_channel_zero` |
| 5 | Safety integration | A move enters a ring that closes before actions | Shared mask rejects the move only when SD warning applies | `SuddenDeathSafetyAcceptance.test_mask_rejects_move_into_ring_that_closes_before_action` |
| 5 | Safety integration | A bomb is safe only through a soon-closing escape corridor | Outside the warning window the mask matches disabled mode; in the window the bomb is rejected | `SuddenDeathSafetyAcceptance.test_mask_changes_only_in_warning_window_and_rejects_closing_escape` |
| 6 | Checkpoint contract | Load a 12-channel checkpoint into a 13-channel network and infer with a zero warning plane | Channels `0..11` copy exactly, channel 12 weights are zero, logits and value match | `SuddenDeathCheckpointAcceptance.test_twelve_channel_checkpoint_migrates_exactly_at_zero_warning` |
| 6 | Negative checkpoint contract | Load 11-channel or wrong-shaped tensors | Loader raises instead of retaining random parameters | `SuddenDeathCheckpointAcceptance.test_incompatible_checkpoint_tensors_raise` |
| 7 | Trainer API/unit | Probability outside `[0,1]`; timing options present | Invalid probability raises; public options exist | `SuddenDeathTrainingAcceptance.test_trainer_public_options_and_probability_validation` |
| 7 | Trainer integration | Repeat episode enablement with one seed; test probabilities `0`, `0.5`, `1` | Same seed reproduces the mixed sequence; endpoints are all off/all on | `SuddenDeathTrainingAcceptance.test_episode_enablement_is_seed_reproducible_and_honors_endpoints` |
| 7 | RNG regression | Phase-D curriculum with probabilities `0` and `1` | Exactly one curriculum draw per episode; endpoint SD choices consume none | `SuddenDeathTrainingAcceptance.test_probability_endpoints_do_not_consume_curriculum_random_draws` |
| 7 | Trainer metrics integration | Learner has one SD-only win and one SD death | Metrics independently expose SD enabled/death/win, active win, passive win, and draw | `SuddenDeathTrainingAcceptance.test_episode_metrics_separate_sd_active_passive_and_draw_outcomes` |
| 8 | Evaluator integration/JSON contract | Seeded evaluation where learner alternately survives or dies by ring closure | Overall, per-corner, and per-game output separates active, SD-only, passive, suicide, SD-death, and draw rates/outcomes | `SuddenDeathEvaluationAcceptance.test_seeded_evaluator_accepts_sd_config_and_separates_all_rates` |
| 8 | CLI contract | Request trainer/evaluator help | SD probability/timing flags are exposed by trainer; enabled/timing flags by evaluator | `SuddenDeathEvaluationAcceptance.test_training_and_evaluation_clis_expose_sd_configuration` |
| 9 | Regression integration | Compare omitted SD config with explicit disabled config under identical seed/actions | Channels `0..11`, rewards, dones, state trajectory, and legacy info keys match; channel 12 is zero | `SuddenDeathEnvironmentAcceptance.test_disabled_mode_preserves_legacy_trajectory_reward_and_info`; existing `test_combat.py` remains unchanged |
| 10 | Smoke integration | Two one-step CPU training episodes with SD always enabled | Rewards/model/checkpoint tensors are finite; final and best checkpoints are readable | `SuddenDeathTrainingAcceptance.test_smoke_training_writes_finite_checkpoints` |

## Commands

- Acceptance suite in the feature checkout: `python -m unittest -v test_sudden_death.py`
- Existing combat regression: `python -m unittest -v test_combat.py`
- RED baseline from `E:\Bomberman\.worktrees\sd-red-baseline`: `python -m unittest -v _phase_c_sd_acceptance_red.py`

## RED Baseline Record

- Baseline checkout: `E:\Bomberman\.worktrees\sd-red-baseline`.
- Exact command: `python -m unittest -v _phase_c_sd_acceptance_red.py`.
- Result: valid RED; `19` test methods ran with `22` assertion failures and `0` errors.
- Failure groups: `9` environment API/closure/warning/safety, `6` trainer API/RNG/metrics/smoke, `1` evaluator API/summary, `2` CLI flags, `1` observation shape, `1` 12-to-13 migration, and `2` malformed-checkpoint rejection failures.
- Setup control: unchanged baseline `python -m unittest -v test_combat.py` passed all `48` tests before RED execution.
- Temporary file: `_phase_c_sd_acceptance_red.py` was copied into the baseline checkout only for execution and removed afterward.
