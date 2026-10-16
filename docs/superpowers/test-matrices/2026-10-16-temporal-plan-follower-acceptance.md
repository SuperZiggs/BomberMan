# Temporal plan follower acceptance matrix

Frozen source: `.superpowers/sdd/temporal-plan-follower/test-author-packet.md`.

All commands run from `E:\Bomberman`. The new trainer scenarios execute one
short deterministic control-loop episode with a fixed policy, disabled backward
pass, disabled optimizer/scheduler, disabled filesystem writes, and disabled
evaluation. They cannot train a model or launch PowerShell.

| Requirement | Scenario / test | Expected observable result | Runnable command |
| --- | --- | --- | --- |
| 1 | `PlanSequenceAcceptance.test_selected_pair_exposes_replayable_ordered_actions` | Selected pair exposes nonempty ordered actions, exactly two BOMB actions, matching first action; replay places at advertised positions and learner survives. | `python -m unittest test_temporal_plan_follower.PlanSequenceAcceptance -v` |
| 2 | `PlanSequenceAcceptance.test_move_first_pair_keeps_prefix_before_first_bomb` | A forced RIGHT prefix remains ahead of the first placement; both placements occur at the advertised positions. | `python -m unittest test_temporal_plan_follower.PlanSequenceAcceptance -v` |
| 3 | `PlanSequenceAcceptance.test_singleton_stops_at_its_only_placement` | One free ammo slot produces one BOMB, with no actions after placement. | `python -m unittest test_temporal_plan_follower.PlanSequenceAcceptance -v` |
| 4 | `FollowerControlLoopAcceptance.test_matching_sample_supervises_whole_pending_sequence`, `test_fire_escape_preempts_pending_sequence`, `test_ring_safety_preempts_pending_sequence`, `test_unsafe_next_action_cancels_pending_sequence` | Teacher retains safe labels across ticks; fire escape and ring safety override pending guidance; newly unsafe moves are not taught. | `python -m unittest test_temporal_plan_follower.FollowerControlLoopAcceptance -v` |
| 5 | `test_sample_mismatch_cancels_before_following_tick`, `test_stale_transition_cancels_before_following_tick`, `test_masked_next_action_cancels_pending_sequence` | Matching observed actions advance; mismatch, blocked actual movement, and masked next action stop subsequent labels. | `python -m unittest test_temporal_plan_follower.FollowerControlLoopAcceptance -v` |
| 6 | `ObservedMetricsAcceptance` | Planning alone counts zero executed bombs/completions; matching observed placements count two bombs and one pair; partial/singleton/rejected placement counts no pair completion. | `python -m unittest test_temporal_plan_follower.ObservedMetricsAcceptance -v` |
| 7 | `ForecastPreservationAcceptance` | A future second placement neither blocks nor burns earlier; time-indexed unreachable refuge remains excluded. | `python -m unittest test_temporal_plan_follower.ForecastPreservationAcceptance -v` |
| 8 | `PreservedContractAcceptance` plus existing nontraining independent regressions | Reward shaping and legal masks retain concrete values; planner remains deterministic/nonmutating. Launcher/output/checkpoint/seeded-evaluation preservation needs independent source-diff review because invoking the launcher/evaluation is prohibited here. | `python -m unittest test_temporal_plan_follower.PreservedContractAcceptance -v`; `python -m unittest test_temporal_trap_regressions.TemporalPlacementRegression.test_future_pair_survival_ignores_old_learner_position test_temporal_trap_regressions.TemporalPlacementRegression.test_future_placement_matches_real_tick_timeline test_temporal_trap_regressions.TemporalPlacementRegression.test_escape_margin_excludes_temporally_unreachable_refuge -v` |
| 9 | All tests | Fixed policy only; no network learning, evaluation, launcher, or run artifacts. Optimizer, backward, saving, directories, reports, and evaluations are intercepted/asserted. | `python -m unittest test_temporal_plan_follower -v` |

## Limits

The requirements do not prescribe an action-sequence attribute name. Planner
tests discover a public sequence of action integers rather than requiring a
particular spelling. Controlled teacher fixtures supply common sequence aliases;
these aliases carry identical actions. The completion metric is discovered by
its temporal/two-bomb completion meaning, with its numeric value asserted.

Requirement 8 includes broad preservation guarantees that a small acceptance
suite cannot prove in full. Existing RuleBot, seeded evaluation, launcher, reward,
and mask implementations must also receive change-scope review; existing tests
that actually train or launch PowerShell are deliberately not executed.
