# Phase D self-play launcher acceptance matrix

Frozen packet: Phase D must be launched separately from existing Phase C
artifacts.  Tests remain dry: they replace the Python entrypoints with
recording stubs and never invoke real training or game evaluation.

| Requirement | Level | Scenario | Observable assertion | Test |
| --- | --- | --- | --- | --- |
| Separate, non-overwriting run | Process contract | Missing warm start / existing Phase D output | Launcher fails before any subprocess; existing sentinel is unchanged | `test_fail_fast_and_never_overwrites_existing_output` |
| Exact Phase D train configuration | Process contract | Successful stubbed train | One train call with Phase D, all frozen options, exact fallback warm start, output directory, and `best_phaseD.pt` postcondition | `test_training_contract_is_exact` |
| Baseline and candidate RuleBot benchmarks | Process contract | Successful stubbed post-train work | Exactly two `evaluate.py` calls, ordered fallback then generated Phase D best, each has identical 200-game seeded environment configuration and separate JSON outputs | `test_two_rulebot_evaluations_are_ordered_and_share_the_frozen_environment` |
| Gate reporting and preserved corner statistics | Process contract | Boundary-shaped evaluation JSON | All five gate labels report PASS at threshold and FAIL just beyond it; both JSON artifacts retain every game and `by_corner` data | `test_gates_are_reported_and_rulebot_artifacts_keep_corner_data` |
| Snapshot-pool diagnostic | Process contract | Successful stubbed post-train work | One extra deterministic, bounded Python diagnostic with its own persisted JSON output; it is distinct from both RuleBot evaluations | `test_snapshot_pool_diagnostic_is_distinct_deterministic_and_bounded` |

The higher-level substitute is a fully staged PowerShell process contract:
real 2,000-episode training is explicitly out of scope for acceptance testing.
Residual risk is model quality and real snapshot gameplay; launch ordering,
arguments, artifacts, gates, and overwrite behavior are covered without
executing training.
