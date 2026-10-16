# Temporal Plan Follower Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach and measure complete safe temporal bomb sequences across learner ticks.

**Architecture:** The planner returns a deterministic immutable action sequence alongside its existing tactical prediction.  A trainer-only follower tracks one selected plan, revalidates its next action in the live environment, and advances only after the sampled policy action matches it.  Observed actions, rather than planner proposals, drive completion and bomb metrics.

**Tech Stack:** Python 3, NumPy, PyTorch, `unittest`, existing Bomberman environment and PowerShell launcher.

**Spec:** `docs/superpowers/specs/2026-10-16-phase-c-temporal-plan-follower-design.md`

## Global Constraints

- Do not alter environment rewards, RuleBot behavior, action-mask semantics, seed protocol, old run directories, checkpoints, or evaluation files.
- Do not change loss coefficients.
- Do not start training during implementation or verification.
- Do not commit, push, delete/reset changes, or overwrite prior run outputs.
- The next future training command must explicitly use Sudden Death probability `1.0`; this plan does not launch it.

## Review Focus

- A move-first pair must retain every movement action before its first `BOMB`; Task 1 adds an exact sequence assertion.
- A policy mismatch must abort before it can inflate a completion metric; Task 2 drives a mismatched sampled action.
- A fire or Sudden Death safety override must cancel a pending plan before labelling its next action; Task 2 tests the override path.
- A legal-looking but masked next action must never be supervised; Task 2 tests mask rejection.
- A singleton must not be counted as a pair completion; Task 2 exercises one- and two-placement plans separately.

---

## File Structure

- `temporal_trap_planner.py`: constructs planner action sequences from selected path nodes.
- `train_actor_critic.py`: owns follower state, live validity checks, label selection, transition accounting, and emitted temporal metrics.
- `test_temporal_trap_planner.py`: existing planner and trainer contract fixtures; add direct unit coverage for the public result and follower helper.
- `test_temporal_trap_regressions.py`: preserve existing independent temporal safety regressions; add only additional behavioral acceptance coverage.
- `run_train_phaseC_sd_temporal_250.ps1`: documents/configures `1.0` Sudden Death for the next fresh run, without invoking training.

### Task 1: Planner action-sequence contract

**Files:**
- Modify: `temporal_trap_planner.py:36-46,226-421`
- Modify: `test_temporal_trap_planner.py:298-740`
- Test: `test_temporal_trap_regressions.py:137-428`

**Interfaces:**
- Produces: read-only `TemporalTrapPlan.action_sequence: tuple[int, ...]`
  without adding a dataclass field.
- Produces: `_actions_for_placements(first: _PathNode, second: _PathNode | None) -> tuple[int, ...]`.
- Consumes: `_PathNode.path`, `_MOVE_DELTA`, and `ACTION_BOMB`.

- [ ] **Step 1: Write independent failing planner acceptance tests**

Add tests that select a real move-first singleton and pair fixture, then assert:

```python
self.assertEqual(plan.first_action, plan.action_sequence[0])
self.assertEqual(plan.action_sequence.count(ACTION_BOMB), 2)
self.assertEqual(plan.action_sequence, (ACTION_RIGHT, ACTION_BOMB, ... , ACTION_BOMB))
```

For a singleton, assert exactly one `BOMB` and that the final action is `BOMB`.

- [ ] **Step 2: Run the new planner tests to establish RED**

Run: `python -m unittest test_temporal_trap_planner.PlannerContractCase test_temporal_trap_regressions.TemporalPlacementRegression -v`

Expected: FAIL because `TemporalTrapPlan` has no `action_sequence`.

- [ ] **Step 3: Add the immutable sequence contract in `temporal_trap_planner.py`**

Store the private sequence on the immutable plan and expose it through a
read-only `action_sequence` property, preserving the existing six dataclass
fields. Implement `_actions_for_placements` by converting adjacent cells in
the selected node paths through `_MOVE_DELTA`, appending `ACTION_BOMB` at each
selected placement, and rejecting an invalid/non-adjacent path with
`ValueError`. Use it in `_candidate_plan` and `_pair_plan`; preserve all
ranking fields and score order.

- [ ] **Step 4: Run planner and regression tests**

Run: `python -m unittest test_temporal_trap_planner.PlannerContractCase test_temporal_trap_regressions.TemporalPlacementRegression -v`

Expected: PASS, including existing delayed-placement and time-indexed escape-margin tests.

### Task 2: Trainer temporal-plan follower and truthful metrics

**Files:**
- Modify: `train_actor_critic.py:206-246,766-883,991-1028`
- Modify: `test_temporal_trap_planner.py:943-1115`
- Test: `test_temporal_trap_regressions.py:137-428`

**Interfaces:**
- Consumes: `TemporalTrapPlan.action_sequence` from Task 1.
- Produces: `TemporalPlanFollower` with `expected_action() -> int`, `advance(sampled_action: int) -> bool`, and `abort(reason: str) -> None`.
- Produces: `_select_temporal_label(env, agent_id, danger, mask, follower, config, diagnostics) -> tuple[int | None, TemporalPlanFollower | None]`.
- Consumes: observed sampled learner action and `env.step` info to update metrics.

- [ ] **Step 1: Write independent failing follower tests**

Use a deterministic plan fixture and assert that the follower:

```python
self.assertEqual(label, ACTION_RIGHT)
self.assertTrue(follower.advance(ACTION_RIGHT))
self.assertFalse(follower.advance(ACTION_STOP))
self.assertEqual(diagnostics["temporal_plan_aborts_mismatch"], 1)
```

Add cases for a masked next action, urgent danger/ring override, exact two-bomb completion, and a singleton that cannot increment pair completion.  Assert executed-bomb and pair-completion counters remain zero until matching observed actions occur.

- [ ] **Step 2: Run follower tests to establish RED**

Run: `python -m unittest test_temporal_trap_planner.TemporalTrainerAcceptance test_temporal_trap_regressions.TemporalPlacementRegression -v`

Expected: FAIL because the follower and observed-action metrics do not exist.

- [ ] **Step 3: Implement follower, selection, and transition accounting**

Create the small trainer-local follower type near `get_attack_action`.  It stores the immutable plan, next sequence index, expected position, and observed pair bomb count.  It verifies the live mask, expected action, safety priority, and agent position before issuing a label.

Replace per-tick temporal proposal counting with sequence-step counting.  After `env.step`, compare the sampled learner action with the expected action; only then advance and increment observed bomb/completion metrics.  On mismatches, safety precedence, masks, or stale position, abort with the named reason.  Retain interception as navigation-only and preserve legacy non-temporal guidance.

- [ ] **Step 4: Run focused trainer tests**

Run: `python -m unittest test_temporal_trap_planner.TemporalTrainerAcceptance test_temporal_trap_planner.PlannerContractCase test_temporal_trap_regressions.TemporalPlacementRegression -v`

Expected: PASS; no test may derive executed bombs or pair completions from a plan proposal alone.

### Task 3: Future-run configuration and launch contract

**Files:**
- Modify: `run_train_phaseC_sd_temporal_250.ps1`
- Modify: `test_phase_c_sd_temporal_250.py:262-503`

**Interfaces:**
- Consumes: launcher's existing `-OutputDirectory` and `-EvaluateOnly` contract.
- Produces: a frozen training invocation whose `--sudden-death-probability` is `1.0` for a new output directory.

- [ ] **Step 1: Write a failing launcher assertion for evaluator-aligned Sudden Death**

Add an assertion that the normal training invocation contains:

```python
self.assertIn("--sudden-death-probability", args)
self.assertEqual(args[args.index("--sudden-death-probability") + 1], "1.0")
```

Keep every existing no-overwrite and `-EvaluateOnly` assertion unchanged.

- [ ] **Step 2: Run the launcher acceptance test to establish RED**

Run: `python -m unittest test_phase_c_sd_temporal_250.PhaseCSDTemporal250LauncherAcceptance.test_training_invocation_matches_frozen_temporal_contract -v`

Expected: FAIL because the launcher currently passes `0.75`.

- [ ] **Step 3: Change only the new training invocation to probability `1.0`**

Update the argument in `run_train_phaseC_sd_temporal_250.ps1`.  Do not invoke the script, create a run directory, or alter its output-directory/evaluation recovery behavior.

- [ ] **Step 4: Run launcher acceptance tests**

Run: `python -m unittest test_phase_c_sd_temporal_250.PhaseCSDTemporal250LauncherAcceptance -v`

Expected: PASS.

### Task 4: Final verification and independent review

**Files:**
- Modify only if a verified test or review finding requires it.

**Interfaces:**
- Consumes: Tasks 1-3 and all current repository tests.
- Produces: verified uncommitted changes ready for a separately authorized training decision.

- [ ] **Step 1: Run the full automated suite**

Run: `python -m unittest discover -v`

Expected: all tests pass, including preserved temporal-placement and launcher acceptance tests.

- [ ] **Step 2: Run static and patch checks**

Run: `python -m py_compile temporal_trap_planner.py train_actor_critic.py evaluate.py; git diff --check`

Expected: both commands exit zero.

- [ ] **Step 3: Obtain an independent code review**

Review the uncommitted diff against the specification, with special attention to actual policy-action accounting, plan invalidation, and unmodified evaluation behavior.  Resolve every Critical or Important finding, then rerun the affected tests.

- [ ] **Step 4: Report verification without training or committing**

Report changed files, test output, remaining experiment risk, and the exact future command configuration.  Do not run a PowerShell training command.

## Self-Review

- Spec coverage: Task 1 covers the sequence result; Task 2 covers persistent guidance, safety invalidation, and observed metrics; Task 3 covers evaluator-aligned Sudden Death; Task 4 covers verification and review.
- Step scan: each code step gives a defined interface, location, and test-driven behavior without prescribing an implementation transcript.
- Type consistency: Task 1 exports `TemporalTrapPlan.action_sequence`; Task 2 consumes that exact tuple and preserves planner ownership of path reconstruction.
- Review focus: all five listed operational failures map to explicit tests in Tasks 1 or 2.
- Proportion: the plan documents contracts and checks, leaving Python implementation details to the code tasks.
