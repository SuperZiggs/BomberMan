# Phase C Temporal Plan Follower Design

Date: 2026-10-16

## Status

Approved direction: persistent temporal-plan follower.  This document defines
the implementation to be reviewed before code or a training run begins.

## Evidence and problem

The current temporal planner can find a two-placement plan, but returns only
`first_action`.  The trainer asks it again on every tick and supervises only
that current action.  It therefore cannot teach a sequence such as
`move -> bomb 1 -> move -> bomb 2` as one tactical unit.

`temporal_trap_bombs` currently counts a planner recommendation whose first
action is `BOMB`; it does not count a bomb that the sampled policy placed.
The metric cannot establish whether the policy followed, completed, or safely
abandoned a plan.

The latest run also sampled Sudden Death on 75 percent of training episodes
while validation and final evaluation enable it on every game.  This changes
the distribution used to judge the Draw and Sudden Death death gates.

## Goals

1. Supervise every executable step of a safe temporal plan, including plans
   whose first bomb follows movement.
2. Advance a plan only after the sampled policy actually took its expected
   action and the next step remains legal and temporally safe.
3. Report policy behavior and pair completion from observed environment
   transitions, not from planner proposals.
4. Retain safety precedence: fire escape and ring safety override temporal
   plans, and no temporal plan bypasses the action mask.
5. Align the next training configuration with the all-Sudden-Death evaluator.

## Non-goals

- Do not alter environment rewards, RuleBot behavior, action-mask semantics,
  seed protocol, existing run directories, checkpoints, or evaluation files.
- Do not change loss coefficients in this change.  Honest sequence metrics
  are required before tuning objective weights.
- Do not start training as part of implementation or verification.

## Interfaces

### Planner result

`TemporalTrapPlan` exposes a deterministic read-only
`action_sequence: tuple[int, ...]` property.  It must preserve the existing
six dataclass fields because the established public API acceptance test freezes
that shape.
It contains each move needed to reach placement one, `BOMB`, each move between
placements, then the second `BOMB` for a pair.  A valid singleton contains its
moves and its one `BOMB`.

The plan continues to retain `first_action`, bomb positions, predicted
pressure, completion tick, escape margin, and elimination prediction for
ranking and diagnostics.  `first_action` must equal `action_sequence[0]`.

### Trainer follower

Each learning agent has at most one pending temporal plan.  At a decision
tick, in this order:

1. Apply urgent fire-escape and ring-safety guidance; this cancels any pending
   plan with a safety-abort reason.
2. If a pending plan exists, recompute legality and temporal safety for its
   next action from the live state.  Abort it if stale, unsafe, masked, or the
   expected agent state no longer matches.
3. If the pending plan remains valid, use its next action as the auxiliary
   label.  Otherwise find a fresh plan and initialize a follower if one is
   returned.
4. Sample the policy action as today.  Advance the follower only when the
   sampled action equals the expected action and the resulting transition is
   valid.  Any mismatch aborts the follower; a later tick may plan afresh.
5. Completion means the second observed `BOMB` action in a two-placement
   sequence.  Escape supervision resumes normally after completion.

The follower is trainer-only state.  Evaluation continues to run the learned
policy without teacher intervention.

### Metrics

Per episode and aggregate metrics add:

- `temporal_plans_started` and `temporal_pair_plans_started`;
- `temporal_steps_supervised`;
- `temporal_policy_action_matches` and a derived agreement rate;
- `temporal_bomb_actions_executed` from observed matching `BOMB` actions;
- `temporal_pair_completions`;
- `temporal_plan_aborts` split by `safety`, `stale`, `masked`, and `mismatch`.

Existing `temporal_trap_labels` remains a count of temporal supervision steps,
not a count of fresh planner proposals.  `temporal_trap_bombs` is replaced by
the observed bomb-action metric in reports and launch diagnostics.

## Safety and correctness rules

- The action sequence is reconstructed from the selected planner path; no
  movement or placement is invented by the follower.
- Before each supervised step, the live action mask, occupancy, scheduled
  bombs, forecast flames, and Sudden Death closure must permit the action.
- A pair may only complete after two distinct observed learner placements in
  the planned order.  A `None` plan never satisfies pair acceptance tests.
- The follower never forces an environment action.  It provides the training
  label and measures the policy's sampled behavior.
- Diagnostics remain bounded scalar counters so training logs do not grow per
  decision tick.

## Configuration decision for the next run

The existing launcher remains runnable for prior configurations.  The next
new-output training command will explicitly use Sudden Death probability
`1.0`, matching validation and final evaluation.  No run is launched by this
change.

## Acceptance criteria

Independent RED tests must prove all of the following before production edits:

1. A selected pair has a nonempty action sequence containing exactly two
   `BOMB` actions; `first_action` is its first element.
2. A move-first plan retains the movement before its first placement.
3. A delayed second bomb does not block or chain-react before its placement.
4. The follower advances only after an observed matching action, aborts on a
   mismatch, and never labels an unsafe or masked next step.
5. Pair completion and executed-bomb metrics use observed actions; merely
   proposing a plan cannot increase them.
6. Existing temporal safety, evaluation-resume, launcher, and repository tests
   stay green.

## Verification and rollout

Run focused planner and trainer tests, launcher tests, the full repository
suite, `py_compile`, and `git diff --check`.  An independent code review must
find no Critical or Important issue.  Training remains a separate user action,
with a new output directory and no overwrite.

This specification is intentionally uncommitted because the user requested no
commit or push.
