# Phase D P0 — fixed-opponent pool protocol

## Goal

Measure whether SD5.3 can improve against varied frozen CNN opponents without
forgetting the RuleBot benchmark. This is a controlled Phase D P0 experiment,
not dynamic self-play: no checkpoint created during training may enter the
opponent pool.

## Frozen inputs

- Learner start: `runs/sd53-20261018/train/epoch_8.pt`, SHA256
  `8f7116d290593fa10de1753c4f0f96366e220a193e9860123ed72e0db46bfd4a`.
- Pool: `runs/phaseD-p0-20261018/pool_manifest.json`, five read-only imported
  checkpoints, each with a required SHA256.
- Environment: 9x9, two agents, 500 maximum steps, Schedule E
  (475/25/40), 13-channel learner observation, current action mask and reward.
- SD5.3 test seed 43000 remains permanently excluded.

## Experimental arms

Both arms use 1,000 episodes and the same frozen baseline, train seed, model,
optimizer, environment, validation configuration, and checkpoint-selection
rule.

| Arm | Opponent schedule |
|---|---|
| `control` | RuleBot for all 1,000 episodes |
| `pool` | Exactly 500 RuleBot episodes and exactly 100 episodes against each of the five manifest snapshots |

The pool order is a deterministic seeded permutation of the fixed multiset.
Every selected opponent and the final counts are recorded. The existing Phase D
automatic snapshot save/add path is disabled for both arms.

## Runner contract

Create `run_phase_d_p0.py` with separate `prepare`, `train`, `validate`, and
`test` stages. `prepare` creates an immutable registration only after verifying
the baseline, every pool hash, and the manifest's 50/10/10/10/10 contract.
Later stages refuse configuration, source, baseline, or pool drift.

`train` creates arm-local checkpoints and records actual opponent IDs/counts.
It must explicitly warm-start from the frozen SD5.3 baseline. It may save
ordinary arm checkpoints but may not mutate the imported pool or add them to
the sampling population.

`validate` evaluates checkpoints at episodes 250/500/750/1000 on seed 44000
(200 games per matchup) and selects the highest equal-weight mean ActiveWR
across RuleBot and the five fixed pool opponents, subject to RuleBot ActiveWR
no lower than the frozen baseline by 5pp and suicide no higher than the frozen
baseline by 1pp. It writes the selection artifact once. `test` opens an
exclusive marker before evaluation and refuses a second invocation. Test uses
seed 45000 (400 games per matchup); all full-test seeds differ from collection,
validation, and 43000.

## Evaluation and decision rule

Evaluate the frozen SD5.3 baseline, selected `control`, and selected `pool`
against RuleBot, each of the five pool opponents, and the held-out 12-channel
PPO checkpoint `runs/ppo-full-20261015-220313/ppo/best_phaseC.pt` (SHA256
`a4f689f2b9361457f395058f55bf11c71d2cd560d0dcf925da26cd37bee477d2`), which
the loader expands to 13 channels but never samples in training. Report
ActiveWR, win rate, draw rate, suicide rate, paired bootstrap confidence
intervals, denominator, seed, and corner split for every matchup.

The result is eligible for a follow-up only when all three predeclared gates
pass: (1) the pool arm's equal-weight ActiveWR mean across the six CNN
opponents exceeds control by at least 5pp and has a paired-bootstrap CI95 lower
bound above 0; (2) pool-vs-control RuleBot ActiveWR CI95 lower bound is at
least -5pp; and (3) pool suicide is no more than control +1pp. No test result
changes the selected checkpoint or pool.

## Acceptance tests

Independent acceptance tests must prove:

1. Invalid/missing manifests, hash drift, duplicate pool hashes, wrong pool
   size, wrong sampling weights, or a baseline that appears in the pool fail
   before training.
2. The deterministic 1,000-episode schedule has exactly 500 RuleBot and 100
   selections for each fixed snapshot; a different seed may change order only.
3. The control schedule contains RuleBot only; fixed P0 never requests or adds
   a live snapshot.
4. The train stage records counts and starts from the registered SD5.3 hash.
5. Validation selection is frozen before test; test uses new seeds, opens its
   exclusive marker first, and cannot run twice.
6. The reported gate uses the pool-vs-control held-out comparison plus RuleBot
   and suicide constraints; it does not tune from test data.

## Non-goals

- No dynamic P1 population, Elo/difficulty selection, reward/optimizer change,
  observation redesign, new checkpoint collection, or Phase E/FFA work.
- No reuse of any prior test seed, especially 43000.
