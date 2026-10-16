# Phase C Temporal Trap Planner and Resumable Evaluation Design

Date: 2026-10-15

## 1. Context

The latest Phase C-SD run warm-started from `runs/phaseC-sd-attack-750/best_phaseC.pt`, trained for 750 episodes, and selected episode 250 as the best checkpoint. Its fresh 200-game evaluation produced:

- Win Rate: 67.5%
- Active Win Rate: 51.0%
- Draw Rate: 21.5%
- Sudden Death death rate: 13.0%
- Suicide Rate: 0.5%
- Average bombs: 31.87
- Average steps: 350.43

Compared with the previous checkpoint, safety improved, but the agent placed more bombs without shortening matches enough. Ninety-eight of 200 games reached step 475 or later, and 35 games ended as draws at step 500. The current two-bomb teacher adds pressure values from independent static positions; it does not simulate the temporal interaction between the first placement, movement, the second placement, explosions, and opponent reachability.

The final evaluation initially did not create `evaluation-200.json`. The same command succeeded when run manually. The failure mode is therefore not an invalid output path: the CLI writes the JSON only after all games finish and gives no progress output. If the process is interrupted, no final or recoverable partial artifact exists. The launcher also rejects the existing run directory, preventing evaluation-only recovery.

## 2. Goals

1. Teach the policy only from safe, temporally valid two-bomb trap sequences.
2. Reduce late games and draws by teaching interception and choke-point movement when no immediate trap exists.
3. Preserve the existing RuleBot, environment rewards, action safety mask, Sudden Death Schedule E, and ring-safety supervision.
4. Make seeded evaluation observable, interruption-safe, resumable, and independently rerunnable.
5. Fine-tune from the current best checkpoint for a bounded 250 episodes and evaluate the selected checkpoint on 200 fresh games.

## 3. Non-Goals

- Do not change RuleBot behavior.
- Do not change environment rewards or kill attribution.
- Do not add Phase D self-play.
- Do not change board size, maximum steps, or Sudden Death timing.
- Do not overwrite any previous run directory or checkpoint.
- Do not claim the target metrics are guaranteed by the planner or tests.

## 4. Selected Approach

### 4.1 Temporal trap search

Add a bounded planner used only for auxiliary supervision. It searches candidate first actions and up to two bomb placements over a short deterministic horizon.

Each search state contains:

- learner row and column;
- simulation tick;
- active learner bomb placements and their timers;
- remaining bomb capacity;
- forecast walls, flames, blast cells, and Sudden Death closures;
- opponent reachable-cell set;
- first learner action that produced the state.

The planner uses existing environment forecast and safety functions rather than creating a second combat simulator. It must not mutate the live environment.

### 4.2 Candidate sequence contract

A two-bomb sequence is eligible only when all conditions hold:

1. The first action and first placement pass the current action mask and survival forecast.
2. The second placement is at a distinct reachable position.
3. The learner can reach the second position at the simulated tick without crossing forecast flames, bombs, walls, occupied cells, or a closing Sudden Death ring.
4. Both placements are legal under ammo and active-bomb capacity.
5. The learner survives the complete combined forecast, including simultaneous counter-bomb assumptions already used by the safety mask.
6. The combined sequence reduces opponent reachable refuges by at least the configured pressure threshold.
7. The second bomb provides additional refuge reduction beyond the first bomb alone.
8. Repeating the same placement, blast coverage, or pressure contribution is rejected.

The planner returns the first action of the best eligible sequence. Sequence ordering is deterministic for a fixed environment state and seed.

### 4.3 Sequence scoring

Eligible sequences are ranked lexicographically by:

1. predicted opponent elimination within the horizon;
2. larger opponent refuge reduction;
3. earlier predicted elimination or trap completion;
4. fewer placement and movement ticks;
5. larger learner escape margin;
6. stable action-order tie-break.

This ranking prioritizes finishing the fight rather than merely increasing bomb count.

### 4.4 Interception fallback

When no eligible two-bomb sequence exists, the teacher searches safe movement actions that reduce the opponent's future reachable region or move the learner toward a choke point. The fallback must not label a bomb unless that bomb independently meets the existing pressure and survival rules.

The fallback replaces repeated no-progress bombing with positional pressure. Existing upgrade collection and wooden-wall navigation remain available after interception guidance fails.

### 4.5 No-progress observability

Training metrics add bounded diagnostics without changing rewards:

- temporal trap labels and selected bombs;
- rejected unsafe sequences;
- rejected duplicate-pressure sequences;
- interception labels;
- predicted refuge reduction;
- predicted trap-completion ticks.

These metrics are used to diagnose behavior and are not reward terms.

## 5. Resumable Seeded Evaluation

### 5.1 Artifact lifecycle

For a requested final path such as `evaluation-200.json`, evaluation uses:

- `evaluation-200.partial.json` while incomplete;
- `evaluation-200.json.tmp` during the final atomic write;
- `evaluation-200.json` only after all requested games are present and summarized.

Every partial write uses a temporary file followed by `os.replace` so interruption cannot leave malformed JSON at the partial path.

### 5.2 Partial artifact schema

The partial artifact contains:

- schema version;
- checkpoint identity using normalized path, file size, and modification time;
- evaluation configuration: seed, board size, agents, episodes, maximum steps, device, and Sudden Death settings;
- completed game records in deterministic episode order;
- completed count and requested count.

Model weights are not duplicated in the partial file.

### 5.3 Resume validation

Evaluation resumes only when every identity and configuration field matches the current request. A mismatch fails clearly and preserves the partial file. The user must choose a different output path or explicitly remove the incompatible partial artifact.

Because each game derives its board seed and learner corner from the global episode index, resuming from the next incomplete index produces the same final result as an uninterrupted run.

### 5.4 Progress and write cadence

The CLI prints progress every 10 games and flushes output immediately. It writes the partial artifact at the same cadence and once more on interruption or handled failure when completed records exist.

On successful completion it:

1. validates that exactly 200 unique ordered game records exist;
2. computes aggregate and per-corner metrics from those records;
3. writes the final JSON atomically;
4. removes the partial artifact;
5. prints the final metrics and saved path.

### 5.5 Evaluation-only recovery

The Phase C launcher accepts an `-EvaluateOnly` switch.

- Normal mode requires a missing output directory, performs training, then evaluation.
- Evaluation-only mode requires an existing output directory and `best_phaseC.pt`, skips training, and resumes or starts the final evaluation.
- Both modes validate the final JSON before printing gates.
- A model gate failure still exits successfully after reporting FAIL; process, checkpoint, schema, configuration, or artifact failures exit nonzero.

The launcher prints explicit stage messages, including a warning not to close the window until `evaluation-200.json` is confirmed.

## 6. Fine-Tune Configuration

The new run uses:

- warm start: `runs/phaseC-sd-target-750/best_phaseC.pt`;
- output: `runs/phaseC-sd-temporal-250`;
- phase: C;
- episodes: 250;
- learning rate: 0.00001;
- entropy coefficient: 0.005;
- board size: 9;
- agents: 2;
- maximum steps: 500;
- Sudden Death probability: 0.75;
- Sudden Death schedule: start 475, interval 25, warning 40;
- ring-safety coefficient: 0.6;
- validation: episode 0, 125, and 250, using 200 games and a fixed validation seed distinct from final evaluation;
- final evaluation: selected best checkpoint, 200 games, CPU, fresh seed;
- unchanged gates: WR >= 70%, ActiveWR >= 55%, Draw <= 15%, SD death <= 10%, Suicide <= 1.5%.

The warm-start checkpoint remains eligible. Checkpoint selection preserves the existing target-aware ranking and suicide eligibility rule.

## 7. Files and Interfaces

Expected production changes:

- `train_actor_critic.py`: temporal planner, interception fallback, configuration, metrics, and trainer integration.
- `evaluate.py`: resumable seeded evaluation, progress callbacks, atomic artifacts, and CLI options.
- `run_train_phaseC_sd_temporal_250.ps1`: one-command train/evaluate pipeline and `-EvaluateOnly` recovery.
- `README.md`: command, artifact lifecycle, recovery command, and gate documentation.

Expected tests:

- temporal sequence success, threshold boundary, deterministic ranking, and first-action labeling;
- unsafe first placement, unsafe second placement, unreachable second placement, duplicate pressure, ammo exhaustion, and Sudden Death closure rejection;
- interception fallback and no unsafe bomb fallback;
- partial evaluation creation, atomic replacement, resume equivalence, mismatch rejection, final validation, and cleanup;
- launcher normal mode, evaluation-only mode, interruption recovery, fail-fast behavior, exact configuration, and gate boundaries;
- preserved RuleBot, environment rewards, safety mask, Schedule E, target ranking, and old-checkpoint behavior.

Tests must use short deterministic fixtures and fake launcher programs. They must not execute real 250-episode training or real 200-game evaluation.

## 8. Error Handling

- Planner failure or no valid sequence falls back to existing safe guidance; it never bypasses the action mask.
- Evaluation keeps a valid partial artifact on interruption or handled failure.
- Invalid or mismatched partial artifacts are never silently discarded or overwritten.
- Final output is not considered valid unless its configuration and game count match the request.
- Existing output directories and final evaluation files are not overwritten in normal training mode.
- Missing warm-start or best checkpoints stop before evaluation.

## 9. Acceptance Criteria

Implementation is complete when:

1. Independent acceptance tests have a valid RED baseline before production edits.
2. All temporal planner safety and uniqueness tests pass.
3. An interrupted deterministic evaluation leaves a valid partial artifact.
4. Resuming produces byte-equivalent game records and equivalent aggregate metrics to an uninterrupted evaluation.
5. The launcher can recover with `-EvaluateOnly` without retraining.
6. The launcher creates and validates `evaluation-200.json` after exactly 200 games.
7. Existing repository tests and the new acceptance suite pass.
8. No real training is started during implementation verification.
9. No commit or push is performed unless separately requested.

## 10. Rollout Decision

After implementation and automated verification, training starts only on an explicit user request. The resulting checkpoint passes Phase C only if all five final evaluation gates pass on the fresh 200-game seed. Otherwise the run is analyzed once and the project does not continue blind hyperparameter tuning.
