# SD5.3 Dense Search Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible SD5.3 runner that distils informative kill-search decisions into a pure 13-channel CNN, then validates and one-shot tests it without changing the game, RuleBot, or search implementation.

**Architecture:** Add `run_sd53_pilot.py` as a stage-based experiment runner. `collect` writes resumable per-seed-shard `.npz` data; `train` loads the fixed split, applies only verified board symmetries, and fine-tunes F with decision soft-label CE plus anchor KL; `validate` selects only epochs 2/4/8 by the frozen rule; `test` creates a one-shot seed-43000 artifact and packages results.

**Tech Stack:** Python 3.11, NumPy, PyTorch CUDA/CPU, `unittest`, existing Bomberman engine, `evaluate`, `kill_search`, `benchmark_t3`, `progress`, and `pilot_two_bomb` helpers.

**Spec:** `C:\Users\DINH LOC\OneDrive\Desktop\2026-10-18-sd5-3-dense-search-distillation-design.md`

## Global Constraints

- Preserve `bomberman_env.py`, `rule_based_bot.py`, `kill_search.py`, the 13-channel CNN architecture, and Schedule E (`475/25/40`) unchanged.
- Use checkpoint F and enforce SHA256 `0315a56f09079333d936db5f55587093f1b6415ed6355d2ef6ae2ab63f8b1483` before every stage.
- Collection uses board seeds `200000..201499`, each from both learner corners; validation is seed `21000`; test opens seed `43000` only once.
- Collect 3000 games, with Schedule E enabled deterministically on 75% of board seeds; train for 8 epochs with Adam `lr=1e-4`, batch 256, 50/50 decision/anchor batches, and anchor lambda `1.0`.
- Candidates are epochs 2, 4, 8 only. Select highest validation ActiveWR with suicide rate no greater than F plus 1pp.
- Test gate is ActiveWR delta >= 5pp, its paired-bootstrap CI95 lower bound > 0, and suicide delta <= 1pp, under Schedule E.
- Keep every stage resumable/reproducible, record normalized-LF source hashes plus runtime provenance, use `progress.py`, and never overwrite an output or rerun an opened test.
- Do not launch the long collect/train/validate/test commands automatically; only execute bounded smoke and unit checks.

## Review Focus

- A resumed collection must not append duplicate shard rows, nor silently use a shard whose seed range or source hash differs from registration.
- Legal-action masking must exclude illegal logits both in decision CE and anchor KL; a zero-probability target must never receive a gradient.
- The train/held-out split must group by board seed, not individual samples, including when each seed appears from both corners.
- Symmetry augmentation must be disabled if the measured transformed-search agreement is below 90%, rather than assuming RuleBot tie-breaking is symmetric.
- `test_opened.json` must be atomically exclusive before any evaluation starts, so interruption cannot permit a second test run.

### Task 1: Define SD5.3 registration, immutable configuration, and pure data helpers

**Files:**
- Create: `run_sd53_pilot.py`
- Create: `test_run_sd53_pilot.py`

**Interfaces:**
- Produces `config_for(output: Path, smoke: bool) -> dict`, `normalized_sha256(path: Path) -> str`, `gate(paired: dict) -> dict`, and `split_board_seeds(samples: np.ndarray, heldout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]`.
- Produces `masked_soft_target(scores: np.ndarray, mask: np.ndarray) -> np.ndarray` for later training.

- [ ] **Step 1: Write independent failing acceptance tests for normalized hashes, soft targets, seed-group splits, and gate boundaries**

Test both tied best actions (equal positive target mass), masked-out actions, no overlapping board seeds, SHA parity for CRLF/LF files, and every gate boundary.

- [ ] **Step 2: Run the targeted test file and record the RED baseline**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot -q`

Expected: FAIL because `run_sd53_pilot` and its public helpers do not yet exist.

- [ ] **Step 3: Implement immutable constants, configuration registration, newline-normalized hashing, target construction, split, and gate helpers**

Use the frozen F checksum, Schedule E, seed ranges, smoke reductions, and non-overwrite checks from the spec. Keep helpers deterministic and free of environment mutation.

- [ ] **Step 4: Re-run the targeted tests**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot -q`

Expected: PASS for the new pure-helper cases.

### Task 2: Build and verify spatial symmetry handling

**Files:**
- Modify: `run_sd53_pilot.py`
- Modify: `test_run_sd53_pilot.py`

**Interfaces:**
- Produces `augment_observation(obs: np.ndarray, rotation: int, mirror: bool) -> np.ndarray` and `augment_action_vector(values: np.ndarray, rotation: int, mirror: bool) -> np.ndarray`.
- Consumes `benchmark_t3.transform` and `benchmark_t3.transform_action` as the canonical coordinate/action mapping.

- [ ] **Step 1: Add failing tests for all eight transforms of observation planes, masks, scores, and action labels**

Use a unique-value 13x9x9 tensor plus a real environment state. Assert the spatial planes transform exactly, scalar planes remain spatially constant, and transformed action scores are indexed by the transformed legal action.

- [ ] **Step 2: Run symmetry tests and record RED**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot.SymmetryTests -q`

Expected: FAIL because no SD5.3 symmetry helpers exist.

- [ ] **Step 3: Implement augmentation and the measured symmetry-agreement audit**

Reconstruct transformed environments only for the audit. Write agreement and enabled/disabled state to `registration.json`; enable augmentation only when agreement is at least 90% over at least 200 decision samples.

- [ ] **Step 4: Re-run symmetry tests**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot.SymmetryTests -q`

Expected: PASS.

### Task 3: Implement shard-safe decision and anchor collection

**Files:**
- Modify: `run_sd53_pilot.py`
- Modify: `test_run_sd53_pilot.py`

**Interfaces:**
- Produces `collect(output: Path, smoke: bool, workers: int, resume: bool) -> dict` and worker-safe shard records under `output/collection/`.
- Each decision record carries observation, mask, per-action search scores, F sampled action, executed action, board seed, corner, and step; each anchor record carries observation, mask, F logits, board seed, corner, and step.

- [ ] **Step 1: Add failing collection tests for informative-only decisions, 5% seeded anchors, deterministic Schedule-E assignment, and resume without duplicate shards**

Use a patched minimal evaluator/search boundary so tests assert persisted artifacts, not mocked internal call counts. Verify a completed shard is skipped only when its registration matches.

- [ ] **Step 2: Run collection tests and record RED**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot.CollectionTests -q`

Expected: FAIL because the collection stage is absent.

- [ ] **Step 3: Implement collect using fixed seed shards and `progress.parallel_map`**

Alternate learner corners per board seed, invoke F plus `search_action`, preserve the real game RNG, write one atomically-completed `.npz` per shard, and aggregate only after validating all shard metadata. Store collection-only diagnostic rates separately from gate results.

- [ ] **Step 4: Re-run collection tests**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot.CollectionTests -q`

Expected: PASS.

### Task 4: Implement supervised distillation with held-out metrics

**Files:**
- Modify: `run_sd53_pilot.py`
- Modify: `test_run_sd53_pilot.py`

**Interfaces:**
- Produces `train(output: Path, smoke: bool, device: str) -> dict`.
- Saves `epoch_2.pt`, `epoch_4.pt`, `epoch_8.pt`, held-out metrics, and model/data provenance below `output/train/`.

- [ ] **Step 1: Add failing tests for masked soft CE, F-to-model anchor KL, balanced batches, epoch checkpoint schedule, and held-out metrics**

Assert CE accepts all tied best legal actions, anchor KL has no contribution from invalid actions, value-head parameters receive no gradient, and all samples from a board seed land in exactly one partition.

- [ ] **Step 2: Run distillation tests and record RED**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot.DistillationTests -q`

Expected: FAIL because no distillation stage exists.

- [ ] **Step 3: Implement F-initialized policy/CNN fine-tuning**

Use Adam with the frozen batch, epoch, LR, CE, and KL values. Recompute masked logits consistently, leave the value-head loss out, emit required held-out metrics every epoch, and retain only required candidate checkpoints.

- [ ] **Step 4: Re-run distillation tests**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot.DistillationTests -q`

Expected: PASS.

### Task 5: Implement frozen validation, one-shot test, and smoke workflow

**Files:**
- Modify: `run_sd53_pilot.py`
- Modify: `test_run_sd53_pilot.py`

**Interfaces:**
- Produces `validate(output: Path, smoke: bool, workers: int) -> dict` and `test(output: Path, smoke: bool, workers: int) -> dict`.
- Writes `validation.json`, `selected.json`, `test_opened.json`, `summary.json`, per-model evaluations, and `output-send.zip` only after successful test completion.

- [ ] **Step 1: Add failing tests for validation selection safety, test-open exclusivity, F/search reporting, and the full smoke sequence**

Verify validation never chooses an unsafe candidate, test creation rejects a second invocation before re-evaluation, and test results include F, selected pure CNN, and F+search under both Schedule E and SD-off while gate uses only the pure CNN Schedule-E pair.

- [ ] **Step 2: Run stage tests and record RED**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot.StageTests -q`

Expected: FAIL because stage functions do not yet exist.

- [ ] **Step 3: Implement validation/test and CLI dispatch**

Use the existing seeded evaluator and paired bootstrap. Open `test_opened.json` with exclusive creation before work, use progress for all long loops, then package the completed output. Expose only `collect`, `train`, `validate`, and `test` stages with `--smoke`, `--resume` (collect only), `--device`, and `--workers` CLI options.

- [ ] **Step 4: Re-run targeted tests and the bounded end-to-end smoke workflow**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot -q`

Then run the four `--smoke` stages in a temporary output folder and verify a single valid ZIP is produced.

Expected: PASS; do not run the full experiment.

### Task 6: Record the SD5.3 implementation status and verify regressions

**Files:**
- Modify: `RESULTS.md`
- Modify: `IMPROVEMENTS.md`
- Modify: `PLAN_SUDDEN_DEATH.md`
- Modify: `PLAN_BOMBERMAN.md`

- [ ] **Step 1: Record implementation-only status without inventing experiment results**

Document the runner, fixed protocol, tests/smoke evidence, unrun full stages, and the frozen decision rule in Vietnamese following nearby entries.

- [ ] **Step 2: Run the SD5.3 targeted tests and repository regression suite**

Run: `.\.venv\Scripts\python.exe -m unittest test_run_sd53_pilot -q`

Run: `.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py" -q`

Expected: all executable tests pass; report skipped tests and any pre-existing failure explicitly.

- [ ] **Step 3: Inspect the diff and update any applicable codebase map**

Run: `git diff --check` and inspect the changed tracker/document sections. No separate codebase overview currently exists, so do not create one solely for this feature.
