# S4-11 CNN Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Preserve the independent acceptance-test requirement in `independent-test-first`; if no separate test-author agent is available, obtain the user's approval for the documented fallback before writing acceptance tests or production code.

**Goal:** Produce and evaluate an 11×11 S4-derived pure CNN against fixed CNN opponents while guarding RuleBot performance.

**Architecture:** Add a separate registered experiment runner and an operator launcher. Reuse existing environment, search, model loader and statistical helpers without editing registered historical source. Stage outputs are source/checkpoint/config locked and the final test seed opens at most once.

**Tech Stack:** Python 3.11, NumPy, PyTorch, unittest, Windows multiprocessing and PowerShell, existing `progress.py`; CUDA only for train where available.

**Spec:** `docs/superpowers/specs/2026-10-31-s4-map11-cnn-distillation-design.md` (approved 31/10/2026).

## Global Constraints

- 11×11, 13 channels, 2 agents, 500 max steps, schedule E `475/25/40`; collect/pilot probability 0.75, evaluation probability 1.
- Fixed S4 starting checkpoint `runs/sd10-20261026/train/epoch_12.pt`, SHA `90d1d0e70c64f879e07f85c68c003969d082043b3cd8065314a3c35faf62bece`; preserve original.
- Collect opponents S2/G/S3-11, each 333 boards × two corners; RuleBot is a pilot/evaluation opponent, not collection data.
- Candidate seed blocks, as proposed in the spec, must be checked again against actual run registrations, opened markers and relevant branch ledger before registration; exclude this plan/spec from their own reservation check.
- A pilot has 160 games, full collection 1998 games, validation 8000 games and final test 7200 games; all numbers and guard conditions are fixed before first game.
- Test marker opens once. An interrupted test is `INCOMPLETE`, never silently replayed. Collection resume accepts only missing shards after metadata, seed/corner and hash validation.
- Preserve existing dirty user files and checkpoints. Source changes must not alter historical experiment runners or their registered dependencies.
- Default 5 CPU workers, 1 torch CPU thread per worker. Do not start the full experiment without passing the approved pilot gate and integrity checks.

## Review Focus

1. Used/reserved or overlapping seed intervals: registration must reject them before writing or starting any game; assert fake ledger and opened-marker conflicts, including the SD10 marker format.
2. Opponent identity and global state: search uses the current opponent CNN's deterministic masked argmax, the real opponent samples; restore the previous `OPPONENT_MODEL` even if gameplay raises.
3. Partial/corrupt artifacts and simultaneous runs: validate complete shard content and provenance, preserve valid work, refuse corruption and foreign/stale locks; no automatic destructive cleanup.
4. Masked/unscored `NaN` values: permit them where the SD10 representation permits, reject nonfinite scored actions, anchor logits and model outputs; verify real 11×11 shapes.
5. Test interruptions and selection leakage: create marker only after valid validation selection, refuse all reruns with the marker, select from validation alone, and compare paired outcomes by board including both corners.

## File Structure

- Create `run_s4_map11_distill.py`: experiment config, registration, pilot, collect, train, validation, test and pure statistical decisions.
- Create `test_run_s4_map11_acceptance.py` and an acceptance matrix/RED transcript: independently authored against approved requirements when a test-author agent is available.
- Create `run_s4_map11_pipeline.ps1` and `S4_MAP11_GUIDE.md`: execute validated stages in order, preserve one-time test and expose logs/progress.
- Update `RESULTS.md`, `IMPROVEMENTS.md`, `PLAN_SUDDEN_DEATH.md` and `PLAN_BOMBERMAN.md` with actual evidence as stages complete; do not overwrite user edits.
- Runtime run and test outputs live in fresh directories under `runs/`; never reuse SD10/S3-11 run folders.

## Task 1: Requirements packet and independent RED acceptance tests

**Files:** `docs/superpowers/specs/2026-10-31-s4-map11-cnn-distillation-design.md`, `docs/superpowers/specs/2026-10-31-s4-map11-acceptance-matrix.md`, `test_run_s4_map11_acceptance.py`, RED transcript.

**Interfaces to test:** `config_for(output, *, smoke=False)`, `initialize(output, *, smoke=False)`, `pilot(output, workers=5)`, `collect(output, workers=5, resume=False)`, `train(output, device='cpu')`, `validate(output, workers=5)`, `test(output, workers=5)`, `main(argv=None)` in `run_s4_map11_distill.py`. Functions return serializable dictionaries and raise on invalid/unsafe state.

- [ ] Freeze a requirements-only packet from the approved spec. It must describe observable inputs, outputs, errors, no-go conditions and non-goals without dictating implementation algorithms to the independent author.
- [ ] Test author writes a traceability matrix and runnable acceptance tests covering Review Focus 1–5 and the stage boundaries, using temporary output directories. At least one integration test exercises real 11×11 environment/model interaction with minimal steps; no real pilot, collect, validation or test seed is opened.
- [ ] Run acceptance tests against the baseline and save RED results that fail because the feature is absent, not because the test environment or dependencies are broken. Preserve the test suite unchanged through implementation unless an independent reviewer demonstrates a defect.

## Task 2: Registration, seed protection, lock and collection integrity

**Files:** `run_s4_map11_distill.py`, `test_run_s4_map11_acceptance.py`.

**Interfaces:** `config_for(output, *, smoke=False) -> dict`, `initialize(output, *, smoke=False) -> dict`, `opponent_for(seed: int) -> str`, `collect(output, workers=5, resume=False) -> dict`.

- [ ] Acceptance tests assert 999 boards `860000–860998`, 333 per CNN, 49 × 20 plus 19-board shards, 1998 games and five disjoint seed ranges; malformed seed types, collisions, existing output and workers outside `1..5` are rejected before work.
- [ ] Tests exercise exclusive locks, source/checkpoint hash mismatch, atomic shard writes, wrong game count, duplicate/missing seed/corner, invalid shape and payload, unexpected shard, resume with completed shards, and refusal to overwrite or remove another process's lock.
- [ ] Implement configuration, provenance bundle, seed ledger checks, lock, validated shard and resume operations. Run the corresponding acceptance tests plus `E:\Tools\python.exe -m unittest test_run_sd10 test_run_s3_map11 test_s3_map11_provenance -q`.

## Task 3: Pilot and CNN-aware teacher collection

**Interfaces:** `pilot(output, workers=5) -> dict`; collection must use the registered S4 teacher and the current fixed opponent for each game.

- [ ] Acceptance tests verify all pilot cells and counts, actual 11×11 observations, sampled real opponent versus deterministic opponent-aware teacher, prior hook restoration on success and exception, and no RuleBot game in full collect.
- [ ] Synthetic tests cover `NaN` for unscored actions, reject nonfinite scored labels/anchor logits, and show pilot stops on missing decision labels or bootstrap CI entirely below zero. A zero/weak point estimate alone is not a statistically conclusive failure.
- [ ] Implement 160-game pilot, per-opponent timing/labels and preregistered readiness rule; implement 1998-game full collect using existing `parallel_map` and progress. Verify controlled smoke with an isolated output and distinct smoke seeds, then run the acceptance tests.

## Task 4: Train and validation

**Interfaces:** `train(output, device='cpu') -> dict`, `validate(output, workers=5) -> dict`, `select(summaries) -> tuple`.

- [ ] Acceptance tests assert board-disjoint 90/10 splitting, model initialization from interpolated S4, 16 epochs with checkpoint 4/8/12/16, finite CE/KL, frozen critic, deterministic recorded RNG/config, and refusal on missing/corrupt inputs or existing output.
- [ ] Synthetic candidate metrics test each RuleBot ActiveWR, RuleBot suicide, CNN suicide guard, strict macro ActiveWR improvement and earlier-epoch tie break. Run exactly 8000 games at full configuration; incomplete runs cannot write selection.
- [ ] Implement train and validation; verify actual CUDA availability before choosing GPU and record fallback if CPU. Run acceptance tests and targeted prior SD10/map11 regression tests.

## Task 5: Final test, paired metrics and report

**Interfaces:** `test(output, workers=5) -> dict`, `gate(comparison) -> dict`; raw game artifacts include opponent, learner, seed and corner.

- [ ] Acceptance tests prove all 7200 configured games and coverage, test marker exclusivity, rejection before playing when selection/hash invalid, and no test replay after interruption or completion.
- [ ] Verify three-opponent macro ActiveWR per board/corner, board-level paired bootstrap with 10000 draws, per-opponent metrics, RuleBot ActiveWR/suicide guards and +5pp/positive-CI gate boundaries. S3/S4 held-out opponents and S3-11 learner remain report-only.
- [ ] Implement test/summary and independently recalculate reported metrics from raw games without generating new games. Record known limitations and do not update the served model automatically.

## Task 6: Launch, monitor, record and hand off

**Files:** `run_s4_map11_pipeline.ps1`, `S4_MAP11_GUIDE.md`, local run directory, project trackers.

- [ ] Run acceptance tests, targeted existing regressions and native short smoke. Check syntax, artifacts, output paths, CPU/RAM, seed freshness and no conflicting process; record exact commands and results.
- [ ] Launch registered pilot with a visible terminal. If the preapproved readiness rule passes, collect → train → validate → test; stop on `VALIDATION_STOP`, test interruption or failure. Preserve seed and checkpoint registration across all stages.
- [ ] Set an hourly monitor while a long stage runs; stay quiet without meaningful change and delete it at completion. Report status and stage transitions without silently replaying a test.
- [ ] Validate complete outputs, update RESULTS/IMPROVEMENTS/PLAN, and report model path, 11×11 CNN gain, RuleBot retention, CI/gate, limitations and all test results. Never auto-publish or switch the served model.

## Plan Self-Review

The approved design is covered by tasks 1–6; each Review Focus item is exercised in tasks 1–5. The one-time test rule takes precedence over collection's resume capability. The pilot sample is descriptive; it cannot tune the final gate. Exact seed blocks are proposals until the final ledger audit and registration. No seed or experiment output is created by writing this plan.
