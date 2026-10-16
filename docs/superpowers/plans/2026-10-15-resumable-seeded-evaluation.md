# Resumable Seeded Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make seeded evaluation observable, atomically persisted, resumable after interruption, and rerunnable without retraining.

**Architecture:** Keep deterministic game execution in `evaluate_seeded`, add validated initial-game resume and per-game checkpoint callbacks, then wrap it in a CLI artifact lifecycle that writes a compatible partial file and atomically promotes a complete final file. Extend the Phase C launcher with evaluation-only recovery and strict final-artifact validation.

**Tech Stack:** Python 3, NumPy, PyTorch, JSON, `pathlib`/`os.replace`, PowerShell, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-15-phase-c-temporal-trap-resumable-evaluation-design.md`

## Global Constraints

- Preserve deterministic board seed and learner-corner rotation for every global episode index.
- Preserve existing `evaluate_seeded` callers when resume options are omitted.
- Write `evaluation-200.partial.json` atomically every 10 completed games.
- Create `evaluation-200.json` only after exactly 200 validated game records exist.
- Never silently discard or overwrite an incompatible partial artifact.
- Preserve model-gate failures as process exit code zero; operational failures remain nonzero.
- Do not execute real 200-game evaluation in automated tests.
- Do not commit or push unless separately requested.

## Review Focus

- A partial file with duplicate or out-of-order episode identities must be rejected, not resumed; pin in Task 2.
- A checkpoint replaced at the same path must fail identity validation; pin in Task 1.
- Interruption between temporary write and `os.replace` must preserve the previous valid partial; pin in Task 1.
- A complete but configuration-mismatched final JSON must not be accepted by the launcher; pin in Task 4.
- `-EvaluateOnly` with a missing best checkpoint must fail before invoking Python; pin in Task 4.

---

### Task 1: Atomic Evaluation Artifact Primitives

**Files:**
- Create: `test_resumable_evaluation.py`
- Modify: `evaluate.py:203`

**Interfaces:**
- Consumes: checkpoint path and evaluation configuration values already accepted by the seeded CLI.
- Produces: `_checkpoint_identity(path: str) -> dict`, `_atomic_write_json(path: str, payload: dict) -> None`, `_load_partial_games(path: str, expected: dict) -> list[dict]`.

- [ ] **Step 1: Delegate the approved requirements to an independent test author**

Require a traceable matrix and a valid RED baseline for checkpoint identity, atomic replacement, matching partial load, mismatch rejection, malformed JSON, and preserved prior partial content. The test author may edit only tests and its matrix.

- [ ] **Step 2: Run the independent primitive tests and record RED**

Run: `venv\Scripts\python.exe -m unittest -v test_resumable_evaluation.EvaluationArtifactPrimitiveAcceptance`

Expected: failures identify missing artifact helpers; no import, fixture, or dependency errors.

- [ ] **Step 3: Implement `_checkpoint_identity(path: str) -> dict`**

Return normalized absolute path, byte size, and nanosecond modification time. Raise `FileNotFoundError` for a missing checkpoint.

- [ ] **Step 4: Implement `_atomic_write_json(path: str, payload: dict) -> None`**

Write UTF-8 JSON to `<path>.tmp`, flush and close it, then use `os.replace`. Do not delete an existing valid target before replacement.

- [ ] **Step 5: Implement `_load_partial_games(path: str, expected: dict) -> list[dict]`**

Validate schema version, checkpoint identity, evaluation configuration, completed count, requested count, and the games list. Raise `ValueError` without modifying the file on any mismatch.

- [ ] **Step 6: Run primitive tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_resumable_evaluation.EvaluationArtifactPrimitiveAcceptance`

Expected: all primitive tests pass.

- [ ] **Step 7: Record task verification without committing**

Record the command and result in the execution ledger; leave changes uncommitted.

### Task 2: Deterministic Resume Inside `evaluate_seeded`

**Files:**
- Modify: `evaluate.py:203`
- Test: `test_resumable_evaluation.py`

**Interfaces:**
- Consumes: `_load_partial_games(...)` from Task 1.
- Produces: `evaluate_seeded(..., initial_games: list[dict] | None = None, on_checkpoint=None) -> dict | None` while preserving all existing parameters and defaults.

- [ ] **Step 1: Add failing resume-equivalence tests**

Test names and assertions:

- `test_resume_skips_completed_indices_and_matches_uninterrupted_games`: a short four-game run resumed from two validated records must produce identical ordered games and aggregates.
- `test_resume_rejects_duplicate_or_wrong_episode_identity`: duplicate `(seed, aid)` or an identity inconsistent with the global index raises `ValueError`.
- `test_checkpoint_callback_receives_ordered_completed_games`: callback counts increase monotonically and records remain ordered.
- `test_existing_callers_without_resume_keep_identical_results`: old signature usage remains deterministic.

- [ ] **Step 2: Run resume tests RED**

Run: `venv\Scripts\python.exe -m unittest -v test_resumable_evaluation.SeededResumeAcceptance`

Expected: failures identify missing `initial_games` and `on_checkpoint` behavior.

- [ ] **Step 3: Extend `evaluate_seeded` without changing default behavior**

Validate initial records against the deterministic index mapping `board_seed = seed + episode_index // num_agents` and `aid = episode_index % num_agents`. Start the loop at `len(initial_games)`, append new records, and call `on_checkpoint(games, completed, total)` after each game.

- [ ] **Step 4: Extract one summary helper shared by resumed and uninterrupted paths**

Define `_summarize_seeded_games(games: list[dict], num_agents: int) -> dict` and use it for the final return value. Reject empty or incomplete caller-provided records before division.

- [ ] **Step 5: Run resume tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_resumable_evaluation.SeededResumeAcceptance`

Expected: all resume tests pass.

- [ ] **Step 6: Run preserved evaluator regressions**

Run: `venv\Scripts\python.exe -m unittest -v test_combat.CombatChecks.test_seeded_evaluation_reuses_boards_rotates_corners_and_restores_rng test_sudden_death.SuddenDeathEvaluationAcceptance.test_seeded_evaluator_accepts_sd_config_and_separates_all_rates`

Expected: both tests pass.

- [ ] **Step 7: Record task verification without committing**

Record commands and results in the execution ledger; leave changes uncommitted.

### Task 3: CLI Partial Lifecycle and Progress

**Files:**
- Modify: `evaluate.py:456`
- Test: `test_resumable_evaluation.py`

**Interfaces:**
- Consumes: Task 1 artifact helpers and Task 2 resume-capable evaluator.
- Produces: CLI options `--resume_partial` and `--partial_every` with default `10`; final output promotion and partial cleanup.

- [ ] **Step 1: Add failing CLI lifecycle tests using a fake short evaluator**

Test names and assertions:

- `test_cli_writes_atomic_partial_at_requested_cadence`: partial payload contains schema, identity, configuration, and completed games.
- `test_handled_interrupt_persists_latest_completed_records`: cancellation between cadence boundaries writes the latest valid partial before returning.
- `test_cli_resume_finishes_and_removes_partial`: compatible partial resumes and final JSON replaces temporary output.
- `test_cli_mismatch_preserves_partial_and_exits_nonzero`: changed seed, checkpoint identity, or game count fails without mutation.
- `test_cli_prints_flushed_progress`: output includes `10/200`-style progress and the saved final path.
- `test_final_output_requires_exact_unique_game_count`: incomplete or duplicate records never promote.

- [ ] **Step 2: Run CLI tests RED**

Run: `venv\Scripts\python.exe -m unittest -v test_resumable_evaluation.SeededEvaluationCLIAcceptance`

Expected: failures identify missing flags and lifecycle behavior.

- [ ] **Step 3: Implement `run_seeded_evaluation_to_file(...) -> dict`**

Place it in `evaluate.py`. Derive `.partial.json` and `.tmp` paths from the requested final output, load only compatible partial records, write every `partial_every` games, atomically promote the final payload, and remove only the compatible partial after success.

- [ ] **Step 4: Wire CLI flags and progress output**

For seeded mode with `--output`, call `run_seeded_evaluation_to_file`. Print progress every configured cadence with `flush=True`. Keep the existing direct in-memory path when no output is requested.

- [ ] **Step 5: Run CLI lifecycle tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_resumable_evaluation.SeededEvaluationCLIAcceptance`

Expected: all CLI tests pass.

- [ ] **Step 6: Record task verification without committing**

Record command and result in the execution ledger; leave changes uncommitted.

### Task 4: Launcher Recovery and Final Validation

**Files:**
- Modify: `run_train_phaseC_sd_target_750.ps1:1`
- Modify: `test_phase_c_sd_target_750.py:555`
- Modify: `README.md:81`

**Interfaces:**
- Consumes: `evaluate.py --resume_partial --partial_every 10` from Task 3.
- Produces: `run_train_phaseC_sd_target_750.ps1 -EvaluateOnly` and strict final JSON validation.

- [ ] **Step 1: Add failing staged-launcher tests**

Test names and assertions:

- `test_evaluate_only_skips_training_and_uses_existing_best`: only the fake evaluator is called.
- `test_evaluate_only_missing_best_fails_before_python`: no subprocess call occurs.
- `test_launcher_passes_resume_and_progress_flags`: exact options are present once.
- `test_launcher_rejects_mismatched_or_incomplete_final_json`: wrong seed, settings, or `num_episodes != 200` exits nonzero.
- `test_launcher_announces_evaluation_stage_before_invocation`: output warns not to close the window.

- [ ] **Step 2: Run launcher recovery tests RED**

Run: `venv\Scripts\python.exe -m unittest -v test_phase_c_sd_target_750.PhaseCSDTarget750LauncherAcceptance`

Expected: new recovery assertions fail while existing launcher tests remain runnable.

- [ ] **Step 3: Add `param([switch]$EvaluateOnly)` and separate preflight branches**

Normal mode keeps no-overwrite training behavior. Evaluation-only mode requires the existing run directory and best checkpoint, skips training, and allows a missing final JSON or compatible partial file.

- [ ] **Step 4: Add visible stage messages and resumable evaluation options**

Print `[Stage 1/2] Training`, `[Stage 2/2] Evaluating`, and the final saved path. Pass `--resume_partial --partial_every 10`.

- [ ] **Step 5: Validate final artifact before gate reporting**

Require seed `31000`, board 9, agents 2, max steps 500, Schedule E, `num_episodes = 200`, and all five gate fields. Fail nonzero on artifact mismatch; preserve exit zero for metric gate failure.

- [ ] **Step 6: Document normal and recovery commands**

Add `run_train_phaseC_sd_target_750.ps1 -EvaluateOnly`, partial-file behavior, and the warning not to delete a compatible partial file.

- [ ] **Step 7: Run launcher tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_phase_c_sd_target_750.PhaseCSDTarget750LauncherAcceptance`

Expected: all launcher tests pass without real training or evaluation.

- [ ] **Step 8: Run Plan 1 verification**

Run:

```powershell
venv\Scripts\python.exe -m unittest -v test_resumable_evaluation.py test_phase_c_sd_target_750.py
venv\Scripts\python.exe -m unittest -v
venv\Scripts\python.exe -m py_compile evaluate.py
git diff --check
```

Expected: all tests pass, compilation succeeds, and diff check reports no errors. Do not launch real evaluation.

- [ ] **Step 9: Record plan completion without committing**

Record final commands and counts in the execution ledger; leave all changes uncommitted and unpushed.
