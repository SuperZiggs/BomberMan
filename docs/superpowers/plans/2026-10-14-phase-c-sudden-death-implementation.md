# Phase C Sudden Death Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional shrinking-ring Sudden Death mode and a checkpoint-compatible Phase C-SD fine-tuning path without changing default gameplay.

**Architecture:** `BombermanFFAEnv` owns ring timing, closure, warning observations and death attribution. The shared survival planner consumes the environment's closure schedule, while trainer and evaluator only configure the mode and report its outcomes. Existing 12-channel checkpoints are expanded to 13 channels through the existing `smart_load_weights` migration path.

**Tech Stack:** Python 3.11, NumPy, PyTorch, `unittest`, PowerShell

**Spec:** `docs/superpowers/specs/2026-10-14-phase-c-sudden-death-design.md`

## Global Constraints

- Sudden Death is disabled by default; existing commands keep their current game rules.
- Observation channel 12 is the only new model input; existing channel meanings and ordering stay unchanged.
- Death by ring never produces bomb kill credit, ActiveWR or terminal win reward.
- Existing checkpoints, metrics and run directories are never overwritten.
- The launcher prints directly to PowerShell and creates no transcript or redirected terminal log.
- Do not change bomb, wall, kill or item rewards.
- Do not commit changes unless the user explicitly requests a commit.

## Review Focus

- Multiple rings becoming due after a large step jump must close in order without leaving stale bombs, flames or power-ups; covered in Task 2 environment tests.
- A ring closure on the same step as a bomb explosion must attribute the caught agent only to Sudden Death; covered in Task 2 environment tests.
- Warning timing at exactly `warning_steps`, one step before closure and immediately after closure must be correct; covered in Task 2 observation tests.
- Survival fallback must not re-enable a move into a cell that closes before the agent can leave; covered in Task 3 mask tests.
- Loading a malformed or newer checkpoint must not silently discard incompatible non-input tensors; covered in Task 4 migration tests.

---

### Task 1: Independent Acceptance Suite

**Files:**
- Create: `docs/superpowers/test-matrices/2026-10-14-phase-c-sudden-death.md`
- Create: `test_sudden_death.py`

**Interfaces:**
- Consumes: approved requirements from the design spec; current public behavior of `BombermanFFAEnv`, `get_action_mask`, `smart_load_weights`, `train`, and `evaluate_seeded`
- Produces: an independently authored acceptance matrix and executable tests with a recorded RED baseline

- [ ] **Step 1: Freeze the requirements packet**

Give the independent test author only the design spec's behavior, acceptance criteria, public contracts, current code and existing test command. Do not provide implementation algorithms from Tasks 2–5.

- [ ] **Step 2: Author the traceability matrix**

Map every acceptance criterion to environment, observation, mask, migration, trainer or evaluator coverage, including boundary and invalid-configuration cases.

- [ ] **Step 3: Add executable acceptance tests**

Tests must assert observable behavior: closure timing and cleanup, death attribution, warning values, disabled-mode equivalence, mask behavior, migrated outputs, seeded curriculum reproducibility and separated evaluator rates.

- [ ] **Step 4: Record a valid RED baseline**

Run: `venv\Scripts\python.exe -m unittest -v test_sudden_death.py`

Expected: tests fail because Sudden Death parameters/contracts are absent, not because of syntax, imports or unavailable dependencies.

### Task 2: Environment, Ring Lifecycle and Observation

**Files:**
- Modify: `bomberman_env.py:59`
- Modify: `bomberman_env.py:98`
- Modify: `bomberman_env.py:137`
- Modify: `bomberman_env.py:182`
- Modify: `bomberman_env.py:298`
- Modify: `bomberman_env.py:458`
- Test: `test_sudden_death.py`

**Interfaces:**
- Consumes: existing grid, bomb, flame, power-up and agent state
- Produces: `BombermanFFAEnv(..., sudden_death=False, sudden_death_start=350, sudden_death_interval=20, sudden_death_warning_steps=40)`, `_next_sudden_death_closure() -> tuple[int, set[tuple[int, int]]] | None`, `_apply_sudden_death() -> list[int]`, `_sudden_death_warning_map() -> np.ndarray`, and `info["sudden_death_deaths"]`

- [ ] **Step 1: Verify environment acceptance tests are RED**

Run the environment and observation test classes from `test_sudden_death.py`; confirm missing constructor arguments or missing closure behavior causes the failure.

- [ ] **Step 2: Add validated Sudden Death configuration and state**

Increase `OBS_CHANNELS` to `13`; add the four constructor parameters, timing validation and `_sudden_death_ring` reset state without changing disabled-mode grid generation.

- [ ] **Step 3: Implement ring geometry and closure schedule**

Add exact ring-cell generation, target-ring calculation and `_next_sudden_death_closure()` using relative environment ticks.

- [ ] **Step 4: Implement start-of-step closure**

Call `_apply_sudden_death()` immediately after incrementing `step_count`; kill caught agents, remove caught bombs while restoring owner capacity, clear transient cell state, and replace the closed ring with rigid walls.

- [ ] **Step 5: Expose death attribution**

Return caught IDs only through `info["sudden_death_deaths"]`; do not add them to `info["kills"]` or `self_blast_deaths`.

- [ ] **Step 6: Add warning channel 12**

Fill the channel with the spec's linear countdown for only the next ring; keep it all zero when disabled or outside the warning window.

- [ ] **Step 7: Run environment tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_sudden_death.SuddenDeathEnvironmentTests test_sudden_death.SuddenDeathObservationTests`

Expected: PASS.

### Task 3: Temporal Safety Integration

**Files:**
- Modify: `rule_based_bot.py:88`
- Modify: `rule_based_bot.py:123`
- Test: `test_sudden_death.py`
- Test: `test_combat.py`

**Interfaces:**
- Consumes: `BombermanFFAEnv._next_sudden_death_closure()` and existing bomb `burns`/`blocked` tensors
- Produces: `_survival_map` and `_safe_action_mask` that treat future ring cells as blocked from their closure tick onward

- [ ] **Step 1: Verify safety tests are RED**

Run the mask acceptance tests and confirm a currently accepted route crosses a cell before its scheduled closure.

- [ ] **Step 2: Project closure into temporal blocking**

Merge only closure cells within the current survival horizon into the existing `blocked` tensor; do not mutate the environment or bomb forecast.

- [ ] **Step 3: Preserve disabled-mode behavior and fallback safety**

Ensure recursive/fallback mask paths consume the same closure projection and cannot re-enable a doomed move; return exactly the previous mask when Sudden Death is disabled.

- [ ] **Step 4: Run safety and combat tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_sudden_death.SuddenDeathSafetyTests test_combat.py`

Expected: PASS with no changes to existing combat assertions.

### Task 4: Checkpoint Migration and Phase C-SD Curriculum

**Files:**
- Modify: `train_actor_critic.py:201`
- Modify: `train_actor_critic.py:266`
- Modify: `train_actor_critic.py:303`
- Modify: `train_actor_critic.py:313`
- Modify: `train_actor_critic.py:430`
- Modify: `train_actor_critic.py:655`
- Modify: `train_actor_critic.py:1043`
- Test: `test_sudden_death.py`
- Test: `test_combat.py:592`

**Interfaces:**
- Consumes: 12-channel state dictionaries, environment Sudden Death configuration and seeded training RNG
- Produces: `train(..., sudden_death_probability=0.0, sudden_death_start=350, sudden_death_interval=20, sudden_death_warning_steps=40)`, per-episode curriculum selection and Sudden Death metrics

- [ ] **Step 1: Verify migration and curriculum tests are RED**

Run the migration, invalid-probability, seeded-curriculum and per-episode metrics acceptance tests.

- [ ] **Step 2: Harden `smart_load_weights` migration**

Copy 12 input channels into the 13-channel convolution, zero channel 12, preserve existing board interpolation, and raise on unsupported incompatible tensors instead of silently retaining random weights.

- [ ] **Step 3: Add curriculum parameters and validation**

Validate probability in `[0, 1]`; avoid consuming curriculum RNG when probability is exactly `0` or `1`; configure the environment before every reset.

- [ ] **Step 4: Track separated episode outcomes**

Add `sudden_death_enabled`, `sudden_death_deaths`, `sudden_death_win`, `active_win`, `passive_win` and `draw` to callback/metrics records. Preserve `_combat_reward` so a ring-only win receives no terminal bonus.

- [ ] **Step 5: Add CLI and config persistence**

Expose the four Phase C-SD options and ensure `config.json` records them through the existing argument serialization.

- [ ] **Step 6: Run migration and trainer tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_sudden_death.SuddenDeathCheckpointTests test_sudden_death.SuddenDeathTrainingTests test_combat.py`

Expected: PASS, including the existing old-checkpoint output-equivalence test.

### Task 5: Evaluator Reporting and PowerShell Launcher

**Files:**
- Modify: `evaluate.py:55`
- Modify: `evaluate.py:187`
- Modify: `evaluate.py:238`
- Modify: `evaluate.py:412`
- Create: `run_train_phaseC_sd.ps1`
- Modify: `README.md`
- Test: `test_sudden_death.py`

**Interfaces:**
- Consumes: `info["sudden_death_deaths"]`, existing kill/self-death data and CLI timing options
- Produces: evaluator fields `active_win_rate`, `sudden_death_win_rate`, `passive_win_rate`, `suicide_rate`, `sudden_death_death_rate`, `draw_rate`; a reproducible 500-episode pilot launcher

- [ ] **Step 1: Verify evaluator tests are RED**

Run the evaluator acceptance tests and confirm the missing separated rates/configuration cause expected failures.

- [ ] **Step 2: Extend match and seeded summaries**

Count ring deaths and ring-only wins separately at agent and aggregate levels; preserve current fields for compatibility and include Sudden Death configuration in seeded JSON.

- [ ] **Step 3: Add evaluator CLI configuration**

Add `--sudden_death`, `--sudden_death_start`, `--sudden_death_interval` and `--sudden_death_warning_steps` to seeded and tournament environment creation.

- [ ] **Step 4: Add the Phase C-SD pilot launcher**

Create `run_train_phaseC_sd.ps1` using `venv\Scripts\python.exe`, 500 episodes, seed `2026101401`, validation seed `20000`, probability `0.75`, start `350`, interval `20`, warning `40`, warm-start `runs\pressure-C-next\final_phaseC.pt`, and a new `runs\phaseC-sd-pilot` directory. Print directly to the visible terminal without log redirection.

- [ ] **Step 5: Document usage and interpretation**

Add concise README commands and state that Sudden Death wins do not count as ActiveWR or justify replacing the current checkpoint.

- [ ] **Step 6: Run evaluator tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_sudden_death.SuddenDeathEvaluationTests`

Expected: PASS.

### Task 6: Integrated Verification

**Files:**
- Verify: all modified files

**Interfaces:**
- Consumes: completed Tasks 1–5
- Produces: evidence that the feature is safe to hand off for a user-started pilot

- [ ] **Step 1: Run the full test suite**

Run: `venv\Scripts\python.exe -m unittest -v test_combat.py test_sudden_death.py`

Expected: all tests PASS.

- [ ] **Step 2: Run syntax checks**

Run: `venv\Scripts\python.exe -m py_compile bomberman_env.py rule_based_bot.py train_actor_critic.py evaluate.py visualize.py test_combat.py test_sudden_death.py`

Expected: exit code `0`.

- [ ] **Step 3: Run a temporary smoke training**

Run a two-episode Phase C-SD training with probability `1`, early closure timings suitable for the short episode, `run_report=False`, and a temporary checkpoint directory.

Expected: finite losses, metrics include separated Sudden Death fields, and 13-channel checkpoints are created outside permanent run directories.

- [ ] **Step 4: Validate the launcher without starting the long pilot**

Parse `run_train_phaseC_sd.ps1`, resolve its interpreter/checkpoint paths and validate argument names; do not execute the 500-episode training.

- [ ] **Step 5: Run repository hygiene checks**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; only intended source, test, documentation and launcher files are modified or untracked.

- [ ] **Step 6: Report evidence**

Report the independent RED baseline, targeted GREEN commands, full-suite result, smoke result, files changed and the exact user command to start the pilot.
