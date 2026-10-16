# Phase C Temporal Trap Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace static two-position pressure supervision with safe temporal two-bomb planning and bounded interception guidance, then provide a verified 250-episode fine-tune launcher.

**Architecture:** Isolate temporal planning in a focused module that consumes the existing environment forecast and safety primitives without mutating the live environment. Integrate the returned first action into the attack teacher, expose diagnostics through training metrics, and launch the bounded fine-tune using the resumable evaluation interface from Plan 1.

**Tech Stack:** Python 3, NumPy, PyTorch, dataclasses, deque-based bounded search, PowerShell, `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-15-phase-c-temporal-trap-resumable-evaluation-design.md`

**Dependency:** Complete `docs/superpowers/plans/2026-10-15-resumable-seeded-evaluation.md` first.

## Global Constraints

- Preserve RuleBot, environment rewards, action safety masking, Schedule E `475/25/40`, and ring-safety coefficient `0.6`.
- Planner search must not mutate the live environment, RNG state, agents, bombs, grid, flames, or power-ups.
- Use at most two learner bomb placements and a bounded default horizon of eight ticks.
- Reject unsafe, unreachable, duplicate-position, duplicate-pressure, and capacity-invalid sequences.
- Preserve existing single-bomb pressure, upgrade, opponent-navigation, and wooden-wall fallbacks.
- Warm-start only from `runs/phaseC-sd-target-750/best_phaseC.pt`; write only to `runs/phaseC-sd-temporal-250`.
- Do not execute real 250-episode training or real 200-game evaluation in automated tests.
- Do not commit or push unless separately requested.

## Review Focus

- Simultaneous counter-bomb placement must not make a sequence appear safe; pin in Task 1.
- A second bomb whose pressure is entirely duplicated by the first must be rejected; pin in Task 1.
- A path that reaches the second placement only after its cell closes in Sudden Death must be rejected; pin in Task 1.
- Interception must not override urgent escape, ring safety, or upgrade collection priority; pin in Task 2.
- Mirrored equivalent states should not diverge because of queue/action iteration order; pin deterministic mirrored behavior in Task 1.

---

### Task 1: Independent Temporal Planner Contract

**Files:**
- Create: `temporal_trap_planner.py`
- Create: `test_temporal_trap_planner.py`
- Create: `docs/superpowers/test-matrices/2026-10-15-temporal-trap-planner.md`

**Interfaces:**
- Consumes: `BombermanFFAEnv`, current danger map, current action mask, `_forecast_blasts`, `_reachable_cells`, `_can_bomb_and_escape`, and action constants.
- Produces: `TemporalTrapPlan`, `TemporalTrapConfig`, and `find_temporal_trap_plan(...)` for Task 2.

- [ ] **Step 1: Delegate planner acceptance tests to an independent test author**

Freeze the approved spec and public interfaces without prescribing the implementation algorithm. Require success, threshold, safety, uniqueness, non-mutation, deterministic, mirrored, counter-bomb, ammo, and Sudden Death boundary scenarios.

- [ ] **Step 2: Define the public planner types in the test contract**

Required signatures:

```python
@dataclass(frozen=True)
class TemporalTrapConfig:
    horizon: int = 8
    max_bombs: int = 2
    min_pressure_gain: float = 0.35

@dataclass(frozen=True)
class TemporalTrapPlan:
    first_action: int
    bomb_positions: tuple[tuple[int, int], ...]
    pressure_gain: float
    completion_tick: int
    escape_margin: int
    predicts_elimination: bool

def find_temporal_trap_plan(
    env: BombermanFFAEnv,
    agent_id: int,
    danger: np.ndarray,
    mask: list[bool],
    config: TemporalTrapConfig = TemporalTrapConfig(),
) -> TemporalTrapPlan | None:
    ...
```

- [ ] **Step 3: Run planner acceptance tests RED**

Run: `venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.py`

Expected: failures identify the missing module/interfaces and behavior; no fixture or dependency errors.

- [ ] **Step 4: Implement immutable bounded search state and live-state snapshot checks**

Represent learner position, tick, proposed bombs, remaining capacity, opponent refuges, first action, and path history. Use copied forecast inputs and validate after every test that the live environment and NumPy/PyTorch RNG states are unchanged.

- [ ] **Step 5: Implement temporally legal movement and placement expansion**

Movement must respect forecast walls, flames, bombs, occupied cells, and due Sudden Death closure. Placement must respect distinct positions, capacity, existing bombs, and combined learner survival including counter-bomb assumptions.

- [ ] **Step 6: Implement additional-pressure and deterministic scoring**

Compute refuge reduction against the same temporal baseline, require the second placement to add unique reduction, then rank by elimination, pressure gain, completion tick, total ticks, escape margin, and stable first-action order.

- [ ] **Step 7: Run planner tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.py`

Expected: all planner tests pass.

- [ ] **Step 8: Record task verification without committing**

Record RED and GREEN evidence in the execution ledger; leave changes uncommitted.

### Task 2: Interception Fallback and Attack-Teacher Integration

**Files:**
- Modify: `temporal_trap_planner.py`
- Modify: `train_actor_critic.py:154`
- Modify: `test_phase_c_attack_750.py:70`
- Test: `test_temporal_trap_planner.py`

**Interfaces:**
- Consumes: `find_temporal_trap_plan(...)` from Task 1.
- Produces: `find_interception_action(env, agent_id, danger, mask, horizon=4) -> int | None` and extended `get_attack_action(...)` temporal options.

- [ ] **Step 1: Add failing interception and priority tests**

Test names and assertions:

- `test_interception_reduces_future_refuges_without_bombing`: chooses a safe choke-point move.
- `test_interception_returns_none_without_strict_reachable_region_gain`: prevents arbitrary movement labels.
- `test_temporal_plan_precedes_static_pressure`: eligible plan supplies its first action.
- `test_urgent_escape_and_ring_safety_remain_higher_priority`: trainer emits only the existing survival label.
- `test_upgrade_and_legacy_navigation_remain_available_without_plan`: preserved fallbacks still operate.

- [ ] **Step 2: Run integration tests RED**

Run: `venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.TemporalInterceptionAcceptance test_phase_c_attack_750.PhaseCAttackTeacherAcceptance`

Expected: new assertions fail because temporal/interception integration is absent.

- [ ] **Step 3: Implement `find_interception_action(...)`**

Evaluate safe first moves over four ticks and return an action only when it strictly reduces the opponent's reachable region or reaches a forecast choke point. Use stable action-order tie-breaking and never return `ACTION_BOMB`.

- [ ] **Step 4: Extend `get_attack_action` configuration**

Add `temporal_trap_horizon: int = 8`, `temporal_trap_max_bombs: int = 2`, and `interception_horizon: int = 4`. Preserve existing options and fallback behavior when the temporal planner returns `None`.

- [ ] **Step 5: Integrate trainer priority without changing rewards**

Keep urgent bomb/flame escape first, ring safety second, temporal trap third, interception fourth, then existing attack/navigation guidance. Classify temporal trap labels as combat and interception labels as navigation.

- [ ] **Step 6: Run integration tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.TemporalInterceptionAcceptance test_phase_c_attack_750.py test_phase_c_sd_target_750.py`

Expected: new and preserved teacher tests pass.

- [ ] **Step 7: Record task verification without committing**

Record commands and results in the execution ledger; leave changes uncommitted.

### Task 3: Planner Diagnostics and Training Configuration

**Files:**
- Modify: `train_actor_critic.py:416`
- Modify: `test_temporal_trap_planner.py`

**Interfaces:**
- Consumes: temporal plan and interception decisions from Task 2.
- Produces: trainer/CLI options and per-episode diagnostic metrics.

- [ ] **Step 1: Add failing trainer configuration and metric tests**

Require CLI and `train(...)` options:

- `temporal_trap_horizon=8`
- `temporal_trap_max_bombs=2`
- `interception_horizon=4`

Require episode metrics:

- `temporal_trap_labels`
- `temporal_trap_bombs`
- `temporal_trap_rejected_unsafe`
- `temporal_trap_rejected_duplicate`
- `interception_labels`
- `predicted_refuge_reduction`
- `predicted_trap_completion_ticks`

- [ ] **Step 2: Run trainer metric tests RED**

Run: `venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.TemporalTrainerAcceptance`

Expected: failures identify missing options and metrics.

- [ ] **Step 3: Thread planner configuration through API and CLI**

Validate positive horizons, `max_bombs == 2`, and pressure ranges. Build `TemporalTrapConfig` once per training call and pass it to the teacher.

- [ ] **Step 4: Accumulate diagnostics without adding reward terms**

Expose counts and means through `on_episode` and `metrics.jsonl`. Empty metrics use numeric zero and must remain JSON serializable.

- [ ] **Step 5: Run trainer metric tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.TemporalTrainerAcceptance`

Expected: all trainer diagnostic tests pass.

- [ ] **Step 6: Record task verification without committing**

Record command and result in the execution ledger; leave changes uncommitted.

### Task 4: Fine-Tune Launcher and Recovery Contract

**Files:**
- Create: `run_train_phaseC_sd_temporal_250.ps1`
- Create: `test_phase_c_sd_temporal_250.py`
- Modify: `README.md:81`

**Interfaces:**
- Consumes: temporal trainer options from Task 3 and resumable evaluator CLI from Plan 1.
- Produces: one-command bounded fine-tune and `-EvaluateOnly` recovery.

- [ ] **Step 1: Add staged-launcher acceptance tests**

Require exact training contract:

- warm start `runs\phaseC-sd-target-750\best_phaseC.pt`;
- output `runs\phaseC-sd-temporal-250`;
- 250 episodes, LR `0.00001`, entropy `0.005`;
- temporal horizon 8, two bombs, interception horizon 4;
- sequence pressure threshold `0.35`;
- training seed `2026101504`, validation seed `32000`, final evaluation seed `33000`;
- validation every 125 episodes, 200 games;
- Schedule E and probability 0.75;
- unchanged target thresholds and suicide gate.

Require final evaluation: selected best checkpoint, 200 CPU games, fresh seed, resumable partial every 10 games, validated final JSON, and all five gate lines.

- [ ] **Step 2: Run launcher tests RED**

Run: `venv\Scripts\python.exe -m unittest -v test_phase_c_sd_temporal_250.py`

Expected: failures identify the missing launcher only; fake programs prevent real training/evaluation.

- [ ] **Step 3: Implement normal-mode preflight and training invocation**

Require Python and warm-start checkpoint, reject an existing output directory, and pass each approved option exactly once.

- [ ] **Step 4: Implement evaluation stage and `-EvaluateOnly`**

Normal mode evaluates after successful training. Evaluation-only mode requires the existing best checkpoint and resumes the compatible partial without invoking training.

- [ ] **Step 5: Validate final JSON and report gates**

Require exact evaluation configuration and 200 games before reporting WR, ActiveWR, Draw, SD death, and Suicide. Gate failure exits zero; operational failure exits nonzero.

- [ ] **Step 6: Document normal and recovery commands**

Document artifact paths, progress cadence, partial resume, no-overwrite behavior, and that training begins only on explicit user request.

- [ ] **Step 7: Run launcher tests GREEN**

Run: `venv\Scripts\python.exe -m unittest -v test_phase_c_sd_temporal_250.py`

Expected: all launcher tests pass.

- [ ] **Step 8: Record task verification without committing**

Record command and result in the execution ledger; leave changes uncommitted.

### Task 5: Whole-Repository Verification and Handoff

**Files:**
- Verify all files from Tasks 1-4.

**Interfaces:**
- Consumes: completed Plan 1 and Tasks 1-4.
- Produces: verified uncommitted implementation ready for an explicitly authorized training run.

- [ ] **Step 1: Run targeted suites**

Run:

```powershell
venv\Scripts\python.exe -m unittest -v test_resumable_evaluation.py
venv\Scripts\python.exe -m unittest -v test_temporal_trap_planner.py
venv\Scripts\python.exe -m unittest -v test_phase_c_sd_temporal_250.py
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run preserved regressions and full suite**

Run:

```powershell
venv\Scripts\python.exe -m unittest -v test_combat.py test_phase_c_attack_750.py test_phase_c_sd_target_750.py test_sudden_death.py
venv\Scripts\python.exe -m unittest -v
```

Expected: all tests pass; report any unrelated baseline failure without changing unrelated code.

- [ ] **Step 3: Run static checks**

Run:

```powershell
venv\Scripts\python.exe -m py_compile temporal_trap_planner.py train_actor_critic.py evaluate.py
git diff --check
```

Expected: compilation succeeds and diff check reports no errors.

- [ ] **Step 4: Verify training preflight without launching it**

Confirm the warm-start checkpoint exists, `runs\phaseC-sd-temporal-250` does not exist, and the launcher help or staged fake execution succeeds. Do not invoke real training.

- [ ] **Step 5: Request one whole-change review**

Review planner safety, non-mutation, evaluation resume identity, launcher recovery, regression risk, and spec coverage. Fix Critical/Important findings through RED→GREEN, then rerun the full suite.

- [ ] **Step 6: Record final handoff without committing**

Report changed files, RED baseline, final test counts, preflight state, and the exact future training command. Leave changes uncommitted and unpushed.
