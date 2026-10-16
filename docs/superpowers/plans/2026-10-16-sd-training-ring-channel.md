# SD3 — Train B dưới sudden death + kênh vòng: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train hai nhánh từ checkpoint B dưới luật SD2 (một nhánh 12 kênh, một nhánh thêm kênh quan sát vòng) và so với B trên seed test mới 40000.

**Architecture:** Kênh vòng chỉ có trong `SuddenDeathEnv`. Trainer/evaluator production không đổi logic; `sudden_death_runtime` patch env/mask như `evaluate_case` đang làm. Runner mới `run_sd_training_pilot.py` tái dùng `train_arm` và các helper protocol hiện có; đánh giá bằng `evaluate_case` có tham số checkpoint.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.5 (`D:\anaconda\envs\torch\python.exe`), `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-16-sd-training-ring-channel-design.md`

## Global Constraints

- Python: `D:\anaconda\envs\torch\python.exe`; test bằng `-m unittest`, không pytest.
- `bomberman_env.py`, `rule_based_bot.py`, `OBS_CHANNELS = 12`, checkpoint B `runs/pressure-C-next/final_phaseC.pt` (SHA256 `93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60`) không đổi.
- Smoke dùng `SMOKE_SD` (start8/interval4/warning6/max_ring2) để vòng thực sự đóng trong 20 bước; full dùng SD2.
- SD2 timings: `{'sudden_death_start': 400, 'sudden_death_interval': 40, 'sudden_death_warning_steps': 40, 'sudden_death_max_ring': 2}`.
- Train: Phase C, 9×9, 2 agent, 200 episode/nhánh, LR 1e-4, max 500 bước, teacher local, MC, train seed `2026101601`, validation seed 21000 / 100 trận ở 0/100/200.
- Test: seed 40000–40199, 400 trận/model, CPU 1 thread, SD2 both_aware, 3 model (B, best sd_12ch, best sd_13ch). Smoke seed 39840.
- Gate: ActiveWR ≥ +5pp so với B, CI95 lower > 0, tự nổ không cao hơn B.
- Không commit trừ khi người dùng yêu cầu. Mỗi task xong cập nhật `RESULTS.md` và `IMPROVEMENTS.md` theo `CLAUDE.md` (gom ở Task 6).

## Review Focus

1. B (12 kênh) chạy trong env 13 kênh phải cho hành động giống hệt trong env 12 kênh cùng seed (Task 3 test parity actions).
2. `ring_channel=False` phải bit-identical với `SuddenDeathEnv` hiện tại, kể cả RNG (Task 1 test).
3. Ô của vòng vượt quá `max_ring` không bao giờ được đánh dấu trên kênh vòng (Task 1 test).
4. Thoát khỏi `sudden_death_runtime` (kể cả khi exception) phải khôi phục mọi hàm/lớp đã patch (Task 4 test).
5. Resume runner không được chấm lại model đã chấm hoặc mở seed test khác (Task 5 test reuse evaluation/seed guard).

---

### Task 1: Kênh quan sát vòng trong `SuddenDeathEnv`

**Files:**
- Modify: `sudden_death_experiment.py` (`__init__`, thêm `_get_obs`)
- Test: `test_sudden_death_experiment.py`

**Interfaces:**
- Produces: `SuddenDeathEnv(..., ring_channel=False)`; thuộc tính `env.obs_channels` (`OBS_CHANNELS` hoặc `OBS_CHANNELS + 1`); `env._get_obs(aid)` trả shape `(obs_channels, s, s)`, kênh cuối = `1/delay` trên ô vòng chưa đóng `<= sudden_death_max_ring`.

- [ ] **Step 1: Write failing tests** (thêm vào class `SuddenDeathTests`)

```python
    def test_ring_channel_marks_pending_rings_with_inverse_delay(self):
        env = arena(sudden_death=True, sudden_death_start=10, sudden_death_interval=5,
                    sudden_death_max_ring=2, ring_channel=True)
        self.assertEqual(env.obs_channels, 13)
        obs = env._get_obs(0)
        self.assertEqual(obs.shape, (13, 9, 9))
        ring1, ring2, ring3 = (env._ring_cells(r) for r in (1, 2, 3))
        for r, c in ring1:
            self.assertAlmostEqual(obs[12, r, c], 1 / 10)
        for r, c in ring2:
            self.assertAlmostEqual(obs[12, r, c], 1 / 15)
        for r, c in ring3 | {(4, 4)}:
            self.assertEqual(obs[12, r, c], 0)
        for _ in range(10):
            obs = env.step([0, 0])[0][0]
        self.assertEqual(env._sudden_death_ring, 1)
        for r, c in ring1:
            self.assertEqual(obs[12, r, c], 0)
            self.assertEqual(obs[0, r, c], 1)
        for r, c in ring2:
            self.assertAlmostEqual(obs[12, r, c], 1 / 5)

    def test_ring_channel_off_or_sd_off_keeps_first_twelve_channels(self):
        plain = SuddenDeathEnv(board_size=9, num_agents=2, seed=18, sudden_death=True)
        ringed = SuddenDeathEnv(board_size=9, num_agents=2, seed=18, sudden_death=True, ring_channel=True)
        probe = SuddenDeathEnv(board_size=9, num_agents=2, seed=18, ring_channel=True)
        self.assertEqual(plain.obs_channels, 12)
        for a, b, c in zip(plain.reset(), ringed.reset(), probe.reset()):
            np.testing.assert_array_equal(a, b[:12])
            np.testing.assert_array_equal(a, c[:12])
            self.assertFalse(np.any(c[12]))
        rng = np.random.default_rng(3)
        for _ in range(60):
            move = rng.integers(0, 6, 2).tolist()
            a, b = plain.step(move), ringed.step(move)
            self.assertEqual(a[1:], b[1:])
            for x, y in zip(a[0], b[0]):
                np.testing.assert_array_equal(x, y[:12])
            self.assertEqual(plain.rng.bit_generator.state, ringed.rng.bit_generator.state)
            if a[3]['episode_done']:
                break
```

- [ ] **Step 2: Run, expect FAIL** (`TypeError: unexpected keyword 'ring_channel'`)

Run: `D:\anaconda\envs\torch\python.exe -m unittest test_sudden_death_experiment -q`

- [ ] **Step 3: Implement**

In `sudden_death_experiment.py`: import `OBS_CHANNELS` from `bomberman_env`; add `ring_channel=False` to `__init__` signature; set before `super().__init__` (reset builds obs):

```python
        self.ring_channel = bool(ring_channel)
        self.obs_channels = OBS_CHANNELS + int(self.ring_channel)
```

Because `super().__init__` may call `reset()`/`_get_obs` before `sudden_death_max_ring` is set, set a provisional `self.sudden_death_max_ring = None` before `super().__init__` and make `_closure_ticks` return `{}` while it is `None`. Add:

```python
    def _get_obs(self, agent_id, danger=None):
        obs = super()._get_obs(agent_id, danger)
        if not self.ring_channel:
            return obs
        plane = np.zeros((1, self.board_size, self.board_size), dtype=np.float32)
        for ring, delay in self._closure_ticks().items():
            for r, c in self._ring_cells(ring):
                plane[0, r, c] = 1.0 / delay
        return np.concatenate([obs, plane])
```

- [ ] **Step 4: Run, expect PASS** (all 17 SD env tests).

---

### Task 2: Trainer tạo mạng theo số kênh của env

**Files:**
- Modify: `train_actor_critic.py:586` (`net = ActorCriticNet(...)` trong `train()`)
- Test: `test_sd_training_pilot.py` (create)

**Interfaces:**
- Consumes: `env.obs_channels` (Task 1), fallback `OBS_CHANNELS`.
- Produces: `train()` builds `ActorCriticNet(in_channels=<env channels>, board_size=..., num_actions=...)`.

- [ ] **Step 1: Write failing test**

```python
import unittest
from unittest.mock import patch

import torch

import train_actor_critic as trainer
from sudden_death_experiment import SuddenDeathEnv


class TrainerChannelTests(unittest.TestCase):
    def test_train_builds_net_with_env_channels(self):
        built = []
        base = trainer.ActorCriticNet
        def net(*args, **kwargs):
            built.append(base(*args, **kwargs))
            return built[-1]
        class RingEnv(SuddenDeathEnv):
            def __init__(self, **kwargs):
                super().__init__(**kwargs, sudden_death=True, ring_channel=True)
        with patch.object(trainer, 'BombermanFFAEnv', RingEnv), \
             patch.object(trainer, 'ActorCriticNet', side_effect=net):
            trainer.train(phase='C', episodes=1, board_size=9, num_agents=2, max_steps=5,
                          device='cpu', run_report=False, validation_every=0, seed=1,
                          checkpoint_dir=self.tmp, log_every=1)
        self.assertEqual(built[0].conv[0].in_channels, 13)

    def setUp(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = self._dir.name

    def tearDown(self):
        self._dir.cleanup()
```

- [ ] **Step 2: Run, expect FAIL** (in_channels 12, or shape error in conv forward).

Run: `D:\anaconda\envs\torch\python.exe -m unittest test_sd_training_pilot -q`

- [ ] **Step 3: Implement** — replace in `train()`:

```python
    net = ActorCriticNet(in_channels=getattr(env, 'obs_channels', OBS_CHANNELS),
                         board_size=board_size, num_actions=NUM_ACTIONS).to(device)
```

- [ ] **Step 4: Run, expect PASS.** Also run `D:\anaconda\envs\torch\python.exe -m unittest test_combat test_training_state test_reward_pilot -q` (default 12-channel path unchanged).

---

### Task 3: `make_ac_fn` nạp checkpoint 12/13 kênh và cắt observation

**Files:**
- Modify: `evaluate.py:174-189` (`make_ac_fn`)
- Test: `test_sd_training_pilot.py`

**Interfaces:**
- Produces: `make_ac_fn(path, board_size, device)` builds net with `in_channels = max(conv.0.weight.shape[1], OBS_CHANNELS)` (fallback `OBS_CHANNELS` if key absent) and feeds `obs[:in_channels]`.

- [ ] **Step 1: Write failing tests**

```python
import tempfile
from pathlib import Path
import numpy as np
import evaluate
from run_sudden_death_experiment import CHECKPOINT

class LoaderTests(unittest.TestCase):
    def test_b_in_ring_env_picks_same_actions_as_without_channel(self):
        policy = evaluate.make_ac_fn(str(CHECKPOINT), 9, 'cpu')
        def play(ring):
            env = SuddenDeathEnv(board_size=9, num_agents=2, seed=40, sudden_death=True,
                                 sudden_death_start=20, sudden_death_interval=5, ring_channel=ring)
            obs, actions = env.reset(), []
            torch.manual_seed(7)
            for _ in range(40):
                a = policy(env, obs, 0)
                actions.append(a)
                obs, _, _, info = env.step([a, 0])
                if info['episode_done']:
                    break
            return actions
        self.assertEqual(play(False), play(True))

    def test_thirteen_channel_checkpoint_round_trip(self):
        net = trainer.ActorCriticNet(in_channels=13, board_size=9)
        trainer.smart_load_weights(net, str(CHECKPOINT), 'cpu')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'ring.pt'
            torch.save(net.state_dict(), path)
            policy = evaluate.make_ac_fn(str(path), 9, 'cpu')
            env = SuddenDeathEnv(board_size=9, num_agents=2, seed=40, sudden_death=True, ring_channel=True)
            obs = env.reset()
            self.assertIn(policy(env, obs, 0), range(6))
            loaded = torch.load(path, weights_only=True)['conv.0.weight']
            self.assertEqual(loaded.shape[1], 13)
            self.assertFalse(torch.any(loaded[:, 12]))

    def test_b_logits_unchanged_in_thirteen_channel_net(self):
        net12, net13 = trainer.ActorCriticNet(board_size=9), trainer.ActorCriticNet(in_channels=13, board_size=9)
        trainer.smart_load_weights(net12, str(CHECKPOINT), 'cpu')
        trainer.smart_load_weights(net13, str(CHECKPOINT), 'cpu')
        env = SuddenDeathEnv(board_size=9, num_agents=2, seed=5, sudden_death=True,
                             sudden_death_start=3, ring_channel=True)
        obs = env.reset()[0]
        with torch.no_grad():
            a = net12(torch.as_tensor(obs[:12]).unsqueeze(0))
            b = net13(torch.as_tensor(obs).unsqueeze(0))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
```

- [ ] **Step 2: Run, expect FAIL** (13-channel obs into 12-channel conv: RuntimeError).

- [ ] **Step 3: Implement** in `make_ac_fn`:

```python
    from train_actor_critic import ActorCriticNet, SnapshotOpponent, smart_load_weights
    from bomberman_env import OBS_CHANNELS

    weights = torch.load(model_path, map_location=device, weights_only=True)
    stored = weights['conv.0.weight'].shape[1] if 'conv.0.weight' in weights else OBS_CHANNELS
    channels = max(stored, OBS_CHANNELS)
    net = ActorCriticNet(in_channels=channels, board_size=board_size).to(device)
    smart_load_weights(net, model_path, device)
    policy = SnapshotOpponent(net, device)

    @torch.no_grad()
    def fn(env, obs_list, agent_id):
        # Extra trailing channels (e.g. the sudden-death ring) are invisible to narrower nets.
        return policy.select_action(obs_list[agent_id][:channels], env, agent_id)
    return fn
```

- [ ] **Step 4: Run, expect PASS.** Re-run `test_sudden_death_runner` (off parity with existing evaluator) and `test_visualize`.

---

### Task 4: `sudden_death_runtime` và `evaluate_case(checkpoint, ring_channel)`

**Files:**
- Modify: `run_sudden_death_experiment.py` (add context manager; `evaluate_case` params)
- Test: `test_sudden_death_runner.py`

**Interfaces:**
- Produces:
  - `sudden_death_runtime(timings: dict, ring_channel: bool = False)` — context manager; patches `train_actor_critic.BombermanFFAEnv`, `evaluate.BombermanFFAEnv` (SD on, timings, ring_channel), `benchmark_t3.BombermanFFAEnv` (SD off, ring_channel), `train_actor_critic.get_action_mask` and `rule_based_bot._safe_action_mask` (wrapped with `sudden_death_mask`; unchanged for envs without `_closure_ticks`).
  - `evaluate_case(case, *, games, seed, max_steps, timings, trace_path=None, progress=None, stop=None, checkpoint=CHECKPOINT, ring_channel=False)`.

- [ ] **Step 1: Write failing tests**

```python
    def test_runtime_patches_and_restores_even_on_error(self):
        import benchmark_t3, rule_based_bot
        import train_actor_critic as trainer
        before = (trainer.BombermanFFAEnv, evaluate.BombermanFFAEnv, benchmark_t3.BombermanFFAEnv,
                  trainer.get_action_mask, rule_based_bot._safe_action_mask)
        with self.assertRaises(RuntimeError):
            with runner.sudden_death_runtime(TIMINGS, ring_channel=True):
                env = trainer.BombermanFFAEnv(board_size=9, num_agents=2, max_steps=20, seed=3)
                self.assertTrue(env.sudden_death and env.obs_channels == 13)
                probe = benchmark_t3.BombermanFFAEnv(board_size=9, num_agents=2, seed=3)
                self.assertFalse(probe.sudden_death)
                self.assertEqual(probe.obs_channels, 13)
                env.reset()
                self.assertEqual(len(trainer.get_action_mask(env, 0)), 6)
                raise RuntimeError
        after = (trainer.BombermanFFAEnv, evaluate.BombermanFFAEnv, benchmark_t3.BombermanFFAEnv,
                 trainer.get_action_mask, rule_based_bot._safe_action_mask)
        self.assertEqual(before, after)

    def test_evaluate_case_accepts_checkpoint_and_ring_channel(self):
        plain = runner.evaluate_case('both_aware', games=2, seed=39888, max_steps=30, timings=TIMINGS)
        ringed = runner.evaluate_case('both_aware', games=2, seed=39888, max_steps=30, timings=TIMINGS,
                                      checkpoint=runner.CHECKPOINT, ring_channel=True)
        for a, b in zip(plain['games'], ringed['games']):
            self.assertEqual(a, b)
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: sudden_death_runtime`, unexpected kw `checkpoint`).

- [ ] **Step 3: Implement**

`evaluate_case`: add `checkpoint=CHECKPOINT, ring_channel=False`; `ObservedEnv.__init__` → `super().__init__(**kwargs, sudden_death=enabled, ring_channel=ring_channel, **timings)`; `make_ac_fn(str(checkpoint), 9, 'cpu')`.

Add after `evaluate_case`:

```python
@contextmanager
def sudden_death_runtime(timings, ring_channel=False):
    """Run trainer/evaluator/skill probes under SD rules with SD-aware learner and RuleBot."""
    import benchmark_t3
    base_mask, base_bot_mask = trainer.get_action_mask, bot_module._safe_action_mask

    class TrainEnv(SuddenDeathEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, sudden_death=True, ring_channel=ring_channel, **timings)

    class ProbeEnv(SuddenDeathEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, ring_channel=ring_channel)

    def aware(base):
        return lambda env, aid, *a, **k: (sudden_death_mask(env, aid, base(env, aid, *a, **k))
                                          if hasattr(env, '_closure_ticks') else base(env, aid, *a, **k))

    with patch.object(trainer, 'BombermanFFAEnv', TrainEnv), \
         patch.object(evaluation, 'BombermanFFAEnv', TrainEnv), \
         patch.object(benchmark_t3, 'BombermanFFAEnv', ProbeEnv), \
         patch.object(trainer, 'get_action_mask', aware(base_mask)), \
         patch.object(bot_module, '_safe_action_mask', aware(base_bot_mask)):
        yield
```

Add `from contextlib import contextmanager` at top; add `'run_sd_training_pilot.py', 'test_sd_training_pilot.py'` to nothing here (SOURCES of SD1 stay as registered).

- [ ] **Step 4: Run, expect PASS** (`test_sudden_death_runner`, `test_sudden_death_experiment`).

---

### Task 5: Runner `run_sd_training_pilot.py`

**Files:**
- Create: `run_sd_training_pilot.py`
- Test: `test_sd_training_pilot.py`

**Interfaces:**
- Consumes: `train_arm`, `pair_ready`, `probe_summary`, `fresh_test_check`, `WARM_START`, `B_SHA256`, `BaseNet`, `ROOT` (run_combat_move_pilot); `skill_probe` (run_t3_curriculum); `paired_active_bootstrap`, `stage_timer` (pilot_two_bomb); `digest`, `dump` (audit_t2); `evaluate_case`, `sudden_death_runtime` (Task 4).
- Produces: `run(output: Path, device='cpu', smoke=False, budget_minutes=150, resume=False) -> dict`; constants `SD2`, `ARMS = {'sd_12ch': False, 'sd_13ch': True}`, `TRAIN_SEED = 2026101601`, `TEST_SEED = 40000`, `SMOKE_SEED = 39840`; CLI `--device --smoke --resume --budget-minutes --output-dir`.

- [ ] **Step 1: Write failing tests**

```python
import run_sd_training_pilot as pilot

class RunnerGuardTests(unittest.TestCase):
    def test_refuses_existing_output_and_bad_budget(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                pilot.run(Path(d), smoke=True)
        with self.assertRaises(ValueError):
            pilot.run(Path('runs/never'), budget_minutes=0)

    def test_gate_requires_gain_ci_and_suicide(self):
        def result(active, suicide=0):
            return {'suicide_rate': suicide, 'games': [
                {'seed': s, 'aid': a, 'active_win': int(s < active), 'win': 0, 'draw': 0, 'suicide': 0}
                for s in range(100) for a in (0, 1)]}
        base = result(30)
        self.assertTrue(pilot.gate({'original': base, 'sd_13ch': result(50)}, 'sd_13ch')[1])
        self.assertFalse(pilot.gate({'original': base, 'sd_13ch': result(32)}, 'sd_13ch')[1])
        self.assertFalse(pilot.gate({'original': base, 'sd_13ch': result(50, 1)}, 'sd_13ch')[1])

    def test_test_seed_is_fresh_and_distinct_from_smoke(self):
        from run_combat_move_pilot import fresh_test_check
        fresh_test_check(pilot.TEST_SEED)
        self.assertNotEqual(pilot.SMOKE_SEED, pilot.TEST_SEED)
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: run_sd_training_pilot`).

- [ ] **Step 3: Implement** `run_sd_training_pilot.py`:

```python
"""SD3: B trained under SD2 rules, without vs with a ring observation channel."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from audit_t2 import digest, dump
from pilot_two_bomb import paired_active_bootstrap, stage_timer
from run_combat_move_pilot import (ROOT, WARM_START, B_SHA256, BaseNet, SOURCE_FILES as SHARED_FILES,
                                   fresh_test_check, train_arm, pair_ready, probe_summary)
from run_t3_curriculum import skill_probe
from run_sudden_death_experiment import evaluate_case, sudden_death_runtime
import train_actor_critic as trainer
from training_state import runtime_state, run_lock

SD2 = {'sudden_death_start': 400, 'sudden_death_interval': 40,
       'sudden_death_warning_steps': 40, 'sudden_death_max_ring': 2}
# Smoke runs 20 steps, so it uses short timings to exercise closures in training and evaluation.
SMOKE_SD = {'sudden_death_start': 8, 'sudden_death_interval': 4,
            'sudden_death_warning_steps': 6, 'sudden_death_max_ring': 2}
ARMS = {'sd_12ch': False, 'sd_13ch': True}
TRAIN_SEED = 2026101601
TEST_SEED = 40000
SMOKE_SEED = 39840
SOURCE_FILES = (*SHARED_FILES, 'sudden_death_experiment.py', 'run_sudden_death_experiment.py',
                'run_sd_training_pilot.py', 'test_sd_training_pilot.py', 'training_state.py')
METRICS = ('win', 'active_win', 'draw', 'suicide')


def gate(results, candidate):
    comparison = paired_active_bootstrap(results[candidate], results['original'])
    eligible = (comparison['difference_pp'] >= 5 and comparison['ci95_pp'][0] > 0
                and results[candidate]['suicide_rate'] <= results['original']['suicide_rate'])
    return comparison, bool(eligible)


def paired(results, left, right):
    out = {}
    for metric in METRICS:
        wrapped = [{**results[name], 'games': [{**g, 'active_win': g[metric]} for g in results[name]['games']]}
                   for name in (left, right)]
        out[metric] = paired_active_bootstrap(*wrapped)
    return out


def evaluate(output, models, stop, smoke, timings):
    seed, games, steps = (SMOKE_SEED, 2, 20) if smoke else (TEST_SEED, 400, 500)
    results = {}
    for name, path in models.items():
        stored = output / f'evaluation_{name}_{seed}.json'
        if stored.exists():
            cached = json.loads(stored.read_text(encoding='utf-8'))
            if (cached['seed'], cached['num_episodes'], cached['checkpoint_sha256']) != (seed, games, digest(path)):
                raise ValueError('Stored evaluation differs from the locked model/protocol')
            results[name] = cached
            continue
        def progress(done, total, name=name):
            if done % 25 == 0:
                print(f'[Evaluation] {name}: {done}/{total}', flush=True)
        result = evaluate_case('both_aware', games=games, seed=seed, max_steps=steps, timings=timings,
                               progress=progress, stop=stop, checkpoint=path, ring_channel=True)
        if result is None:
            break
        result.update(checkpoint=str(path), checkpoint_sha256=digest(path))
        dump(stored, result)
        results[name] = result
    return results


def run(output, device='cpu', smoke=False, budget_minutes=150, resume=False):
    if not 0 < budget_minutes <= 150:
        raise ValueError('Budget must be in (0, 150] minutes')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested training device is unavailable')
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f'Refusing to overwrite {output}')
    if resume and not (output / 'config.json').exists():
        raise ValueError('No registered run at the requested resume directory')
    with run_lock(output):
        return _run(output, device, smoke, budget_minutes, resume)


def _run(output, device, smoke, budget_minutes, resume):
    if digest(WARM_START) != B_SHA256:
        raise ValueError('Warm-start B changed')
    if not smoke and not (output / 'test_opened.json').exists():
        fresh_test_check(TEST_SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sources = {name: digest(ROOT / name) for name in SOURCE_FILES}
    timings = SMOKE_SD if smoke else SD2
    config = {'experiment': 'SD3', 'full_protocol': not smoke, 'warm_start': str(WARM_START),
        'warm_start_sha256': B_SHA256, 'source_sha256': sources, 'arms': ARMS, 'sudden_death': timings,
        'train_seed': TRAIN_SEED, 'episodes_per_arm': 20 if smoke else 200, 'phase': 'C',
        'board_size': 9, 'num_agents': 2, 'max_steps': 20 if smoke else 500, 'lr': 1e-4,
        'train_device': device, 'attack_teacher': 'local', 'advantage_estimator': 'mc',
        'opponent': 'RuleBot, sudden-death aware', 'validation_seed': 21000,
        'validation_episodes': 2 if smoke else 100, 'validation_every': 10 if smoke else 100,
        'evaluation': {'seed': SMOKE_SEED if smoke else TEST_SEED, 'games_per_model': 2 if smoke else 400,
                       'case': 'both_aware', 'ring_channel_env': True, 'device': 'cpu', 'threads': 1},
        'budget_minutes': {'overall': budget_minutes, 'training': 100, 'evaluation': 45},
        'gate': {'gain_vs_B_pp': 5, 'ci95_lower_above': 0, 'suicide_not_above_B': True},
        'runtime': runtime_state(), 'torch': torch.__version__,
        'hardware': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}
    if resume:
        stored = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        if any(stored[k] != config[k] for k in config if k != 'runtime') or stored['runtime'] != config['runtime']:
            raise ValueError('Resume config/source/runtime differs from the registered run')
        config = stored
    else:
        output.mkdir(parents=True)
        (output / 'source').mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(ROOT / name, output / 'source' / name)
        config['registered_utc'] = datetime.now(timezone.utc).isoformat()
        dump(output / 'config.json', config)
    deadline = datetime.fromisoformat(config['registered_utc']) + timedelta(minutes=budget_minutes)
    if resume and deadline <= datetime.now(timezone.utc):
        raise ValueError('Registered deadline expired; register a separately budgeted continuation')
    started = perf_counter()
    arms, results, comparisons, reason, eligible, timer = {}, {}, {}, None, False, None

    def phase(minutes):
        left = (deadline - datetime.now(timezone.utc)).total_seconds()
        return stage_timer(max(0, min(minutes * 60, left)))

    try:
        stop, timer = phase(100)
        for arm, ring in ARMS.items():
            if stop.is_set():
                break
            saved = output / arm / 'summary.json'
            if resume and saved.exists() and pair_ready({arm: json.loads(saved.read_text())}, (arm,)):
                arms[arm] = json.loads(saved.read_text())
                continue
            with sudden_death_runtime(timings, ring_channel=ring):
                net = BaseNet(in_channels=12 + int(ring), board_size=9)
                trainer.smart_load_weights(net, str(WARM_START), 'cpu')
                baseline = skill_probe(net, 1, stop, smoke=False)
                if baseline is None:
                    break
                dump(output / f'baseline_skill_{arm}.json', probe_summary(baseline))
                arms[arm] = train_arm(output, arm, device, smoke, stop, baseline,
                    attack_teacher='local', normalize_advantage=False, advantage_estimator='mc',
                    contact_diagnostics=True, train_seed=TRAIN_SEED, save_training_state=True,
                    resume=resume and (output / arm).exists(), smoke_episodes=20, smoke_steps=20)
            if not (arms[arm]['training_complete'] and arms[arm]['validation_complete']
                    and arms[arm]['skill_probes_complete']):
                break
        if not pair_ready(arms, ARMS):
            reason = 'training_validation_or_skill_incomplete'
        else:
            models = {'original': WARM_START, **{arm: Path(arms[arm]['checkpoints']['best']['path']) for arm in ARMS}}
            selected = {name: {'path': str(p), 'sha256': digest(p)} for name, p in models.items()}
            if (output / 'selected_models.json').exists():
                if json.loads((output / 'selected_models.json').read_text()) != selected:
                    raise ValueError('Test models are locked; selecting another checkpoint is forbidden')
            else:
                dump(output / 'selected_models.json', selected)
                dump(output / 'test_opened.json', {'seed': SMOKE_SEED if smoke else TEST_SEED,
                     'num_episodes': 2 if smoke else 400, 'num_agents': 2})
            timer.cancel()
            stop, timer = phase(45)
            results = evaluate(output, models, stop, smoke, timings)
            if len(results) != 3:
                reason = 'evaluation_incomplete'
            elif not smoke:
                for arm in ARMS:
                    comparisons[arm] = {'gate': gate(results, arm)[0], 'eligible': gate(results, arm)[1],
                                        'vs_original': paired(results, arm, 'original')}
                comparisons['sd_13ch_vs_sd_12ch'] = paired(results, 'sd_13ch', 'sd_12ch')
                eligible = any(comparisons[arm]['eligible'] for arm in ARMS)
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        if timer is not None:
            timer.cancel()
        unchanged = digest(WARM_START) == B_SHA256 and sources == {
            name: digest(ROOT / name) if (ROOT / name).exists() else None for name in SOURCE_FILES}
        if not unchanged:
            reason, eligible = 'source_or_warm_start_changed', False
        complete = reason is None and pair_ready(arms, ARMS) and len(results) == 3
        summary = {'status': 'complete' if complete else 'incomplete', 'full_protocol': not smoke,
            'reason': reason, 'arms': arms, 'hashes_unchanged': unchanged, 'comparisons': comparisons,
            'eligible_for_replication': complete and not smoke and eligible,
            'decision': 'replicate_before_replacing' if complete and not smoke and eligible else 'retain_original',
            'elapsed_seconds_this_process': perf_counter() - started, 'resumed': resume,
            'models': {n: {k: v for k, v in r.items() if k != 'games'} for n, r in results.items()}}
        dump(output / 'summary.json', summary)
        shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(json.dumps({k: v for k, v in summary.items() if k not in ('arms', 'models')}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--budget-minutes', type=float, default=150)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'runs' / ('sd3-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.device, args.smoke, args.budget_minutes, args.resume)
```

- [ ] **Step 4: Run tests, expect PASS.** `D:\anaconda\envs\torch\python.exe -m unittest test_sd_training_pilot -q`

- [ ] **Step 5: Smoke CPU and CUDA**

```
D:\anaconda\envs\torch\python.exe -u run_sd_training_pilot.py --smoke --device cpu --output-dir runs\sd3-smoke-cpu-<time>
D:\anaconda\envs\torch\python.exe -u run_sd_training_pilot.py --smoke --device cuda --output-dir runs\sd3-smoke-cuda-<time>
```
Expected: `status: complete`, `full_protocol: false`, `hashes_unchanged: true`, 20 episodes/arm, both arms' `conv.0.weight` in_channels 12 / 13, 3 evaluation files. Inspect `sd_13ch/final_phaseC.pt` channel 12 weights non-zero after updates (learned).

---

### Task 6: Regression, docs, handoff

**Files:**
- Modify: `RESULTS.md` (SD3 section status), `IMPROVEMENTS.md` (new section 24)

- [ ] **Step 1:** Run regression: `D:\anaconda\envs\torch\python.exe -m unittest test_sudden_death_experiment test_sudden_death_runner test_sd_training_pilot test_combat test_training_state test_reward_pilot test_visualize -q`; record counts.
- [ ] **Step 2:** Update SD3 section in `RESULTS.md` (implementation/smoke DONE, full WAIT RUN, smoke dirs), add `IMPROVEMENTS.md` §24 (kênh vòng, `obs_channels`, `make_ac_fn` slicing, runtime, runner, tests).
- [ ] **Step 3:** Give the user the cmd command for the full run:

```
D:\anaconda\envs\torch\python.exe -u run_sd_training_pilot.py --device cuda --output-dir runs\sd3-full > runs\sd3-full-console.log 2>&1
```
and the resume command (same with `--resume`).
