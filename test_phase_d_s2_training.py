"""Requirements-derived schedules and actual bounded D-S2 state restoration."""
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import random
import subprocess
import tempfile
import threading
import unittest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from test_phase_d_s2_protocol import (ROOT, RUNNER, PYTHON, CHECKPOINTS, copy_inputs,
                                     descriptor, function, read_json)


def register_case(test, base, device='cpu'):
    function(test, 'train_arm')
    input_root = copy_inputs(Path(base) / 'inputs')
    output = Path(base) / 'output'
    completed = subprocess.run([str(PYTHON), str(RUNNER), 'register', '--output', str(output),
                                '--input-root', str(input_root), '--smoke', '--device', device,
                                '--workers', '1'], cwd=ROOT, capture_output=True, text=True, check=False)
    test.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
    return input_root, output


class ScheduleContractTests(unittest.TestCase):
    def schedule(self, arm, **kwargs):
        return function(self, 'build_train_schedule')(arm, **kwargs)

    def test_full_pool_ratios_seeds_corners_and_fallback_are_exact(self):
        result = self.schedule('pool')
        rows = result['episodes']
        self.assertEqual(len(rows), 2000)
        self.assertEqual([r['episode'] for r in rows], list(range(1, 2001)))
        self.assertEqual(Counter(r['opponent_type'] for r in rows), {'RB': 400, 'fixed': 800, 'recent': 800})
        self.assertEqual(Counter(r['opponent_id'] for r in rows if r['opponent_type'] == 'fixed'),
                         {'G': 200, 'F': 200, 'S': 200, 'S2': 200})
        for pair, (first, second) in enumerate(zip(rows[::2], rows[1::2])):
            with self.subTest(pair=pair):
                self.assertEqual([first['learner_corner'], second['learner_corner']], [0, 1])
                self.assertEqual(first['board_seed'], 610000 + pair)
                for key in ('board_seed', 'opponent_id', 'opponent_sha256', 'block', 'slot', 'opponent_type'):
                    self.assertEqual(first[key], second[key])
                self.assertEqual(first['block'], pair // 10)
                self.assertEqual(first['slot'], pair % 10)
        recent = [r for r in rows if r['opponent_type'] == 'recent']
        self.assertTrue(all(r['recent_fallback'] and r['opponent_id'] == 'S2' for r in recent))
        self.assertTrue(all(r['opponent_sha256'] == CHECKPOINTS['S2'][1] for r in recent))

    def test_control_uses_same_board_and_corner_schedule_but_only_rb(self):
        control = self.schedule('control')['episodes']
        pool = self.schedule('pool')['episodes']
        self.assertEqual(len(control), 2000)
        self.assertTrue(all(r['opponent_id'] == 'RB' and r['opponent_type'] == 'RB' for r in control))
        self.assertEqual([(r['episode'], r['board_seed'], r['learner_corner']) for r in control],
                         [(r['episode'], r['board_seed'], r['learner_corner']) for r in pool])

    def test_recent_uses_only_two_latest_pool_snapshots_available_before_block(self):
        snapshots = [descriptor(f'pool-{ep}', ep, 'pool') for ep in (200, 400, 600)]
        snapshots.append(descriptor('control-600', 600, 'control'))
        rows = self.schedule('pool', snapshots=snapshots, start_episode=401, episodes=20)['episodes']
        recent = [r for r in rows[::2] if r['opponent_type'] == 'recent']
        self.assertEqual([r['opponent_id'] for r in recent], ['pool-200', 'pool-400', 'pool-200', 'pool-400'])
        self.assertTrue(all(not r['recent_fallback'] for r in recent))
        rows = self.schedule('pool', snapshots=snapshots, start_episode=601, episodes=20)['episodes']
        recent = [r['opponent_id'] for r in rows[::2] if r['opponent_type'] == 'recent']
        self.assertEqual(recent, ['pool-400', 'pool-600', 'pool-400', 'pool-600'])

    def test_one_snapshot_is_reused_and_mid_block_publication_waits_for_next_block(self):
        snapshots = [descriptor('early', 10, 'pool')]
        rows = self.schedule('pool', snapshots=snapshots, episodes=40)['episodes']
        self.assertTrue(all(r['recent_fallback'] for r in rows[:20] if r['opponent_type'] == 'recent'))
        self.assertEqual({r['opponent_id'] for r in rows[20:] if r['opponent_type'] == 'recent'}, {'early'})

    def test_recent_cursor_and_split_at_block_boundary_preserve_schedule(self):
        snapshots = [descriptor(f'pool-{ep}', ep, 'pool') for ep in (200, 400)]
        full = self.schedule('pool', snapshots=snapshots, start_episode=401, episodes=60, recent_cursor=1)
        first = self.schedule('pool', snapshots=snapshots, start_episode=401, episodes=20, recent_cursor=1)
        rest = self.schedule('pool', snapshots=snapshots, start_episode=421, episodes=40,
                             recent_cursor=first['recent_cursor'])
        self.assertEqual(full['episodes'], first['episodes'] + rest['episodes'])
        self.assertEqual(full['recent_cursor'], rest['recent_cursor'])

    def test_smoke_uses_only_two_paired_smoke_boards(self):
        rows = self.schedule('pool', smoke=True, episodes=4)['episodes']
        self.assertEqual([(r['board_seed'], r['learner_corner']) for r in rows],
                         [(690000, 0), (690000, 1), (690001, 0), (690001, 1)])


class ActualTrainingContractTests(unittest.TestCase):
    def assert_same(self, left, right, path='state'):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor, path)
            self.assertTrue(torch.equal(left.cpu(), right.cpu()), path)
        elif isinstance(left, np.ndarray):
            self.assertTrue(np.array_equal(left, right), path)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for key in left:
                self.assert_same(left[key], right[key], f'{path}.{key}')
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assert_same(a, b, f'{path}[{index}]')
        else:
            self.assertEqual(left, right, path)

    def read_state(self, output, arm):
        return function(self, 'read_training_state')(output, arm)

    def test_both_arms_start_from_s2_with_fresh_adam_and_cnn_opponents_stay_frozen(self):
        train = function(self, 'train_arm')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-acceptance-') as temporary:
            _, output = register_case(self, temporary)
            states = {}
            for arm in ('control', 'pool'):
                train(output, arm)
                state = self.read_state(output, arm)
                states[arm] = state
                self.assertEqual(state['initial_sha256'], CHECKPOINTS['S2'][1])
                self.assertEqual(state['initial_optimizer_steps'], 0)
                self.assertEqual(state['episode'], 4)
                self.assertEqual(len(state['metrics']), 4)
                self.assertEqual(state['scheduler']['T_max'], 4)
                self.assertAlmostEqual(state['scheduler']['eta_min'], 2.5e-5 * .05)
                self.assertTrue(state['optimizer']['state'], 'Real Adam update state must be persisted')
                self.assertTrue(all(int(v['step']) == 4 for v in state['optimizer']['state'].values()))
            audit = states['pool']['opponent_audit']
            self.assertTrue(any(row['id'] == 'G' for row in audit), 'Smoke must execute a frozen CNN opponent')
            for row in audit:
                self.assertEqual(row['before_sha256'], row['after_sha256'])
                self.assertTrue(row['eval_mode'])
                self.assertEqual(row['gradient_count'], 0)

    def continuation(self, *, interruption='episode', device='cpu'):
        train = function(self, 'train_arm')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-acceptance-') as temporary:
            base = Path(temporary)
            _, full_output = register_case(self, base / 'full', device)
            _, split_output = register_case(self, base / 'split', device)
            train(full_output, 'pool')
            full = self.read_state(full_output, 'pool')
            stop = threading.Event()
            if interruption == 'episode':
                def on_episode(row):
                    if row['episode'] == 2:
                        stop.set()
                train(split_output, 'pool', stop_event=stop, on_episode=on_episode)
                expected = 2
            elif interruption == 'mid_episode':
                def on_step(env, episode, aid, actions, opponents):
                    if episode == 2 and env.step_count >= 2:
                        stop.set()
                train(split_output, 'pool', stop_event=stop, on_step=on_step)
                expected = 1
            else:
                event = interruption
                def failure(event_name, episode):
                    if event_name == event and episode == 2:
                        raise RuntimeError('independent acceptance injected commit interruption')
                with self.assertRaisesRegex(RuntimeError, 'injected commit interruption'):
                    train(split_output, 'pool', commit_hook=failure)
                expected = 1 if event == 'before_state_commit' else 2
            partial = self.read_state(split_output, 'pool')
            self.assertEqual(partial['episode'], expected)
            self.assertEqual(len(partial['metrics']), expected)
            self.assertTrue(all(s['episode'] <= expected for s in partial['snapshots']))
            random.random(); np.random.random(); torch.rand(7)
            train(split_output, 'pool', resume=True)
            resumed = self.read_state(split_output, 'pool')
            for key in ('episode', 'model', 'optimizer', 'scheduler', 'rng', 'schedule_state', 'metrics'):
                self.assert_same(full[key], resumed[key], key)
            self.assertEqual([r['episode'] for r in resumed['metrics']], [1, 2, 3, 4])
            for first, second in zip(resumed['metrics'][::2], resumed['metrics'][1::2]):
                for key in ('board_seed', 'opponent_sha256'):
                    self.assertEqual(first[key], second[key])
                self.assertEqual([first['learner_corner'], second['learner_corner']], [0, 1])
            self.assertEqual(sorted(s['episode'] for s in resumed['snapshots']), [2, 4])

    def test_cpu_resume_preserves_real_model_adam_scheduler_rng_and_metrics(self):
        self.continuation()

    def test_partial_episode_is_discarded_and_replayed(self):
        self.continuation(interruption='mid_episode')

    def test_failure_before_atomic_state_commit_replays_episode_once(self):
        self.continuation(interruption='before_state_commit')

    def test_failure_after_atomic_state_commit_does_not_repeat_episode(self):
        self.continuation(interruption='after_state_commit')

    @unittest.skipUnless(os.environ.get('D_S2_ACCEPTANCE_CUDA') == '1' and torch.cuda.is_available(),
                         'CUDA acceptance requires explicit D_S2_ACCEPTANCE_CUDA=1 and available CUDA')
    def test_opt_in_cuda_resume_uses_actual_registered_gpu(self):
        self.continuation(device='cuda:0')

    def test_second_writer_cannot_start_or_advance_training(self):
        train = function(self, 'train_arm')
        module = __import__('run_phase_d_s2')
        from training_state import run_lock
        lock = getattr(module, 'writer_lock', run_lock)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-acceptance-') as temporary:
            _, output = register_case(self, temporary)
            with lock(output):
                with self.assertRaises((ValueError, RuntimeError)):
                    train(output, 'control')


if __name__ == '__main__':
    unittest.main()
