"""T4-B1 paired GAE protocol guards; numerical and boundary tests live in test_gae."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import torch
import torch.nn.functional as F

import run_gae_pilot as runner
import run_combat_move_pilot as shared
import train_actor_critic as trainer
from test_combat_move_pilot import ready_arm, result


class ProtocolChecks(unittest.TestCase):
    def test_preflight_checks_gae_and_honors_cancellation(self):
        stop = threading.Event()
        report = runner.preflight(stop)
        self.assertTrue(report['passed'])
        self.assertEqual(report['gae_lambda'], .95)
        self.assertFalse(report['normalization'])
        stop.set()
        self.assertIsNone(runner.preflight(stop))

    def run_mocked(self, broken=None, smoke=False, edit=False, phase_calls=None, clock=None):
        output = runner.ROOT / ('runs/uncreated-gae-' + uuid4().hex)
        arms = {a: ready_arm(output, a) for a in runner.ARMS}
        if broken:
            arms['gae'][broken] = False
        changed = False
        saved, order = {}, []
        def train(*args, **kwargs):
            nonlocal changed
            order.append(('train', args[1], kwargs))
            changed = edit
            return arms[args[1]]
        def save(path, data):
            saved[path.name] = data
            order.append(('save', path.name))
        def digest(path):
            if path == runner.WARM_START:
                return runner.B_SHA256
            return 'changed' if changed and path.name == 'train_actor_critic.py' else 'test-only-source'
        def evaluate(*args, **kwargs):
            self.assertIn('selected_models.json', saved)
            self.assertIn('test_opened.json', saved)
            self.assertEqual(kwargs['test_seed'], 35000)
            return {n: result(100 if n == 'gae' else 0, seed=35000)
                    for n in ('original', 'control', 'gae')}
        def timer(seconds):
            if phase_calls is not None:
                phase_calls.append(seconds)
            return threading.Event(), Mock()
        with patch.object(runner, 'digest', side_effect=digest), patch.object(runner, 'fresh_test_check') as fresh, \
             patch.object(runner, 'BaseNet'), patch.object(runner.trainer, 'smart_load_weights'), \
             patch.object(runner, 'skill_probe', return_value={'records': []}), \
             patch.object(runner, 'train_arm', side_effect=train), \
             patch.object(runner, 'evaluate_models', side_effect=evaluate) as evaluation, \
             patch.object(runner, 'stage_timer', side_effect=timer), \
             patch.object(runner, 'perf_counter', side_effect=clock or runner.perf_counter), \
             patch.object(Path, 'mkdir'), patch.object(Path, 'write_text'), \
             patch.object(runner.shutil, 'copy2'), patch.object(runner.shutil, 'make_archive'), \
             patch.object(runner, 'dump', side_effect=save), patch('builtins.print'):
            if edit:
                with self.assertRaises(AssertionError):
                    runner.run(output)
                summary = saved['summary.json']
            else:
                summary = runner.run(output, smoke=smoke)
        return summary, order, saved, fresh, evaluation

    def test_paired_config_only_estimator_differs_and_selection_precedes_test(self):
        summary, order, saved, fresh, _ = self.run_mocked()
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['decision'], 'replicate_before_replacing')
        fresh.assert_called_once_with(35000)
        self.assertEqual(order[0], ('save', 'config.json'))
        configs = [entry[2] for entry in order if entry[0] == 'train']
        self.assertEqual(configs[0]['normalize_advantage'], False)
        self.assertEqual(configs[1]['normalize_advantage'], False)
        self.assertEqual(configs[0]['advantage_estimator'], 'mc')
        self.assertEqual(configs[1]['advantage_estimator'], 'gae')
        self.assertEqual(configs[0]['gae_lambda'], .95)
        self.assertEqual(configs[1]['gae_lambda'], .95)
        self.assertEqual({k: v for k, v in configs[0].items() if k != 'advantage_estimator'},
                         {k: v for k, v in configs[1].items() if k != 'advantage_estimator'})
        self.assertEqual(configs[0]['attack_teacher'], 'local')
        self.assertEqual(saved['config.json']['critic_target'], 'MC returns')

    def test_incomplete_arms_and_changed_source_never_open_test(self):
        for broken in ('training_complete', 'validation_complete', 'skill_probes_complete'):
            summary, _, _, _, evaluation = self.run_mocked(broken=broken)
            self.assertEqual(summary['status'], 'incomplete')
            evaluation.assert_not_called()
        summary, _, _, _, evaluation = self.run_mocked(edit=True)
        self.assertFalse(summary['hashes_unchanged'])
        self.assertEqual(summary['decision'], 'retain_original')
        evaluation.assert_not_called()

    def test_smoke_never_eligible(self):
        summary, _, _, fresh, _ = self.run_mocked(smoke=True)
        self.assertFalse(summary['full_protocol'])
        self.assertFalse(summary['eligible_for_replication'])
        fresh.assert_not_called()

    def test_budget_clobber_and_gpu_validation(self):
        with self.assertRaises(ValueError):
            runner.run(runner.ROOT / 'unused', budget_minutes=151)
        with self.assertRaises(FileExistsError):
            runner.run(runner.ROOT)
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(ValueError):
                runner.run(runner.ROOT / 'unused', device='cuda')

    def test_training_cap_100_and_evaluation_clipped_to_overall_150(self):
        phases = []
        summary, _, saved, _, _ = self.run_mocked(phase_calls=phases,
            clock=[0, 0, 12 * 60, 112 * 60, 130 * 60])
        self.assertEqual(phases, [15 * 60, 100 * 60, 38 * 60])
        self.assertEqual(saved['config.json']['budget_minutes']['training'], 100)
        self.assertEqual(summary['elapsed_seconds'], 130 * 60)

    def test_new_seed_range_and_cpu_evaluation_are_wired_without_teacher(self):
        with patch.object(Path, 'rglob', return_value=[Path('evaluation_seen.json')]), \
             patch.object(Path, 'read_text', return_value=json.dumps(result(seed=35000))):
            with self.assertRaisesRegex(ValueError, 'Test35000 already opened'):
                shared.fresh_test_check(runner.TEST_SEED)
        with patch.object(shared, 'make_ac_fn', return_value=object()) as load, \
             patch.object(shared, 'evaluate_seeded', side_effect=lambda *a, **k: result(seed=a[4])) as evaluate, \
             patch.object(shared, 'certify_combat_move') as teacher, \
             patch.object(shared, 'digest', return_value='test-only'), patch.object(shared, 'dump'):
            results = shared.evaluate_models(runner.ROOT / 'unused', threading.Event(), False,
                                            arm_names=runner.ARMS, test_seed=runner.TEST_SEED)
        self.assertEqual(set(results), {'original', 'control', 'gae'})
        self.assertTrue(all(c.args[3:6] == (400, 35000, 500) for c in evaluate.call_args_list))
        self.assertTrue(all(c.args[1:] == (9, 'cpu') for c in load.call_args_list))
        teacher.assert_not_called()

    def test_interrupted_test_range_is_reserved_even_without_evaluation_file(self):
        marker = {'seed': 35000, 'num_episodes': 400, 'num_agents': 2}
        with patch.object(Path, 'rglob', side_effect=[[], [Path('test_opened.json')]]), \
             patch.object(Path, 'read_text', return_value=json.dumps(marker)):
            with self.assertRaisesRegex(ValueError, 'already opened'):
                shared.fresh_test_check(35000)


if __name__ == '__main__':
    unittest.main()
