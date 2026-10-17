"""Paired configuration, source guards, selection gates and incomplete-run routing."""
from copy import deepcopy
import io
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, mock_open, patch
from uuid import uuid4

import torch

import run_combat_move_pilot as runner


def ready_arm(output, arm):
    return {'completed_episodes': 200, 'expected_episodes': 200, 'optimizer_steps': 200,
        'optimizer_instances': 1, 'training_complete': True, 'validation_complete': True,
        'skill_probes_complete': True, 'checkpoints': {'best': {
            'path': str(output / arm / 'best_phaseC.pt'), 'sha256': 'test-only-source'}}}


def result(active=0, suicide=0, seed=33000, count=400):
    return {'num_episodes': count, 'win_rate': active, 'active_win_rate': active,
        'suicide_rate': suicide, 'seed': seed, 'num_agents': 2,
        'by_corner': {str(a): {'win_rate': active, 'active_win_rate': active} for a in (0, 1)},
        'games': [{'seed': seed + i // 2, 'aid': i % 2,
                   'active_win': int(i < active * count / 100)} for i in range(count)]}


class PilotChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def run_mocked(self, arms=None, profile=None, results=None, smoke=False, edit_source=False):
        output = runner.ROOT / ('runs/uncreated-combat-move-' + uuid4().hex)
        arms = arms or {arm: ready_arm(output, arm) for arm in runner.ARMS}
        results = results if results is not None else {name: result(100 if name == 'combat_move' else 0)
                                                     for name in ('original', 'control', 'combat_move')}
        changed = False
        def train(*args):
            nonlocal changed
            if edit_source:
                changed = True
            return arms[args[1]]
        def digest(path):
            if path == runner.WARM_START:
                return runner.B_SHA256
            return 'edited-source' if changed and path.name == 'train_actor_critic.py' else 'test-only-source'
        baseline = {'model_state_sha256': 'test-only-model', 'records': []}
        with patch.object(runner, 'digest', side_effect=digest), patch.object(runner, 'fresh_test_check') as fresh, \
             patch.object(runner, 'preflight', return_value=profile if profile is not None else {'passed': True}), \
             patch.object(runner, 'BaseNet'), patch.object(runner.trainer, 'smart_load_weights'), \
             patch.object(runner, 'skill_probe', return_value=baseline), \
             patch.object(runner, 'train_arm', side_effect=train) as training, \
             patch.object(runner, 'evaluate_models', return_value=results) as evaluation, \
             patch.object(runner, 'stage_timer', side_effect=lambda seconds: (threading.Event(), Mock())), \
             patch('pathlib.Path.mkdir'), patch('pathlib.Path.write_text'), \
             patch.object(runner.shutil, 'copy2'), patch.object(runner.shutil, 'make_archive'), \
             patch.object(runner, 'dump') as save, patch('builtins.print'):
            if edit_source:
                with self.assertRaises(AssertionError):
                    runner.run(output, smoke=smoke)
                summary = next(c.args[1] for c in save.call_args_list if c.args[0].name == 'summary.json')
            else:
                summary = runner.run(output, smoke=smoke)
        return summary, training, evaluation, fresh, save

    def test_complete_pair_registers_protocol_before_training_and_gate_allows_replication_only(self):
        summary, training, evaluation, fresh, save = self.run_mocked()
        self.assertEqual(summary['status'], 'complete')
        self.assertTrue(summary['eligible_for_replication'])
        self.assertEqual(summary['decision'], 'replicate_before_replacing')
        self.assertEqual([c.args[1] for c in training.call_args_list], ['control', 'combat_move'])
        self.assertEqual(evaluation.call_count, 1)
        fresh.assert_called_once()
        self.assertEqual(save.call_args_list[0].args[0].name, 'config.json')
        names = [c.args[0].name for c in save.call_args_list]
        self.assertLess(names.index('selected_models.json'), names.index('summary.json'))

    def test_failed_preflight_starts_no_train_or_test(self):
        summary, training, evaluation, _, _ = self.run_mocked(profile={'passed': False})
        self.assertEqual(summary['status'], 'incomplete')
        self.assertFalse(summary['eligible_for_replication'])
        training.assert_not_called()
        evaluation.assert_not_called()

    def test_missing_validation_or_skill_probe_never_opens_test(self):
        output = runner.ROOT / 'unused'
        for key in ('validation_complete', 'skill_probes_complete', 'training_complete'):
            arms = {arm: ready_arm(output, arm) for arm in runner.ARMS}
            arms['combat_move'][key] = False
            summary, _, evaluation, _, _ = self.run_mocked(arms=arms)
            self.assertEqual(summary['status'], 'incomplete')
            self.assertFalse(summary['evaluation_complete'])
            evaluation.assert_not_called()

    def test_incomplete_test_and_smoke_never_qualify(self):
        summary, _, _, _, _ = self.run_mocked(results={'original': result()})
        self.assertEqual(summary['reason'], 'evaluation_incomplete')
        self.assertFalse(summary['eligible_for_replication'])
        summary, _, _, fresh, _ = self.run_mocked(smoke=True)
        self.assertTrue(summary['evaluation_complete'])
        self.assertFalse(summary['full_protocol'])
        self.assertFalse(summary['eligible_for_replication'])
        self.assertEqual(summary['comparisons'], {})
        fresh.assert_not_called()

    def test_source_change_blocks_evaluation_and_retains_original(self):
        summary, _, evaluation, _, _ = self.run_mocked(edit_source=True)
        self.assertFalse(summary['hashes_unchanged'])
        self.assertEqual(summary['decision'], 'retain_original')
        evaluation.assert_not_called()

    def test_ready_requires_one_optimizer_exact_updates_and_episode_count(self):
        arms = {arm: ready_arm(runner.ROOT / 'unused', arm) for arm in runner.ARMS}
        self.assertTrue(runner.pair_ready(arms))
        for key, value in (('completed_episodes', 199), ('optimizer_steps', 199), ('optimizer_instances', 2)):
            broken = deepcopy(arms)
            broken['combat_move'][key] = value
            self.assertFalse(runner.pair_ready(broken))

    def test_each_gain_ci_and_suicide_gate_is_required(self):
        models = {'original': result(), 'control': result(), 'combat_move': result(5)}
        comparisons, passed = runner.comparison_gate(models)
        self.assertTrue(passed)
        self.assertEqual(comparisons['control']['board_pairs'], 200)
        models['combat_move'] = result(4)
        self.assertFalse(runner.comparison_gate(models)[1])
        models['combat_move'] = result(100, suicide=0.5)
        self.assertFalse(runner.comparison_gate(models)[1])
        models['combat_move'] = result(100)
        with patch.object(runner, 'paired_active_bootstrap', return_value={'difference_pp': 6, 'ci95_pp': [0, 12]}):
            self.assertFalse(runner.comparison_gate(models)[1])

    def test_evaluation_uses_cpu_categorical_without_teacher_on_paired_fresh_seeds(self):
        def evaluate(policy, board, agents, count, seed, steps, **kwargs):
            self.assertEqual((board, agents, count, seed, steps), (9, 2, 400, 33000, 500))
            return result()
        with patch.object(runner, 'make_ac_fn', return_value=object()) as load, \
             patch.object(runner, 'evaluate_seeded', side_effect=evaluate), \
             patch.object(runner, 'certify_combat_move') as certifier, \
             patch.object(runner, 'digest', return_value='test-only'), patch.object(runner, 'dump'):
            results = runner.evaluate_models(runner.ROOT / 'unused', threading.Event(), smoke=False)
        self.assertEqual(set(results), {'original', 'control', 'combat_move'})
        self.assertTrue(all(c.args[1:] == (9, 'cpu') for c in load.call_args_list))
        certifier.assert_not_called()

    def test_existing_output_bad_budget_and_unavailable_gpu_stop_before_writes(self):
        with patch.object(runner, 'train_arm') as training, patch('pathlib.Path.mkdir') as mkdir:
            with self.assertRaises(FileExistsError):
                runner.run(runner.ROOT)
            with self.assertRaises(ValueError):
                runner.run(runner.ROOT / 'unused', budget_minutes=121)
            with patch.object(runner.torch.cuda, 'is_available', return_value=False):
                with self.assertRaises(ValueError):
                    runner.run(runner.ROOT / 'unused', device='cuda')
        training.assert_not_called()
        mkdir.assert_not_called()

    def test_used_test_seed_range_is_rejected_before_training(self):
        with patch.object(Path, 'rglob', return_value=[Path('evaluation_seen.json')]), \
             patch.object(Path, 'read_text', return_value=json.dumps(result())):
            with self.assertRaisesRegex(ValueError, 'already opened'):
                runner.fresh_test_check()


if __name__ == '__main__':
    unittest.main()
