"""The full-run helper must reject stale controls and never overwrite old runs."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch, mock_open

import run_two_bomb_complete as runner


class CompleteRunChecks(unittest.TestCase):
    def archived_reference(self):
        real_digest = runner.digest
        def frozen_digest(path):
            if path.parent == runner.ROOT and path.name in runner.CORE_FILES:
                return real_digest(runner.REFERENCE / 'source' / path.name)
            return real_digest(path)
        with patch.object(runner, 'digest', side_effect=frozen_digest):
            return runner.reference_inputs()

    def test_real_reference_is_complete_and_unchanged(self):
        manifest, original, local = self.archived_reference()
        self.assertEqual(manifest['training']['episodes'], 200)
        self.assertNotEqual(original, local)

    def test_current_trainer_cannot_reuse_the_old_control(self):
        with self.assertRaisesRegex(ValueError, 'rerun both teachers'):
            runner.reference_inputs()

    def test_changed_source_cannot_reuse_control(self):
        real_digest = runner.digest
        with patch.object(runner, 'digest', side_effect=lambda p: 'changed' if p.name == 'tactical_teacher.py' else real_digest(p)):
            with self.assertRaisesRegex(ValueError, 'rerun both teachers'):
                runner.reference_inputs()

    def test_existing_output_is_not_overwritten_or_trained(self):
        with patch.object(runner, 'reference_inputs', return_value=self.archived_reference()), \
             patch.object(runner.torch.cuda, 'is_available', return_value=True), \
             patch.object(runner, 'train') as train:
            with self.assertRaises(FileExistsError):
                runner.run(runner.ROOT)
            train.assert_not_called()

    def test_missing_cuda_rejects_training_before_creating_output(self):
        with patch.object(runner, 'reference_inputs', return_value=self.archived_reference()), \
             patch.object(runner.torch.cuda, 'is_available', return_value=False), \
             patch.object(runner, 'train') as train, patch('pathlib.Path.mkdir') as mkdir:
            with self.assertRaisesRegex(RuntimeError, 'control used CUDA'):
                runner.run(runner.ROOT / 'unused-test-output')
            train.assert_not_called()
            mkdir.assert_not_called()

    def test_complete_flow_uses_original_warm_start_and_fresh_paired_seeds(self):
        def fake_train(**config):
            for ep in range(1, 201):
                config['on_episode']({'episode': ep})
            return None, list(range(200)), None

        def fake_evaluate(*args, **kwargs):
            self.assertEqual(args[1:], (9, 2, 400, 26000, 500))
            return {'num_episodes': 400, 'win_rate': 0, 'active_win_rate': 0, 'suicide_rate': 0,
                    'by_corner': {str(a): {'win_rate': 0, 'active_win_rate': 0} for a in (0, 1)},
                    'games': [{'seed': s, 'aid': a, 'active_win': 0}
                              for s in range(26000, 26200) for a in (0, 1)]}

        reference = self.archived_reference()
        validations = '\n'.join(json.dumps({'episode': ep, 'num_episodes': 100, 'seed': 21000})
                                for ep in (0, 100, 200))
        with patch.object(runner, 'reference_inputs', return_value=reference), \
             patch.object(runner.torch.cuda, 'is_available', return_value=True), \
             patch.object(runner.torch.cuda, 'get_device_name', return_value='test-only GPU'), \
             patch.object(runner, 'digest', return_value='test-only hash'), \
             patch('pathlib.Path.mkdir'), patch('pathlib.Path.open', mock_open()), \
             patch('pathlib.Path.read_text', side_effect=[validations, '{"episode": 0}']), \
             patch.object(runner.shutil, 'copy2'), patch.object(runner, 'save_json') as save, \
             patch.object(runner, 'train', side_effect=fake_train) as train, \
             patch.object(runner, 'make_ac_fn', return_value=object()), \
             patch.object(runner, 'evaluate_seeded', side_effect=fake_evaluate) as evaluate, \
             patch('builtins.print'):
            output = runner.ROOT / 'unused-test-output'
            summary = runner.run(output)
            self.assertEqual(Path(train.call_args.kwargs['resume_from']), reference[1])
            self.assertEqual(train.call_args.kwargs['attack_teacher'], 'two_bomb')
            self.assertEqual(train.call_args.kwargs['seed'], 2026101303)
            self.assertNotIn('stop_event', train.call_args.kwargs)
            self.assertEqual(evaluate.call_count, 3)
            self.assertTrue(summary['training_complete'])
            self.assertFalse(summary['eligible_for_replication'])
            self.assertEqual(summary['decision'], 'retain_original')
            self.assertEqual(summary['comparisons']['local']['board_pairs'], 200)
            self.assertEqual(save.call_args.args[0], output / 'summary.json')


if __name__ == '__main__':
    unittest.main()
