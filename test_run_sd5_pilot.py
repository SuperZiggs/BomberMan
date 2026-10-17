import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import run_sd5_pilot as pilot


class ConfigTests(unittest.TestCase):
    def test_arms_differ_only_in_the_search_teacher(self):
        control, search = (pilot.train_config(arm, Path('x'), 'cpu') for arm in pilot.ARMS)
        self.assertEqual({k for k in control if control[k] != search[k]}, {'search_teacher', 'checkpoint_dir'})
        self.assertTrue(search['search_teacher'])
        self.assertEqual((control['episodes'], control['resume_from'], control['validation_seed'],
                          control['validation_episodes'], control['validation_every']),
                         (3000, str(pilot.F), 21000, 400, 500))
        self.assertEqual((control['sudden_death_probability'], control['sudden_death_start'],
                          control['sudden_death_interval'], control['sudden_death_warning_steps']), (0.75, 475, 25, 40))
        self.assertTrue(control['save_training_state'])

    def test_gate_needs_five_points_over_f_a_positive_ci_and_no_loss_to_control(self):
        def paired(diff, low):
            return {'difference_pp': diff, 'ci95_pp': [low, diff + 4]}
        ok = {'search_vs_f': {'active_win': paired(6, 1), 'suicide': paired(0.5, -1)},
              'search_vs_control': {'active_win': paired(0, -4)}}
        self.assertTrue(pilot.gate(ok)['pass'])
        for path, value in ((('search_vs_f', 'active_win'), paired(4.75, 1)),
                            (('search_vs_f', 'active_win'), paired(6, 0)),
                            (('search_vs_control', 'active_win'), paired(-0.25, -5)),
                            (('search_vs_f', 'suicide'), paired(1.25, 0))):
            bad = json.loads(json.dumps(ok))
            bad[path[0]][path[1]] = value
            self.assertFalse(pilot.gate(bad)['pass'], path)


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_smoke_train_refuses_overwrite_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder, patch('builtins.print'):
            out = Path(folder) / 'run'
            pilot.train_arm(out, 'search', 'cpu', smoke=True)
            arm = out / 'search'
            self.assertTrue((arm / 'best_phaseC.pt').exists() and (arm / 'training_state.pt').exists())
            self.assertTrue((arm / 'source' / 'kill_search.py').exists() and (arm / 'registration.json').exists())
            rows = [json.loads(line) for line in (arm / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['episode'] for r in rows], list(range(1, pilot.SMOKE['episodes'] + 1)))
            self.assertIn('search', rows[0])
            with self.assertRaises(FileExistsError):
                pilot.train_arm(out, 'search', 'cpu', smoke=True)
            pilot.train_arm(out, 'search', 'cpu', smoke=True, resume=True)  # completed state: no new episodes
            rows = (arm / 'metrics.jsonl').read_text().splitlines()
            self.assertEqual(len(rows), pilot.SMOKE['episodes'])

    def test_test_stage_needs_both_models_and_opens_once(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with self.assertRaises(FileNotFoundError):
                pilot.test(out, smoke=True)
            for arm in pilot.ARMS:
                (out / arm).mkdir()
                (out / arm / 'best_phaseC.pt').write_bytes(pilot.F.read_bytes())
            with patch('builtins.print'):
                summary = pilot.test(out, smoke=True)
            self.assertIn('pass', summary['gate'])
            self.assertEqual(set(summary['results']), {f'{m}_{r}' for m in pilot.MODELS for r in pilot.RULES})
            with self.assertRaises(FileExistsError):
                pilot.test(out, smoke=True)


if __name__ == '__main__':
    unittest.main()
