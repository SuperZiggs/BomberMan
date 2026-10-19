import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import run_sd7_pilot as pilot


class ConfigTests(unittest.TestCase):
    def test_group_arm_is_the_sd51_control_plus_group_updates(self):
        control = json.loads(pilot.CONTROL_CONFIG.read_text(encoding='utf-8'))
        group = pilot.train_config(Path('x'))
        self.assertEqual({k for k in group if group.get(k) != control.get(k)},
                         {'checkpoint_dir', *pilot.GROUP})
        self.assertTrue(group['group_rl'])
        self.assertEqual((group['episodes'], group['seed'], group['resume_from']),
                         (3000, control['seed'], control['resume_from']))

    def test_gate_and_mechanism_rule(self):
        def paired(diff, low):
            return {'difference_pp': diff, 'ci95_pp': [low, diff + 4]}
        ok = {'group_vs_f': {'active_win': paired(6, 1), 'suicide': paired(0.5, -1)},
              'group_vs_control': {'active_win': paired(0, -4)}}
        self.assertTrue(pilot.gate(ok)['pass'])
        bad = json.loads(json.dumps(ok)); bad['group_vs_f']['active_win'] = paired(4.75, 1)
        self.assertFalse(pilot.gate(bad)['pass'])
        self.assertTrue(pilot.mechanism_passes({'difference_pp': 10.0, 'ci95_pp': [2.0, 18.0]}))
        self.assertFalse(pilot.mechanism_passes({'difference_pp': 9.5, 'ci95_pp': [2.0, 18.0]}))
        self.assertFalse(pilot.mechanism_passes({'difference_pp': 12.0, 'ci95_pp': [0.0, 20.0]}))

    def test_paired_state_bootstrap(self):
        result = pilot.paired_states([[1, 1], [0, 1], [0, 0]], [[0, 1], [0, 0], [0, 0]])
        self.assertAlmostEqual(result['difference_pp'], 100 * (2 / 6))
        self.assertLessEqual(result['ci95_pp'][0], result['difference_pp'])


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_smoke_train_refuses_overwrite_and_resumes(self):
        with tempfile.TemporaryDirectory() as folder, patch('builtins.print'):
            out = Path(folder) / 'run'
            pilot.train_arm(out, 'cpu', smoke=True)
            arm = out / 'group'
            rows = [json.loads(line) for line in (arm / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), pilot.SMOKE['episodes'])
            self.assertIn('group', rows[0])
            with self.assertRaises(FileExistsError):
                pilot.train_arm(out, 'cpu', smoke=True)
            pilot.train_arm(out, 'cpu', smoke=True, resume=True)
            self.assertEqual(len((arm / 'metrics.jsonl').read_text().splitlines()), pilot.SMOKE['episodes'])

    def test_mechanism_shard_replays_every_model_from_the_same_states(self):
        models = {name: pilot.F for name in pilot.MODELS}
        rows = pilot.mechanism_shard(models, 6650, 2, 60, queue.Queue())
        for row in rows:
            self.assertEqual(set(row['outcomes']), set(pilot.MODELS))
            self.assertEqual(row['outcomes']['F'], row['outcomes']['group'])  # same weights, same seeds

    def test_test_stage_opens_once(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'group').mkdir()
            (out / 'group' / 'best_phaseC.pt').write_bytes(pilot.F.read_bytes())
            with patch.object(pilot, 'CONTROL_BEST', pilot.F), patch('builtins.print'):
                summary = pilot.test(out, smoke=True)
                self.assertIn('pass', summary['gate'])
                with self.assertRaises(FileExistsError):
                    pilot.test(out, smoke=True)


if __name__ == '__main__':
    unittest.main()
