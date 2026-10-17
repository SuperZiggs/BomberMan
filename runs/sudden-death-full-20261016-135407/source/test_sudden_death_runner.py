import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate
import run_sudden_death_experiment as runner
from bomberman_env import EMPTY, WOODEN_WALL


TIMINGS = {'sudden_death_start': 4, 'sudden_death_interval': 20,
           'sudden_death_warning_steps': 3}


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_off_matches_existing_evaluator_and_preserves_rng(self):
        policy = evaluate.make_ac_fn(str(runner.CHECKPOINT), 9, 'cpu')
        reference = evaluate.evaluate_seeded(policy, 9, 2, 2, 39888, 20,
                                               contact_diagnostics=True)
        numpy_before, torch_before = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
        actual = runner.evaluate_case('off', games=2, seed=39888, max_steps=20, timings=TIMINGS)
        self.assertEqual(numpy_before, pickle.dumps(np.random.get_state()))
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        for left, right in zip(reference['games'], actual['games']):
            self.assertEqual(left, {k: right[k] for k in left})
        for key in ('win_rate', 'active_win_rate', 'draw_rate', 'suicide_rate', 'avg_kills'):
            self.assertEqual(reference[key], actual[key])

    def test_sd_win_is_passive_and_simultaneous_sd_is_draw(self):
        def stationary(*_args):
            return 0
        def escape_ring(env, _obs, aid):
            return 2 if env.step_count < 2 else ((4 if aid == 0 else 3) if env.step_count == 2 else 0)
        original_reset = runner.SuddenDeathEnv.reset
        def open_board(env):
            obs = original_reset(env)
            env.grid[env.grid == WOODEN_WALL] = EMPTY  # scripted path needs no wood
            return obs
        with patch.object(evaluate, 'make_ac_fn', return_value=escape_ring), \
             patch.object(evaluate, 'make_rule_based_fn', return_value=stationary), \
             patch.object(runner.SuddenDeathEnv, 'reset', open_board):
            result = runner.evaluate_case('rings_only', games=2, seed=39888, max_steps=5, timings=TIMINGS)
        self.assertEqual(result['win_rate'], 100)
        self.assertEqual(result['active_win_rate'], 0)
        self.assertEqual(result['passive_win_rate'], 100)
        self.assertEqual(result['sudden_death']['wins_opponent_sd'], 2)
        with patch.object(evaluate, 'make_ac_fn', return_value=stationary), \
             patch.object(evaluate, 'make_rule_based_fn', return_value=stationary):
            result = runner.evaluate_case('rings_only', games=2, seed=39888, max_steps=5, timings=TIMINGS)
        self.assertEqual(result['draw_rate'], 100)
        self.assertEqual(result['sudden_death']['both_sd_draws'], 2)
        self.assertEqual(result['suicide_rate'], 0)

    def test_paired_metrics_distinguish_wr_from_active_wr(self):
        results = {}
        for case in runner.CASES:
            win = int(case != 'off')
            results[case] = {'games': [{'seed': 9, 'aid': aid, 'win': win, 'active_win': 0,
                                        'draw': 1 - win, 'suicide': 0} for aid in (0, 1)]}
        comparison = runner.comparisons(results)['rings_vs_off']
        self.assertEqual(comparison['win']['difference_pp'], 100)
        self.assertEqual(comparison['active_win']['difference_pp'], 0)
        self.assertEqual(comparison['draw']['difference_pp'], -100)
        self.assertEqual(comparison['win']['board_pairs'], 1)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / 'runs') as folder:
            with self.assertRaises(FileExistsError):
                runner.run(folder, smoke=True)


if __name__ == '__main__':
    unittest.main()
