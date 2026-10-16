import unittest

import torch

import probe_sd_endgame as probe

TIMINGS = {'sudden_death_start': 12, 'sudden_death_interval': 6,
           'sudden_death_warning_steps': 8, 'sudden_death_max_ring': 2}


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_switch_after_game_end_keeps_b_games(self):
        base = probe.evaluate_arm('b', seed=21000, games=4, timings=TIMINGS, max_steps=30)
        for arm in ('local', 'rulebot'):
            late = probe.evaluate_arm(arm, seed=21000, games=4, switch=10_000, timings=TIMINGS, max_steps=30)
            self.assertEqual(base['games'], late['games'])

    def test_teacher_arms_run_and_refuse_test_seeds(self):
        for arm in (*probe.ARMS, 'search'):
            result = probe.evaluate_arm(arm, seed=21000, games=2, switch=4, timings=TIMINGS, max_steps=30)
            self.assertEqual(len(result['games']), 2)
        with self.assertRaises(ValueError):
            probe.evaluate_arm('local', seed=42000, games=2)


if __name__ == '__main__':
    unittest.main()
