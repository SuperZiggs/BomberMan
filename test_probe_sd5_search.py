import unittest

import torch

import probe_sd5_search as probe

FAST = dict(sudden_death_start=12, sudden_death_interval=6, sudden_death_warning_steps=8)


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shards_reproduce_one_run_and_rates_follow_games(self):
        whole = probe.evaluate_arm('f', seed=21000, games=4, max_steps=30, schedule=FAST)
        parts = [probe.evaluate_arm('f', seed=21000 + i, games=2, max_steps=30, schedule=FAST) for i in (0, 1)]
        merged = probe.merge(parts)
        self.assertEqual(merged['games'], whole['games'])
        for key in probe.RATES:
            self.assertAlmostEqual(merged[f'{key}_rate'], whole[f'{key}_rate'])
        self.assertTrue(all('contact' in g for g in whole['games']))

    def test_search_arm_runs_and_test_seeds_are_refused(self):
        result = probe.evaluate_arm('search', seed=21000, games=2, max_steps=30, schedule=FAST)
        self.assertEqual(len(result['games']), 2)
        with self.assertRaises(ValueError):
            probe.evaluate_arm('f', seed=42000, games=2)

    def test_draw_breakdown_and_decision_rule(self):
        games = [dict(draw=1, steps=500, contact={'first_close_step': None}),
                 dict(draw=1, steps=230, contact={'first_close_step': 40}),
                 dict(draw=0, steps=90, contact={'first_close_step': 20})]
        self.assertEqual(probe.draw_breakdown(games), {'draws': 2, 'timeout': 1, 'both_died': 1,
                                                       'timeout_never_close': 1})
        self.assertTrue(probe.decide({'difference_pp': 3.0, 'ci95_pp': [0.25, 6.0]}))
        self.assertFalse(probe.decide({'difference_pp': 4.0, 'ci95_pp': [0.0, 8.0]}))
        self.assertFalse(probe.decide({'difference_pp': 2.75, 'ci95_pp': [0.5, 5.0]}))


if __name__ == '__main__':
    unittest.main()
