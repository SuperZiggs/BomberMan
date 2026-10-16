import unittest

import torch

import diagnose_sd_ring_deaths as diag
import run_sudden_death_experiment as runner

TIMINGS = {'sudden_death_start': 12, 'sudden_death_interval': 6,
           'sudden_death_warning_steps': 8, 'sudden_death_max_ring': 2}


class DiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_recording_does_not_change_games_and_aligns(self):
        plain = runner.evaluate_case('both_aware', games=4, seed=21000, max_steps=30, timings=TIMINGS)
        result, per_game = diag.run_diagnosis(4, 21000, TIMINGS, max_steps=30)
        self.assertEqual(plain['games'], result['games'])
        events = diag.ring_events(result, per_game, TIMINGS)
        self.assertTrue(events)
        for e in events:
            if e['side'] == 'learner':
                game = next(g for g in result['games'] if g['seed'] == e['seed'] and g['aid'] == e['learner_aid'])
                if e['died']:
                    self.assertEqual(game['learner_death'], 'sudden_death')
        diag.summarize(events)

    def test_refuses_test_seeds(self):
        with self.assertRaises(ValueError):
            diag.run_diagnosis(2, 41000)


if __name__ == '__main__':
    unittest.main()
