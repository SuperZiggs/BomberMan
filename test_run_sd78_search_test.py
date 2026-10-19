import unittest

import run_sd78_search_test as sd78


class SearchTestTests(unittest.TestCase):
    def test_gate(self):
        def paired(active, lower, suicide):
            return {'active_win': {'difference_pp': active, 'ci95_pp': [lower, active + 5]},
                    'suicide': {'difference_pp': suicide}}
        self.assertTrue(sd78.gate(paired(5.0, 0.25, 1.0))['pass'])
        self.assertFalse(sd78.gate(paired(4.75, 1.0, 0.0))['pass'])
        self.assertFalse(sd78.gate(paired(8.0, 0.0, 0.0))['pass'])
        self.assertFalse(sd78.gate(paired(8.0, 1.0, 1.25))['pass'])

    def test_summarize_rates(self):
        games = [dict(win=1, active_win=1, passive_win=0, sudden_death_win=0, draw=0, suicide=0, sudden_death_death=0,
                      kills=1, bombs=30, steps=200),
                 dict(win=0, active_win=0, passive_win=0, sudden_death_win=0, draw=1, suicide=0, sudden_death_death=0,
                      kills=0, bombs=40, steps=500)]
        s = sd78.summarize(games)
        self.assertEqual((s['active_win_rate'], s['draw_rate'], s['avg_bombs']), (50.0, 50.0, 35.0))


if __name__ == '__main__':
    unittest.main()
