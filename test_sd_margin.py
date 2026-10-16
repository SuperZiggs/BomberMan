import unittest

import run_sd_margin as margin


def row(k, active, sd_deaths):
    return {'margin': k, 'active_win_rate': active, 'learner_sd_deaths': sd_deaths}


def games(flags, metric):
    return [{'seed': s, 'aid': a, 'win': 0, 'active_win': 0, 'draw': 0, 'suicide': 0,
             'learner_death': 'alive', metric: flags(s)} for s in range(100) for a in (0, 1)]


class MarginTests(unittest.TestCase):
    def test_select_needs_fewer_ring_deaths_then_best_active_then_smaller_k(self):
        rows = [row(0, 30, 100), row(5, 33, 90), row(10, 33, 80), row(15, 40, 100), row(20, 20, 50)]
        self.assertEqual(margin.select(rows)['margin'], 5)
        self.assertIsNone(margin.select([row(0, 30, 100), row(5, 40, 100)]))

    def test_gate(self):
        base = {'suicide_rate': 1.0, 'games': games(lambda s: 'sudden_death' if s < 40 else 'alive', 'learner_death')}
        good = {'suicide_rate': 1.5, 'games': games(lambda s: 'sudden_death' if s < 10 else 'alive', 'learner_death')}
        self.assertTrue(margin.gate(good, base)['passed'])
        self.assertFalse(margin.gate({**good, 'suicide_rate': 2.5}, base)['passed'])
        self.assertFalse(margin.gate(base, base)['passed'])

    def test_sweep_refuses_test_seeds(self):
        with self.assertRaises(ValueError):
            margin.sweep(None, seed=41000)


if __name__ == '__main__':
    unittest.main()
