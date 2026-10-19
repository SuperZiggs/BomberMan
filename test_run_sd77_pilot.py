import unittest

import run_sd76_pilot as sd76
import run_sd77_pilot as pilot


class PilotTests(unittest.TestCase):
    def test_arms_are_sd76_buffer_plus_anchor(self):
        anchor, full = pilot.train_config('out', 'anchor', 4.0), pilot.train_config('out', 'full', 4.0)
        buffer, base_full = sd76.train_config('out', 'buffer'), sd76.train_config('out', 'full')
        self.assertEqual({k for k in anchor if buffer.get(k) != anchor[k]}, {'anchor_coef', 'checkpoint_dir'})
        self.assertEqual({k for k in full if base_full.get(k) != full[k]}, {'anchor_coef'})
        self.assertEqual((anchor['episodes'], full['episodes'], anchor['resume_from']), (1000, 3000, str(sd76.G)))

    def test_beta_rule(self):
        def pick(rows):
            return pilot.choose({'G': {'p_bomb': 0.64},
                                 **{f'beta_{b:g}': {'p_bomb': pb, 'p_kill': pk} for b, (pb, pk) in rows.items()}})
        rows = {0.0: (0.9, 0.95), 1.0: (0.75, 0.93), 4.0: (0.68, 0.85), 16.0: (0.65, 0.78)}
        self.assertEqual(pick(rows), 4.0)      # beta 0 and 1 still spill bombs
        rows[16.0] = (0.65, 0.85)
        self.assertEqual(pick(rows), 4.0)      # tie on P(kill): the smaller beta
        self.assertIsNone(pick({b: (0.8, 0.9) for b in pilot.BETAS}))

    def test_mechanism_rule(self):
        def summary(diff=12.0, lower=1.0, deaths=0.0, kills=0.47, bombs=35.0, active=0.0):
            return {'anchor_vs_control': {'difference_pp': diff, 'ci95_pp': [lower, diff + 3]},
                    'deaths': {'control': 0.0, 'anchor': deaths},
                    'validation': {'anchor': {'avg_kills': kills, 'avg_bombs': bombs},
                                   'control': {'avg_kills': 0.46, 'avg_bombs': 32.13},
                                   'anchor_vs_control': {'difference_pp': active}}}
        self.assertTrue(pilot.mechanism_passes(summary()))
        for bad in (dict(diff=9.9), dict(lower=0.0), dict(deaths=0.011), dict(kills=0.45),
                    dict(bombs=35.4), dict(active=-0.25)):
            self.assertFalse(pilot.mechanism_passes(summary(**bad)), bad)


if __name__ == '__main__':
    unittest.main()
