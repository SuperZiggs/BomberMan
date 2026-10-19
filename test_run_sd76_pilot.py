import json
import unittest

import run_sd7_pilot as sd7
import run_sd76_pilot as pilot


class PilotTests(unittest.TestCase):
    def test_arms_start_from_g_and_differ_only_in_group_keys(self):
        control, buffer, full = (pilot.train_config('out', arm) for arm in ('control', 'buffer', 'full'))
        base = json.loads(sd7.CONTROL_CONFIG.read_text(encoding='utf-8'))
        self.assertEqual({k for k in buffer if control.get(k) != buffer[k]}, set(pilot.BUFFER) | {'checkpoint_dir'})
        self.assertEqual(control['resume_from'], str(pilot.G))
        self.assertEqual((control['episodes'], control['validation_every'], control['seed']), (1000, 0, base['seed']))
        self.assertEqual(full, {**base, 'resume_from': str(pilot.G), **pilot.BUFFER, 'checkpoint_dir': full['checkpoint_dir']})
        self.assertEqual((full['episodes'], full['validation_every']), (3000, 500))

    def test_mechanism_rule(self):
        def summary(diff, lower, deaths, validation):
            return {'buffer_vs_control': {'difference_pp': diff, 'ci95_pp': [lower, diff + 3]},
                    'deaths': {'control': 0.01, 'buffer': deaths},
                    'validation': {'buffer_vs_control': {'difference_pp': validation}}}
        self.assertTrue(pilot.mechanism_passes(summary(10.0, 0.5, 0.02, -2.0)))
        self.assertFalse(pilot.mechanism_passes(summary(9.9, 0.5, 0.0, 0.0)))
        self.assertFalse(pilot.mechanism_passes(summary(12.0, 0.0, 0.0, 0.0)))
        self.assertFalse(pilot.mechanism_passes(summary(12.0, 1.0, 0.021, 0.0)))
        self.assertFalse(pilot.mechanism_passes(summary(12.0, 1.0, 0.0, -2.25)))

    def test_gate(self):
        def paired(active, lower, suicide):
            return {'full_vs_g': {'active_win': {'difference_pp': active, 'ci95_pp': [lower, active + 5]},
                                  'suicide': {'difference_pp': suicide}}}
        self.assertTrue(pilot.gate(paired(5.0, 0.25, 1.0))['pass'])
        self.assertFalse(pilot.gate(paired(4.75, 0.25, 0.0))['pass'])
        self.assertFalse(pilot.gate(paired(6.0, 0.0, 0.0))['pass'])
        self.assertFalse(pilot.gate(paired(6.0, 1.0, 1.25))['pass'])


if __name__ == '__main__':
    unittest.main()
