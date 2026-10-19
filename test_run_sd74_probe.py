import json
import unittest

import run_sd7_pilot as sd7
import run_sd74_probe as probe


class ProbeTests(unittest.TestCase):
    def test_arms_differ_only_in_group_keys(self):
        control, group = probe.train_config('out', 'control'), probe.train_config('out', 'group')
        base = json.loads(sd7.CONTROL_CONFIG.read_text(encoding='utf-8'))
        self.assertEqual({k for k in group if control.get(k) != group[k]}, set(probe.GROUP) | {'checkpoint_dir'})
        self.assertNotIn('group_rl', control)
        self.assertEqual((control['episodes'], control['validation_every'], control['seed']), (1000, 0, base['seed']))

    def test_full_is_sd5_control_plus_group(self):
        full = probe.train_config('out', 'full')
        base = json.loads(sd7.CONTROL_CONFIG.read_text(encoding='utf-8'))
        self.assertEqual(full, {**base, **probe.GROUP, 'checkpoint_dir': full['checkpoint_dir']})
        self.assertEqual((full['episodes'], full['validation_every']), (3000, 500))

    def test_rule(self):
        def summary(diff, lower, deaths):
            return {'group_vs_control': {'difference_pp': diff, 'ci95_pp': [lower, diff + 3]},
                    'deaths': {'control': 0.01, 'group': deaths}}
        self.assertTrue(probe.probe_passes(summary(5.0, 0.5, 0.02)))
        self.assertFalse(probe.probe_passes(summary(4.9, 0.5, 0.0)))
        self.assertFalse(probe.probe_passes(summary(8.0, 0.0, 0.0)))
        self.assertFalse(probe.probe_passes(summary(8.0, 1.0, 0.021)))


if __name__ == '__main__':
    unittest.main()
