import queue
import unittest

from bomberman_env import ACTION_STOP, ACTION_UP
import probe_sd6_cutoff as probe
from rule_based_bot import _safe_action_mask
from test_cutoff_teacher import corridor


class ProbeTests(unittest.TestCase):
    def test_cutoff_arm_overrides_only_when_the_policy_misses_the_trap(self):
        env = corridor()
        self.assertEqual(probe.with_cutoff(lambda e, o, a: ACTION_STOP)(env, None, 0), ACTION_UP)
        self.assertEqual(probe.with_cutoff(lambda e, o, a: ACTION_UP)(env, None, 0), ACTION_UP)
        env.bombs.clear()
        self.assertEqual(probe.with_cutoff(lambda e, o, a: ACTION_STOP)(env, None, 0), ACTION_STOP)

    def test_decision_rule(self):
        self.assertTrue(probe.decide({'difference_pp': 3.0, 'ci95_pp': [0.25, 6.0]}))
        self.assertFalse(probe.decide({'difference_pp': 2.75, 'ci95_pp': [0.25, 6.0]}))
        self.assertFalse(probe.decide({'difference_pp': 5.0, 'ci95_pp': [0.0, 9.0]}))

    def test_shards_cover_every_board_once(self):
        jobs = probe.shards(21000, 400, 3)
        boards = [s + i for s, g in jobs for i in range(g // 2)]
        self.assertEqual(boards, list(range(21000, 21200)))
        with self.assertRaises(ValueError):
            probe.shards(43000, 400, 2)  # test seeds stay closed

    def test_smoke_arms_and_profile(self):
        ticks = queue.Queue()
        jobs = [('f', 21000, 2, 30), ('cutoff', 21000, 2, 30), ('profile', 15000, 2, 30)]
        parts = [probe._job(*job, ticks) for job in jobs]
        self.assertEqual([len(r['games']) for r in parts[:2]], [2, 2])
        self.assertLessEqual(parts[2]['move_kill_hits'], parts[2]['move_kill_overrides'])
        results, summary = probe.summarize(jobs, parts, 2)
        self.assertEqual(set(results), set(probe.ARMS))
        self.assertIn('ci95_pp', summary['paired_cutoff_vs_f']['active_win'])
        self.assertEqual(summary['profile']['games'], 2)


if __name__ == '__main__':
    unittest.main()
