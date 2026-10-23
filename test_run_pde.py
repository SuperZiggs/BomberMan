import unittest

import evaluate
import run_pde_check as pde


def rows(values, corners):
    return [{'seed': b, 'aid': c, 'value': v} for (b, c), v in zip([(b, c) for b in range(len(values) // corners)
                                                                   for c in range(corners)], values)]


class PDETests(unittest.TestCase):
    def test_bootstrap_keeps_all_corners_of_a_board(self):
        out = pde.paired_bootstrap(rows([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], 4), rows([0] * 12, 4))
        self.assertAlmostEqual(out['difference_pp'], 100 * 4 / 12)
        self.assertEqual(out['boards'], 3)
        with self.assertRaises(ValueError):
            pde.paired_bootstrap(rows([1] * 12, 4), rows([1] * 9, 3))

    def test_arms_and_comparisons_cover_registered_design(self):
        names = {pde.arm_name(s, l, o) for s, l, o, _ in pde.ARMS}
        self.assertEqual(len(names), len(pde.ARMS))
        for s, a, b in pde.COMPARE:
            if s != 'D':
                self.assertIn(f'{s}_{a}', names); self.assertIn(f'{s}_{b}', names)
        self.assertEqual(pde.SEED, 50000)
        for s, _, _, n in pde.ARMS:
            self.assertGreater(n, 0)

    def test_opponent_fn_default_is_rulebot_and_custom_seats_are_used(self):
        seen = []

        def stop(env, obs, aid):
            return 0

        def opponent(i):
            def fn(env, obs, aid):
                seen.append(aid); return 0
            return fn
        base = evaluate.evaluate_seeded(stop, 9, 2, 2, 21000, 6)
        again = evaluate.evaluate_seeded(stop, 9, 2, 2, 21000, 6, opponent_fn=evaluate.make_rule_based_fn)
        self.assertEqual(base['games'], again['games'])
        evaluate.evaluate_seeded(stop, 9, 3, 3, 21000, 2, opponent_fn=opponent)
        self.assertEqual(sorted(set(seen)), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
