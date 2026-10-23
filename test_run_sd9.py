import math
import unittest

import numpy as np

import run_sd9_distill as sd9


def diff(value, lower=1.0):
    return {'difference_pp': value, 'ci95_pp': [lower, value + 1]}


class SD9Tests(unittest.TestCase):
    def test_teacher_scores_prefer_three_move_plans(self):
        info = {'skipped': False, 'scores': {0: 0.0, 3: 0.0, 4: 0.0, 5: -1.0}, 'plans': {0: 0.0, 3: 0.0, 4: 0.0},
                'deep_plans': {0: 0.0, 3: 0.0, 4: 1.0}}
        scores = sd9.teacher_scores(info)
        self.assertEqual(scores[4], 1.0)
        self.assertEqual(scores[5], -1.0)
        self.assertTrue(math.isnan(scores[1]))
        target = sd9.sd53.masked_soft_target(scores, ~np.isnan(scores))
        self.assertEqual(target.tolist(), [0, 0, 0, 0, 1, 0])

    def test_select_compares_to_s2(self):
        row = lambda active, suicide: {'active_win_rate': active, 'suicide_rate': suicide}
        summaries = {'S2': row(83, 0.0), 'epoch_4': row(85, 1.0), 'epoch_8': row(90, 1.5), 'epoch_16': row(88, 0.5)}
        self.assertEqual(sd9.select(summaries), 16)

    def test_gate_is_against_s2(self):
        self.assertTrue(sd9.gate({'active_win': diff(5.0), 'suicide': diff(1.0)})['pass'])
        self.assertFalse(sd9.gate({'active_win': diff(4.75), 'suicide': diff(0.0)})['pass'])
        self.assertFalse(sd9.gate({'active_win': diff(6.0, lower=-0.5), 'suicide': diff(0.0)})['pass'])
        self.assertIn('active_vs_S2_5pp', sd9.gate({'active_win': diff(6.0), 'suicide': diff(0.0)}))

    def test_new_boards_and_seeds(self):
        config = sd9.config_for('x', smoke=False)
        self.assertEqual(config['collect_games'], 3000)
        boards = [b for s, e in sd9._shards(config) for b in range(s, e + 1)]
        self.assertEqual(boards, list(range(212000, 213500)))
        self.assertEqual(sd9.TEST_SEED, 57000)
        self.assertEqual(config['s2_sha256'], sd9.S2_SHA256)


if __name__ == '__main__':
    unittest.main()
