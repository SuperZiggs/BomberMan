import math
import unittest

import numpy as np

import run_sd8a_distill as sd8a
import run_sd8b_search3 as sd8b


def diff(value, lower=1.0):
    return {'difference_pp': value, 'ci95_pp': [lower, value + 1]}


class SD8ATests(unittest.TestCase):
    def test_teacher_scores_prefer_two_move_plans(self):
        info = {'skipped': False, 'scores': {0: 0.0, 3: 0.0, 5: -1.0}, 'plans': {0: 0.0, 3: 1.0}}
        scores = sd8a.teacher_scores(info)
        self.assertEqual(scores[3], 1.0)
        self.assertEqual(scores[5], -1.0)
        self.assertTrue(math.isnan(scores[1]))
        target = sd8a.sd53.masked_soft_target(scores, ~np.isnan(scores))
        self.assertEqual(target.tolist(), [0, 0, 0, 1, 0, 0])

    def test_select_respects_suicide_limit_and_ties(self):
        row = lambda active, suicide: {'active_win_rate': active, 'suicide_rate': suicide}
        summaries = {'G': row(48, 1.0), 'epoch_4': row(60, 1.0), 'epoch_8': row(60, 2.0),
                     'epoch_12': row(70, 2.5), 'epoch_16': row(55, 0.0)}
        self.assertEqual(sd8a.select(summaries), 4)

    def test_gate(self):
        self.assertTrue(sd8a.gate({'active_win': diff(5.0), 'suicide': diff(1.0)})['pass'])
        self.assertFalse(sd8a.gate({'active_win': diff(6.0, lower=-0.5), 'suicide': diff(0.0)})['pass'])
        self.assertFalse(sd8a.gate({'active_win': diff(6.0), 'suicide': diff(1.25)})['pass'])

    def test_config_and_shards_cover_every_board_once(self):
        config = sd8a.config_for('x', smoke=False)
        self.assertEqual(config['collect_games'], 3000)
        boards = [b for s, e in sd8a._shards(config) for b in range(s, e + 1)]
        self.assertEqual(boards, list(range(210000, 211500)))
        self.assertEqual(sd8a.TEST_SEED, 48000)


class SD8BTests(unittest.TestCase):
    def test_decide_threshold(self):
        self.assertTrue(sd8b.decide({'active_win': diff(2.0)})['open_test_49000'])
        self.assertFalse(sd8b.decide({'active_win': diff(1.75)})['open_test_49000'])
        self.assertEqual(sd8b.TEST_SEED, 49000)


if __name__ == '__main__':
    unittest.main()
