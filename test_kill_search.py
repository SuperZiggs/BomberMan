import pickle
import unittest

import numpy as np

from bomberman_env import ACTION_BOMB, ACTION_STOP, RIGID_WALL
from kill_search import search_action
from rule_based_bot import _safe_action_mask
from sudden_death_experiment import SuddenDeathEnv


def trapped_opponent():
    env = SuddenDeathEnv(board_size=9, num_agents=2, wooden_wall_density=0, powerup_prob=0, seed=17)
    env.reset()
    env.agents[0].row, env.agents[0].col = 1, 3   # learner
    env.agents[1].row, env.agents[1].col = 1, 1   # opponent in a dead end
    env.grid[2, 1] = RIGID_WALL
    return env


class KillSearchTests(unittest.TestCase):
    def test_finds_trapping_bomb_and_leaves_state_untouched(self):
        env = trapped_opponent()
        mask = _safe_action_mask(env, 0)
        before, rng = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state())
        action, info = search_action(env, 0, mask, ACTION_STOP)
        self.assertEqual(action, ACTION_BOMB)
        self.assertGreater(info['scores'][ACTION_BOMB], info['scores'][ACTION_STOP])
        self.assertEqual(before, pickle.dumps(env.__dict__))
        self.assertEqual(rng, pickle.dumps(np.random.get_state()))

    def test_keeps_default_when_nothing_is_better(self):
        env = trapped_opponent()
        env.agents[1].row, env.agents[1].col = 7, 7   # far away: no kill line
        mask = _safe_action_mask(env, 0)
        action, info = search_action(env, 0, mask, ACTION_STOP)
        self.assertEqual(action, ACTION_STOP)
        self.assertTrue(info['skipped'])


if __name__ == '__main__':
    unittest.main()
