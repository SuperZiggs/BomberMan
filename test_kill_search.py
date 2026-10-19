import pickle
import unittest
from unittest.mock import patch

import numpy as np

from bomberman_env import ACTION_BOMB, ACTION_LEFT, ACTION_STOP, RIGID_WALL, WOODEN_WALL
from kill_search import search_action, search_action2, search_action3
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

    def test_two_move_search_finds_step_then_bomb(self):
        env = trapped_opponent()
        env.agents[0].row, env.agents[0].col = 1, 4   # one step too far for a bomb now
        mask = _safe_action_mask(env, 0)
        before, rng = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state())
        self.assertEqual(max(search_action(env, 0, mask, ACTION_STOP)[1]['scores'].values()), 0.0)
        action, info = search_action2(env, 0, mask, ACTION_STOP)
        self.assertEqual(info['plans'][action], 1.0)
        self.assertNotEqual(action, ACTION_STOP)
        self.assertEqual(before, pickle.dumps(env.__dict__))
        self.assertEqual(rng, pickle.dumps(np.random.get_state()))
        # With a one-move kill available, it returns exactly the one-move answer.
        env.agents[0].col = 3
        self.assertEqual(search_action2(env, 0, mask, ACTION_STOP)[0], search_action(env, 0, mask, ACTION_STOP)[0])

    def test_three_move_search_finds_step_step_bomb(self):
        import kill_search
        real = kill_search._rollout

        def rollout(env, aid, action, horizon):  # the only kill line: a bomb dropped at (1, 3)
            at = (env.agents[aid].row, env.agents[aid].col)
            return 1.0 if action == ACTION_BOMB and at == (1, 3) else min(real(env, aid, action, horizon), 0.0)
        env = trapped_opponent()
        env.agents[0].col = 5
        env.agents[1].row, env.agents[1].col = 5, 5   # within reach, too far to block row 1
        env.grid[3, 3] = WOODEN_WALL   # a bomb target only from (1, 3), so the mask allows BOMB there
        mask = _safe_action_mask(env, 0)
        before, rng = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state())
        with patch.object(kill_search, '_rollout', rollout):
            self.assertEqual(search_action2(env, 0, mask, ACTION_STOP)[0], ACTION_STOP)
            action, info = search_action3(env, 0, mask, ACTION_STOP)
            self.assertEqual(action, ACTION_LEFT)
            self.assertEqual(info['deep_plans'][ACTION_LEFT], 1.0)
            self.assertEqual(search_action3(env, 0, mask, ACTION_LEFT)[0], ACTION_LEFT)  # CNN move kept
        self.assertEqual(before, pickle.dumps(env.__dict__))
        self.assertEqual(rng, pickle.dumps(np.random.get_state()))


if __name__ == '__main__':
    unittest.main()
