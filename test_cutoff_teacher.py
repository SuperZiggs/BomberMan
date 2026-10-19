import pickle
import unittest

from bomberman_env import ACTION_STOP, ACTION_UP, RIGID_WALL, Bomb, BombermanFFAEnv
from cutoff_teacher import trap_actions
from kill_search import search_action
from rule_based_bot import _safe_action_mask


def corridor():
    """Learner bomb at (1,1) range 3 sweeps row 1; the opponent's only refuge is (2,3)."""
    env = BombermanFFAEnv(board_size=9, num_agents=2, wooden_wall_density=0, powerup_prob=0, seed=17)
    env.reset()
    env.agents[0].row, env.agents[0].col = 3, 3   # learner, one step below the refuge
    env.agents[1].row, env.agents[1].col = 1, 2   # opponent inside the blast
    env.grid[1, 5] = RIGID_WALL
    env.bombs.append(Bomb(1, 1, 0, 4, 3))
    env.agents[0].bombs_placed = 1
    return env


class TrapActionTests(unittest.TestCase):
    def test_finds_the_move_that_blocks_the_last_refuge(self):
        env = corridor()
        mask = _safe_action_mask(env, 0)
        before = pickle.dumps(env.__dict__)
        self.assertEqual(trap_actions(env, 0, mask), [ACTION_UP])
        self.assertEqual(before, pickle.dumps(env.__dict__))
        self.assertEqual(search_action(env, 0, mask, ACTION_STOP)[0], ACTION_UP)  # kill-search agrees

    def test_holding_the_doorway_is_the_trap_when_leaving_would_free_it(self):
        # Kill-search's usual killing "move" (RESULTS SD6.1): stay put; F runs and opens the exit.
        env = corridor()
        env.agents[0].row, env.agents[0].col = 2, 3
        self.assertEqual(trap_actions(env, 0, _safe_action_mask(env, 0)), [ACTION_STOP])

    def test_no_bombs_or_no_escape_left_needs_no_override(self):
        env = corridor()
        env.bombs.clear()
        self.assertEqual(trap_actions(env, 0, _safe_action_mask(env, 0)), [])
        env = corridor()
        env.grid[2, 3] = RIGID_WALL  # the opponent is already doomed wherever the learner goes
        self.assertEqual(trap_actions(env, 0, _safe_action_mask(env, 0)), [])

    def test_ignores_moves_the_mask_forbids(self):
        env = corridor()
        mask = _safe_action_mask(env, 0)
        mask[ACTION_UP] = False
        self.assertEqual(trap_actions(env, 0, mask), [])


if __name__ == '__main__':
    unittest.main()
