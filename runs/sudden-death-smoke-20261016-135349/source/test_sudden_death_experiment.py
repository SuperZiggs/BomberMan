"""Ring mechanics, timed masks and unchanged production behavior."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch

import numpy as np

from bomberman_env import (BombermanFFAEnv, Bomb, RIGID_WALL, WOODEN_WALL,
                          ACTION_STOP, ACTION_DOWN, POWERUP_AMMO)
from rule_based_bot import _safe_action_mask
from sudden_death_experiment import SuddenDeathEnv, sudden_death_mask


def arena(**kwargs):
    env = SuddenDeathEnv(board_size=9, num_agents=2, wooden_wall_density=0,
                         powerup_prob=0, seed=17, **kwargs)
    env.reset()
    for agent, pos in zip(env.agents, ((3, 3), (5, 5))):
        agent.row, agent.col = pos
    return env


class SuddenDeathTests(unittest.TestCase):
    def test_off_is_exact_current_engine(self):
        original = BombermanFFAEnv(board_size=9, num_agents=2, seed=18)
        variant = SuddenDeathEnv(board_size=9, num_agents=2, seed=18)
        for left, right in zip(original.reset(), variant.reset()):
            np.testing.assert_array_equal(left, right)
        actions = np.random.default_rng(19)
        for _ in range(100):
            move = actions.integers(0, 6, 2).tolist()
            a, b = original.step(move), variant.step(move)
            self.assertEqual(a[1:], b[1:])
            for x, y in zip(a[0], b[0]):
                np.testing.assert_array_equal(x, y)
            self.assertEqual(original.rng.bit_generator.state, variant.rng.bit_generator.state)
            if a[3]['episode_done']:
                break

    def test_ring_schedule_matches_loc_update(self):
        path = Path(__file__).parent / '_comparison/BomberMan-Loc_update/bomberman_env.py'
        spec = importlib.util.spec_from_file_location('_sudden_reference_env', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        local = arena(sudden_death=True)
        reference = module.BombermanFFAEnv(board_size=9, num_agents=2, sudden_death=True)
        for tick in (0, 159, 199, 200, 219, 220, 239, 240, 500):
            local.step_count = reference.step_count = tick
            self.assertEqual(local._target_sudden_death_ring(), reference._target_sudden_death_ring())
        for ring in (1, 2, 3):
            self.assertEqual(local._ring_cells(ring), reference._ring_cells(ring))

    def test_closure_happens_before_move_and_has_no_bomb_credit(self):
        env = arena(sudden_death=True, sudden_death_start=2)
        env.agents[0].row, env.agents[0].col = 1, 3
        env.step_count = 1
        _, rewards, dones, info = env.step([ACTION_DOWN, ACTION_STOP])
        self.assertEqual(info['sudden_death_kills'], [0])
        self.assertFalse(env.agents[0].alive)
        self.assertEqual((env.agents[0].row, env.agents[0].col), (1, 3))
        self.assertEqual(rewards[0], -1)
        self.assertTrue(dones[0])
        self.assertEqual(info['kills'], [])
        self.assertEqual(env.agents[1].kills, 0)
        self.assertEqual(info['self_blast_deaths'], [])

    def test_removed_bomb_returns_capacity_and_all_flame_sources_are_cleared(self):
        env = arena(sudden_death=True, sudden_death_start=1)
        pos = (1, 3)
        env.bombs = [Bomb(*pos, 0, 1, 4)]
        env.agents[0].bombs_placed = 1
        env.flames[pos] = 2
        env._flame_owners[pos] = 0
        env._flame_sources[pos] = {0: 2}
        env.powerups[pos] = POWERUP_AMMO
        info = env.step([0, 0])[3]
        self.assertEqual(env.bombs, [])
        self.assertEqual(env.agents[0].bombs_placed, 0)
        self.assertEqual(info['kills'], [])
        self.assertEqual(env.grid[pos], RIGID_WALL)
        for mapping in (env.flames, env._flame_owners, env._flame_sources, env.powerups):
            self.assertNotIn(pos, mapping)

    def test_final_ring_closes_all_floor_on_9x9_and_reset_reopens(self):
        env = arena(sudden_death=True, sudden_death_start=1, sudden_death_interval=1)
        for _ in range(4):
            env.step([0, 0])
        self.assertEqual(env._sudden_death_ring, 3)
        self.assertEqual(env.grid[3, 4], RIGID_WALL)
        self.assertEqual(env.grid[4, 4], RIGID_WALL)
        self.assertFalse(np.any(env.grid == 0))
        self.assertFalse(any(a.alive for a in env.agents))
        env.reset()
        self.assertEqual(env._sudden_death_ring, 0)
        self.assertEqual(env.grid[1, 3], 0)
        self.assertEqual(env._get_obs(0).shape, (12, 9, 9))

    def test_forecast_matches_step_with_chain_wood_lingering_flame_and_ring(self):
        env = arena(sudden_death=True, sudden_death_start=3, sudden_death_interval=3)
        env.grid[3, 4] = WOODEN_WALL
        env.bombs = [Bomb(1, 3, 0, 5, 4), Bomb(3, 3, 0, 1, 3), Bomb(3, 5, 1, 6, 3)]
        env.agents[0].bombs_placed = 2
        env.agents[1].bombs_placed = 1
        env.flames[(5, 3)] = 2
        env._flame_owners[(5, 3)] = 0
        env._flame_sources[(5, 3)] = {0: 2}
        before = pickle.dumps(env.__dict__)
        burns, blocked, closing = env.forecast_closures(min_horizon=10)
        self.assertEqual(before, pickle.dumps(env.__dict__))
        actual = deepcopy(env)
        for tick in range(len(burns)):
            expected_blocked, expected_damage = None, None
            original_tick = actual._tick_bombs_and_explode
            def record():
                nonlocal expected_blocked, expected_damage
                expected_blocked = actual.grid != 0
                for bomb in actual.bombs:
                    expected_blocked[bomb.row, bomb.col] = True
                original_tick()
                expected_damage = np.zeros_like(actual.grid, dtype=bool)
                for pos in actual.flames:
                    expected_damage[pos] = True
                for ring in range(1, actual._sudden_death_ring + 1):
                    for pos in actual._ring_cells(ring):
                        expected_damage[pos] = True
            with patch.object(actual, '_tick_bombs_and_explode', side_effect=record):
                actual.step([0, 0])
            np.testing.assert_array_equal(blocked[tick], expected_blocked)
            np.testing.assert_array_equal(burns[tick], expected_damage)
        self.assertTrue(closing[2, 1, 3])
        self.assertFalse(burns[4, 1, 3] == 0)

    def test_joint_mask_leaves_ring_one_tick_before_closure(self):
        env = arena(sudden_death=True, sudden_death_start=2)
        env.agents[0].row, env.agents[0].col = 1, 3
        base = _safe_action_mask(env, 0)
        before = pickle.dumps(env.__dict__)
        mask = sudden_death_mask(env, 0, base)
        self.assertFalse(mask[ACTION_STOP])
        self.assertTrue(mask[ACTION_DOWN])
        self.assertTrue(all(not new or old for old, new in zip(base, mask)))
        self.assertEqual(before, pickle.dumps(env.__dict__))
        env.step([ACTION_DOWN, 0])
        env.step([0, 0])
        self.assertTrue(env.agents[0].alive)

    def test_unavoidable_source_closure_is_logged_and_no_empty_mask(self):
        env = arena(sudden_death=True, sudden_death_start=1)
        env.agents[0].row, env.agents[0].col = 1, 3
        base = _safe_action_mask(env, 0)
        stats = {}
        self.assertEqual(sudden_death_mask(env, 0, base, stats), base)
        self.assertEqual(stats['unavoidable_pre_move_closure'], 1)

    def test_disabled_and_distant_closure_do_not_change_masks(self):
        for enabled in (False, True):
            env = arena(sudden_death=enabled)
            base = _safe_action_mask(env, 0)
            self.assertEqual(sudden_death_mask(env, 0, base), base)
            if not enabled:
                a, b = env._forecast_blasts(min_horizon=10), env.forecast_closures(min_horizon=10)
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)

    def test_timing_validation(self):
        for kwargs in ({'sudden_death_start': 0}, {'sudden_death_interval': -1},
                       {'sudden_death_warning_steps': -1}, {'sudden_death_start': 1.5}):
            with self.assertRaises(ValueError):
                arena(**kwargs)


if __name__ == '__main__':
    unittest.main()
