"""Timing, legality and supervision checks for the two-bomb teacher."""
import pickle
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch

from bomberman_env import Bomb, RIGID_WALL, WOODEN_WALL, ACTION_BOMB, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP, _MOVE_DELTA
from rule_based_bot import _reachable_cells
from tactical_teacher import get_two_bomb_plan, _learner_blockers
from train_actor_critic import train, get_action_mask
from test_combat import arena, put_bomb
from pilot_two_bomb import paired_active_bootstrap, complete_validation, final_evaluation
from pathlib import Path


def combo_arena(ammo=2):
    env = arena(2)
    env.grid[1:6, 1:6] = RIGID_WALL
    for pos in [(3, 3), (3, 4), (3, 5), (2, 3), (2, 4), (2, 5),
                (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (3, 2)]:
        env.grid[pos] = 0
    env.agents[0].row, env.agents[0].col = 3, 3
    env.agents[1].row, env.agents[1].col = 3, 5
    env.agents[0].ammo = ammo
    return env


class TacticalChecks(unittest.TestCase):
    def assert_forecast_matches_steps(self, env, schedule, actions):
        burns, blocked = env._forecast_blasts(scheduled_bombs=schedule)
        actual = deepcopy(env)
        observed_fire, observed_blockers = [], []
        original_tick = actual._tick_bombs_and_explode

        def tick():
            walls = actual.grid != 0
            for bomb in actual.bombs:
                walls[bomb.row, bomb.col] = True
            observed_blockers.append(walls.copy())
            original_tick()
            fire = np.zeros_like(walls)
            for pos in actual.flames:
                fire[pos] = True
            observed_fire.append(fire)

        with patch.object(actual, '_tick_bombs_and_explode', side_effect=tick):
            for t in range(len(burns)):
                actual.step([actions[t] if t < len(actions) else ACTION_STOP, ACTION_STOP])
        np.testing.assert_array_equal(burns, observed_fire)
        np.testing.assert_array_equal(blocked, observed_blockers)
        return actual

    def test_delayed_chain_and_wood_forecast_matches_real_steps(self):
        env = combo_arena()
        env.grid[4, 3] = WOODEN_WALL
        env.grid[5, 3] = 0
        env.agents[0].blast_range = 4
        env.flames[(5, 5)] = 2
        schedule = [(0, Bomb(3, 3, 0, 8, 4)), (2, Bomb(2, 3, 0, 8, 4))]
        burns, blocked, slots = env._forecast_blasts(scheduled_bombs=schedule, return_slots=True)
        self.assertFalse(blocked[0, 2, 3], 'The future bomb cannot block an earlier move')
        self.assertTrue(blocked[2, 2, 3])
        self.assertFalse(burns[6, 2, 5])
        self.assertTrue(burns[7, 2, 5], 'The second fuse is shortened by the first bomb')
        self.assertEqual(slots[2, 0], 0)
        self.assertEqual(slots[8, 0], 2)
        self.assertTrue(burns[8, 2, 5], 'Lingering fire lasts through the next tick')
        self.assertTrue(burns[7, 5, 3], 'The chained ray traverses the wood just destroyed')
        actual = self.assert_forecast_matches_steps(
            env, schedule, [ACTION_BOMB, ACTION_UP, ACTION_BOMB, ACTION_UP, ACTION_LEFT])
        self.assertEqual(actual.grid[4, 3], 0)
        self.assertTrue(actual.agents[0].alive)

    def test_ammo_one_waits_for_return_and_existing_chain_returns_capacity(self):
        env = combo_arena(1)
        first = (0, Bomb(3, 3, 0, 8, 2))
        with self.assertRaises(ValueError):
            env._forecast_blasts(scheduled_bombs=[first, (2, Bomb(2, 3, 0, 8, 2))])
        schedule = [first, (8, Bomb(1, 2, 0, 8, 2))]
        actions = [ACTION_BOMB, ACTION_UP, ACTION_UP, ACTION_LEFT,
                   ACTION_STOP, ACTION_STOP, ACTION_STOP, ACTION_STOP,
                   ACTION_BOMB, ACTION_LEFT, ACTION_DOWN]
        self.assert_forecast_matches_steps(env, schedule, actions)
        # An existing long fuse can release its slot earlier through a chain.
        env = combo_arena(1)
        put_bomb(env, 3, 3, 0, timer=8)
        put_bomb(env, 3, 4, 1, timer=1)
        _, _, slots = env._forecast_blasts(min_horizon=3, return_slots=True)
        self.assertEqual(slots[0, 0], 0)
        self.assertEqual(slots[1, 0], 1)
        env._forecast_blasts(scheduled_bombs=[(1, Bomb(2, 3, 0, 8, 2))])

    def test_two_single_bombs_leave_refuges_but_combo_traps_and_learner_survives(self):
        env = combo_arena()
        schedule = [(0, Bomb(3, 3, 0, 8, 2)), (2, Bomb(2, 3, 0, 8, 2))]
        for bomb in schedule:
            burns, blocked = env._forecast_blasts(scheduled_bombs=[bomb])
            self.assertTrue(_reachable_cells((3, 5), burns, blocked, len(burns)).any())
        burns, blocked = env._forecast_blasts(scheduled_bombs=schedule)
        self.assertFalse(_reachable_cells((3, 5), burns, blocked, len(burns)).any())
        mask = get_action_mask(env, 0)
        plan, stats = get_two_bomb_plan(env, 0, mask)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan['bombs']), 2)
        self.assertEqual(plan['pressure'], 1)
        self.assertLessEqual(stats['plans'], 64)
        self.assertTrue(all(mask[a] for a in plan['first_actions']))
        forecast = env._forecast_blasts(scheduled_bombs=plan['bombs'])
        pos = (env.agents[0].row, env.agents[0].col)
        for t, action in enumerate(plan['path']):
            dr, dc = _MOVE_DELTA.get(action, (0, 0))
            pos = pos[0] + dr, pos[1] + dc
            self.assertFalse(forecast[0][t][pos])
        suffix = len(plan['path'])
        self.assertTrue(_reachable_cells(pos, forecast[0][suffix:],
                        _learner_blockers(env, 0, forecast[1])[suffix:], len(forecast[0]) - suffix).any())

    def test_planner_preserves_state_rng_mask_and_capacity(self):
        for ammo in (1, 2):
            env = combo_arena(ammo)
            before = pickle.dumps(env.__dict__)
            numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
            mask = get_action_mask(env, 0)
            plan, stats = get_two_bomb_plan(env, 0, mask)
            self.assertEqual(pickle.dumps(env.__dict__), before)
            np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
            self.assertEqual(numpy_rng[2:], np.random.get_state()[2:])
            self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
            self.assertLessEqual(stats['plans'], 64)
            if plan:
                self.assertTrue(all(mask[a] for a in plan['first_actions']))
                env._forecast_blasts(scheduled_bombs=plan['bombs'])
                if len(plan['bombs']) == 2 and ammo == 1:
                    self.assertGreaterEqual(plan['bombs'][1][0] - plan['bombs'][0][0], env.bomb_timer)

    def test_collision_and_already_doomed_enemy_are_rejected(self):
        env = combo_arena()
        blocked = _learner_blockers(env, 0, env._forecast_blasts()[1])
        self.assertTrue(blocked[0, 3, 4], 'Both agents can target this cell on the first step')
        collision = deepcopy(env)
        collision.step([ACTION_RIGHT, ACTION_LEFT])
        self.assertEqual((collision.agents[0].row, collision.agents[0].col), (3, 3))
        self.assertFalse(get_action_mask(env, 0)[ACTION_RIGHT])
        # Closing the upper doorway leaves the enemy already trapped by a known bomb.
        env.grid[2, 4] = env.grid[2, 5] = RIGID_WALL
        put_bomb(env, 3, 3, 0, timer=4)
        env.agents[0].row, env.agents[0].col = 1, 1
        plan, stats = get_two_bomb_plan(env, 0, get_action_mask(env, 0))
        self.assertIsNone(plan)
        self.assertGreater(stats['rejected'].get('enemy_already_doomed', 0), 0)

    def test_equivalent_first_actions_are_kept_and_bit_reachability_matches_grids(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 5
        env.agents[0].ammo = 2
        plan, _ = get_two_bomb_plan(env, 0, [False, True, True, False, False, False])
        self.assertEqual(plan['first_actions'], [ACTION_UP, ACTION_DOWN])
        rng = np.random.default_rng(42)
        for size in (7, 9, 11):
            for ticks in (0, 1, 12):
                fire = rng.random((ticks, size, size)) < 0.08
                walls = rng.random((ticks, size, size)) < 0.3
                start = size // 2, size // 2
                reachable = np.zeros((size, size), dtype=bool)
                reachable[start] = True
                for t in range(ticks):
                    moves = np.zeros_like(reachable)
                    moves[1:] |= reachable[:-1]
                    moves[:-1] |= reachable[1:]
                    moves[:, 1:] |= reachable[:, :-1]
                    moves[:, :-1] |= reachable[:, 1:]
                    reachable = (reachable | (moves & ~walls[t])) & ~fire[t]
                np.testing.assert_array_equal(_reachable_cells(start, fire, walls, ticks), reachable)

    def test_combat_soft_target_replaces_old_label_without_forcing_action(self):
        captured, labels, actions = [], [], []
        plan = {'first_actions': [ACTION_UP, ACTION_RIGHT], 'bombs': (), 'pressure': 0.5}
        diagnostics = {'plans': 3, 'accepted': 1, 'two_bomb_labels': 1,
                       'pressure': 0.5, 'milliseconds': 2.0, 'rejected': {}}
        def teacher(env, aid, mask):
            captured.append(mask)
            return plan, deepcopy(diagnostics)
        def logits(value, dim):
            labels.append(value.detach().clone())
            return torch.log_softmax(value, dim=dim)
        def step(env, ep, aid, selected, opponents):
            if env.step_count:
                actions.append(selected[aid])
        with patch('train_actor_critic.get_two_bomb_plan', side_effect=teacher), \
             patch('train_actor_critic.get_attack_action') as local, \
             patch('train_actor_critic.get_action_mask', return_value=[True] * 6), \
             patch('train_actor_critic.F.log_softmax', side_effect=logits), \
             patch('torch.distributions.Categorical.sample', return_value=torch.tensor([ACTION_STOP])), \
             patch('train_actor_critic.torch.save'), patch('train_actor_critic.os.makedirs'), patch('builtins.print'):
            metrics = []
            train(episodes=1, board_size=7, max_steps=1, device='cpu', run_report=False,
                  attack_teacher='two_bomb', on_episode=metrics.append, on_step=step)
        local.assert_not_called()
        self.assertEqual(len(labels), 1)
        self.assertGreater(metrics[0]['combat_loss'], 0)
        self.assertEqual(metrics[0]['navigation_loss'], 0)
        self.assertEqual(metrics[0]['teacher']['accepted'], 1)
        expected = -torch.log_softmax(labels[0], dim=-1)[0, [ACTION_UP, ACTION_RIGHT]].mean()
        self.assertAlmostEqual(metrics[0]['combat_loss'], expected.item(), places=6)
        self.assertEqual(actions, [ACTION_STOP], 'The sampled CNN action must not be replaced by the teacher')

    def test_bootstrap_resamples_board_pairs_and_requires_matching_complete_pairs(self):
        candidate = {'games': [{'seed': s, 'aid': aid, 'active_win': int(aid == 0)}
                               for s in range(10) for aid in (0, 1)]}
        reference = {'games': [{**g, 'active_win': 1 - g['active_win']} for g in candidate['games']]}
        result = paired_active_bootstrap(candidate, reference)
        self.assertEqual(result['difference_pp'], 0)
        self.assertEqual(result['ci95_pp'], [0, 0], 'Do not independently resample A0 and A1')
        with self.assertRaises(ValueError):
            paired_active_bootstrap({'games': candidate['games'][:-1]}, reference)
        with self.assertRaises(ValueError):
            paired_active_bootstrap({'games': candidate['games'][2:]}, reference)

    def test_deadline_during_last_validation_cannot_count_as_complete_pilot(self):
        rows = [{'episode': ep, 'num_episodes': 100, 'seed': 21000} for ep in (0, 100, 200)]
        self.assertTrue(complete_validation(rows))
        self.assertFalse(complete_validation(rows[:-1]))
        self.assertFalse(complete_validation(rows[:1] * 3))
        rows[-1]['num_episodes'] = 50
        self.assertFalse(complete_validation(rows))

    def test_final_evaluation_does_not_publish_a_cancelled_partial_score(self):
        result = {'win_rate': 10, 'active_win_rate': 8, 'by_corner': {
            str(aid): {'win_rate': 10, 'active_win_rate': 8} for aid in (0, 1)}}
        with patch('pilot_two_bomb.make_ac_fn', return_value=object()), \
             patch('pilot_two_bomb.evaluate_seeded', side_effect=[result, None]) as evaluate, \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.read_bytes', return_value=b'weights'), \
             patch('pilot_two_bomb.save_json') as save:
            results = final_evaluation(Path('unused'), Path('original.pt'), seconds=5)
        self.assertEqual(list(results), ['original'])
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(evaluate.call_args.args[3:6], (400, 25000, 500))
        self.assertEqual(save.call_count, 1)


if __name__ == '__main__':
    unittest.main()
