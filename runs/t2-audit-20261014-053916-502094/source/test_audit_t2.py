"""Audit parity, RNG isolation and diagnostic denominators."""
from copy import deepcopy
import pickle
import unittest

import numpy as np
import torch

from audit_t2 import (ROOT, PlanTracker, check_plan, escape_suffix, load_net, observed_teacher,
                      opponent_action, rate, reachable_comparison, replay_game, summarize)
from bomberman_env import ACTION_BOMB, ACTION_LEFT, ACTION_STOP, ACTION_UP, Bomb
from evaluate import evaluate_seeded
from rule_based_bot import RuleBasedAgent
from tactical_teacher import get_two_bomb_plan
from test_combat import arena, put_bomb
from test_tactical_teacher import combo_arena
from train_actor_critic import get_action_mask, SnapshotOpponent


class AuditChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.net = load_net(ROOT / 'runs/pressure-C-next/final_phaseC.pt')

    def test_instrumented_teacher_matches_original_including_equivalent_paths(self):
        cases = [combo_arena(1), combo_arena(2), arena(2)]
        cases[-1].agents[0].row, cases[-1].agents[0].col = 3, 3
        cases[-1].agents[1].row, cases[-1].agents[1].col = 3, 5
        cases[-1].agents[0].ammo = 2
        masks = [get_action_mask(e, 0) for e in cases[:2]] + [[False, True, True, False, False, False]]
        for env, mask in zip(cases, masks):
            state = pickle.dumps(env.__dict__)
            np_state, torch_state = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
            expected, expected_stats = get_two_bomb_plan(env, 0, mask)
            actual, stats, tied = observed_teacher(env, 0, mask)
            self.assertEqual(actual, expected)
            self.assertEqual({k: v for k, v in stats.items() if k != 'milliseconds'},
                             {k: v for k, v in expected_stats.items() if k != 'milliseconds'})
            self.assertEqual({p['path'][0] for p in tied}, set(actual['first_actions']))
            self.assertEqual(pickle.dumps(env.__dict__), state)
            self.assertEqual(pickle.dumps(np.random.get_state()), np_state)
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_doomed_enemy_has_no_label_or_tied_plan(self):
        env = combo_arena()
        env.grid[2, 4] = env.grid[2, 5] = 1
        put_bomb(env, 3, 3, 0, timer=4)
        env.agents[0].row, env.agents[0].col = 1, 1
        plan, _, tied = observed_teacher(env, 0, get_action_mask(env, 0))
        self.assertIsNone(plan)
        self.assertEqual(tied, [])

    def test_clone_suffix_survives_combo_and_does_not_change_state_or_rng(self):
        for ammo in (1, 2):
            env = combo_arena(ammo)
            _, _, tied = observed_teacher(env, 0, get_action_mask(env, 0))
            before = pickle.dumps(env.__dict__)
            np_state, torch_state = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
            for kind in ('static', 'evader', 'rule'):
                result = check_plan(env, 0, tied[0], kind, True)
                self.assertEqual(result['prefix_status'], 'complete')
                self.assertTrue(result['suffix_found'])
                self.assertTrue(result['full_horizon_alive'])
            self.assertEqual(pickle.dumps(env.__dict__), before)
            self.assertEqual(pickle.dumps(np.random.get_state()), np_state)
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_physical_prefix_can_pass_while_real_mask_blocks_it(self):
        env = combo_arena()
        env.agents[0].row, env.agents[0].col = 1, 1
        plan = {'path': (ACTION_BOMB,), 'bombs': ((0, Bomb(1, 1, 0, 8, 2)),), 'horizon': 9}
        physical = check_plan(env, 0, plan, 'static', False)
        masked = check_plan(env, 0, plan, 'static', True)
        self.assertEqual(physical['prefix_status'], 'complete')
        self.assertTrue(physical['full_horizon_alive'])
        self.assertEqual(masked['prefix_status'], 'mask_blocked')
        self.assertEqual(masked['blocked_tick'], 0)
        self.assertIsNone(masked['full_horizon_alive'])

    def test_evader_never_places_a_bomb(self):
        env = combo_arena()
        env.bombs.append(Bomb(3, 3, 0, 4, 2))
        bot = RuleBasedAgent(1)
        for _ in range(30):
            self.assertNotEqual(opponent_action(env, 1, 'evader', bot), ACTION_BOMB)

    def test_common_horizon_combo_traps_while_each_single_leaves_refuges(self):
        env = combo_arena()
        plan, _ = get_two_bomb_plan(env, 0, get_action_mask(env, 0))
        before = pickle.dumps(env.__dict__)
        quality = reachable_comparison(env, 0, plan)['survivors']
        self.assertEqual(quality['combo'], [0])
        self.assertGreater(quality['baseline'][0], 0)
        self.assertGreater(quality['first_only'][0], 0)
        self.assertGreater(quality['second_only'][0], 0)
        self.assertEqual(pickle.dumps(env.__dict__), before)

    def test_suffix_rejects_unavoidable_fire_and_zero_ticks_is_empty(self):
        env = combo_arena()
        self.assertEqual(escape_suffix(env, 0, 0), [])
        env.flames[(3, 3)] = 2
        for r, c in ((2, 3), (4, 3), (3, 2), (3, 4)):
            env.flames[(r, c)] = 2
        self.assertIsNone(escape_suffix(env, 0, 1))

    def test_tracker_keeps_failure_and_early_terminal_denominators(self):
        env = combo_arena()
        plan, _ = get_two_bomb_plan(env, 0, get_action_mask(env, 0))
        tracker = PlanTracker(plan, 0, 'case', env, 0)
        before = (3, 3)
        _, _, _, info = env.step([ACTION_STOP, ACTION_STOP])
        self.assertIsNone(tracker.advance(0, ACTION_STOP, before, env, info))
        env.agents[0].alive = False
        info.update(episode_done=True, alive=[False, True])
        row = tracker.advance(1, ACTION_UP, before, env, info)
        self.assertEqual(row['status'], 'failure')
        self.assertFalse(row['strict_prefix'])
        self.assertFalse(row['first_bomb'])
        self.assertFalse(row['full_horizon_observed'])

        env = combo_arena()
        tracker = PlanTracker(plan, 0, 'early', env, 0)
        _, _, _, info = env.step([ACTION_BOMB, ACTION_STOP])
        env.agents[1].alive = False
        info.update(episode_done=True, alive=[True, False])
        row = tracker.advance(0, ACTION_BOMB, (3, 3), env, info)
        self.assertEqual(row['status'], 'early_terminal')
        self.assertTrue(row['first_bomb'])
        summary = summarize([], [], [row], [])
        self.assertEqual(summary['second_bomb_given_first'], rate(0, 1))
        self.assertEqual(summary['two_bomb_complete_placements'], rate(0, 1))

    def test_churn_uses_absolute_future_ticks_and_all_tied_schedules(self):
        env = combo_arena()
        plan, _ = get_two_bomb_plan(env, 0, get_action_mask(env, 0))
        tracker = PlanTracker(plan, 10, 'case', env, 0)
        # At tick 11 the first bomb is history, not a missing part of the next forecast.
        future = {**plan, 'bombs': tuple((d - 1, deepcopy(b)) for d, b in plan['bombs'] if d >= 1)}
        tracker.compare(11, [future], [])
        self.assertEqual(tracker.row['churn_transitions'], 0)
        tracker.compare(11, [], ['enemy_bomb'])
        self.assertEqual(tracker.row['churn_transitions'], 1)
        self.assertEqual(tracker.row['churn_reasons']['enemy_bomb'], 1)

    def test_trace_and_clone_checks_match_unobserved_evaluator_both_corners(self):
        for aid in (0, 1):
            traced = replay_game(self.net, 28000, aid, 50, check_every=10, max_checks=2)
            np_state, torch_state = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
            plain = replay_game(self.net, 28000, aid, 50, traced=False)
            self.assertEqual(traced[:2], plain[:2])
            self.assertEqual(pickle.dumps(np.random.get_state()), np_state)
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            self.assertTrue(traced[4])
            self.assertTrue(all(r['mask'][a] for r in traced[2] for a in r['first_actions']))
        policy = SnapshotOpponent(self.net, 'cpu')
        evaluated = evaluate_seeded(lambda e, o, a: policy.select_action(o[a], e, a),
                                    num_episodes=2, seed=28000, max_steps=50)
        for g in evaluated['games']:
            self.assertEqual(g, replay_game(self.net, g['seed'], g['aid'], 50, traced=False)[0])

    def test_empty_metrics_are_unknown_instead_of_zero_success(self):
        summary = summarize([], [], [], [])
        self.assertIsNone(summary['agreement']['percent'])
        self.assertIsNone(summary['second_bomb_given_first']['percent'])
        self.assertIsNone(summary['clone_checks']['rule_masked']['survival_observed']['percent'])


if __name__ == '__main__':
    unittest.main()
