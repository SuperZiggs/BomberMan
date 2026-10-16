"""Reference certification, split isolation, mask fidelity and CNN-only play."""
from copy import deepcopy
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmark_t3 import (scenarios, make_env, validate_env, reference_plan, genuine_combo,
                          certify, play_case, summarize_games, transform_action, FAMILIES)
from bomberman_env import _MOVE_DELTA, Bomb
from train_actor_critic import get_action_mask


class ScenarioChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_symmetries_roles_have_legal_states_and_same_split_per_family(self):
        cases = scenarios(tuple(FAMILIES.values()))
        self.assertEqual(len(cases), 336)
        self.assertEqual(len({c['id'] for c in cases}), 336)
        for c in cases:
            self.assertEqual(c['split'], FAMILIES[c['family']])
            env = make_env(c)
            validate_env(env)
            self.assertEqual(len(get_action_mask(env, c['aid'])), 6)
        self.assertFalse(any(c['split'] == 'test' for c in scenarios()))
        for mirror in (False, True):
            for rot in range(4):
                self.assertEqual({transform_action(a, rot, mirror) for a in _MOVE_DELTA}, set(_MOVE_DELTA))

    def test_combo_reference_really_requires_both_bombs_and_full_mask_escape(self):
        for c in scenarios(tuple(FAMILIES.values())):
            if c['expected'] != 'combo':
                continue
            env = make_env(c)
            before = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
            plan, quality = reference_plan(env, c)
            self.assertTrue(genuine_combo(quality), c['id'])
            proof = certify(env, c['aid'], plan)
            self.assertTrue(proof['full_mask_pass'], (c['id'], proof))
            clone = deepcopy(env)
            for action in proof['action_tape']:
                self.assertTrue(get_action_mask(clone, c['aid'])[action])
                actions = [0, 0]
                actions[c['aid']] = action
                clone.step(actions)
            self.assertTrue(clone.agents[c['aid']].alive)
            self.assertFalse(clone.bombs or clone.flames, 'Certificate must reach the end of hazards')
            self.assertEqual(pickle.dumps(env.__dict__), before[0])
            self.assertEqual(pickle.dumps(np.random.get_state()), before[1])
            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_ammo_one_illegal_overlap_is_rejected_and_delayed_return_is_legal(self):
        for c in scenarios(smoke=True):
            env = make_env(c)
            if c['expected'] == 'illegal_capacity':
                with self.assertRaises(ValueError):
                    reference_plan(env, c)
            if c['expected'] == 'delayed_capacity':
                plan, _ = reference_plan(env, c)
                self.assertEqual([d for d, _ in plan['bombs']], [0, 8])
                proof = certify(env, c['aid'], plan)
                self.assertTrue(proof['full_mask_pass'], proof)

    def test_single_bomb_and_approach_positive_controls_have_mask_valid_solutions(self):
        for c in scenarios(tuple(FAMILIES.values())):
            if c['expected'] not in ('single_trap', 'static_kill'):
                continue
            env = make_env(c)
            plan, quality = reference_plan(env, c)
            if c['expected'] == 'single_trap':
                self.assertEqual(quality['survivors']['combo'], [0], c['id'])
            self.assertGreater(quality['survivors']['baseline'][0], 0)
            proof = certify(env, c['aid'], plan)
            self.assertTrue(proof['full_mask_pass'], c['id'])
            self.assertEqual(proof['owned_kills'], 1, c['id'])

    def test_certificate_rejects_a_future_mask_block_and_invalid_state(self):
        c = next(c for c in scenarios(smoke=True) if c['expected'] == 'combo')
        env = make_env(c)
        plan, _ = reference_plan(env, c)
        real_mask = get_action_mask
        def mask(e, aid):
            result = real_mask(e, aid)
            if e.step_count == 2:
                result[5] = False
            return result
        with patch('benchmark_t3.get_action_mask', side_effect=mask):
            proof = certify(env, c['aid'], plan)
        self.assertEqual(proof['status'], 'mask_blocked')
        self.assertEqual(proof['blocked_delay'], 2)
        env.agents[0].row, env.agents[0].col = env.agents[1].row, env.agents[1].col
        with self.assertRaises(ValueError):
            validate_env(env)

    def test_death_in_prefix_is_known_failure_not_unknown_survival(self):
        c = next(c for c in scenarios(smoke=True) if c['expected'] == 'combo')
        env = make_env(c)
        a = env.agents[c['aid']]
        env.bombs.append(Bomb(a.row, a.col, c['aid'], 1, a.blast_range))
        a.bombs_placed = 1
        proof = certify(env, c['aid'], {'path': [0], 'bombs': [], 'horizon': 1})
        self.assertEqual(proof['status'], 'learner_died')
        self.assertFalse(proof['full_horizon_alive'])
        self.assertFalse(proof['full_mask_pass'])

    def test_play_uses_only_policy_mask_and_distinguishes_truncation_from_draw(self):
        c = next(c for c in scenarios(smoke=True) if c['group'] == 'escape')
        def safe(e, obs, aid):
            self.assertEqual(obs[aid].shape, (13, 9, 9))
            return next(a for a, allowed in enumerate(get_action_mask(e, aid)) if allowed)
        with patch('benchmark_t3.collect_candidates', side_effect=AssertionError('Teacher invoked during CNN test')):
            game = play_case(c, safe, 'test', 'static', 123, 10)
        self.assertTrue(game['skill_horizon_truncated'])
        self.assertFalse(game['actual_draw'])
        self.assertFalse(game['owned_kill_success'])
        self.assertEqual(summarize_games([game])['games'], 1)
        self.assertEqual(summarize_games([])['owned_kill_success']['percent'], None)

    def test_played_pair_quality_identifies_reference_combo_without_overriding_policy(self):
        c = next(c for c in scenarios(smoke=True) if c['expected'] == 'combo')
        env = make_env(c)
        plan, _ = reference_plan(env, c)
        tape = certify(env, c['aid'], plan)['action_tape']
        def scripted(e, obs, aid):
            return tape[e.step_count] if e.step_count < len(tape) else 0
        game = play_case(c, scripted, 'reference_test_only', 'static', 123, 20)
        self.assertTrue(game['owned_kill_success'])
        self.assertTrue(game['has_genuine_combo_pair'])
        self.assertEqual([(p['first_tick'], p['second_tick']) for p in game['pairs']], [(0, 2)])


if __name__ == '__main__':
    unittest.main()
