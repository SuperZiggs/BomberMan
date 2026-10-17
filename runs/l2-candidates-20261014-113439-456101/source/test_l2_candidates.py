"""Candidate capture parity, feasible alternatives, tied labels and budget coverage."""
import pickle
from time import perf_counter
import unittest

import numpy as np
import torch

from audit_l2_candidates import collect_candidates, inspect_state, selection, summarize_states, viable
from tactical_teacher import get_two_bomb_plan
from test_combat import arena
from test_tactical_teacher import combo_arena
from train_actor_critic import get_action_mask


def open_combat():
    env = arena(2)
    env.agents[0].row, env.agents[0].col = 3, 3
    env.agents[1].row, env.agents[1].col = 3, 5
    env.agents[0].ammo = 2
    return env


class CandidateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_collects_lower_ranked_paths_without_changing_original_teacher(self):
        for env in (combo_arena(1), combo_arena(2), open_combat()):
            state, np_rng, torch_rng = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
            mask = get_action_mask(env, 0)
            expected, old_stats = get_two_bomb_plan(env, 0, mask)
            original, stats, candidates = collect_candidates(env, 0, mask)
            self.assertEqual(original, expected)
            self.assertEqual({k: v for k, v in stats.items() if k != 'milliseconds'},
                             {k: v for k, v in old_stats.items() if k != 'milliseconds'})
            self.assertTrue(any(c['score'] < candidates[0]['score'] for c in candidates))
            self.assertLessEqual(stats['plans'], 64)
            self.assertEqual(pickle.dumps(env.__dict__), state)
            self.assertEqual(pickle.dumps(np.random.get_state()), np_rng)
            self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))

    def test_real_future_mask_rejects_old_best_and_allows_alternative(self):
        r = inspect_state(open_combat(), 0, 0, perf_counter() + 30)
        self.assertTrue(r['complete'])
        self.assertFalse(r['old_representative_viable'])
        self.assertFalse(r['old_best_any_tied_viable'])
        self.assertIsNotNone(r['best_viable'])
        selected = r['candidates'][r['best_viable']['candidate_index']]
        self.assertTrue(viable(selected['check']))
        self.assertLess(selected['score'], r['candidates'][0]['score'])
        self.assertFalse(r['best_viable']['two_bomb'])
        summary = summarize_states([r])
        self.assertEqual(summary['rescued_original_invalid']['numerator'], 1)
        self.assertEqual(summary['rescued_original_invalid']['denominator'], 1)

    def test_genuine_combo_that_passes_mask_is_retained(self):
        env = combo_arena(2)
        before = pickle.dumps(env.__dict__)
        r = inspect_state(env, 0, 0, perf_counter() + 30)
        self.assertTrue(r['old_representative_viable'])
        self.assertTrue(r['best_viable']['two_bomb'])
        self.assertEqual(r['best_viable']['trapped'], 1)
        self.assertTrue(r['genuine_combo_candidate_indices'])
        self.assertTrue(r['top_k']['1']['same_best_score'])
        self.assertTrue(r['top_k']['1']['same_first_actions'])
        self.assertEqual(pickle.dumps(env.__dict__), before)

    def test_tied_selection_keeps_all_equivalent_first_actions(self):
        env = open_combat()
        original, _, candidates = collect_candidates(env, 0, [False, True, True, False, False, False])
        for i, c in enumerate(candidates):
            c['candidate_index'] = i
        result = selection(candidates)
        self.assertEqual(result['first_actions'], original['first_actions'])
        self.assertEqual(result['first_actions'], [1, 2])
        self.assertGreater(result['tied_paths'], 1)
        self.assertIsNone(selection([]))

    def test_deadline_marks_partial_states_and_excludes_them_from_denominators(self):
        r = inspect_state(combo_arena(), 0, 0, perf_counter() - 1)
        self.assertFalse(r['complete'])
        self.assertEqual(r['checked_paths'], 0)
        self.assertIsNone(r['best_viable'])
        summary = summarize_states([r])
        self.assertEqual(summary['fully_checked_states'], 0)
        self.assertIsNone(summary['any_viable_candidate']['percent'])
        self.assertIsNone(summary['top_k']['4']['best_score_recall']['percent'])

    def test_stress_policies_are_reproducible_and_preserve_numpy_torch_rng(self):
        np_before, torch_before = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
        a = inspect_state(combo_arena(), 0, 0, perf_counter() + 30, stress_seed=123)
        b = inspect_state(combo_arena(), 0, 0, perf_counter() + 30, stress_seed=123)
        for kind in ('evader', 'rule'):
            self.assertEqual({k: v for k, v in a['selected_stress_checks'][kind].items() if k != 'elapsed_ms'},
                             {k: v for k, v in b['selected_stress_checks'][kind].items() if k != 'elapsed_ms'})
        self.assertEqual(pickle.dumps(np.random.get_state()), np_before)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
