"""Label dispatch, replay integrity and delayed counter-bomb certificates."""
from copy import deepcopy
import gzip
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

import audit_t3_approach as audit
from benchmark_t3 import make_env, play_case, scenarios
from bomberman_env import Bomb
from counterbomb_fixtures import (certify_fixtures, make_fixture, masked_surviving_path,
                                  parity_replay, surviving_path)


class ApproachChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_local_dispatch_preserves_loss_groups_weights_and_missing(self):
        case = next(c for c in scenarios(('validation',)) if c['group'] == 'approach')
        env, mask = make_env(case), [True] * 6
        zero, danger = np.zeros((9, 9)), np.ones((9, 9))
        with patch.object(audit, '_build_danger_map', return_value=zero):
            with patch.object(audit, 'get_attack_action', return_value=4):
                self.assertEqual(audit.local_label(env, 0, mask), (4, 'navigation', 0.2))
            with patch.object(audit, 'get_attack_action', return_value=5), \
                 patch.object(audit, '_bomb_escape_pressure', return_value=0.2):
                self.assertEqual(audit.local_label(env, 0, mask), (5, 'combat', 0.5))
            with patch.object(audit, 'get_attack_action', return_value=None):
                self.assertEqual(audit.local_label(env, 0, mask), (None, 'missing', 0))
        with patch.object(audit, '_build_danger_map', return_value=danger):
            self.assertEqual(audit.local_label(env, 0, mask), (5, 'combat', 0.5))
            mask[5] = False
            with patch.object(audit, '_bfs_safe_cell', return_value=1):
                self.assertEqual(audit.local_label(env, 0, mask), (1, 'escape', 0.5))
            mask[1] = False
            with patch.object(audit, '_bfs_safe_cell', return_value=1):
                self.assertEqual(audit.local_label(env, 0, mask), (None, 'missing', 0))

    def test_saved_replay_outcomes_and_masks_must_match_and_labels_are_pure(self):
        case = next(c for c in scenarios(('validation',)) if c['group'] == 'approach')
        def teacher(env, observations, aid):
            mask = audit.get_action_mask(env, aid)
            label = audit.local_label(env, aid, mask)[0]
            return label if label is not None else next(a for a, allowed in enumerate(mask) if allowed)
        game = play_case(case, teacher, 'teacher', 'static', 2026101401, 80)
        frames = audit.replay_labels(game, case)
        self.assertEqual(len(frames), game['steps'])
        self.assertTrue(all(r['agrees'] for r in frames if r['label'] is not None))
        broken = deepcopy(game)
        broken['tape'][0]['mask'][0] = not broken['tape'][0]['mask'][0]
        with self.assertRaises(AssertionError):
            audit.replay_labels(broken, case)
        broken = deepcopy(game)
        broken['owned_kill_success'] = not game['owned_kill_success']
        with self.assertRaises(AssertionError):
            audit.replay_labels(broken, case)

    def test_no_test_split_or_unwanted_model_is_selected(self):
        base = {'split': 'validation', 'group': 'approach', 'opponent': 'static', 'model': 'original'}
        records = [base, {**base, 'split': 'test'}, {**base, 'opponent': 'rule'}, {**base, 'model': 'other'}]
        data = gzip.compress(('\n'.join(json.dumps(r) for r in records)).encode())
        with patch('gzip.open', side_effect=lambda *a, **k: io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(data)))):
            self.assertEqual(list(audit.read_selected('in_memory.gz', baseline=True)), [base])

    def test_closed_loop_teacher_is_masked_and_state_rng_pure(self):
        for case in scenarios(('validation',), smoke=True):
            if case['group'] in ('escape', 'approach'):
                result = audit.teacher_rollout(case, 2026101401)
                self.assertEqual(result['missing_labels'], sum(row['kind'] == 'missing' for row in result['tape']))
                self.assertTrue(all(row['mask'][row['actions'][case['aid']]] for row in result['tape']))

    def test_common_path_probabilities_use_same_states_and_masked_labels(self):
        case = next(c for c in scenarios(('validation',), smoke=True) if c['group'] == 'approach')
        rollout = audit.teacher_rollout(case, 2026101401)
        with patch.object(audit, 'ActorCriticNet') as constructor, patch.object(audit, 'smart_load_weights'):
            constructor.return_value.return_value = (torch.zeros(1, 6), torch.zeros(1, 1))
            result = audit.teacher_probabilities([case], [rollout])
        for records in result.values():
            self.assertEqual(len(records), 8)
            for record in records:
                reference = rollout['tape'][record['tick']]
                self.assertEqual(record['label'], reference['actions'][case['aid']])
                self.assertAlmostEqual(record['teacher_probability'], 1 / sum(reference['mask']))
                self.assertTrue(all(p == 0 for p, allowed in zip(record['probabilities'], reference['mask']) if not allowed))


class CounterBombChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_delayed_counter_and_four_controls_all_orientations_roles(self):
        results = certify_fixtures()
        self.assertEqual(len(results), 96)
        for result in results:
            unsafe = result['variant'] in ('returned_capacity', 'occupied_tile')
            self.assertEqual(result['surviving_path_with_fixed_enemy_route'] is None, unsafe)
            self.assertEqual(any(result['aid'] in row['self_blast_deaths'] for row in result['tape']), unsafe)

    def test_ammo_one_cannot_schedule_before_capacity_returns(self):
        env, schedule, _ = make_fixture('returned_capacity')
        early = [(0, schedule[0][1]), (3, schedule[1][1])]
        with self.assertRaises(ValueError):
            env._forecast_blasts(scheduled_bombs=early)
        env._forecast_blasts(scheduled_bombs=schedule)
        env, schedule, _ = make_fixture('capacity_unavailable')
        with self.assertRaises(ValueError):
            env._forecast_blasts(scheduled_bombs=schedule + [(3, Bomb(1, 2, 1, 8, 2))])

    def test_late_bomb_does_not_block_early_and_diagnostics_do_not_mutate(self):
        env, schedule, route = make_fixture('occupied_tile')
        before = audit.fingerprint(env)
        burns, blocked = env._forecast_blasts(scheduled_bombs=schedule)
        self.assertFalse(blocked[2, 1, 2])
        self.assertTrue(blocked[3, 1, 2])
        self.assertFalse(burns[3, 1, 2])
        surviving_path(env, 0, schedule)
        self.assertIsNone(masked_surviving_path(env, 0, schedule, route))
        parity_replay(env, 0, schedule, [5, 1, 1, 2, 1, 2, 1, 0], route)
        self.assertEqual(audit.fingerprint(env), before)

    def test_timed_reachability_cannot_stand_inside_new_opponent_bomb(self):
        env, schedule, route = make_fixture('occupied_tile')
        # Blocking before placement would wrongly remove the early-time tile.
        self.assertFalse(env._forecast_blasts(scheduled_bombs=schedule)[1][0, 1, 2])
        path = surviving_path(env, 0, schedule)
        if path is not None:
            pos = (3, 1)
            from bomberman_env import _MOVE_DELTA
            for tick, action in enumerate(path):
                if action in _MOVE_DELTA:
                    dr, dc = _MOVE_DELTA[action]
                    pos = pos[0] + dr, pos[1] + dc
                if tick == 3:
                    self.assertNotEqual(pos, (1, 2))


if __name__ == '__main__':
    unittest.main()
