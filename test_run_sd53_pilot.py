"""Independent, bounded acceptance contracts for the SD5.3 public runner."""

import hashlib
import importlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from benchmark_t3 import transform, transform_action


class PublicContracts(unittest.TestCase):
    def setUp(self):
        self.pilot = importlib.import_module('run_sd53_pilot')

    def test_required_stage_and_helper_interfaces_exist(self):
        for name in ('config_for', 'normalized_sha256', 'masked_soft_target',
                     'split_board_seeds', 'gate', 'augment_observation',
                     'augment_action_vector', 'collect', 'train', 'validate', 'test'):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(self.pilot, name)))

    def test_full_and_smoke_configuration_preserve_frozen_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            for smoke in (False, True):
                with self.subTest(smoke=smoke):
                    config = self.pilot.config_for(Path(folder), smoke)
                    self.assertIsInstance(config, dict)
                    serialized = json.dumps(config, default=str)
                    for invariant in ('0315a56f09079333d936db5f55587093f1b6415ed6355d2ef6ae2ab63f8b1483',
                                      '475', '25', '40', '200000', '201499', '21000', '43000'):
                        self.assertIn(invariant, serialized)

    def test_source_hash_normalizes_crlf_to_lf(self):
        with tempfile.TemporaryDirectory() as folder:
            lf, crlf = Path(folder) / 'lf.py', Path(folder) / 'crlf.py'
            lf.write_bytes(b'alpha\nbeta\n')
            crlf.write_bytes(b'alpha\r\nbeta\r\n')
            expected = hashlib.sha256(lf.read_bytes()).hexdigest()
            self.assertEqual(self.pilot.normalized_sha256(lf), expected)
            self.assertEqual(self.pilot.normalized_sha256(crlf), expected)

    def test_soft_target_uses_only_tied_positive_legal_maxima(self):
        scores = np.array([10.0, 4.0, 4.0, 0.0, -2.0, np.nan])
        mask = np.array([False, True, True, True, True, True])
        np.testing.assert_allclose(self.pilot.masked_soft_target(scores, mask),
                                   [0.0, 0.5, 0.5, 0.0, 0.0, 0.0])

    def test_soft_target_keeps_safe_zero_over_lethal_negative_action(self):
        np.testing.assert_array_equal(
            self.pilot.masked_soft_target(np.array([0.0, -1.0, np.nan]), np.array([True, True, True])),
            [1.0, 0.0, 0.0])

    def test_soft_target_has_no_mass_for_illegal_or_uninformative_states(self):
        for scores, mask in (([0.0, 0.0, np.nan], [True, True, True]),
                             ([7.0, 4.0, 3.0], [False, False, False]),
                             ([np.inf, 0.0, 2.0], [True, True, False])):
            with self.subTest(scores=scores):
                target = self.pilot.masked_soft_target(np.asarray(scores), np.asarray(mask))
                np.testing.assert_array_equal(target, np.zeros(len(scores)))

    def test_board_seed_split_keeps_both_corners_together_and_is_deterministic(self):
        seeds = np.repeat(np.arange(200000, 200020), 2)
        first = self.pilot.split_board_seeds(seeds, heldout_fraction=0.25, seed=17)
        second = self.pilot.split_board_seeds(seeds, heldout_fraction=0.25, seed=17)
        self.assertEqual(len(first), 2)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        train, heldout = (np.asarray(part) for part in first)
        self.assertFalse(set(train) & set(heldout))
        self.assertEqual(sorted(np.concatenate((train, heldout)).tolist()), sorted(seeds.tolist()))
        for seed in np.unique(seeds):
            self.assertIn(int(np.count_nonzero(train == seed)), (0, 2))
            self.assertIn(int(np.count_nonzero(heldout == seed)), (0, 2))
        self.assertEqual(len(heldout), 10)

    def test_joint_seed_holdout_never_splits_decision_and_anchor_partitions(self):
        decisions = np.repeat(np.arange(200000, 200020), 2)
        anchors = np.repeat(np.arange(200000, 200030), 2)
        held = self.pilot._heldout_board_ids(np.concatenate((decisions, anchors)), 0.1, 530)
        decision_train, decision_held = self.pilot._partition_indices(decisions, held)
        anchor_train, anchor_held = self.pilot._partition_indices(anchors, held)
        self.assertFalse(set(decisions[decision_train]) & set(anchors[anchor_held]))
        self.assertFalse(set(decisions[decision_held]) & set(anchors[anchor_train]))

    def test_full_config_uses_only_the_three_frozen_candidate_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            config = self.pilot.config_for(Path(folder), smoke=False)
        self.assertEqual(config['candidate_epochs'], [2, 4, 8])

    def test_collection_guard_rejects_missing_or_pending_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            config = self.pilot.config_for(output, smoke=True)
            collection = output / 'collection'
            collection.mkdir()
            shard = collection / 'shard-200000-200003.npz'
            shard.touch()
            with mock.patch.object(self.pilot, '_valid_shard', return_value=True):
                with self.assertRaisesRegex(ValueError, 'symmetry audit'):
                    self.pilot._require_complete_collection(output, config, {'symmetry': {'state': 'pending'}})

    def test_invalid_completed_shard_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'shard-200000-200003.npz'
            config = self.pilot.config_for(Path(folder), smoke=True)
            metadata = {'first_seed': 200000, 'last_seed': 200003, 'F_sha256': 'wrong',
                        'config_sha256': 'wrong', 'source_sha256': {}}
            self.pilot._write_shard(path, metadata, [], [], [])
            self.assertFalse(self.pilot._valid_shard(path, 200000, 200003, config))

    def test_safe_selection_excludes_suicide_regression(self):
        results = {'F': {'suicide_rate': 1.0},
                   'epoch_2': {'suicide_rate': 2.0, 'active_win_rate': 90.0},
                   'epoch_4': {'suicide_rate': 2.01, 'active_win_rate': 99.0}}
        self.assertEqual([epoch for epoch, _ in self.pilot._safe_candidates(results)], [2])

    def test_validate_is_frozen_once_test_seed_is_opened(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'test_opened.json').write_text('{}', encoding='utf-8')
            with mock.patch.object(self.pilot, '_require_f'), mock.patch.object(self.pilot, '_stage'):
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    self.pilot.validate(output, smoke=True)

    def test_masked_ce_and_kl_have_no_illegal_logit_gradient(self):
        import torch
        import torch.nn.functional as functional
        logits = torch.tensor([[1.0, 20.0, 0.5]], requires_grad=True)
        mask = torch.tensor([[True, False, True]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        student = functional.log_softmax(self.pilot._masked_logits(logits, mask), dim=1)
        teacher = functional.softmax(self.pilot._masked_logits(torch.tensor([[0.0, -10.0, 1.0]]), mask), dim=1)
        loss = -(target * student).sum() + functional.kl_div(student, teacher, reduction='batchmean')
        loss.backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.0)

    def test_gate_thresholds_include_exact_five_and_exact_one(self):
        def paired(active, lower, suicide):
            return {'search_vs_f': {
                'active_win': {'difference_pp': active, 'ci95_pp': [lower, active + 2]},
                'suicide': {'difference_pp': suicide, 'ci95_pp': [suicide - 2, suicide + 2]}}}

        cases = ((5.0, 0.01, 1.0, True),
                 (4.999, 0.01, 1.0, False),
                 (5.0, 0.0, 1.0, False),
                 (5.0, -0.01, 1.0, False),
                 (5.0, 0.01, 1.001, False),
                 (6.0, 0.5, -2.0, True))
        for active, lower, suicide, expected in cases:
            with self.subTest(active=active, lower=lower, suicide=suicide):
                self.assertIs(self.pilot.gate(paired(active, lower, suicide))['pass'], expected)

    def test_all_eight_board_symmetries_match_benchmark(self):
        obs = np.arange(13 * 9 * 9, dtype=np.float32).reshape(13, 9, 9)
        values = np.arange(6, dtype=np.float32) + 0.25
        for rotation in range(4):
            for mirror in (False, True):
                with self.subTest(rotation=rotation, mirror=mirror):
                    transformed = self.pilot.augment_observation(obs, rotation, mirror)
                    self.assertEqual(transformed.shape, obs.shape)
                    for row in range(9):
                        for col in range(9):
                            transformed_row, transformed_col = transform((row, col), rotation, mirror)
                            np.testing.assert_array_equal(transformed[:, transformed_row, transformed_col],
                                                          obs[:, row, col])
                    actions = self.pilot.augment_action_vector(values, rotation, mirror)
                    self.assertEqual(actions.shape, values.shape)
                    for action in range(len(values)):
                        self.assertEqual(actions[transform_action(action, rotation, mirror)], values[action])

    def test_opened_test_seed_refuses_second_invocation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'test_opened.json').write_text(json.dumps({'seed': 43000}), encoding='utf-8')
            with self.assertRaises(FileExistsError):
                self.pilot.test(output, smoke=True)


if __name__ == '__main__':
    unittest.main()
