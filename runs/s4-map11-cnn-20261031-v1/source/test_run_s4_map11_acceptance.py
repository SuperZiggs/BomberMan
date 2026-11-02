"""Acceptance contracts for S4-11 (same-author fallback, not independent QA)."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import run_s4_map11_distill as runner
except ModuleNotFoundError as error:
    if error.name != 'run_s4_map11_distill':
        raise
    class _MissingRunner:
        def __getattr__(self, name):
            raise AssertionError(f'S4-11 behavior is not implemented: {name}')
    runner = _MissingRunner()


class ConfigurationTests(unittest.TestCase):
    def test_full_config_counts_and_board_shapes(self):
        config = runner.config_for('example-output')
        self.assertEqual((config['board_size'], config['num_agents'], config['obs_channels']), (11, 2, 13))
        self.assertEqual((config['max_steps'], config['collect_games']), (500, 1998))
        self.assertEqual(config['collect_board_seeds'], list(range(860000, 860999)))
        self.assertEqual(config['collection_shard_boards'], 20)
        self.assertEqual((config['pilot_boards'], config['evaluation_boards']), (10, 200))
        self.assertEqual((config['pilot_games'], config['validation_games'], config['test_games']), (160, 8000, 7200))
        self.assertEqual(config['candidate_epochs'], [4, 8, 12, 16])
        self.assertEqual((config['train_epochs'], config['learning_rate'], config['batch_size'], config['anchor_lambda']),
                         (16, 1e-4, 256, 1.0))
        self.assertEqual(config['schedule_e'], {'sudden_death_start': 475, 'sudden_death_interval': 25,
                                                'sudden_death_warning_steps': 40})
        self.assertEqual(config['schedule_e_probability'], 0.75)

    def test_three_opponents_have_exactly_333_boards_and_rulebot_absent(self):
        config = runner.config_for('example-output')
        assigned = [runner.opponent_for(seed, config) for seed in config['collect_board_seeds']]
        self.assertEqual({name: assigned.count(name) for name in ('S2', 'G', 'S3_11')},
                         {'S2': 333, 'G': 333, 'S3_11': 333})
        self.assertNotIn('rulebot', assigned)

    def test_smoke_is_small_marked_and_separate_from_full_seed_blocks(self):
        full = runner.config_for('full')
        smoke = runner.config_for('smoke', smoke=True)
        self.assertTrue(smoke['smoke'])
        self.assertEqual((smoke['max_steps'], smoke['train_epochs'], smoke['candidate_epochs']), (12, 2, [2]))
        self.assertTrue(all(900000 <= s <= 900003 for s in smoke['collect_board_seeds']))
        self.assertTrue(set(smoke['collect_board_seeds']).isdisjoint(full['collect_board_seeds']))
        for key in ('pilot_seed', 'validation_seed', 'test_seed'):
            self.assertTrue(900000 <= smoke[key] <= 900003)

    def test_non_integer_negative_overflow_or_overlapping_ranges_rejected(self):
        for invalid in (True, -1, 1.5, '860000', 2**32):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                runner.config_for('x', collect_first_seed=invalid)
        with self.assertRaises(ValueError):
            runner.config_for('x', pilot_seed=860000)

    def test_workers_are_bounded_and_bools_rejected(self):
        for value in (1, 5):
            self.assertEqual(runner.validate_workers(value), value)
        for value in (0, 6, True, 1.5, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_workers(value)


def candidate_summaries():
    result = {}
    for model, active in [('S4', 10), ('epoch_4', 15), ('epoch_8', 15), ('epoch_12', 14), ('epoch_16', 9)]:
        for opponent in ('S2', 'G', 'S3_11'):
            result[f'{model}_vs_{opponent}'] = {'active_win_rate': active, 'suicide_rate': 2,
                                                 'win_rate': active, 'draw_rate': 40}
        result[f'{model}_vs_rulebot'] = {'active_win_rate': 80, 'suicide_rate': 1,
                                        'win_rate': 80, 'draw_rate': 5}
    return result


def gate_comparison():
    return {'macro_cnn': {'active_win': {'difference_pp': 5.0, 'ci95_pp': [0.1, 8.0]},
                          'suicide': {'difference_pp': 1.0}},
            'rulebot': {'active_win': {'difference_pp': -3.0}, 'suicide': {'difference_pp': 1.0}}}


class SelectionAndGateTests(unittest.TestCase):
    def test_selection_uses_cnn_macro_and_earliest_epoch_on_tie(self):
        epoch, proceed, checks = runner.select(candidate_summaries())
        self.assertEqual((epoch, proceed), (4, True))
        self.assertEqual(checks[4]['macro_active'], 15)

    def test_rulebot_and_suicide_validation_guards(self):
        data = candidate_summaries()
        data['epoch_4_vs_rulebot']['active_win_rate'] = 76.99
        data['epoch_8_vs_rulebot']['suicide_rate'] = 2.01
        for name in ('S2', 'G', 'S3_11'):
            data[f'epoch_12_vs_{name}']['suicide_rate'] = 3.01
        self.assertEqual(runner.select(data)[:2], (16, False))

    def test_validation_strictly_requires_improvement(self):
        data = candidate_summaries()
        for epoch in (4, 8, 12, 16):
            for opponent in ('S2', 'G', 'S3_11'):
                data[f'epoch_{epoch}_vs_{opponent}']['active_win_rate'] = 10
        self.assertEqual(runner.select(data)[:2], (4, False))

    def test_test_gate_boundary_and_each_guard(self):
        self.assertTrue(runner.gate(gate_comparison())['pass'])
        for path, value in [(('macro_cnn', 'active_win', 'difference_pp'), 4.99),
                            (('macro_cnn', 'active_win', 'ci95_pp'), [0.0, 8]),
                            (('macro_cnn', 'suicide', 'difference_pp'), 1.01),
                            (('rulebot', 'active_win', 'difference_pp'), -3.01),
                            (('rulebot', 'suicide', 'difference_pp'), 1.01)]:
            comparison = gate_comparison()
            node = comparison
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path):
                self.assertFalse(runner.gate(comparison)['pass'])

    def test_pilot_readiness_rejects_no_labels_and_entirely_negative_ci(self):
        report = {'decisions_by_opponent': {'S2': 1, 'G': 1, 'S3_11': 1}, 'anchors': 1,
                  'macro_active_ci95_pp': [-10.0, 1.0]}
        self.assertTrue(runner.pilot_ready(report))
        bad = json.loads(json.dumps(report)); bad['decisions_by_opponent']['G'] = 0
        self.assertFalse(runner.pilot_ready(bad))
        bad = json.loads(json.dumps(report)); bad['anchors'] = 0
        self.assertFalse(runner.pilot_ready(bad))
        bad = json.loads(json.dumps(report)); bad['macro_active_ci95_pp'] = [-9, -0.01]
        self.assertFalse(runner.pilot_ready(bad))

    def test_masked_nan_is_allowed_but_infinity_or_no_score_is_not(self):
        mask = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
        self.assertTrue(runner.valid_scores(np.array([0., 1., np.nan, np.nan, np.nan, np.nan]), mask))
        self.assertFalse(runner.valid_scores(np.array([0., np.inf, np.nan, np.nan, np.nan, np.nan]), mask))
        self.assertFalse(runner.valid_scores(np.full(6, np.nan), mask))
        self.assertFalse(runner.valid_scores(np.array([0., 1., 1., np.nan, np.nan, np.nan]), mask))


class SafetyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='s4-map11-accept-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.output = self.base / 'run'

    def test_init_refuses_existing_output_without_touching_it(self):
        self.output.mkdir()
        sentinel = self.output / 'sentinel.txt'; sentinel.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            runner.initialize(self.output, smoke=True)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ['sentinel.txt'])

    def test_used_test_seed_refused_before_creating_output(self):
        other = self.base / 'old-run'; other.mkdir()
        (other / 'test_opened.json').write_text(json.dumps({'seed': 880000, 'boards': 200}), encoding='utf-8')
        with mock.patch.object(runner, 'LEDGER_ROOTS', (self.base,)), self.assertRaisesRegex(ValueError, 'seed|overlap'):
            runner.initialize(self.output)
        self.assertFalse(self.output.exists())

    def test_foreign_lock_is_never_removed(self):
        self.output.mkdir()
        marker = self.output / 'run.lock.json'; marker.write_text('foreign-lock', encoding='utf-8')
        with self.assertRaises(FileExistsError), runner.run_lock(self.output, 'hash'):
            pass
        self.assertEqual(marker.read_text(encoding='utf-8'), 'foreign-lock')

    def test_own_lock_released_on_exception_and_nested_entry_refused(self):
        self.output.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with runner.run_lock(self.output, 'hash'):
                self.assertTrue((self.output / 'run.lock.json').exists())
                with self.assertRaises(FileExistsError), runner.run_lock(self.output, 'other'):
                    pass
                raise RuntimeError('sentinel')
        self.assertFalse((self.output / 'run.lock.json').exists())

    def test_bundle_tampering_blocks_pilot_before_game(self):
        runner.initialize(self.output, smoke=True)
        source = self.output / 'source' / 'bomberman_env.py'
        with source.open('a', encoding='utf-8') as f:
            f.write('\n# tampered for test\n')
        with mock.patch.object(runner, 'parallel_map', side_effect=AssertionError('unexpected game')), \
                self.assertRaisesRegex(ValueError, 'source|hash|bundle'):
            runner.pilot(self.output, workers=1)
        self.assertFalse((self.output / 'pilot.json').exists())

    def test_test_marker_refuses_replay_before_any_game(self):
        runner.initialize(self.output, smoke=True)
        marker = self.output / 'test_opened.json'
        marker.write_text('{"already_open":true}', encoding='utf-8')
        with mock.patch.object(runner, 'parallel_map', side_effect=AssertionError('unexpected game')), \
                self.assertRaises((FileExistsError, ValueError)):
            runner.test(self.output, workers=1)
        self.assertEqual(json.loads(marker.read_text(encoding='utf-8')), {'already_open': True})

    def test_no_validation_selection_does_not_open_test(self):
        runner.initialize(self.output, smoke=True)
        with mock.patch.object(runner, 'parallel_map', side_effect=AssertionError('unexpected game')), \
                self.assertRaises((FileNotFoundError, ValueError)):
            runner.test(self.output, workers=1)
        self.assertFalse((self.output / 'test_opened.json').exists())

    def test_real_11x11_game_uses_correct_opponent_and_restores_hook_on_error(self):
        import kill_search
        config = runner.config_for(self.output, smoke=True)
        net = runner.load_net(runner.CHECKPOINTS['S4'])
        models = runner.load_opponent_models(('S2',))
        seen = []
        previous = kill_search.OPPONENT_MODEL
        sentinel = lambda env, aid: 0
        kill_search.OPPONENT_MODEL = sentinel
        self.addCleanup(setattr, kill_search, 'OPPONENT_MODEL', previous)
        def fail_search(env, aid, mask, sampled):
            seen.append((env.board_size, kill_search.OPPONENT_MODEL))
            raise RuntimeError('test interruption')
        with mock.patch.object(kill_search, 'search_action3', side_effect=fail_search), \
                self.assertRaisesRegex(RuntimeError, 'test interruption'):
            runner.record_game(900000, 0, 'S2', config, net, models, use_search=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], 11)
        self.assertIs(seen[0][1], models['S2'][1])
        self.assertIs(kill_search.OPPONENT_MODEL, sentinel)


class CollectionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='s4-shard-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = runner.config_for(self.root / 'out', smoke=True)

    def _write(self):
        from run_s3_map11_distill import _write_shard
        start, end = runner._shards(self.config)[0]
        games = [dict(seed=s, aid=a, opponent=runner.opponent_for(s, self.config))
                 for s in range(start, end + 1) for a in (0, 1)]
        path = self.root / 'shard.npz'
        _write_shard(path, runner._shard_meta(start, end, self.config), [], [], games)
        return path, start, end

    def test_shard_board_coverage_and_tampering(self):
        path, start, end = self._write()
        data, games = runner._read_shard(path, start, end, self.config)
        self.assertEqual(len(games), 4)
        games[-1] = games[0]
        data['games'] = np.array(json.dumps(games))
        np.savez_compressed(path, **data)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            runner._read_shard(path, start, end, self.config)

    def test_full_shards_are_49_forty_and_one_thirty_eight(self):
        shards = runner._shards(runner.config_for('full'))
        self.assertEqual(len(shards), 50)
        self.assertEqual([(b-a+1)*2 for a, b in shards], [40]*49 + [38])

    def test_pilot_macro_requires_all_paired_opponents_and_corners(self):
        seeds = range(2)
        rows = [dict(seed=s, aid=a, opponent=o, search=search, active_win=search)
                for s in seeds for a in (0, 1) for o in (*runner.CNN_OPPONENTS, 'rulebot')
                for search in (False, True)]
        delta, ci = runner._pilot_macro(rows, seeds)
        self.assertEqual((delta, ci), (100.0, [100.0, 100.0]))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            runner._pilot_macro(rows[:-1], seeds)


class EvaluationIntegrityTests(unittest.TestCase):
    def test_macro_bootstrap_pairs_both_corners_and_opponents(self):
        results = {}
        for name in ('new', 'old'):
            for opponent in runner.CNN_OPPONENTS:
                results[f'{name}_vs_{opponent}'] = [dict(seed=s, aid=a, active_win=(name == 'new'))
                                                     for s in (1, 2) for a in (0, 1)]
        paired = runner._paired(results, 'new', 'old', runner.CNN_OPPONENTS, 'active_win')
        self.assertEqual((paired['difference_pp'], paired['ci95_pp'], paired['board_pairs']),
                         (100.0, [100.0, 100.0], 2))
        results['new_vs_S2'].pop()
        with self.assertRaises(ValueError):
            runner._paired(results, 'new', 'old', runner.CNN_OPPONENTS, 'active_win')

    def test_interrupted_test_leaves_marker_and_refuses_replay(self):
        import hashlib
        with tempfile.TemporaryDirectory(prefix='s4-test-once-') as directory:
            output = Path(directory)
            checkpoint = output / 'candidate.pt'; checkpoint.write_bytes(b'candidate')
            sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            (output / 'selected.json').write_text(json.dumps({'epoch': 2, 'proceed_to_test': True,
                    'checkpoint': str(checkpoint), 'sha256': sha}), encoding='utf-8')
            (output / 'validation.json').write_text(json.dumps({'selected_epoch': 2, 'proceed_to_test': True}), encoding='utf-8')
            config = runner.config_for(output, smoke=True)
            with mock.patch.object(runner, '_registration', return_value=config), \
                 mock.patch.object(runner, '_candidate_paths', return_value={'epoch_2': checkpoint}), \
                 mock.patch.object(runner, '_evaluation', side_effect=RuntimeError('interrupted')) as evaluate:
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    runner.test(output, workers=1)
                self.assertTrue((output / 'test_opened.json').exists())
                with self.assertRaises(FileExistsError):
                    runner.test(output, workers=1)
                self.assertEqual(evaluate.call_count, 1)


if __name__ == '__main__':
    unittest.main()
