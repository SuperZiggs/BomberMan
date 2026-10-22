"""Independent mathematical and manifest acceptance tests for D-S2."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from test_phase_d_s2_protocol import (ROOT, CHECKPOINTS, descriptor, function, game_row)


CNN = ('G', 'F', 'S', 'S2')


def metrics(active=20., win=40., draw=50., suicide=2.):
    return {'active_win_rate': active, 'win_rate': win, 'draw_rate': draw, 'suicide_rate': suicide}


def candidate(name='S2', episode=0, *, active=20., win=40., draw=50., suicide=2.,
              rb_active=40., rb_suicide=1., arm=None):
    result = descriptor(name, episode, arm)
    result['metrics'] = {opponent: metrics(active, win, draw, suicide) for opponent in CNN}
    result['metrics']['RB'] = metrics(rb_active, 60., 20., rb_suicide)
    return result


def candidates(control=(), pool=()):
    return {'S2': candidate(), 'control': list(control), 'pool': list(pool)}


def comparison(delta, lower=1., upper=9.):
    return {'difference_pp': delta, 'ci95_pp': [lower, upper]}


def passing_gate_fixture():
    comparisons = {
        'pool_minus_S2': {'macro_cnn': {'active_win': comparison(5.), 'draw': comparison(0., -1., 1.),
                                       'suicide': comparison(1., 0., 2.)},
                          'RB': {'active_win': comparison(-2., -4., 1.), 'suicide': comparison(1., -1., 2.)}},
        'pool_minus_control': {'macro_cnn': {'active_win': comparison(1., .1, 2.)}},
    }
    matchups = {opponent: {'games': 200, 'win_rate': 60.} for opponent in (*CNN, 'RB', 'snapshot-200')}
    matchups['B'] = {'games': 200, 'win_rate': 0.}
    return comparisons, matchups, {'games': 200, 'win_rate': 60.}, ['snapshot-200']


class SelectionContractTests(unittest.TestCase):
    def select(self, data):
        return function(self, 'select_candidates')(data)

    def test_shared_s2_fallback_is_always_available_and_can_win(self):
        result = self.select(candidates(control=[candidate('c500', 500, active=19.)],
                                        pool=[candidate('p500', 500, active=19.)]))
        self.assertEqual(result['models']['control']['sha256'], CHECKPOINTS['S2'][1])
        self.assertEqual(result['models']['pool']['episode'], 0)
        self.assertFalse(result['open_test'])

    def test_rb_performance_does_not_replace_cnn_macro_and_b_is_excluded(self):
        good = candidate('good', 500, active=26., rb_active=40., arm='pool')
        bad = candidate('bad', 1000, active=21., rb_active=90., arm='pool')
        bad['metrics']['B'] = metrics(100., 100., 0., 0.)
        result = self.select(candidates(pool=[bad, good]))
        self.assertEqual(result['models']['pool']['sha256'], good['sha256'])

    def test_guard_thresholds_are_inclusive_and_each_excess_is_rejected(self):
        accepted = candidate('edge', 500, active=30., suicide=3., rb_active=38., rb_suicide=2.)
        result = self.select(candidates(pool=[accepted]))
        self.assertEqual(result['models']['pool']['episode'], 500)
        for field in ('rb_active', 'rb_suicide', 'suicide'):
            bad = candidate('bad', 1000, active=90., suicide=3.0001 if field == 'suicide' else 3.,
                            rb_active=37.9999 if field == 'rb_active' else 38.,
                            rb_suicide=2.0001 if field == 'rb_suicide' else 2.)
            with self.subTest(field=field):
                result = self.select(candidates(pool=[accepted, bad]))
                self.assertEqual(result['models']['pool']['sha256'], accepted['sha256'])

    def test_cnn_guard_and_macro_use_all_four_opponents_equally(self):
        c = candidate('macro', 500, active=20., suicide=2.)
        c['metrics']['G']['active_win_rate'] = 40.
        c['metrics']['RB']['active_win_rate'] = 99.
        rival = candidate('uniform', 1000, active=26.)
        result = self.select(candidates(pool=[c, rival]))
        self.assertEqual(result['models']['pool']['sha256'], rival['sha256'])

    def test_tuple_orders_draw_then_win_then_suicide_then_earlier_episode(self):
        cases = [
            (candidate('a', 500, active=30., draw=40.), candidate('b', 1000, active=30., draw=39.)),
            (candidate('a', 500, active=30., win=40.), candidate('b', 1000, active=30., win=41.)),
            (candidate('a', 500, active=30., suicide=2.), candidate('b', 1000, active=30., suicide=1.)),
            (candidate('b', 1000, active=30.), candidate('a', 500, active=30.)),
        ]
        for first, expected in cases:
            with self.subTest(expected=expected['id']):
                result = self.select(candidates(pool=[first, expected]))
                self.assertEqual(result['models']['pool']['sha256'], expected['sha256'])

    def test_tie_tolerance_is_one_e_minus_twelve(self):
        early = candidate('early', 500, active=30.)
        near = candidate('near', 1000, active=30. + 5e-13)
        far = candidate('far', 1000, active=30. + 2e-12)
        self.assertEqual(self.select(candidates(pool=[near, early]))['models']['pool']['episode'], 500)
        self.assertEqual(self.select(candidates(pool=[far, early]))['models']['pool']['episode'], 1000)

    def test_open_test_requires_three_pp_and_no_worse_than_selected_control(self):
        control = candidate('control', 500, active=23., arm='control')
        for active, expected in ((22.9999, False), (23., True)):
            with self.subTest(active=active):
                result = self.select(candidates(control=[control], pool=[candidate('pool', 500, active=active, arm='pool')]))
                self.assertEqual(result['open_test'], expected)
        superior_control = candidate('control', 500, active=25., arm='control')
        result = self.select(candidates(control=[superior_control], pool=[candidate('pool', 500, active=24., arm='pool')]))
        self.assertFalse(result['open_test'])


class PairedMacroContractTests(unittest.TestCase):
    def rows(self, active_by_board, learner):
        return [game_row(board=630000 + board, corner=corner, opponent=opponent, learner=learner,
                         active=active, win=active, draw=1-active)
                for board, active in enumerate(active_by_board) for opponent in CNN for corner in (0, 1)]

    def test_known_constant_delta_has_degenerate_correct_board_ci(self):
        paired = function(self, 'paired_macro_comparison')
        result = paired(self.rows([1, 1, 1], 'pool'), self.rows([0, 0, 0], 'S2'), 'active_win')
        self.assertEqual(result['board_pairs'], 3)
        self.assertEqual(result['difference_pp'], 100.)
        self.assertEqual(result['ci95_pp'], [100., 100.])

    def test_correlated_opponent_corner_rows_are_one_board_cluster(self):
        paired = function(self, 'paired_macro_comparison')
        left = self.rows([1, 0], 'pool')
        right = self.rows([0, 0], 'S2')
        result = paired(left, right, 'active_win')
        self.assertEqual(result['board_pairs'], 2)
        self.assertEqual(result['difference_pp'], 50.)
        self.assertEqual(result['ci95_pp'], [0., 100.], 'Row-level resampling incorrectly narrows the board CI')
        self.assertEqual(result, paired(list(reversed(left)), list(reversed(right)), 'active_win'))

    def test_macro_averages_four_opponents_and_two_corners_before_bootstrap(self):
        paired = function(self, 'paired_macro_comparison')
        left = self.rows([0, 0], 'pool'); right = self.rows([0, 0], 'S2')
        for row in left:
            if row['opponent_id'] == 'G' and row['corner'] == 0:
                row.update(active_win=1, win=1, draw=0, kills=1, opponent_outcome='loss')
        result = paired(left, right, 'active_win')
        self.assertEqual(result['difference_pp'], 12.5)
        self.assertEqual(result['ci95_pp'], [12.5, 12.5])

    def test_incomplete_opponents_corners_duplicate_and_misaligned_boards_reject(self):
        paired = function(self, 'paired_macro_comparison')
        valid = self.rows([1, 0], 'pool'); right = self.rows([0, 0], 'S2')
        cases = [valid[:-1], valid + [deepcopy(valid[0])],
                 [r for r in valid if r['opponent_id'] != 'S'],
                 [r for r in valid if r['board'] != 630001]]
        for rows in cases:
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                paired(rows, right, 'active_win')

    def test_rb_and_b_comparisons_are_separate_from_cnn_macro(self):
        paired = function(self, 'paired_macro_comparison')
        for opponent in ('RB', 'B'):
            left = [game_row(board=b, corner=c, opponent=opponent, win=1, active=1, draw=0)
                    for b in (630000, 630001) for c in (0, 1)]
            right = [game_row(board=b, corner=c, opponent=opponent, learner='S2')
                     for b in (630000, 630001) for c in (0, 1)]
            self.assertEqual(paired(left, right, 'active_win', opponents=(opponent,))['difference_pp'], 100.)


class DecisionGateContractTests(unittest.TestCase):
    def decide(self, fixture):
        return function(self, 'decide_gates')(*fixture)

    def test_all_inclusive_point_estimate_boundaries_pass_and_b_is_diagnostic(self):
        result = self.decide(passing_gate_fixture())
        self.assertTrue(result['pilot_gate']['pass'])
        self.assertEqual(result['pilot_gate']['status'], 'PILOT_PASS')
        self.assertTrue(result['phase_d_gate']['pass'])
        self.assertEqual(result['phase_d_gate']['status'], 'PHASE_D_PASS')

    def test_every_pilot_condition_can_fail_independently(self):
        changes = [('pool_minus_S2', 'macro_cnn', 'active_win', 'difference_pp', 4.9999),
                   ('pool_minus_S2', 'macro_cnn', 'active_win', 'ci95_pp', [0., 9.]),
                   ('pool_minus_control', 'macro_cnn', 'active_win', 'ci95_pp', [0., 2.]),
                   ('pool_minus_S2', 'macro_cnn', 'draw', 'difference_pp', .0001),
                   ('pool_minus_S2', 'macro_cnn', 'suicide', 'difference_pp', 1.0001),
                   ('pool_minus_S2', 'RB', 'active_win', 'difference_pp', -2.0001),
                   ('pool_minus_S2', 'RB', 'suicide', 'difference_pp', 1.0001)]
        for first, second, third, key, value in changes:
            fixture = passing_gate_fixture()
            fixture[0][first][second][third][key] = value
            with self.subTest(condition=(first, second, third, key)):
                result = self.decide(fixture)
                self.assertFalse(result['pilot_gate']['pass'])
                self.assertEqual(result['pilot_gate']['status'], 'PILOT_NOT_PASS')
                self.assertFalse(result['phase_d_gate']['pass'])

    def test_phase_d_requires_all_fixed_and_all_older_snapshots_full_200_games(self):
        for opponent in (*CNN, 'snapshot-200'):
            for change in ('missing', 'games', 'win_rate'):
                fixture = passing_gate_fixture()
                if change == 'missing':
                    fixture[1].pop(opponent)
                else:
                    fixture[1][opponent][change] = 199 if change == 'games' else 59.9999
                with self.subTest(opponent=opponent, change=change):
                    result = self.decide(fixture)
                    self.assertTrue(result['pilot_gate']['pass'])
                    self.assertFalse(result['phase_d_gate']['pass'])

    def test_no_older_snapshot_and_rb_regression_cannot_complete_phase_d(self):
        fixture = passing_gate_fixture(); fixture[3].clear()
        self.assertFalse(self.decide(fixture)['phase_d_gate']['pass'])
        fixture = passing_gate_fixture(); fixture[1]['RB']['win_rate'] = 59.9999
        self.assertFalse(self.decide(fixture)['phase_d_gate']['pass'])

    def test_wr_uses_all_games_and_cannot_be_replaced_by_conditional_wr(self):
        fixture = passing_gate_fixture()
        fixture[1]['G'].update(win_rate=30., draw_rate=50., decisive_win_rate=60.)
        self.assertFalse(self.decide(fixture)['phase_d_gate']['pass'])


class TestManifestContractTests(unittest.TestCase):
    def fixture(self):
        registration = {'format': 'd-s2-registration-v1', 'registration_digest': 'a' * 64, 'mode': 'full',
                        'config': function(self, 'config_for')(smoke=False),
                        'checkpoints': {name: descriptor(name) for name in (*CNN, 'B')},
                        'seed_manifest': {'mode': 'full', 'ranges': {
                            'train': {'start': 610000, 'end_inclusive': 610999},
                            'validation': {'start': 620000, 'end_inclusive': 620039},
                            'test_reference': {'start': 630000, 'end_inclusive': 630099},
                            'test_snapshots': {'start': 640000, 'end_inclusive': 640099}}, 'inventory_digest': 'b' * 64}}
        registration['checkpoints']['RB'] = {'id': 'RB', 'checkpoint': None, 'sha256': 'c' * 64,
                                            'episode': None, 'role': 'anchor'}
        models = {'S2': descriptor('S2', 0), 'control': descriptor('control', 500, 'control'),
                  'pool': descriptor('pool', 1000, 'pool')}
        snapshots = [descriptor(f'pool-{ep}', ep, 'pool') for ep in range(200, 2001, 200)]
        snapshots.append(descriptor('control-200', 200, 'control'))
        return registration, models, snapshots

    def test_all_three_learners_all_reference_opponents_and_all_earlier_pool_snapshots(self):
        build = function(self, 'build_test_manifest')
        registration, models, snapshots = self.fixture()
        tasks = build(registration, models, snapshots)
        self.assertEqual(len(tasks), 3 * 6 * 100 + 4 * 100)
        self.assertEqual(len({t['id'] for t in tasks}), len(tasks))
        for learner in models.values():
            own = [t for t in tasks if t['learner']['sha256'] == learner['sha256']]
            references = [t for t in own if t['board'] < 640000]
            self.assertEqual({t['opponent']['id'] for t in references}, {*CNN, 'B', 'RB'})
            self.assertEqual({t['board'] for t in references}, set(range(630000, 630100)))
        older = [t for t in tasks if t['board'] >= 640000]
        self.assertEqual({t['opponent']['episode'] for t in older}, {200, 400, 600, 800})
        self.assertTrue(all(t['learner']['sha256'] == models['pool']['sha256'] for t in older))
        self.assertTrue(all(t['corners'] == [0, 1] and t['max_steps'] == 500 for t in tasks))
        self.assertTrue(any(t['learner']['sha256'] == t['opponent']['sha256'] == CHECKPOINTS['S2'][1]
                            and t['opponent']['id'] == 'S2' for t in tasks))

    def test_identical_learner_sha_reuses_task_ids_without_alias_in_identity(self):
        build = function(self, 'build_test_manifest')
        registration, models, snapshots = self.fixture()
        models['control'] = {**models['S2'], 'id': 'control-alias'}
        tasks = build(registration, models, snapshots)
        self.assertEqual(len(tasks), 2 * 6 * 100 + 4 * 100)
        again_models = deepcopy(models); again_models['control']['id'] = 'different-alias'
        self.assertEqual({t['id'] for t in tasks}, {t['id'] for t in build(registration, again_models, snapshots)})


class StrictCheckpointLoadContractTests(unittest.TestCase):
    def test_b_zero_padding_preserves_logits_with_arbitrary_thirteenth_channel(self):
        load = function(self, 'load_registered_model')
        from train_actor_critic import ActorCriticNet
        torch.set_num_threads(1)
        original = ActorCriticNet(in_channels=12, board_size=9).eval()
        original.load_state_dict(torch.load(ROOT / CHECKPOINTS['B'][0], map_location='cpu', weights_only=True))
        expanded = load(ROOT / CHECKPOINTS['B'][0], 'B', device='cpu')
        self.assertFalse(expanded.training)
        self.assertTrue(torch.equal(expanded.conv[0].weight[:, 12], torch.zeros_like(expanded.conv[0].weight[:, 12])))
        inputs = torch.randn(3, 13, 9, 9, generator=torch.Generator().manual_seed(808))
        with torch.no_grad():
            old_logits, old_value = original(inputs[:, :12])
            logits, value = expanded(inputs)
        torch.testing.assert_close(logits, old_logits, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(value, old_value, rtol=1e-6, atol=1e-6)

    def test_non_b_12_channels_wrong_board_and_malformed_tensor_reject(self):
        load = function(self, 'load_registered_model')
        with self.assertRaises((ValueError, RuntimeError)):
            load(ROOT / CHECKPOINTS['B'][0], 'G', device='cpu')
        from train_actor_critic import ActorCriticNet
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-load-') as temporary:
            path = Path(temporary) / 'bad.pt'
            torch.save(ActorCriticNet(board_size=11).state_dict(), path)
            with self.assertRaises((ValueError, RuntimeError)):
                load(path, 'G', device='cpu')
            weights = ActorCriticNet(board_size=9).state_dict(); weights['actor_head.bias'] = torch.zeros(7)
            torch.save(weights, path)
            with self.assertRaises((ValueError, RuntimeError)):
                load(path, 'G', device='cpu')


if __name__ == '__main__':
    unittest.main()
