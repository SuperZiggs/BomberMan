import unittest

import kill_search
import run_sd10_distill as sd10


def diff(value, lower=1.0):
    return {'difference_pp': value, 'ci95_pp': [lower, value + 1]}


def summaries(**models):
    """models: name -> (macro active, macro suicide, rulebot active, rulebot suicide); same rate for each CNN."""
    out = {}
    for name, (active, suicide, rb_active, rb_suicide) in models.items():
        for o in ('S2', 'G', 'S3'):
            out[f'{name}_vs_{o}'] = {'active_win_rate': active, 'suicide_rate': suicide, 'draw_rate': 50, 'win_rate': 20}
        out[f'{name}_vs_rulebot'] = {'active_win_rate': rb_active, 'suicide_rate': rb_suicide, 'draw_rate': 1, 'win_rate': 95}
    return out


class SD10Tests(unittest.TestCase):
    def test_boards_opponents_and_seeds(self):
        config = sd10.config_for('x', smoke=False)
        self.assertEqual(config['collect_games'], 2000)
        boards = [b for s, e in sd10._shards(config) for b in range(s, e + 1)]
        self.assertEqual(boards, list(range(214000, 215000)))
        self.assertEqual([sd10.opponent_for(b) for b in boards[:5]], ['rulebot', 'S2', 'G', 'S3', 'rulebot'])
        self.assertEqual(sum(sd10.opponent_for(b) == 'S2' for b in boards), 250)
        self.assertEqual((sd10.TEST_SEED, sd10.VALIDATION_SEED), (58000, 21000))

    def test_select_rule(self):
        s = summaries(S3=(8, 6, 90, 0.5), epoch_4=(12, 6.5, 88, 1.0), epoch_8=(14, 7.5, 89, 0.5),
                      epoch_12=(13, 6, 86, 0.5), epoch_16=(12, 6, 90, 0.5))
        epoch, proceed, checks = sd10.select(s, [4, 8, 12, 16])
        # 8: suicide +1.5 out; 12: RuleBot -4 out; 4 and 16 tie on 12 -> earlier epoch
        self.assertEqual((epoch, proceed), (4, True))
        self.assertFalse(checks[8]['macro_suicide_within_1pp'])
        self.assertFalse(checks[12]['rulebot_active_within_3pp'])
        epoch, proceed, _ = sd10.select(summaries(S3=(8, 6, 90, 0.5), epoch_4=(8, 6, 90, 0.5)), [4])
        self.assertEqual((epoch, proceed), (4, False))  # not better than S3 -> VALIDATION_STOP
        self.assertEqual(sd10.select(summaries(S3=(8, 6, 90, 0.5), epoch_4=(20, 6, 80, 0.5)), [4])[:2], (None, False))

    def test_gate(self):
        good = {'macro_cnn': {'active_win': diff(5.0), 'suicide': diff(1.0)}, 'rulebot': {'active_win': diff(-3.0)}}
        self.assertTrue(sd10.gate(good)['pass'])
        for key, value in (('macro_cnn', {'active_win': diff(4.9), 'suicide': diff(0)}),
                           ('macro_cnn', {'active_win': diff(6, lower=-0.1), 'suicide': diff(0)}),
                           ('macro_cnn', {'active_win': diff(6), 'suicide': diff(1.1)}),
                           ('rulebot', {'active_win': diff(-3.1)})):
            self.assertFalse(sd10.gate({**good, key: value})['pass'])

    def test_cnn_game_uses_opponent_model_and_resets_it(self):
        import run_sd53_pilot as sd53
        config = sd10.config_for('x', smoke=True)  # max_steps 60
        net, models = sd53._load_net(sd10.S3, 'cpu'), sd10._opponent_models()
        seen = []
        original = kill_search.search_action3

        def spy(env, aid, mask, default):
            seen.append(kill_search.OPPONENT_MODEL)
            return original(env, aid, mask, default)
        kill_search.search_action3 = spy
        try:
            game, decisions, _ = sd10._record_game(214001, 0, config, net, models)  # board 214001 -> S2
        finally:
            kill_search.search_action3 = original
        self.assertEqual(game['opponent'], 'S2')
        self.assertTrue(seen and all(m is models['S2'][1] for m in seen))
        self.assertIsNone(kill_search.OPPONENT_MODEL)
        self.assertEqual(game['decisions'], len(decisions))
        again, _, _ = sd10._record_game(214001, 0, config, net, models)
        self.assertEqual(again, game)  # deterministic per board


if __name__ == '__main__':
    unittest.main()
