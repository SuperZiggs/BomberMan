import json
from pathlib import Path
import tempfile
import unittest

import kill_search
import probe_search_cnn_model as sdm
import run_pde_check as pde
from rule_based_bot import _safe_action_mask
from test_kill_search import trapped_opponent
from train_actor_critic import get_action_mask


class OpponentModelTests(unittest.TestCase):
    def tearDown(self):
        kill_search.OPPONENT_MODEL = None

    def test_rollouts_use_the_opponent_model_when_set(self):
        env = trapped_opponent()
        calls = []

        def model(e, aid):
            calls.append(aid); return 0
        kill_search.OPPONENT_MODEL = model
        kill_search.search_action3(env, 0, _safe_action_mask(env, 0), 0)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {1})

    def test_cnn_model_is_deterministic_and_legal(self):
        env = trapped_opponent()
        model = sdm.cnn_model(pde.CKPT['S'], 9)
        action = model(env, 1)
        self.assertEqual(action, model(env, 1))
        self.assertTrue(get_action_mask(env, 1)[action])

    def test_self_model_predicts_with_g_whoever_the_opponent_is(self):
        self.assertIsNone(sdm.model_ckpt('rulebot', 'S'))
        self.assertEqual(sdm.model_ckpt('self', 'rulebot'), pde.CKPT['G'])
        self.assertEqual(sdm.model_ckpt('oracle', 'S'), pde.CKPT['S'])

    def test_reused_shards_are_not_replayed_and_compared_to_rulebot(self):
        sdm_shards = Path('runs/sdm-21000/shards')
        if not sdm_shards.exists():
            self.skipTest('needs runs/sdm-21000')
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp, 'old', 'shards'); old.mkdir(parents=True)
            for model, src in (('rulebot', 'rulebot_vs_S_21000.json'), ('self', 'oracle_vs_S_21000.json')):
                name, sec, games = json.loads((sdm_shards / src).read_text(encoding='utf-8'))
                (old / f'{model}_vs_S_21000.json').write_text(json.dumps([f'{model}_vs_S', sec, games]))
            summary = sdm.run(Path(tmp, 'new'), 21000, 5, 1, opponents=('S',), models=('rulebot', 'self'),
                              reuse=Path(tmp, 'old'))
            self.assertEqual(set(summary['results']), {'rulebot_vs_S', 'self_vs_S'})
            self.assertEqual(summary['results']['self_vs_S']['num_episodes'], 10)
            self.assertIn('self - rulebot vs S', summary['comparisons'])
            self.assertIn('kill_search_sha256', json.loads(Path(tmp, 'new', 'opened.json').read_text()))
        with self.assertRaises(ValueError):
            sdm.run(Path(tmp, 'x'), 21000, 5, 1, opponents=('rulebot',), models=('rulebot', 'oracle'))


if __name__ == '__main__':
    unittest.main()
