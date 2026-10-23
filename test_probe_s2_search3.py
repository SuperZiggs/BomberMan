import json
from pathlib import Path
import shutil
import tempfile
import unittest

import probe_s2_search3 as sd90
import run_pde_check as pde
from test_kill_search import trapped_opponent
from train_actor_critic import get_action_mask

RUNS = Path('runs')


class S2Search3Tests(unittest.TestCase):
    def test_s2_search3_seat_plays_legal_moves(self):
        env = trapped_opponent()
        obs = [env._get_obs(a.agent_id) for a in env.agents]
        action = pde._seat('S2_search3', 9)(0)(env, obs, 0)
        self.assertTrue(get_action_mask(env, 0)[action])

    def test_reuses_g_search3_shards_and_compares_learners(self):
        if not all((RUNS / d / 'shards').exists() for d in ('sdm-21000', 'sdm2-21000')):
            self.skipTest('needs runs/sdm-21000 and runs/sdm2-21000')
        with tempfile.TemporaryDirectory() as tmp:
            runs, out = Path(tmp, 'runs'), Path(tmp, 'out')
            for d in ('sdm-21000', 'sdm2-21000'):
                (runs / d / 'shards').mkdir(parents=True)
                for f in (RUNS / d / 'shards').glob('*_21000.json'):
                    shutil.copy(f, runs / d / 'shards' / f.name)
            shutil.copy(RUNS / 'sdm2-21000' / 'opened.json', runs / 'sdm2-21000' / 'opened.json')
            # stand-ins for the shards that would need real games, so run() plays nothing
            fake = (RUNS / 'sdm-21000' / 'shards' / 'oracle_vs_G_21000.json').read_text(encoding='utf-8')
            (out / 'shards').mkdir(parents=True)
            for l in sd90.LEARNERS:
                for o in sd90.OPPONENTS:
                    if (l, o) not in sd90.REUSE:
                        (out / 'shards' / f'{l}_vs_{o}_21000.json').write_text(fake, encoding='utf-8')
            (out / 'opened.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):  # resume refuses a different registered design
                sd90.run(out, boards=5, workers=1, resume=True, runs=runs)
            opened = {'seed': 21000, 'boards': 5, 'max_steps': 500, 'learners': sd90.LEARNERS,
                      'opponents': sd90.OPPONENTS, 'G_sha256': pde.sd76.G_SHA256, 'S2_sha256': sd90.S2_SHA256,
                      'kill_search_sha256': pde.sd5._hash(sd90.ROOT / 'kill_search.py')}
            (out / 'opened.json').write_text(json.dumps(opened), encoding='utf-8')
            summary = sd90.run(out, boards=5, workers=1, resume=True, runs=runs)
            self.assertTrue(summary['reused'])
            self.assertEqual((out / 'shards' / 'G_search3_vs_rulebot_21000.json').read_text(encoding='utf-8'),
                             (RUNS / 'sdm2-21000' / 'shards' / 'rulebot_vs_rulebot_21000.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['results']['G_search3_vs_G']['num_episodes'], 10)
            self.assertEqual(set(summary['comparisons']), {f'S2_search3 - G_search3 vs {o}' for o in sd90.OPPONENTS})
        with self.assertRaises(FileExistsError):  # never overwrites a finished output
            sd90.run(RUNS / 'sdm2-21000', boards=5, workers=1, runs=runs)


if __name__ == '__main__':
    unittest.main()
