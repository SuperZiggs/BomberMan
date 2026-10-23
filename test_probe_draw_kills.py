import json
from pathlib import Path
import tempfile
import unittest

import probe_draw_kills as sd92


def pooled(percent, control=80):
    return {'draws_with_kill_oracle': {'percent': percent}, 'control_active_wins_with_kill_oracle': {'percent': control}}


class DrawKillProbeTests(unittest.TestCase):
    def test_verdict_boundaries(self):
        self.assertEqual(sd92.verdict(pooled(50))['call'], 'missed_kills_direction_A')
        self.assertEqual(sd92.verdict(pooled(20))['call'], 'structural_consider_rules')
        self.assertEqual(sd92.verdict(pooled(35))['call'], 'mixed')
        self.assertEqual(sd92.verdict(pooled(90, control=49))['call'], 'invalid_control')
        self.assertEqual(sd92.verdict(pooled(90, control=None))['call'], 'invalid_control')

    def test_share_resamples_boards(self):
        games = [{'seed': 1, 'x': True}, {'seed': 1, 'x': False}, {'seed': 2, 'x': True}, {'seed': 2, 'x': True}]
        share = sd92._share(games, lambda g: g['x'])
        self.assertEqual(share['percent'], 75.0)
        self.assertLessEqual(share['ci95'][0], 75.0)
        self.assertGreaterEqual(share['ci95'][1], 75.0)

    def test_tiny_run_checks_draw_states_under_both_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, 'out')
            summary = sd92.run(out, boards=1, workers=2, max_steps=12, stride=1, shard=1, reference=Path(tmp))
            self.assertEqual(set(summary['per_opponent']), set(sd92.OPPONENTS))
            self.assertEqual(summary['pooled']['games'], 6)
            self.assertGreater(summary['pooled']['draw_states'], 0)  # 12 steps: start corners are within REACH
            for o in sd92.OPPONENTS:
                games = json.loads(next((out / 'shards').glob(f'S3_vs_{o}_*.json')).read_text(encoding='utf-8'))[2]
                for g in games:
                    self.assertEqual(len(g['states']), g['near_states'] if g['draw'] or g['active_win'] else 0)
                    self.assertTrue(all(set(s) == {'step', 'oracle', 'rulebot'} for s in g['states']))
            self.assertIn(summary['verdict']['call'], ('missed_kills_direction_A', 'structural_consider_rules', 'mixed', 'invalid_control'))
            (out / 'opened.json').write_text(json.dumps({}), encoding='utf-8')
            with self.assertRaises(ValueError):  # resume refuses a different registered design
                sd92.run(out, boards=1, workers=2, max_steps=12, stride=1, shard=1, resume=True, reference=Path(tmp))
            with self.assertRaises(FileExistsError):  # never overwrites
                sd92.run(out, boards=1, workers=2, max_steps=12, stride=1, shard=1, reference=Path(tmp))


if __name__ == '__main__':
    unittest.main()
