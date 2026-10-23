import json
from pathlib import Path
import tempfile
import unittest

import probe_s3_cnn as sd91


def diff(value, low, high):
    return {'difference_pp': value, 'ci95_pp': [low, high]}


class S3CnnProbeTests(unittest.TestCase):
    def test_verdict_needs_fewer_draws_and_more_active_wins(self):
        good = {'draw': diff(-20, -30, -10), 'active_win': diff(8, 2, 14)}
        self.assertTrue(sd91.verdict(good)['s3_fixes_draws'])
        self.assertFalse(sd91.verdict({**good, 'draw': diff(-14.5, -25, -4)})['s3_fixes_draws'])
        self.assertFalse(sd91.verdict({**good, 'draw': diff(-20, -40, 1)})['s3_fixes_draws'])
        self.assertFalse(sd91.verdict({**good, 'active_win': diff(8, -1, 14)})['s3_fixes_draws'])

    def test_tiny_run_plays_every_arm_with_search_as_opponent(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, 'out')
            summary = sd91.run(out, boards=2, search_boards=1, workers=2, max_steps=12, shard=1)
            self.assertEqual(set(summary['results']), {f'{l}_vs_{o}' for l in sd91.LEARNERS for o in sd91.OPPONENTS})
            self.assertTrue(all(r['num_episodes'] == (2 if o == 'G_search3' else 4)
                                for k, r in summary['results'].items() for o in [k.split('_vs_')[1]]))
            self.assertEqual(summary['macro_cnn']['draw']['boards'], 2)
            self.assertIn('s3_fixes_draws', summary['verdict'])
            (out / 'opened.json').write_text(json.dumps({}), encoding='utf-8')
            with self.assertRaises(ValueError):  # resume refuses a different registered design
                sd91.run(out, boards=2, search_boards=1, workers=2, max_steps=12, shard=1, resume=True)
            with self.assertRaises(FileExistsError):  # never overwrites
                sd91.run(out, boards=2, search_boards=1, workers=2, max_steps=12, shard=1)


if __name__ == '__main__':
    unittest.main()
