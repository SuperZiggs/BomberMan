import json
from pathlib import Path
import queue
import tempfile
import unittest

import probe_sd_schedule as sd93


def diff(value, low, high):
    return {'difference_pp': value, 'ci95_pp': [low, high]}


def macro(active, draw=(-20, -30, -10), suicide=0):
    return {'active_win': diff(*active), 'draw': diff(*draw), 'suicide': diff(suicide, -1, 1)}


class ScheduleProbeTests(unittest.TestCase):
    def test_verdict_rules_and_choice(self):
        good, rb_ok = macro((8, 2, 14)), {'active_win': diff(-1, -4, 2)}
        alts = ('S400', 'S350', 'S300')
        v = sd93.verdict({'S400': good, 'S350': macro((8.5, 3, 14)), 'S300': macro((12, 6, 18))}, dict.fromkeys(alts, rb_ok))
        self.assertEqual(v['chosen'], 'S300')  # clearly larger gain wins
        v = sd93.verdict({'S400': good, 'S350': macro((8.5, 3, 14)), 'S300': macro((8.9, 3, 14))}, dict.fromkeys(alts, rb_ok))
        self.assertEqual(v['chosen'], 'S400')  # within 1pp of the best -> latest start
        bad = {'S400': macro((8, 2, 14), draw=(-14, -20, -5)), 'S350': macro((4.9, 1, 9)),
               'S300': macro((8, 2, 14), suicide=1.5)}
        v = sd93.verdict(bad, dict.fromkeys(alts, rb_ok))
        self.assertIsNone(v['chosen'])
        v = sd93.verdict(dict.fromkeys(alts, good), dict.fromkeys(alts, {'active_win': diff(-3.5, -6, 0)}))
        self.assertIsNone(v['chosen'])  # must not lose more than 3pp against RuleBot

    def test_third_ring_ends_every_game_by_start_plus_50(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, games = sd93.play('S300', 'S2', 21000, 2, 500, str(Path(tmp, 'x.json')), queue.Queue())
        self.assertTrue(all(g['steps'] <= 350 for g in games))

    def test_tiny_run_covers_every_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, 'out')
            summary = sd93.run(out, boards=2, workers=2, max_steps=12, shard=1, reference=Path(tmp))
            self.assertEqual(set(summary['results']), {f'{s}_vs_{o}' for s in sd93.SCHEDULES for o in sd93.OPPONENTS})
            self.assertEqual(summary['macro_cnn']['S300']['draw']['boards'], 2)
            self.assertIn('chosen', summary['verdict'])
            (out / 'opened.json').write_text(json.dumps({}), encoding='utf-8')
            with self.assertRaises(ValueError):  # resume refuses a different registered design
                sd93.run(out, boards=2, workers=2, max_steps=12, shard=1, resume=True, reference=Path(tmp))
            with self.assertRaises(FileExistsError):  # never overwrites
                sd93.run(out, boards=2, workers=2, max_steps=12, shard=1, reference=Path(tmp))


if __name__ == '__main__':
    unittest.main()
