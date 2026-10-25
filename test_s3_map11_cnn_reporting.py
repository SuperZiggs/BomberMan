"""Supporting boundary tests requested by the independent test author."""
import importlib
from pathlib import Path
import tempfile
import unittest


class ReportingBoundaries(unittest.TestCase):
    def setUp(self):
        self.probe = importlib.import_module('probe_s3_map11_cnn')

    def test_full_gate_boundaries_and_smoke_exclusion(self):
        def macro(gain=5.0, lower=0.01, suicide=1.0):
            return {'active_win': {'difference_pp': gain, 'ci95_pp': [lower, gain+1]},
                    'suicide': {'difference_pp': suicide, 'ci95_pp': [-1, 2]}}
        self.assertTrue(self.probe._verdict(macro(), smoke=False)['pass'])
        for data in (macro(gain=4.99), macro(lower=0), macro(lower=-.01), macro(suicide=1.01)):
            with self.subTest(macro=data):
                self.assertFalse(self.probe._verdict(data, smoke=False)['pass'])
        self.assertFalse(self.probe._verdict(macro(gain=100), smoke=True)['pass'])

    def test_full_shard_plan_covers_200_boards_and_both_learners_pool(self):
        config = self.probe.config_for(Path('unused-cnn-test-output'), seed=1900500000)
        jobs = self.probe._jobs(config, Path(config['output']))
        self.assertEqual(len(jobs), 120)
        groups = {}
        for learner, opponent, first, last, _, _ in jobs:
            self.assertEqual(last-first+1, 10)
            groups.setdefault((learner, opponent), []).extend(range(first, last+1))
        self.assertEqual(set(groups), {(l,o) for l in ('S3_11','S3') for o in ('S3','S2','G')})
        for seeds in groups.values():
            self.assertEqual(seeds, list(range(1900500000,1900500200)))

    def payload(self, **changes):
        config = self.probe.config_for(Path('unused-cnn-test-output'), seed=1900500000, smoke=True)
        row = dict(seed=1900500000, aid=0, win=0, active_win=0, draw=1, suicide=0,
                   passive_win=0, sudden_death_win=0, sudden_death_death=0,
                   kills=0, bombs=1, items=0, steps=12)
        row.update(changes)
        payload = dict(metadata=self.probe._metadata('S3_11','S3',1900500000,1900500000,config),
                       games=[row,dict(row,aid=1)],seconds=.01)
        return config,payload

    def test_consistent_win_draw_and_loss_outcomes_are_accepted(self):
        cases = ({},dict(draw=0),dict(win=1,draw=0,active_win=1,kills=1),
                 dict(win=1,draw=0,passive_win=1),dict(win=1,draw=0,sudden_death_win=1),
                 dict(draw=0,suicide=1),dict(draw=0,sudden_death_death=1))
        for fields in cases:
            with self.subTest(fields=fields):
                config,payload=self.payload(**fields)
                self.probe._validate_payload(payload,'S3_11','S3',1900500000,1900500000,config)

    def test_impossible_outcome_partitions_and_death_winners_are_refused(self):
        cases = (dict(win=1,draw=1,active_win=1,kills=1),
                 dict(win=1,draw=0,active_win=1,passive_win=1,kills=1),
                 dict(win=1,draw=0),dict(win=1,draw=0,active_win=1,kills=0),
                 dict(win=1,draw=0,active_win=1,kills=1,suicide=1),
                 dict(win=1,draw=0,sudden_death_win=1,sudden_death_death=1))
        for fields in cases:
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                config,payload=self.payload(**fields)
                self.probe._validate_payload(payload,'S3_11','S3',1900500000,1900500000,config)

    def test_fractional_and_boolean_game_counters_are_refused(self):
        for fields in (dict(steps=True),dict(steps=1.5),dict(kills=.5),dict(bombs=True),dict(items=.5)):
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                config,payload=self.payload(**fields)
                self.probe._validate_payload(payload,'S3_11','S3',1900500000,1900500000,config)

    def test_missing_or_modified_source_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'run'
            self.probe.initialize(output,seed=1900500000,smoke=True)
            path=output/'source'/'evaluate.py'
            original=path.read_bytes()
            path.unlink()
            with self.assertRaisesRegex(ValueError,'bundle|archive'):
                self.probe._registration(output)
            path.write_bytes(original+b'\n# modified archived source\n')
            with self.assertRaisesRegex(ValueError,'bundle|archive'):
                self.probe._registration(output)


if __name__ == '__main__':
    unittest.main()
