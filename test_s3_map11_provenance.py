"""Regression for the source-closure gap identified by independent review."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_s3_map11_distill as runner


class SourceClosureTests(unittest.TestCase):
    def initialize(self, output):
        return runner.initialize(output, collect_first_seed=1900300000, collect_boards=4,
                                 validation_seed=1900301000, test_seed=1900302000, smoke=True)

    def test_transitive_dependencies_are_hashed_and_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            registration = self.initialize(output)
            for name in ('audit_l2_candidates.py', 'audit_t2.py', 'visualize.py'):
                with self.subTest(source=name):
                    self.assertIn(name, registration['sources_normalized_lf_sha256'])
                    self.assertEqual((output / 'source' / name).read_bytes(), (runner.ROOT / name).read_bytes())

    def test_transitive_source_change_stops_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            self.initialize(output)
            original = runner.sd53.normalized_sha256
            def changed(path):
                return '0' * 64 if Path(path).name == 'visualize.py' else original(path)
            with mock.patch.object(runner.sd53, 'normalized_sha256', side_effect=changed), \
                    mock.patch.object(runner, '_run_jobs', side_effect=AssertionError('unexpected evaluation')), \
                    self.assertRaisesRegex(ValueError, 'source|hash'):
                runner.baseline(output, workers=1)
            self.assertFalse((output / 'baseline.json').exists())


if __name__ == '__main__':
    unittest.main()
