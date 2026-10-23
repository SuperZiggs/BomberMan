"""Independent acceptance tests for verified historical S2-KL seed resolution."""
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import s2_kl_protocol as protocol


ROOT = Path(__file__).resolve().parent
ARTIFACTS = {
    'runs/sd6-diagnostics-6600/overrides.json':
        '09f2b9a62205a22aebc880bd523529d1abe2489a642556bf1a2bba94767af4be',
    'runs/t2-full-20261014-060320/replay_cases.json':
        'c0be883584d68b5351fa1885f8ce98b998352345fdf2f1d092ea3eb0e4d05e78',
    'runs/t2-smoke-20261014-060320/replay_cases.json':
        'cd49441f14e02c275235ab28929841ea7d6dadfd92a31dc7a17c7378366984fb',
}
EVIDENCE = 'runs/sd6-diagnostics-6600/override_probe.py'
EVIDENCE_SHA = 'a3c9c532faf965747a4a67d45a6c4f5b22b9bfa2b7db38fa3bcc4039f6c70d17'
PINNED = {**ARTIFACTS, EVIDENCE: EVIDENCE_SHA}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fixture(root, relatives=PINNED):
    for relative in relatives:
        target = Path(root) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return Path(root)


def proposed(seed):
    return {'ranges': {'probe': {'start': seed, 'end_inclusive': seed}}}


class VerifiedHistoricalInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for relative, expected in PINNED.items():
            if sha(ROOT / relative) != expected:
                raise AssertionError('Author fixture identity drift: ' + relative)
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-history-fixture-')
        cls.inputs = fixture(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def accepted_inventory(self, root=None):
        inventory = protocol._inventory(root or self.inputs, ())
        self.assertEqual(inventory['unresolved'], [],
                         'Exact verified history must resolve without suppressing other unknown logs')
        return inventory

    def assert_blocked(self, root):
        inventory = protocol._inventory(root, ())
        self.assertTrue(inventory['unresolved'], 'Unverified historical input must fail closed')
        with self.assertRaisesRegex(ValueError, '(?i)(unresolved|history|seed|index|hash)'):
            protocol._check_history(protocol.seed_manifest(False), inventory, {}, 'acceptance')

    def test_actual_repository_history_accepts_disjoint_full_s2_kl_manifest(self):
        inventory = self.accepted_inventory(ROOT)
        protocol._check_history(protocol.seed_manifest(False), inventory, {}, 'acceptance')

    def test_exact_artifacts_and_generator_are_hashed_and_all_historical_boards_covered(self):
        inventory = self.accepted_inventory()
        for relative, expected in PINNED.items():
            with self.subTest(identity=relative):
                matching = [value for source, value in inventory['sources'].items()
                            if Path(source).resolve() == (self.inputs / relative).resolve()]
                self.assertIn(expected, matching, 'Inventory must retain exact artifact/evidence identity')
        for seed in (*range(6600, 6650), *range(28000, 28004)):
            with self.subTest(seed=seed):
                self.assertTrue(any(a <= seed <= b for a, b, _ in inventory['ranges']),
                                'Historical boards must remain reserved conservatively')
        protocol._check_history(protocol.seed_manifest(False), inventory, {}, 'acceptance')

    def test_resolution_is_read_only_for_original_and_copied_history(self):
        self.accepted_inventory()
        for relative, expected in PINNED.items():
            for base in (ROOT, self.inputs):
                with self.subTest(base=str(base), artifact=relative):
                    self.assertEqual(sha(base / relative), expected)

    def test_resolved_historical_boundaries_and_interior_still_reject_overlap(self):
        inventory = self.accepted_inventory()
        for seed in (6600, 6609, 6610, 6619, 6620, 6629, 6630, 6639, 6649, 28000, 28001, 28003):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, '(?i)overlap'):
                protocol._check_history(proposed(seed), inventory, {}, 'acceptance')

    def test_changed_known_artifact_bytes_fail_closed_even_when_json_is_still_valid(self):
        for relative in ARTIFACTS:
            with self.subTest(artifact=relative), tempfile.TemporaryDirectory(
                    dir=ROOT, prefix='s2-kl-history-drift-') as temporary:
                root = fixture(temporary, (relative, EVIDENCE))
                with (root / relative).open('ab') as handle:
                    handle.write(b' ')
                self.assert_blocked(root)

    def test_missing_or_modified_generator_evidence_fails_closed(self):
        artifact = 'runs/sd6-diagnostics-6600/overrides.json'
        for changed in (False, True):
            with self.subTest(modified=changed), tempfile.TemporaryDirectory(
                    dir=ROOT, prefix='s2-kl-history-evidence-') as temporary:
                root = fixture(temporary, (artifact, EVIDENCE) if changed else (artifact,))
                if changed:
                    with (root / EVIDENCE).open('ab') as handle:
                        handle.write(b'\n# acceptance evidence drift\n')
                self.assert_blocked(root)

    def test_null_seed_in_unrelated_metadata_remains_unresolved(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-history-null-') as temporary:
            path = Path(temporary) / 'runs/unverified/overrides.json'
            path.parent.mkdir(parents=True)
            path.write_text('[{"seed":null,"episodes":20}]', encoding='utf-8')
            self.assert_blocked(Path(temporary))

    def test_unknown_oversized_valid_json_remains_unresolved(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-history-large-') as temporary:
            path = Path(temporary) / 'runs/unverified/replay_cases.json'
            path.parent.mkdir(parents=True)
            with path.open('wb') as handle:
                handle.write(b'{"seed":28000}')
                for _ in range(33):
                    handle.write(b' ' * (1024 * 1024))
            self.assert_blocked(Path(temporary))

    def test_every_executed_local_resolution_module_is_frozen_in_source_closure(self):
        executed = set()
        previous = sys.getprofile()

        def observe(frame, event, arg):
            if event == 'call':
                path = Path(frame.f_code.co_filename)
                if path.is_file():
                    path = path.resolve()
                    if path.parent == ROOT and path.suffix == '.py' and path != Path(__file__).resolve():
                        executed.add(path)

        try:
            sys.setprofile(observe)
            inventory = protocol._inventory(self.inputs, ())
        finally:
            sys.setprofile(previous)
        self.assertEqual(inventory['unresolved'], [], 'Resolver must execute successfully before its closure is audited')
        closure = protocol.source_closure()
        self.assertIn(ROOT / 's2_kl_protocol.py', executed)
        for path in executed:
            with self.subTest(module=path.name):
                self.assertEqual(closure.get(path.name), sha(path),
                                 'Every executed local resolver/dependency must be registered by exact source hash')


if __name__ == '__main__':
    unittest.main()
