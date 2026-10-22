"""Additional seed-audit regressions derived from full-run preflight requirements."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_s2_kl_support import ROOT, copy_inputs, function, write_json


class HistoricalSeedTests(unittest.TestCase):
    def test_disjoint_single_board_is_not_expanded_by_its_numeric_seed(self):
        register = function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-point-history-') as temporary:
            inputs = copy_inputs(Path(temporary) / 'inputs')
            write_json(inputs / 'runs/old/raw.json', {'board': 620000, 'corner': 0})
            try:
                reg = register(Path(temporary) / 'output', inputs, device='cpu', smoke=False)
            except ValueError as error:
                self.fail('Historical board620000 is disjoint from710000/720000; registration must succeed: ' + str(error))
            self.assertEqual(reg['seed_manifest']['ranges']['train']['start'], 710000)

    def test_unknown_nested_seed_range_is_rejected_before_registration(self):
        register = function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-unknown-history-') as temporary:
            inputs = copy_inputs(Path(temporary) / 'inputs')
            write_json(inputs / 'game UI/runs/nested/metadata.json',
                {'evaluation': {'start_seed': 709950, 'count': 100}})
            output = Path(temporary) / 'output'
            with self.assertRaisesRegex(ValueError, '(?i)(unresolved|seed|mapping)'):
                register(output, inputs, device='cpu', smoke=False)
            self.assertFalse(output.exists())

    def test_known_mechanism_span_still_rejects_offset_overlap(self):
        register = function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-offset-history-') as temporary:
            inputs = copy_inputs(Path(temporary) / 'inputs')
            write_json(inputs / 'runs/old/config.json', {'mechanism_seed': 709990, 'mechanism_games': 20})
            with self.assertRaisesRegex(ValueError, '(?i)(overlap|seed)'):
                register(Path(temporary) / 'output', inputs, device='cpu', smoke=False)

    def test_training_rng_inside_proposed_board_range_is_not_a_board(self):
        register = function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-rng-history-') as temporary:
            inputs = copy_inputs(Path(temporary) / 'inputs')
            write_json(inputs / 'game UI/runs/old/config.json', {'phase': 'C', 'episodes': 3, 'seed': 710003})
            reg = register(Path(temporary) / 'output', inputs, device='cpu', smoke=False)
            self.assertEqual(reg['mode'], 'full')

    def test_inclusive_export_index_rejects_interior_overlap(self):
        register = function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-index-history-') as temporary:
            inputs = copy_inputs(Path(temporary) / 'inputs')
            write_json(inputs / 'game UI/runs/old/metadata.json', {'first_seed': 709999, 'last_seed': 710300})
            with self.assertRaisesRegex(ValueError, '(?i)overlap'):
                register(Path(temporary) / 'output', inputs, device='cpu', smoke=False)

    def test_bootstrap_rng_is_not_an_evaluation_board(self):
        register = function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-bootstrap-history-') as temporary:
            inputs = copy_inputs(Path(temporary) / 'inputs')
            write_json(inputs / 'game UI/runs/old/metadata.json', {'bootstrap_seed': 710005})
            reg = register(Path(temporary) / 'output', inputs, device='cpu', smoke=False)
            self.assertEqual(reg['mode'], 'full')

    def test_malformed_known_seed_span_is_rejected_instead_of_assumed_one_board(self):
        register = function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-invalid-span-') as temporary:
            inputs = copy_inputs(Path(temporary) / 'inputs')
            write_json(inputs / 'game UI/runs/old/metadata.json',
                {'mechanism_seed': 709990, 'mechanism_games': '20'})
            with self.assertRaisesRegex(ValueError, '(?i)(unresolved|seed|count)'):
                register(Path(temporary) / 'output', inputs, device='cpu', smoke=False)


class EquivalentModelTests(unittest.TestCase):
    def test_deduplicated_sha_keeps_both_endpoint_aliases_in_report(self):
        import numpy as np
        from s2_kl_protocol import config_for, seed_manifest, canonical_digest, S2_SHA, SAMPLE_SHA, S2_REL
        from s2_kl_training import model_manifest
        from s2_kl_ledger import _report
        from test_s2_kl_support import model_descriptor
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-equivalent-') as temporary:
            output = Path(temporary)
            reg = dict(output=str(output), config=config_for(True), mode='smoke',
                seed_manifest=seed_manifest(True), sources={'rule_based_bot.py': 'b' * 64},
                registration_digest='a' * 64)
            baseline = model_descriptor(checkpoint=ROOT / S2_REL)
            descriptors = [baseline] + [dict(baseline, id=f'{arm}-{ep}', arm=arm, episode=ep)
                for arm in ('control', 'anchor') for ep in (2, 4)]
            manifest = model_manifest(reg, descriptors)
            self.assertEqual(len(manifest['models']), 1)
            write_json(output / 'model_manifest.json', manifest)
            samples = {kind: dict(ids=[f'{kind}-{i}' for i in range(1024)],
                boards=np.array([210000 + i // 4 for i in range(1024)])) for kind in ('decision', 'anchor')}
            raw = {kind: [dict(id=row_id, board=int(board), kl_s2_to_model=0., bomb_probability=.1, value=0.,
                **(dict(hit=1, mass=1., ce=0.) if kind == 'decision' else {}))
                for row_id, board in zip(s['ids'], s['boards'])] for kind, s in samples.items()}
            diagnostic = dict(format='s2-kl-diagnostics-v1', model_sha256=S2_SHA,
                registration_digest=reg['registration_digest'], sample_manifest_sha256=SAMPLE_SHA,
                raw=raw, summary=dict(R=100., Q=100.))
            diagnostic['diagnostic_digest'] = canonical_digest(diagnostic)
            rows = [dict(learner_sha256=S2_SHA, opponent_id='RB', opponent_sha256='b' * 64,
                board=790002 + i // 2, corner=i % 2, win=0, active_win=0, draw=1, passive_win=0,
                sudden_death_win=0, suicide=0, sudden_death_death=0, kills=0, bombs=0, steps=20,
                opponent_outcome='draw') for i in range(4)]
            saved = {str(i): {'records': rows[i:i + 2]} for i in (0, 2)}
            with patch('s2_kl_evaluation.load_samples', return_value=samples):
                report = _report(reg, manifest['models'], {S2_SHA: diagnostic}, saved)
            self.assertTrue(report['complete'], 'Identical weights still have both trained endpoint identities')
            self.assertEqual(report['status'], 'SMOKE_ONLY')
            self.assertEqual(report['counts']['games'], 4)
            self.assertEqual(report['model_manifest_digest'], manifest['model_manifest_digest'])
            self.assertEqual(report['endpoints']['anchor500']['episode'], 4)


class FinalReviewTests(unittest.TestCase):
    def test_copied_output_cannot_reuse_original_experiment_identity(self):
        import shutil
        from test_s2_kl_support import register_case
        verify = function(self, 'verify_run')
        for smoke in (False, True):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-copy-') as temporary:
                _, output, _ = register_case(self, temporary, smoke=smoke)
                copied = Path(temporary) / 'copied'
                shutil.copytree(output, copied)
                with self.assertRaisesRegex(ValueError, '(?i)(output|identity|experiment)'):
                    verify(copied)

    def test_validation_requires_all_diagnostics_before_any_game_or_projection(self):
        from test_s2_kl_support import ready_case
        validate = function(self, 'validate_run')
        diagnose = function(self, 'diagnose_model')
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-prerequisite-') as temporary:
                _, output, reg, models = ready_case(self, temporary)
                if partial:
                    write_json(output / 'diagnostics' / (models[0]['sha256'] + '.json'), diagnose(models[0], reg))
                with patch('s2_kl_ledger._frozen_job') as worker:
                    with self.assertRaisesRegex(ValueError, '(?i)diagnostic'):
                        validate(output)
                    worker.assert_not_called()
                self.assertFalse((output / 'validation.json').exists())
                self.assertFalse(list((output / 'validation/tasks').glob('*.json')))


if __name__ == '__main__':
    unittest.main()
