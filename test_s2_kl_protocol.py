"""Frozen S2-KL configuration, provenance, seed and transaction acceptance."""
from copy import deepcopy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from test_s2_kl_support import (ROOT, RUNNER, PYTHON, S2_REL, S2_SHA, SAMPLE_REL, SAMPLE_SHA, copy_inputs,
    digest, function, read_json, register_case, sha, write_json)


class ConfigurationTests(unittest.TestCase):
    def test_full_one_variable_protocol_is_frozen(self):
        config = function(self, 'config_for')()
        expected = dict(phase='D', episodes=500, board_size=9, num_agents=2, max_steps=500,
            gamma=.99, lr=2.5e-5, value_coef=.5, entropy_coef=.0075,
            wall_destroy_bonus=.15, seed=2026102301, validation_every=0, run_report=False,
            attack_teacher='local', teacher_version='v2', normalize_advantage=False,
            advantage_estimator='mc', policy_objective='ac', sudden_death_probability=1.,
            sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40,
            attack_lookahead=2, min_pressure_gain=.15, trap_sequence_depth=2,
            sequence_pressure_gain=.35, escape_loss_coef=.5, combat_loss_coef=.9,
            navigation_loss_coef=.4, ring_safety_coef=.6, late_game_start=350,
            late_game_attack_multiplier=1.75, search_teacher=False, group_rl=False,
            advantage_diagnostics=True, reward_diagnostics=False)
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(config['train'].get(key), value)
        self.assertNotIn('anchor_coef', config['train'])
        self.assertEqual(config['anchor_coef'], {'control': 0., 'anchor': 1.})
        self.assertEqual(config['checkpoint_episodes'], [0, 100, 250, 500])
        self.assertEqual((config['endpoint_episode'], config['workers']), (500, 2))
        self.assertTrue(all(isinstance(v, (str, int, float, bool, type(None))) for v in config['train'].values()))

    def test_smoke_keeps_both_coefficients_and_uses_separate_seeds(self):
        config = function(self, 'config_for')(smoke=True)
        self.assertEqual((config['train']['episodes'], config['train']['max_steps']), (4, 20))
        self.assertEqual(config['checkpoint_episodes'], [0, 2, 4])
        self.assertEqual(config['endpoint_episode'], 4)
        self.assertEqual(config['anchor_coef'], {'control': 0., 'anchor': 1.})
        seed_manifest = function(self, 'seed_manifest')
        for smoke, expected in ((False, {'train': (710000, 710249), 'validation': (720000, 720099)}),
                                (True, {'train': (790000, 790001), 'validation': (790002, 790003)})):
            manifest = seed_manifest(smoke=smoke)
            self.assertEqual(set(manifest['ranges']), set(expected))
            self.assertEqual({k: (v['start'], v['end_inclusive']) for k, v in manifest['ranges'].items()}, expected)


class RegistrationTests(unittest.TestCase):
    def test_real_locked_inputs_and_source_closure_are_copied_and_digest_locked(self):
        function(self, 'verify_run')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-input-') as tmp:
            inputs, output, reg = register_case(self, tmp)
            self.assertEqual(reg['format'], 's2-kl-registration-v1')
            self.assertEqual(reg['experiment_id'], 'S2-KL:' + str(output.resolve()))
            self.assertEqual(reg['registration_digest'], digest(reg, 'registration_digest'))
            self.assertEqual(reg['checkpoints']['S2']['sha256'], S2_SHA)
            self.assertEqual(sha(output / 'offline_sample_manifest.json'), SAMPLE_SHA)
            for key in ('runtime', 'sources', 'spec_sha256', 'source_root', 'input_root', 'device', 'seed_manifest'):
                self.assertIn(key, reg)
            serialized = json.dumps(reg)
            for expected in read_json(inputs / SAMPLE_REL)['input_sha256'].values():
                self.assertIn(expected, serialized, 'Every locked shard SHA must be registered')
            for required in ('train_actor_critic.py', 'training_state.py', 'bomberman_env.py',
                             'rule_based_bot.py', 'evaluate.py', 'tactical_teacher.py',
                             'combat_move_teacher.py', 'kill_search.py', 'group_rl.py', 'progress.py'):
                self.assertIn(required, reg['sources'])
            for name, expected in reg['sources'].items():
                self.assertEqual(sha(output / 'sources' / name), expected)
            self.assertEqual(function(self, 'verify_run')(output), reg)
            self.assertFalse((inputs / 'runs/phase_d_s2_seed_registry.json').exists())
            self.assertFalse(any(p.is_symlink() for p in inputs.rglob('*')))

    def test_workers_are_only_one_or_two_and_invalid_device_never_falls_back(self):
        register = function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-bounds-') as tmp:
            inputs = copy_inputs(Path(tmp) / 'inputs')
            for index, kwargs in enumerate(({'workers': 0}, {'workers': 3}, {'workers': True},
                                            {'device': 'gpu'}, {'device': 'cuda:1'})):
                output = Path(tmp) / str(index)
                with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                    register(output, inputs, smoke=True, device='cpu', **kwargs) if 'device' not in kwargs else register(
                        output, inputs, smoke=True, **kwargs)
                self.assertFalse((output / 'registration.json').exists())
            with patch('torch.cuda.is_available', return_value=False), self.assertRaisesRegex(ValueError, '(?i)cuda'):
                register(Path(tmp) / 'cuda', inputs, smoke=True, device='cuda:0')

    def test_wrong_s2_or_selected_identity_is_rejected_before_publication(self):
        register = function(self, 'register')
        for kind in ('checkpoint', 'selected', 'opened', 'manifest', 'shard'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-bad-input-') as tmp:
                inputs = copy_inputs(Path(tmp) / 'inputs'); output = Path(tmp) / 'output'
                if kind == 'checkpoint':
                    with (inputs / S2_REL).open('ab') as f: f.write(b'drift')
                elif kind in ('selected', 'opened'):
                    path = inputs / 'runs/sd8a-20261020' / ('selected.json' if kind == 'selected' else 'test_opened.json')
                    value = read_json(path)
                    if kind == 'selected': value['checkpoint'] = 'runs/other/train/epoch_16.pt'
                    else: value['models']['S2'] = '0' * 64
                    write_json(path, value)
                elif kind == 'manifest':
                    value = read_json(inputs / SAMPLE_REL); value['seed'] += 1; write_json(inputs / SAMPLE_REL, value)
                else:
                    relative = next(iter(read_json(inputs / SAMPLE_REL)['input_sha256']))
                    with (inputs / Path(relative)).open('ab') as f: f.write(b'drift')
                with self.assertRaises(ValueError): register(output, inputs, device='cpu', smoke=True)
                self.assertFalse((output / 'registration.json').exists())

    def test_each_registered_projection_or_source_drift_is_rejected(self):
        verify = function(self, 'verify_run')
        for artifact in ('registration.json', 'runtime.json', 'seed_manifest.json',
                         'offline_sample_manifest.json', 'sources/bomberman_env.py'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-drift-') as tmp:
                _, output, _ = register_case(self, tmp)
                path = output / artifact
                with path.open('ab') as f: f.write(b'\n# tamper' if path.suffix == '.py' else b' ')
                # Metadata bytes are allowed to be reformatted; alter a semantic value as well.
                if path.suffix == '.json':
                    value = read_json(path); value['unexpected_drift'] = True; write_json(path, value)
                with self.assertRaises(ValueError): verify(output)

    def test_runtime_actual_profile_is_checked_even_if_projection_digests_are_recomputed(self):
        verify = function(self, 'verify_run')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-runtime-') as tmp:
            _, output, reg = register_case(self, tmp)
            reg['runtime']['threads'] = 7
            reg['registration_digest'] = digest(reg, 'registration_digest')
            write_json(output / 'registration.json', reg); write_json(output / 'runtime.json', reg['runtime'])
            with self.assertRaisesRegex(ValueError, '(?i)runtime'): verify(output)

    def test_smoke_can_repeat_without_touching_full_registry_and_resume_identity_is_fixed(self):
        register = function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-repeat-') as tmp:
            inputs = copy_inputs(Path(tmp) / 'inputs')
            registry = inputs / 'runs/phase_d_s2_seed_registry.json'
            write_json(registry, {'reservations': {'D-S2:original': {'status': 'COMPLETE', 'ranges': {'old': {'start': 630000, 'end_inclusive': 640099}}}}})
            original = registry.read_bytes()
            for name in ('first', 'second'):
                output = Path(tmp) / name
                reg = register(output, inputs, smoke=True, device='cpu', workers=1)
                self.assertEqual(register(output, inputs, smoke=True, device='cpu', workers=1, resume=True), reg)
                for changed in ({'workers': 2}, {'smoke': False}, {'device': 'cuda:0'}):
                    args = dict(smoke=True, device='cpu', workers=1, resume=True); args.update(changed)
                    with self.subTest(changed=changed), self.assertRaises(ValueError): register(output, inputs, **args)
            self.assertEqual(registry.read_bytes(), original)


class SeedSafetyTests(unittest.TestCase):
    def test_nested_ui_history_raw_ledger_mechanism_and_reservation_conflicts_fail_closed(self):
        register = function(self, 'register')
        cases = [
            ('runs/legacy/raw.jsonl', {'records': [{'board': 710012, 'corner': 0}]}),
            ('game UI/runs/deep/nested/run/metadata.json', {'training': {'board_ids': [720003]}}),
            ('game UI/runs/deep/nested/run/registration.json', {'unknown_seed_scheme': 'unresolved'}),
            ('runs/mechanism/registration.json', {'mechanism_seed': 710000, 'mechanism_games': 20}),
            ('runs/phase_d_s2_seed_registry.json', {'reservations': {'other': {'status': 'PENDING',
                'ranges': {'train': {'start': 710000, 'end_inclusive': 710010}}}}}),
        ]
        for relative, payload in cases:
            with self.subTest(path=relative), tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-seeds-') as tmp:
                inputs = copy_inputs(Path(tmp) / 'inputs'); output = Path(tmp) / 'output'
                path = inputs / relative
                if path.suffix == '.jsonl':
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
                    expected_error = '(?i)overlap'
                else:
                    write_json(path, payload)
                    expected_error = '(?i)(overlap|conflict|unresolved|mapping|seed)'
                with self.assertRaisesRegex(ValueError, expected_error):
                    register(output, inputs, smoke=False, device='cpu')
                self.assertFalse(output.exists())

    def test_source_root_nested_ui_inventory_is_also_checked(self):
        register = function(self, 'register')
        ui_runs = ROOT / 'game UI/runs'
        self.assertTrue(ui_runs.is_dir(), 'Expected existing UI runs root for cross-root acceptance')
        with tempfile.TemporaryDirectory(dir=ui_runs, prefix='s2-kl-acceptance-') as history, tempfile.TemporaryDirectory(
                dir=ROOT, prefix='s2-kl-source-inventory-') as tmp:
            write_json(Path(history) / 'nested/metadata.json', {'records': [{'board': 710000}]})
            inputs = copy_inputs(Path(tmp) / 'inputs')
            with self.assertRaisesRegex(ValueError, '(?i)(overlap|conflict|seed)'):
                register(Path(tmp) / 'output', inputs, smoke=False, device='cpu')


class TransactionTests(unittest.TestCase):
    def test_two_registration_processes_cannot_reserve_the_same_ranges(self):
        function(self, 'register')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-race-') as tmp:
            inputs=copy_inputs(Path(tmp)/'inputs')
            processes=[]
            for index in (1,2):
                processes.append(subprocess.Popen([str(PYTHON),str(RUNNER),'register','--input-root',str(inputs),
                    '--output',str(Path(tmp)/str(index)),'--device','cpu','--workers','1'],cwd=ROOT,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
            outputs=[p.communicate(timeout=120) for p in processes]
            self.assertEqual(sorted(p.returncode for p in processes),[0,1],str(outputs))
            book=read_json(inputs/'runs/phase_d_s2_seed_registry.json')
            self.assertEqual(len(book['reservations']),1)
            self.assertEqual(next(iter(book['reservations'].values()))['status'],'COMPLETE')

    def test_inventory_is_rechecked_under_original_registry_lock(self):
        register=function(self,'register')
        import training_state
        original=training_state.run_lock
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='s2-kl-under-lock-') as tmp:
            inputs=copy_inputs(Path(tmp)/'inputs');output=Path(tmp)/'output'
            registry=inputs/'runs/phase_d_s2_seed_registry.json'
            acquired=[]
            @contextmanager
            def inject(path):
                with original(path):
                    if Path(path)==registry:
                        acquired.append(True)
                        write_json(inputs/'game UI/runs/deep/metadata.json',{'board':710001})
                    yield
            with patch.object(training_state,'run_lock',inject), self.assertRaisesRegex(
                    ValueError,'(?i)(seed|history|inventory|overlap|conflict)'):
                register(output,inputs,smoke=False,device='cpu')
            self.assertTrue(acquired,'Preflight must acquire the original shared registry lock')
            self.assertFalse(output.exists())
            self.assertFalse(registry.exists())

    def test_original_ds2_registry_lock_prevents_new_namespace_reservation(self):
        register = function(self, 'register')
        from training_state import run_lock
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-lock-') as tmp:
            inputs = copy_inputs(Path(tmp) / 'inputs'); output = Path(tmp) / 'output'
            registry = inputs / 'runs/phase_d_s2_seed_registry.json'
            with run_lock(registry), self.assertRaisesRegex(ValueError, '(?i)(active|writer|lock)'):
                register(output, inputs, smoke=False, device='cpu')
            self.assertFalse(output.exists())
            self.assertFalse(registry.exists())

    def test_writer_lock_is_held_for_whole_registration(self):
        register = function(self, 'register')
        from training_state import run_lock
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-writer-') as tmp:
            inputs = copy_inputs(Path(tmp) / 'inputs'); output = Path(tmp) / 'output'
            with run_lock(output), self.assertRaisesRegex(ValueError, '(?i)(active|writer|lock)'):
                register(output, inputs, smoke=True, device='cpu')
            self.assertFalse(output.exists())

    def test_pending_publish_crashes_recover_same_identity_and_preserve_old_entries(self):
        register = function(self, 'register'); verify = function(self, 'verify_run')
        for when in ('before_publish', 'after_publish'):
            with self.subTest(when=when), tempfile.TemporaryDirectory(dir=ROOT, prefix='s2-kl-transaction-') as tmp:
                inputs = copy_inputs(Path(tmp) / 'inputs'); output = Path(tmp) / 'output'
                registry = inputs / 'runs/phase_d_s2_seed_registry.json'
                old = {'status': 'COMPLETE', 'ranges': {'old': {'start': 610000, 'end_inclusive': 640099}},
                       'registration_digest': '7' * 64, 'output': 'untouched-old-experiment'}
                write_json(registry, {'reservations': {'D-S2:old': old}, 'unrelated_metadata': 'keep'})
                replace = os.replace; fired = []
                def interrupted(source, destination):
                    if when == 'before_publish' and Path(destination) == output:
                        fired.append(True); raise RuntimeError('acceptance crash before publish')
                    if when == 'after_publish' and Path(destination) == registry and output.exists():
                        value = read_json(source)
                        if any(e.get('status') == 'COMPLETE' for k, e in value['reservations'].items() if k != 'D-S2:old'):
                            fired.append(True); raise RuntimeError('acceptance crash after publish')
                    return replace(source, destination)
                with patch('os.replace', interrupted), self.assertRaisesRegex(RuntimeError, 'acceptance crash'):
                    register(output, inputs, smoke=False, device='cpu', workers=1)
                self.assertTrue(fired)
                pending = read_json(registry)
                self.assertEqual(pending['reservations']['D-S2:old'], old)
                new = [v for k, v in pending['reservations'].items() if k != 'D-S2:old']
                self.assertEqual(len(new), 1); self.assertEqual(new[0]['status'], 'PENDING')
                if output.exists():
                    with self.assertRaises(ValueError): verify(output)
                    verify(output, allow_pending=True)
                with self.assertRaises(ValueError):
                    register(output, inputs, smoke=False, device='cpu', workers=2, resume=True)
                reg = register(output, inputs, smoke=False, device='cpu', workers=1, resume=True)
                self.assertEqual(verify(output), reg)
                book = read_json(registry)
                self.assertEqual(len(book['reservations']), 2)
                self.assertEqual(book['reservations']['D-S2:old'], old)
                self.assertEqual(book['unrelated_metadata'], 'keep')
                self.assertEqual(book['reservations'][reg['experiment_id']]['status'], 'COMPLETE')


if __name__ == '__main__': unittest.main()
