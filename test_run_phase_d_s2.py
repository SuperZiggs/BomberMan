"""Black-box CLI and real ledger integration, using disposable smoke inputs only."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from test_phase_d_s2_protocol import (ROOT, RUNNER, PYTHON, CHECKPOINTS, api, copy_inputs,
                                     digest, function, game_row, raw_sha, read_json, write_json)
from test_phase_d_s2_training import register_case


def cli(test, *arguments):
    api(test)
    return subprocess.run([str(PYTHON), str(RUNNER), *map(str, arguments)], cwd=ROOT,
                          capture_output=True, text=True, check=False)


def task_payloads(output, stage=None):
    result = {}
    for path in Path(output).rglob('*.json'):
        try:
            payload = read_json(path)
        except (ValueError, OSError):
            continue
        if isinstance(payload, dict) and payload.get('format') == 'd-s2-board-result-v1':
            if stage is None or (stage == 'validation') == (payload.get('selection_digest') is None):
                result[path] = payload
    return result


def games_fingerprint(output):
    files = dict((str(p.relative_to(output)), raw_sha(p)) for p in task_payloads(output))
    for name in ('selected.json', 'test_opened.json', 'report.json'):
        path = Path(output) / name
        if path.is_file():
            files[name] = raw_sha(path)
    return files


def synthetic_result(task):
    """Known validation: episode4 wins, episode2 splits corners, S2 draws.

    Both selected arms tie, so validation opens test. Its pool-control CI is zero,
    a known PILOT_NOT_PASS outcome; no new full PASS claim can come from smoke.
    """
    episode = int(task['learner'].get('episode') or 0)
    records = []
    for corner in task['corners']:
        active = int(episode >= 4 or (episode == 2 and corner == 0))
        records.append({**game_row(board=task['board'], corner=corner, opponent=task['opponent']['id'],
                                   active=active, win=active, draw=1-active, steps=min(task['max_steps'], 10)),
                        'learner_sha256': task['learner']['sha256'],
                        'opponent_sha256': task['opponent']['sha256']})
    return {'format': 'd-s2-board-result-v1', 'id': task['id'],
            'registration_digest': task['registration_digest'], 'selection_digest': task['selection_digest'],
            'records': records}


def ready_case(test, temporary):
    _, output = register_case(test, temporary)
    train = function(test, 'train_arm')
    for arm in ('control', 'pool'):
        train(output, arm)
    return output


def validate_with_fixture(test, output):
    stage = function(test, 'evaluate_stage')
    stage(output, 'validation', evaluator=synthetic_result)
    selected = read_json(Path(output) / 'selected.json')
    test.assertTrue(selected['open_test'], 'Known validation fixture must permit the test journey')
    test.assertEqual(selected['models']['pool']['episode'], 4)
    return selected


class CLIContractTests(unittest.TestCase):
    def test_help_has_all_stages_and_flags_but_no_escape_hatches(self):
        completed = cli(self, '--help')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for stage in ('register', 'train', 'validate', 'test', 'package'):
            self.assertIn(stage, completed.stdout)
            result = cli(self, stage, '--help')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--output', result.stdout)
            for forbidden in ('--force-test', '--ignore-hash', '--reset-seed', '--episodes', '--lr'):
                self.assertNotIn(forbidden, result.stdout)
        register = cli(self, 'register', '--help')
        for option in ('--input-root', '--device', '--workers', '--smoke', '--resume'):
            self.assertIn(option, register.stdout)
        self.assertIn('--arm', cli(self, 'train', '--help').stdout)

    def test_invalid_stage_order_missing_arm_and_budget_override_never_make_games(self):
        api(self)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-cli-') as temporary:
            output = Path(temporary) / 'absent'
            for arguments in (('train',), ('train', '--arm', 'pool'), ('validate',), ('test',),
                              ('train', '--arm', 'pool', '--episodes', '8'), ('test', '--force-test')):
                with self.subTest(arguments=arguments):
                    completed = cli(self, *arguments, '--output', output)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertFalse(task_payloads(output))
                    self.assertFalse((output / 'test_opened.json').exists())

    def test_worker_bounds_are_validated_before_registration(self):
        api(self)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-workers-') as temporary:
            root = copy_inputs(Path(temporary) / 'inputs')
            for workers in ('0', '6'):
                output = Path(temporary) / workers
                result = cli(self, 'register', '--output', output, '--input-root', root,
                             '--device', 'cpu', '--workers', workers, '--smoke')
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((output / 'registration.json').exists())

    def test_package_does_not_implicitly_train_or_evaluate(self):
        function(self, 'train_arm')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-package-') as temporary:
            _, output = register_case(self, temporary)
            before = task_payloads(output)
            result = cli(self, 'package', '--output', output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(before, task_payloads(output))
            self.assertFalse(task_payloads(output))
            self.assertFalse((output / 'test_opened.json').exists())
            self.assertFalse((output / 'selected.json').exists())


class RegistrationIntegrationTests(unittest.TestCase):
    def test_registration_locks_all_public_inputs_and_copies_source_dependency_closure(self):
        function(self, 'train_arm')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-register-') as temporary:
            input_root, output = register_case(self, temporary)
            registration = read_json(output / 'registration.json')
            self.assertEqual(registration['format'], 'd-s2-registration-v1')
            self.assertEqual(registration['mode'], 'smoke')
            self.assertEqual(Path(registration['input_root']).resolve(), input_root.resolve())
            self.assertEqual(registration['registration_digest'], digest(registration, 'registration_digest'))
            for key in ('experiment_id', 'config', 'checkpoints', 'sources', 'spec_sha256', 'runtime', 'git', 'seed_manifest'):
                self.assertIn(key, registration)
            self.assertRegex(registration['spec_sha256'], r'^[0-9a-f]{64}$')
            for name, (_, expected_sha) in CHECKPOINTS.items():
                self.assertEqual(registration['checkpoints'][name]['sha256'], expected_sha)
            self.assertIn('RB', registration['checkpoints'], 'RuleBot needs its registered source descriptor')
            self.assertIsNone(registration['checkpoints']['RB']['checkpoint'])
            self.assertEqual(registration['checkpoints']['RB']['sha256'], raw_sha(ROOT / 'rule_based_bot.py'))
            source_text = json.dumps(registration['sources'])
            for filename in ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'training_state.py',
                             'evaluate.py', 'tactical_teacher.py', 'combat_move_teacher.py', 'progress.py', 'run_phase_d_s2.py'):
                self.assertIn(filename, source_text)
                snapshots = list(output.rglob(filename))
                self.assertTrue(snapshots, f'No frozen source snapshot for {filename}')
                self.assertTrue(any(raw_sha(path) == raw_sha(ROOT / filename) for path in snapshots))
            runtime_keys = json.dumps(registration['runtime']).lower()
            for word in ('python', 'torch', 'numpy', 'cuda', 'cudnn', 'gpu', 'device', 'thread', 'determin', 'tf32'):
                self.assertIn(word, runtime_keys)
            self.assertTrue('os' in runtime_keys or 'platform' in runtime_keys)
            self.assertIn('head', json.dumps(registration['git']).lower())
            self.assertTrue(any(word in json.dumps(registration['git']).lower()
                                for word in ('status', 'tracked', 'dirty')),
                            'Git HEAD alone does not register relevant tracked/untracked working state')
            manifest = read_json(output / 'seed_manifest.json')
            self.assertEqual(manifest, registration['seed_manifest'])
            self.assertEqual(manifest['mode'], 'smoke')
            expected = {'train': (690000, 690001), 'validation': (690002, 690002),
                        'test_reference': (690003, 690003), 'test_snapshots': (690003, 690003)}
            self.assertEqual({k: (v['start'], v['end_inclusive']) for k, v in manifest['ranges'].items()}, expected)
            self.assertFalse(task_payloads(output))

    def test_full_ranges_are_reserved_atomically_and_second_output_is_refused(self):
        api(self)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-registry-') as temporary:
            root = copy_inputs(Path(temporary) / 'inputs')
            output = Path(temporary) / 'first'
            first = cli(self, 'register', '--input-root', root, '--output', output, '--device', 'cpu')
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            registry = root / 'runs/phase_d_s2_seed_registry.json'
            self.assertTrue(registry.is_file())
            before = registry.read_bytes()
            manifest = read_json(output / 'seed_manifest.json')
            expected = {'train': (610000, 610999), 'validation': (620000, 620039),
                        'test_reference': (630000, 630099), 'test_snapshots': (640000, 640099)}
            self.assertEqual({k: (v['start'], v['end_inclusive']) for k, v in manifest['ranges'].items()}, expected)
            registry_text = json.dumps(read_json(registry))
            for first_seed, last_seed in expected.values():
                self.assertIn(str(first_seed), registry_text)
                self.assertIn(str(last_seed), registry_text)
            second = cli(self, 'register', '--input-root', root, '--output', Path(temporary) / 'second', '--device', 'cpu')
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(before, registry.read_bytes())
            self.assertFalse(task_payloads(output))

    def test_two_simultaneous_registrations_cannot_reserve_same_namespace(self):
        api(self)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-race-') as temporary:
            root = copy_inputs(Path(temporary) / 'inputs')
            processes = [subprocess.Popen([str(PYTHON), str(RUNNER), 'register', '--input-root', str(root),
                                          '--output', str(Path(temporary) / name), '--device', 'cpu'],
                                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                         for name in ('one', 'two')]
            results = [(process.communicate(timeout=120), process.returncode) for process in processes]
            self.assertEqual(sum(code == 0 for _, code in results), 1, results)
            read_json(root / 'runs/phase_d_s2_seed_registry.json')

    def test_cross_context_interval_raw_board_and_registry_overlaps_are_rejected(self):
        api(self)
        cases = {
            'interval': ('runs/legacy/registration.json', {'seed_manifest': {'ranges': {
                'collect': {'start': 609999, 'end_inclusive': 610001}}}}),
            'raw_board': ('runs/unrelated-ffa/raw-records.json', [{'context': 'FFA_11', 'board': 620020, 'corner': 0}]),
            'registry': ('runs/pde_seed_registry.json', {'reservations': {'OTHER:630050': {'seed': 630050, 'games': 2}}}),
        }
        for case, (relative, payload) in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-overlap-') as temporary:
                root = copy_inputs(Path(temporary) / 'inputs'); write_json(root / relative, payload)
                output = Path(temporary) / 'output'
                result = cli(self, 'register', '--input-root', root, '--output', output, '--device', 'cpu')
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse((output / 'registration.json').exists())
                self.assertFalse(task_payloads(output))

    def test_unresolved_legacy_seed_metadata_is_not_silently_certified_clean(self):
        api(self)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-unknown-') as temporary:
            root = copy_inputs(Path(temporary) / 'inputs')
            write_json(root / 'runs/unknown/registration.json',
                       {'experiment': 'legacy', 'train': {'episodes': 100}, 'collection': {'seed': None}})
            output = Path(temporary) / 'output'
            result = cli(self, 'register', '--input-root', root, '--output', output, '--device', 'cpu')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((output / 'registration.json').exists())

    def test_checkpoint_and_frozen_source_drift_refuse_stage_before_games(self):
        train = function(self, 'train_arm')
        for kind in ('checkpoint', 'source'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-drift-') as temporary:
                input_root, output = register_case(self, temporary)
                if kind == 'checkpoint':
                    path = input_root / CHECKPOINTS['G'][0]
                else:
                    matches = list(output.rglob('rule_based_bot.py'))
                    self.assertTrue(matches, 'Source snapshot required before training')
                    path = matches[0]
                with path.open('ab') as handle:
                    handle.write(b'\n# independent drift fixture\n')
                before = games_fingerprint(output)
                with self.assertRaises((ValueError, RuntimeError)):
                    train(output, 'control')
                self.assertEqual(before, games_fingerprint(output))

    def test_config_and_actual_runtime_drift_are_rejected(self):
        train = function(self, 'train_arm')
        for kind in ('config', 'runtime'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-drift-') as temporary:
                _, output = register_case(self, temporary)
                before = games_fingerprint(output)
                if kind == 'config':
                    path = output / 'registration.json'; registration = read_json(path)
                    registration['config']['train']['lr'] = .001; write_json(path, registration)
                    with self.assertRaises((ValueError, RuntimeError)):
                        train(output, 'control')
                else:
                    with patch.object(torch, '__version__', '999.invalid-runtime'):
                        with self.assertRaises((ValueError, RuntimeError)):
                            train(output, 'control')
                self.assertEqual(before, games_fingerprint(output))


class EvaluationLedgerIntegrationTests(unittest.TestCase):
    def test_known_validation_and_test_fixture_open_once_then_complete_resume_reuses_exact_report(self):
        stage = function(self, 'evaluate_stage')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-ledger-') as temporary:
            output = ready_case(self, temporary)
            selected = validate_with_fixture(self, output)
            self.assertEqual(selected['selection_digest'], digest(selected, 'selection_digest'))
            self.assertFalse((output / 'test_opened.json').exists())
            stage(output, 'test', evaluator=synthetic_result)
            marker = read_json(output / 'test_opened.json')
            self.assertEqual(marker['selection_digest'], selected['selection_digest'])
            report = read_json(output / 'report.json')
            self.assertTrue(report['complete'])
            self.assertEqual(report['actual_games'], report['expected_games'])
            self.assertNotIn(report['status'], ('PILOT_PASS', 'PHASE_D_PASS'))
            self.assertFalse(report['pilot_gate']['pass'], 'Known pool-control tie has no strictly positive CI')
            before = games_fingerprint(output)
            calls = []
            def should_not_run(task):
                calls.append(task['id']); return synthetic_result(task)
            stage(output, 'test', resume=True, evaluator=should_not_run)
            self.assertFalse(calls)
            self.assertEqual(before, games_fingerprint(output))
            repeated = cli(self, 'test', '--output', output)
            self.assertNotEqual(repeated.returncode, 0)
            resumed = cli(self, 'test', '--output', output, '--resume')
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual(before, games_fingerprint(output))

    def test_validation_stop_is_scientific_result_and_never_opens_test(self):
        stage = function(self, 'evaluate_stage')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-stop-') as temporary:
            output = ready_case(self, temporary)
            def draws(task):
                task = deepcopy(task); task['learner']['episode'] = 0
                return synthetic_result(task)
            stage(output, 'validation', evaluator=draws)
            selected = read_json(output / 'selected.json')
            self.assertFalse(selected['open_test'])
            self.assertEqual(read_json(output / 'report.json')['status'], 'VALIDATION_STOP')
            before = games_fingerprint(output)
            result = cli(self, 'validate', '--output', output, '--resume')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            failed = cli(self, 'test', '--output', output)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(before, games_fingerprint(output))
            self.assertFalse((output / 'test_opened.json').exists())

    def test_interrupted_validation_reuses_only_committed_boards(self):
        stage = function(self, 'evaluate_stage')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-resume-') as temporary:
            output = ready_case(self, temporary)
            calls = []
            def interrupt(task):
                calls.append(task['id'])
                if len(calls) == 3:
                    raise RuntimeError('independent task interruption')
                return synthetic_result(task)
            with self.assertRaisesRegex(RuntimeError, 'independent task interruption'):
                stage(output, 'validation', evaluator=interrupt)
            saved = task_payloads(output, 'validation')
            self.assertEqual(len(saved), 2)
            self.assertFalse((output / 'selected.json').exists(), 'Incomplete validation must not publish selection')
            hashes = {path: raw_sha(path) for path in saved}
            replayed = []
            def resume(task):
                replayed.append(task['id']); return synthetic_result(task)
            stage(output, 'validation', resume=True, evaluator=resume)
            self.assertTrue(set(payload['id'] for payload in saved.values()).isdisjoint(replayed))
            self.assertIn(calls[-1], replayed)
            for path, expected in hashes.items():
                self.assertEqual(raw_sha(path), expected)
            self.assertEqual(len(replayed) + len(saved), len(task_payloads(output, 'validation')))

    def test_before_and_after_atomic_task_commit_have_exactly_once_resume(self):
        stage = function(self, 'evaluate_stage')
        for event in ('before_task_commit', 'after_task_commit'):
            with self.subTest(event=event), tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-atomic-') as temporary:
                output = ready_case(self, temporary)
                injected = []
                def failure(name, task_id):
                    if name == event:
                        injected.append(task_id)
                        raise RuntimeError('independent atomic task interruption')
                with self.assertRaisesRegex(RuntimeError, 'independent atomic task interruption'):
                    stage(output, 'validation', evaluator=synthetic_result, commit_hook=failure)
                saved = task_payloads(output, 'validation')
                self.assertEqual(len(saved), int(event == 'after_task_commit'))
                replayed = []
                def evaluator(task):
                    replayed.append(task['id']); return synthetic_result(task)
                stage(output, 'validation', resume=True, evaluator=evaluator)
                self.assertEqual(injected[0] in replayed, event == 'before_task_commit')

    def test_interrupted_test_preserves_open_marker_and_resumes_same_manifest_only(self):
        stage = function(self, 'evaluate_stage')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-test-resume-') as temporary:
            output = ready_case(self, temporary); validate_with_fixture(self, output)
            calls = []
            def interrupt(task):
                calls.append(task['id'])
                if len(calls) == 3:
                    raise RuntimeError('independent held-out interruption')
                return synthetic_result(task)
            with self.assertRaisesRegex(RuntimeError, 'independent held-out interruption'):
                stage(output, 'test', evaluator=interrupt)
            marker = (output / 'test_opened.json').read_bytes()
            saved = task_payloads(output, 'test')
            self.assertEqual(len(saved), 2)
            report = read_json(output / 'report.json')
            self.assertFalse(report['complete'])
            self.assertLess(report['actual_games'], report['expected_games'])
            reused = {payload['id'] for payload in saved.values()}
            replayed = []
            def evaluator(task):
                replayed.append(task['id']); return synthetic_result(task)
            stage(output, 'test', resume=True, evaluator=evaluator)
            self.assertTrue(reused.isdisjoint(replayed))
            self.assertIn(calls[-1], replayed)
            self.assertEqual((output / 'test_opened.json').read_bytes(), marker)
            report = read_json(output / 'report.json')
            self.assertTrue(report['complete'])
            self.assertEqual(report['actual_games'], report['expected_games'])

    def test_corrupt_or_duplicate_ledger_refuses_before_running_any_new_match(self):
        stage = function(self, 'evaluate_stage')
        for corruption in ('identity', 'duplicate'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-corrupt-') as temporary:
                output = ready_case(self, temporary)
                validate_with_fixture(self, output)
                saved = task_payloads(output, 'validation'); path = next(iter(saved))
                if corruption == 'identity':
                    payload = saved[path]; payload['records'][0]['board'] += 1; write_json(path, payload)
                else:
                    shutil.copy2(path, path.with_name('acceptance-duplicate-' + path.name))
                calls = []
                def evaluator(task):
                    calls.append(task['id']); return synthetic_result(task)
                with self.assertRaises((ValueError, RuntimeError)):
                    stage(output, 'validation', resume=True, evaluator=evaluator)
                self.assertFalse(calls)

    def test_test_open_marker_and_selection_are_immutable(self):
        stage = function(self, 'evaluate_stage')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-once-') as temporary:
            output = ready_case(self, temporary); validate_with_fixture(self, output)
            stage(output, 'test', evaluator=synthetic_result)
            marker_bytes = (output / 'test_opened.json').read_bytes()
            selected = read_json(output / 'selected.json'); selected['models']['pool']['episode'] += 1
            write_json(output / 'selected.json', selected)
            calls = []
            def evaluator(task):
                calls.append(task['id']); return synthetic_result(task)
            with self.assertRaises((ValueError, RuntimeError)):
                stage(output, 'test', resume=True, evaluator=evaluator)
            self.assertFalse(calls)
            self.assertEqual((output / 'test_opened.json').read_bytes(), marker_bytes)

    def test_missing_required_older_snapshot_refuses_before_opening_test(self):
        stage = function(self, 'evaluate_stage')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-snapshot-') as temporary:
            output = ready_case(self, temporary); selected = validate_with_fixture(self, output)
            tasks = selected['test_manifest']
            if isinstance(tasks, dict):
                tasks = tasks['tasks']
            older = [task['opponent'] for task in tasks if task['board'] == 690003
                     and task['opponent'].get('arm') == 'pool' and task['opponent'].get('episode', 0) == 2]
            self.assertTrue(older, 'Selected episode4 must test every older pool snapshot, including episode2')
            checkpoint = Path(older[0]['checkpoint'])
            if not checkpoint.is_absolute():
                checkpoint = output / checkpoint
            self.assertTrue(checkpoint.is_file(), checkpoint)
            checkpoint.unlink()
            calls = []
            def evaluator(task):
                calls.append(task['id']); return synthetic_result(task)
            with self.assertRaises((ValueError, RuntimeError)):
                stage(output, 'test', evaluator=evaluator)
            self.assertFalse(calls)
            self.assertFalse((output / 'test_opened.json').exists())


class CPUSmokeJourneyTests(unittest.TestCase):
    def test_real_cli_smoke_train_validate_package_and_cmd_redirect(self):
        function(self, 'train_arm')
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='d-s2-smoke-') as temporary:
            input_root, output = register_case(self, temporary)
            historical = {relative: raw_sha(input_root / relative) for relative, _ in CHECKPOINTS.values()}
            for arm in ('control', 'pool'):
                result = cli(self, 'train', '--output', output, '--arm', arm)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            logfile = Path(temporary) / 'validation.log'
            # Invoke the user's actual CMD redirection syntax, with stdlib quoting.
            command = subprocess.list2cmdline([str(PYTHON), str(RUNNER), 'validate', '--output', str(output)])
            batch = Path(temporary) / 'validate.cmd'
            batch.write_text('@echo off\n' + command + ' > "' + str(logfile) + '" 2>&1\n', encoding='utf-8')
            redirected = subprocess.run(['cmd.exe', '/d', '/c', str(batch)], cwd=ROOT,
                                        capture_output=True, text=True, check=False)
            diagnostic = logfile.read_text(encoding='utf-8', errors='replace') if logfile.is_file() else redirected.stdout + redirected.stderr
            self.assertEqual(redirected.returncode, 0, diagnostic)
            self.assertTrue(logfile.is_file())
            report = read_json(output / 'report.json')
            if read_json(output / 'selected.json')['open_test']:
                result = cli(self, 'test', '--output', output)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = read_json(output / 'report.json')
            self.assertNotIn(report['status'], ('PILOT_PASS', 'PHASE_D_PASS'))
            for key in ('status', 'complete', 'actual_games', 'expected_games', 'summary', 'comparisons',
                        'pilot_gate', 'phase_d_gate', 'limitations'):
                self.assertIn(key, report)
            for payload in task_payloads(output).values():
                self.assertEqual(len(payload['records']), 2)
                self.assertTrue(all(690000 <= row['board'] <= 690003 for row in payload['records']))
            for relative, before in historical.items():
                self.assertEqual(raw_sha(input_root / relative), before)
            packaged = cli(self, 'package', '--output', output)
            self.assertEqual(packaged.returncode, 0, packaged.stdout + packaged.stderr)
            archives = list(output.rglob('*.zip')) + list(output.parent.glob(output.name + '*.zip'))
            self.assertTrue(archives, 'Package stage must produce a reviewable ZIP')
            with zipfile.ZipFile(archives[0]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertTrue(any(name.endswith('report.json') for name in archive.namelist()))


if __name__ == '__main__':
    unittest.main()
