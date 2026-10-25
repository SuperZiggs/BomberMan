"""Independent acceptance suite for the frozen S3-map11 CNN packet.

Only the process scheduler is replaced: a single cached smoke executes all
24 real games through the existing loader, two networks and 11x11 environment.
Integrity cases restore that smoke's exact filesystem snapshot at the same
registered output path. No benchmark seed, training, or intended production
implementation is used by this test author.
"""
from contextlib import ExitStack, contextmanager, redirect_stdout
import ast
import builtins
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SEED = 1900410000
LEARNERS = ('S3_11', 'S3')
OPPONENTS = ('S3', 'S2', 'G')
PAIRS = {f'{learner}_vs_{opponent}' for learner in LEARNERS for opponent in OPPONENTS}
CHECKPOINTS = {
    'S3_11': ('runs/s3-map11-20261024-auto-v1/train/epoch_16.pt',
              '17748471874686388f1a9b373b0debdfe53ba04a9b1c821a9695efdd8825a95f'),
    'S3': ('runs/sd9-20261023/train/epoch_12.pt',
           '4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7'),
    'S2': ('runs/sd8a-20261020/train/epoch_16.pt',
           'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814'),
    'G': ('runs/sd75-20261018/full/best_phaseC.pt',
          'bb0c775c58161aa1bbdd229b9528170aab174d5aabf8f4a6739b848919cdc7ac'),
}
METRICS = ('win', 'active_win', 'draw', 'suicide', 'passive_win',
           'sudden_death_win', 'sudden_death_death')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def snapshot(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in Path(folder).rglob('*') if path.is_file()}


def field(mapping, *names):
    """Accept spelling variations; requirements constrain values, not key names."""
    for name in names:
        if name in mapping:
            return mapping[name]
    raise AssertionError(f'missing artifact field, expected one of {names}')


def serial_map(fn, jobs, total, desc, workers):
    return [fn(*job, queue.Queue()) for job in jobs]


@contextmanager
def scheduler(replacement=serial_map):
    import progress
    original = progress.parallel_map
    with ExitStack() as stack:
        for module in tuple(sys.modules.values()):
            if module is not None and vars(module).get('parallel_map') is original:
                stack.enter_context(mock.patch.object(module, 'parallel_map', replacement))
        yield


@contextmanager
def no_game():
    from bomberman_env import BombermanFFAEnv
    with mock.patch.object(BombermanFFAEnv, 'step', side_effect=AssertionError('unexpected game')):
        yield


class CnnMap11Acceptance(unittest.TestCase):
    _cached = None
    _base_temporary = None

    def setUp(self):
        # Every pre-implementation RED must originate here: the new module is missing.
        self.probe = importlib.import_module('probe_s3_map11_cnn')
        self.temporary = tempfile.TemporaryDirectory(prefix='cnn11-acceptance-')
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / 'run'
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    @classmethod
    def tearDownClass(cls):
        if cls._base_temporary is not None:
            cls._base_temporary.cleanup()

    def initialize(self, smoke=True):
        with no_game():
            self.probe.initialize(self.output, seed=SEED, smoke=smoke)
        return read_json(self.output / 'registration.json')

    @contextmanager
    def observe_real_world(self):
        import evaluate
        import kill_search
        import train_actor_critic as trainer
        from bomberman_env import BombermanFFAEnv
        from rule_based_bot import RuleBasedAgent
        seen = dict(envs=[], forwards=[], loads=[], masks=[], samples=0, marker_checks=0)
        env_init, forward = BombermanFFAEnv.__init__, trainer.ActorCriticNet.forward
        step, load, mask = BombermanFFAEnv.step, evaluate.make_ac_fn, trainer.get_action_mask
        sample = torch.distributions.Categorical.sample

        def create(env, *args, **kwargs):
            env_init(env, *args, **kwargs)
            seen['envs'].append((env.board_size, env.num_agents, env.max_steps,
                                 env.sudden_death, kwargs))

        def inference(net, observations):
            seen['forwards'].append((tuple(observations.shape), str(observations.device),
                                     torch.is_grad_enabled(), torch.get_num_threads(), id(net)))
            return forward(net, observations)

        def loading(path, board_size, device='cpu'):
            seen['loads'].append((str(Path(path).resolve()), board_size, str(device)))
            return load(path, board_size, device)

        def masking(env, aid):
            seen['masks'].append(aid)
            return mask(env, aid)

        def sampling(distribution, *args, **kwargs):
            seen['samples'] += 1
            return sample(distribution, *args, **kwargs)

        def advance(env, actions):
            marker = read_json(self.output / 'test_opened.json')
            self.assertEqual((marker['seed'], marker['games'], marker['total_games']), (SEED, 4, 24))
            seen['marker_checks'] += 1
            return step(env, actions)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(BombermanFFAEnv, '__init__', create))
            stack.enter_context(mock.patch.object(BombermanFFAEnv, 'step', advance))
            stack.enter_context(mock.patch.object(trainer.ActorCriticNet, 'forward', inference))
            stack.enter_context(mock.patch.object(evaluate, 'make_ac_fn', loading))
            stack.enter_context(mock.patch.object(trainer, 'get_action_mask', masking))
            stack.enter_context(mock.patch.object(torch.distributions.Categorical, 'sample', sampling))
            stack.enter_context(mock.patch.object(RuleBasedAgent, 'select_action', side_effect=AssertionError('RuleBot in CNN match')))
            for name in ('search_action', 'search_action2', 'search_action3'):
                stack.enter_context(mock.patch.object(kill_search, name, side_effect=AssertionError('search in pure CNN match')))
            stack.enter_context(mock.patch.object(torch.Tensor, 'backward', side_effect=AssertionError('backward in evaluation')))
            stack.enter_context(mock.patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('optimizer in evaluation')))
            stack.enter_context(scheduler())
            yield seen

    def fixture(self):
        cls = type(self)
        if cls._cached is None:
            cls._base_temporary = tempfile.TemporaryDirectory(prefix='cnn11-real24-')
            self.output = Path(cls._base_temporary.name) / 'run'
            self.initialize()
            with self.observe_real_world() as seen, redirect_stdout(io.StringIO()):
                summary = self.probe.run(self.output, workers=1)
            cls._cached = dict(output=self.output, files=snapshot(self.output), seen=seen, summary=summary)
        else:
            self.output = cls._cached['output']
            shutil.rmtree(self.output)
            self.output.mkdir()
            for name, content in cls._cached['files'].items():
                target = self.output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        return cls._cached

    def shard_paths(self):
        paths = sorted((self.output / 'shards').glob('*.json'))
        self.assertEqual(len(paths), 12)
        return paths

    def shard(self, path):
        payload = read_json(path)
        self.assertIsInstance(payload, dict)
        self.assertIsInstance(payload['games'], list)
        return payload

    def metadata(self, payload):
        return payload.get('metadata', payload)

    def refuse_resume_unchanged(self):
        before = snapshot(self.output)
        with no_game(), scheduler(), self.assertRaises((ValueError, FileExistsError, RuntimeError)):
            self.probe.run(self.output, workers=1, resume=True)
        self.assertEqual(snapshot(self.output), before)

    def summary_rebuild(self):
        (self.output / 'summary.json').unlink()
        with no_game(), scheduler(), redirect_stdout(io.StringIO()):
            return self.probe.run(self.output, workers=1, resume=True)

    def test_public_api(self):
        for name in ('config_for', 'initialize', 'run', 'main'):
            self.assertTrue(callable(getattr(self.probe, name)), name)
        self.assertEqual(inspect.signature(self.probe.config_for).parameters['seed'].default, inspect.Parameter.empty)
        self.assertEqual(inspect.signature(self.probe.initialize).parameters['seed'].default, inspect.Parameter.empty)

    def test_import_and_help_do_not_play_or_train(self):
        script = ("import torch\nfrom unittest.mock import patch\n"
                  "from bomberman_env import BombermanFFAEnv\n"
                  "with patch.object(BombermanFFAEnv, 'step', side_effect=AssertionError('game on import')), "
                  "patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('gradient on import')):\n"
                  " import probe_s3_map11_cnn\nprint('SAFE_IMPORT_OK')\n")
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('SAFE_IMPORT_OK', result.stdout)
        with no_game(), redirect_stdout(io.StringIO()) as shown:
            with self.assertRaises(SystemExit) as stopped:
                self.probe.main(['--help'])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn('init', shown.getvalue())
        self.assertIn('run', shown.getvalue())

    def check_config(self, config, smoke):
        self.assertEqual((config['board_size'], config['num_agents'], config['obs_channels']), (11, 2, 13))
        self.assertEqual(tuple(config['learners']), LEARNERS)
        self.assertEqual(tuple(config['opponents']), OPPONENTS)
        self.assertEqual(config['smoke'], smoke)
        self.assertEqual(config['max_steps'], 12 if smoke else 500)
        self.assertEqual(field(config, 'boards', 'evaluation_boards'), 2 if smoke else 200)
        self.assertEqual(field(config, 'shard_boards', 'evaluation_shard_boards'), 1 if smoke else 10)
        self.assertEqual(field(config, 'games_per_pair', 'evaluation_games', 'games'), 4 if smoke else 400)
        self.assertEqual(config['total_games'], 24 if smoke else 2400)
        self.assertEqual(field(config, 'schedule_e', 'schedule'),
                         dict(sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40))
        self.assertEqual(field(config, 'bootstrap_samples', 'bootstrap_resamples'), 10000)
        self.assertEqual(config['bootstrap_seed'], 20261014)

    def test_full_config_is_fixed_11x11_six_pairs(self):
        with no_game():
            self.check_config(self.probe.config_for(self.output, seed=SEED), False)
        self.assertFalse(self.output.exists())

    def test_smoke_config_is_separate_and_bounded(self):
        with no_game():
            self.check_config(self.probe.config_for(self.output, seed=SEED, smoke=True), True)
        self.assertFalse(self.output.exists())

    def test_invalid_seed_types_and_uint32_overflow_refused(self):
        for value in (-1, True, False, 1.2, '123', None, 2**32, 2**32-1, 2**32-199):
            with self.subTest(seed=value), no_game(), self.assertRaises(ValueError):
                self.probe.initialize(self.output, seed=value)
            self.assertFalse(self.output.exists())

    def test_used_reserved_seed_ranges_refused_before_output(self):
        for value in (57000, 57999, 820000, 820199, 819999, 630000, 640000, 720000):
            with self.subTest(seed=value), no_game(), self.assertRaises(ValueError):
                self.probe.initialize(self.output, seed=value)
            self.assertFalse(self.output.exists())

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        (self.output / 'precious.txt').write_text('preserve', encoding='utf-8')
        before = snapshot(self.output)
        with no_game(), self.assertRaises(FileExistsError):
            self.probe.initialize(self.output, seed=SEED, smoke=True)
        self.assertEqual(snapshot(self.output), before)

    def test_registration_contains_frozen_checkpoint_source_bundle_and_runtime(self):
        registration = self.initialize()
        self.assertIn('config_sha256', registration)
        self.assertIn('runtime', registration)
        sources = registration['sources_normalized_lf_sha256']
        for name in ('probe_s3_map11_cnn.py', 'evaluate.py', 'bomberman_env.py',
                     'train_actor_critic.py', 'rule_based_bot.py', 'progress.py'):
            self.assertIn(name, sources)
        for name, digest in sources.items():
            original = (ROOT / name).read_bytes()
            self.assertEqual((self.output / 'source' / name).read_bytes(), original)
            self.assertEqual(digest, hashlib.sha256(original.replace(b'\r\n', b'\n').replace(b'\r', b'\n')).hexdigest())
            # Full local dependency closure, including function-local imports.
            # In particular evaluate -> audit_t2 -> visualize was a known gap
            # in the existing runner's source registration conventions.
            tree = ast.parse(original.decode('utf-8-sig'), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [alias.name.split('.')[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = [node.module.split('.')[0]]
                else:
                    continue
                for module in imported:
                    dependency = module + '.py'
                    if (ROOT / dependency).is_file():
                        self.assertIn(dependency, sources, f'{name} imports unregistered {dependency}')
        encoded = json.dumps(registration)
        for name, (path, digest) in CHECKPOINTS.items():
            self.assertIn(digest, encoded, name)
            self.assertIn(Path(path).name, encoded, name)
        self.assertFalse((self.output / 'test_opened.json').exists())

    def test_changed_config_refused_before_game(self):
        self.initialize()
        path = self.output / 'registration.json'
        data = read_json(path)
        data['config']['board_size'] = 9
        write_json(path, data)
        self.refuse_resume_unchanged()

    def test_changed_registered_source_hash_refused(self):
        self.initialize()
        path = self.output / 'registration.json'
        data = read_json(path)
        data['sources_normalized_lf_sha256']['evaluate.py'] = '0' * 64
        write_json(path, data)
        self.refuse_resume_unchanged()
        # Also exercise actual source-byte drift without editing shared source.
        self.output = Path(self.temporary.name) / 'live-source-run'
        self.initialize()
        target = ROOT / 'evaluate.py'
        original_read = Path.read_bytes

        def source_bytes(path):
            data = original_read(path)
            return data + b'\n# acceptance source drift\n' if path.resolve() == target.resolve() else data

        with mock.patch.object(Path, 'read_bytes', source_bytes):
            self.refuse_resume_unchanged()

    def test_live_checkpoint_change_refused_before_game(self):
        self.initialize()
        target = (ROOT / CHECKPOINTS['S3_11'][0]).resolve()
        original = target.read_bytes()
        corrupted = bytes([original[0] ^ 1]) + original[1:]
        path_open, builtin_open = Path.open, builtins.open

        def intercept_path(path, mode='r', *args, **kwargs):
            if path.resolve() == target and 'r' in mode:
                return io.BytesIO(corrupted)
            return path_open(path, mode, *args, **kwargs)

        def intercept_builtin(path, mode='r', *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target and 'r' in mode:
                return io.BytesIO(corrupted)
            return builtin_open(path, mode, *args, **kwargs)

        with mock.patch.object(Path, 'open', intercept_path), mock.patch.object(builtins, 'open', intercept_builtin):
            self.refuse_resume_unchanged()

    def test_real_smoke_uses_two_cnn_networks_masks_sampling_cpu_no_grad(self):
        seen = self.fixture()['seen']
        self.assertEqual(len(seen['envs']), 24)
        self.assertGreater(seen['marker_checks'], 0)
        self.assertTrue(seen['forwards'])
        self.assertGreaterEqual(len(seen['loads']), 24)
        self.assertEqual({path for path, _, _ in seen['loads']},
                         {str((ROOT / path).resolve()) for path, _ in CHECKPOINTS.values()})
        self.assertTrue(all(board == 11 and device == 'cpu' for _, board, device in seen['loads']))
        self.assertEqual(set(seen['masks']), {0, 1})
        self.assertGreaterEqual(seen['samples'], len(seen['forwards']))
        for shape, device, gradient, threads, _ in seen['forwards']:
            self.assertEqual(shape[1:], (13, 11, 11))
            self.assertEqual(device, 'cpu')
            self.assertFalse(gradient)
            self.assertEqual(threads, 1)
        for board, agents, steps, sudden_death, kwargs in seen['envs']:
            self.assertEqual((board, agents, steps, sudden_death), (11, 2, 12, True))
            self.assertEqual(tuple(kwargs[key] for key in ('sudden_death_start', 'sudden_death_interval', 'sudden_death_warning_steps')), (475, 25, 40))

    def test_marker_compatible_game_counts_and_hashes(self):
        self.fixture()
        marker = read_json(self.output / 'test_opened.json')
        registration = read_json(self.output / 'registration.json')
        self.assertEqual((marker['seed'], marker['games'], marker['total_games'], marker['smoke']), (SEED, 4, 24, True))
        self.assertEqual(marker['config_sha256'], registration['config_sha256'])
        for _, digest in CHECKPOINTS.values():
            self.assertIn(digest, json.dumps(marker))

    def test_raw_and_summary_retain_six_pairs_all_corners_and_metrics(self):
        self.fixture()
        groups = {}
        registration = read_json(self.output / 'registration.json')
        for path in self.shard_paths():
            data = self.shard(path)
            meta = self.metadata(data)
            key = f"{meta['learner']}_vs_{meta['opponent']}"
            self.assertEqual(meta['config_sha256'], registration['config_sha256'])
            self.assertEqual(meta['first_seed'], meta['last_seed'])
            self.assertTrue(any('source' in key for key in meta), 'source provenance required per shard')
            self.assertIn(CHECKPOINTS[meta['learner']][1], json.dumps(meta))
            self.assertIn(CHECKPOINTS[meta['opponent']][1], json.dumps(meta))
            for game in data['games']:
                for metric in METRICS:
                    self.assertIn(game[metric], (0, 1), metric)
                self.assertTrue(1 <= game['steps'] <= 12)
            groups.setdefault(key, []).extend(data['games'])
        self.assertEqual(set(groups), PAIRS)
        for games in groups.values():
            self.assertEqual(sorted((g['seed'], g['aid']) for g in games),
                             [(s, a) for s in (SEED, SEED+1) for a in (0, 1)])
        summary = read_json(self.output / 'summary.json')
        self.assertEqual(set(summary['results']), PAIRS)
        self.assertTrue(summary['smoke'])
        for key, rates in summary['results'].items():
            self.assertEqual(rates['num_episodes'], 4)
            for metric in METRICS:
                self.assertAlmostEqual(rates[f'{metric}_rate'], 25 * sum(g[metric] for g in groups[key]))
            self.assertIn('loss_rate', rates)
            self.assertIn('draw_at_max_steps', rates)
        self.assertFalse((self.output / 'run.lock.json').exists())

    def test_nonresume_refuses_existing_shards_marker_or_summary(self):
        for name in ('shards', 'test_opened.json', 'summary.json'):
            with self.subTest(artifact=name):
                if self.output.exists():
                    shutil.rmtree(self.output)
                self.initialize()
                path = self.output / name
                path.mkdir() if name == 'shards' else path.write_text('{}', encoding='utf-8')
                before = snapshot(self.output)
                with no_game(), scheduler(), self.assertRaises((ValueError, FileExistsError)):
                    self.probe.run(self.output, workers=1)
                self.assertEqual(snapshot(self.output), before)

    def test_shard_wrong_count_refused_without_overwrite(self):
        self.fixture()
        path = self.shard_paths()[0]
        data = self.shard(path)
        data['games'].pop()
        write_json(path, data)
        self.refuse_resume_unchanged()

    def test_duplicate_or_missing_corner_refused_without_overwrite(self):
        for kind in ('duplicate', 'seed', 'corner'):
            with self.subTest(kind=kind):
                self.fixture()
                path = self.shard_paths()[0]
                data = self.shard(path)
                if kind == 'duplicate':
                    data['games'][1] = dict(data['games'][0])
                else:
                    data['games'][0]['seed' if kind == 'seed' else 'aid'] = SEED + 99 if kind == 'seed' else 2
                write_json(path, data)
                self.refuse_resume_unchanged()

    def test_mismatched_metadata_and_metric_shape_refused(self):
        for kind in ('config', 'pair', 'range', 'metric', 'nan'):
            with self.subTest(kind=kind):
                self.fixture()
                path = self.shard_paths()[0]
                data = self.shard(path)
                meta = self.metadata(data)
                if kind == 'config':
                    meta['config_sha256'] = '0' * 64
                elif kind == 'pair':
                    meta['opponent'] = 'RuleBot'
                elif kind == 'range':
                    meta['last_seed'] += 1
                else:
                    data['games'][0]['active_win'] = [1] if kind == 'metric' else float('nan')
                write_json(path, data)
                self.refuse_resume_unchanged()

    def test_resume_runs_only_missing_shard_and_preserves_marker_and_other_shards(self):
        self.fixture()
        path = self.shard_paths()[0]
        (self.output / 'summary.json').unlink()
        path.unlink()
        before = snapshot(self.output)
        seen_jobs = []

        def counted(fn, jobs, total, desc, workers):
            jobs = list(jobs)
            seen_jobs.extend(jobs)
            self.assertEqual(total, 2)
            return serial_map(fn, jobs, total, desc, workers)

        with scheduler(counted), redirect_stdout(io.StringIO()):
            summary = self.probe.run(self.output, workers=1, resume=True)
        self.assertEqual(len(seen_jobs), 1)
        after = snapshot(self.output)
        for name, content in before.items():
            self.assertEqual(after[name], content, name)
        self.assertEqual(set(summary['results']), PAIRS)

    def test_incomplete_scheduler_never_emits_summary_and_releases_own_lock(self):
        self.initialize()
        with scheduler(lambda *args, **kwargs: []), redirect_stdout(io.StringIO()):
            try:
                self.probe.run(self.output, workers=1)
            except (ValueError, FileNotFoundError, RuntimeError):
                pass
        self.assertFalse((self.output / 'summary.json').exists())
        self.assertTrue((self.output / 'test_opened.json').exists())
        self.assertFalse((self.output / 'run.lock.json').exists())

    def test_completed_resume_returns_identical_report_without_write_or_game(self):
        self.fixture()
        before = snapshot(self.output)
        old = read_json(self.output / 'summary.json')
        with no_game(), scheduler(), redirect_stdout(io.StringIO()):
            actual = self.probe.run(self.output, workers=1, resume=True)
        self.assertEqual(actual, old)
        self.assertEqual(snapshot(self.output), before)

    def test_summary_disagreeing_with_raw_refused_without_overwrite(self):
        self.fixture()
        path = self.output / 'summary.json'
        data = read_json(path)
        data['results']['S3_11_vs_S3']['active_win_rate'] += 25
        write_json(path, data)
        self.refuse_resume_unchanged()

    def test_foreign_lock_is_exclusive_and_never_deleted_or_changed(self):
        self.initialize()
        lock = self.output / 'run.lock.json'
        # A nonexistent PID still belongs to the operator: no stale-lock guessing.
        write_json(lock, dict(pid=2147483000, owner='operator-only', created='frozen'))
        before = snapshot(self.output)
        with no_game(), scheduler(), self.assertRaises((FileExistsError, ValueError, RuntimeError)):
            self.probe.run(self.output, workers=1, resume=True)
        self.assertEqual(snapshot(self.output), before)

    def test_worker_bounds_and_failure_release_own_pid_lock(self):
        self.initialize()
        for workers in (0, 6, -1, True, 1.5, '2'):
            with self.subTest(workers=workers), no_game(), self.assertRaises(ValueError):
                self.probe.run(self.output, workers=workers)
            self.assertFalse((self.output / 'run.lock.json').exists())
            self.assertFalse((self.output / 'test_opened.json').exists())

        def fail(*args, **kwargs):
            import os
            self.assertEqual(read_json(self.output / 'run.lock.json')['pid'], os.getpid())
            raise RuntimeError('injected scheduler failure')

        with scheduler(fail), self.assertRaisesRegex(RuntimeError, 'injected scheduler failure'):
            self.probe.run(self.output, workers=1)
        self.assertFalse((self.output / 'run.lock.json').exists())
        self.assertFalse((self.output / 'summary.json').exists())

    def test_macro_bootstrap_pairs_opponents_then_resamples_whole_boards(self):
        self.fixture()
        for path in self.shard_paths():
            data = self.shard(path)
            meta = self.metadata(data)
            for game in data['games']:
                win = int(meta['learner'] == 'S3_11' and
                          (game['seed'] == SEED + 1 or meta['opponent'] == 'S3'))
                game.update(win=win, active_win=win, draw=1-win, suicide=0, passive_win=0,
                            sudden_death_win=0, sudden_death_death=0, kills=win,
                            bombs=1, steps=12)
            write_json(path, data)
        summary = self.summary_rebuild()
        macro = field(summary, 'macro_cnn', 'macro')['active_win']
        self.assertAlmostEqual(macro['difference_pp'], 200/3)
        np.testing.assert_allclose(macro['ci95_pp'], [100/3, 100], rtol=0, atol=1e-9)
        self.assertEqual((macro['boards'], macro['resamples'], macro['bootstrap_seed']), (2, 10000, 20261014))
        comparisons = summary['comparisons']
        self.assertEqual(len(comparisons), 3)
        for opponent in OPPONENTS:
            entry = field(comparisons, f'S3_11 - S3 vs {opponent}', opponent)
            value = entry['active_win']
            self.assertAlmostEqual(value['difference_pp'], 100 if opponent == 'S3' else 50)
            self.assertEqual(value['boards'], 2)
        for key, rates in summary['results'].items():
            self.assertAlmostEqual(rates['draw_at_max_steps'], rates['draw_rate'])
            self.assertEqual(rates['loss_rate'], 0)

    def test_smoke_never_qualifies_as_full_pass_even_with_perfect_gain(self):
        self.fixture()
        for path in self.shard_paths():
            data = self.shard(path)
            win = int(self.metadata(data)['learner'] == 'S3_11')
            for game in data['games']:
                game.update(win=win, active_win=win, draw=1-win, suicide=0, passive_win=0,
                            sudden_death_win=0, sudden_death_death=0, kills=win, bombs=1, steps=12)
            write_json(path, data)
        summary = self.summary_rebuild()
        self.assertAlmostEqual(field(summary, 'macro_cnn', 'macro')['active_win']['difference_pp'], 100)
        verdict = field(summary, 'verdict', 'gate')
        for key in ('pass', 'qualified_gain', 'full_pass'):
            if key in verdict:
                self.assertFalse(verdict[key], f'smoke cannot claim {key}')
        self.assertTrue(any(key in verdict for key in ('pass', 'qualified_gain', 'full_pass')))
        self.assertTrue(summary['smoke'])


if __name__ == '__main__':
    unittest.main()
