"""Independent acceptance tests for the frozen S3-map11 step-1 packet.

Bounded: only temporary smoke runs, real 11x11 env/network, synthetic train
records and stubbed search/evaluation for protocol edge cases. No real test
seed is evaluated. Existing SD9 NPZ/JSON artifact conventions are retained.
"""
from contextlib import ExitStack, contextmanager, redirect_stdout
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
S3 = ROOT / 'runs/sd9-20261023/train/epoch_12.pt'
S3_SHA = '4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7'
COLLECT, VALIDATION, TEST = 1900000000, 1900000100, 1900001000


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def snapshot(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in Path(folder).rglob('*') if p.is_file()}


def serial_map(fn, jobs, total, desc, workers):
    """Scheduler seam only: preserve the real worker and its progress queue."""
    return [fn(*job, queue.Queue()) for job in jobs]


@contextmanager
def synchronous(pilot):
    import progress
    original = progress.parallel_map
    with ExitStack() as stack:
        # Existing runners may bind the shared scheduling helper at import.
        for module in tuple(sys.modules.values()):
            if module is not None and vars(module).get('parallel_map') is original:
                stack.enter_context(mock.patch.object(module, 'parallel_map', serial_map))
        yield


class Map11Acceptance(unittest.TestCase):
    def setUp(self):
        # Missing module is intentionally the only missing-behavior RED.
        self.pilot = importlib.import_module('run_s3_map11_distill')
        self.temporary = tempfile.TemporaryDirectory(prefix='s3-map11-acceptance-')
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / 'run'
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)

    def initialize(self, *, smoke=True, output=None, **changes):
        options = dict(collect_first_seed=COLLECT, collect_boards=4,
                       validation_seed=VALIDATION, test_seed=TEST, smoke=smoke)
        options.update(changes)
        self.pilot.initialize(output or self.output, **options)
        return read_json((output or self.output) / 'registration.json')

    def config(self, smoke=False, **changes):
        options = dict(collect_first_seed=COLLECT, collect_boards=100,
                       validation_seed=VALIDATION, test_seed=TEST, smoke=smoke)
        options.update(changes)
        return self.pilot.config_for(self.output, **options)

    @contextmanager
    def no_work(self):
        from bomberman_env import BombermanFFAEnv
        with mock.patch.object(BombermanFFAEnv, 'step', side_effect=AssertionError('unexpected game')), \
                mock.patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('unexpected gradient')):
            yield

    @contextmanager
    def observed_world(self):
        from bomberman_env import BombermanFFAEnv
        from train_actor_critic import ActorCriticNet
        from rule_based_bot import RuleBasedAgent
        original_init, original_forward = BombermanFFAEnv.__init__, ActorCriticNet.forward
        original_bot = RuleBasedAgent.select_action
        seen = {'envs': [], 'inputs': [], 'bots': 0}

        def create(env, *args, **kwargs):
            original_init(env, *args, **kwargs)
            seen['envs'].append((env.board_size, env.num_agents, env.max_steps, kwargs))

        def forward(net, data):
            seen['inputs'].append(tuple(data.shape))
            return original_forward(net, data)

        def bot(agent, env):
            seen['bots'] += 1
            return original_bot(agent, env)

        with mock.patch.object(BombermanFFAEnv, '__init__', create), \
                mock.patch.object(ActorCriticNet, 'forward', forward), \
                mock.patch.object(RuleBasedAgent, 'select_action', bot), synchronous(self.pilot):
            yield seen

    def assert_world(self, seen):
        self.assertTrue(seen['envs'], 'real environment must be constructed')
        self.assertTrue(seen['inputs'], 'real network must receive observations')
        self.assertGreater(seen['bots'], 0, 'RuleBot must execute')
        for board, agents, max_steps, kwargs in seen['envs']:
            self.assertEqual((board, agents, max_steps), (11, 2, 12))
            self.assertEqual(kwargs.get('sudden_death_start'), 475)
            self.assertEqual(kwargs.get('sudden_death_interval'), 25)
            self.assertEqual(kwargs.get('sudden_death_warning_steps'), 40)
        for shape in seen['inputs']:
            self.assertEqual(shape[1:], (13, 11, 11))

    def search(self, mode='mixed', seen=None):
        def choose(env, aid, mask, default):
            if seen is not None:
                seen.append((env.board_size, env.num_agents, env.seed if hasattr(env, 'seed') else None))
            informative = mode == 'decisions' or (mode == 'mixed' and env.step_count % 2 == 0)
            legal = np.flatnonzero(mask).tolist()
            scores = {action: 0.0 for action in legal}
            if informative and len(legal) > 1:
                # Keep the fixture alive and informative; choosing BOMB here
                # can leave a one-legal-action state and accidentally add anchors.
                scores[legal[0]] = 1.0
                return legal[0], {'skipped': False, 'scores': scores, 'plans': {}, 'deep_plans': {}}
            return int(default), {'skipped': True, 'scores': {}, 'plans': {}, 'deep_plans': {}}
        return choose

    def collect_fixture(self, mode='mixed'):
        import kill_search
        with synchronous(self.pilot), mock.patch.object(kill_search, 'search_action3', self.search(mode)):
            return self.pilot.collect(self.output, workers=1)

    def shards(self):
        paths = sorted((self.output / 'collection').glob('shard-*.npz'))
        self.assertTrue(paths, 'collection must persist NPZ shards')
        return paths

    def synthetic_records(self):
        """Keep registered metadata/games; replace records with tiny valid 11x11 data."""
        self.collect_fixture()
        net = self.pilot.load_net(S3)
        for path in self.shards():
            with np.load(path, allow_pickle=False) as old:
                data = {key: old[key].copy() for key in old.files}
            games = json.loads(str(data['games']))
            seeds = sorted({int(game['seed']) for game in games})
            ids = np.repeat(np.asarray(seeds, dtype=np.int64), 2)
            corners = np.tile(np.array([0, 1], np.int8), len(seeds))
            observations = np.zeros((len(ids), 13, 11, 11), np.float32)
            observations[:, 0, 0, 0] = ids - COLLECT + 1
            masks = np.ones((len(ids), 6), bool)
            with torch.no_grad():
                logits = net(torch.from_numpy(observations))[0].numpy()
            for prefix in ('decision', 'anchor'):
                data.update({f'{prefix}_obs': observations.copy(), f'{prefix}_mask': masks.copy(),
                             f'{prefix}_seed': ids.copy(), f'{prefix}_corner': corners.copy(),
                             f'{prefix}_step': np.zeros(len(ids), np.int16)})
            data['decision_scores'] = np.tile(np.arange(6, dtype=np.float32), (len(ids), 1))
            data['decision_sampled_action'] = np.zeros(len(ids), np.int8)
            data['decision_executed_action'] = np.full(len(ids), 5, np.int8)
            data['anchor_logits'] = logits.copy()
            with path.open('wb') as handle:
                np.savez_compressed(handle, **data)

    def candidates(self):
        destination = self.output / 'train'
        destination.mkdir()
        state = self.pilot.load_net(S3).state_dict()
        for epoch in read_json(self.output / 'registration.json')['config']['candidate_epochs']:
            torch.save(state, destination / f'epoch_{epoch}.pt')

    @contextmanager
    def fake_measurements(self, rates):
        """Public evaluation seam; leave model loading/stage selection logic real."""
        import evaluate
        original_load = torch.load
        last = {'model': 'S3'}
        calls = []

        def load(path, *args, **kwargs):
            candidate = Path(path) if isinstance(path, (str, Path)) else None
            name = candidate.stem if candidate is not None else 'S3'
            last['model'] = ('S3' if candidate is None or candidate.resolve() == S3.resolve()
                             else name if name.startswith('epoch_') else 'S3')
            return original_load(path, *args, **kwargs)

        def measure(agent_fn, board_size=9, num_agents=2, num_episodes=100,
                    seed=7000, max_steps=500, **kwargs):
            self.assertEqual((board_size, num_agents), (11, 2))
            model = last['model']
            active, suicide = rates.get(model, rates['S3'])
            calls.append((model, seed, num_episodes))
            rows = []
            for index in range(num_episodes):
                win = int(index < round(active * num_episodes / 100))
                own = int(index >= num_episodes - round(suicide * num_episodes / 100))
                rows.append(dict(seed=seed + index // 2, aid=index % 2, win=win, active_win=win,
                                 draw=1-win, suicide=own, passive_win=0, sudden_death_win=0,
                                 sudden_death_death=0, kills=win, bombs=1, items=0, steps=1))
            return {'games': rows, 'seed': seed, 'board_size': board_size, 'num_agents': num_agents}

        with synchronous(self.pilot), mock.patch.object(torch, 'load', load), \
                mock.patch.object(evaluate, 'evaluate_seeded', measure), redirect_stdout(io.StringIO()):
            yield calls

    def test_public_stage_api(self):
        for name in ('config_for', 'initialize', 'load_net', 'baseline', 'collect', 'train', 'validate', 'test', 'main'):
            self.assertTrue(callable(getattr(self.pilot, name)), name)

    def test_import_is_side_effect_free(self):
        script = ("import torch\nfrom unittest.mock import patch\n"
                  "from bomberman_env import BombermanFFAEnv\n"
                  "with patch.object(BombermanFFAEnv, 'step', side_effect=AssertionError('game on import')), "
                  "patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('gradient on import')):\n"
                  " import run_s3_map11_distill\nprint('SAFE_IMPORT_OK')\n")
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('SAFE_IMPORT_OK', result.stdout)

    def test_help_lists_stages_without_work(self):
        with ExitStack() as stack:
            called = [stack.enter_context(mock.patch.object(self.pilot, name))
                      for name in ('initialize', 'baseline', 'collect', 'train', 'validate', 'test')]
            stream = stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(self.no_work())
            with self.assertRaises(SystemExit) as stopped:
                self.pilot.main(['--help'])
            self.assertEqual(stopped.exception.code, 0)
            for stage in ('init', 'baseline', 'collect', 'train', 'validate', 'test'):
                self.assertIn(stage, stream.getvalue())
            for stage in called:
                stage.assert_not_called()

    def test_cli_dispatches_only_requested_stage(self):
        for stage in ('init', 'baseline', 'collect', 'train', 'validate', 'test'):
            with self.subTest(stage=stage), ExitStack() as stack:
                calls = {name: stack.enter_context(mock.patch.object(self.pilot, name, return_value={}))
                         for name in ('initialize', 'baseline', 'collect', 'train', 'validate', 'test')}
                stack.enter_context(redirect_stdout(io.StringIO()))
                args = [stage, '--output', str(self.output)]
                if stage == 'init':
                    args += ['--collect-first-seed', str(COLLECT), '--test-seed', str(TEST), '--smoke']
                self.pilot.main(args)
                for name, call in calls.items():
                    self.assertEqual(call.call_count, int(name == ('initialize' if stage == 'init' else stage)))

    def test_init_cli_requires_collect_and_test_seed(self):
        for args in ([], ['--collect-first-seed', str(COLLECT)], ['--test-seed', str(TEST)]):
            with self.subTest(args=args), mock.patch.object(self.pilot, 'initialize') as initialize, \
                    mock.patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    self.pilot.main(['init', '--output', str(self.output), *args])
                self.assertNotEqual(stopped.exception.code, 0)
                initialize.assert_not_called()

    def test_full_configuration(self):
        config = self.config()
        expected = dict(board_size=11, num_agents=2, max_steps=500, collect_games=200,
                        baseline_games=100, evaluation_games=400, schedule_e_probability=.75,
                        anchor_probability=.05, learning_rate=1e-4, batch_size=256,
                        train_epochs=16, candidate_epochs=[4, 8, 12, 16], anchor_lambda=1., heldout_fraction=.1,
                        validation_seed=VALIDATION, test_seed=TEST, smoke=False)
        for key, value in expected.items():
            self.assertEqual(config[key], value, key)
        self.assertEqual(config['collect_board_seeds'], list(range(COLLECT, COLLECT + 100)))
        self.assertEqual(config['schedule_e'], dict(sudden_death_start=475, sudden_death_interval=25,
                                                   sudden_death_warning_steps=40))
        self.assertIn(S3_SHA, json.dumps(config))

    def test_smoke_configuration(self):
        config = self.config(smoke=True, collect_boards=4)
        for key, value in dict(board_size=11, num_agents=2, max_steps=12, collect_games=8,
                               baseline_games=4, evaluation_games=4, batch_size=16,
                               train_epochs=2, candidate_epochs=[2], smoke=True).items():
            self.assertEqual(config[key], value, key)

    def test_initialize_records_provenance_without_games_or_gradients(self):
        with self.no_work():
            registration = self.initialize()
        self.assertEqual(registration['config']['board_size'], 11)
        self.assertIn(S3_SHA, json.dumps(registration))
        self.assertTrue(registration['runtime'])
        sources = registration['sources_normalized_lf_sha256']
        for name in ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py',
                     'evaluate.py', 'kill_search.py', 'run_s3_map11_distill.py'):
            expected = hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            self.assertEqual(sources[name], expected)
        self.assertEqual(list(self.output.rglob('*.pt')), [])
        self.assertFalse((self.output / 'test_opened.json').exists())

    def test_initialize_never_overwrites_existing_output(self):
        self.output.mkdir()
        (self.output / 'keep.txt').write_text('unchanged', encoding='utf-8')
        before = snapshot(self.output)
        with self.no_work(), self.assertRaises(FileExistsError):
            self.initialize()
        self.assertEqual(snapshot(self.output), before)

    def test_invalid_seed_ranges_are_rejected(self):
        cases = [dict(collect_first_seed=-1), dict(validation_seed=-1), dict(test_seed=-1),
                 dict(collect_boards=0), dict(collect_boards=1), dict(validation_seed=COLLECT + 3),
                 dict(test_seed=COLLECT + 3), dict(test_seed=VALIDATION + 1),
                 dict(collect_first_seed=VALIDATION + 1), dict(collect_first_seed=TEST + 1)]
        for index, changes in enumerate(cases):
            with self.subTest(changes=changes), self.no_work(), self.assertRaises(ValueError):
                self.initialize(output=Path(self.temporary.name) / f'invalid-{index}', **changes)

    def test_initialize_refuses_known_used_test_seed(self):
        with self.no_work(), self.assertRaisesRegex(ValueError, 'used|opened|fresh|seed|test'):
            self.initialize(smoke=False, test_seed=57000)
        self.assertFalse((self.output / 'test_opened.json').exists())

    def test_load_net_transfers_real_s3_to_11x11(self):
        from train_actor_critic import ActorCriticNet, smart_load_weights
        before = hashlib.sha256(S3.read_bytes()).hexdigest()
        self.assertEqual(before, S3_SHA)
        net = self.pilot.load_net(S3, device='cpu')
        reference = ActorCriticNet(in_channels=13, board_size=11)
        smart_load_weights(reference, str(S3), 'cpu')
        self.assertEqual(net.board_size, 11)
        self.assertEqual(net.conv[0].in_channels, 13)
        for key, expected in reference.state_dict().items():
            self.assertTrue(torch.equal(net.state_dict()[key], expected), key)
        with torch.no_grad():
            logits, value = net(torch.zeros(2, 13, 11, 11))
        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(hashlib.sha256(S3.read_bytes()).hexdigest(), before)

    def test_load_net_rejects_missing_and_invalid_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            self.pilot.load_net(Path(self.temporary.name) / 'missing.pt')
        invalid = Path(self.temporary.name) / 'invalid.pt'
        torch.save({'unrelated': torch.ones(1)}, invalid)
        with self.assertRaises(ValueError):
            self.pilot.load_net(invalid)

    def test_stages_refuse_unregistered_output(self):
        for name in ('baseline', 'collect', 'train', 'validate', 'test'):
            with self.subTest(stage=name), self.no_work(), self.assertRaises((ValueError, FileNotFoundError)):
                getattr(self.pilot, name)(self.output)
        self.assertFalse(self.output.exists())

    def test_registered_source_hash_tampering_is_refused(self):
        registration = self.initialize()
        registration['sources_normalized_lf_sha256']['bomberman_env.py'] = '0' * 64
        (self.output / 'registration.json').write_text(json.dumps(registration), encoding='utf-8')
        with self.no_work(), self.assertRaisesRegex(ValueError, 'source|hash|provenance'):
            self.pilot.baseline(self.output, workers=1)

    def test_registered_checkpoint_hash_tampering_is_refused(self):
        self.initialize()
        path = self.output / 'registration.json'
        original = path.read_text(encoding='utf-8')
        self.assertIn(S3_SHA, original)
        path.write_text(original.replace(S3_SHA, '0' * 64), encoding='utf-8')
        with self.no_work(), self.assertRaisesRegex(ValueError, 'checkpoint|hash|config|registration'):
            self.pilot.baseline(self.output, workers=1)

    def test_registered_configuration_tampering_is_refused(self):
        registration = self.initialize()
        registration['config']['max_steps'] = 13
        (self.output / 'registration.json').write_text(json.dumps(registration), encoding='utf-8')
        with self.no_work(), self.assertRaisesRegex(ValueError, 'config|hash|registration'):
            self.pilot.baseline(self.output, workers=1)

    def test_worker_limits_are_enforced(self):
        self.initialize()
        for stage in ('baseline', 'collect', 'validate', 'test'):
            for workers in (0, 6):
                with self.subTest(stage=stage, workers=workers), self.no_work(), \
                        self.assertRaisesRegex(ValueError, 'worker|1.*5'):
                    getattr(self.pilot, stage)(self.output, workers=workers)

    def test_collection_uses_real_11x11_teacher_and_new_anchor_logits(self):
        import kill_search
        self.initialize()
        search_calls = []
        with self.observed_world() as seen, \
                mock.patch.object(kill_search, 'search_action3', self.search(seen=search_calls)):
            self.pilot.collect(self.output, workers=1)
        self.assert_world(seen)
        self.assertTrue(search_calls)
        self.assertEqual(len(seen['envs']), 8)
        net = self.pilot.load_net(S3)
        pairs, anchors = [], 0
        for path in self.shards():
            with np.load(path, allow_pickle=False) as data:
                self.assertTrue(json.loads(str(data['metadata'])))
                pairs += [(g['seed'], g['aid']) for g in json.loads(str(data['games']))]
                for key in ('decision_obs', 'anchor_obs'):
                    self.assertEqual(data[key].shape[1:], (13, 11, 11))
                if len(data['anchor_obs']):
                    with torch.no_grad():
                        expected = net(torch.from_numpy(data['anchor_obs']).float())[0].numpy()
                    np.testing.assert_allclose(data['anchor_logits'], expected, rtol=1e-5, atol=1e-5)
                    anchors += len(expected)
        self.assertEqual(sorted(pairs), [(seed, corner) for seed in range(COLLECT, COLLECT+4) for corner in (0, 1)])
        self.assertGreater(anchors, 0, 'bounded fixture seeds must yield fresh anchors')

    def test_empty_decision_shards_keep_11x11_shape(self):
        self.initialize()
        self.collect_fixture(mode='anchors')
        for path in self.shards():
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data['decision_obs'].shape, (0, 13, 11, 11))

    def test_empty_anchor_shards_keep_11x11_shape(self):
        self.initialize()
        self.collect_fixture(mode='decisions')
        found_empty = False
        for path in self.shards():
            with np.load(path, allow_pickle=False) as data:
                if not len(data['anchor_obs']):
                    found_empty = True
                    self.assertEqual(data['anchor_obs'].shape, (0, 13, 11, 11))
        self.assertTrue(found_empty, 'informative-only fixture must exercise an empty anchor shard')

    def test_corrupt_shard_is_refused_without_overwrite(self):
        self.initialize()
        self.collect_fixture()
        path = self.shards()[0]
        path.write_bytes(b'not a valid NPZ shard')
        (self.output / 'collection_summary.json').unlink(missing_ok=True)
        before = path.read_bytes()
        with self.no_work(), synchronous(self.pilot), self.assertRaisesRegex(ValueError, 'shard|invalid|corrupt'):
            self.pilot.collect(self.output, workers=1, resume=True)
        self.assertEqual(path.read_bytes(), before)

    def test_resume_collects_only_missing_shards(self):
        import kill_search
        self.initialize()
        self.collect_fixture()
        paths = self.shards()
        self.assertGreaterEqual(len(paths), 2)
        with np.load(paths[-1], allow_pickle=False) as data:
            missing_seeds = {g['seed'] for g in json.loads(str(data['games']))}
        kept = {path: path.read_bytes() for path in paths[:-1]}
        paths[-1].unlink()
        (self.output / 'collection_summary.json').unlink(missing_ok=True)
        actual_seeds = []
        from bomberman_env import BombermanFFAEnv
        original = BombermanFFAEnv.__init__
        def create(env, *args, **kwargs):
            actual_seeds.append(kwargs.get('seed'))
            original(env, *args, **kwargs)
        with synchronous(self.pilot), mock.patch.object(BombermanFFAEnv, '__init__', create), \
                mock.patch.object(kill_search, 'search_action3', self.search()):
            self.pilot.collect(self.output, workers=1, resume=True)
        self.assertEqual(set(actual_seeds), missing_seeds)
        self.assertEqual(len(actual_seeds), len(missing_seeds) * 2)
        for path, content in kept.items():
            self.assertEqual(path.read_bytes(), content)

    def test_completed_collection_is_not_overwritten(self):
        self.initialize()
        self.collect_fixture()
        before = snapshot(self.output / 'collection')
        with self.no_work(), self.assertRaises(FileExistsError):
            self.pilot.collect(self.output, workers=1)
        self.assertEqual(snapshot(self.output / 'collection'), before)

    def test_baseline_uses_real_11x11_env_and_registered_validation_seed(self):
        import kill_search
        self.initialize()
        with self.observed_world() as seen, redirect_stdout(io.StringIO()), \
                mock.patch.object(kill_search, 'search_action3', self.search()):
            self.pilot.baseline(self.output, workers=1)
        self.assert_world(seen)
        self.assertEqual(len(seen['envs']), 8)  # Four per S3 / S3+search3 arm.
        self.assertEqual({entry[3]['seed'] for entry in seen['envs']}, {VALIDATION, VALIDATION+1})
        self.assertTrue(all(entry[3]['sudden_death'] for entry in seen['envs']))

    def test_baseline_is_not_overwritten(self):
        import kill_search
        self.initialize()
        with synchronous(self.pilot), redirect_stdout(io.StringIO()), \
                mock.patch.object(kill_search, 'search_action3', self.search()):
            self.pilot.baseline(self.output, workers=1)
        before = (self.output / 'baseline.json').read_bytes()
        with self.no_work(), self.assertRaises(FileExistsError):
            self.pilot.baseline(self.output, workers=1)
        self.assertEqual((self.output / 'baseline.json').read_bytes(), before)

    def test_train_synthetic_11x11_data_updates_actor_preserves_critic_and_board_split(self):
        import run_sd53_pilot as sd53
        from train_actor_critic import ActorCriticNet
        self.initialize()
        self.synthetic_records()
        initial = self.pilot.load_net(S3).state_dict()
        original = ActorCriticNet.forward
        train_ids, heldout_ids = set(), set()
        def observe(net, data):
            self.assertEqual(tuple(data.shape[1:]), (13, 11, 11))
            ids = set(int(x) for x in data[:, 0, 0, 0].detach().cpu().numpy())
            (train_ids if torch.is_grad_enabled() and net.training else heldout_ids).update(ids)
            return original(net, data)
        with mock.patch.object(ActorCriticNet, 'forward', observe), \
                mock.patch.object(sd53, '_augment_batch', side_effect=AssertionError('9x9 augmentation')), \
                redirect_stdout(io.StringIO()):
            self.pilot.train(self.output, device='cpu')
        candidate = self.output / 'train/epoch_2.pt'
        self.assertTrue(candidate.is_file())
        learned = self.pilot.load_net(candidate).state_dict()
        self.assertTrue(any(not torch.equal(initial[k], learned[k]) for k in initial if k.startswith('actor_head.')))
        for key in initial:
            if key.startswith('critic_head.'):
                self.assertTrue(torch.equal(initial[key], learned[key]), key)
        self.assertTrue(train_ids and heldout_ids)
        self.assertFalse(train_ids & heldout_ids, (train_ids, heldout_ids))
        self.assertEqual(train_ids | heldout_ids, {1, 2, 3, 4})
        self.assertEqual(hashlib.sha256(S3.read_bytes()).hexdigest(), S3_SHA)
        self.assertNotIn('NaN', (self.output / 'train/metrics.json').read_text(encoding='utf-8'))
        before = candidate.read_bytes()
        with self.assertRaises(FileExistsError):
            self.pilot.train(self.output, device='cpu')
        self.assertEqual(candidate.read_bytes(), before)

    def test_train_rejects_9x9_decision_or_anchor_observations(self):
        self.initialize()
        self.synthetic_records()
        path = self.shards()[0]
        with np.load(path, allow_pickle=False) as original:
            valid = {key: original[key].copy() for key in original.files}
        for field in ('decision_obs', 'anchor_obs'):
            data = {key: value.copy() for key, value in valid.items()}
            data[field] = data[field][:, :, :9, :9]
            with path.open('wb') as handle:
                np.savez_compressed(handle, **data)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'shape|11|observation|invalid'):
                self.pilot.train(self.output)
            self.assertFalse(list(self.output.glob('train/*.pt')))

    def test_train_rejects_wrong_action_shape(self):
        self.initialize()
        self.synthetic_records()
        path = self.shards()[0]
        with np.load(path, allow_pickle=False) as original:
            valid = {key: original[key].copy() for key in original.files}
        for field in ('decision_mask', 'anchor_logits'):
            data = {key: value.copy() for key, value in valid.items()}
            data[field] = data[field][:, :5]
            with path.open('wb') as handle:
                np.savez_compressed(handle, **data)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'shape|action|6|invalid'):
                self.pilot.train(self.output)
            self.assertFalse(list(self.output.glob('train/*.pt')))

    def test_validation_selects_highest_safe_activewr_and_earliest_tie(self):
        self.initialize(smoke=False)
        self.candidates()
        rates = {'S3': (50, 1), 'epoch_4': (60, 2), 'epoch_8': (60, 1),
                 'epoch_12': (90, 2.25), 'epoch_16': (55, 0)}
        with self.fake_measurements(rates) as calls:
            self.pilot.validate(self.output, workers=1)
        selected = read_json(self.output / 'selected.json')
        self.assertEqual(selected['epoch'], 4)
        self.assertTrue(calls)
        self.assertTrue(all(seed == VALIDATION and games == 400 for _, seed, games in calls))
        self.assertFalse((self.output / 'test_opened.json').exists())

    def test_validation_without_improvement_keeps_test_closed(self):
        self.initialize()
        self.candidates()
        with self.fake_measurements({'S3': (50, 0), 'epoch_2': (50, 0)}):
            self.pilot.validate(self.output, workers=1)
        with self.fake_measurements({'S3': (50, 0)}) as calls:
            with self.assertRaisesRegex(ValueError, 'validation|improv|baseline|stop|gate'):
                self.pilot.test(self.output, workers=1)
            self.assertEqual(calls, [])
        self.assertFalse((self.output / 'test_opened.json').exists())

    def test_smoke_test_is_labelled_and_cannot_run_twice(self):
        self.initialize()
        self.candidates()
        rates = {'S3': (0, 0), 'epoch_2': (100, 0)}
        with self.fake_measurements(rates):
            self.pilot.validate(self.output, workers=1)
        with self.fake_measurements(rates) as calls:
            self.pilot.test(self.output, workers=1)
        self.assertTrue(calls)
        self.assertTrue(all(seed != TEST for _, seed, _ in calls), 'smoke must not consume registered real test seed')
        summary = read_json(self.output / 'summary.json')
        self.assertIs(summary.get('smoke'), True)
        self.assertTrue((self.output / 'test_opened.json').is_file())
        before = (self.output / 'test_opened.json').read_bytes()
        with self.fake_measurements(rates) as again:
            with self.assertRaises(FileExistsError):
                self.pilot.test(self.output, workers=1)
            self.assertEqual(again, [])
        self.assertEqual((self.output / 'test_opened.json').read_bytes(), before)

    def test_validation_cannot_be_rerun_after_selection_or_test_open(self):
        self.initialize()
        self.candidates()
        path = self.output / 'selected.json'
        path.write_text('{"epoch": 2}', encoding='utf-8')
        with self.no_work(), self.assertRaises(FileExistsError):
            self.pilot.validate(self.output, workers=1)
        self.assertEqual(read_json(path), {'epoch': 2})
        path.unlink()
        (self.output / 'test_opened.json').write_text('{}', encoding='utf-8')
        with self.no_work(), self.assertRaises((ValueError, FileExistsError)):
            self.pilot.validate(self.output, workers=1)


if __name__ == '__main__':
    unittest.main()
