"""Independent tests of the frozen D-S2 contract; no experimental seeds run."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import unittest


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / 'run_phase_d_s2.py'
PYTHON = Path(sys.executable)
CHECKPOINTS = {
    'S2': ('runs/sd8a-20261020/train/epoch_16.pt', 'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814'),
    'G': ('runs/sd75-20261018/full/best_phaseC.pt', 'bb0c775c58161aa1bbdd229b9528170aab174d5aabf8f4a6739b848919cdc7ac'),
    'F': ('runs/phaseC-sd-best-fallback/best_phaseC.pt', '0315a56f09079333d936db5f55587093f1b6415ed6355d2ef6ae2ab63f8b1483'),
    'S': ('runs/sd53-20261018/train/epoch_8.pt', '8f7116d290593fa10de1753c4f0f96366e220a193e9860123ed72e0db46bfd4a'),
    'B': ('runs/pressure-C-next/final_phaseC.pt', '93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60'),
}
RATE_KEYS = ('win', 'active_win', 'draw', 'passive_win', 'sudden_death_win', 'suicide', 'sudden_death_death')


def api(test):
    test.assertTrue(RUNNER.is_file(), 'D-S2 public runner run_phase_d_s2.py is missing')
    return importlib.import_module('run_phase_d_s2')


def function(test, name):
    module = api(test)
    value = getattr(module, name, None)
    test.assertTrue(callable(value), f'Missing documented D-S2 public acceptance seam: {name}')
    return value


def raw_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value, digest_key=None):
    value = {k: v for k, v in value.items() if k != digest_key} if digest_key else value
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def copy_inputs(destination):
    """Copy only immutable inputs, never link/mutate the real runs directory."""
    destination = Path(destination)
    for relative, _ in CHECKPOINTS.values():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for name in ('selected.json', 'test_opened.json'):
        target = destination / 'runs/sd8a-20261020' / name
        shutil.copy2(ROOT / 'runs/sd8a-20261020' / name, target)
    return destination


def descriptor(name, episode=None, arm=None, sha=None):
    result = {'id': name, 'checkpoint': f'{name}.pt',
              'sha256': sha or (CHECKPOINTS[name][1] if name in CHECKPOINTS else hashlib.sha256(name.encode()).hexdigest()),
              'episode': episode, 'role': 'candidate' if arm else 'reference'}
    if arm:
        result['arm'] = arm
    return result


def game_row(board=630000, corner=0, opponent='G', learner='pool', win=0, active=0, draw=1, **metrics):
    result = {'learner_sha256': descriptor(learner)['sha256'], 'opponent_id': opponent,
              'opponent_sha256': descriptor(opponent)['sha256'], 'board': board, 'corner': corner,
              **{key: 0 for key in RATE_KEYS}, 'win': win, 'active_win': active, 'draw': draw,
              'kills': active, 'bombs': 1, 'steps': 20,
              'opponent_outcome': 'draw' if draw else ('loss' if win else 'win')}
    result.update(metrics)
    return result


def task_fixture():
    task = {'id': 'task-identity', 'stage': 'test', 'learner': descriptor('pool', 1000, 'pool'),
            'opponent': descriptor('G'), 'board': 630000, 'corners': [0, 1], 'max_steps': 500,
            'registration_digest': 'a' * 64, 'selection_digest': 'b' * 64}
    payload = {'format': 'd-s2-board-result-v1', 'id': task['id'],
               'registration_digest': task['registration_digest'], 'selection_digest': task['selection_digest'],
               'records': [game_row(corner=c) for c in (0, 1)]}
    return task, payload


class ConfigurationContractTests(unittest.TestCase):
    def test_full_scalar_training_config_and_budget_are_frozen(self):
        config = function(self, 'config_for')(smoke=False)
        expected = {'phase': 'D', 'episodes': 2000, 'board_size': 9, 'num_agents': 2, 'max_steps': 500,
                    'gamma': .99, 'lr': 2.5e-5, 'value_coef': .5, 'entropy_coef': .0075,
                    'wall_destroy_bonus': .15, 'snapshot_every': 200, 'seed': 2026102201,
                    'validation_every': 0, 'attack_teacher': 'local', 'teacher_version': 'v2',
                    'normalize_advantage': False, 'advantage_estimator': 'mc', 'policy_objective': 'ac',
                    'sudden_death_probability': 1.0, 'sudden_death_start': 475,
                    'sudden_death_interval': 25, 'sudden_death_warning_steps': 40,
                    'attack_lookahead': 2, 'min_pressure_gain': .15, 'trap_sequence_depth': 2,
                    'sequence_pressure_gain': .35, 'escape_loss_coef': .5, 'combat_loss_coef': .9,
                    'navigation_loss_coef': .4, 'ring_safety_coef': .6, 'late_game_start': 350,
                    'late_game_attack_multiplier': 1.75, 'search_teacher': False, 'group_rl': False,
                    'anchor_coef': 0.0}
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(config['train'].get(key), value)
        self.assertTrue(all(isinstance(v, (str, int, float, bool, type(None))) for v in config['train'].values()))
        self.assertEqual(config['pool_slots'], ['RB', 'G', 'F', 'S', 'S2', 'RB', 'recent', 'recent', 'recent', 'recent'])
        self.assertEqual(config['candidate_episodes'], [0, 500, 1000, 1500, 2000])
        self.assertEqual((config['validation_games'], config['test_games'], config['workers']), (80, 200, 2))

    def test_smoke_has_separate_small_budget(self):
        config = function(self, 'config_for')(smoke=True)
        self.assertEqual(config['train']['episodes'], 4)
        self.assertEqual(config['train']['max_steps'], 20)
        self.assertEqual(config['train']['snapshot_every'], 2)
        self.assertEqual(config['candidate_episodes'], [0, 2, 4])
        self.assertEqual((config['validation_games'], config['test_games']), (2, 2))

    def test_canonical_digest_uses_utf8_sorted_compact_json_and_only_own_digest_is_omitted(self):
        canonical = function(self, 'canonical_digest')
        value = {'z': [1, True, None], 'a': 'bàn cờ', 'digest': 'ignored', 'nested': {'digest': 'keep'}}
        self.assertEqual(canonical(value, digest_key='digest'), digest(value, 'digest'))
        self.assertEqual(canonical(dict(reversed(list(value.items()))), digest_key='digest'), digest(value, 'digest'))
        self.assertNotEqual(canonical({'x': 1}), canonical({'x': 1.0}))

    def test_nonfinite_metadata_is_rejected(self):
        canonical = function(self, 'canonical_digest')
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                canonical({'metric': value})


class BoardPayloadContractTests(unittest.TestCase):
    def test_valid_complete_payload_is_accepted_without_mutation(self):
        validate = function(self, 'validate_task_payload')
        task, payload = task_fixture()
        before = deepcopy(payload)
        validate(task, payload)
        self.assertEqual(before, payload)

    def test_two_corners_and_exact_identity_are_mandatory(self):
        validate = function(self, 'validate_task_payload')
        task, valid = task_fixture()
        cases = []
        for key, value in (('id', 'other'), ('registration_digest', '0' * 64), ('selection_digest', None)):
            case = deepcopy(valid); case[key] = value; cases.append(case)
        for key, value in (('board', 630001), ('corner', 1), ('learner_sha256', '0' * 64),
                           ('opponent_id', 'B'), ('opponent_sha256', '0' * 64)):
            case = deepcopy(valid); case['records'][0][key] = value; cases.append(case)
        case = deepcopy(valid); case['records'].pop(); cases.append(case)
        case = deepcopy(valid); case['records'].append(deepcopy(case['records'][0])); cases.append(case)
        for index, case in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate(task, case)

    def test_impossible_metric_and_outcome_combinations_are_rejected(self):
        validate = function(self, 'validate_task_payload')
        task, valid = task_fixture()
        bad = ({'active_win': 1}, {'win': 1}, {'draw': .5}, {'win': 2}, {'kills': -1},
               {'bombs': float('inf')}, {'steps': float('nan')}, {'steps': 501},
               {'suicide': 1, 'win': 1, 'draw': 0, 'opponent_outcome': 'loss'},
               {'opponent_outcome': 'loss'}, {'sudden_death_win': 1},
               {'passive_win': 1, 'sudden_death_win': 1, 'win': 1, 'draw': 0, 'opponent_outcome': 'loss'})
        for update in bad:
            payload = deepcopy(valid); payload['records'][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate(task, payload)

    def test_sudden_death_win_is_valid_and_is_not_silently_passive_or_active(self):
        validate = function(self, 'validate_task_payload')
        task, payload = task_fixture()
        payload['records'][0].update(win=1, draw=0, sudden_death_win=1, opponent_outcome='loss')
        validate(task, payload)
        self.assertEqual(payload['records'][0]['active_win'], 0)
        self.assertEqual(payload['records'][0]['passive_win'], 0)


if __name__ == '__main__':
    unittest.main()
