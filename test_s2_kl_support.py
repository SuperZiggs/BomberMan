"""Acceptance-side fixtures for the frozen S2-KL packet; no production helpers."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import unittest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / 'run_s2_kl_probe.py'
PYTHON = Path(sys.executable)
S2_REL = Path('runs/sd8a-20261020/train/epoch_16.pt')
S2_SHA = 'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814'
SAMPLE_REL = Path('runs/phaseD-s2-20261022-analysis-20261023/offline_sample_manifest.json')
SAMPLE_SHA = '9b36b8e07ed8f65175e5c9fbf53a18eca16ae791f0073d19730b28e75529e333'
STAGES = ('register', 'train', 'diagnose', 'validate', 'package')


def function(test, name):
    test.assertTrue(RUNNER.is_file(), 'Missing S2-KL public runner run_s2_kl_probe.py')
    module = importlib.import_module('run_s2_kl_probe')
    value = getattr(module, name, None)
    test.assertTrue(callable(value), 'Missing documented S2-KL public function: ' + name)
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value, key=None):
    if key is not None:
        value = {k: v for k, v in value.items() if k != key}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def copy_inputs(destination):
    """Actual immutable inputs only: 1 S2, metadata, 1 manifest, 60 tiny shards."""
    destination = Path(destination)
    manifest = read_json(ROOT / SAMPLE_REL)
    relatives = [S2_REL, SAMPLE_REL]
    relatives += [Path('runs/sd8a-20261020') / name for name in
                  ('selected.json', 'test_opened.json', 'registration.json', 'collection_summary.json')]
    relatives += [Path(p.replace('\\', '/')) for p in manifest['input_sha256']]
    for relative in relatives:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    if sha(destination / S2_REL) != S2_SHA or sha(destination / SAMPLE_REL) != SAMPLE_SHA:
        raise AssertionError('Author fixture drift: pinned S2/manifest are unavailable or changed')
    for relative, expected in manifest['input_sha256'].items():
        if sha(destination / Path(relative.replace('\\', '/'))) != expected:
            raise AssertionError('Author fixture drift: locked collection shard ' + relative)
    return destination


def register_case(test, base, *, device='cpu', smoke=True, workers=1):
    register = function(test, 'register')
    inputs = copy_inputs(Path(base) / 'inputs')
    output = Path(base) / 'output'
    reg = register(output, inputs, device=device, workers=workers, smoke=smoke)
    test.assertIsInstance(reg, dict)
    return inputs, output, reg


def ready_case(test, base, *, device='cpu'):
    train = function(test, 'train_arm')
    inputs, output, reg = register_case(test, base, device=device)
    for arm in ('control', 'anchor'):
        result = train(output, arm)
        test.assertEqual(result, dict(arm=arm, episode=4, target=4, complete=True))
    models = function(test, 'candidates_for_run')(output, reg)
    return inputs, output, reg, models


def cli(test, *args):
    function(test, 'main')
    return subprocess.run([str(PYTHON), str(RUNNER), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, check=False, timeout=180)


def read_state(output, arm):
    return torch.load(Path(output) / 'arms' / arm / 'training_state.pt',
                      map_location='cpu', weights_only=True)


def metrics(output, arm):
    return [json.loads(line) for line in (Path(output) / 'arms' / arm / 'metrics.jsonl').read_text().splitlines()
            if line.strip()]


def assert_same(test, left, right, path='state'):
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor, path)
        test.assertTrue(torch.equal(left.cpu(), right.cpu()), path)
    elif isinstance(left, np.ndarray):
        test.assertTrue(np.array_equal(left, right), path)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys(), path)
        for key in left:
            assert_same(test, left[key], right[key], path + '.' + str(key))
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right), path)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(test, a, b, path + '[' + str(index) + ']')
    else:
        test.assertEqual(left, right, path)


def rng_snapshot():
    return dict(python=random.getstate(), numpy=deepcopy(np.random.get_state()),
                torch=torch.get_rng_state().clone(),
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def model_descriptor(name='baselineS2', arm=None, episode=0, checkpoint=None, model_sha=S2_SHA):
    return dict(id=name, arm=arm, episode=episode, checkpoint=str(checkpoint or ROOT / S2_REL), sha256=model_sha)


def gate_fixture():
    baseline = dict(R=95., Q=90., AW=80., SU=3.)
    control = dict(R=80., Q=70., AW=70., SU=4.)
    anchor = dict(R=90., Q=85., AW=78., SU=4.)
    comparison = dict(difference_pp=10., ci95_pp=[.1, 20.], board_clusters=10,
                      samples=20, resamples=10000, bootstrap_seed=20261023)
    return baseline, control, anchor, {'anchor_minus_control': {'R': comparison}}
