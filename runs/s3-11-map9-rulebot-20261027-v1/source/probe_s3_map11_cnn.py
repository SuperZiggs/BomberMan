"""Frozen 11x11 CNN benchmark: S3-11 and S3 against S3/S2/G, no training.

Register a fresh seed with init, then run (or resume only missing shards).
See S3_MAP11_CNN_GUIDE.md. Import/help/init do not play games.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from progress import parallel_map
import run_s3_map11_distill as base

ROOT = Path(__file__).resolve().parent
LEARNERS = ('S3_11', 'S3')
OPPONENTS = ('S3', 'S2', 'G')
CHECKPOINTS = {
    'S3_11': ROOT / 'runs/s3-map11-20261024-auto-v1/train/epoch_16.pt',
    'S3': ROOT / 'runs/sd9-20261023/train/epoch_12.pt',
    'S2': ROOT / 'runs/sd8a-20261020/train/epoch_16.pt',
    'G': ROOT / 'runs/sd75-20261018/full/best_phaseC.pt',
}
SHA256 = {
    'S3_11': '17748471874686388f1a9b373b0debdfe53ba04a9b1c821a9695efdd8825a95f',
    'S3': '4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7',
    'S2': 'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814',
    'G': 'bb0c775c58161aa1bbdd229b9528170aab174d5aabf8f4a6739b848919cdc7ac',
}
# Include the existing source closure plus its conditionally imported group-RL
# module, without modifying that historical runner's frozen source list.
SOURCE_FILES = ('probe_s3_map11_cnn.py', *base.SOURCE_FILES, 'group_rl.py')
METRICS = ('win', 'active_win', 'draw', 'suicide', 'passive_win',
           'sudden_death_win', 'sudden_death_death')
COMPARISON_METRICS = (*METRICS, 'loss', 'timeout_draw')


def config_for(output, *, seed, smoke=False):
    boards = 2 if smoke else 200
    if type(seed) is not int or seed < 0 or seed + boards - 1 > np.iinfo(np.uint32).max:
        raise ValueError('seed must be an integer with its full range inside uint32')
    return dict(protocol='S3-11 frozen CNN evaluation', output=str(Path(output).resolve()),
                seed=seed, boards=boards, board_size=11, num_agents=2, obs_channels=13,
                learners=list(LEARNERS), opponents=list(OPPONENTS),
                games_per_pair=boards * 2, total_games=boards * 2 * 6,
                shard_boards=1 if smoke else 10, max_steps=12 if smoke else 500,
                schedule_e=base.SCHEDULE_E.copy(), device='cpu', policy='sampled_with_mask',
                checkpoint_paths={k: str(v) for k, v in CHECKPOINTS.items()},
                checkpoint_sha256=SHA256.copy(), bootstrap_samples=10000,
                bootstrap_seed=20261014, smoke=bool(smoke))


def _sources():
    return {name: base.sd53.normalized_sha256(ROOT / name) for name in SOURCE_FILES}


def _require_checkpoints():
    for name, path in CHECKPOINTS.items():
        if base._hash(path) != SHA256[name]:
            raise ValueError(f'{name} checkpoint hash changed: {path}')


def initialize(output, *, seed, smoke=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = config_for(output, seed=seed, smoke=smoke)
    if not smoke:
        base._fresh_test_seed(seed, config['boards'])
    _require_checkpoints()
    registration = dict(config=config, config_sha256=base._digest(config),
                        sources_normalized_lf_sha256=_sources(), runtime=base.sd53._runtime())
    output.mkdir(parents=True)
    bundle = output / 'source'; bundle.mkdir()
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, bundle / name)
    base.sd53._atomic_json(output / 'registration.json', registration)
    return registration


def _registration(output):
    registration = json.loads((Path(output) / 'registration.json').read_text(encoding='utf-8'))
    config = registration['config']
    expected = config_for(output, seed=config['seed'], smoke=config['smoke'])
    if config != expected or registration.get('config_sha256') != base._digest(config):
        raise ValueError('registered configuration/hash changed; refused')
    if registration.get('sources_normalized_lf_sha256') != _sources():
        raise ValueError('registered source hash changed; refused')
    for name, digest in registration['sources_normalized_lf_sha256'].items():
        archived = Path(output) / 'source' / name
        if not archived.is_file() or base.sd53.normalized_sha256(archived) != digest:
            raise ValueError(f'archived source bundle changed or missing: {archived}')
    _require_checkpoints()
    runtime = base.sd53._runtime()
    if any(registration['runtime'].get(k) != runtime[k] for k in ('python', 'numpy', 'torch')):
        raise ValueError('registered Python/NumPy/PyTorch runtime changed; refused')
    return config


def _jobs(config, output):
    return [(learner, opponent, first, min(first + config['shard_boards'] - 1,
                                          config['seed'] + config['boards'] - 1),
             str(output), config)
            for learner in LEARNERS for opponent in OPPONENTS
            for first in range(config['seed'], config['seed'] + config['boards'], config['shard_boards'])]


def _shard_path(output, learner, opponent, first, last):
    return Path(output) / 'shards' / f'{learner}_vs_{opponent}_{first}-{last}.json'


def _metadata(learner, opponent, first, last, config):
    return dict(learner=learner, opponent=opponent, first_seed=first, last_seed=last,
                board_size=11, num_agents=2, config_sha256=base._digest(config),
                sources_normalized_lf_sha256=_sources(), checkpoint_sha256=SHA256.copy())


def _validate_payload(payload, learner, opponent, first, last, config):
    if payload.get('metadata') != _metadata(learner, opponent, first, last, config):
        raise ValueError('shard metadata mismatch')
    rows = payload['games']
    if not isinstance(rows, list) or sorted((g['seed'], g['aid']) for g in rows) != [
            (seed, aid) for seed in range(first, last + 1) for aid in (0, 1)]:
        raise ValueError('shard count/seed/corner coverage mismatch')
    seconds = payload['seconds']
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('invalid shard duration')
    for row in rows:
        for name in METRICS:
            value = row[name]
            if not isinstance(value, (int, float, bool)) or not math.isfinite(value) or value not in (0, 1):
                raise ValueError(f'invalid scalar game metric {name}')
        for name in ('steps', 'kills', 'bombs', 'items'):
            value = row[name]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or not float(value).is_integer()):
                raise ValueError(f'invalid discrete game counter {name}')
        if not 1 <= row['steps'] <= config['max_steps']:
            raise ValueError('game steps outside registered limit')
        if row['win'] + row['draw'] > 1:
            raise ValueError('game cannot be both a win and a draw')
        if row['active_win'] + row['passive_win'] + row['sudden_death_win'] != row['win']:
            raise ValueError('game win categories must be exclusive and sum to win')
        if row['active_win'] and row['kills'] < 1:
            raise ValueError('active win requires at least one kill')
        if row['win'] and (row['suicide'] or row['sudden_death_death']):
            raise ValueError('a dead learner cannot win')
    return payload


def _read_shard(path, learner, opponent, first, last, config):
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return _validate_payload(payload, learner, opponent, first, last, config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'invalid shard {path}: {error}') from error


def play(learner, opponent, first, last, output, config, ticks):
    import torch
    import evaluate
    torch.set_num_threads(1)
    path = _shard_path(output, learner, opponent, first, last)
    if path.exists():
        raise FileExistsError(f'completed shard will not be overwritten: {path}')
    start = perf_counter()
    for name in (learner, opponent):
        if base._hash(config['checkpoint_paths'][name]) != config['checkpoint_sha256'][name]:
            raise ValueError(f'{name} checkpoint changed before worker load')
    learner_fn = evaluate.make_ac_fn(config['checkpoint_paths'][learner], 11, 'cpu')
    opponent_policy = evaluate.make_ac_fn(config['checkpoint_paths'][opponent], 11, 'cpu')
    done = [0]

    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value

    result = evaluate.evaluate_seeded(learner_fn, board_size=11, num_agents=2,
                                      num_episodes=(last - first + 1) * 2, seed=first,
                                      max_steps=config['max_steps'], sudden_death=True,
                                      opponent_fn=lambda aid: opponent_policy,
                                      on_progress=advance, **config['schedule_e'])
    payload = dict(metadata=_metadata(learner, opponent, first, last, config),
                   seconds=perf_counter()-start, games=result['games'])
    _validate_payload(payload, learner, opponent, first, last, config)
    base.sd53._atomic_json(path, payload)
    return {'path': str(path), 'games': len(payload['games'])}


def _value(row, metric, max_steps):
    if metric == 'loss':
        return float(not row['win'] and not row['draw'])
    if metric == 'timeout_draw':
        return float(row['draw'] and row['steps'] >= max_steps)
    return float(row[metric])


def _summarize(rows, max_steps):
    rates = {f'{m}_rate': 100 * np.mean([_value(g, m, max_steps) for g in rows])
             for m in COMPARISON_METRICS}
    return dict(num_episodes=len(rows), **{k: float(v) for k, v in rates.items()},
                draw_at_max_steps=float(rates['timeout_draw_rate']),
                **{f'avg_{k}': float(np.mean([g[k] for g in rows])) for k in ('kills', 'bombs', 'steps')})


def _paired(left, right, config):
    from pilot_two_bomb import paired_active_bootstrap
    comparisons = {}
    for metric in COMPARISON_METRICS:
        pair = [{'games': [dict(seed=g['seed'], aid=g['aid'],
                               active_win=_value(g, metric, config['max_steps'])) for g in rows]}
                for rows in (left, right)]
        result = paired_active_bootstrap(*pair, samples=config['bootstrap_samples'], seed=config['bootstrap_seed'])
        result['boards'] = result.pop('board_pairs')
        comparisons[metric] = result
    return comparisons


def _macro_rows(results, learner, config):
    grouped = {}
    for opponent in OPPONENTS:
        for row in results[f'{learner}_vs_{opponent}']:
            grouped.setdefault((row['seed'], row['aid']), []).append(row)
    rows = []
    for (seed, aid), games in sorted(grouped.items()):
        if len(games) != len(OPPONENTS):
            raise ValueError('macro pool does not cover every opponent')
        values = {m: float(np.mean([_value(g, m, config['max_steps']) for g in games]))
                  for m in COMPARISON_METRICS}
        rows.append(dict(seed=seed, aid=aid, **values))
    return rows


def _macro_paired(left, right, config):
    from pilot_two_bomb import paired_active_bootstrap
    comparisons = {}
    for metric in COMPARISON_METRICS:
        pair = [{'games': [dict(seed=g['seed'], aid=g['aid'], active_win=g[metric]) for g in rows]}
                for rows in (left, right)]
        result = paired_active_bootstrap(*pair, samples=config['bootstrap_samples'], seed=config['bootstrap_seed'])
        result['boards'] = result.pop('board_pairs')
        comparisons[metric] = result
    return comparisons


def _verdict(macro, smoke=False):
    checks = dict(active_gain_5pp=macro['active_win']['difference_pp'] >= 5,
                   active_ci_lower_positive=macro['active_win']['ci95_pp'][0] > 0,
                   suicide_within_1pp=macro['suicide']['difference_pp'] <= 1)
    return dict(checks, smoke=bool(smoke), **{'pass': not smoke and all(checks.values())})


def _report(output, config, jobs):
    results, seconds, shard_hashes = {}, {}, {}
    for learner, opponent, first, last, _, _ in jobs:
        path = _shard_path(output, learner, opponent, first, last)
        payload = _read_shard(path, learner, opponent, first, last, config)
        key = f'{learner}_vs_{opponent}'
        results.setdefault(key, []).extend(payload['games'])
        seconds[key] = seconds.get(key, 0.0) + payload['seconds']
        shard_hashes[path.name] = base._hash(path)
    expected_pairs = [(s, a) for s in range(config['seed'], config['seed']+config['boards']) for a in (0, 1)]
    for rows in results.values():
        rows.sort(key=lambda g: (g['seed'], g['aid']))
        if [(g['seed'], g['aid']) for g in rows] != expected_pairs:
            raise ValueError('incomplete pair coverage in final report')
    if len(results) != 6 or sum(map(len, results.values())) != config['total_games']:
        raise ValueError('incomplete full benchmark')
    comparisons = {f'S3_11 - S3 vs {o}': _paired(results[f'S3_11_vs_{o}'], results[f'S3_vs_{o}'], config)
                   for o in OPPONENTS}
    macro_rows = {l: _macro_rows(results, l, config) for l in LEARNERS}
    macro = _macro_paired(macro_rows['S3_11'], macro_rows['S3'], config)
    macro_results = {l: {f'{m}_rate': 100*float(np.mean([g[m] for g in rows]))
                          for m in COMPARISON_METRICS} for l, rows in macro_rows.items()}
    return dict(complete=True, smoke=config['smoke'], config_sha256=base._digest(config),
                seed=config['seed'], boards=config['boards'], board_size=11, num_agents=2,
                max_steps=config['max_steps'], schedule_e=config['schedule_e'],
                learners=list(LEARNERS), opponents=list(OPPONENTS), total_games=config['total_games'],
                checkpoint_sha256=SHA256.copy(), sources_normalized_lf_sha256=_sources(),
                results={k: _summarize(v, config['max_steps']) for k, v in results.items()},
                comparisons=comparisons, macro_cnn=macro, macro_results=macro_results,
                verdict=_verdict(macro, smoke=config['smoke']),
                summed_shard_seconds=seconds, seconds_per_game={k: s/len(results[k]) for k, s in seconds.items()},
                shard_sha256=shard_hashes, raw_games=results)


def _marker(config):
    return dict(seed=config['seed'], games=config['games_per_pair'], total_games=config['total_games'],
                smoke=config['smoke'], config_sha256=base._digest(config), checkpoint_sha256=SHA256.copy())


def run(output, workers=5, resume=False):
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError('workers must be an integer from 1 to 5')
    output = Path(output); config = _registration(output)
    lock = output / 'run.lock.json'
    lock_text = json.dumps(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                                config_sha256=base._digest(config)), sort_keys=True)
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(lock_text)
    try:
        marker, shards, summary = output / 'test_opened.json', output / 'shards', output / 'summary.json'
        if not resume:
            if any(path.exists() for path in (marker, shards, summary)):
                raise FileExistsError('benchmark artifacts exist; only validated interrupted runs support --resume')
            if not config['smoke']:
                base._fresh_test_seed(config['seed'], config['boards'])
            with marker.open('x', encoding='utf-8') as handle:
                json.dump(_marker(config), handle, indent=2)
            shards.mkdir()
        elif not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')) != _marker(config):
            raise ValueError('opened marker/config mismatch; resume refused')
        jobs = _jobs(config, output)
        expected = {_shard_path(output, l, o, a, b) for l, o, a, b, *_ in jobs}
        if set(shards.glob('*.json')) - expected:
            raise ValueError('unexpected benchmark shard')
        todo = []
        for learner, opponent, first, last, *_ in jobs:
            path = _shard_path(output, learner, opponent, first, last)
            if path.exists():
                _read_shard(path, learner, opponent, first, last, config)
            else:
                todo.append((learner, opponent, first, last, str(output), config))
        if summary.exists() and todo:
            raise ValueError('completed summary has missing shards; refused')
        if todo:
            parallel_map(play, todo, total=sum((b-a+1)*2 for _, _, a, b, *_ in todo),
                         desc='S3-11 vs CNN', workers=min(workers, len(todo)))
        _registration(output)  # refuse final results if sources/models changed during work
        report = _report(output, config, jobs)
        if summary.exists():
            if json.loads(summary.read_text(encoding='utf-8')) != report:
                raise ValueError('completed summary disagrees with raw shards; refused')
        else:
            base.sd53._atomic_json(summary, report)
        return report
    finally:
        # Never remove a foreign/replaced lock, including a stale operator lock.
        if lock.is_file() and lock.read_text(encoding='utf-8') == lock_text:
            lock.unlink()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    init = stages.add_parser('init'); init.add_argument('--output', type=Path, required=True)
    init.add_argument('--seed', type=int, required=True); init.add_argument('--smoke', action='store_true')
    execute = stages.add_parser('run'); execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--workers', type=int, default=5); execute.add_argument('--resume', action='store_true')
    args = vars(parser.parse_args(argv)); stage = args.pop('stage')
    result = (initialize if stage == 'init' else run)(**args)
    # Raw games are already persisted; keep the console report readable.
    print(json.dumps({k: v for k, v in result.items() if k not in ('raw_games', 'shard_sha256')},
                     indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
