"""SD9.1 probe: does S3 (SD9 CNN) still draw most games against CNNs? No training.

Spec: docs/superpowers/specs/2026-10-23-sd9-1-s3-vs-cnn-probe.md (rules fixed before running).
Learners S3 and S2 (pure CNN, sampled) vs S2, G, S3 and G_search3; 1v1 9x9, schedule E, max_steps 500,
validation seed 21000, 200 boards x 2 corners vs CNNs and 50 boards x 2 corners vs G_search3 (slow),
paired by board between the two learners.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from progress import parallel_map
import run_pde_check as pde
import run_sd5_pilot as sd5
from run_sd78_search_test import summarize

ROOT = Path(__file__).resolve().parent
LEARNERS = ('S3', 'S2')
OPPONENTS = ('S2', 'G', 'S3', 'G_search3')
CNN_OPPONENTS = ('S2', 'G', 'S3')  # macro: each learner meets itself once and the other two
SHA256 = {'G': pde.sd76.G_SHA256, 'S2': 'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814',
          'S3': '4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7'}


def play(learner, opponent, seed, boards, max_steps, shard_path, ticks):
    return pde.play('D', learner, opponent, seed, boards, max_steps, shard_path, ticks)


def _macro(results, learner, metric):
    """One row per (board, corner): metric averaged over the CNN opponents."""
    rows = {}
    for o in CNN_OPPONENTS:
        for g in results[f'{learner}_vs_{o}']:
            rows.setdefault((g['seed'], g['aid']), []).append(g[metric])
    return [{'seed': s, 'aid': a, 'value': float(np.mean(v))} for (s, a), v in rows.items()]


def verdict(macro):
    """Rule fixed in the spec: S3 clearly draws less against CNNs than S2."""
    d, a = macro['draw'], macro['active_win']
    checks = {'draw_down_15pp': d['difference_pp'] <= -15.0, 'draw_ci_upper_negative': d['ci95_pp'][1] < 0,
              'active_up_5pp': a['difference_pp'] >= 5.0, 'active_ci_lower_positive': a['ci95_pp'][0] > 0}
    return {**checks, 's3_fixes_draws': all(checks.values())}


def run(output, seed=21000, boards=200, search_boards=50, workers=5, max_steps=500, shard=5, resume=False):
    for key, sha in SHA256.items():
        if sd5._hash(pde.CKPT[key]) != sha:
            raise ValueError(f'{key} checkpoint changed')
    output = Path(output)
    opened = {'seed': seed, 'boards': boards, 'search_boards': search_boards, 'max_steps': max_steps, 'learners': LEARNERS, 'opponents': OPPONENTS,
              'sha256': SHA256, 'kill_search_sha256': sd5._hash(ROOT / 'kill_search.py')}
    if resume:
        if json.loads((output / 'opened.json').read_text(encoding='utf-8')) != json.loads(json.dumps(opened)):
            raise ValueError('registered design differs from this run; refusing to resume')
    else:
        output.mkdir(parents=True)  # FileExistsError: never overwrite
        (output / 'opened.json').write_text(json.dumps(opened, indent=2), encoding='utf-8')
    (output / 'shards').mkdir(exist_ok=True)
    jobs = [(l, o, seed + first, min(shard, n - first), max_steps,
             str(output / 'shards' / f'{l}_vs_{o}_{seed + first}.json'))
            for l in LEARNERS for o in OPPONENTS for n in [search_boards if o == 'G_search3' else boards]
            for first in range(0, n, shard)]
    todo = sorted((j for j in jobs if not Path(j[-1]).exists()), key=lambda j: j[1] != 'G_search3')  # slow first
    parallel_map(play, todo, total=sum(j[3] * 2 for j in todo), desc=f'SD9.1 seed {seed}', workers=workers)
    results, seconds = {}, {}
    for l, o, *_, path in jobs:
        _, sec, games = json.loads(Path(path).read_text(encoding='utf-8'))
        results.setdefault(f'{l}_vs_{o}', []).extend(games)
        seconds[f'{l}_vs_{o}'] = seconds.get(f'{l}_vs_{o}', 0) + sec
    for rows in results.values():
        rows.sort(key=lambda g: (g['seed'], g['aid']))
    value = lambda rows, m: [{**g, 'value': g[m]} for g in rows]
    comparisons = {f'S3 - S2 vs {o}': {m: pde.paired_bootstrap(value(results[f'S3_vs_{o}'], m),
                                                               value(results[f'S2_vs_{o}'], m))
                                       for m in pde.METRICS} for o in OPPONENTS}
    macro = {m: pde.paired_bootstrap(_macro(results, 'S3', m), _macro(results, 'S2', m)) for m in pde.METRICS}
    summary = {**opened, 'schedule': sd5.SCHEDULE_E,
               'seconds_per_game': {k: v / len(results[k]) for k, v in seconds.items()},
               'results': {k: {**summarize(v), 'loss_rate': 100 * sum(not g['win'] and not g['draw'] for g in v) / len(v),
                               'draw_at_max_steps': 100 * sum(g['draw'] and g['steps'] >= max_steps for g in v) / len(v)}
                           for k, v in results.items()},
               'comparisons': comparisons, 'macro_cnn': macro, 'verdict': verdict(macro)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for k, v in sorted(summary['results'].items()):
        print(f"{k:16s} ActiveWR {v['active_win_rate']:6.2f} WR {v['win_rate']:6.2f} draw {v['draw_rate']:6.2f} "
              f"(timeout {v['draw_at_max_steps']:6.2f}) loss {v['loss_rate']:6.2f} "
              f"s/game {summary['seconds_per_game'][k]:6.1f}", flush=True)
    for k, c in {**comparisons, 'S3 - S2 macro CNN': macro}.items():
        a, d = c['active_win'], c['draw']
        print(f"{k}: ActiveWR {a['difference_pp']:+.2f} [{a['ci95_pp'][0]:+.2f}, {a['ci95_pp'][1]:+.2f}], "
              f"draw {d['difference_pp']:+.2f} [{d['ci95_pp'][0]:+.2f}, {d['ci95_pp'][1]:+.2f}]", flush=True)
    print('verdict', summary['verdict'], flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=21000)
    p.add_argument('--boards', type=int, default=200)
    p.add_argument('--search_boards', type=int, default=50)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--max_steps', type=int, default=500)
    p.add_argument('--resume', action='store_true', help='keep finished shards in OUTPUT/shards, run the rest')
    a = p.parse_args()
    run(a.output, a.seed, a.boards, a.search_boards, a.workers, a.max_steps, resume=a.resume)
