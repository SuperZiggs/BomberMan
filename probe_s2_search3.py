"""SD9.0 probe: is three-move kill-search on S2 (SD8A CNN) better than on G? No training.

Spec: docs/superpowers/specs/2026-10-22-sd9-0-s2-search3-probe.md (rules fixed before running).
Learners G_search3 and S2_search3 (search models opponents with RuleBot, as in SD8B) vs RuleBot, S2 and G;
1v1 9x9, schedule E, max_steps 500, validation seed 21000, 50 boards x 2 corners, paired by board.
G_search3 vs RuleBot / vs G reuse the finished SDM2 / SDM shards (same code, boards and 5-board split).
"""
import argparse
import json
from pathlib import Path
import shutil

from progress import parallel_map
import run_pde_check as pde
import run_sd5_pilot as sd5
from run_sd78_search_test import summarize

ROOT = Path(__file__).resolve().parent
LEARNERS = ('G_search3', 'S2_search3')
OPPONENTS = ('rulebot', 'S2', 'G')
S2_SHA256 = 'd222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814'
REUSE = {('G_search3', 'rulebot'): ('sdm2-21000', 'rulebot_vs_rulebot'), ('G_search3', 'G'): ('sdm-21000', 'rulebot_vs_G')}


def play(learner, opponent, seed, boards, max_steps, shard_path, ticks):
    return pde.play('D', learner, opponent, seed, boards, max_steps, shard_path, ticks)


def run(output, seed=21000, boards=50, workers=5, max_steps=500, shard=5, resume=False, runs=ROOT / 'runs'):
    for key, sha in (('G', pde.sd76.G_SHA256), ('S2', S2_SHA256)):
        if sd5._hash(pde.CKPT[key]) != sha:
            raise ValueError(f'{key} checkpoint changed')
    kill_sha = sd5._hash(ROOT / 'kill_search.py')
    output = Path(output)
    opened = {'seed': seed, 'boards': boards, 'max_steps': max_steps, 'learners': LEARNERS, 'opponents': OPPONENTS,
              'G_sha256': pde.sd76.G_SHA256, 'S2_sha256': S2_SHA256, 'kill_search_sha256': kill_sha}
    if resume:
        if json.loads((output / 'opened.json').read_text(encoding='utf-8')) != json.loads(json.dumps(opened)):
            raise ValueError('registered design differs from this run; refusing to resume')
    else:
        output.mkdir(parents=True)  # FileExistsError: never overwrite
        (output / 'opened.json').write_text(json.dumps(opened, indent=2), encoding='utf-8')
    (output / 'shards').mkdir(exist_ok=True)
    jobs = [(l, o, seed + first, min(shard, boards - first), max_steps,
             str(output / 'shards' / f'{l}_vs_{o}_{seed + first}.json'))
            for l in LEARNERS for o in OPPONENTS for first in range(0, boards, shard)]
    sdm2 = Path(runs) / 'sdm2-21000' / 'opened.json'
    same_code = sdm2.exists() and json.loads(sdm2.read_text(encoding='utf-8')).get('kill_search_sha256') == kill_sha
    for l, o, first, n, steps, path in jobs:
        if same_code and (l, o) in REUSE and steps == 500 and n == shard and not Path(path).exists():
            src = Path(runs) / REUSE[l, o][0] / 'shards' / f'{REUSE[l, o][1]}_{first}.json'
            if src.exists():
                shutil.copy(src, path)
    todo = sorted((j for j in jobs if not Path(j[-1]).exists()), key=lambda j: j[1] == 'rulebot')  # slow CNN duels first
    parallel_map(play, todo, total=sum(j[3] * 2 for j in todo), desc=f'SD9.0 seed {seed}', workers=workers)
    results, seconds = {}, {}
    for l, o, *_, path in jobs:
        _, sec, games = json.loads(Path(path).read_text(encoding='utf-8'))
        results.setdefault(f'{l}_vs_{o}', []).extend(games)
        seconds[f'{l}_vs_{o}'] = seconds.get(f'{l}_vs_{o}', 0) + sec
    for rows in results.values():
        rows.sort(key=lambda g: (g['seed'], g['aid']))
    summary = {**opened, 'schedule': sd5.SCHEDULE_E, 'reused': same_code,
               'seconds_per_game': {k: v / len(results[k]) for k, v in seconds.items()},
               'results': {k: {**summarize(v), 'loss_rate': 100 * sum(not g['win'] and not g['draw'] for g in v) / len(v),
                               'draw_at_max_steps': 100 * sum(g['draw'] and g['steps'] >= max_steps for g in v) / len(v)}
                           for k, v in results.items()},
               'comparisons': {f'S2_search3 - G_search3 vs {o}': {
                   m: pde.paired_bootstrap([{**g, 'value': g[m]} for g in results[f'S2_search3_vs_{o}']],
                                           [{**g, 'value': g[m]} for g in results[f'G_search3_vs_{o}']])
                   for m in pde.METRICS} for o in OPPONENTS}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for k, v in sorted(summary['results'].items()):
        print(f"{k:24s} ActiveWR {v['active_win_rate']:6.2f} WR {v['win_rate']:6.2f} draw {v['draw_rate']:6.2f} "
              f"loss {v['loss_rate']:6.2f} s/game {summary['seconds_per_game'][k]:6.1f}", flush=True)
    for k, c in summary['comparisons'].items():
        a = c['active_win']
        print(f"{k}: ActiveWR {a['difference_pp']:+.2f} [{a['ci95_pp'][0]:+.2f}, {a['ci95_pp'][1]:+.2f}], "
              f"draw {c['draw']['difference_pp']:+.2f}", flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=21000)
    p.add_argument('--boards', type=int, default=50)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--max_steps', type=int, default=500)
    p.add_argument('--resume', action='store_true', help='keep finished shards in OUTPUT/shards, run the rest')
    a = p.parse_args()
    run(a.output, a.seed, a.boards, a.workers, a.max_steps, resume=a.resume)
