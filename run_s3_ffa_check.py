"""S3-FFA: the distilled CNN S3 in free-for-all against RuleBots (descriptive, no gate, no model is chosen).

Same protocol as PDE (run_pde_check.py) on a fresh block, seed 59000 (used once):
  E1:  9x9,  3 agents, learner vs 2 RuleBots, 60 boards x 3 seats = 180 games per learner: rulebot, G, S3.
  E2: 11x11, 4 agents, learner vs 3 RuleBots, 40 boards x 4 seats = 160 games per learner:
      rulebot, G and S3 (9x9 weights interpolated, as PDE did for G), and S3_11 (the native 11x11 student).
G is re-run on the same boards so S3 - G is paired; G + search3 stays on PDE's block 50000 (not paired).
Reported, fixed before running: per-arm ActiveWR/WR/draw/self-kill with board-bootstrap CIs, and paired
S3 - G (E1, E2), S3_11 - S3 and S3_11 - G (E2). Run: D:\\anaconda\\envs\\torch\\python.exe run_s3_ffa_check.py --output runs/s3-ffa-59000
"""
import argparse
import json
from pathlib import Path

from progress import parallel_map
import run_pde_check as pde
import run_sd5_pilot as sd5
import run_sd76_pilot as sd76
from run_sd78_search_test import summarize

ROOT = Path(__file__).resolve().parent
SEED, MAX_STEPS, SHARDS = 59000, 500, 4
pde.CKPT['S3_11'] = ROOT / 'runs/s3-map11-20261024-auto-v1/train/epoch_16.pt'   # spawned workers re-import this
SHA = {'G': sd76.G_SHA256, 'S3': '4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7',
       'S3_11': '17748471874686388f1a9b373b0debdfe53ba04a9b1c821a9695efdd8825a95f'}
ARMS = ([('E1', l, 60) for l in ('rulebot', 'G', 'S3')]
        + [('E2', l, 40) for l in ('rulebot', 'G', 'S3', 'S3_11')])
COMPARE = [('E1', 'S3', 'G'), ('E2', 'S3', 'G'), ('E2', 'S3_11', 'S3'), ('E2', 'S3_11', 'G')]


def run(output, workers=5, smoke=False):
    for key, sha in SHA.items():
        if sd5._hash(pde.CKPT[key]) != sha:
            raise ValueError(f'{key} checkpoint changed')
    output = Path(output)
    seed, steps = (21000, 40) if smoke else (SEED, MAX_STEPS)
    arms = [(s, l, 1 if smoke else n) for s, l, n in ARMS]
    output.mkdir(parents=True)  # FileExistsError: never overwrite a finished block
    (output / 'opened.json').write_text(json.dumps({'seed': seed, 'max_steps': steps, 'arms': arms, 'sha256': SHA,
                                                    'schedule': sd5.SCHEDULE_E, 'compare': COMPARE}, indent=2),
                                        encoding='utf-8')
    (output / 'shards').mkdir()
    jobs = []
    for s, l, n in arms:
        k = min(SHARDS, n)
        for i in range(k):
            first, count = seed + i * n // k, (i + 1) * n // k - i * n // k
            jobs.append((s, l, 'rulebot', first, count, steps, str(output / 'shards' / f'{s}_{l}_{first}.json')))
    total = sum(j[4] * pde.SETTINGS[j[0]][1] for j in jobs)
    parts = parallel_map(pde.play, jobs, total=total, desc=f'S3-FFA seed {seed}', workers=workers)
    results = {}
    for name, _, games in parts:
        results.setdefault(name, []).extend(games)
    for name, rows in results.items():
        rows.sort(key=lambda g: (g['seed'], g['aid']))
        (output / f'evaluation_{name}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    comparisons = {f'{s}_{a} - {s}_{b}': {m: pde.paired_bootstrap([{**g, 'value': g[m]} for g in results[f'{s}_{a}']],
                                                                  [{**g, 'value': g[m]} for g in results[f'{s}_{b}']])
                                          for m in pde.METRICS} for s, a, b in COMPARE}
    summary = {'seed': seed, 'max_steps': steps, 'schedule': sd5.SCHEDULE_E,
               'results': {k: summarize(v) for k, v in results.items()}, 'comparisons': comparisons}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: (v['active_win_rate'], v['win_rate'], v['draw_rate'], v['suicide_rate'])
                      for k, v in summary['results'].items()}, indent=1), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(a.output, a.workers, a.smoke)
