"""PDE: how the strongest agent (G + three-move kill-search) fares in the Phase D/E settings. No training.

Spec: docs/superpowers/specs/2026-10-20-pde-check-design.md (rules fixed before running).

  E2: 11x11, 4 agents, learner vs 3 RuleBots (old Phase E2 setting; G's weights are interpolated 9x9 -> 11x11).
  E1:  9x9,  3 agents, learner vs 2 RuleBots (old Phase E1 setting).
  D:   9x9,  2 agents, learner vs a frozen CNN opponent (Phase D self-play pool: G, F, S, B, old Phase D).
Every learner seat rotates over all corners of each board; all settings use schedule E and board seeds from SEED.
Descriptive: no gate, no model is chosen. Paired bootstrap resamples whole boards (all corners together).
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from progress import parallel_map
import run_sd5_pilot as sd5
import run_sd76_pilot as sd76
from run_sd78_search_test import summarize

ROOT = Path(__file__).resolve().parent
SEED, MAX_STEPS = 50000, 500
CKPT = {'G': sd76.G, 'F': sd5.F, 'S': ROOT / 'runs/sd53-20261018/train/epoch_8.pt',
        'S2': ROOT / 'runs/sd8a-20261020/train/epoch_16.pt',  # SD8A (friend), added after PDE ran
        'S3': ROOT / 'runs/sd9-20261023/train/epoch_12.pt',  # SD9, added after PDE ran
        'B': ROOT / 'runs/pressure-C-next/final_phaseC.pt', 'oldD': ROOT / 'checkpoints/best_phaseD.pt',
        'oldE3': ROOT / 'checkpoints/best_phaseE_3a.pt', 'oldE4': ROOT / 'checkpoints/best_phaseE_4a.pt'}
SETTINGS = {'E2': (11, 4), 'E1': (9, 3), 'D': (9, 2)}
# (setting, learner, opponent, boards); learner/opponent: 'rulebot', a CKPT key (CNN), or 'G_search3'
ARMS = ([('E2', l, 'rulebot', 40) for l in ('rulebot', 'oldE4', 'G', 'G_search3')]
        + [('E1', l, 'rulebot', 60) for l in ('rulebot', 'oldE3', 'G', 'G_search3')]
        + [('D', l, o, 100) for o in ('G', 'F', 'S', 'B', 'oldD') for l in ('G', 'G_search3')])
SHARDS = {'G_search3': 10}  # slow arms are split by board (~190 s/game in E2); others run whole
COST = {'E2': 190, 'E1': 124, 'D': 30}  # measured s/game of G_search3, used only to start slow jobs first
COMPARE = ([(s, a, b) for s in ('E2', 'E1') for a, b in
            (('G_search3', 'G'), ('G', 'rulebot'), ('G_search3', 'rulebot'), ('G', f'old{"E4" if s == "E2" else "E3"}'))]
           + [('D', 'G_search3', 'G')])
METRICS = ('active_win', 'win', 'draw', 'passive_win', 'suicide')


def arm_name(setting, learner, opponent):
    return f'{setting}_{learner}' + (f'_vs_{opponent}' if setting == 'D' else '')


def _seat(kind, board):
    import evaluate
    import train_actor_critic as trainer
    from kill_search import search_action3
    if kind == 'rulebot':
        return evaluate.make_rule_based_fn
    search = kind.endswith('_search3')  # '<CKPT key>_search3' = that CNN + three-move kill-search
    policy = evaluate.make_ac_fn(str(CKPT[kind[:-len('_search3')] if search else kind]), board)
    if search:
        return lambda i: (lambda env, obs, aid: search_action3(env, aid, trainer.get_action_mask(env, aid),
                                                               policy(env, obs, aid))[0])
    return lambda i: policy


def play(setting, learner, opponent, seed, boards, max_steps, shard_path, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    board, agents = SETTINGS[setting]
    learner_fn = _seat(learner, board)(0)
    opponent_fn = _seat(opponent, board)
    done = [0]

    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    start = time.time()
    result = evaluate.evaluate_seeded(learner_fn, board, agents, boards * agents, seed, max_steps, on_progress=advance,
                                      sudden_death=True, opponent_fn=opponent_fn, **sd5.SCHEDULE_E)
    part = (arm_name(setting, learner, opponent), time.time() - start, result['games'])
    Path(shard_path).write_text(json.dumps(part), encoding='utf-8')  # kept so --resume skips finished jobs
    return part


def paired_bootstrap(candidate, reference, samples=10000, seed=20261014):
    """Percentile CI (pp) of the mean per-board difference; each board keeps all its corners."""
    def boards(rows):
        out = {}
        for g in rows:
            out.setdefault(g['seed'], {})[g['aid']] = g['value']
        return out
    a, b = boards(candidate), boards(reference)
    if a.keys() != b.keys() or any(a[s].keys() != b[s].keys() for s in a):
        raise ValueError('candidate and reference must cover the same boards and corners')
    diffs = np.array([np.mean([a[s][i] - b[s][i] for i in a[s]]) for s in sorted(a)])
    draws = diffs[np.random.default_rng(seed).integers(0, len(diffs), (samples, len(diffs)))].mean(axis=1) * 100
    low, high = np.quantile(draws, [0.025, 0.975])
    return {'difference_pp': float(diffs.mean() * 100), 'ci95_pp': [float(low), float(high)],
            'boards': len(diffs), 'resamples': samples, 'bootstrap_seed': seed}


def comparisons(results):
    out = {}
    pairs = [(f'{s}_{a}', f'{s}_{b}') for s, a, b in COMPARE if s != 'D']
    pairs += [(arm_name('D', 'G_search3', o), arm_name('D', 'G', o)) for o in ('G', 'F', 'S', 'B', 'oldD')]
    for a, b in pairs:
        out[f'{a} - {b}'] = {m: paired_bootstrap([{**g, 'value': g[m]} for g in results[a]],
                                                 [{**g, 'value': g[m]} for g in results[b]]) for m in METRICS}
    return out


def run(output, workers=5, smoke=False, resume=False):
    for key in ('G',):
        if sd5._hash(CKPT[key]) != sd76.G_SHA256:
            raise ValueError('G checkpoint changed')
    output = Path(output)
    seed, steps = (21000, 40) if smoke else (SEED, MAX_STEPS)
    arms = [(s, l, o, 1 if smoke else n) for s, l, o, n in ARMS]
    opened = {'seed': seed, 'max_steps': steps, 'arms': arms, 'G_sha256': sd76.G_SHA256}
    if resume:
        if json.loads((output / 'opened.json').read_text(encoding='utf-8')) != json.loads(json.dumps(opened)):
            raise ValueError('registered design differs from this run; refusing to resume')
    else:
        output.mkdir(parents=True)  # FileExistsError: never overwrite
        (output / 'opened.json').write_text(json.dumps(opened, indent=2), encoding='utf-8')
    (output / 'shards').mkdir(exist_ok=True)
    jobs = []
    for s, l, o, n in arms:
        k = min(SHARDS.get(l, 1), n)
        for i in range(k):
            first, count = seed + i * n // k, (i + 1) * n // k - i * n // k
            jobs.append((s, l, o, first, count, steps, str(output / 'shards' / f'{arm_name(s, l, o)}_{first}.json')))
    done = [j for j in jobs if Path(j[-1]).exists()]
    todo = sorted((j for j in jobs if j not in done),
                  key=lambda j: -(COST[j[0]] if j[1] == 'G_search3' else 3) * j[4] * SETTINGS[j[0]][1])
    total = sum(j[4] * SETTINGS[j[0]][1] for j in todo)
    parts = [tuple(json.loads(Path(j[-1]).read_text(encoding='utf-8'))) for j in done]
    parts += parallel_map(play, todo, total=total, desc=f'PDE seed {seed}', workers=workers)
    results, seconds = {}, {}
    for name, sec, games in parts:
        results.setdefault(name, []).extend(games); seconds[name] = seconds.get(name, 0) + sec
    for name, rows in results.items():
        rows.sort(key=lambda g: (g['seed'], g['aid']))
        (output / f'evaluation_{name}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    summary = {'seed': seed, 'max_steps': steps, 'schedule': sd5.SCHEDULE_E,
               'seconds_per_game': {k: v / len(results[k]) for k, v in seconds.items()},
               'results': {k: {**summarize(v), 'by_corner': {str(c): summarize([g for g in v if g['aid'] == c])
                                                              for c in sorted({g['aid'] for g in v})}}
                           for k, v in results.items()},
               'comparisons': comparisons(results)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not smoke:
        shutil.make_archive(str(output) + '-send', 'zip', output, '.')
    print(json.dumps({k: (v['active_win_rate'], v['win_rate'], v['draw_rate']) for k, v in summary['results'].items()},
                     indent=1), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true', help='keep finished shards in OUTPUT/shards, run the rest')
    a = p.parse_args()
    run(a.output, a.workers, a.smoke, a.resume)
