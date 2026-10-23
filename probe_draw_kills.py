"""SD9.2 probe: in S3's draws against CNNs, did a kill exist that S3 missed? No training.

Spec: docs/superpowers/specs/2026-10-23-sd9-2-draw-kill-probe.md (rules fixed before running).
Replays S3 (sampled) vs S2, G, S3 (sampled) on validation boards exactly as SD9.1 did (board seed fixes
every RNG), snapshots the state every STRIDE steps while the opponent is within kill_search.REACH, and
after each drawn game asks the three-move kill search (`kill_search._best_plan`, depth 3) whether a kill
plan exists from that state when the opponent is modelled by (a) its own CNN (argmax) and (b) RuleBot.
The same check on S3's active wins is a positive control (a kill did happen there).
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from progress import parallel_map
import probe_s3_cnn as sd91
import run_pde_check as pde
import run_sd5_pilot as sd5

ROOT = Path(__file__).resolve().parent
LEARNER = 'S3'
OPPONENTS = ('S2', 'G', 'S3')
MODELS = ('oracle', 'rulebot')  # oracle = the opponent's own CNN (argmax), rulebot = what S3's teacher assumed
SD91 = ROOT / 'runs/sd91-21000'  # replayed games must match these shards


def _near(env, aid):
    import kill_search
    me = env.agents[aid]
    return me.alive and any(a.alive and a.agent_id != aid and abs(a.row - me.row) + abs(a.col - me.col) <= kill_search.REACH
                            for a in env.agents)


def play(opponent, seed, boards, max_steps, stride, shard_path, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    import kill_search
    from probe_search_cnn_model import cnn_model
    policy = pde._seat(LEARNER, 9)(0)
    snaps, cur = [], {'env': None}

    def learner(env, obs, aid):
        if env is not cur['env']:  # new game: evaluate_seeded builds a fresh env per game
            cur['env'] = env
            snaps.append([])
        if env.step_count % stride == 0 and _near(env, aid):
            snaps[-1].append((env.step_count, deepcopy(env), aid))  # deepcopy uses no RNG
        return policy(env, obs, aid)

    done = [0]

    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    start = time.time()
    games = evaluate.evaluate_seeded(learner, 9, 2, boards * 2, seed, max_steps, on_progress=advance,
                                     sudden_death=True, opponent_fn=pde._seat(opponent, 9), **sd5.SCHEDULE_E)['games']
    assert len(games) == len(snaps)
    oracle = cnn_model(pde.CKPT[opponent], 9)
    try:
        for game, states in zip(games, snaps):
            game['states'] = []
            if game['draw'] or game['active_win']:  # active wins = positive control: a kill did happen
                for step, env, aid in states:
                    row = {'step': step}
                    for m in MODELS:
                        kill_search.OPPONENT_MODEL = oracle if m == 'oracle' else None
                        row[m] = bool(kill_search._best_plan(env, aid, 3, kill_search.HORIZON) > 0)
                    game['states'].append(row)
            game['near_states'] = len(states)
            ticks.put(1)
    finally:
        kill_search.OPPONENT_MODEL = None
    part = (f'{LEARNER}_vs_{opponent}', time.time() - start, games)
    Path(shard_path).write_text(json.dumps(part), encoding='utf-8')
    return part


def _share(games, test, rng_seed=20261014, samples=10000):
    """Percent of games passing `test`, with a percentile CI95 resampling whole boards."""
    by_board = {}
    for g in games:
        by_board.setdefault(g['seed'], []).append(bool(test(g)))
    if not by_board:
        return {'percent': None, 'ci95': None, 'games': 0}
    hits = np.array([sum(v) for v in by_board.values()]); counts = np.array([len(v) for v in by_board.values()])
    idx = np.random.default_rng(rng_seed).integers(0, len(hits), (samples, len(hits)))
    boot = hits[idx].sum(1) / np.maximum(counts[idx].sum(1), 1) * 100
    return {'percent': 100 * hits.sum() / counts.sum(), 'ci95': [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            'games': int(counts.sum())}


def describe(games):
    draws = [g for g in games if g['draw']]
    wins = [g for g in games if g['active_win']]
    states = [s for g in draws for s in g['states']]
    out = {'games': len(games), 'draws': len(draws), 'draw_states': len(states)}
    for m in MODELS:
        out[f'state_kill_{m}_percent'] = 100 * sum(s[m] for s in states) / len(states) if states else None
        out[f'draws_with_kill_{m}'] = _share(draws, lambda g: any(s[m] for s in g['states']))
        out[f'draws_with_kill_{m}_before_475'] = _share(draws, lambda g: any(s[m] and s['step'] < 475 for s in g['states']))
        out[f'control_active_wins_with_kill_{m}'] = _share(wins, lambda g: any(s[m] for s in g['states']))
    return out


def verdict(pooled):
    """Rule fixed in the spec, on the pooled share of draws with at least one oracle kill state.
    Invalid if the search misses the kill in most of S3's own active wins (positive control)."""
    p, control = pooled['draws_with_kill_oracle']['percent'], pooled['control_active_wins_with_kill_oracle']['percent']
    if control is None or control < 50:
        return {'percent': p, 'control': control, 'call': 'invalid_control'}
    return {'percent': p, 'control': control,
            'call': 'missed_kills_direction_A' if p >= 50 else 'structural_consider_rules' if p <= 20 else 'mixed'}


def run(output, seed=21000, boards=50, workers=5, max_steps=500, stride=5, shard=5, resume=False, reference=SD91):
    for key in ('G', 'S2', 'S3'):
        if sd5._hash(pde.CKPT[key]) != sd91.SHA256[key]:
            raise ValueError(f'{key} checkpoint changed')
    output = Path(output)
    opened = {'seed': seed, 'boards': boards, 'max_steps': max_steps, 'stride': stride, 'learner': LEARNER,
              'opponents': OPPONENTS, 'models': MODELS, 'depth': 3, 'sha256': sd91.SHA256,
              'kill_search_sha256': sd5._hash(ROOT / 'kill_search.py')}
    if resume:
        if json.loads((output / 'opened.json').read_text(encoding='utf-8')) != json.loads(json.dumps(opened)):
            raise ValueError('registered design differs from this run; refusing to resume')
    else:
        output.mkdir(parents=True)  # FileExistsError: never overwrite
        (output / 'opened.json').write_text(json.dumps(opened, indent=2), encoding='utf-8')
    (output / 'shards').mkdir(exist_ok=True)
    jobs = [(o, seed + first, min(shard, boards - first), max_steps, stride,
             str(output / 'shards' / f'{LEARNER}_vs_{o}_{seed + first}.json'))
            for o in OPPONENTS for first in range(0, boards, shard)]
    todo = [j for j in jobs if not Path(j[-1]).exists()]
    parallel_map(play, todo, total=sum(j[2] * 4 for j in todo), desc=f'SD9.2 seed {seed}', workers=workers)
    results, seconds = {}, {}
    for o, *_, path in jobs:
        name, sec, games = json.loads(Path(path).read_text(encoding='utf-8'))
        results.setdefault(o, []).extend(games); seconds[o] = seconds.get(o, 0) + sec
    replay = {}  # same boards, code and torch as SD9.1 -> the games should be identical
    for o, games in results.items():
        old = {}
        for f in (Path(reference) / 'shards').glob(f'{LEARNER}_vs_{o}_*.json'):
            if f.stem.split('_vs_')[1].rsplit('_', 1)[0] == o:
                old.update({(g['seed'], g['aid']): g for g in json.loads(f.read_text(encoding='utf-8'))[2]})
        keys = ('win', 'draw', 'steps', 'active_win')
        replay[o] = {'compared': sum((g['seed'], g['aid']) in old for g in games),
                     'identical': sum(all(old[(g['seed'], g['aid'])][k] == g[k] for k in keys)
                                      for g in games if (g['seed'], g['aid']) in old)}
    per = {o: describe(g) for o, g in results.items()}
    pooled = describe([g for o in OPPONENTS for g in results[o]])
    summary = {**opened, 'schedule': sd5.SCHEDULE_E, 'seconds_per_game': {o: seconds[o] / len(results[o]) for o in results},
               'replay_matches_sd91': replay, 'per_opponent': per, 'pooled': pooled, 'verdict': verdict(pooled)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for o, d in {**per, 'pooled': pooled}.items():
        print(f"{o:7s} draws {d['draws']:3d}/{d['games']:3d} states {d['draw_states']:5d} | "
              + ' | '.join(f"{m}: states {d[f'state_kill_{m}_percent'] or 0:5.1f}% draws "
                           f"{d[f'draws_with_kill_{m}']['percent'] or 0:5.1f}% {d[f'draws_with_kill_{m}']['ci95']}"
                           for m in MODELS), flush=True)
    print('control (active wins with a kill state)', {m: pooled[f'control_active_wins_with_kill_{m}'] for m in MODELS})
    print('replay', replay, '\nverdict', summary['verdict'], flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=21000)
    p.add_argument('--boards', type=int, default=50)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--max_steps', type=int, default=500)
    p.add_argument('--resume', action='store_true', help='keep finished shards in OUTPUT/shards, run the rest')
    a = p.parse_args()
    run(a.output, a.seed, a.boards, a.workers, a.max_steps, a.stride, resume=a.resume)
