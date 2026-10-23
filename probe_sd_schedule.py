"""SD9.3 probe: does an earlier sudden-death schedule turn S3's draws against CNNs into active wins? No training.

Spec: docs/superpowers/specs/2026-10-24-sd9-3-sd-schedule-probe.md (rules fixed before running).
S3 (sampled) vs S2, G, S3 (sampled) and RuleBot, 1v1 9x9, max_steps 500, validation seed 21000,
200 boards x 2 corners per (schedule, opponent). On 9x9 the third ring fills every free cell, so a
schedule start/25 ends every game by step start + 50. Schedule E (475/25) is the reference.
"""
import argparse
import json
from pathlib import Path
import time

from progress import parallel_map
import probe_s3_cnn as sd91
import run_pde_check as pde
import run_sd5_pilot as sd5
from run_sd78_search_test import summarize

ROOT = Path(__file__).resolve().parent
SCHEDULES = {'E': sd5.SCHEDULE_E,
             **{f'S{s}': dict(sudden_death_start=s, sudden_death_interval=25, sudden_death_warning_steps=40)
                for s in (400, 350, 300)}}
OPPONENTS = ('S2', 'G', 'S3', 'rulebot')
METRICS = pde.METRICS + ('sudden_death_win', 'sudden_death_death')
SD91 = ROOT / 'runs/sd91-21000'


def play(schedule, opponent, seed, boards, max_steps, shard_path, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    done = [0]

    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    start = time.time()
    games = evaluate.evaluate_seeded(pde._seat('S3', 9)(0), 9, 2, boards * 2, seed, max_steps, on_progress=advance,
                                     sudden_death=True, opponent_fn=pde._seat(opponent, 9),
                                     **SCHEDULES[schedule])['games']
    part = (f'{schedule}_vs_{opponent}', time.time() - start, games)
    Path(shard_path).write_text(json.dumps(part), encoding='utf-8')
    return part


def verdict(macro, rulebot, chosen_order=('S400', 'S350', 'S300')):
    """Rule fixed in the spec. macro[x] / rulebot[x]: paired bootstraps of schedule x minus E."""
    out = {}
    for x in chosen_order:
        m, r = macro[x], rulebot[x]
        checks = {'draw_down_15pp': m['draw']['difference_pp'] <= -15, 'draw_ci_upper_negative': m['draw']['ci95_pp'][1] < 0,
                  'active_up_5pp': m['active_win']['difference_pp'] >= 5, 'active_ci_lower_positive': m['active_win']['ci95_pp'][0] > 0,
                  'suicide_within_1pp': m['suicide']['difference_pp'] <= 1,
                  'rulebot_active_within_3pp': r['active_win']['difference_pp'] >= -3}
        out[x] = {**checks, 'helps': all(checks.values())}
    helping = [x for x in chosen_order if out[x]['helps']]
    best = max((macro[x]['active_win']['difference_pp'] for x in helping), default=None)
    # tie within 1pp -> the latest start (closest to E); chosen_order runs from latest to earliest start
    out['chosen'] = next((x for x in helping if macro[x]['active_win']['difference_pp'] >= best - 1), None)
    return out


def run(output, seed=21000, boards=200, workers=5, max_steps=500, shard=10, resume=False, reference=SD91):
    for key in ('G', 'S2', 'S3'):
        if sd5._hash(pde.CKPT[key]) != sd91.SHA256[key]:
            raise ValueError(f'{key} checkpoint changed')
    output = Path(output)
    opened = {'seed': seed, 'boards': boards, 'max_steps': max_steps, 'learner': 'S3', 'opponents': OPPONENTS,
              'schedules': SCHEDULES, 'sha256': sd91.SHA256}
    if resume:
        if json.loads((output / 'opened.json').read_text(encoding='utf-8')) != json.loads(json.dumps(opened)):
            raise ValueError('registered design differs from this run; refusing to resume')
    else:
        output.mkdir(parents=True)  # FileExistsError: never overwrite
        (output / 'opened.json').write_text(json.dumps(opened, indent=2), encoding='utf-8')
    (output / 'shards').mkdir(exist_ok=True)
    jobs = [(s, o, seed + first, min(shard, boards - first), max_steps,
             str(output / 'shards' / f'{s}_vs_{o}_{seed + first}.json'))
            for s in SCHEDULES for o in OPPONENTS for first in range(0, boards, shard)]
    todo = [j for j in jobs if not Path(j[-1]).exists()]
    parallel_map(play, todo, total=sum(j[3] * 2 for j in todo), desc=f'SD9.3 seed {seed}', workers=workers)
    results, seconds = {}, {}
    for s, o, *_, path in jobs:
        name, sec, games = json.loads(Path(path).read_text(encoding='utf-8'))
        results.setdefault(name, []).extend(games); seconds[name] = seconds.get(name, 0) + sec
    for rows in results.values():
        rows.sort(key=lambda g: (g['seed'], g['aid']))
    replay = {}  # schedule E vs CNNs replays SD9.1's S3 games exactly
    for o in sd91.CNN_OPPONENTS:
        old = {}
        for f in (Path(reference) / 'shards').glob(f'S3_vs_{o}_*.json'):
            if f.stem.split('_vs_')[1].rsplit('_', 1)[0] == o:
                old.update({(g['seed'], g['aid']): g for g in json.loads(f.read_text(encoding='utf-8'))[2]})
        mine = [g for g in results[f'E_vs_{o}'] if (g['seed'], g['aid']) in old]
        replay[o] = {'compared': len(mine), 'identical': sum(all(old[(g['seed'], g['aid'])][k] == g[k]
                                                                 for k in ('win', 'draw', 'steps', 'active_win')) for g in mine)}
    value = lambda rows, m: [{**g, 'value': g[m]} for g in rows]
    alternatives = [s for s in SCHEDULES if s != 'E']
    macro = {x: {m: pde.paired_bootstrap(sd91._macro(results, x, m), sd91._macro(results, 'E', m)) for m in METRICS}
             for x in alternatives}
    per_opponent = {f'{x} - E vs {o}': {m: pde.paired_bootstrap(value(results[f'{x}_vs_{o}'], m), value(results[f'E_vs_{o}'], m))
                                        for m in METRICS} for x in alternatives for o in OPPONENTS}
    rulebot = {x: per_opponent[f'{x} - E vs rulebot'] for x in alternatives}
    summary = {**opened, 'seconds_per_game': {k: v / len(results[k]) for k, v in seconds.items()},
               'replay_E_matches_sd91': replay,
               'results': {k: {**summarize(v), 'loss_rate': 100 * sum(not g['win'] and not g['draw'] for g in v) / len(v),
                               'draw_at_max_steps': 100 * sum(g['draw'] and g['steps'] >= max_steps for g in v) / len(v)}
                           for k, v in results.items()},
               'macro_cnn': macro, 'comparisons': per_opponent, 'verdict': verdict(macro, rulebot)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for k, v in sorted(summary['results'].items()):
        print(f"{k:14s} ActiveWR {v['active_win_rate']:6.2f} WR {v['win_rate']:6.2f} SDwin {v['sudden_death_win_rate']:6.2f} "
              f"passive {v['passive_win_rate']:6.2f} draw {v['draw_rate']:6.2f} loss {v['loss_rate']:6.2f} "
              f"SDdeath {v['sudden_death_death_rate']:6.2f}", flush=True)
    for x, c in macro.items():
        print(f"{x} - E macro CNN: " + ', '.join(f"{m} {c[m]['difference_pp']:+.2f} [{c[m]['ci95_pp'][0]:+.2f}, {c[m]['ci95_pp'][1]:+.2f}]"
                                                 for m in ('active_win', 'draw', 'sudden_death_win', 'passive_win')), flush=True)
    print('replay', replay, '\nverdict', summary['verdict'], flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=21000)
    p.add_argument('--boards', type=int, default=200)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--max_steps', type=int, default=500)
    p.add_argument('--resume', action='store_true', help='keep finished shards in OUTPUT/shards, run the rest')
    a = p.parse_args()
    run(a.output, a.seed, a.boards, a.workers, a.max_steps, resume=a.resume)
