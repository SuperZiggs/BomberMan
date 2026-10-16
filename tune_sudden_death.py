"""Grid search of sudden-death timings for frozen checkpoint B on validation seeds.

Reuses run_sudden_death_experiment.evaluate_case; each config runs in its own
single-thread process. Test seeds are never touched here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
from time import perf_counter

KEYS = ('win_rate', 'active_win_rate', 'passive_win_rate', 'draw_rate', 'suicide_rate', 'avg_kills')


def run_one(case, start, interval, warning, max_ring, games, seed, max_steps):
    import torch
    torch.set_num_threads(1)
    import run_sudden_death_experiment as runner
    begin = perf_counter()
    timings = {'sudden_death_start': start, 'sudden_death_interval': interval,
               'sudden_death_warning_steps': warning, 'sudden_death_max_ring': max_ring}
    result = runner.evaluate_case(case, games=games, seed=seed, max_steps=max_steps, timings=timings)
    row = {'case': case, **timings, 'games': games, 'seed': seed,
           **{k: result[k] for k in KEYS}, 'sudden_death': result['sudden_death'],
           'seconds': perf_counter() - begin}
    return row, result['games']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--cases', nargs='+', default=['learner_aware', 'both_aware'])
    p.add_argument('--starts', nargs='+', type=int, default=[100, 150, 200, 250, 300])
    p.add_argument('--intervals', nargs='+', type=int, default=[10, 20, 40])
    p.add_argument('--warnings', nargs='+', type=int, default=[40])
    p.add_argument('--max-rings', nargs='+', type=int, default=[3])
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--seed', type=int, default=21000)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.seed >= 38000:
        raise ValueError('tuning must stay on validation seeds')
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    jobs = [('off', 200, 20, 40, 3)] + list(itertools.product(
        args.cases, args.starts, args.intervals, args.warnings, args.max_rings))
    rows = []
    if args.resume:  # keep finished configs (same seed/games only)
        for path in args.output_dir.glob('*_r*.json'):
            row = json.loads(path.read_text(encoding='utf-8'))
            if (row['seed'], row['games']) != (args.seed, args.games):
                raise ValueError(f'{path.name} was run with another seed/games')
            row.pop('games_detail')
            rows.append(row)
        done = {(r['case'], r['sudden_death_start'], r['sudden_death_interval'],
                 r['sudden_death_warning_steps'], r['sudden_death_max_ring']) for r in rows}
        jobs = [j for j in jobs if j not in done]
        print(f'resume: {len(rows)} done, {len(jobs)} left', flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, *job, args.games, args.seed, args.max_steps) for job in jobs]
        for f in as_completed(futures):
            row, games = f.result()
            rows.append(row)
            name = (f"{row['case']}_s{row['sudden_death_start']}_i{row['sudden_death_interval']}"
                    f"_w{row['sudden_death_warning_steps']}_r{row['sudden_death_max_ring']}")
            (args.output_dir / f'{name}.json').write_text(json.dumps({**row, 'games_detail': games}), encoding='utf-8')
            print(f"{name}: WR={row['win_rate']:.1f} Active={row['active_win_rate']:.1f} "
                  f"Draw={row['draw_rate']:.1f} Self={row['suicide_rate']:.1f} ({row['seconds']:.0f}s)", flush=True)
    rows.sort(key=lambda r: (-r['win_rate'], r['draw_rate'], -r['active_win_rate']))
    (args.output_dir / 'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
