"""Pre-registered selection on validation sweep + one confirmation on fresh test seeds.

Selection (fixed before reading sweep results): case both_aware; maximise WR subject to
ActiveWR >= off ActiveWR - 3pp and suicide <= off suicide + 1pp; ties -> lower draw.
"""
import argparse
import json
from pathlib import Path

import torch

import run_sudden_death_experiment as runner
from pilot_two_bomb import paired_active_bootstrap


def select(summary_path, case='both_aware'):
    rows = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    off = next(r for r in rows if r['case'] == 'off')
    ok = [r for r in rows if r['case'] == case
          and r['active_win_rate'] >= off['active_win_rate'] - 3
          and r['suicide_rate'] <= off['suicide_rate'] + 1]
    if not ok:
        raise SystemExit('no config passes the guards')
    return max(ok, key=lambda r: (r['win_rate'], -r['draw_rate'])), off


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sweep', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--seed', type=int, default=39000)
    p.add_argument('--games', type=int, default=400)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    best, off_val = select(args.sweep)
    timings = {k: best[k] for k in ('sudden_death_start', 'sudden_death_interval',
                                    'sudden_death_warning_steps', 'sudden_death_max_ring')}
    (args.output_dir / 'selected.json').write_text(json.dumps(
        {'selected_validation': best, 'off_validation': off_val, 'timings': timings}, indent=2), encoding='utf-8')
    print('selected', timings, flush=True)
    torch.set_num_threads(1)
    results = {}
    for case in ('off', 'both_aware', 'learner_aware'):
        r = runner.evaluate_case(case, games=args.games, seed=args.seed, max_steps=500, timings=timings)
        results[case] = r
        runner.dump(args.output_dir / f'evaluation_{case}.json', r)
        print(f"[{case}] WR={r['win_rate']:.2f} ActiveWR={r['active_win_rate']:.2f} "
              f"Draw={r['draw_rate']:.2f} Self={r['suicide_rate']:.2f}", flush=True)
    comp = {}
    for label, (left, right) in {'both_aware_vs_off': ('both_aware', 'off'),
                                 'learner_aware_vs_off': ('learner_aware', 'off')}.items():
        comp[label] = {}
        for metric in ('win', 'active_win', 'draw', 'suicide'):
            wrapped = [{**results[c], 'games': [{**g, 'active_win': g[metric]} for g in results[c]['games']]}
                       for c in (left, right)]
            comp[label][metric] = paired_active_bootstrap(*wrapped)
    summary = {'timings': timings, 'seed': args.seed, 'games': args.games,
               'results': {c: {k: v for k, v in r.items() if k != 'games'} for c, r in results.items()},
               'comparisons': comp}
    runner.dump(args.output_dir / 'summary.json', summary)
    for label, v in comp.items():
        print(label, {m: (round(x['difference_pp'], 2), x['ci95_pp']) for m, x in v.items()}, flush=True)


if __name__ == '__main__':
    main()
