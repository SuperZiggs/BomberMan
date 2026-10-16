"""SD4.1: learner-only ring safety margin k, swept on validation, confirmed once on a fresh test seed.

Rules fixed with the user before any SD4.1 result (PLAN_SUDDEN_DEATH.md):
- sweep k in {0, 5, 10, 15, 20} on validation 21000, 400 games each, B under SD2 both_aware;
- select: among k > 0 with fewer learner ring deaths than k=0, highest ActiveWR; ties -> smaller k;
- gate on test 41000 (k=0 vs chosen k, 400 games each): ring-death difference CI95 upper < 0,
  ActiveWR difference CI95 lower >= -3pp, suicide rate not more than +1pp.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from pilot_two_bomb import paired_active_bootstrap

MARGINS = (0, 5, 10, 15, 20)
VALIDATION_SEED, TEST_SEED, GAMES = 21000, 41000, 400
KEYS = ('win_rate', 'active_win_rate', 'passive_win_rate', 'draw_rate', 'suicide_rate', 'avg_kills')
METRICS = {'ring_death': lambda g: int(g['learner_death'] == 'sudden_death'),
           'active_win': lambda g: g['active_win'], 'win': lambda g: g['win'],
           'draw': lambda g: g['draw'], 'suicide': lambda g: g['suicide']}


def evaluate(k, seed, games=GAMES):
    import torch
    torch.set_num_threads(1)
    import run_sudden_death_experiment as runner
    from run_sd_training_pilot import SD2
    result = runner.evaluate_case('both_aware', games=games, seed=seed, max_steps=500, timings=SD2,
                                  learner_margin=k)
    result.update(margin=k, seed=seed, num_episodes=games, num_agents=2,
                  learner_sd_deaths=result['sudden_death']['learner_deaths'].get('sudden_death', 0))
    return result


def select(rows):
    base = next(r for r in rows if r['margin'] == 0)
    ok = [r for r in rows if r['margin'] > 0 and r['learner_sd_deaths'] < base['learner_sd_deaths']]
    return max(ok, key=lambda r: (r['active_win_rate'], -r['margin'])) if ok else None


def compare(left, right, metric):
    wrap = lambda r: {**r, 'games': [{**g, 'active_win': METRICS[metric](g)} for g in r['games']]}
    return paired_active_bootstrap(wrap(left), wrap(right))


def gate(candidate, base):
    out = {m: compare(candidate, base, m) for m in METRICS}
    out['suicide_rate_diff_pp'] = candidate['suicide_rate'] - base['suicide_rate']
    out['passed'] = bool(out['ring_death']['ci95_pp'][1] < 0 and out['active_win']['ci95_pp'][0] >= -3
                         and out['suicide_rate_diff_pp'] <= 1)
    return out


def brief(r):
    return {'margin': r['margin'], **{k: r[k] for k in KEYS}, 'learner_sd_deaths': r['learner_sd_deaths'],
            'sudden_death': {k: v for k, v in r['sudden_death'].items()}}


def run_parallel(jobs, seed, output):
    with ProcessPoolExecutor(min(len(jobs), 5)) as pool:  # machine RAM: <= ~6 torch processes
        results = list(pool.map(evaluate, jobs, [seed] * len(jobs)))
    for r in results:
        (output / f"evaluation_k{r['margin']}_{seed}.json").write_text(json.dumps(r), encoding='utf-8')
        print(f"k={r['margin']}: WR={r['win_rate']:.2f} Active={r['active_win_rate']:.2f} "
              f"Draw={r['draw_rate']:.2f} Self={r['suicide_rate']:.2f} ringDeaths={r['learner_sd_deaths']}", flush=True)
    return results


def sweep(output, seed=VALIDATION_SEED):
    if seed >= 31000:
        raise ValueError('sweep must stay on validation seeds')
    output.mkdir(parents=True)
    results = run_parallel(MARGINS, seed, output)
    rows = [brief(r) for r in results]
    chosen = select(rows)
    (output / 'summary.json').write_text(json.dumps({'seed': seed, 'games': GAMES, 'rows': rows,
        'selected_margin': chosen and chosen['margin']}, indent=2), encoding='utf-8')
    print('selected margin:', chosen and chosen['margin'], flush=True)


def confirm(sweep_dir, output):
    from run_combat_move_pilot import fresh_test_check
    chosen = json.loads((sweep_dir / 'summary.json').read_text(encoding='utf-8'))['selected_margin']
    if not chosen:
        raise SystemExit('No margin passed validation selection; test seed stays closed')
    fresh_test_check(TEST_SEED)
    output.mkdir(parents=True)
    (output / 'test_opened.json').write_text(json.dumps({'seed': TEST_SEED, 'num_episodes': GAMES,
        'num_agents': 2, 'margin': chosen, 'sweep': str(sweep_dir)}), encoding='utf-8')
    base, candidate = sorted(run_parallel((0, chosen), TEST_SEED, output), key=lambda r: r['margin'])
    result = gate(candidate, base)
    (output / 'summary.json').write_text(json.dumps({'seed': TEST_SEED, 'games': GAMES, 'margin': chosen,
        'models': [brief(base), brief(candidate)], 'gate': result}, indent=2), encoding='utf-8')
    print(json.dumps({m: result[m] if m in ('passed', 'suicide_rate_diff_pp')
                      else (result[m]['difference_pp'], result[m]['ci95_pp']) for m in result}, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('sweep', 'confirm'))
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--sweep-dir', type=Path, help='confirm: folder of the finished validation sweep')
    a = p.parse_args()
    sweep(a.output_dir) if a.mode == 'sweep' else confirm(a.sweep_dir, a.output_dir)
