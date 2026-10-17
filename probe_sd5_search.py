"""SD5.0: does kill-search still add active wins on top of F? Validation only, no training.

Schedule E (engine sudden death 475/25/40), seed 21000, 400 games per arm, both corners:
  f       F unchanged (control), with contact diagnostics for the draw breakdown
  search  F's sampled action unless `kill_search` finds a strictly better line (from step 0)
Both arms consume the same torch RNG stream, so games only diverge where search overrides.
Decision rule (PLAN_SUDDEN_DEATH.md SD5.0, fixed before running): open the SD5.1 search arm if
search - f >= +3pp ActiveWR with the paired CI95 lower bound > 0.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

F = 'runs/phaseC-sd-best-fallback/best_phaseC.pt'
SCHEDULE_E = dict(sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
SEED, GAMES, MAX_STEPS = 21000, 400, 500
RATES = ('win', 'active_win', 'sudden_death_win', 'passive_win', 'draw', 'suicide', 'sudden_death_death')


def evaluate_arm(arm, seed=SEED, games=GAMES, max_steps=MAX_STEPS, schedule=SCHEDULE_E):
    if seed >= 31000:
        raise ValueError('probe must stay on validation seeds')
    import torch
    torch.set_num_threads(1)
    import evaluate
    import train_actor_critic as trainer
    from kill_search import search_action
    policy = evaluate.make_ac_fn(F, 9)
    agent = policy if arm == 'f' else (
        lambda env, obs, aid: search_action(env, aid, trainer.get_action_mask(env, aid), policy(env, obs, aid))[0])
    result = evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, sudden_death=True,
                                      contact_diagnostics=True, **schedule)
    result['arm'] = arm
    return result


def merge(parts):
    """Concatenate board-seed shards; recompute the rates from per-game rows."""
    games = [g for r in sorted(parts, key=lambda r: r['games'][0]['seed']) for g in r['games']]
    out = {'arm': parts[0]['arm'], 'games': games,
           **{f'{k}_rate': 100 * sum(g[k] for g in games) / len(games) for k in RATES},
           'avg_kills': sum(g['kills'] for g in games) / len(games),
           'avg_steps': sum(g['steps'] for g in games) / len(games)}
    from evaluate import summarize_contact
    out['contact'] = summarize_contact(games)
    return out


def draw_breakdown(games, max_steps=MAX_STEPS):
    draws = [g for g in games if g['draw']]
    timeout = [g for g in draws if g['steps'] >= max_steps]
    return {'draws': len(draws), 'timeout': len(timeout), 'both_died': len(draws) - len(timeout),
            'timeout_never_close': sum(g['contact']['first_close_step'] is None for g in timeout)}


def decide(paired):
    return paired['difference_pp'] >= 3.0 and paired['ci95_pp'][0] > 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--shards', nargs=2, type=int, default=[1, 4], metavar=('F', 'SEARCH'),
                   help='board-seed shards per arm (one process each, at most 5 in total)')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True)
    # Every game reseeds from its board seed, so shards reproduce one long run exactly.
    jobs = [(arm, SEED + i * GAMES // 2 // n, GAMES // n)
            for arm, n in zip(('f', 'search'), a.shards) for i in range(n)]
    with ProcessPoolExecutor(min(len(jobs), 5)) as pool:
        parts = list(pool.map(evaluate_arm, *zip(*jobs)))
    results = {arm: merge([r for r in parts if r['arm'] == arm]) for arm in ('f', 'search')}
    from pilot_two_bomb import paired_active_bootstrap
    rows = {}
    for arm, r in results.items():
        (a.output_dir / f'evaluation_{arm}_{SEED}.json').write_text(json.dumps(r), encoding='utf-8')
        rows[arm] = {k: v for k, v in r.items() if k != 'games'} | {'draw_breakdown': draw_breakdown(r['games'])}
    paired = {m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in results[arm]['games']]}
                                           for arm in ('search', 'f'))) for m in ('active_win', 'win', 'draw')}
    summary = {'seed': SEED, 'games': GAMES, 'schedule': SCHEDULE_E, 'checkpoint': F, 'rows': rows,
               'search_vs_f': paired, 'open_sd5_1_search_arm': decide(paired['active_win'])}
    (a.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
