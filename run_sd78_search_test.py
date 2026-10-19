"""SD7.8: G vs G + kill-search at play time, test seed 46000, once. No training.

Spec: docs/superpowers/specs/2026-10-19-sd7-8-g-plus-search-design.md (fixed before running).
Gate (schedule E): G+search - G ActiveWR >= +5pp, CI95 lower > 0, suicide <= +1pp.
"""
import argparse
import json
from pathlib import Path
import shutil

from progress import parallel_map
import run_sd5_pilot as sd5
import run_sd76_pilot as sd76

ARMS, RULES, SEED, GAMES, MAX_STEPS = ('G', 'search'), ('E', 'off'), 46000, 400, 500


def play(arm, rule, seed, games, max_steps, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    import train_actor_critic as trainer
    from kill_search import search_action
    policy = evaluate.make_ac_fn(str(sd76.G), 9)
    agent = policy if arm == 'G' else (
        lambda env, obs, aid: search_action(env, aid, trainer.get_action_mask(env, aid), policy(env, obs, aid))[0])

    def advance(done, total):
        if done:
            ticks.put(1)
    result = evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=advance,
                                      sudden_death=rule == 'E', **sd5.SCHEDULE_E)
    return arm, rule, result['games']


def summarize(games):
    n = len(games)
    rates = {f'{k}_rate': 100 * sum(g[k] for g in games) / n
             for k in ('win', 'active_win', 'passive_win', 'sudden_death_win', 'draw', 'suicide', 'sudden_death_death')}
    return {'num_episodes': n, **rates, **{f'avg_{k}': sum(g[k] for g in games) / n for k in ('kills', 'bombs', 'steps')}}


def gate(paired):
    checks = {'active_5pp': paired['active_win']['difference_pp'] >= 5.0,
              'active_ci_lower_positive': paired['active_win']['ci95_pp'][0] > 0,
              'suicide_within_1pp': paired['suicide']['difference_pp'] <= 1.0}
    return {**checks, 'pass': all(checks.values())}


def main(output, workers=5, smoke=False):
    from pilot_two_bomb import paired_active_bootstrap
    if sd5._hash(sd76.G) != sd76.G_SHA256:
        raise ValueError('G checkpoint changed')
    output = Path(output)
    output.mkdir(parents=True)  # FileExistsError: never overwrite
    seed, games, steps = (21000, 4, 40) if smoke else (SEED, GAMES, MAX_STEPS)
    with open(output / 'test_opened.json', 'x', encoding='utf-8') as handle:  # the test seed is used once
        json.dump({'seed': seed, 'games': games, 'G_sha256': sd76.G_SHA256}, handle, indent=2)
    shards = 4 if not smoke else 2  # search games are slow: split each search run by board seed
    jobs = [(arm, rule, seed + i * games // 2 // n, games // n, steps)
            for arm in ARMS for rule in RULES for n in [shards if arm == 'search' else 1] for i in range(n)]
    parts = parallel_map(play, jobs, total=len(ARMS) * len(RULES) * games, desc=f'SD7.8 test seed {seed}', workers=workers)
    results = {f'{a}_{r}': sorted([g for pa, pr, gs in parts if (pa, pr) == (a, r) for g in gs],
                                  key=lambda g: (g['seed'], g['aid'])) for a in ARMS for r in RULES}
    for key, rows in results.items():
        (output / f'evaluation_{key}_{seed}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    paired = {rule: {m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in results[f'{a}_{rule}']]}
                                                  for a in ('search', 'G')))
                     for m in ('active_win', 'win', 'draw', 'passive_win', 'suicide')} for rule in RULES}
    summary = {'seed': seed, 'games': games, 'results': {k: summarize(v) for k, v in results.items()},
               'search_vs_g': paired, 'gate': gate(paired['E'])}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not smoke:
        shutil.make_archive(str(output) + '-send', 'zip', output, '.')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    main(a.output, a.workers, a.smoke)
