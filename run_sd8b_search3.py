"""SD8B: G + three-move kill-search (`kill_search.search_action3`) vs G + two-move (current best agent).

No training. Spec: docs/superpowers/specs/2026-10-20-sd8-distill-and-deep-search-design.md (fixed before running).

  validate: validation seed 21000, 400 games per arm, schedule E. Rule: search3 - search2 ActiveWR >= +2pp
            opens the test. `--base DIR` reuses G+search2 games from an earlier validation run on the same
            machine/torch (e.g. runs/sd79b-search2-21000) instead of replaying them.
  test:     test seed 49000, once, schedule E, 400 games per arm. Gate: ActiveWR >= +5pp, CI95 lower > 0,
            suicide <= +1pp. Refused unless the validation rule passed.
  ablate:   (descriptive, no gate) replay both arms with schedule off on the test boards and pair them with
            the schedule-E games of `--test DIR`: E - off per arm, off search3 - search2, and the interaction.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from progress import parallel_map
import run_sd5_pilot as sd5
import run_sd76_pilot as sd76
from run_sd78_search_test import gate, summarize

ARMS = ('G_search2', 'G_search3')
VALIDATION_SEED, TEST_SEED, GAMES, MAX_STEPS, SHARDS = 21000, 49000, 400, 500, 5
METRICS = ('active_win', 'win', 'draw', 'passive_win', 'suicide')


def play(arm, seed, games, max_steps, rule, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    import train_actor_critic as trainer
    from kill_search import search_action2, search_action3
    search = {'G_search2': search_action2, 'G_search3': search_action3}[arm]
    policy = evaluate.make_ac_fn(str(sd76.G), 9)
    agent = lambda env, obs, aid: search(env, aid, trainer.get_action_mask(env, aid), policy(env, obs, aid))[0]
    done = [0]

    def advance(value, total):
        ticks.put(value - done[0]); done[0] = value
    start = time.time()
    result = evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=advance,
                                      sudden_death=rule == 'E', **sd5.SCHEDULE_E)
    return arm, time.time() - start, result['games']


def run_arms(arms, seed, games, steps, shards, workers, desc, rule='E'):
    jobs = [(arm, seed + i * games // 2 // shards, games // shards, steps, rule) for arm in arms for i in range(shards)]
    parts = parallel_map(play, jobs, total=len(arms) * games, desc=desc, workers=workers)
    results = {arm: sorted([g for a, _, gs in parts if a == arm for g in gs], key=lambda g: (g['seed'], g['aid']))
               for arm in arms}
    seconds = {arm: sum(s for a, s, _ in parts if a == arm) / games for arm in arms}
    return results, seconds


def compare(results):
    from pilot_two_bomb import paired_active_bootstrap
    return {m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in results[a]]}
                                         for a in ('G_search3', 'G_search2'))) for m in METRICS}


def sd_effects(on, off):
    """Schedule E minus off on the same boards per arm, off search3 - search2, and the ActiveWR interaction."""
    from pilot_two_bomb import paired_active_bootstrap
    for arm in ARMS:
        if [(g['seed'], g['aid']) for g in on[arm]] != [(g['seed'], g['aid']) for g in off[arm]]:
            raise ValueError(f'{arm}: E and off games are not on the same boards')
    wrap = lambda rows, m: {'games': [{**g, 'active_win': g[m]} for g in rows]}
    gain = lambda arm: {'games': [{**e, 'active_win': e['active_win'] - o['active_win']} for e, o in zip(on[arm], off[arm])]}
    late = lambda rows: 100 * sum(g['active_win'] and g['steps'] > 475 for g in rows) / len(rows)
    return {'E_minus_off': {arm: {m: paired_active_bootstrap(wrap(on[arm], m), wrap(off[arm], m)) for m in METRICS}
                            for arm in ARMS},
            'off_search3_vs_search2': compare(off),
            'interaction_active_pp': paired_active_bootstrap(gain('G_search3'), gain('G_search2')),
            'active_win_after_step_475_rate': {arm: {'E': late(on[arm]), 'off': late(off[arm])} for arm in ARMS}}


def decide(paired):
    return {'search3_minus_search2': paired['active_win']['difference_pp'],
            'open_test_49000': paired['active_win']['difference_pp'] >= 2.0}


def _check_g():
    if sd5._hash(sd76.G) != sd76.G_SHA256:
        raise ValueError('G checkpoint changed')


def validate(output, workers=5, smoke=False, base=None):
    _check_g()
    output = Path(output)
    output.mkdir(parents=True)  # FileExistsError: never overwrite
    games, steps, shards = (4, 40, 2) if smoke else (GAMES, MAX_STEPS, SHARDS)
    arms = ('G_search3',) if base else ARMS
    results, seconds = run_arms(arms, VALIDATION_SEED, games, steps, shards, workers, f'SD8B validation {VALIDATION_SEED}')
    if base:
        results['G_search2'] = json.loads((Path(base) / f'evaluation_G_search2_{VALIDATION_SEED}.json').read_text(encoding='utf-8'))['games']
        if len(results['G_search2']) != games:
            raise ValueError('base run has a different number of games')
    for arm, rows in results.items():
        (output / f'evaluation_{arm}_{VALIDATION_SEED}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    paired = compare(results)
    summary = {'seed': VALIDATION_SEED, 'games': games, 'base': str(base) if base else None,
               'seconds_per_game': seconds, 'results': {k: summarize(v) for k, v in results.items()},
               'search3_vs_search2': paired, 'decision': decide(paired)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def test(output, validation, workers=5, smoke=False):
    _check_g()
    decision = json.loads((Path(validation) / 'summary.json').read_text(encoding='utf-8'))['decision']
    if not decision['open_test_49000'] and not smoke:
        raise ValueError(f'validation rule failed ({decision}); the test seed stays closed')
    output = Path(output)
    output.mkdir(parents=True)
    seed, games, steps, shards = (VALIDATION_SEED, 4, 40, 2) if smoke else (TEST_SEED, GAMES, MAX_STEPS, SHARDS)
    with open(output / 'test_opened.json', 'x', encoding='utf-8') as handle:  # the test seed is used once
        json.dump({'seed': seed, 'games': games, 'G_sha256': sd76.G_SHA256, 'validation': str(validation)}, handle, indent=2)
    results, seconds = run_arms(ARMS, seed, games, steps, shards, workers, f'SD8B test seed {seed}')
    for arm, rows in results.items():
        (output / f'evaluation_{arm}_E_{seed}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    paired = compare(results)
    summary = {'seed': seed, 'games': games, 'seconds_per_game': seconds,
               'results': {k: summarize(v) for k, v in results.items()},
               'search3_vs_search2': paired, 'gate': gate(paired)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not smoke:
        shutil.make_archive(str(output) + '-send', 'zip', output, '.')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def ablate(output, test_dir, workers=5, smoke=False):
    _check_g()
    output = Path(output)
    output.mkdir(parents=True)
    seed, games, steps, shards = (VALIDATION_SEED, 4, 40, 2) if smoke else (TEST_SEED, GAMES, MAX_STEPS, SHARDS)
    on = {arm: json.loads((Path(test_dir) / f'evaluation_{arm}_E_{seed}.json').read_text(encoding='utf-8'))['games']
          for arm in ARMS}
    off, seconds = run_arms(ARMS, seed, games, steps, shards, workers, f'SD8B ablation off seed {seed}', rule='off')
    for arm, rows in off.items():
        (output / f'evaluation_{arm}_off_{seed}.json').write_text(json.dumps({'games': rows}), encoding='utf-8')
    summary = {'seed': seed, 'games': games, 'test': str(test_dir), 'seconds_per_game_off': seconds,
               'results': {f'{arm}_{rule}': summarize(rows[arm]) for rule, rows in (('E', on), ('off', off)) for arm in ARMS},
               **sd_effects(on, off)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('validate', 'test', 'ablate'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--base', type=Path, help='validate: earlier run holding G+search2 games on seed 21000')
    p.add_argument('--validation', type=Path, help='test: the validate output whose rule must have passed')
    p.add_argument('--test', type=Path, help='ablate: the test output holding the schedule-E games')
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.stage == 'validate':
        validate(a.output, a.workers, a.smoke, a.base)
    elif a.stage == 'ablate':
        if not a.test:
            p.error('ablate needs --test')
        ablate(a.output, a.test, a.workers, a.smoke)
    else:
        if not a.validation:
            p.error('test needs --validation')
        test(a.output, a.validation, a.workers, a.smoke)
