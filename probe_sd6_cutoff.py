"""SD6.1: does the opponent-aware escape rule add active wins on top of F? Validation only.

Schedule E (engine sudden death 475/25/40), both corners, CNN action sampled as usual:
  f       F unchanged
  cutoff  F, except when `cutoff_teacher.trap_actions` finds a safe move that leaves the
          opponent no escape and F's sampled action is not one of them
Arms: seed 21000, 400 games each. Profile (seed 15000, 400 games, F + kill-search): how often
the rule picks the move kill-search picked when search overrides F with a killing move.
SD6.1b (rule fixed after SD6.1 matched 0/85 search kills: compare against the learner's other
allowed moves, not against staying put); same seeds and decision rule.
Decision rule (PLAN_SUDDEN_DEATH.md SD6.1, fixed before running): open SD6.2 if
cutoff - f >= +3pp ActiveWR with the paired CI95 lower bound > 0.
"""
import argparse
import json
from pathlib import Path

from progress import parallel_map

F = 'runs/phaseC-sd-best-fallback/best_phaseC.pt'
SCHEDULE_E = dict(sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
ARMS = ('f', 'cutoff')
SEED, PROFILE_SEED, GAMES, MAX_STEPS = 21000, 15000, 400, 500


def shards(seed, games, n):
    if seed >= 31000:
        raise ValueError('probe must stay on validation/diagnostic seeds')
    cuts = [games // 2 * i // n for i in range(n + 1)]  # split boards (2 corners each), not games
    return [(seed + cuts[i], 2 * (cuts[i + 1] - cuts[i])) for i in range(n)]


def with_cutoff(policy):
    from cutoff_teacher import trap_actions
    from train_actor_critic import get_action_mask

    def agent(env, obs, aid):
        action = policy(env, obs, aid)
        traps = trap_actions(env, aid, get_action_mask(env, aid))
        return action if not traps or action in traps else traps[0]
    return agent


def _ticker(ticks):
    def advance(done, total):
        if done:
            ticks.put(1)
    return advance


def evaluate_arm(arm, seed, games, max_steps, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    policy = evaluate.make_ac_fn(F, 9)
    agent = policy if arm == 'f' else with_cutoff(policy)
    result = evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=_ticker(ticks),
                                      sudden_death=True, contact_diagnostics=True, **SCHEDULE_E)
    ticks.put(1)  # the last game
    result['arm'] = arm
    return result


def profile(seed, games, max_steps, ticks):
    """F + kill-search; at each override with a killing move, does trap_actions contain it?"""
    import torch
    torch.set_num_threads(1)
    import evaluate
    from bomberman_env import ACTION_BOMB
    from cutoff_teacher import trap_actions
    from kill_search import search_action
    from train_actor_critic import get_action_mask
    policy = evaluate.make_ac_fn(F, 9)
    rows = dict(games=games, overrides=0, kill_bomb_overrides=0, move_kill_overrides=0, move_kill_hits=0,
                rule_fires=0, rule_fires_search_agrees=0)

    def agent(env, obs, aid):
        mask = get_action_mask(env, aid)
        action = policy(env, obs, aid)
        choice, info = search_action(env, aid, mask, action)
        traps = trap_actions(env, aid, mask)
        if traps and action not in traps:
            rows['rule_fires'] += 1
            rows['rule_fires_search_agrees'] += int(choice in traps)
        if choice != action:
            rows['overrides'] += 1
            if max(info['scores'].values()) == 1.0:
                if choice == ACTION_BOMB:
                    rows['kill_bomb_overrides'] += 1
                else:
                    rows['move_kill_overrides'] += 1
                    rows['move_kill_hits'] += int(choice in traps)
        return choice
    result = evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=_ticker(ticks),
                                      sudden_death=True, **SCHEDULE_E)
    ticks.put(1)
    rows['active_win_rate'] = result['active_win_rate']
    return rows


def _job(kind, seed, games, max_steps, ticks):
    return (profile if kind == 'profile' else evaluate_arm)(*((seed,) if kind == 'profile' else (kind, seed)),
                                                           games, max_steps, ticks)


def decide(paired):
    return paired['difference_pp'] >= 3.0 and paired['ci95_pp'][0] > 0


def summarize(jobs, parts, games):
    from probe_sd5_search import merge
    from pilot_two_bomb import paired_active_bootstrap
    results = {arm: merge([r for (k, *_), r in zip(jobs, parts) if k == arm]) for arm in ARMS}
    prof = [r for (k, *_), r in zip(jobs, parts) if k == 'profile']
    prof = {k: sum(r[k] for r in prof) for k in prof[0] if k != 'active_win_rate'} | {
        'active_win_rate': sum(r['active_win_rate'] * r['games'] for r in prof) / games}
    prof['move_kill_recall'] = prof['move_kill_hits'] / max(prof['move_kill_overrides'], 1)
    prof['rule_precision_vs_search'] = prof['rule_fires_search_agrees'] / max(prof['rule_fires'], 1)
    paired = {m: paired_active_bootstrap(*({'games': [{**g, 'active_win': g[m]} for g in results[arm]['games']]}
                                           for arm in ('cutoff', 'f'))) for m in ('active_win', 'win', 'draw', 'suicide')}
    summary = {'seed': SEED, 'games': games, 'results': {k: {m: v for m, v in r.items() if m not in ('games', 'contact')}
                                                          for k, r in results.items()},
               'paired_cutoff_vs_f': paired, 'decision_open_sd6_2': decide(paired['active_win']),
               'profile': {'seed': PROFILE_SEED, **prof}}
    return results, summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--workers', type=int, default=5)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True)
    # The kill-search profile is ~4x slower per game, so it gets more shards.
    jobs = ([('f', s, g, MAX_STEPS) for s, g in shards(SEED, GAMES, 2)]
            + [('cutoff', s, g, MAX_STEPS) for s, g in shards(SEED, GAMES, 2)]
            + [('profile', s, g, MAX_STEPS) for s, g in shards(PROFILE_SEED, GAMES, 4)])
    parts = parallel_map(_job, jobs, total=3 * GAMES, desc='SD6.1 probe', workers=a.workers)
    # Raw shards first: a failure while summarizing must not lose the games.
    (a.output_dir / 'parts.json').write_text(json.dumps({'jobs': jobs, 'parts': parts}), encoding='utf-8')
    results, summary = summarize(jobs, parts, GAMES)
    for arm, r in results.items():
        (a.output_dir / f'evaluation_{arm}_{SEED}.json').write_text(json.dumps(r), encoding='utf-8')
    (a.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
