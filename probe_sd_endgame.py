"""SD4.2a: can a teacher convert the sudden-death endgame into active wins? Validation only.

Same boards, RNG and B actions until `switch` (default step 360 = start of the ring-1 warning);
afterwards the learner seat is driven by another controller, all under SD2 + learner margin 15:
  b          B unchanged (control)
  local      local attack teacher (`get_attack_action`) when not in danger, else B
  two_bomb   two-bomb plan, else local teacher, when not in danger, else B
  rulebot    RuleBot (its own ring-aware mask, no margin)
  search     B's action unless `kill_search` finds a strictly better line (SD4.6)
Decision rule (PLAN_SUDDEN_DEATH.md, fixed before running): a teacher arm must raise ActiveWR
by >= 5pp over `b` on validation to justify training (SD4.2b).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

ARMS = ('b', 'local', 'two_bomb', 'rulebot')  # SD4.2a; SD4.6 adds 'search'
MARGIN, SEED, GAMES, SWITCH = 15, 21000, 400, 360
KEYS = ('win_rate', 'active_win_rate', 'passive_win_rate', 'draw_rate', 'suicide_rate', 'avg_kills')


def controller(arm, policy):
    import evaluate
    import train_actor_critic as trainer
    from tactical_teacher import get_two_bomb_plan
    from bomberman_env import ACTION_STOP
    bots = {}

    def act(env, obs, aid):
        if arm == 'rulebot':
            return bots.setdefault(aid, evaluate.make_rule_based_fn(aid))(env, obs, aid)
        if arm == 'search':
            from kill_search import search_action
            return search_action(env, aid, trainer.get_action_mask(env, aid), policy(env, obs, aid))[0]
        agent = env.agents[aid]
        danger = trainer._build_danger_map(env)
        if danger[agent.row, agent.col] == 0:
            mask = trainer.get_action_mask(env, aid)  # the patched SD + margin mask
            action = None
            if arm == 'two_bomb':
                plan, _ = get_two_bomb_plan(env, aid, mask)
                action = min(plan['first_actions']) if plan else None
            if action is None:
                action = trainer.get_attack_action(env, aid, danger, mask)
            if action is not None and mask[action]:
                return action
        return policy(env, obs, aid)
    return act


def evaluate_arm(arm, seed=SEED, games=GAMES, switch=SWITCH, timings=None, max_steps=500):
    import torch
    torch.set_num_threads(1)
    import evaluate
    import run_sudden_death_experiment as runner
    from run_sd_training_pilot import SD2
    if seed >= 31000:
        raise ValueError('probe must stay on validation seeds')
    original = evaluate.make_ac_fn

    def make(*args, **kwargs):
        policy = original(*args, **kwargs)
        other = controller(arm, policy)
        return lambda env, obs, aid: other(env, obs, aid) if env.step_count >= switch else policy(env, obs, aid)

    with patch.object(evaluate, 'make_ac_fn', make) if arm != 'b' else patch.object(evaluate, 'make_ac_fn', original):
        result = runner.evaluate_case('both_aware', games=games, seed=seed, max_steps=max_steps,
                                      timings=timings or SD2, learner_margin=MARGIN)
    result.update(arm=arm, switch=switch)
    return result


def merge(parts):
    """Concatenate board-seed shards; recompute the headline rates from per-game rows."""
    if len(parts) == 1:
        return parts[0]
    from collections import Counter
    games = [g for r in sorted(parts, key=lambda r: r['games'][0]['seed']) for g in r['games']]
    pct = lambda key: 100 * sum(g[key] for g in games) / len(games)
    out = {'arm': parts[0]['arm'], 'switch': parts[0]['switch'], 'games': games,
           'win_rate': pct('win'), 'active_win_rate': pct('active_win'), 'draw_rate': pct('draw'),
           'suicide_rate': pct('suicide'), 'avg_kills': sum(g['kills'] for g in games) / len(games),
           'sudden_death': {'learner_deaths': dict(Counter(g['learner_death'] for g in games)),
                            'opponent_deaths': dict(Counter(g['opponent_death'] for g in games))}}
    out['passive_win_rate'] = out['win_rate'] - out['active_win_rate']
    return out


def late_phase(result, switch):
    late = [g for g in result['games'] if g['steps'] > switch]
    return {'games_past_switch': len(late), 'active_wins_past_switch': sum(g['active_win'] for g in late),
            'kills_past_switch': sum(g['kills'] for g in late)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--arms', nargs='+', default=list(ARMS), help='first arm is the control')
    p.add_argument('--switches', nargs='+', type=int, help='one switch step per arm (default 360)')
    p.add_argument('--shards', nargs='+', type=int, help='seed shards per arm (default 1 each)')
    a = p.parse_args()
    switches = a.switches or [SWITCH] * len(a.arms)
    shards = a.shards or [1] * len(a.arms)
    a.output_dir.mkdir(parents=True)
    # Every game reseeds from its board seed, so board-seed shards reproduce one long run exactly.
    jobs = [(arm, SEED + i * GAMES // 2 // n, GAMES // n, sw)
            for arm, sw, n in zip(a.arms, switches, shards) for i in range(n)]
    with ProcessPoolExecutor(min(len(jobs), 5)) as pool:
        parts = list(pool.map(evaluate_arm, *zip(*jobs)))
    results = [merge([r for r, job in zip(parts, jobs) if job[0] == arm and job[3] == sw])
               for arm, sw in zip(a.arms, switches)]
    from pilot_two_bomb import paired_active_bootstrap
    base = results[0]
    rows = []
    for r in results:
        (a.output_dir / f"evaluation_{r['arm']}_s{r['switch']}_{SEED}.json").write_text(json.dumps(r), encoding='utf-8')
        row = {'arm': r['arm'], 'switch': r['switch'], **{k: r[k] for k in KEYS}, **late_phase(r, r['switch']),
               'learner_deaths': r['sudden_death']['learner_deaths'],
               'opponent_deaths': r['sudden_death']['opponent_deaths'],
               'active_vs_b': paired_active_bootstrap(r, base) if r is not base else None}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.output_dir / 'summary.json').write_text(json.dumps({'seed': SEED, 'games': GAMES, 'switches': switches,
        'margin': MARGIN, 'rows': rows}, indent=2), encoding='utf-8')
