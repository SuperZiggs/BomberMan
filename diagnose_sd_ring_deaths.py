"""SD4.0: why does the learner die to closing rings? Validation-only replay diagnostics.

Wraps the sudden-death mask during a normal `evaluate_case` run (same games, same RNG) and
records, for every agent and every step inside the warning window, whether it stands on the
next closing ring, whether any ring-safe move still exists, and BFS distance to the inside.
At the first step a doomed agent loses every ring-safe move, it re-checks feasibility with one
obstacle class removed at a time to name what closed the last route.
"""
import argparse
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_sudden_death_experiment as runner
from bomberman_env import EMPTY, WOODEN_WALL, _MOVE_DELTA
from run_sd_training_pilot import SD2
from sudden_death_experiment import _ring_survival

FORBIDDEN_TEST_SEEDS = range(31000, 50000)  # test seeds, used or reserved


def feasible(env, aid):
    """Some stay/move keeps a ring-and-bomb-safe continuation (ignores the base mask)."""
    safe, blocked, closing = _ring_survival(env, aid)
    agent = env.agents[aid]
    pos = agent.row, agent.col
    if closing[pos]:
        return False
    if safe[pos]:
        return True
    for dr, dc in _MOVE_DELTA.values():
        r, c = agent.row + dr, agent.col + dc
        if 0 <= r < env.board_size and 0 <= c < env.board_size and safe[r, c] and not blocked[r, c]:
            return True
    return False


def inside_distance(env, aid, cells):
    """BFS steps through floor (bombs/agents ignored) to a tile outside `cells`; None if walled in."""
    agent = env.agents[aid]
    start = agent.row, agent.col
    if start not in cells:
        return 0
    seen, queue = {start}, deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        for dr, dc in _MOVE_DELTA.values():
            nxt = r + dr, c + dc
            if nxt in seen or env.grid[nxt] != EMPTY:
                continue
            if nxt not in cells:
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def blocking_causes(env, aid):
    """Obstacle classes whose removal alone restores a ring-safe move."""
    enemy = [a.agent_id for a in env.agents if a.agent_id != aid]
    variants = {
        'opponent_body': lambda e: [setattr(e.agents[i], 'alive', False) for i in enemy],
        'opponent_bombs': lambda e: setattr(e, 'bombs', [b for b in e.bombs if b.owner == aid]),
        'own_bombs': lambda e: setattr(e, 'bombs', [b for b in e.bombs if b.owner != aid]),
        'flames': lambda e: e.flames.clear(),
        'wood': lambda e: e.grid.__setitem__(e.grid == WOODEN_WALL, EMPTY),
    }
    out = []
    for name, edit in variants.items():
        copy = deepcopy(env)
        edit(copy)
        if feasible(copy, aid):
            out.append(name)
    return out or ['none_single']


def run_diagnosis(games, seed, timings=SD2, max_steps=500):
    if seed in FORBIDDEN_TEST_SEEDS or seed + games // 2 - 1 in FORBIDDEN_TEST_SEEDS:
        raise ValueError('Diagnostics must stay off test seeds')
    original = runner.sudden_death_mask
    per_game, current = [], [None]

    def recorded(env, aid, base_mask, diagnostics=None, **kwargs):
        if current[0] is not env:
            current[0] = env
            per_game.append({})
        clocks = env._closure_ticks()
        if clocks and min(clocks.values()) <= env.sudden_death_warning_steps:
            ring = min(clocks, key=clocks.get)
            cells = env._ring_cells(ring)
            agent = env.agents[aid]
            ok = feasible(env, aid)
            # Learner's mask is queried twice per step (inside the bot mask and its own); keep one row.
            row = {'step': env.step_count, 'aid': aid, 'ring': ring, 'delay': clocks[ring],
                   'in_ring': (agent.row, agent.col) in cells, 'feasible': ok,
                   'distance': inside_distance(env, aid, cells)}
            rows = per_game[-1].setdefault(aid, {})
            if not ok and env.step_count not in rows:
                row['causes'] = blocking_causes(env, aid)
            rows.setdefault(env.step_count, row)
        return original(env, aid, base_mask, diagnostics, **kwargs)

    with patch.object(runner, 'sudden_death_mask', recorded):
        result = runner.evaluate_case('both_aware', games=games, seed=seed, max_steps=max_steps,
                                      timings=timings)
    if len(per_game) != len(result['games']):
        raise RuntimeError('Mask trace does not align with games')
    return result, per_game


def ring_events(result, per_game, timings=SD2):
    events = []
    for game, trace in zip(result['games'], per_game):
        learner = game['aid']
        sides = {learner: ('learner', game['learner_death']), 1 - learner: ('opponent', game['opponent_death'])}
        end_ring = 1 + (int(game['steps']) - timings['sudden_death_start']) // timings['sudden_death_interval']
        for aid, (side, death) in sides.items():
            rows = sorted(trace.get(aid, {}).values(), key=lambda r: r['step'])
            for ring in sorted({r['ring'] for r in rows}):
                window = [r for r in rows if r['ring'] == ring]
                died = death == 'sudden_death' and end_ring == ring
                inside = [r['delay'] for r in window if r['in_ring']]
                infeasible = [r for r in window if not r['feasible'] and r['delay'] > 1]
                at = {d: next((r for r in window if r['delay'] == d), None) for d in (10, 5, 2)}
                events.append({
                    'seed': game['seed'], 'learner_aid': learner, 'side': side, 'ring': ring, 'died': died,
                    'window_start_delay': window[0]['delay'], 'window_start_in_ring': window[0]['in_ring'],
                    'last_in_ring_delay': min(inside) if inside else None,
                    'steps_in_ring': len(inside),
                    'first_infeasible_delay': infeasible[0]['delay'] if infeasible else None,
                    'first_infeasible_causes': infeasible[0].get('causes') if infeasible else None,
                    **{f'in_ring_at_{d}': (r['in_ring'] if r else None) for d, r in at.items()},
                    **{f'distance_at_{d}': (r['distance'] if r else None) for d, r in at.items()}})
    return events


def bucket(delay):
    if delay is None:
        return 'never_in_ring'
    return next(name for low, name in ((20, '>=20'), (10, '10-19'), (5, '5-9'), (2, '2-4'), (1, '1'))
                if delay >= low)


def summarize(events):
    out = {}
    for side in ('learner', 'opponent'):
        rows = [e for e in events if e['side'] == side]
        for died in (True, False):
            group = [e for e in rows if e['died'] == died]
            key = f"{side}_{'died' if died else 'survived'}"
            out[key] = {
                'events': len(group),
                'by_ring': dict(Counter(e['ring'] for e in group)),
                'last_in_ring_delay': dict(Counter(bucket(e['last_in_ring_delay']) for e in group)),
                'in_ring_at_10': dict(Counter(str(e['in_ring_at_10']) for e in group)),
                'in_ring_at_5': dict(Counter(str(e['in_ring_at_5']) for e in group)),
                'had_infeasible_before_last_tick': sum(e['first_infeasible_delay'] is not None for e in group),
                'first_infeasible_delay': dict(Counter(bucket(e['first_infeasible_delay'])
                                                       for e in group if e['first_infeasible_delay'])),
                'first_infeasible_causes': dict(Counter(c for e in group
                                                        for c in (e['first_infeasible_causes'] or ()))),
                'walled_in_at_10': sum(bool(e['in_ring_at_10'] and e['distance_at_10'] is None) for e in group),
                'median_distance_at_10_in_ring': float(np.median(
                    [e['distance_at_10'] for e in group if e['in_ring_at_10'] and e['distance_at_10'] is not None]
                    or [np.nan]))}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=400)
    parser.add_argument('--seed', type=int, default=21000)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('runs') / ('sd4-diag-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True)  # refuses an existing folder
    result, per_game = run_diagnosis(args.games, args.seed)
    events = ring_events(result, per_game)
    summary = {'seed': args.seed, 'games': args.games, 'timings': SD2, 'checkpoint': str(runner.CHECKPOINT),
               'checkpoint_sha256': runner.digest(runner.CHECKPOINT),
               'model': {k: result[k] for k in ('win_rate', 'active_win_rate', 'draw_rate', 'suicide_rate')},
               'sudden_death': {k: v for k, v in result['sudden_death'].items() if k != 'mask'},
               'diagnosis': summarize(events)}
    runner.dump(args.output_dir / 'summary.json', summary)
    with gzip.open(args.output_dir / 'events.jsonl.gz', 'wt', encoding='utf-8') as f:
        f.writelines(json.dumps(e) + '\n' for e in events)
    print(json.dumps(summary['diagnosis'], indent=1))
