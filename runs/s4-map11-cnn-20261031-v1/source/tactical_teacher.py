"""Bounded attack labels; never selects the learner's actual action."""
from collections import Counter, deque
from time import perf_counter

import numpy as np

from bomberman_env import Bomb, ACTION_STOP, ACTION_BOMB, _MOVE_DELTA, POWERUP_RANGE, MAX_BLAST_RANGE
from rule_based_bot import _reachable_cells


def _learner_blockers(env, aid, blocked):
    blocked = blocked.copy()
    agent = env.agents[aid]
    for enemy in env.agents:
        if not enemy.alive or enemy.agent_id == aid:
            continue
        blocked[:, enemy.row, enemy.col] = True
        for dr, dc in _MOVE_DELTA.values():
            r, c = enemy.row + dr, enemy.col + dc
            if (0 <= r < env.board_size and 0 <= c < env.board_size
                    and (r, c) != (agent.row, agent.col)):
                blocked[0, r, c] = True  # simultaneous movement can collide
    # A range pickup would change the planned blast, so omit that route.
    if agent.blast_range < MAX_BLAST_RANGE:
        for pos, kind in env.powerups.items():
            if kind == POWERUP_RANGE:
                blocked[:, pos[0], pos[1]] = True
    return blocked


def _placement_routes(env, aid, start, start_tick, forecast, max_moves, mask=None):
    """Earliest legal placement at each tile / first action, including slot waits."""
    burns, blocked, slots = forecast
    quiet = not burns.any()
    blocked = _learner_blockers(env, aid, blocked)
    queue = deque([(start, start_tick, 0, ())])
    seen = {}
    placements = {}
    found = set()
    moves = {ACTION_STOP: (0, 0), **_MOVE_DELTA}
    while queue:
        pos, tick, distance, path = queue.popleft()
        first = path[0] if path and mask is not None else None
        key = (pos, tick, first)
        if distance >= seen.get(key, max_moves + 1) or tick >= len(burns):
            continue
        seen[key] = distance
        placement_first = path[0] if path else ACTION_BOMB
        candidate = (pos, placement_first if mask is not None else None)
        if (candidate not in found and slots[tick, aid] > 0
                and not blocked[tick, pos[0], pos[1]] and not burns[tick][pos]
                and not (mask is not None and quiet and pos == start and distance > 0)
                and (mask is None or mask[placement_first])):
            placements.setdefault((tick, pos), []).append(path + (ACTION_BOMB,))
            found.add(candidate)
        for action, (dr, dc) in moves.items():
            if (action == ACTION_STOP and mask is not None and slots[tick, aid] > 0
                    and quiet):
                continue  # no reason to postpone the first bomb on a quiet board
            length = distance + int(action != ACTION_STOP)
            if length > max_moves or (not path and mask is not None and not mask[action]):
                continue
            dest = (pos[0] + dr, pos[1] + dc)
            if not (0 <= dest[0] < env.board_size and 0 <= dest[1] < env.board_size):
                continue
            if burns[tick][dest] or (action != ACTION_STOP and blocked[tick][dest]):
                continue
            queue.append((dest, tick + 1, length, path + (action,)))
    return [(tick, pos, paths) for (tick, pos), paths in placements.items()]


def get_two_bomb_plan(env, aid, mask):
    """Return an equally weighted set of best first actions, or None for local fallback.

    shortcut: forecasts known bombs and fixed enemy occupancy for learner exits;
    repeat with opponent models before claiming robustness to future counter-bombs.
    """
    started = perf_counter()
    reasons = Counter()
    stats = {"plans": 0, "accepted": 0, "two_bomb_labels": 0, "pressure": 0.0,
             "rejected": reasons, "milliseconds": 0.0}
    agent = env.agents[aid]
    enemies = [a for a in env.agents if a.alive and a.agent_id != aid]
    best = None
    best_score = None
    baseline = {}

    def finish():
        stats["milliseconds"] = (perf_counter() - started) * 1000
        stats["rejected"] = dict(reasons)
        if best is not None:
            best["first_actions"] = sorted(best["first_actions"])
            stats.update(accepted=1, two_bomb_labels=int(len(best["bombs"]) == 2),
                         pressure=best["pressure"])
        return best, stats

    if not agent.alive or not enemies:
        reasons["no_enemy"] += 1
        return finish()
    horizon = max([0, *(b.timer for b in env.bombs)]) + env.flame_duration + 3
    base = env._forecast_blasts(min_horizon=horizon, return_slots=True)
    known_fire = np.flatnonzero(base[0].any(axis=(1, 2)))
    baseline_end = int(known_fire[-1]) + 1 if len(known_fire) else 1
    first_routes = _placement_routes(env, aid, (agent.row, agent.col), 0, base, 2, mask)

    def priority(route):
        tick, pos, paths = route
        distance = min(abs(pos[0] - a.row) + abs(pos[1] - a.col) for a in enemies)
        return distance, min(sum(a in _MOVE_DELTA for a in p) for p in paths), tick, pos

    # Eight first placements, each with a single and up to seven follow-ups: <=64.
    first_routes.sort(key=priority)
    if len(first_routes) > 8:
        reasons["search_limit"] += len(first_routes) - 8
    first_routes = first_routes[:8]

    def evaluate(schedule, pos, paths, forecast):
        nonlocal best, best_score
        stats["plans"] += 1
        burns, blocked, _ = forecast
        last_fire = int(np.flatnonzero(burns.any(axis=(1, 2)))[-1]) + 1
        ticks = max(baseline_end, last_fire)
        burns, blocked = burns[:ticks], blocked[:ticks]
        end_tick = schedule[-1][0]
        learner_blocked = _learner_blockers(env, aid, blocked)
        remaining = len(burns) - end_tick - 1
        if burns[end_tick][pos] or (remaining and not _reachable_cells(
                pos, burns[end_tick + 1:], learner_blocked[end_tick + 1:], remaining).any()):
            reasons["learner_dies"] += 1
            return
        if ticks not in baseline:
            original = env._forecast_blasts(min_horizon=ticks)
            baseline[ticks] = [np.count_nonzero(_reachable_cells(
                (a.row, a.col), *original, ticks)) for a in enemies]
        before = baseline[ticks]
        after = [np.count_nonzero(_reachable_cells((a.row, a.col), burns, blocked, ticks))
                 for a in enemies]
        live = [(b, a) for b, a in zip(before, after) if b > 0]
        if not live:
            reasons["enemy_already_doomed"] += 1
            return
        pressure = max(1.0 - a / b for b, a in live)
        trapped = sum(a == 0 for _, a in live)
        if pressure <= 0:
            reasons["no_reduction"] += 1
            return
        for path in paths:
            if not mask[path[0]]:
                reasons["masked_first_action"] += 1
                continue
            score = (trapped, round(pressure, 10), -len(schedule),
                     -sum(a in _MOVE_DELTA for a in path), -len(path))
            if best_score is None or score > best_score:
                best_score = score
                best = {"first_actions": {path[0]}, "bombs": schedule, "path": path,
                        "pressure": pressure, "trapped": trapped, "before": before,
                        "after": after, "horizon": ticks}
            elif score == best_score:
                best["first_actions"].add(path[0])

    for tick, pos, prefixes in first_routes:
        bomb = Bomb(*pos, aid, env.bomb_timer, agent.blast_range)
        schedule = ((tick, bomb),)
        single = env._forecast_blasts(scheduled_bombs=schedule,
                                     min_horizon=tick + env.bomb_timer + env.flame_duration + 3,
                                     return_slots=True)
        evaluate(schedule, pos, prefixes, env._forecast_blasts(
            scheduled_bombs=schedule, return_slots=True))
        second_routes = _placement_routes(env, aid, pos, tick + 1, single, 3)
        second_routes.sort(key=priority)
        if len(second_routes) > 7:
            reasons["search_limit"] += len(second_routes) - 7
        for next_tick, dest, suffixes in second_routes[:7]:
            two = schedule + ((next_tick, Bomb(*dest, aid, env.bomb_timer, agent.blast_range)),)
            try:
                forecast = env._forecast_blasts(scheduled_bombs=two, return_slots=True)
            except ValueError:
                stats["plans"] += 1
                reasons["illegal_placement"] += 1
                continue
            paths = [prefix + suffix for prefix in prefixes for suffix in suffixes]
            evaluate(two, dest, paths, forecast)
    if not first_routes:
        reasons["no_safe_route"] += 1
    return finish()
