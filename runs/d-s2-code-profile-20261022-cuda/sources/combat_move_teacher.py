"""Certify an existing local MOVE label before assigning it to combat loss."""
from copy import deepcopy
from time import perf_counter

from bomberman_env import ACTION_BOMB, ACTION_STOP, _MOVE_DELTA
from rule_based_bot import _safe_action_mask, _bfs_safe_cell, _bomb_escape_pressure


def certify_combat_move(env, aid, action, mask):
    """One MOVE, one BOMB, then a masked escape with other agents stationary.

    shortcut: bounded greedy certificate; reject to navigation if it fails.
    Known bombs are simulated; future enemy moves/placements are not certified.
    """
    started = perf_counter()
    result = {'accepted': False, 'reason': None, 'pressure_here': 0.0,
              'pressure_next': 0.0, 'pressure_gain': 0.0, 'path': [], 'milliseconds': 0.0}

    def finish(reason):
        result['reason'] = reason
        result['accepted'] = reason == 'accepted'
        result['milliseconds'] = (perf_counter() - started) * 1000
        return result

    agent = env.agents[aid]
    if action not in _MOVE_DELTA or not mask[action]:
        return finish('not_allowed_move')
    if not agent.alive or not any(a.alive and a.agent_id != aid for a in env.agents):
        return finish('no_live_opponent')
    if mask[ACTION_BOMB]:
        result['pressure_here'] = _bomb_escape_pressure(env, aid, agent.row, agent.col)
    clone = deepcopy(env)
    delta = _MOVE_DELTA[action]
    expected = agent.row + delta[0], agent.col + delta[1]

    def step(selected):
        actions = [ACTION_STOP] * clone.num_agents
        actions[aid] = selected
        return clone.step(actions)[3]

    step(action)
    learner = clone.agents[aid]
    if not learner.alive or (learner.row, learner.col) != expected:
        return finish('move_failed')
    next_mask = _safe_action_mask(clone, aid)
    if not next_mask[ACTION_BOMB]:
        return finish('next_bomb_masked')
    result['pressure_next'] = _bomb_escape_pressure(clone, aid, learner.row, learner.col)
    result['pressure_gain'] = result['pressure_next'] - result['pressure_here']
    if result['pressure_next'] <= 0 or result['pressure_gain'] <= 1e-6:
        return finish('no_pressure_increase')
    info = step(ACTION_BOMB)
    if aid not in info['bombs_placed'] or not learner.alive:
        return finish('placement_failed')
    result['path'] = [action, ACTION_BOMB]
    horizon = max([0, *clone.flames.values(),
                   *(b.timer + clone.flame_duration - 1 for b in clone.bombs)])
    if horizon > 32:
        return finish('horizon_limit')
    for _ in range(horizon):
        if not clone.bombs and not clone.flames:
            break
        escape_mask = _safe_action_mask(clone, aid)
        danger = clone._danger_map()
        if danger[learner.row, learner.col] == 0 and escape_mask[ACTION_STOP]:
            escape = ACTION_STOP
        else:
            escape = _bfs_safe_cell(clone, learner.row, learner.col, danger, action_mask=escape_mask)
        if escape is None or escape == ACTION_BOMB or not escape_mask[escape]:
            return finish('escape_not_certified')
        step(escape)
        result['path'].append(escape)
        if not learner.alive:
            return finish('escape_died')
    if clone.bombs or clone.flames:
        return finish('hazards_remaining')
    return finish('accepted')
