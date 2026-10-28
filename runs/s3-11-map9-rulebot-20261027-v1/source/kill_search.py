"""Short-horizon kill search with the fixed RuleBot as the opponent model (SD4.6 probe).

For each action the mask allows, simulate `horizon` ticks on a copy: the learner plays the
action, then only dodges (BFS to a safe tile, else STOP); every other agent plays RuleBot.
Score +1 if the learner's bomb kills an opponent while the learner is alive, -1 if the learner
dies, else 0. The default (CNN) action is kept unless another action scores strictly higher.

shortcut: perfect RuleBot model and dodge-only learner rollouts; an opponent other than
RuleBot, or plans needing a second learner bomb, need a real planner (MCTS) instead.
"""
from copy import deepcopy

import numpy as np

from bomberman_env import ACTION_BOMB, ACTION_STOP
from rule_based_bot import RuleBasedAgent, _bfs_safe_cell, _build_danger_map

HORIZON = 12   # bomb_timer 8 + flame 2 + slack
REACH = 6      # Manhattan distance beyond which no kill line fits the horizon
# Opponent model used inside rollouts: None = RuleBot (default), else fn(env, agent_id) -> action (stateless).
OPPONENT_MODEL = None


def _opponents(env, aid):
    if OPPONENT_MODEL is None:
        bots = {a.agent_id: RuleBasedAgent(a.agent_id) for a in env.agents if a.agent_id != aid}
        return lambda e, i: bots[i].select_action(e)
    return OPPONENT_MODEL


def _dodge(env, aid):
    agent = env.agents[aid]
    danger = _build_danger_map(env)
    if danger[agent.row, agent.col] > 0:
        move = _bfs_safe_cell(env, agent.row, agent.col, danger)
        if move is not None:
            return move
    return ACTION_STOP


def _rollout(env, aid, action, horizon):
    model = _opponents(env, aid)
    for tick in range(horizon):
        actions = [ACTION_STOP] * env.num_agents
        for other in env.agents:
            if other.alive and other.agent_id != aid:
                actions[other.agent_id] = model(env, other.agent_id)
        actions[aid] = action if tick == 0 else _dodge(env, aid)
        _, _, _, info = env.step(actions)
        if any(k == aid and v != aid for k, v in info['kills']) and env.agents[aid].alive:
            return 1.0
        if not env.agents[aid].alive:
            return -1.0
        if info['episode_done']:
            return 0.0
    return 0.0


def search_action(env, aid, mask, default_action, horizon=HORIZON):
    agent = env.agents[aid]
    near = any(a.alive and a.agent_id != aid and abs(a.row - agent.row) + abs(a.col - agent.col) <= REACH
               for a in env.agents)
    if not near:
        return default_action, {'skipped': True}
    state = np.random.get_state()  # RuleBot's rare random fallback must not shift the real game
    try:
        scores = {a: _rollout(deepcopy(env), aid, a, horizon) for a in range(len(mask)) if mask[a]}
    finally:
        np.random.set_state(state)
    best = max(scores.values(), default=0.0)
    if scores.get(default_action, -np.inf) >= best:
        return default_action, {'skipped': False, 'scores': scores}
    # Prefer the bomb among equally good overrides (it is the kill), then the lowest action id.
    choice = ACTION_BOMB if scores.get(ACTION_BOMB) == best else min(a for a, s in scores.items() if s == best)
    return choice, {'skipped': False, 'scores': scores}


def search_action2(env, aid, mask, default_action, horizon=HORIZON):
    """Two-move search (SD7.9b): when no single move kills, try every safe move followed by every
    legal second move next tick (then dodge). Returns the first move of a kill plan, else defers
    to the one-move result. About 6x the cost of `search_action` on steps where it fires."""
    from rule_based_bot import _safe_action_mask
    action, info = search_action(env, aid, mask, default_action, horizon)
    if info['skipped'] or max(info['scores'].values(), default=0.0) > 0:
        return action, info
    state = np.random.get_state()
    plans = {}
    try:
        for first, score in info['scores'].items():
            if score < 0:
                continue
            after = deepcopy(env)
            actions = [ACTION_STOP] * after.num_agents
            model = _opponents(after, aid)
            for a in after.agents:
                if a.alive and a.agent_id != aid:
                    actions[a.agent_id] = model(after, a.agent_id)
            actions[aid] = first
            _, _, _, step_info = after.step(actions)
            if step_info['episode_done'] or not after.agents[aid].alive:
                continue
            second = _safe_action_mask(after, aid)
            plans[first] = max((_rollout(deepcopy(after), aid, b, horizon - 1) for b in range(len(second)) if second[b]),
                               default=0.0)
    finally:
        np.random.set_state(state)
    if max(plans.values(), default=0.0) <= 0 or plans.get(default_action, -1.0) > 0:
        return action, {**info, 'plans': plans}
    return min(a for a, s in plans.items() if s > 0), {**info, 'plans': plans}


def _advance(env, aid, action):
    """Copy of env one tick later (learner plays `action`, others the opponent model), or None if the line ends."""
    after = deepcopy(env)
    actions = [ACTION_STOP] * after.num_agents
    model = _opponents(after, aid)
    for a in after.agents:
        if a.alive and a.agent_id != aid:
            actions[a.agent_id] = model(after, a.agent_id)
    actions[aid] = action
    _, _, _, step_info = after.step(actions)
    return None if step_info['episode_done'] or not after.agents[aid].alive else after


def _best_plan(env, aid, depth, horizon):
    """Best score of up to `depth` learner moves then dodging: 1 if some line kills, else the one-move best."""
    from rule_based_bot import _safe_action_mask
    mask = _safe_action_mask(env, aid)
    scores = {b: _rollout(deepcopy(env), aid, b, horizon) for b in range(len(mask)) if mask[b]}
    best = max(scores.values(), default=0.0)
    if depth == 1 or best > 0:
        return best
    for b, score in scores.items():
        after = _advance(env, aid, b) if score >= 0 else None
        if after is not None and _best_plan(after, aid, depth - 1, horizon - 1) > 0:
            return 1.0
    return best


def search_action_deep(env, aid, mask, default_action, depth=3, horizon=HORIZON):
    """Multi-move search (SD8B): like `search_action2`, but a kill plan may use up to `depth` learner
    moves before dodging. Shallower plans are tried first, so it returns `search_action2`'s answer
    whenever that one finds a kill. Cost grows ~6x per extra move on steps with no shorter kill."""
    action, info = search_action2(env, aid, mask, default_action, horizon)
    if depth <= 2 or 'plans' not in info or max(info['plans'].values(), default=0.0) > 0:
        return action, info
    state = np.random.get_state()
    plans = {}
    try:
        for first, score in info['scores'].items():
            after = _advance(env, aid, first) if score >= 0 else None
            if after is not None:
                plans[first] = _best_plan(after, aid, depth - 1, horizon - 1)
    finally:
        np.random.set_state(state)
    if max(plans.values(), default=0.0) <= 0 or plans.get(default_action, -1.0) > 0:
        return action, {**info, 'deep_plans': plans}
    return min(a for a, s in plans.items() if s > 0), {**info, 'deep_plans': plans}


def search_action3(env, aid, mask, default_action, horizon=HORIZON):
    return search_action_deep(env, aid, mask, default_action, 3, horizon)
