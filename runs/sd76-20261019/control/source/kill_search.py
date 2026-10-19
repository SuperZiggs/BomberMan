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


def _dodge(env, aid):
    agent = env.agents[aid]
    danger = _build_danger_map(env)
    if danger[agent.row, agent.col] > 0:
        move = _bfs_safe_cell(env, agent.row, agent.col, danger)
        if move is not None:
            return move
    return ACTION_STOP


def _rollout(env, aid, action, horizon):
    bots = {a.agent_id: RuleBasedAgent(a.agent_id) for a in env.agents if a.agent_id != aid}
    for tick in range(horizon):
        actions = [ACTION_STOP] * env.num_agents
        for other, bot in bots.items():
            if env.agents[other].alive:
                actions[other] = bot.select_action(env)
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
