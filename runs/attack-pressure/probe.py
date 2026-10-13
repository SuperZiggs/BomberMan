"""Development probe; no checkpoint weights or live game state are changed."""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bomberman_env import Bomb, ACTION_BOMB, ACTION_STOP, _MOVE_DELTA
from evaluate import evaluate_seeded
from rule_based_bot import _build_danger_map, _bfs_safe_cell
from train_actor_critic import get_action_mask, get_attack_action

torch.set_num_threads(1)


def reachable(env, start, burns, blocked, ticks):
    positions = np.zeros_like(burns[0])
    positions[start] = True
    for tick in range(ticks):
        if tick < len(burns):
            damage, obstacles = burns[tick], blocked[tick]
        else:
            damage = np.zeros_like(burns[0])
            obstacles = blocked[-1] & ~burns[-1]
        moves = np.zeros_like(positions)
        moves[1:] |= positions[:-1]
        moves[:-1] |= positions[1:]
        moves[:, 1:] |= positions[:, :-1]
        moves[:, :-1] |= positions[:, 1:]
        positions = (positions | (moves & ~obstacles)) & ~damage
    return positions.sum()


def pressure(env, aid, r, c, base):
    agent = env.agents[aid]
    if not env._is_walkable(r, c):
        return 0.0
    bomb = Bomb(r, c, aid, env.bomb_timer, agent.blast_range)
    burns, blocked = env._forecast_blasts(bomb)
    hit_ticks = np.flatnonzero(burns[:, r, c])
    if not len(hit_ticks):
        return 0.0
    ticks = int(hit_ticks[0]) + 1
    scores = []
    for enemy in env.agents:
        if not enemy.alive or enemy.agent_id == aid:
            continue
        start = (enemy.row, enemy.col)
        before = reachable(env, start, *base, ticks)
        after = reachable(env, start, burns, blocked, ticks)
        scores.append(max(0.0, 1 - after / before) if before else 0.0)
    return max(scores, default=0.0)


def target(env, aid, danger, mask):
    agent = env.agents[aid]
    r, c = agent.row, agent.col
    close = any(a.alive and a.agent_id != aid and
                abs(a.row-r) + abs(a.col-c) <= agent.blast_range + 1 for a in env.agents)
    if danger[r, c] == 0 and close:
        base = env._forecast_blasts()
        here = pressure(env, aid, r, c, base) if mask[ACTION_BOMB] else 0.0
        best, move = here, None
        occupied = {(a.row, a.col) for a in env.agents if a.alive and a.agent_id != aid}
        for action, (dr, dc) in _MOVE_DELTA.items():
            pos = (r+dr, c+dc)
            if not mask[action] or pos in occupied or danger[pos]:
                continue
            value = pressure(env, aid, *pos, base)
            if value > best + 1e-6:
                best, move = value, action
        if move is not None:
            return move
        if here > 0:
            return ACTION_BOMB
    return get_attack_action(env, aid, danger, mask)


def teacher(improved):
    def choose(env, obs, aid):
        a = env.agents[aid]
        danger = _build_danger_map(env)
        mask = get_action_mask(env, aid)
        if danger[a.row, a.col]:
            action = _bfs_safe_cell(env, a.row, a.col, danger)
        else:
            action = (target if improved else get_attack_action)(env, aid, danger, mask)
        if action is not None and mask[action]:
            return action
        return next((i for i in range(ACTION_BOMB) if mask[i]), ACTION_STOP)
    return choose


if __name__ == '__main__':
    results = {}
    for name, improved in [('original_teacher', False), ('pressure_teacher', True)]:
        print(name, flush=True)
        result = evaluate_seeded(teacher(improved), board_size=9, num_agents=2,
                                 num_episodes=40, seed=6300, max_steps=500)
        results[name] = result
        print(json.dumps({k:v for k,v in result.items() if k not in ('games','by_corner')}), flush=True)
    (Path(__file__).parent/'teacher_probe.json').write_text(json.dumps(results,indent=2))
