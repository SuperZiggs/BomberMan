"""
rule_based_bot.py

Giai đoạn A — Rule-based agent dùng heuristic đơn giản:
  1. Né tới chỗ trú theo thời gian nổ và lửa, giữ đường thoát.
  2. Đặt bom nếu tia nổ thật trúng đối thủ và còn đường sống.
  3. Nhặt nâng cấp hữu ích ở gần trên đường an toàn.
  4. Phá gỗ / mở đường rồi tiếp cận đối thủ, chỉ chọn hành động qua safety mask.

Dùng hàm `select_action(obs, env, agent_id)` hoặc class `RuleBasedAgent`.

Script evaluation: 2 RuleBasedAgent đấu nhau 100 episode trên BombermanFFAEnv(num_agents=2),
in thống kê thắng/thua, độ dài trung bình, bom đặt trung bình.
"""

from __future__ import annotations

import sys
import numpy as np
from collections import deque
from typing import Optional

from bomberman_env import (
    BombermanFFAEnv,
    Bomb,
    EMPTY,
    WOODEN_WALL,
    NUM_ACTIONS,
    ACTION_STOP,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_BOMB,
    _MOVE_DELTA,
    POWERUP_AMMO,
    POWERUP_RANGE,
    MAX_AMMO,
    MAX_BLAST_RANGE,
)


# ─────────────────────────── helper: danger map ───────────────────────────

def _build_danger_map(env: BombermanFFAEnv) -> np.ndarray:
    """Earliest damage tick, including shortened fuses from chain reactions."""
    return env._danger_map()


def _survival_map(env, start_r, start_c, extra_bomb=None, avoid_contests=True,
                  safety_margin=0, reserve_exits=False):
    """Backward reachability: survive all known flames without losing the exit."""
    burns, blocked = env._forecast_blasts(extra_bomb)
    actual_burns = burns.copy() if safety_margin else burns
    for shift in range(1, min(safety_margin + 1, len(burns))):
        burns[:-shift] |= actual_burns[shift:]
    for a in env.agents:
        if a.alive and (a.row, a.col) != (start_r, start_c):
            blocked[:, a.row, a.col] = True
            if avoid_contests:
                tick = min(1 if extra_bomb is not None else 0, len(blocked) - 1)
                for dr, dc in _MOVE_DELTA.values():
                    pos = (a.row + dr, a.col + dc)
                    if pos != (start_r, start_c) and env._is_walkable(*pos):
                        # A new bomb must not rely on the opponent leaving its doorway.
                        if extra_bomb is not None and reserve_exits:
                            blocked[tick:, pos[0], pos[1]] = True
                        else:
                            blocked[tick, pos[0], pos[1]] = True
    safe = np.zeros_like(burns)
    next_safe = np.ones_like(burns[0])
    for tick in range(len(burns) - 1, -1, -1):
        continuation = next_safe.copy()  # staying, including on one's own bomb
        if tick + 1 < len(burns):
            dest = next_safe & ~blocked[tick + 1]
            continuation[1:, :] |= dest[:-1, :]
            continuation[:-1, :] |= dest[1:, :]
            continuation[:, 1:] |= dest[:, :-1]
            continuation[:, :-1] |= dest[:, 1:]
        safe[tick] = ~burns[tick] & continuation
        next_safe = safe[tick]
    return safe, blocked, burns


def _safe_action_mask(env: BombermanFFAEnv, agent_id: int, avoid_contests=True,
                      safety_margin=1, robust=True) -> list[bool]:
    if not robust:
        safety_margin = 0
    agent = env.agents[agent_id]
    mask = [False] * NUM_ACTIONS
    if not agent.alive:
        mask[ACTION_STOP] = True
        return mask
    r, c = agent.row, agent.col
    safe, blocked, burns = _survival_map(env, r, c, avoid_contests=avoid_contests,
                                        safety_margin=safety_margin)
    mask[ACTION_STOP] = bool(safe[0, r, c])
    for action, (dr, dc) in _MOVE_DELTA.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.board_size and 0 <= nc < env.board_size:
            mask[action] = bool(not blocked[0, nr, nc] and safe[0, nr, nc])
    if burns[:, r, c].any():
        # Leave promptly: a merely possible last-second exit is easily blocked.
        exits = _escape_first_actions(env, r, c, safe, blocked, burns, robust=robust)
        if exits:
            mask = [a in exits for a in range(NUM_ACTIONS)]
    else:
        blast = env._blast_cells(r, c, agent.blast_range)
        targets = any(env.grid[pos] == WOODEN_WALL for pos in blast)
        for other in env.agents:
            if other.alive and other.agent_id != agent_id:
                reachable = {(other.row, other.col)}
                reachable.update((other.row + dr, other.col + dc) for dr, dc in _MOVE_DELTA.values()
                                 if env._is_walkable(other.row + dr, other.col + dc))
                targets |= bool(reachable.intersection(blast))
        mask[ACTION_BOMB] = targets and _can_bomb_and_escape(env, agent_id, r, c, None, robust=robust)
    if not any(mask) and safety_margin:
        return _safe_action_mask(env, agent_id, avoid_contests=avoid_contests, safety_margin=0, robust=robust)
    if not any(mask) and avoid_contests:
        return _safe_action_mask(env, agent_id, avoid_contests=False, safety_margin=0, robust=robust)
    if not any(mask):
        # A future opponent move may reopen a route; keep immediate survival options.
        mask[ACTION_STOP] = bool(not burns[0, r, c])
        for action, (dr, dc) in _MOVE_DELTA.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.board_size and 0 <= nc < env.board_size:
                mask[action] = bool(not blocked[0, nr, nc] and not burns[0, nr, nc])
        if not any(mask):
            mask[ACTION_STOP] = True
    return mask


# ─────────────────────── helper: BFS for safe cell ────────────────────────

def _escape_first_actions(env, start_r, start_c, safe, blocked, burns, robust=True):
    """Prefer a refuge outside nearby counter-bomb rays, then the quickest route."""
    threatened = np.zeros_like(burns[0])
    for other in env.agents if robust else ():
        if other.alive and (other.row, other.col) != (start_r, start_c):
            positions = [(other.row, other.col)]
            positions.extend((other.row + dr, other.col + dc) for dr, dc in _MOVE_DELTA.values()
                             if env._is_walkable(other.row + dr, other.col + dc))
            for r, c in positions:
                for pos in env._blast_cells(r, c, other.blast_range):
                    threatened[pos] = True
    queue = deque([(start_r, start_c, 0, None)])
    visited = {(start_r, start_c, 0, None)}
    moves = {**_MOVE_DELTA, ACTION_STOP: (0, 0)}
    exits = []
    best_score = (2, len(burns) + 1)
    while queue:
        r, c, tick, first = queue.popleft()
        if best_score[0] == 0 and tick > best_score[1]:
            break
        if first is not None and not burns[tick - 1:, r, c].any():
            score = (int(threatened[r, c]), tick)
            if score < best_score:
                best_score, exits = score, [first]
            elif score == best_score and first not in exits:
                exits.append(first)
            if not threatened[r, c]:
                continue
        if tick >= len(burns):
            continue
        for action, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            first_action = action if first is None else first
            state = (nr, nc, tick + 1, first_action)
            if not (0 <= nr < env.board_size and 0 <= nc < env.board_size):
                continue
            if state in visited or not safe[tick, nr, nc]:
                continue
            if action != ACTION_STOP and blocked[tick, nr, nc]:
                continue
            visited.add(state)
            queue.append((nr, nc, tick + 1, first_action))
    moves_out = [a for a in exits if a != ACTION_STOP]
    return (moves_out or exits) if robust else exits


def _bfs_safe_cell(env: BombermanFFAEnv, start_r: int, start_c: int,
                   danger: np.ndarray, robust=True) -> Optional[int]:
    """First move to the nearest timed refuge, including a necessary wait."""
    attempts = ((1, True), (0, True), (0, False)) if robust else ((0, True), (0, False))
    for margin, avoid in attempts:
        safe, blocked, burns = _survival_map(env, start_r, start_c, avoid_contests=avoid,
                                            safety_margin=margin)
        exits = _escape_first_actions(env, start_r, start_c, safe, blocked, burns, robust=robust)
        if exits:
            return exits[0]
    return None


# ─────────────────── helper: BFS toward nearest wooden wall ───────────────

def _bfs_toward_wood(env: BombermanFFAEnv, start_r: int, start_c: int,
                     danger: np.ndarray) -> Optional[int]:
    """
    BFS from agent to find the nearest wooden wall.  Returns the first
    action to move *adjacent* to a wooden wall (not *onto* it, which is
    not walkable), or None.
    """
    s = env.board_size
    bomb_positions = {(b.row, b.col) for b in env.bombs}

    visited = set()
    visited.add((start_r, start_c))
    queue: deque[tuple[int, int, int]] = deque()
    for action, (dr, dc) in _MOVE_DELTA.items():
        nr, nc = start_r + dr, start_c + dc
        if not (0 <= nr < s and 0 <= nc < s):
            continue
        if env.grid[nr, nc] != EMPTY:
            continue
        if (nr, nc) in bomb_positions:
            continue
        # Avoid walking into danger zones that are about to blow
        if danger[nr, nc] > 0 and danger[nr, nc] <= 3:
            continue
        visited.add((nr, nc))
        queue.append((nr, nc, action))

    while queue:
        r, c, first_action = queue.popleft()
        # Check if adjacent to a wooden wall.
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < s and 0 <= cc < s and env.grid[rr, cc] == WOODEN_WALL:
                return first_action
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if not (0 <= nr < s and 0 <= nc < s):
                continue
            if env.grid[nr, nc] != EMPTY:
                continue
            if (nr, nc) in bomb_positions:
                continue
            if danger[nr, nc] > 0 and danger[nr, nc] <= 3:
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, first_action))
    return None


# ─────────────────────── helper: adjacent to wooden wall? ─────────────────

def _adjacent_to_wood(env: BombermanFFAEnv, r: int, c: int) -> bool:
    s = env.board_size
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < s and 0 <= cc < s and env.grid[rr, cc] == WOODEN_WALL:
            return True
    return False


# ─────────────────────── helper: adjacent to opponent? ────────────────────

def _adjacent_to_opponent(env: BombermanFFAEnv, agent_id: int, r: int, c: int) -> bool:
    blast = env._blast_cells(r, c, env.agents[agent_id].blast_range)
    return any(a.alive and a.agent_id != agent_id and (a.row, a.col) in blast
               for a in env.agents)


# ─────────────────────── helper: can safely place bomb? ───────────────────

def _can_bomb_and_escape(env: BombermanFFAEnv, agent_id: int,
                         r: int, c: int, danger: np.ndarray, robust=True) -> bool:
    """Spend this tick placing, then survive every visible blast and its flames."""
    agent = env.agents[agent_id]
    if agent.bombs_placed >= agent.ammo or any((b.row, b.col) == (r, c) for b in env.bombs):
        return False
    placements = [Bomb(r, c, agent_id, env.bomb_timer, agent.blast_range)]
    occupied = {(b.row, b.col) for b in env.bombs}
    # Check a simultaneous counter-bomb before committing to our placement.
    for other in env.agents if robust else ():
        if (other.alive and other.agent_id != agent_id
                and other.bombs_placed < other.ammo
                and (other.row, other.col) not in occupied):
            placements.append(Bomb(other.row, other.col, other.agent_id,
                                   env.bomb_timer, other.blast_range))
    placements.sort(key=lambda b: b.owner)  # Match simultaneous placement order in env.step.
    safe, _, _ = _survival_map(env, r, c, placements, safety_margin=2 if robust else 0,
                              reserve_exits=robust)
    return bool(safe[0, r, c])


def _bfs_toward_item(env, agent_id, danger, max_distance=3):
    """Take a short safe detour for an upgrade that is still useful."""
    agent = env.agents[agent_id]
    targets = {pos for pos, kind in env.powerups.items()
               if (kind == POWERUP_AMMO and agent.ammo < MAX_AMMO)
               or (kind == POWERUP_RANGE and agent.blast_range < MAX_BLAST_RANGE)}
    occupied = {(a.row, a.col) for a in env.agents if a.alive and a.agent_id != agent_id}
    start = (agent.row, agent.col)
    queue = deque([(start, ACTION_STOP, 0)])
    visited = {start}
    while queue:
        pos, first, distance = queue.popleft()
        if pos in targets and danger[pos] == 0:
            return first
        if distance == max_distance:
            continue
        for action, (dr, dc) in _MOVE_DELTA.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if nxt in visited or nxt in occupied or not env._is_walkable(*nxt):
                continue
            if danger[nxt] > 0:
                continue
            visited.add(nxt)
            queue.append((nxt, action if distance == 0 else first, distance + 1))
    return None


# ──────────────────────────── RuleBasedAgent ──────────────────────────────

class RuleBasedAgent:
    """
    A deterministic rule-based agent for BombermanFFAEnv.

    Decision priority:
      1. DODGE if in danger (bomb blast zone / flame).
      2. BOMB an opponent in the actual blast rays, if escape is possible.
      3. Collect a useful nearby item on a safe route.
      4. BOMB / MOVE toward nearest wooden wall.
      5. Move toward opponent.
      6. STOP.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def select_action(self, env: BombermanFFAEnv) -> int:
        agent = env.agents[self.agent_id]
        if not agent.alive:
            return ACTION_STOP

        r, c = agent.row, agent.col
        danger = _build_danger_map(env)
        # Keep this opponent baseline fixed while improving the learned policy's filter.
        mask = _safe_action_mask(env, self.agent_id, robust=False)

        # Priority 1: If in danger, dodge!
        if danger[r, c] > 0:
            escape_action = _bfs_safe_cell(env, r, c, danger, robust=False)
            if escape_action is not None and mask[escape_action]:
                return escape_action
        else:
            if mask[ACTION_BOMB] and _adjacent_to_opponent(env, self.agent_id, r, c):
                return ACTION_BOMB
            move = _bfs_toward_item(env, self.agent_id, danger)
            if move is not None and mask[move]:
                return move
            if mask[ACTION_BOMB] and _adjacent_to_wood(env, r, c):
                return ACTION_BOMB
            for move in (_bfs_toward_wood(env, r, c, danger),
                         _bfs_toward_opponent(env, self.agent_id, r, c, danger)):
                if move is not None and mask[move]:
                    return move
        choices = [a for a in range(ACTION_BOMB) if mask[a]]
        return int(np.random.choice(choices)) if choices else ACTION_STOP


def _bfs_toward_opponent(env: BombermanFFAEnv, agent_id: int,
                         start_r: int, start_c: int,
                         danger: np.ndarray) -> Optional[int]:
    """BFS toward the nearest alive opponent."""
    s = env.board_size
    bomb_positions = {(b.row, b.col) for b in env.bombs}
    opponent_positions = set()
    for a in env.agents:
        if a.agent_id != agent_id and a.alive:
            opponent_positions.add((a.row, a.col))
    if not opponent_positions:
        return None

    visited = set()
    visited.add((start_r, start_c))
    queue: deque[tuple[int, int, int]] = deque()
    for action, (dr, dc) in _MOVE_DELTA.items():
        nr, nc = start_r + dr, start_c + dc
        if not (0 <= nr < s and 0 <= nc < s):
            continue
        if env.grid[nr, nc] != EMPTY:
            continue
        if (nr, nc) in bomb_positions:
            continue
        if danger[nr, nc] > 0 and danger[nr, nc] <= 3:
            continue
        visited.add((nr, nc))
        queue.append((nr, nc, action))

    while queue:
        r, c, first_action = queue.popleft()
        if (r, c) in opponent_positions:
            return first_action
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited:
                continue
            if not (0 <= nr < s and 0 <= nc < s):
                continue
            if env.grid[nr, nc] != EMPTY:
                continue
            if (nr, nc) in bomb_positions:
                continue
            if danger[nr, nc] > 0 and danger[nr, nc] <= 3:
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, first_action))
    return None


# ──────────────────────── evaluation script ───────────────────────────────

def evaluate_rule_based(num_episodes: int = 100, board_size: int = 11,
                        seed: int = 42, verbose: bool = True):
    """
    Pit two RuleBasedAgents against each other for `num_episodes` episodes.
    Print statistics: win rate, avg episode length, avg bombs per episode.
    """
    env = BombermanFFAEnv(board_size=board_size, num_agents=2, seed=seed)
    bots = [RuleBasedAgent(0), RuleBasedAgent(1)]

    wins = [0, 0]
    draws = 0
    total_steps = 0
    total_bombs = [0, 0]

    for ep in range(num_episodes):
        env.reset()
        bombs_this_ep = [0, 0]

        for t in range(env.max_steps):
            actions = []
            for bot in bots:
                action = bot.select_action(env)
                if action == ACTION_BOMB:
                    bombs_this_ep[bot.agent_id] += 1
                actions.append(action)

            _, _, _, info = env.step(actions)
            if info["episode_done"]:
                break

        total_steps += info["step"]
        for i in range(2):
            total_bombs[i] += bombs_this_ep[i]

        alive = info["alive"]
        if alive[0] and not alive[1]:
            wins[0] += 1
        elif alive[1] and not alive[0]:
            wins[1] += 1
        else:
            draws += 1

    avg_steps = total_steps / num_episodes
    avg_bombs = [total_bombs[i] / num_episodes for i in range(2)]

    if verbose:
        print(f"{'=' * 50}")
        print(f"Rule-based bot evaluation: {num_episodes} episodes")
        print(f"  Board size: {board_size}x{board_size}, 2 agents")
        print(f"{'=' * 50}")
        print(f"  Agent 0 wins: {wins[0]:4d} ({100 * wins[0] / num_episodes:.1f}%)")
        print(f"  Agent 1 wins: {wins[1]:4d} ({100 * wins[1] / num_episodes:.1f}%)")
        print(f"  Draws:        {draws:4d} ({100 * draws / num_episodes:.1f}%)")
        print(f"  Avg episode length: {avg_steps:.1f} steps")
        print(f"  Avg bombs/episode:  Agent 0 = {avg_bombs[0]:.1f}, Agent 1 = {avg_bombs[1]:.1f}")
        print(f"{'=' * 50}")

        # Gate checks
        win_diff = abs(wins[0] - wins[1]) / num_episodes * 100
        print(f"\n--- Gate Checks ---")
        print(f"  Win rate difference: {win_diff:.1f}% ", end="")
        print("[PASS]" if win_diff < 15 else "[FAIL] (need < 15%)")
        print(f"  Avg bombs > 0: Agent 0 = {avg_bombs[0]:.1f}, Agent 1 = {avg_bombs[1]:.1f} ", end="")
        print("[PASS]" if avg_bombs[0] > 0 and avg_bombs[1] > 0 else "[FAIL]")
        print(f"  No crashes: [PASS] (completed {num_episodes} episodes)")

    return {
        "wins": wins,
        "draws": draws,
        "avg_steps": avg_steps,
        "avg_bombs": avg_bombs,
    }


if __name__ == "__main__":
    evaluate_rule_based(num_episodes=100, board_size=11, seed=42)
