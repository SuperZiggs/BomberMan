"""
rule_based_bot.py

Giai đoạn A — Rule-based agent dùng heuristic đơn giản:
  1. Nếu đang đứng trong vùng nguy hiểm (lửa, bom sắp nổ) → né (BFS tìm ô an toàn gần nhất).
  2. Nếu đứng cạnh tường gỗ và còn ammo → đặt bom rồi né.
  3. Nếu không → di chuyển về phía tường gỗ gần nhất.
  4. Không làm gì → STOP.

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
    EMPTY,
    RIGID_WALL,
    WOODEN_WALL,
    NUM_ACTIONS,
    ACTION_STOP,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_BOMB,
    _MOVE_DELTA,
)


# ─────────────────────────── helper: danger map ───────────────────────────

def _build_danger_map(env: BombermanFFAEnv) -> np.ndarray:
    """
    Return an (H, W) int array.  danger[r][c] > 0 means the cell is in
    the blast zone of a bomb that is about to explode in `danger[r][c]`
    ticks (lower = more urgent).  0 = safe (or flame, handled separately).
    For flames that are *currently* burning we set danger = 1 (immediate).
    """
    s = env.board_size
    danger = np.zeros((s, s), dtype=np.int32)

    # Mark existing flames as immediate danger.
    for (r, c) in env.flames:
        danger[r, c] = 1

    # Project bomb blast zones.
    for bomb in env.bombs:
        timer = max(bomb.timer, 1)  # timer <= 0 means exploding now
        # Center cell.
        if danger[bomb.row, bomb.col] == 0 or timer < danger[bomb.row, bomb.col]:
            danger[bomb.row, bomb.col] = timer
        # Four directions.
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for step in range(1, bomb.blast_range + 1):
                rr, cc = bomb.row + dr * step, bomb.col + dc * step
                if not (0 <= rr < s and 0 <= cc < s):
                    break
                if env.grid[rr, cc] == RIGID_WALL:
                    break
                if danger[rr, cc] == 0 or timer < danger[rr, cc]:
                    danger[rr, cc] = timer
                if env.grid[rr, cc] == WOODEN_WALL:
                    break  # blast stops here
    return danger


# ─────────────────────── helper: BFS for safe cell ────────────────────────

def _bfs_safe_cell(env: BombermanFFAEnv, start_r: int, start_c: int,
                   danger: np.ndarray, max_depth: int = 12) -> Optional[int]:
    """
    BFS from (start_r, start_c) to find the nearest *safe* walkable cell
    (danger == 0, no bomb on it, not a wall).  Returns the first action
    (UP/DOWN/LEFT/RIGHT) to take on the shortest path, or None if no safe
    cell is reachable.
    """
    s = env.board_size
    bomb_positions = {(b.row, b.col) for b in env.bombs}
    agent_positions = {(a.row, a.col) for a in env.agents if a.alive}

    visited = set()
    visited.add((start_r, start_c))
    # Queue items: (row, col, first_action)
    queue: deque[tuple[int, int, int]] = deque()
    for action, (dr, dc) in _MOVE_DELTA.items():
        nr, nc = start_r + dr, start_c + dc
        if not (0 <= nr < s and 0 <= nc < s):
            continue
        if env.grid[nr, nc] != EMPTY:
            continue
        if (nr, nc) in bomb_positions:
            continue
        # Don't walk into other agents
        if (nr, nc) in agent_positions and (nr, nc) != (start_r, start_c):
            continue
        visited.add((nr, nc))
        queue.append((nr, nc, action))

    while queue:
        r, c, first_action = queue.popleft()
        # Is this cell safe?
        if danger[r, c] == 0:
            return first_action
        if len(visited) > max_depth * 10:
            break  # cap search
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
            visited.add((nr, nc))
            queue.append((nr, nc, first_action))
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
    for a in env.agents:
        if a.agent_id != agent_id and a.alive:
            if abs(a.row - r) + abs(a.col - c) <= 2:  # within blast range
                return True
    return False


# ─────────────────────── helper: can safely place bomb? ───────────────────

def _can_bomb_and_escape(env: BombermanFFAEnv, agent_id: int,
                         r: int, c: int, danger: np.ndarray) -> bool:
    """
    Simulate: if we place a bomb here, is there at least one adjacent cell
    we can step into that is safe from this new bomb's blast?
    """
    agent = env.agents[agent_id]
    s = env.board_size
    bomb_positions = {(b.row, b.col) for b in env.bombs}

    # Simulate new bomb blast zone.
    new_blast = {(r, c)}
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for step in range(1, agent.blast_range + 1):
            rr, cc = r + dr * step, c + dc * step
            if not (0 <= rr < s and 0 <= cc < s):
                break
            if env.grid[rr, cc] == RIGID_WALL:
                break
            new_blast.add((rr, cc))
            if env.grid[rr, cc] == WOODEN_WALL:
                break

    # BFS to find a safe cell not in new_blast and not in existing danger.
    visited = {(r, c)}
    queue = deque()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < s and 0 <= nc < s):
            continue
        if env.grid[nr, nc] != EMPTY:
            continue
        if (nr, nc) in bomb_positions:
            continue
        visited.add((nr, nc))
        queue.append((nr, nc, 1))

    while queue:
        cr, cc, dist = queue.popleft()
        if dist > env.bomb_timer - 1:
            break  # too far, bomb would have exploded
        if (cr, cc) not in new_blast and danger[cr, cc] == 0:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if (nr, nc) in visited:
                continue
            if not (0 <= nr < s and 0 <= nc < s):
                continue
            if env.grid[nr, nc] != EMPTY:
                continue
            if (nr, nc) in bomb_positions:
                continue
            visited.add((nr, nc))
            queue.append((nr, nc, dist + 1))
    return False


# ──────────────────────────── RuleBasedAgent ──────────────────────────────

class RuleBasedAgent:
    """
    A deterministic rule-based agent for BombermanFFAEnv.

    Decision priority:
      1. DODGE if in danger (bomb blast zone / flame).
      2. BOMB if adjacent to wooden wall, has ammo, and can safely escape.
      3. BOMB if close to opponent, has ammo, and can safely escape.
      4. MOVE toward nearest wooden wall.
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

        # Priority 1: If in danger, dodge!
        if danger[r, c] > 0:
            escape_action = _bfs_safe_cell(env, r, c, danger)
            if escape_action is not None:
                return escape_action
            # Can't find safe cell — stay put (might survive if flame expires).
            return ACTION_STOP

        # Priority 2: Bomb if adjacent to wooden wall and can escape.
        if _adjacent_to_wood(env, r, c):
            if agent.bombs_placed < agent.ammo:
                if _can_bomb_and_escape(env, self.agent_id, r, c, danger):
                    return ACTION_BOMB

        # Priority 3: Bomb if near opponent and can escape.
        if _adjacent_to_opponent(env, self.agent_id, r, c):
            if agent.bombs_placed < agent.ammo:
                if _can_bomb_and_escape(env, self.agent_id, r, c, danger):
                    return ACTION_BOMB

        # Priority 4: Move toward nearest wooden wall.
        move = _bfs_toward_wood(env, r, c, danger)
        if move is not None:
            return move

        # Priority 5: Move toward nearest opponent.
        move = _bfs_toward_opponent(env, self.agent_id, r, c, danger)
        if move is not None:
            return move

        # Priority 6: Random safe move or stop.
        return _random_safe_move(env, r, c, danger)


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


def _random_safe_move(env: BombermanFFAEnv, r: int, c: int,
                      danger: np.ndarray) -> int:
    """Pick a random walkable, safe neighbor; or STOP."""
    s = env.board_size
    bomb_positions = {(b.row, b.col) for b in env.bombs}
    safe_actions = []
    for action, (dr, dc) in _MOVE_DELTA.items():
        nr, nc = r + dr, c + dc
        if not (0 <= nr < s and 0 <= nc < s):
            continue
        if env.grid[nr, nc] != EMPTY:
            continue
        if (nr, nc) in bomb_positions:
            continue
        if danger[nr, nc] == 0:
            safe_actions.append(action)
    if safe_actions:
        return int(np.random.choice(safe_actions))
    return ACTION_STOP


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
