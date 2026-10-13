"""
bomberman_env.py

A lightweight, Pommerman-inspired free-for-all Bomberman environment,
built from scratch with numpy only (no external RL framework dependency).

Design (mirrors the classic Pommerman FFA board):
- Grid with a checkerboard pattern of indestructible (rigid) walls, plus a
  border of rigid walls.
- Remaining open cells are randomly filled with destructible (wooden)
  walls, except for 3x3 safe zones around each agent's spawn corner.
- Up to `num_agents` agents (default 4) spawn in the four corners.
- Agents can move (up/down/left/right), stay, or place a bomb.
- Bombs explode after a fixed timer, producing a "+"-shaped blast that
  destroys wooden walls (revealing power-ups with some probability) and
  kills any agent caught in the flame. Blasts can chain-detonate other
  bombs.
- Power-ups: extra bomb capacity ("ammo") and blast range ("power").
- Episode ends when <= 1 agent remains alive, or a max step count is hit.

This is intentionally dependency-light so it can run on Kaggle/Colab
without installing gym/pommerman. The observation is a multi-channel
numpy array (agent-centric channel ordering) suitable as input to a CNN.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------- Cell / action constants -----------------------------

EMPTY = 0
RIGID_WALL = 1     # indestructible
WOODEN_WALL = 2     # destructible, may drop a power-up when destroyed

ACTION_STOP = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_BOMB = 5
NUM_ACTIONS = 6

_MOVE_DELTA = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

POWERUP_AMMO = "ammo"
POWERUP_RANGE = "range"


@dataclass
class Bomb:
    row: int
    col: int
    owner: int
    timer: int          # steps remaining before explosion
    blast_range: int


@dataclass
class Agent:
    agent_id: int
    row: int
    col: int
    alive: bool = True
    ammo: int = 1            # max bombs that can be placed simultaneously
    bombs_placed: int = 0    # currently active bombs owned by this agent
    blast_range: int = 2
    kills: int = 0


class BombermanFFAEnv:
    """
    Multi-agent free-for-all Bomberman environment.

    API (intentionally close to a simplified PettingZoo/Pommerman style):

        env = BombermanFFAEnv(num_agents=4)
        obs_list = env.reset()                      # list[np.ndarray], one per agent
        obs_list, rewards, dones, info = env.step(actions)  # actions: list[int], one per agent

    `dones` is a list of per-agent booleans (True once that agent is out or
    the episode has ended). `info["episode_done"]` is True once the whole
    episode is over (<=1 agent left or max steps reached).
    """

    def __init__(
        self,
        board_size: int = 11,
        num_agents: int = 4,
        max_steps: int = 500,
        wooden_wall_density: float = 0.7,
        bomb_timer: int = 8,
        flame_duration: int = 2,
        powerup_prob: float = 0.3,
        seed: Optional[int] = None,
    ):
        assert board_size % 2 == 1, "board_size should be odd, like classic Pommerman (e.g. 11)"
        assert 2 <= num_agents <= 4, "supports 2-4 agents in FFA"

        self.board_size = board_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.wooden_wall_density = wooden_wall_density
        self.bomb_timer = bomb_timer
        self.flame_duration = flame_duration
        self.powerup_prob = powerup_prob
        self.rng = np.random.default_rng(seed)

        # Corner spawn positions (row, col) for up to 4 agents.
        s = self.board_size
        self._spawn_corners = [(1, 1), (1, s - 2), (s - 2, 1), (s - 2, s - 2)]

        self.grid: np.ndarray = None
        self.bombs: list[Bomb] = []
        self.flames: dict[tuple[int, int], int] = {}   # (r,c) -> remaining duration
        self._flame_owners: dict[tuple[int, int], int] = {}
        self.powerups: dict[tuple[int, int], str] = {}  # (r,c) -> powerup type
        self.agents: list[Agent] = []
        self.step_count = 0
        self._episode_done = False

    # --------------------------------- setup ---------------------------------

    def reset(self) -> list[np.ndarray]:
        s = self.board_size
        self.grid = np.zeros((s, s), dtype=np.int8)

        # Border + checkerboard rigid walls (classic Pommerman pattern).
        self.grid[0, :] = RIGID_WALL
        self.grid[-1, :] = RIGID_WALL
        self.grid[:, 0] = RIGID_WALL
        self.grid[:, -1] = RIGID_WALL
        for r in range(2, s - 2, 2):
            for c in range(2, s - 2, 2):
                self.grid[r, c] = RIGID_WALL

        # Spawn agents.
        self.agents = []
        safe_cells = set()
        for i in range(self.num_agents):
            r, c = self._spawn_corners[i]
            self.agents.append(Agent(agent_id=i, row=r, col=c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < s and 0 <= cc < s:
                        safe_cells.add((rr, cc))

        # Scatter wooden walls on remaining open cells.
        for r in range(1, s - 1):
            for c in range(1, s - 1):
                if self.grid[r, c] == EMPTY and (r, c) not in safe_cells:
                    if self.rng.random() < self.wooden_wall_density:
                        self.grid[r, c] = WOODEN_WALL

        self.bombs = []
        self.flames = {}
        self._flame_owners = {}
        self.powerups = {}
        self.step_count = 0
        self._episode_done = False

        return [self._get_obs(a.agent_id) for a in self.agents]

    # --------------------------------- step ---------------------------------

    def step(self, actions: list[int]):
        """
        actions: list of ints (one per agent, ACTION_STOP..ACTION_BOMB).
        An action for a dead agent is ignored.
        """
        assert len(actions) == self.num_agents
        self.step_count += 1

        rewards = [0.0 for _ in range(self.num_agents)]

        # 1) Resolve movement / bomb placement intents.
        intended_positions = {}
        bombs_placed = []
        for agent in self.agents:
            if not agent.alive:
                continue
            a = actions[agent.agent_id]
            if a in _MOVE_DELTA:
                dr, dc = _MOVE_DELTA[a]
                nr, nc = agent.row + dr, agent.col + dc
                if self._is_walkable(nr, nc):
                    intended_positions[agent.agent_id] = (nr, nc)
                else:
                    intended_positions[agent.agent_id] = (agent.row, agent.col)
            elif a == ACTION_BOMB:
                if self._try_place_bomb(agent):
                    bombs_placed.append(agent.agent_id)
                intended_positions[agent.agent_id] = (agent.row, agent.col)
            else:  # ACTION_STOP
                intended_positions[agent.agent_id] = (agent.row, agent.col)

        # Prevent two agents from swapping into the same cell in a way that
        # lets them pass through each other; simple rule: if two agents
        # target the same cell, both just stay put this step.
        target_counts = {}
        for pos in intended_positions.values():
            target_counts[pos] = target_counts.get(pos, 0) + 1
        for agent in self.agents:
            if not agent.alive:
                continue
            target = intended_positions[agent.agent_id]
            if target_counts[target] > 1:
                continue  # collision -> stay in place
            agent.row, agent.col = target

        # 2) Tick bombs, explode any whose timer hits zero (with chain reactions).
        self._tick_bombs_and_explode()

        # 3) Resolve flame damage.
        died_this_step = []
        for agent in self.agents:
            if agent.alive and (agent.row, agent.col) in self.flames:
                agent.alive = False
                died_this_step.append(agent.agent_id)
                owner = self._flame_owners.get((agent.row, agent.col))
                if owner is not None:
                    self._last_kills.append((owner, agent.agent_id))

        # 4) Pick up power-ups.
        for agent in self.agents:
            if not agent.alive:
                continue
            pos = (agent.row, agent.col)
            if pos in self.powerups:
                kind = self.powerups.pop(pos)
                if kind == POWERUP_AMMO:
                    agent.ammo += 1
                elif kind == POWERUP_RANGE:
                    agent.blast_range += 1

        # 5) Decay flames.
        for pos in list(self.flames.keys()):
            self.flames[pos] -= 1
            if self.flames[pos] <= 0:
                del self.flames[pos]
                self._flame_owners.pop(pos, None)

        # 6) Reward shaping.
        alive_ids = [a.agent_id for a in self.agents if a.alive]
        for agent in self.agents:
            if agent.agent_id in died_this_step:
                rewards[agent.agent_id] += -1.0
            elif agent.alive:
                rewards[agent.agent_id] += -0.001  # small time penalty -> discourages passivity

        # Each death has one owner, including deaths in lingering flames.
        for killer_id, victim_id in self._last_kills:
            if killer_id != victim_id:
                self.agents[killer_id].kills += 1
                if killer_id in alive_ids:
                    rewards[killer_id] += 1.0

        # 7) Episode / done bookkeeping.
        num_alive = len(alive_ids)
        self._episode_done = (num_alive <= 1) or (self.step_count >= self.max_steps)
        if self._episode_done:
            for agent in self.agents:
                if agent.alive and num_alive == 1:
                    rewards[agent.agent_id] += 10.0   # sole survivor bonus
            for aid in died_this_step:
                pass  # death penalty already applied above

        dones = [(not a.alive) or self._episode_done for a in self.agents]
        obs_list = [self._get_obs(a.agent_id) for a in self.agents]
        info = {
            "episode_done": self._episode_done,
            "alive": [a.alive for a in self.agents],
            "step": self.step_count,
            "walls_destroyed": getattr(self, '_last_walls_destroyed', {}),
            "kills": self._last_kills,
            "bombs_placed": bombs_placed,
        }
        return obs_list, rewards, dones, info

    # ------------------------------ bomb / blast logic ------------------------------

    def _try_place_bomb(self, agent: Agent):
        if not agent.alive:
            return False
        if agent.bombs_placed >= agent.ammo:
            return False
        pos = (agent.row, agent.col)
        if any(b.row == pos[0] and b.col == pos[1] for b in self.bombs):
            return False  # already a bomb here
        self.bombs.append(Bomb(row=pos[0], col=pos[1], owner=agent.agent_id,
                                timer=self.bomb_timer, blast_range=agent.blast_range))
        agent.bombs_placed += 1
        return True

    def _tick_bombs_and_explode(self):
        self._last_kills = []  # list of (killer_id, victim_id), one per death
        self._last_walls_destroyed = {}  # {owner_id: count} walls destroyed this tick
        exploded_this_pass = True
        # Decrement timers once, then keep resolving chain reactions until
        # no more bombs are ready to go off.
        for b in self.bombs:
            b.timer -= 1

        while exploded_this_pass:
            exploded_this_pass = False
            ready = [b for b in self.bombs if b.timer <= 0]
            if not ready:
                break
            for b in ready:
                self._explode_bomb(b)
                self.bombs.remove(b)
                exploded_this_pass = True

    def _blast_cells(self, row: int, col: int, blast_range: int):
        """Project a blast against the current walls without changing the board."""
        cells = [(row, col)]
        for dr, dc in _MOVE_DELTA.values():
            for step in range(1, blast_range + 1):
                r, c = row + dr * step, col + dc * step
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    break
                if self.grid[r, c] == RIGID_WALL:
                    break
                cells.append((r, c))
                if self.grid[r, c] == WOODEN_WALL:
                    break  # blast stops at the first wooden wall it destroys
        return cells

    def _explode_bomb(self, bomb: Bomb):
        owner_agent = self.agents[bomb.owner]
        owner_agent.bombs_placed = max(0, owner_agent.bombs_placed - 1)
        cells = self._blast_cells(bomb.row, bomb.col, bomb.blast_range)
        walls_destroyed_count = 0
        for r, c in cells:
            if self.grid[r, c] == WOODEN_WALL:
                self.grid[r, c] = EMPTY
                walls_destroyed_count += 1
                if self.rng.random() < self.powerup_prob:
                    kind = self.rng.choice([POWERUP_AMMO, POWERUP_RANGE])
                    self.powerups[(r, c)] = kind
        # Track wall destructions per bomb owner (for reward shaping).
        if walls_destroyed_count > 0:
            self._last_walls_destroyed[bomb.owner] = (
                self._last_walls_destroyed.get(bomb.owner, 0) + walls_destroyed_count
            )

        for pos in cells:
            self.flames[pos] = max(self.flames.get(pos, 0), self.flame_duration)
            # The first active flame owns an overlap; one victim earns one kill.
            self._flame_owners.setdefault(pos, bomb.owner)
            # Chain-detonate any bomb caught in the blast.
            for other in self.bombs:
                if (other.row, other.col) == pos and other.timer > 0:
                    other.timer = 0

    def _is_walkable(self, r: int, c: int) -> bool:
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return False
        if self.grid[r, c] != EMPTY:
            return False
        if any(b.row == r and b.col == c for b in self.bombs):
            return False
        return True

    # --------------------------------- observation ---------------------------------

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """
        Multi-channel observation, agent-centric (the requesting agent's
        position is always in the "self" channel, others in "opponents").
        Shape: (C, H, W), dtype float32 -- ready to feed a CNN.

        Channels:
          0: rigid walls
          1: wooden walls
          2: bombs, value = normalized remaining timer (0-1)
          3: flame
          4: ammo power-ups
          5: range power-ups
          6: self position
          7: opponents' positions (combined)
        """
        s = self.board_size
        obs = np.zeros((8, s, s), dtype=np.float32)
        obs[0] = (self.grid == RIGID_WALL).astype(np.float32)
        obs[1] = (self.grid == WOODEN_WALL).astype(np.float32)
        for b in self.bombs:
            obs[2, b.row, b.col] = max(b.timer, 0) / self.bomb_timer
        for (r, c), dur in self.flames.items():
            obs[3, r, c] = dur / self.flame_duration
        for (r, c), kind in self.powerups.items():
            if kind == POWERUP_AMMO:
                obs[4, r, c] = 1.0
            else:
                obs[5, r, c] = 1.0
        for agent in self.agents:
            if not agent.alive:
                continue
            if agent.agent_id == agent_id:
                obs[6, agent.row, agent.col] = 1.0
            else:
                obs[7, agent.row, agent.col] = 1.0
        return obs

    def render(self) -> str:
        """Return a simple ASCII rendering, useful for debugging in a notebook."""
        symbols = {EMPTY: ".", RIGID_WALL: "#", WOODEN_WALL: "?"}
        s = self.board_size
        grid_chars = [[symbols[self.grid[r, c]] for c in range(s)] for r in range(s)]
        for (r, c) in self.flames:
            grid_chars[r][c] = "*"
        for b in self.bombs:
            grid_chars[b.row][b.col] = "o"
        for agent in self.agents:
            if agent.alive:
                grid_chars[agent.row][agent.col] = str(agent.agent_id)
        return "\n".join("".join(row) for row in grid_chars)


# --------------------------------- quick smoke test ---------------------------------

if __name__ == "__main__":
    env = BombermanFFAEnv(board_size=11, num_agents=4, seed=0)
    obs = env.reset()
    print("Initial board:")
    print(env.render())
    print("Obs shape per agent:", obs[0].shape)

    rng = np.random.default_rng(1)
    for t in range(60):
        actions = [int(rng.integers(0, NUM_ACTIONS)) for _ in range(env.num_agents)]
        obs, rewards, dones, info = env.step(actions)
        if info["episode_done"]:
            print(f"Episode ended at step {info['step']}, alive={info['alive']}")
            break
    print("\nFinal board:")
    print(env.render())
    print("Rewards last step:", rewards)
