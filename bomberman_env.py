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
    chain_triggered: bool = False


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
        sudden_death: bool = False,
        sudden_death_start: int = 200,
        sudden_death_interval: int = 20,
        sudden_death_warning_steps: int = 40,
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
        self.sudden_death = sudden_death
        self.sudden_death_start = sudden_death_start
        self.sudden_death_interval = sudden_death_interval
        self.sudden_death_warning_steps = sudden_death_warning_steps
        if sudden_death_start < 1 or sudden_death_interval < 1 or sudden_death_warning_steps < 0:
            raise ValueError("sudden-death timings must be non-negative, with start and interval >= 1")
        self.rng = np.random.default_rng(seed)

        # Corner spawn positions (row, col) for up to 4 agents.
        s = self.board_size
        self._spawn_corners = [(1, 1), (1, s - 2), (s - 2, 1), (s - 2, s - 2)]

        self.grid: np.ndarray = None
        self.bombs: list[Bomb] = []
        self.flames: dict[tuple[int, int], int] = {}   # (r,c) -> remaining duration
        self.powerups: dict[tuple[int, int], str] = {}  # (r,c) -> powerup type
        self.agents: list[Agent] = []
        self.step_count = 0
        self._episode_done = False
        self._sudden_death_ring = 0

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
        self.powerups = {}
        self.step_count = 0
        self._episode_done = False
        self._sudden_death_ring = 0

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
        sudden_death_kills = self._apply_sudden_death()
        self._last_bomb_placements = []

        # 1) Resolve movement / bomb placement intents.
        intended_positions = {}
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
                self._try_place_bomb(agent)
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
        died_this_step = list(sudden_death_kills)
        for agent in self.agents:
            if agent.alive and (agent.row, agent.col) in self.flames:
                agent.alive = False
                died_this_step.append(agent.agent_id)

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

        # 6) Reward shaping.
        alive_ids = [a.agent_id for a in self.agents if a.alive]
        for agent in self.agents:
            if agent.agent_id in died_this_step:
                rewards[agent.agent_id] += -1.0
            elif agent.alive:
                rewards[agent.agent_id] += -0.001  # small time penalty -> discourages passivity

        # Credit eliminations to bomb owners (approximate: whoever owned any
        # bomb whose flame is at a dying agent's cell this step). We
        # recompute this inside _tick_bombs_and_explode via self._last_kills.
        for killer_id, victim_id in self._last_kills:
            if killer_id != victim_id and killer_id in alive_ids:
                rewards[killer_id] += 1.0
                for a in self.agents:
                    if a.agent_id == killer_id:
                        a.kills += 1

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
            "bomb_outcomes": getattr(self, '_last_bomb_outcomes', []),
            "tactical_bomb_owners": [
                owner for owner, tactical, _, _ in self._last_bomb_placements if tactical
            ],
            "offensive_bomb_owners": [
                owner for owner, _, offensive, _ in self._last_bomb_placements if offensive
            ],
            "bomb_pressure_scores": [
                (owner, pressure_score)
                for owner, _, _, pressure_score in self._last_bomb_placements
            ],
            "sudden_death_ring": self._sudden_death_ring,
            "sudden_death_kills": sudden_death_kills,
        }
        return obs_list, rewards, dones, info

    # ------------------------------ bomb / blast logic ------------------------------

    def _target_sudden_death_ring(self) -> int:
        if not self.sudden_death or self.step_count < self.sudden_death_start:
            return 0
        return min(
            1 + (self.step_count - self.sudden_death_start) // self.sudden_death_interval,
            (self.board_size // 2) - 1,
        )

    def _ring_cells(self, ring: int) -> set[tuple[int, int]]:
        s = self.board_size
        cells = set()
        for index in range(ring, s - ring):
            cells.add((ring, index))
            cells.add((s - 1 - ring, index))
            cells.add((index, ring))
            cells.add((index, s - 1 - ring))
        return cells

    def _apply_sudden_death(self) -> list[int]:
        """Close newly due rings and eliminate agents caught by the closing wall."""
        target_ring = self._target_sudden_death_ring()
        if target_ring <= self._sudden_death_ring:
            return []

        closing_cells = set()
        for ring in range(self._sudden_death_ring + 1, target_ring + 1):
            closing_cells.update(self._ring_cells(ring))

        killed = []
        for agent in self.agents:
            if agent.alive and (agent.row, agent.col) in closing_cells:
                agent.alive = False
                killed.append(agent.agent_id)

        for bomb in list(self.bombs):
            if (bomb.row, bomb.col) in closing_cells:
                owner = self.agents[bomb.owner]
                owner.bombs_placed = max(0, owner.bombs_placed - 1)
                self.bombs.remove(bomb)

        for row, col in closing_cells:
            self.grid[row, col] = RIGID_WALL
            self.flames.pop((row, col), None)
            self.powerups.pop((row, col), None)

        self._sudden_death_ring = target_ring
        return killed

    def _sudden_death_warning_map(self) -> np.ndarray:
        """Return cells scheduled to close soon, giving the policy time to react."""
        warning = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        if not self.sudden_death:
            return warning

        if self.step_count < self.sudden_death_start:
            if self.sudden_death_start - self.step_count > self.sudden_death_warning_steps:
                return warning
            next_ring = 1
        else:
            next_ring = self._sudden_death_ring + 1

        if next_ring >= self.board_size // 2:
            return warning
        for row, col in self._ring_cells(next_ring):
            warning[row, col] = 1.0
        return warning

    def _bomb_blast_cells(self, row: int, col: int, blast_range: int) -> set[tuple[int, int]]:
        """Return cells reached by a hypothetical bomb without mutating the board."""
        cells = {(row, col)}
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for step in range(1, blast_range + 1):
                target_row, target_col = row + dr * step, col + dc * step
                if not (0 <= target_row < self.board_size and 0 <= target_col < self.board_size):
                    break
                if self.grid[target_row, target_col] == RIGID_WALL:
                    break
                cells.add((target_row, target_col))
                if self.grid[target_row, target_col] == WOODEN_WALL:
                    break
        return cells

    def _bomb_pressure_score(self, agent: Agent) -> int:
        """Count immediate opponent escape options covered by a hypothetical bomb."""
        blast_cells = self._bomb_blast_cells(agent.row, agent.col, agent.blast_range)
        bomb_position = (agent.row, agent.col)
        pressure_score = 0
        for opponent in self.agents:
            if not opponent.alive or opponent.agent_id == agent.agent_id:
                continue
            candidates = {(opponent.row, opponent.col)}
            for dr, dc in _MOVE_DELTA.values():
                row, col = opponent.row + dr, opponent.col + dc
                if (row, col) != bomb_position and self._is_walkable(row, col):
                    candidates.add((row, col))
            pressure_score += len(candidates & blast_cells)
        return pressure_score

    def _bomb_target_flags(self, agent: Agent) -> tuple[bool, bool, int]:
        """Return tactical, offensive, and opponent-pressure placement signals."""
        tactical = False
        pressure_score = self._bomb_pressure_score(agent)
        offensive = pressure_score > 0
        opponent_positions = {
            (other.row, other.col)
            for other in self.agents
            if other.alive and other.agent_id != agent.agent_id
        }
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for step in range(1, agent.blast_range + 1):
                row, col = agent.row + dr * step, agent.col + dc * step
                if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                    break
                if self.grid[row, col] == RIGID_WALL:
                    break
                if (row, col) in opponent_positions:
                    tactical = True
                    offensive = True
                if self.grid[row, col] == WOODEN_WALL:
                    tactical = True
                    break
        return tactical or offensive, offensive, pressure_score

    def _try_place_bomb(self, agent: Agent):
        if not agent.alive:
            return
        if agent.bombs_placed >= agent.ammo:
            return
        pos = (agent.row, agent.col)
        if any(b.row == pos[0] and b.col == pos[1] for b in self.bombs):
            return  # already a bomb here
        tactical, offensive, pressure_score = self._bomb_target_flags(agent)
        self.bombs.append(Bomb(row=pos[0], col=pos[1], owner=agent.agent_id,
                                timer=self.bomb_timer, blast_range=agent.blast_range))
        agent.bombs_placed += 1
        self._last_bomb_placements.append(
            (agent.agent_id, tactical, offensive, pressure_score)
        )

    def _tick_bombs_and_explode(self):
        self._last_kills = []  # list of (killer_id, victim_id) approximations
        self._last_walls_destroyed = {}  # {owner_id: count} walls destroyed this tick
        self._last_bomb_outcomes = []
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
            ready_ids = {id(bomb) for bomb in ready}
            for b in ready:
                self._explode_bomb(b)
                exploded_this_pass = True
            self.bombs = [b for b in self.bombs if id(b) not in ready_ids]

    def _explode_bomb(self, bomb: Bomb):
        owner_agent = next((a for a in self.agents if a.agent_id == bomb.owner), None)
        if owner_agent is not None:
            owner_agent.bombs_placed = max(0, owner_agent.bombs_placed - 1)

        cells = [(bomb.row, bomb.col)]
        walls_destroyed_count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for step in range(1, bomb.blast_range + 1):
                r, c = bomb.row + dr * step, bomb.col + dc * step
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    break
                if self.grid[r, c] == RIGID_WALL:
                    break
                cells.append((r, c))
                if self.grid[r, c] == WOODEN_WALL:
                    self.grid[r, c] = EMPTY
                    walls_destroyed_count += 1
                    if self.rng.random() < self.powerup_prob:
                        kind = self.rng.choice([POWERUP_AMMO, POWERUP_RANGE])
                        self.powerups[(r, c)] = kind
                    break  # blast stops at the first wooden wall it destroys
        # Track wall destructions per bomb owner (for reward shaping).
        if walls_destroyed_count > 0:
            self._last_walls_destroyed[bomb.owner] = (
                self._last_walls_destroyed.get(bomb.owner, 0) + walls_destroyed_count
            )

        for pos in cells:
            self.flames[pos] = max(self.flames.get(pos, 0), self.flame_duration)
            # Chain-detonate any bomb caught in the blast.
            for other in self.bombs:
                if (other.row, other.col) == pos and other.timer > 0:
                    other.chain_triggered = True
                    other.timer = 0

        # Approximate kill attribution: any agent standing in `cells` right
        # now is credited to this bomb's owner (used by step() after this
        # call to decide rewards).
        victims = []
        for agent in self.agents:
            if agent.alive and (agent.row, agent.col) in cells:
                self._last_kills.append((bomb.owner, agent.agent_id))
                victims.append(agent.agent_id)
        self._last_bomb_outcomes.append({
            "owner": bomb.owner,
            "position": (bomb.row, bomb.col),
            "walls_destroyed": walls_destroyed_count,
            "victims": victims,
            "chain_reaction": bomb.chain_triggered,
        })

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
        Shape: (9, H, W), dtype float32 -- ready to feed a CNN.

        Channels:
          0: rigid walls
          1: wooden walls
          2: bombs, value = normalized remaining timer (0-1)
          3: flame
          4: ammo power-ups
          5: range power-ups
          6: self position
          7: opponents' positions (combined)
          8: cells that sudden death will close next (when enabled)
        """
        s = self.board_size
        obs = np.zeros((9, s, s), dtype=np.float32)
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
        obs[8] = self._sudden_death_warning_map()
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
