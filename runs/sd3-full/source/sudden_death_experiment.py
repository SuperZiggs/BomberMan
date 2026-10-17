"""Opt-in experiment using Loc_update afbf6f9 ring rules on the current engine.

Keep the production engine, 12-channel observation and trained weights unchanged.
The adapted mask adds timed ring survival to the existing bomb-safety mask.
"""
from dataclasses import replace

import numpy as np

from bomberman_env import (
    BombermanFFAEnv, Bomb, EMPTY, RIGID_WALL, WOODEN_WALL,
    ACTION_STOP, ACTION_BOMB, OBS_CHANNELS, _MOVE_DELTA,
)


class SuddenDeathEnv(BombermanFFAEnv):
    def __init__(self, *args, sudden_death=False, sudden_death_start=200,
                 sudden_death_interval=20, sudden_death_warning_steps=40,
                 sudden_death_max_ring=None, ring_channel=False, **kwargs):
        timings = (sudden_death_start, sudden_death_interval, sudden_death_warning_steps)
        if any(type(x) is not int for x in timings) or min(timings[:2]) < 1 or timings[2] < 0:
            raise ValueError('Invalid sudden-death timings')
        self.sudden_death = sudden_death
        self.sudden_death_start = sudden_death_start
        self.sudden_death_interval = sudden_death_interval
        self.sudden_death_warning_steps = sudden_death_warning_steps
        self._sudden_death_ring = 0
        # Opt-in extra plane appended after the 12 production channels.
        self.ring_channel = bool(ring_channel)
        self.obs_channels = OBS_CHANNELS + int(self.ring_channel)
        super().__init__(*args, **kwargs)
        # Last ring that closes; None closes all (on 9x9 that leaves no walkable tile).
        full = self.board_size // 2 - 1
        if sudden_death_max_ring is None:
            sudden_death_max_ring = full
        if type(sudden_death_max_ring) is not int or not 1 <= sudden_death_max_ring <= full:
            raise ValueError('Invalid sudden-death max ring')
        self.sudden_death_max_ring = sudden_death_max_ring

    def reset(self):
        self._sudden_death_ring = 0
        return super().reset()

    def _target_sudden_death_ring(self, tick=None):
        tick = self.step_count if tick is None else tick
        if not self.sudden_death or tick < self.sudden_death_start:
            return 0
        return min(1 + (tick - self.sudden_death_start) // self.sudden_death_interval,
                   self.sudden_death_max_ring)

    def _ring_cells(self, ring):
        last = self.board_size - 1 - ring
        return {(r, c) for i in range(ring, last + 1)
                for r, c in ((ring, i), (last, i), (i, ring), (i, last))}

    def step(self, actions):
        if not self.sudden_death:
            return super().step(actions)
        assert len(actions) == self.num_agents
        target = self._target_sudden_death_ring(self.step_count + 1)
        cells = set().union(*(self._ring_cells(r)
                              for r in range(self._sudden_death_ring + 1, target + 1)))
        killed = []
        # Loc_update closes walls BEFORE movement and bomb placement on this tick.
        for agent in self.agents:
            if agent.alive and (agent.row, agent.col) in cells:
                agent.alive = False
                killed.append(agent.agent_id)
        for bomb in list(self.bombs):
            if (bomb.row, bomb.col) in cells:
                owner = self.agents[bomb.owner]
                owner.bombs_placed = max(0, owner.bombs_placed - 1)
                self.bombs.remove(bomb)
        for pos in cells:
            self.grid[pos] = RIGID_WALL
            for mapping in (self.flames, self._flame_owners, self._flame_sources, self.powerups):
                mapping.pop(pos, None)
        self._sudden_death_ring = target
        obs, rewards, dones, info = super().step(actions)
        for aid in killed:
            rewards[aid] -= 1.0
        info.update(sudden_death_ring=target, sudden_death_kills=killed)
        return obs, rewards, dones, info

    def _closure_ticks(self):
        if not self.sudden_death:
            return {}
        return {ring: self.sudden_death_start + (ring - 1) * self.sudden_death_interval - self.step_count
                for ring in range(self._sudden_death_ring + 1, self.sudden_death_max_ring + 1)}

    def _get_obs(self, agent_id, danger=None):
        obs = super()._get_obs(agent_id, danger)
        if not self.ring_channel:
            return obs
        plane = np.zeros((1, self.board_size, self.board_size), dtype=np.float32)
        for ring, delay in self._closure_ticks().items():
            for r, c in self._ring_cells(ring):
                plane[0, r, c] = 1.0 / delay
        return np.concatenate([obs, plane])

    def forecast_closures(self, extra_bomb=None, min_horizon=1):
        """Known bombs plus closing walls; closing tiles stay lethal/blocked.

        Unlike the production forecast, this also returns pre-movement closures.
        It does not assume any future opponent bomb placements or movements.
        """
        clocks = self._closure_ticks()
        next_tick = min(clocks.values(), default=0)
        bombs = [replace(b) for b in self.bombs]
        if extra_bomb is not None:
            bombs.append(replace(extra_bomb))
        horizon = max(min_horizon, 1, *self.flames.values(),
                      *(max(1, b.timer) + self.flame_duration - 1 for b in bombs),
                      next_tick if next_tick <= self.sudden_death_warning_steps else 0)
        burns = np.zeros((horizon, self.board_size, self.board_size), dtype=bool)
        blocked = np.zeros_like(burns)
        closing = np.zeros_like(burns)
        grid = self.grid.copy()
        for pos, duration in self.flames.items():
            burns[:duration, *pos] = True
        for tick in range(horizon):
            cells = set().union(*(self._ring_cells(ring) for ring, delay in clocks.items()
                                  if delay == tick + 1))
            for pos in cells:
                grid[pos] = RIGID_WALL
                closing[tick, *pos] = True
                burns[tick:, *pos] = True
            bombs = [b for b in bombs if (b.row, b.col) not in cells]
            blocked[tick] = grid != EMPTY
            for bomb in bombs:
                blocked[tick, bomb.row, bomb.col] = True
                bomb.timer -= 1
            while True:
                ready = [b for b in bombs if b.timer <= 0]
                if not ready:
                    break
                for bomb in ready:
                    blast = self._blast_cells(bomb.row, bomb.col, bomb.blast_range, grid)
                    for pos in blast:
                        burns[tick:tick + self.flame_duration, *pos] = True
                        if grid[pos] == WOODEN_WALL:
                            grid[pos] = EMPTY
                    for other in bombs:
                        if (other.row, other.col) in blast:
                            other.timer = 0
                    bombs.remove(bomb)
        return burns, blocked, closing


def _ring_survival(env, aid, extra_bomb=None):
    burns, blocked, closing = env.forecast_closures(extra_bomb)
    agent = env.agents[aid]
    for other in env.agents:
        if other.alive and other.agent_id != aid:
            blocked[:, other.row, other.col] = True
            tick = min(int(extra_bomb is not None), len(blocked) - 1)
            for dr, dc in _MOVE_DELTA.values():
                pos = other.row + dr, other.col + dc
                if pos != (agent.row, agent.col) and env._is_walkable(*pos):
                    blocked[tick, *pos] = True
    safe = np.zeros_like(burns)
    future = np.ones_like(burns[0])
    for tick in range(len(burns) - 1, -1, -1):
        continuation = future.copy()
        if tick + 1 < len(burns):
            dest = future & ~blocked[tick + 1]
            continuation[1:] |= dest[:-1]
            continuation[:-1] |= dest[1:]
            continuation[:, 1:] |= dest[:, :-1]
            continuation[:, :-1] |= dest[:, 1:]
            # A source tile closing before the NEXT move cannot be escaped then.
            continuation &= ~closing[tick + 1]
        safe[tick] = ~burns[tick] & continuation
        future = safe[tick]
    return safe[0], blocked[0], closing[0]


def sudden_death_mask(env, aid, base_mask, diagnostics=None):
    """Intersect with timed ring survival; retain the old fallback if infeasible."""
    clocks = env._closure_ticks()
    if not clocks or min(clocks.values()) > env.sudden_death_warning_steps:
        return list(base_mask)
    if diagnostics is not None:
        diagnostics['queries'] = diagnostics.get('queries', 0) + 1
    safe, blocked, closing = _ring_survival(env, aid)
    agent = env.agents[aid]
    pos = agent.row, agent.col
    if closing[pos]:
        if diagnostics is not None:
            diagnostics['unavoidable_pre_move_closure'] = diagnostics.get('unavoidable_pre_move_closure', 0) + 1
        return list(base_mask)
    mask = [False] * len(base_mask)
    mask[ACTION_STOP] = bool(base_mask[ACTION_STOP] and safe[pos])
    for action, (dr, dc) in _MOVE_DELTA.items():
        target = agent.row + dr, agent.col + dc
        if base_mask[action] and 0 <= target[0] < env.board_size and 0 <= target[1] < env.board_size:
            mask[action] = bool(not blocked[target] and safe[target])
    if base_mask[ACTION_BOMB]:
        bomb = Bomb(*pos, aid, env.bomb_timer, agent.blast_range)
        bomb_safe, _, _ = _ring_survival(env, aid, bomb)
        mask[ACTION_BOMB] = bool(bomb_safe[pos])
    if diagnostics is not None:
        diagnostics['blocked_actions'] = diagnostics.get('blocked_actions', 0) + sum(
            old and not new for old, new in zip(base_mask, mask))
        diagnostics['empty_intersections'] = diagnostics.get('empty_intersections', 0) + int(not any(mask))
    return mask if any(mask) else list(base_mask)
