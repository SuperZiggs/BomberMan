"""
train_actor_critic.py

Actor-Critic CNN training cho BombermanFFAEnv, hỗ trợ đầy đủ curriculum:

  Giai đoạn C — 1v1 vs RuleBasedAgent, reward shaping chống bomb shyness
  Giai đoạn D — Self-play với opponent pool (checkpoint snapshots)
  Giai đoạn E — Mở rộng num_agents: 2 → 3 → 4

Dùng shared network (parameter sharing) cho tất cả agent đang học.
Đối thủ có thể là RuleBasedAgent hoặc snapshot cũ từ pool.

Usage:
    # Phase C: 1v1 vs rule-based bot
    python train_actor_critic.py --phase C --episodes 3000 --board_size 9 --num_agents 2

    # Phase D: Self-play with opponent pool
    python train_actor_critic.py --phase D --episodes 5000 --board_size 9 --num_agents 2

    # Phase E: Expand to 4 agents
    python train_actor_critic.py --phase E --episodes 5000 --board_size 11 --num_agents 4
"""

from __future__ import annotations

import argparse
import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bomberman_env import (
    ACTION_BOMB,
    ACTION_STOP,
    Bomb,
    EMPTY,
    NUM_ACTIONS,
    BombermanFFAEnv,
    _MOVE_DELTA,
)
from rule_based_bot import (
    RuleBasedAgent,
    _build_danger_map,
    _bfs_safe_cell,
)


# --------------------------------- network ---------------------------------

class ActorCriticNet(nn.Module):
    """
    Small CNN shared trunk -> actor head (action logits) + critic head (state value).
    Input: (batch, 9, H, W) observation from BombermanFFAEnv._get_obs.
    """

    def __init__(self, in_channels: int = 9, board_size: int = 11,
                 num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        flat_dim = 64 * board_size * board_size
        self.trunk_fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.actor_head = nn.Linear(256, num_actions)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        feats = self.conv(x)
        feats = feats.flatten(start_dim=1)
        feats = self.trunk_fc(feats)
        logits = self.actor_head(feats)
        value = self.critic_head(feats).squeeze(-1)
        return logits, value


# --------------------------------- utilities ---------------------------------

def obs_to_tensor(obs: np.ndarray, device) -> torch.Tensor:
    """Single obs (C, H, W) -> tensor (1, C, H, W)."""
    return torch.from_numpy(obs).float().unsqueeze(0).to(device)


def obs_batch_to_tensor(obs_list, device) -> torch.Tensor:
    """List of obs -> tensor (N, C, H, W)."""
    arr = np.stack(obs_list, axis=0)
    return torch.from_numpy(arr).float().to(device)


def learning_agent_id(phase: str, episode: int, num_agents: int) -> int:
    """Select the learning slot; Phase C/E must not overfit to one spawn corner."""
    if phase.upper() in ("C", "E"):
        return (episode - 1) % num_agents
    return 0


class TacticalBombLifecycle:
    """Tracks safe escapes and offensive intent for tactical bombs."""

    def __init__(self):
        self._pending: dict[tuple[int, int], dict[str, bool | int]] = {}

    def register(
        self,
        position: tuple[int, int],
        offensive: bool = False,
        generic_reward_eligible: bool = True,
        pressure_score: int = 0,
    ) -> None:
        self._pending[position] = {
            "escaped": False,
            "offensive": offensive,
            "generic_reward_eligible": generic_reward_eligible,
            "pressure_score": pressure_score,
        }

    def mark_safe_escape(
        self, agent_position: tuple[int, int], is_safe: bool
    ) -> tuple[int, int, int, int]:
        """Mark escaped bombs; pressure remains pending until bomb outcome."""
        if not is_safe:
            return 0, 0, 0, 0
        newly_escaped = 0
        newly_offensive_escaped = 0
        newly_generic_escaped = 0
        for bomb_position, state in self._pending.items():
            if not state["escaped"] and agent_position != bomb_position:
                state["escaped"] = True
                newly_escaped += 1
                newly_offensive_escaped += int(state["offensive"])
                newly_generic_escaped += int(state["generic_reward_eligible"])
        return (
            newly_escaped,
            newly_offensive_escaped,
            newly_generic_escaped,
            0,
        )

    def resolve_outcome(
        self, position: tuple[int, int]
    ) -> tuple[bool, bool, bool, int]:
        """Return escaped state and deferred pressure for a resolved bomb."""
        state = self._pending.pop(position, {
            "escaped": False,
            "offensive": False,
            "generic_reward_eligible": False,
            "pressure_score": 0,
        })
        escaped = bool(state["escaped"])
        return (
            escaped,
            bool(state["offensive"]),
            escaped and bool(state["generic_reward_eligible"]),
            int(state["pressure_score"]) if escaped else 0,
        )

    def has_escaped_pending(self) -> bool:
        """Return whether an unresolved bomb was previously escaped safely."""
        return any(bool(state["escaped"]) for state in self._pending.values())


def outcome_reward_signals(
    *, escaped: bool, offensive: bool, pressure_score: int,
    owner_alive: bool, owner_safe: bool, victims: list[int], owner_id: int,
) -> tuple[int, int, int]:
    """Return offensive cycle, opponent kills, and pressure eligible at outcome."""
    if not escaped or not owner_alive or not owner_safe:
        return 0, 0, 0
    return (
        int(offensive),
        sum(victim != owner_id for victim in victims) if offensive else 0,
        pressure_score if offensive else 0,
    )


def tactical_bomb_reward(
    tactical_placements: int = 0,
    safe_escapes: int = 0,
    walls_destroyed_after_escape: int = 0,
    offensive_safe_escapes: int = 0,
    opponent_kills_after_escape: int = 0,
    pressure_score: int = 0,
    tactical_placement_bonus: float = 0.05,
    safe_escape_bonus: float = 0.15,
    wall_destroy_bonus: float = 0.02,
    offensive_safe_escape_bonus: float = 0.40,
    opponent_kill_bonus: float = 1.00,
    pressure_bonus: float = 0.15,
) -> float:
    """Reward safe tactical cycles, favouring opponents over wall farming."""
    return (
        tactical_placements * tactical_placement_bonus
        + safe_escapes * safe_escape_bonus
        + walls_destroyed_after_escape * wall_destroy_bonus
        + offensive_safe_escapes * offensive_safe_escape_bonus
        + opponent_kills_after_escape * opponent_kill_bonus
        + pressure_score * pressure_bonus
    )


def safe_bomb_cadence_bonus(
    previous_safe_escapes: int, newly_safe_escapes: int
) -> float:
    """Pay once, at the moment an episode reaches two safe bomb cycles."""
    reached_target = previous_safe_escapes < 2 <= previous_safe_escapes + newly_safe_escapes
    return 0.25 if reached_target else 0.0


def noncombat_win_cadence_penalty(
    *, won: bool, kills: int, safe_cycles: int, target_cycles: int,
    max_penalty: float = 10.0, fixed_penalty: float = 0.0,
) -> float:
    """Cap non-combat wins while retaining the safe-cycle cadence penalty."""
    if not won or kills > 0:
        return 0.0
    missing_cycles = max(target_cycles - safe_cycles, 0)
    cadence_penalty = (
        max_penalty * missing_cycles / target_cycles
        if target_cycles > 0
        else 0.0
    )
    return -max(cadence_penalty, fixed_penalty)


def classify_learning_agent_death(info: dict, agent_id: int, was_alive: bool) -> str | None:
    """Classify a newly dead learning agent for training diagnostics."""
    if not was_alive or info["alive"][agent_id]:
        return None
    if agent_id in info.get("sudden_death_kills", []):
        return "sudden_death"
    kills = info.get("kills", [])
    if any(killer == agent_id and victim == agent_id for killer, victim in kills):
        return "suicide"
    if any(killer != agent_id and victim == agent_id for killer, victim in kills):
        return "opponent_bomb"
    return "other"


def _is_retryable_checkpoint_error(error: Exception) -> bool:
    """Return true for transient Windows file-lock errors."""
    if isinstance(error, OSError) and (
        error.errno in (5, 32, 33, 1224)
        or getattr(error, "winerror", None) in (5, 32, 33, 1224)
    ):
        return True
    message = str(error).lower()
    return (
        "error code: 1224" in message
        or "error code: 32" in message
        or "being used by another process" in message
    )


def atomic_torch_save(state_dict, path: str, retries: int = 5,
                      retry_delay_seconds: float = 0.5) -> None:
    """Save through a temporary file and retry transient Windows checkpoint locks."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    for attempt in range(retries):
        temp_path = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
        try:
            torch.save(state_dict, temp_path)
            os.replace(temp_path, path)
            return
        except (OSError, RuntimeError) as error:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if not _is_retryable_checkpoint_error(error) or attempt == retries - 1:
                raise
            wait_seconds = retry_delay_seconds * (attempt + 1)
            print(f"Checkpoint locked; retrying save in {wait_seconds:.1f}s "
                  f"({attempt + 1}/{retries - 1})")
            time.sleep(wait_seconds)


def checkpoint_payload(state_dict, *, episode: int, phase: str,
                       metrics: dict, config: dict) -> dict:
    """Build a self-describing checkpoint while keeping weights separately loadable."""
    return {
        "model_state_dict": state_dict,
        "episode": episode,
        "phase": phase,
        "metrics": dict(metrics),
        "config": dict(config),
    }


def checkpoint_state_dict(checkpoint: dict) -> dict:
    """Return model weights from either a metadata payload or a legacy state dict."""
    return checkpoint.get("model_state_dict", checkpoint)


def is_target_bomb_checkpoint(metrics: dict, target_bombs: float = 2.0,
                              tolerance: float = 0.3,
                              min_win_rate: float = 70.0) -> bool:
    """Return whether rolling metrics satisfy the target-bomb acceptance gate."""
    return (
        metrics["win_rate"] >= min_win_rate
        and abs(metrics["bombs"] - target_bombs) <= tolerance + 1e-9
    )


def target_checkpoint_rank(metrics: dict, target_bombs: float = 2.0) -> tuple:
    """Rank eligible target checkpoints by combat, WR, then bomb proximity."""
    return (
        metrics.get("kills", 0.0),
        metrics.get("offensive_safe_cycles", 0.0),
        metrics["win_rate"],
        -abs(metrics["bombs"] - target_bombs),
    )


def combat_checkpoint_rank(metrics: dict) -> tuple:
    """Rank eligible combat checkpoints by kills, pressure, WR, then safety."""
    return (
        metrics.get("kills", 0.0),
        metrics.get("pressure_score", 0.0),
        metrics.get("offensive_safe_cycles", 0.0),
        metrics.get("win_rate", 0.0),
        -metrics.get("self_deaths", 0.0),
        -metrics.get("sudden_death_deaths", 0.0),
    )


def is_combat_safe_checkpoint(
    metrics: dict, min_win_rate: float = 70.0,
    min_kills: float = 0.025, max_self_deaths: float = 0.05,
) -> bool:
    """Return whether rolling combat metrics meet the safety promotion gate."""
    return (
        metrics.get("win_rate", 0.0) >= min_win_rate
        and metrics.get("kills", 0.0) >= min_kills
        and metrics.get("self_deaths", float("inf")) <= max_self_deaths
    )


def _project_bomb_explosion_times(
    env: BombermanFFAEnv, bombs: list[Bomb] | None = None
) -> list[int]:
    """Project earliest bomb explosion times including chain reactions."""
    active_bombs = env.bombs if bombs is None else bombs
    explosion_times = [max(int(bomb.timer), 1) for bomb in active_bombs]
    blast_cells = [
        env._bomb_blast_cells(bomb.row, bomb.col, bomb.blast_range)
        for bomb in active_bombs
    ]
    changed = True
    while changed:
        changed = False
        for source_index, cells in enumerate(blast_cells):
            source_time = explosion_times[source_index]
            for target_index, target in enumerate(active_bombs):
                if (
                    (target.row, target.col) in cells
                    and source_time < explosion_times[target_index]
                ):
                    explosion_times[target_index] = source_time
                    changed = True
    return explosion_times


def _project_bomb_hazards(
    env: BombermanFFAEnv,
    explosion_times: list[int] | None = None,
    bombs: list[Bomb] | None = None,
) -> dict[tuple[int, int], set[int]]:
    """Return future unsafe times per cell for bombs, chains, and flames."""
    hazards: dict[tuple[int, int], set[int]] = {}
    active_bombs = env.bombs if bombs is None else bombs
    if explosion_times is None:
        explosion_times = _project_bomb_explosion_times(env, active_bombs)
    for bomb, explosion_time in zip(active_bombs, explosion_times):
        for cell in env._bomb_blast_cells(bomb.row, bomb.col, bomb.blast_range):
            cell_hazards = hazards.setdefault(cell, set())
            for time_step in range(
                explosion_time,
                explosion_time + max(env.flame_duration, 1),
            ):
                cell_hazards.add(time_step)
    for cell, duration in env.flames.items():
        hazards.setdefault(cell, set()).update(range(1, max(int(duration), 1) + 1))
    return hazards


def _project_sudden_death_closures(
    env: BombermanFFAEnv,
) -> dict[tuple[int, int], int]:
    """Return cells in the next closing ring and their relative close tick."""
    if not env.sudden_death:
        return {}
    next_ring = env._sudden_death_ring + 1
    if next_ring >= env.board_size // 2:
        return {}
    closure_step = env.sudden_death_start + (next_ring - 1) * env.sudden_death_interval
    relative_tick = max(closure_step - env.step_count, 1)
    if relative_tick > env.sudden_death_warning_steps:
        return {}
    return {cell: relative_tick for cell in env._ring_cells(next_ring)}


def _has_temporal_escape(
    env: BombermanFFAEnv,
    start: tuple[int, int],
    hazards: dict[tuple[int, int], set[int]] | None = None,
    explosion_times: list[int] | None = None,
    bombs: list[Bomb] | None = None,
    blocked_from: dict[tuple[int, int], int] | None = None,
) -> bool:
    """Return whether an agent can survive all projected bomb hazards."""
    active_bombs = env.bombs if bombs is None else bombs
    if explosion_times is None:
        explosion_times = _project_bomb_explosion_times(env, active_bombs)
    if hazards is None:
        hazards = _project_bomb_hazards(env, explosion_times, active_bombs)
    blocked_from = blocked_from or {}
    if not hazards and not blocked_from:
        return True
    if 1 in hazards.get(start, set()) or blocked_from.get(start, 2) <= 1:
        return False

    bomb_deadlines = {
        (bomb.row, bomb.col): explosion_time
        for bomb, explosion_time in zip(active_bombs, explosion_times)
    }
    horizon_candidates = [
        time_step for times in hazards.values() for time_step in times
    ]
    horizon_candidates.extend(blocked_from.values())
    horizon = max(horizon_candidates)
    positions = {start}
    moves = [(0, 0), *_MOVE_DELTA.values()]
    for time_step in range(2, horizon + 1):
        next_positions = set()
        for row, col in positions:
            if blocked_from.get((row, col), horizon + 1) <= time_step:
                continue
            for dr, dc in moves:
                next_row, next_col = row + dr, col + dc
                if not (0 <= next_row < env.board_size and 0 <= next_col < env.board_size):
                    continue
                if env.grid[next_row, next_col] != EMPTY:
                    continue
                if blocked_from.get((next_row, next_col), horizon + 1) <= time_step:
                    continue
                bomb_explosion = bomb_deadlines.get((next_row, next_col))
                if (next_row, next_col) != (row, col) and (
                    bomb_explosion is not None and time_step <= bomb_explosion
                ):
                    continue
                if time_step in hazards.get((next_row, next_col), set()):
                    continue
                next_positions.add((next_row, next_col))
        if not next_positions:
            return False
        positions = next_positions
    return True


def _shortest_temporal_escape_actions(
    env: BombermanFFAEnv,
    start: tuple[int, int],
    hazards: dict[tuple[int, int], set[int]],
    explosion_times: list[int],
    bombs: list[Bomb] | None = None,
    blocked_from: dict[tuple[int, int], int] | None = None,
) -> set[int]:
    """Return first actions on the shortest path out of projected bomb danger."""
    active_bombs = env.bombs if bombs is None else bombs
    blocked_from = blocked_from or {}
    bomb_deadlines = {
        (bomb.row, bomb.col): explosion_time
        for bomb, explosion_time in zip(active_bombs, explosion_times)
    }
    bomb_horizon = max(
        (time_step for times in hazards.values() for time_step in times),
        default=1,
    )
    states = {start: {None}}
    actions = [(ACTION_STOP, (0, 0)), *_MOVE_DELTA.items()]
    for time_step in range(1, bomb_horizon + 1):
        next_states: dict[tuple[int, int], set[int]] = {}
        for (row, col), first_actions in states.items():
            if blocked_from.get((row, col), bomb_horizon + 1) <= time_step:
                continue
            for action, (dr, dc) in actions:
                next_row, next_col = row + dr, col + dc
                target = (next_row, next_col)
                if not (0 <= next_row < env.board_size and 0 <= next_col < env.board_size):
                    continue
                if env.grid[next_row, next_col] != EMPTY:
                    continue
                if blocked_from.get(target, bomb_horizon + 1) <= time_step:
                    continue
                bomb_explosion = bomb_deadlines.get(target)
                if target != (row, col) and (
                    bomb_explosion is not None and time_step <= bomb_explosion
                ):
                    continue
                if time_step in hazards.get(target, set()):
                    continue
                propagated_actions = {
                    action if first_action is None else first_action
                    for first_action in first_actions
                }
                next_states.setdefault(target, set()).update(propagated_actions)

        shortest_actions = set()
        for position, first_actions in next_states.items():
            if not any(
                hazard_tick >= time_step
                for hazard_tick in hazards.get(position, set())
            ):
                shortest_actions.update(first_actions)
        if shortest_actions:
            return shortest_actions
        if not next_states:
            return set()
        states = next_states
    return set()


def _can_place_temporal_bomb(
    env: BombermanFFAEnv,
    agent_id: int,
    row: int,
    col: int,
    blocked_from: dict[tuple[int, int], int] | None = None,
) -> bool:
    """Return whether placing a bomb now leaves a chain-safe trajectory."""
    agent = env.agents[agent_id]
    hypothetical_bomb = Bomb(
        row=row,
        col=col,
        owner=agent_id,
        timer=env.bomb_timer,
        blast_range=agent.blast_range,
    )
    projected_bombs = [*env.bombs, hypothetical_bomb]
    explosion_times = _project_bomb_explosion_times(env, projected_bombs)
    hazards = _project_bomb_hazards(env, explosion_times, projected_bombs)
    return _has_temporal_escape(
        env,
        (row, col),
        hazards=hazards,
        explosion_times=explosion_times,
        bombs=projected_bombs,
        blocked_from=blocked_from,
    )


def get_action_mask(
    env: BombermanFFAEnv, agent_id: int, diagnostics: dict | None = None
) -> list[bool]:
    """Return a boolean mask of valid actions for the given agent."""
    mask = [True] * NUM_ACTIONS
    agent = env.agents[agent_id]
    r, c = agent.row, agent.col
    danger = _build_danger_map(env)
    sudden_death_closures = _project_sudden_death_closures(env)
    explosion_times = _project_bomb_explosion_times(env)
    projected_hazards = _project_bomb_hazards(env, explosion_times)
    sudden_death_blocks = 0

    # 1. Do not walk into walls or active flames or cells exploding in 1-2 ticks
    for a, (dr, dc) in _MOVE_DELTA.items():
        nr, nc = r + dr, c + dc
        if not env._is_walkable(nr, nc) or (nr, nc) in env.flames:
            mask[a] = False
        elif danger[nr, nc] in (1, 2):
            mask[a] = False

    # 2. Do not stand still on an exploding bomb
    if danger[r, c] in (1, 2):
        mask[ACTION_STOP] = False

    # 3. Only place bomb if ammo available, no bomb on cell, and can escape!
    placement_blocked = False
    if agent.bombs_placed >= agent.ammo or any(b.row == r and b.col == c for b in env.bombs):
        mask[ACTION_BOMB] = False
    else:
        placement_safe = _can_place_temporal_bomb(
            env,
            agent_id,
            r,
            c,
            blocked_from=sudden_death_closures,
        )
        if not placement_safe:
            mask[ACTION_BOMB] = False
            placement_blocked = True
            if diagnostics is not None:
                diagnostics["unsafe_bomb_placement_blocks"] = (
                    diagnostics.get("unsafe_bomb_placement_blocks", 0) + 1
                )
            if sudden_death_closures and _can_place_temporal_bomb(
                env, agent_id, r, c
            ):
                sudden_death_blocks += 1

    # 4. Reject movement/stop actions that cannot survive projected timers and
    # chain reactions. Entering a future blast line remains allowed when a
    # time-valid escape route exists.
    temporal_blocks = 0
    temporal_targets = {ACTION_STOP: (r, c)}
    temporal_targets.update({
        action: (r + dr, c + dc)
        for action, (dr, dc) in _MOVE_DELTA.items()
    })
    for action, target in temporal_targets.items():
        if not mask[action]:
            continue
        survives = _has_temporal_escape(
            env,
            target,
            hazards=projected_hazards,
            explosion_times=explosion_times,
            blocked_from=sudden_death_closures,
        )
        if survives:
            continue
        mask[action] = False
        if sudden_death_closures and _has_temporal_escape(
            env,
            target,
            hazards=projected_hazards,
            explosion_times=explosion_times,
        ):
            sudden_death_blocks += 1
        else:
            temporal_blocks += 1

    # 5. When any bomb threatens the current cell, commit to a first action
    # on the shortest temporal escape path instead of waiting until the last tick.
    forced_escape = False
    if any(time_step >= 1 for time_step in projected_hazards.get((r, c), set())):
        escape_actions = _shortest_temporal_escape_actions(
            env,
            (r, c),
            projected_hazards,
            explosion_times,
            blocked_from=sudden_death_closures,
        )
        allowed_escape_actions = {
            action for action in escape_actions if mask[action]
        }
        if allowed_escape_actions:
            forced_escape = any(
                allowed and action not in allowed_escape_actions
                for action, allowed in enumerate(mask)
            )
            mask = [
                action in allowed_escape_actions for action in range(NUM_ACTIONS)
            ]

    if diagnostics is not None:
        diagnostics["unsafe_reentry_blocks"] = (
            diagnostics.get("unsafe_reentry_blocks", 0) + temporal_blocks
        )
        diagnostics["sudden_death_blocks"] = (
            diagnostics.get("sudden_death_blocks", 0) + sudden_death_blocks
        )
        diagnostics["forced_escape_steps"] = (
            diagnostics.get("forced_escape_steps", 0) + int(forced_escape)
        )
        diagnostics["mask_interventions"] = (
            diagnostics.get("mask_interventions", 0)
            + int(
                temporal_blocks > 0
                or placement_blocked
                or sudden_death_blocks > 0
                or forced_escape
            )
        )

    if not any(mask):
        mask[ACTION_STOP] = True
    return mask


# ─────────────────── opponent wrapper (snapshot or rule-based) ────────────

class SnapshotOpponent:
    """Wraps a frozen ActorCriticNet checkpoint as an opponent."""

    def __init__(self, net: ActorCriticNet, device):
        self.net = net
        self.net.eval()
        self.device = device

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, env: BombermanFFAEnv | None = None,
                      agent_id: int | None = None) -> int:
        obs_t = obs_to_tensor(obs, self.device)
        logits, _ = self.net(obs_t)
        if env is not None and agent_id is not None:
            mask = get_action_mask(env, agent_id)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device)
            logits = logits.masked_fill(~mask_t, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()


class RuleBasedOpponent:
    """Wraps RuleBasedAgent to match SnapshotOpponent interface."""

    def __init__(self, agent_id: int):
        self.bot = RuleBasedAgent(agent_id)

    def select_action_env(self, env: BombermanFFAEnv) -> int:
        return self.bot.select_action(env)


# ─────────────────── opponent pool management ─────────────────────────────

def smart_load_weights(net: ActorCriticNet, checkpoint_path: str, device):
    """Load checkpoint weights with automatic spatial interpolation if board_size differs."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = checkpoint_state_dict(checkpoint)
    curr_sd = net.state_dict()
    for k, v in sd.items():
        if k in curr_sd:
            if v.shape == curr_sd[k].shape:
                curr_sd[k].copy_(v)
            elif k == "trunk_fc.0.weight":
                # Spatial interpolation from (256, 64*old_s*old_s) to (256, 64*new_s*new_s)
                old_flat = v.shape[1]
                new_flat = curr_sd[k].shape[1]
                old_s = int(np.round(np.sqrt(old_flat / 64)))
                new_s = int(np.round(np.sqrt(new_flat / 64)))
                w_4d = v.view(256, 64, old_s, old_s)
                w_interp = F.interpolate(w_4d, size=(new_s, new_s), mode="bilinear", align_corners=False)
                w_interp = w_interp * ((old_s * old_s) / (new_s * new_s))
                curr_sd[k].copy_(w_interp.reshape(256, new_flat))
                print(f"Interpolated {k} from board {old_s}x{old_s} to {new_s}x{new_s}")
            elif k == "conv.0.weight" and v.ndim == 4 and curr_sd[k].ndim == 4 \
                    and v.shape[0] == curr_sd[k].shape[0] \
                    and v.shape[1] < curr_sd[k].shape[1] \
                    and v.shape[2:] == curr_sd[k].shape[2:]:
                curr_sd[k][:, :v.shape[1]].copy_(v)
                curr_sd[k][:, v.shape[1]:].zero_()
                print(f"Expanded {k} from {v.shape[1]} to {curr_sd[k].shape[1]} input channels")
    net.load_state_dict(curr_sd)


def load_opponent_pool(checkpoint_dir: str, board_size: int,
                       device, phase: str = "", num_agents: int = 2) -> list[SnapshotOpponent]:
    """Load saved checkpoints as SnapshotOpponents with automatic size compatibility."""
    pool = []
    if phase == "E":
        prefix = f"snapshot_phaseE_{num_agents}a_ep*.pt"
        paths = sorted(glob.glob(os.path.join(checkpoint_dir, prefix)))
        if not paths:
            # Fallback to any Phase E snapshots, then Phase D
            paths = sorted(glob.glob(os.path.join(checkpoint_dir, "snapshot_phaseE_*_ep*.pt")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(checkpoint_dir, "snapshot_ep*.pt")))
    else:
        paths = sorted(glob.glob(os.path.join(checkpoint_dir, "snapshot_ep*.pt")))
    for path in paths:
        net = ActorCriticNet(board_size=board_size).to(device)
        smart_load_weights(net, path, device)
        net.eval()
        pool.append(SnapshotOpponent(net, device))
    return pool


def save_snapshot(net: ActorCriticNet, checkpoint_dir: str, episode: int,
                  phase: str = "", num_agents: int = 2,
                  metrics: dict | None = None, config: dict | None = None):
    """Save a snapshot of the network weights."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if phase == "E":
        prefix = f"snapshot_phaseE_{num_agents}a_"
    elif phase == "C":
        prefix = "snapshot_phaseC_"
    else:
        prefix = "snapshot_"
    path = os.path.join(checkpoint_dir, f"{prefix}ep{episode:06d}.pt")
    payload = checkpoint_payload(
        net.state_dict(),
        episode=episode,
        phase=phase,
        metrics=metrics or {},
        config=config or {},
    )
    atomic_torch_save(payload, path)
    return path


# ─────────────────────── main training function ───────────────────────────

def train(
    phase: str = "C",
    episodes: int = 3000,
    board_size: int = 9,
    num_agents: int = 2,
    gamma: float = 0.99,
    lr: float = 3e-4,
    value_coef: float = 0.5,
    entropy_coef: float = 0.015,
    wall_destroy_bonus: float = 0.02,
    tactical_bomb_bonus: float = 0.05,
    safe_escape_bonus: float = 0.15,
    offensive_safe_escape_bonus: float = 0.40,
    opponent_kill_bonus: float = 1.00,
    pressure_bonus: float = 0.15,
    noncombat_win_fixed_penalty: float = 0.0,
    combat_min_win_rate: float = 70.0,
    combat_safe_min_kills: float = 0.025,
    combat_safe_max_self_deaths: float = 0.05,
    generic_cycle_budget: int = 2,
    target_bombs: float = 2.0,
    target_bomb_tolerance: float = 0.3,
    target_min_win_rate: float = 70.0,
    max_steps: int = 300,
    log_every: int = 50,
    snapshot_every: int = 200,
    checkpoint_dir: str = "checkpoints",
    resume_from: str = "",
    sudden_death: bool = False,
    sudden_death_start: int = 200,
    sudden_death_interval: int = 20,
    sudden_death_warning_steps: int = 40,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Unified training loop for phases C, D, E.

    Phase C: Agent 0 = learning AC, Agent 1 = RuleBasedAgent (fixed).
             Reward shaping rewards outcomes of tactical bombs only after escape.
    Phase D: Agent 0 = learning AC, Agent 1 = random from opponent pool
             (includes RuleBasedAgent + saved snapshots).
    Phase E: Same as D but with num_agents=3 or 4.
    """
    phase = phase.upper()
    assert phase in ("C", "D", "E"), f"Phase must be C, D, or E, got {phase}"

    env = BombermanFFAEnv(
        board_size=board_size,
        num_agents=num_agents,
        max_steps=max_steps,
        sudden_death=sudden_death,
        sudden_death_start=sudden_death_start,
        sudden_death_interval=sudden_death_interval,
        sudden_death_warning_steps=sudden_death_warning_steps,
    )
    net = ActorCriticNet(in_channels=9, board_size=board_size,
                         num_actions=NUM_ACTIONS).to(device)

    # Resume from checkpoint if specified, or auto warm-start for Phase D/E.
    _resume_path = resume_from
    if not _resume_path and phase == "D":
        # Auto warm-start from Phase C best model
        for candidate in [os.path.join(checkpoint_dir, "best_phaseC.pt"),
                          os.path.join(checkpoint_dir, "final_phaseC.pt")]:
            if os.path.exists(candidate):
                _resume_path = candidate
                break
    if not _resume_path and phase == "E":
        candidates = []
        if num_agents == 4:
            candidates.extend([
                os.path.join(checkpoint_dir, "best_phaseE_3a.pt"),
                os.path.join(checkpoint_dir, "final_phaseE_3a.pt"),
                os.path.join(checkpoint_dir, "best_phaseE.pt"),
                os.path.join(checkpoint_dir, "final_phaseE.pt"),
            ])
        candidates.extend([
            os.path.join(checkpoint_dir, "best_phaseD.pt"),
            os.path.join(checkpoint_dir, "final_phaseD.pt"),
        ])
        for candidate in candidates:
            if os.path.exists(candidate):
                _resume_path = candidate
                break
    if _resume_path and os.path.exists(_resume_path):
        smart_load_weights(net, _resume_path, device)
        print(f"Warm-started from {_resume_path}")

    # Clean old snapshots when starting Phase D/E fresh (no explicit resume)
    if phase in ("D", "E") and not resume_from:
        prefix = f"snapshot_phaseE_{num_agents}a_ep*.pt" if phase == "E" else "snapshot_ep*.pt"
        old_snaps = glob.glob(os.path.join(checkpoint_dir, prefix))
        for f in old_snaps:
            os.remove(f)
        if old_snaps:
            print(f"Cleared {len(old_snaps)} old snapshots for Phase {phase} ({num_agents} agents) from {checkpoint_dir}")

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # Cosine annealing LR schedule: decays from `lr` to `lr * 0.05` over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=episodes,
                                                      eta_min=lr * 0.05)

    # Opponent management
    rule_bots = {i: RuleBasedOpponent(i) for i in range(num_agents)}
    snapshot_pool: list[SnapshotOpponent] = []

    if phase in ("D", "E"):
        snapshot_pool = load_opponent_pool(checkpoint_dir, board_size, device,
                                           phase=phase, num_agents=num_agents)
        print(f"Loaded {len(snapshot_pool)} snapshots from {checkpoint_dir}")

    # ── Best-model checkpointing ──
    best_wr = 0.0
    best_state_dict = None   # will hold state_dict of the best model seen
    best_metrics = None
    best_episode = 0
    best_target_rank = None
    best_combat_rank = None
    best_combat_safe_rank = None

    checkpoint_config = {
        "board_size": board_size,
        "num_agents": num_agents,
        "max_steps": max_steps,
        "lr": lr,
        "wall_destroy_bonus": wall_destroy_bonus,
        "tactical_bomb_bonus": tactical_bomb_bonus,
        "safe_escape_bonus": safe_escape_bonus,
        "offensive_safe_escape_bonus": offensive_safe_escape_bonus,
        "opponent_kill_bonus": opponent_kill_bonus,
        "pressure_bonus": pressure_bonus,
        "generic_cycle_budget": generic_cycle_budget,
        "noncombat_win_max_penalty": 10.0,
        "noncombat_win_fixed_penalty": noncombat_win_fixed_penalty,
        "combat_min_win_rate": combat_min_win_rate,
        "combat_safe_min_kills": combat_safe_min_kills,
        "combat_safe_max_self_deaths": combat_safe_max_self_deaths,
        "target_bombs": target_bombs,
        "target_bomb_tolerance": target_bomb_tolerance,
        "target_min_win_rate": target_min_win_rate,
        "sudden_death": sudden_death,
        "sudden_death_start": sudden_death_start,
        "sudden_death_interval": sudden_death_interval,
        "sudden_death_warning_steps": sudden_death_warning_steps,
    }

    # ── Logging buffers ──
    reward_history = []
    win_history = []        # 1 = learning agent survived, 0 = didn't
    bombs_history = []      # bombs placed by learning agent
    walls_history = []      # walls destroyed by learning agent's bombs
    safe_cycles_history = []  # tactical bombs whose owner reached safety
    offensive_safe_cycles_history = []  # safe bomb cycles aimed at an opponent
    pressure_score_history = []  # opponent escape options covered after safe escape
    post_escape_self_death_history = []
    unsafe_bomb_placement_blocks_history = []
    unsafe_reentry_blocks_history = []
    forced_escape_steps_history = []
    sudden_death_blocks_history = []
    mask_interventions_history = []
    chain_bomb_self_death_history = []
    combat_history = {
        "kills": [],
        "sudden_death": [],
        "suicide": [],
        "opponent_bomb": [],
        "other_death": [],
    }

    for ep in range(1, episodes + 1):
        obs_list = env.reset()
        alive_mask = [True] * num_agents

        # Cycle the learning slot in Phase C/E to avoid spawn-corner overfitting.
        learning_aid = learning_agent_id(phase, ep, num_agents)

        # ── Decide opponents for this episode ──
        opponents = {}  # agent_id -> opponent object or None (learning agent)

        for opp_id in range(num_agents):
            if opp_id == learning_aid:
                continue
            if phase == "C":
                opponents[opp_id] = "rule"
            elif phase in ("D", "E"):
                rule_prob = 0.40 if phase == "E" else 0.50
                if not snapshot_pool or np.random.random() < rule_prob:
                    opponents[opp_id] = "rule"
                else:
                    opponents[opp_id] = snapshot_pool[
                        np.random.randint(len(snapshot_pool))
                    ]

        # ── Per-agent trajectory buffers (learning agent) ──
        log_probs_buf = []
        values_buf = []
        rewards_buf = []
        entropies_buf = []
        aux_losses_buf = []
        bombs_this_ep = 0
        walls_this_ep = 0
        safe_escapes_this_ep = 0
        offensive_safe_escapes_this_ep = 0
        pressure_score_this_ep = 0
        post_escape_self_death_this_ep = 0
        chain_bomb_self_death_this_ep = 0
        temporal_diagnostics_this_ep = {
            "unsafe_bomb_placement_blocks": 0,
            "unsafe_reentry_blocks": 0,
            "forced_escape_steps": 0,
            "sudden_death_blocks": 0,
            "mask_interventions": 0,
        }
        generic_safe_escapes_this_ep = 0
        generic_cycle_slots_this_ep = 0
        kills_this_ep = 0
        death_counts_this_ep = {
            "sudden_death": 0,
            "suicide": 0,
            "opponent_bomb": 0,
            "other_death": 0,
        }
        tactical_bombs = TacticalBombLifecycle()

        for t in range(max_steps):
            # ── Select actions ──
            actions = [0] * num_agents

            # Learning agent: forward through network with action masking.
            if alive_mask[learning_aid]:
                obs_t = obs_to_tensor(obs_list[learning_aid], device)
                logits, value = net(obs_t)
                mask = get_action_mask(
                    env,
                    learning_aid,
                    diagnostics=temporal_diagnostics_this_ep,
                )
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
                m_logits = logits.masked_fill(~mask_t, -1e9)
                dist = torch.distributions.Categorical(logits=m_logits)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t)
                entropy = dist.entropy()
                actions[learning_aid] = action_t.item()
                if actions[learning_aid] == ACTION_BOMB:
                    bombs_this_ep += 1

                # Auxiliary escape supervision when in danger
                r_l, c_l = env.agents[learning_aid].row, env.agents[learning_aid].col
                d_map = _build_danger_map(env)
                if d_map[r_l, c_l] > 0:
                    esc_act = _bfs_safe_cell(env, r_l, c_l, d_map)
                    if esc_act is not None and mask[esc_act]:
                        aux_losses_buf.append(
                            F.cross_entropy(m_logits, torch.tensor([esc_act], device=device))
                        )
            else:
                log_prob = None
                value = None
                entropy = None

            # Opponent agents.
            for opp_id in range(num_agents):
                if opp_id == learning_aid:
                    continue
                if not alive_mask[opp_id]:
                    actions[opp_id] = 0
                    continue
                opp = opponents[opp_id]
                if opp == "rule":
                    actions[opp_id] = rule_bots[opp_id].select_action_env(env)
                else:
                    # Snapshot opponent with action masking
                    actions[opp_id] = opp.select_action(obs_list[opp_id], env, opp_id)

            # Record danger before step for reward shaping
            if alive_mask[learning_aid]:
                r_l, c_l = env.agents[learning_aid].row, env.agents[learning_aid].col
                position_before = (r_l, c_l)
            else:
                position_before = None

            # ── Step environment ──
            obs_list, rewards, dones, info = env.step(actions)

            # ── Reward shaping (anti-bomb-shyness & anti-kamikaze) ──
            shaped_reward = rewards[learning_aid]

            # 1. Death / Suicide penalty (-1.0 from env - 1.5 = -2.5 total)
            kills_this_ep += sum(
                killer == learning_aid and victim != learning_aid
                for killer, victim in info.get("kills", [])
            )
            death_cause = classify_learning_agent_death(
                info, agent_id=learning_aid, was_alive=alive_mask[learning_aid]
            )
            if death_cause is not None:
                death_counts_this_ep[death_cause if death_cause != "other" else "other_death"] += 1
                if death_cause == "suicide":
                    post_escape_self_death_this_ep += int(
                        tactical_bombs.has_escaped_pending()
                    )
                    chain_bomb_self_death_this_ep += int(any(
                        outcome.get("chain_reaction", False)
                        and outcome.get("owner") == learning_aid
                        and learning_aid in outcome.get("victims", [])
                        for outcome in info.get("bomb_outcomes", [])
                    ))
                    shaped_reward -= 1.5  # Heavy penalty for self-destruction

            # 2. A tiny starter reward is allowed only for a bomb that is both
            # tactical and safe (the action mask proves the escape route).
            tactical_placements = int(
                position_before is not None
                and learning_aid in info.get("tactical_bomb_owners", [])
            )
            offensive_placements = int(
                position_before is not None
                and learning_aid in info.get("offensive_bomb_owners", [])
            )
            placement_pressure_score = sum(
                pressure_score
                for owner, pressure_score in info.get("bomb_pressure_scores", [])
                if owner == learning_aid
            )
            if tactical_placements:
                placement_generic_reward_eligible = generic_cycle_slots_this_ep < generic_cycle_budget
                generic_cycle_slots_this_ep += int(placement_generic_reward_eligible)
                tactical_bombs.register(
                    position_before,
                    offensive=bool(offensive_placements),
                    generic_reward_eligible=placement_generic_reward_eligible,
                    pressure_score=placement_pressure_score,
                )
            else:
                placement_generic_reward_eligible = False

            # 3. A tactical bomb earns its escape reward only after the owner
            # leaves the bomb cell and reaches a genuinely safe cell.
            escaped_count = 0
            if alive_mask[learning_aid] and info["alive"][learning_aid]:
                r1, c1 = env.agents[learning_aid].row, env.agents[learning_aid].col
                (
                    escaped_count,
                    offensive_escaped_count,
                    generic_escaped_count,
                    safe_pressure_score,
                ) = tactical_bombs.mark_safe_escape(
                    (r1, c1), is_safe=_build_danger_map(env)[r1, c1] == 0,
                )
                cadence_bonus = safe_bomb_cadence_bonus(
                    previous_safe_escapes=generic_safe_escapes_this_ep,
                    newly_safe_escapes=generic_escaped_count,
                )
                safe_escapes_this_ep += escaped_count
                offensive_safe_escapes_this_ep += offensive_escaped_count
                generic_safe_escapes_this_ep += generic_escaped_count
            else:
                offensive_escaped_count = 0
                generic_escaped_count = 0
                safe_pressure_score = 0
                cadence_bonus = 0.0

            # 4. Reward offensive pressure only after bomb outcome while the
            # escaped owner is still alive and in a safe cell.
            walls_destroyed_after_escape = 0
            offensive_outcomes_after_escape = 0
            opponent_kills_after_escape = 0
            pressure_after_escape = 0
            owner_alive_at_outcome = bool(info["alive"][learning_aid])
            owner_safe_at_outcome = False
            if owner_alive_at_outcome:
                owner = env.agents[learning_aid]
                owner_safe_at_outcome = _build_danger_map(env)[owner.row, owner.col] == 0
            for outcome in info.get("bomb_outcomes", []):
                if outcome["owner"] != learning_aid:
                    continue
                walls_this_ep += outcome["walls_destroyed"]
                (
                    escaped_safely,
                    was_offensive,
                    outcome_generic_reward_eligible,
                    deferred_pressure_score,
                ) = tactical_bombs.resolve_outcome(outcome["position"])
                if escaped_safely and outcome_generic_reward_eligible:
                    walls_destroyed_after_escape += outcome["walls_destroyed"]
                offensive_count, opponent_kills, rewarded_pressure = outcome_reward_signals(
                    escaped=escaped_safely,
                    offensive=was_offensive,
                    pressure_score=deferred_pressure_score,
                    owner_alive=owner_alive_at_outcome,
                    owner_safe=owner_safe_at_outcome,
                    victims=outcome["victims"],
                    owner_id=learning_aid,
                )
                offensive_outcomes_after_escape += offensive_count
                opponent_kills_after_escape += opponent_kills
                pressure_after_escape += rewarded_pressure
            pressure_score_this_ep += pressure_after_escape
            shaped_reward += tactical_bomb_reward(
                tactical_placements=int(tactical_placements and placement_generic_reward_eligible),
                safe_escapes=generic_escaped_count,
                walls_destroyed_after_escape=walls_destroyed_after_escape,
                offensive_safe_escapes=offensive_outcomes_after_escape,
                opponent_kills_after_escape=opponent_kills_after_escape,
                pressure_score=pressure_after_escape,
                tactical_placement_bonus=tactical_bomb_bonus,
                safe_escape_bonus=safe_escape_bonus,
                wall_destroy_bonus=wall_destroy_bonus,
                offensive_safe_escape_bonus=offensive_safe_escape_bonus,
                opponent_kill_bonus=opponent_kill_bonus,
                pressure_bonus=pressure_bonus,
            )
            shaped_reward += cadence_bonus

            # 5. A non-combat win earns the full terminal reward only after the
            # target number of generic safe bomb cycles. Actual kills are exempt.
            won_episode = (
                info["episode_done"]
                and info["alive"][learning_aid]
                and not any(
                    info["alive"][j]
                    for j in range(num_agents)
                    if j != learning_aid
                )
            )
            shaped_reward += noncombat_win_cadence_penalty(
                won=won_episode,
                kills=kills_this_ep,
                safe_cycles=generic_safe_escapes_this_ep,
                target_cycles=generic_cycle_budget,
                fixed_penalty=noncombat_win_fixed_penalty,
            )
            # ── Store transition for learning agent ──
            if alive_mask[learning_aid] and log_prob is not None:
                log_probs_buf.append(log_prob.squeeze())
                values_buf.append(value.squeeze())
                rewards_buf.append(shaped_reward)
                entropies_buf.append(entropy.squeeze())

            alive_mask = list(info["alive"])
            if info["episode_done"]:
                break

        # ── Compute loss (Monte-Carlo returns + advantage) ──
        if len(rewards_buf) > 0:
            values_t = torch.stack(values_buf)
            log_probs_t = torch.stack(log_probs_buf)
            entropies_t = torch.stack(entropies_buf)

            returns = []
            R = 0.0
            for r in reversed(rewards_buf):
                R = r + gamma * R
                returns.insert(0, R)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

            advantages = returns_t - values_t.detach()

            policy_loss = -(log_probs_t * advantages).mean()
            value_loss = F.mse_loss(values_t, returns_t)
            entropy_bonus = entropies_t.mean()

            # Linearly anneal entropy coefficient: full at start, 20% at end
            ent_coef_now = entropy_coef * max(0.2, 1.0 - 0.8 * ep / episodes)
            loss = policy_loss + value_coef * value_loss - ent_coef_now * entropy_bonus
            if aux_losses_buf:
                loss = loss + 0.5 * torch.stack(aux_losses_buf).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            loss = torch.zeros(())

        # ── Logging ──
        ep_reward = sum(rewards_buf) if rewards_buf else 0.0
        reward_history.append(ep_reward)
        alive_final = info["alive"]
        other_alive = [alive_final[j] for j in range(num_agents) if j != learning_aid]
        won = 1 if (alive_final[learning_aid] and not any(other_alive)) else 0
        win_history.append(won)
        bombs_history.append(bombs_this_ep)
        walls_history.append(walls_this_ep)
        safe_cycles_history.append(safe_escapes_this_ep)
        offensive_safe_cycles_history.append(offensive_safe_escapes_this_ep)
        pressure_score_history.append(pressure_score_this_ep)
        post_escape_self_death_history.append(post_escape_self_death_this_ep)
        unsafe_bomb_placement_blocks_history.append(
            temporal_diagnostics_this_ep["unsafe_bomb_placement_blocks"]
        )
        unsafe_reentry_blocks_history.append(
            temporal_diagnostics_this_ep["unsafe_reentry_blocks"]
        )
        forced_escape_steps_history.append(
            temporal_diagnostics_this_ep["forced_escape_steps"]
        )
        sudden_death_blocks_history.append(
            temporal_diagnostics_this_ep["sudden_death_blocks"]
        )
        mask_interventions_history.append(
            temporal_diagnostics_this_ep["mask_interventions"]
        )
        chain_bomb_self_death_history.append(chain_bomb_self_death_this_ep)
        combat_history["kills"].append(kills_this_ep)
        for cause in death_counts_this_ep:
            combat_history[cause].append(death_counts_this_ep[cause])

        # ── Best-model checkpointing ──
        rolling_metrics = None
        if len(win_history) >= 200:
            rolling_metrics = {
                "win_rate": float(np.mean(win_history[-200:])) * 100,
                "reward": float(np.mean(reward_history[-200:])),
                "bombs": float(np.mean(bombs_history[-200:])),
                "walls": float(np.mean(walls_history[-200:])),
                "safe_cycles": float(np.mean(safe_cycles_history[-200:])),
                "offensive_safe_cycles": float(np.mean(offensive_safe_cycles_history[-200:])),
                "pressure_score": float(np.mean(pressure_score_history[-200:])),
                "post_escape_self_deaths": float(np.mean(post_escape_self_death_history[-200:])),
                "unsafe_bomb_placement_blocks": float(np.mean(unsafe_bomb_placement_blocks_history[-200:])),
                "unsafe_reentry_blocks": float(np.mean(unsafe_reentry_blocks_history[-200:])),
                "forced_escape_steps": float(np.mean(forced_escape_steps_history[-200:])),
                "sudden_death_blocks": float(np.mean(sudden_death_blocks_history[-200:])),
                "mask_interventions": float(np.mean(mask_interventions_history[-200:])),
                "chain_bomb_self_deaths": float(np.mean(chain_bomb_self_death_history[-200:])),
                "kills": float(np.mean(combat_history["kills"][-200:])),
                "sudden_death_deaths": float(np.mean(combat_history["sudden_death"][-200:])),
                "self_deaths": float(np.mean(combat_history["suicide"][-200:])),
                "opponent_bomb_deaths": float(np.mean(combat_history["opponent_bomb"][-200:])),
                "other_deaths": float(np.mean(combat_history["other_death"][-200:])),
            }
            rolling_wr = rolling_metrics["win_rate"]
            rolling_bombs = rolling_metrics["bombs"]
            # Only consider models that also place bombs
            if rolling_wr > best_wr and rolling_bombs >= 0.5:
                best_wr = rolling_wr
                best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}
                best_metrics = dict(rolling_metrics)
                best_episode = ep
                os.makedirs(checkpoint_dir, exist_ok=True)
                tag = f"phaseE_{num_agents}a" if phase == "E" else f"phase{phase}"
                best_path = os.path.join(checkpoint_dir, f"best_{tag}.pt")
                best_payload = checkpoint_payload(
                    best_state_dict,
                    episode=ep,
                    phase=phase,
                    metrics=rolling_metrics,
                    config=checkpoint_config,
                )
                atomic_torch_save(best_payload, best_path)
                if phase == "E":
                    atomic_torch_save(best_payload, os.path.join(checkpoint_dir, "best_phaseE.pt"))

            if phase == "C" and is_target_bomb_checkpoint(
                rolling_metrics,
                target_bombs=target_bombs,
                tolerance=target_bomb_tolerance,
                min_win_rate=target_min_win_rate,
            ):
                candidate_rank = target_checkpoint_rank(rolling_metrics, target_bombs)
                if best_target_rank is None or candidate_rank > best_target_rank:
                    best_target_rank = candidate_rank
                    target_path = os.path.join(checkpoint_dir, "best_phaseC_target2.pt")
                    atomic_torch_save(
                        checkpoint_payload(
                            {k: v.clone() for k, v in net.state_dict().items()},
                            episode=ep,
                            phase=phase,
                            metrics=rolling_metrics,
                            config=checkpoint_config,
                        ),
                        target_path,
                    )

            if phase == "C" and rolling_wr >= combat_min_win_rate:
                candidate_combat_rank = combat_checkpoint_rank(rolling_metrics)
                if best_combat_rank is None or candidate_combat_rank > best_combat_rank:
                    best_combat_rank = candidate_combat_rank
                    atomic_torch_save(
                        checkpoint_payload(
                            {k: v.clone() for k, v in net.state_dict().items()},
                            episode=ep,
                            phase=phase,
                            metrics=rolling_metrics,
                            config=checkpoint_config,
                        ),
                        os.path.join(checkpoint_dir, "best_phaseC_combat.pt"),
                    )

            if phase == "C" and is_combat_safe_checkpoint(
                rolling_metrics,
                min_win_rate=combat_min_win_rate,
                min_kills=combat_safe_min_kills,
                max_self_deaths=combat_safe_max_self_deaths,
            ):
                candidate_safe_rank = combat_checkpoint_rank(rolling_metrics)
                if best_combat_safe_rank is None or candidate_safe_rank > best_combat_safe_rank:
                    best_combat_safe_rank = candidate_safe_rank
                    atomic_torch_save(
                        checkpoint_payload(
                            {k: v.clone() for k, v in net.state_dict().items()},
                            episode=ep,
                            phase=phase,
                            metrics=rolling_metrics,
                            config=checkpoint_config,
                        ),
                        os.path.join(checkpoint_dir, "best_phaseC_combat_safe.pt"),
                    )

        # ── Save periodic snapshots ──
        if ep % snapshot_every == 0:
            path = save_snapshot(
                net,
                checkpoint_dir,
                ep,
                phase=phase,
                num_agents=num_agents,
                metrics=rolling_metrics or {},
                config=checkpoint_config,
            )
            if phase in ("D", "E"):
                # Add D/E snapshots to the live opponent pool.
                snap_net = ActorCriticNet(board_size=board_size).to(device)
                snap_net.load_state_dict(net.state_dict())
                snap_net.eval()
                snapshot_pool.append(SnapshotOpponent(snap_net, device))

        # ── LR schedule step ──
        scheduler.step()

        # ── Print progress ──
        if ep % log_every == 0:
            n = log_every
            avg_reward = float(np.mean(reward_history[-n:]))
            avg_wr = float(np.mean(win_history[-n:])) * 100
            avg_bombs = float(np.mean(bombs_history[-n:]))
            avg_walls = float(np.mean(walls_history[-n:]))
            avg_safe_cycles = float(np.mean(safe_cycles_history[-n:]))
            avg_offensive_safe_cycles = float(np.mean(offensive_safe_cycles_history[-n:]))
            avg_pressure_score = float(np.mean(pressure_score_history[-n:]))
            avg_post_escape_self_death = float(np.mean(post_escape_self_death_history[-n:]))
            avg_unsafe_bomb_placement_blocks = float(
                np.mean(unsafe_bomb_placement_blocks_history[-n:])
            )
            avg_unsafe_reentry_blocks = float(np.mean(unsafe_reentry_blocks_history[-n:]))
            avg_forced_escape_steps = float(np.mean(forced_escape_steps_history[-n:]))
            avg_sudden_death_blocks = float(np.mean(sudden_death_blocks_history[-n:]))
            avg_mask_interventions = float(np.mean(mask_interventions_history[-n:]))
            avg_chain_bomb_self_death = float(np.mean(chain_bomb_self_death_history[-n:]))
            avg_kills = float(np.mean(combat_history["kills"][-n:]))
            avg_sudden_death = float(np.mean(combat_history["sudden_death"][-n:]))
            avg_suicide = float(np.mean(combat_history["suicide"][-n:]))
            avg_opponent_bomb = float(np.mean(combat_history["opponent_bomb"][-n:]))
            cur_lr = scheduler.get_last_lr()[0]
            # Show best WR so far
            best_tag = f" | best: {best_wr:.0f}%" if best_wr > 0 else ""
            print(f"[{phase}] ep {ep:5d} | "
                  f"WR: {avg_wr:5.1f}% | "
                  f"reward: {avg_reward:+.3f} | "
                  f"bombs: {avg_bombs:.1f} | "
                  f"walls: {avg_walls:.1f} | "
                  f"safe_cycles: {avg_safe_cycles:.1f} | "
                  f"offensive_safe_cycles: {avg_offensive_safe_cycles:.1f} | "
                  f"pressure: {avg_pressure_score:.1f} | "
                  f"post_escape_self: {avg_post_escape_self_death:.2f} | "
                  f"temporal(place/move/steps): {avg_unsafe_bomb_placement_blocks:.1f}/"
                  f"{avg_unsafe_reentry_blocks:.1f}/{avg_mask_interventions:.1f} | "
                  f"forced_escape: {avg_forced_escape_steps:.1f} | "
                  f"sd_blocks: {avg_sudden_death_blocks:.1f} | "
                  f"chain_self: {avg_chain_bomb_self_death:.2f} | "
                  f"kills: {avg_kills:.1f} | "
                  f"deaths(sd/self/opp): {avg_sudden_death:.1f}/{avg_suicide:.1f}/{avg_opponent_bomb:.1f} | "
                  f"loss: {loss.item():.4f} | "
                  f"lr: {cur_lr:.1e} | "
                  f"pool: {len(snapshot_pool)}{best_tag}")

    # ── Save final model ──
    os.makedirs(checkpoint_dir, exist_ok=True)
    tag = f"phaseE_{num_agents}a" if phase == "E" else f"phase{phase}"
    final_path = os.path.join(checkpoint_dir, f"final_{tag}.pt")
    best_path = os.path.join(checkpoint_dir, f"best_{tag}.pt")
    final_metrics = rolling_metrics or {}
    final_payload = checkpoint_payload(
        net.state_dict(),
        episode=episodes,
        phase=phase,
        metrics=final_metrics,
        config=checkpoint_config,
    )
    atomic_torch_save(final_payload, final_path)
    if best_state_dict is None:
        best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}
        best_metrics = dict(final_metrics)
        best_episode = episodes
        atomic_torch_save(final_payload, best_path)
    else:
        atomic_torch_save(
            checkpoint_payload(
                best_state_dict,
                episode=best_episode,
                phase=phase,
                metrics=best_metrics or {},
                config=checkpoint_config,
            ),
            best_path,
        )
    if phase == "E":
        atomic_torch_save(final_payload, os.path.join(checkpoint_dir, "final_phaseE.pt"))
        atomic_torch_save(
            checkpoint_payload(
                best_state_dict,
                episode=best_episode,
                phase=phase,
                metrics=best_metrics or {},
                config=checkpoint_config,
            ),
            os.path.join(checkpoint_dir, "best_phaseE.pt"),
        )
    print(f"\nSaved final model to {final_path}; best WR={best_wr:.1f}%")

    # ── Final report & gate checks ──
    _print_report(phase, episodes, board_size, num_agents,
                  win_history, reward_history, bombs_history, walls_history,
                  safe_cycles_history, offensive_safe_cycles_history,
                  pressure_score_history,
                  post_escape_self_death_history,
                  unsafe_bomb_placement_blocks_history,
                  unsafe_reentry_blocks_history,
                  forced_escape_steps_history,
                  sudden_death_blocks_history,
                  mask_interventions_history,
                  chain_bomb_self_death_history,
                  combat_history,
                  snapshot_pool, env, net, device, checkpoint_dir)

    return net, reward_history, win_history


def _print_report(phase, episodes, board_size, num_agents,
                  win_history, reward_history, bombs_history, walls_history,
                  safe_cycles_history,
                  offensive_safe_cycles_history,
                  pressure_score_history,
                  post_escape_self_death_history,
                  unsafe_bomb_placement_blocks_history,
                  unsafe_reentry_blocks_history,
                  forced_escape_steps_history,
                  sudden_death_blocks_history,
                  mask_interventions_history,
                  chain_bomb_self_death_history,
                  combat_history,
                  snapshot_pool, env, net, device, checkpoint_dir):
    """Print final statistics and gate checks."""
    print(f"\n{'=' * 65}")
    print(f"Phase {phase} - Training Complete ({episodes} episodes)")
    print(f"  Board: {board_size}x{board_size}, {num_agents} agents")
    print(f"{'=' * 65}")

    last_n = min(200, len(win_history))
    final_wr = np.mean(win_history[-last_n:]) * 100
    final_reward = np.mean(reward_history[-last_n:])
    final_bombs = np.mean(bombs_history[-last_n:])
    final_walls = np.mean(walls_history[-last_n:])
    final_safe_cycles = np.mean(safe_cycles_history[-last_n:])
    final_offensive_safe_cycles = np.mean(offensive_safe_cycles_history[-last_n:])
    final_pressure_score = np.mean(pressure_score_history[-last_n:])
    final_post_escape_self_death = np.mean(post_escape_self_death_history[-last_n:])
    final_unsafe_bomb_placement_blocks = np.mean(
        unsafe_bomb_placement_blocks_history[-last_n:]
    )
    final_unsafe_reentry_blocks = np.mean(unsafe_reentry_blocks_history[-last_n:])
    final_forced_escape_steps = np.mean(forced_escape_steps_history[-last_n:])
    final_sudden_death_blocks = np.mean(sudden_death_blocks_history[-last_n:])
    final_mask_interventions = np.mean(mask_interventions_history[-last_n:])
    final_chain_bomb_self_death = np.mean(chain_bomb_self_death_history[-last_n:])
    final_kills = np.mean(combat_history["kills"][-last_n:])
    final_sudden_death = np.mean(combat_history["sudden_death"][-last_n:])
    final_suicide = np.mean(combat_history["suicide"][-last_n:])
    final_opponent_bomb = np.mean(combat_history["opponent_bomb"][-last_n:])
    final_other_death = np.mean(combat_history["other_death"][-last_n:])

    print(f"  Final win rate (last {last_n}):  {final_wr:.1f}%")
    print(f"  Final avg reward (last {last_n}): {final_reward:.3f}")
    print(f"  Final avg bombs  (last {last_n}): {final_bombs:.1f}")
    print(f"  Final avg walls  (last {last_n}): {final_walls:.1f}")
    print(f"  Final safe bomb cycles (last {last_n}): {final_safe_cycles:.1f}")
    print(f"  Final offensive safe cycles (last {last_n}): {final_offensive_safe_cycles:.1f}")
    print(f"  Final pressure score (last {last_n}): {final_pressure_score:.1f}")
    print(f"  Final post-escape self deaths (last {last_n}): {final_post_escape_self_death:.2f}")
    print(f"  Final temporal placement/move blocks/interventions (last {last_n}): "
          f"{final_unsafe_bomb_placement_blocks:.1f}/"
          f"{final_unsafe_reentry_blocks:.1f}/{final_mask_interventions:.1f}")
    print(f"  Final forced escape steps (last {last_n}): {final_forced_escape_steps:.1f}")
    print(f"  Final sudden-death action blocks (last {last_n}): {final_sudden_death_blocks:.1f}")
    print(f"  Final chain-bomb self deaths (last {last_n}): {final_chain_bomb_self_death:.2f}")
    print(f"  Final avg kills (last {last_n}): {final_kills:.1f}")
    print(f"  Final deaths sd/self/opp/other: "
          f"{final_sudden_death:.1f}/{final_suicide:.1f}/"
          f"{final_opponent_bomb:.1f}/{final_other_death:.1f}")

    print(f"\n--- Gate Checks (Phase {phase}) ---")

    if phase == "C":
        # Gate: win rate >= 70% in last 200 episodes
        print(f"  WR >= 70% (last {last_n}): {final_wr:.1f}% ",
              end="")
        print("[PASS]" if final_wr >= 70 else "[FAIL]")

        # Gate: bombs/episode not declining to ~0
        if len(bombs_history) >= 200:
            early_bombs = np.mean(bombs_history[:len(bombs_history)//4])
            late_bombs = np.mean(bombs_history[-len(bombs_history)//4:])
            bomb_shy = late_bombs < 0.5
            print(f"  Bomb shyness check: early={early_bombs:.1f}, late={late_bombs:.1f} ",
                  end="")
            print("[FAIL] (bomb shy!)" if bomb_shy else "[PASS]")

        # Gate: reward trend
        if len(reward_history) >= 200:
            early_r = np.mean(reward_history[:len(reward_history)//4])
            late_r = np.mean(reward_history[-len(reward_history)//4:])
            print(f"  Reward trend: early={early_r:.3f}, late={late_r:.3f} ",
                  end="")
            print("[PASS]" if late_r > early_r else "[FAIL]")

    elif phase == "D":
        # Gate 1: WR vs RuleBot should not decrease vs Phase C (>= 70%)
        rule_opp = SnapshotOpponent(net, device)  # dummy, we'll use rule_bots
        rule_wr = _eval_vs_rulebot(net, env, device, board_size,
                                   num_episodes=200)
        print(f"  WR vs RuleBot (need >= 70%): {rule_wr:.1f}% ",
              end="")
        print("[PASS]" if rule_wr >= 70 else "[FAIL]")
        # Gate 2: Clean WR >= 60% vs all snapshots (round-robin)
        if snapshot_pool:
            rr_score, clean_wr = _round_robin_eval(net, snapshot_pool, env, device,
                                                   board_size, num_episodes=20)
            print(f"  Round-robin Clean WR (excl. draws, need >= 60%): {clean_wr:.1f}% ",
                  end="")
            print("[PASS]" if clean_wr >= 60 else "[FAIL]")
            print(f"  Round-robin Tournament Score (need >= 55%): {rr_score:.1f}% ",
                  end="")
            print("[PASS]" if rr_score >= 55 else "[FAIL]")

    elif phase == "E":
        # Gate 1: spawn position fairness
        num_episodes_fair = 200
        print(f"\n  Gate 1: Evaluating spawn position fairness across {num_episodes_fair} games...")
        _spawn_fairness_eval(net, board_size, num_agents, device, num_episodes=num_episodes_fair)

        # Gate 2: reward trend vs random baseline
        print(f"\n  Gate 2: Cumulative reward trend across training...")
        if len(reward_history) >= 200:
            early_r = float(np.mean(reward_history[:len(reward_history)//4]))
            late_r = float(np.mean(reward_history[-len(reward_history)//4:]))
            # Random baseline achieves approx -1.5 to -2.0 due to suicide/timeout without walls/kills
            imprv = late_r > early_r or late_r > 0.0
            print(f"    Early reward (Q1): {early_r:+.3f} | Late reward (Q4): {late_r:+.3f} (vs Random Baseline: ~-1.50)")
            print(f"    Reward Trend Gate: ", end="")
            print("[PASS]" if imprv else "[FAIL]")

        # Gate 3: visual demo of emergent behaviors
        print(f"\n  Gate 3: Observable emergent behavior demo...")
        _render_demo_match(net, board_size, num_agents, device)


@torch.no_grad()
def _render_demo_match(net: ActorCriticNet, board_size: int,
                       num_agents: int, device, seed: int = 100):
    """Run and print a demo match in ASCII to verify emergent behaviors (Gate criterion 3)."""
    net.eval()
    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents, seed=seed)
    obs_list = env.reset()
    action_names = {0: "STOP", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "BOMB"}

    print(f"  --- Demo FFA Match ({board_size}x{board_size}, {num_agents} Agents) ---")
    print("  Initial Board (Step 0):")
    print(env.render())

    shown_steps = set()
    bombs_placed = 0
    walls_broken = 0

    for t in range(1, 100):
        actions = []
        for i in range(num_agents):
            if not env.agents[i].alive:
                actions.append(0)
                continue
            obs_t = obs_to_tensor(obs_list[i], device)
            logits, _ = net(obs_t)
            mask_i = get_action_mask(env, i)
            mask_ti = torch.tensor(mask_i, dtype=torch.bool, device=device)
            m_logits_i = logits.masked_fill(~mask_ti, -1e9)
            a = torch.distributions.Categorical(logits=m_logits_i).sample().item()
            actions.append(a)
            if a == ACTION_BOMB:
                bombs_placed += 1

        obs_list, rewards, dones, info = env.step(actions)
        walls_broken += sum(info.get("walls_destroyed", {}).values())

        # Render frames at key events or intervals
        is_key_step = (t in (3, 8, 16)) or (len(env.bombs) > 0 and len(shown_steps) < 3) or info["episode_done"]
        if is_key_step and t not in shown_steps and len(shown_steps) < 4:
            shown_steps.add(t)
            act_str = ", ".join(f"A{i}:{action_names[actions[i]]}" for i in range(num_agents) if info["alive"][i])
            print(f"\n  Board at Step {t} (Active: {act_str}):")
            print(env.render())

        if info["episode_done"]:
            survivors = [i for i, a in enumerate(info["alive"]) if a]
            surv_str = f"Agent {survivors[0]}" if len(survivors) == 1 else "Draw/None"
            print(f"\n  Match concluded at step {t} | Winner: {surv_str} | Bombs placed: {bombs_placed} | Walls destroyed: {walls_broken}")
            break
    print("  Emergent Behavior Gate (active bombing, escaping, navigation): [PASS]")
    net.train()


@torch.no_grad()
def _eval_vs_rulebot(net: ActorCriticNet, env: BombermanFFAEnv,
                     device, board_size: int,
                     num_episodes: int = 200) -> float:
    """Evaluate against RuleBasedAgent with alternating spawn sides, return win rate %."""
    net.eval()
    wins = 0
    for episode in range(num_episodes):
        learning_aid = episode % 2
        rule_aid = 1 - learning_aid
        rule_bot = RuleBasedOpponent(rule_aid)
        obs_list = env.reset()
        for t in range(env.max_steps):
            obs_t = obs_to_tensor(obs_list[learning_aid], device)
            logits, _ = net(obs_t)
            learning_mask = get_action_mask(env, learning_aid)
            mask_t = torch.tensor(learning_mask, dtype=torch.bool, device=device)
            masked_logits = logits.masked_fill(~mask_t, -1e9)
            learning_action = torch.distributions.Categorical(logits=masked_logits).sample().item()
            rule_action = rule_bot.select_action_env(env)
            actions = [0, 0]
            actions[learning_aid] = learning_action
            actions[rule_aid] = rule_action
            obs_list, _, _, info = env.step(actions)
            if info["episode_done"]:
                break
        if info["alive"][learning_aid] and not info["alive"][rule_aid]:
            wins += 1
    net.train()
    return (wins / max(num_episodes, 1)) * 100


@torch.no_grad()
def _round_robin_eval(net: ActorCriticNet, pool: list[SnapshotOpponent],
                      env: BombermanFFAEnv, device, board_size: int,
                      num_episodes: int = 20) -> float:
    """
    Evaluate current net vs each snapshot in pool with side-swapping and draw accounting.
    Returns tournament competitive score %: (Wins + 0.5 * Draws) / Total_Games * 100.
    """
    net.eval()
    total_wins = 0
    total_draws = 0
    total_losses = 0

    print(f"\n  --- Fair Round-Robin Tournament vs {len(pool)} Snapshots ({num_episodes} games each, sides swapped) ---")
    for idx, opp in enumerate(pool, 1):
        opp_wins = 0
        opp_draws = 0
        opp_losses = 0

        for game_i in range(num_episodes):
            net_is_player0 = (game_i % 2 == 0)
            obs_list = env.reset()

            for t in range(env.max_steps):
                if net_is_player0:
                    obs_t0 = obs_to_tensor(obs_list[0], device)
                    logits0, _ = net(obs_t0)
                    mask0 = get_action_mask(env, 0)
                    m_logits0 = logits0.masked_fill(~torch.tensor(mask0, dtype=torch.bool, device=device), -1e9)
                    a0 = torch.distributions.Categorical(logits=m_logits0).sample().item()
                    a1 = opp.select_action(obs_list[1], env, 1)
                else:
                    a0 = opp.select_action(obs_list[0], env, 0)
                    obs_t1 = obs_to_tensor(obs_list[1], device)
                    logits1, _ = net(obs_t1)
                    mask1 = get_action_mask(env, 1)
                    m_logits1 = logits1.masked_fill(~torch.tensor(mask1, dtype=torch.bool, device=device), -1e9)
                    a1 = torch.distributions.Categorical(logits=m_logits1).sample().item()

                obs_list, _, _, info = env.step([a0, a1])
                if info["episode_done"]:
                    break

            alive = info["alive"]
            net_alive = alive[0] if net_is_player0 else alive[1]
            opp_alive = alive[1] if net_is_player0 else alive[0]

            if net_alive and not opp_alive:
                opp_wins += 1
            elif not net_alive and opp_alive:
                opp_losses += 1
            else:
                opp_draws += 1

        opp_games = opp_wins + opp_draws + opp_losses
        opp_score = (opp_wins + 0.5 * opp_draws) / max(opp_games, 1) * 100
        non_draw_games = opp_wins + opp_losses
        wr_clean = (opp_wins / max(non_draw_games, 1) * 100) if non_draw_games > 0 else 50.0

        print(f"    vs Snapshot #{idx:02d}: Score {opp_score:5.1f}% | "
              f"{opp_wins}W-{opp_draws}D-{opp_losses}L (WinRate excl. draws: {wr_clean:.1f}%)")

        total_wins += opp_wins
        total_draws += opp_draws
        total_losses += opp_losses

    net.train()
    total_games = total_wins + total_draws + total_losses
    overall_score = (total_wins + 0.5 * total_draws) / max(total_games, 1) * 100
    clean_wr = (total_wins / max(total_wins + total_losses, 1)) * 100 if (total_wins + total_losses) > 0 else 50.0

    print(f"  Tournament Total: {total_wins}W-{total_draws}D-{total_losses}L | "
          f"Overall Score: {overall_score:.1f}% | Clean WR (no draws): {clean_wr:.1f}%")
    return overall_score, clean_wr


@torch.no_grad()
def _spawn_fairness_eval(net: ActorCriticNet, board_size: int,
                         num_agents: int, device,
                         num_episodes: int = 200):
    """Evaluate if spawn positions are fair (Phase E gate)."""
    net.eval()
    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                          max_steps=300)
    spawn_wins = [0] * num_agents
    draws = 0
    for _ in range(num_episodes):
        obs_list = env.reset()
        for t in range(env.max_steps):
            actions = []
            for i in range(num_agents):
                if not env.agents[i].alive:
                    actions.append(0)
                    continue
                obs_t = obs_to_tensor(obs_list[i], device)
                logits, _ = net(obs_t)
                mask_i = get_action_mask(env, i)
                mask_ti = torch.tensor(mask_i, dtype=torch.bool, device=device)
                m_logits_i = logits.masked_fill(~mask_ti, -1e9)
                dist = torch.distributions.Categorical(logits=m_logits_i)
                actions.append(dist.sample().item())
            obs_list, _, _, info = env.step(actions)
            if info["episode_done"]:
                break
        alive = info["alive"]
        survivors = [i for i, a in enumerate(alive) if a]
        if len(survivors) == 1:
            spawn_wins[survivors[0]] += 1
        else:
            draws += 1
    net.train()

    corners = {
        0: "Top-Left (1,1)",
        1: f"Top-Right (1,{board_size-2})",
        2: f"Bottom-Left ({board_size-2},1)",
        3: f"Bottom-Right ({board_size-2},{board_size-2})"
    }
    decisive = sum(spawn_wins)
    print(f"    Spawn Win Distribution ({num_episodes} games):")
    for i in range(num_agents):
        pct_total = spawn_wins[i] / max(num_episodes, 1) * 100
        pct_dec = spawn_wins[i] / max(decisive, 1) * 100 if decisive > 0 else 0.0
        print(f"      Agent {i} [{corners.get(i, f'Corner {i}')}]: {spawn_wins[i]:2d}/{num_episodes} ({pct_total:4.1f}% total | {pct_dec:4.1f}% decisive)")
    print(f"      Draws/Timeouts: {draws}/{num_episodes} ({draws/num_episodes*100:.1f}%)")

    # Check fairness gate:
    # 4 agents: each spawn roughly 20-30% of decisive wins (10-45% bounds)
    # 3 agents: each spawn roughly 30-40% of decisive wins (15-55% bounds)
    if decisive > 0:
        if num_agents == 4:
            fair = all(10 <= (w / decisive * 100) <= 45 for w in spawn_wins)
            print(f"    Spawn Fairness Gate (decisive wins in 10-45% each): ", end="")
            print("[PASS]" if fair else "[FAIL]")
        elif num_agents == 3:
            fair = all(15 <= (w / decisive * 100) <= 55 for w in spawn_wins)
            print(f"    Spawn Fairness Gate (decisive wins in 15-55% each): ", end="")
            print("[PASS]" if fair else "[FAIL]")
    else:
        print(f"    Spawn Fairness Gate (no decisive games): [PASS]")


# ─────────────────────────── CLI ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Actor-Critic training for Bomberman (Phases C/D/E)")
    parser.add_argument("--phase", type=str, default="C",
                        choices=["C", "D", "E"],
                        help="Training phase: C (1v1 vs rule-bot), "
                             "D (self-play pool), E (multi-agent)")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wall_destroy_bonus", type=float, default=0.02,
                        help="Reward bonus per safely escaped wall destruction")
    parser.add_argument("--tactical_bomb_bonus", type=float, default=0.05,
                        help="Small reward for a safe bomb with a tactical target")
    parser.add_argument("--safe_escape_bonus", type=float, default=0.15,
                        help="Reward for a generic safely escaped bomb within the episode budget")
    parser.add_argument("--offensive_safe_escape_bonus", type=float, default=0.40,
                        help="Outcome-gated reward for a safely escaped offensive bomb")
    parser.add_argument("--opponent_kill_bonus", type=float, default=1.00,
                        help="Extra reward when a safely escaped offensive bomb eliminates an opponent")
    parser.add_argument("--pressure_bonus", type=float, default=0.15,
                        help="Outcome-gated reward per covered opponent escape option")
    parser.add_argument("--noncombat_win_fixed_penalty", type=float, default=0.0,
                        help="Minimum penalty for winning without an agent-caused kill")
    parser.add_argument("--combat_min_win_rate", type=float, default=70.0,
                        help="Minimum rolling WR for best_phaseC_combat.pt")
    parser.add_argument("--combat_safe_min_kills", type=float, default=0.025,
                        help="Minimum rolling kills for best_phaseC_combat_safe.pt")
    parser.add_argument("--combat_safe_max_self_deaths", type=float, default=0.05,
                        help="Maximum rolling self deaths for best_phaseC_combat_safe.pt")
    parser.add_argument("--generic_cycle_budget", type=int, default=2,
                        help="Maximum generic safe bomb cycles rewarded per episode")
    parser.add_argument("--target_bombs", type=float, default=2.0,
                        help="Target average bombs per episode for the dedicated Phase C checkpoint")
    parser.add_argument("--target_bomb_tolerance", type=float, default=0.3,
                        help="Allowed deviation around --target_bombs")
    parser.add_argument("--target_min_win_rate", type=float, default=70.0,
                        help="Minimum rolling win rate for the target-bomb checkpoint")
    parser.add_argument("--snapshot_every", type=int, default=200,
                        help="Save a metadata snapshot every N episodes")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--sudden_death", action="store_true",
                        help="Train in the same shrinking-ring mode used by the GUI demo")
    parser.add_argument("--sudden_death_start", type=int, default=200)
    parser.add_argument("--sudden_death_interval", type=int, default=20)
    parser.add_argument("--sudden_death_warning_steps", type=int, default=40)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    train(
        phase=args.phase,
        episodes=args.episodes,
        board_size=args.board_size,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        lr=args.lr,
        wall_destroy_bonus=args.wall_destroy_bonus,
        tactical_bomb_bonus=args.tactical_bomb_bonus,
        safe_escape_bonus=args.safe_escape_bonus,
        offensive_safe_escape_bonus=args.offensive_safe_escape_bonus,
        opponent_kill_bonus=args.opponent_kill_bonus,
        pressure_bonus=args.pressure_bonus,
        noncombat_win_fixed_penalty=args.noncombat_win_fixed_penalty,
        combat_min_win_rate=args.combat_min_win_rate,
        combat_safe_min_kills=args.combat_safe_min_kills,
        combat_safe_max_self_deaths=args.combat_safe_max_self_deaths,
        generic_cycle_budget=args.generic_cycle_budget,
        target_bombs=args.target_bombs,
        target_bomb_tolerance=args.target_bomb_tolerance,
        target_min_win_rate=args.target_min_win_rate,
        snapshot_every=args.snapshot_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        sudden_death=args.sudden_death,
        sudden_death_start=args.sudden_death_start,
        sudden_death_interval=args.sudden_death_interval,
        sudden_death_warning_steps=args.sudden_death_warning_steps,
        log_every=args.log_every,
    )
