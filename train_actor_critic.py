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
import json
from collections import deque
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bomberman_env import BombermanFFAEnv, NUM_ACTIONS, OBS_CHANNELS, ACTION_BOMB, ACTION_STOP, WOODEN_WALL, _MOVE_DELTA
from rule_based_bot import (
    RuleBasedAgent,
    _build_danger_map,
    _safe_action_mask,
    _bfs_safe_cell,
    _bfs_toward_opponent,
    _bfs_toward_wood,
    _bfs_toward_item,
    _bomb_escape_pressure,
    _can_bomb_and_escape,
    _survival_map,
)
from temporal_trap_planner import (
    TemporalTrapConfig,
    TemporalTrapPlan,
    find_interception_action,
    find_temporal_trap_plan,
)


# --------------------------------- network ---------------------------------

class ActorCriticNet(nn.Module):
    """
    Small CNN shared trunk -> actor head (action logits) + critic head (state value).
    Input: (batch, OBS_CHANNELS, H, W) from BombermanFFAEnv._get_obs.
    """

    def __init__(self, in_channels: int = OBS_CHANNELS, board_size: int = 11,
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


def get_action_mask(env: BombermanFFAEnv, agent_id: int) -> list[bool]:
    """Shared timed survival filter for training, evaluation, GUI and RuleBots."""
    return _safe_action_mask(env, agent_id)


def _pressure_attack_action(env: BombermanFFAEnv, agent_id: int, danger, mask,
                            attack_lookahead: int, min_pressure_gain: float):
    agent = env.agents[agent_id]
    start = (agent.row, agent.col)
    base = env._forecast_blasts(min_horizon=env.bomb_timer + env.flame_duration - 1)
    best_pressure = min_pressure_gain
    best_action = None

    if mask[ACTION_BOMB]:
        pressure = _bomb_escape_pressure(env, agent_id, *start, base)
        if pressure >= best_pressure:
            best_pressure = pressure
            best_action = ACTION_BOMB

    occupied = {(a.row, a.col) for a in env.agents if a.alive and a.agent_id != agent_id}
    bomb_positions = {(bomb.row, bomb.col) for bomb in env.bombs}
    visited = {start}
    queue = deque()
    for action, (dr, dc) in _MOVE_DELTA.items():
        position = (start[0] + dr, start[1] + dc)
        if not mask[action] or position in occupied or danger[position]:
            continue
        visited.add(position)
        queue.append((position, 1, action))

    while queue:
        (row, col), depth, first_action = queue.popleft()
        pressure = _bomb_escape_pressure(env, agent_id, row, col, base)
        if pressure >= min_pressure_gain and pressure > best_pressure + 1e-6:
            best_pressure = pressure
            best_action = first_action
        elif pressure >= min_pressure_gain and best_action is None:
            best_pressure = pressure
            best_action = first_action
        if depth >= attack_lookahead:
            continue
        for dr, dc in _MOVE_DELTA.values():
            position = (row + dr, col + dc)
            if position in visited or position in occupied or position in bomb_positions:
                continue
            if not env._is_walkable(*position) or danger[position]:
                continue
            visited.add(position)
            queue.append((position, depth + 1, first_action))
    return best_action


def _sequence_attack_action(env: BombermanFFAEnv, agent_id: int, danger, mask,
                            attack_lookahead: int, trap_sequence_depth: int,
                            sequence_pressure_gain: float):
    """Label a safe opening bomb when a distinct follow-up completes the trap."""
    agent = env.agents[agent_id]
    if (trap_sequence_depth < 2 or not mask[ACTION_BOMB]
            or agent.ammo - agent.bombs_placed < 2):
        return None

    start = (agent.row, agent.col)
    base = env._forecast_blasts(min_horizon=env.bomb_timer + env.flame_duration - 1)
    first_pressure = _bomb_escape_pressure(env, agent_id, *start, base)
    if first_pressure <= 0.0:
        return None

    occupied = {(a.row, a.col) for a in env.agents
                if a.alive and a.agent_id != agent_id}
    bomb_positions = {(bomb.row, bomb.col) for bomb in env.bombs}
    visited = {start}
    queue = deque()
    for action, (dr, dc) in _MOVE_DELTA.items():
        position = (start[0] + dr, start[1] + dc)
        if (not mask[action] or position in occupied or position in bomb_positions
                or danger[position] or not env._is_walkable(*position)):
            continue
        visited.add(position)
        queue.append((position, 1))

    while queue:
        (row, col), depth = queue.popleft()
        followup_pressure = _bomb_escape_pressure(env, agent_id, row, col, base)
        if (followup_pressure > 0.0
                and first_pressure + followup_pressure + 1e-9 >= sequence_pressure_gain
                and _can_bomb_and_escape(env, agent_id, row, col, danger)):
            return ACTION_BOMB
        if depth >= attack_lookahead:
            continue
        for dr, dc in _MOVE_DELTA.values():
            position = (row + dr, col + dc)
            if (position in visited or position in occupied or position in bomb_positions
                    or danger[position] or not env._is_walkable(*position)):
                continue
            visited.add(position)
            queue.append((position, depth + 1))
    return None


class TemporalPlanFollower:
    """Pending trainer labels; policy samples remain in control of the agent."""

    def __init__(self, plan: TemporalTrapPlan, position: tuple[int, int],
                 diagnostics: dict):
        self.plan = plan
        self.next_index = 0
        self.expected_position = position
        self.observed_bombs = 0
        self.abort_reason = None
        self.diagnostics = diagnostics

    @property
    def complete(self) -> bool:
        return self.next_index == len(self.plan.action_sequence)

    def expected_action(self) -> int:
        return self.plan.action_sequence[self.next_index]

    def advance(self, sampled_action: int) -> bool:
        if self.abort_reason is not None or self.complete:
            return False
        if sampled_action != self.expected_action():
            self.abort("mismatch")
            return False
        if sampled_action in _MOVE_DELTA:
            dr, dc = _MOVE_DELTA[sampled_action]
            self.expected_position = (self.expected_position[0] + dr,
                                      self.expected_position[1] + dc)
        self.next_index += 1
        return True

    def abort(self, reason: str) -> None:
        if self.abort_reason is None and not self.complete:
            self.abort_reason = reason
            key = "temporal_plan_aborts_" + reason
            self.diagnostics[key] = self.diagnostics.get(key, 0) + 1


def _select_temporal_label(env, agent_id, danger, mask, follower, config,
                           diagnostics) -> tuple[int | None, TemporalPlanFollower | None]:
    """Retain only a legal, safe route at its observed expected position."""
    agent = env.agents[agent_id]
    position = (agent.row, agent.col)
    reason = None
    if not agent.alive:
        reason = "inactive"
    elif position in env.flames or danger[position] == 1:
        reason = "danger"
    elif _ring_safety_action(env, agent_id, danger, mask) is not None:
        reason = "ring"
    if reason is not None:
        if follower is not None:
            follower.abort(reason)
        return None, None

    if follower is not None:
        if follower.abort_reason is not None or follower.complete:
            return None, None
        if position != follower.expected_position:
            follower.abort("stale_position")
            return None, None
    else:
        # Preserve the existing non-urgent bomb/escape teacher without a route.
        if danger[position] > 0:
            return None, None
        plan = find_temporal_trap_plan(
            env, agent_id, danger, mask, config, diagnostics=diagnostics,
        )
        if plan is None:
            return None, None
        follower = TemporalPlanFollower(plan, position, diagnostics)
        diagnostics["temporal_trap_plans"] = diagnostics.get("temporal_trap_plans", 0) + 1
        key = "temporal_pair_plans_started"
        diagnostics[key] = diagnostics.get(key, 0) + int(len(plan.bomb_positions) == 2)

    action = follower.expected_action()
    if not 0 <= action < len(mask) or not mask[action]:
        follower.abort("mask")
        return None, None
    dr, dc = _MOVE_DELTA.get(action, (0, 0))
    destination = (position[0] + dr, position[1] + dc)
    if destination in env.flames or danger[destination] == 1:
        follower.abort("danger")
        return None, None
    # The shared mask may fall back to immediate survival in a doomed state.
    # A temporal teacher needs an actual continuation through the live forecast.
    if action == ACTION_BOMB:
        safe_step = _can_bomb_and_escape(env, agent_id, *position, danger)
    else:
        safe, blocked, _ = _survival_map(env, *position, safety_margin=0)
        safe_step = bool(safe[0, destination[0], destination[1]])
        if action in _MOVE_DELTA:
            safe_step &= not blocked[0, destination[0], destination[1]]
    if not safe_step:
        follower.abort("unsafe")
        return None, None
    diagnostics["source"] = "temporal"
    diagnostics["temporal_trap_labels"] = diagnostics.get("temporal_trap_labels", 0) + 1
    return action, follower


def _observe_temporal_transition(env, agent_id, sampled_action, info, follower):
    """Count placements only after a matching sample and actual transition."""
    if follower is None:
        return None
    expected = follower.expected_action()
    if sampled_action != expected:
        follower.abort("mismatch")
        return None
    diagnostics = follower.diagnostics
    key = "temporal_policy_action_matches"
    diagnostics[key] = diagnostics.get(key, 0) + 1
    if expected == ACTION_BOMB and agent_id not in info["bombs_placed"]:
        follower.abort("placement")
        return None
    dr, dc = _MOVE_DELTA.get(expected, (0, 0))
    position = (follower.expected_position[0] + dr, follower.expected_position[1] + dc)
    agent = env.agents[agent_id]
    if (agent.row, agent.col) != position:
        follower.abort("stale_position")
        return None
    if not follower.advance(sampled_action):
        return None
    if expected == ACTION_BOMB:
        follower.observed_bombs += 1
        diagnostics = follower.diagnostics
        diagnostics["temporal_trap_bombs"] = diagnostics.get("temporal_trap_bombs", 0) + 1
    if follower.complete:
        if len(follower.plan.bomb_positions) == 2 and follower.observed_bombs == 2:
            diagnostics = follower.diagnostics
            key = "temporal_trap_pair_completions"
            diagnostics[key] = diagnostics.get(key, 0) + 1
        return None
    return follower


def _temporal_metrics(diagnostics: dict) -> dict:
    """Publish episode or summed counters with rates derived from their totals."""
    metrics = {key: value for key, value in diagnostics.items() if key != "source"}
    aliases = {
        "temporal_plans_started": "temporal_trap_plans",
        "temporal_steps_supervised": "temporal_trap_labels",
        "temporal_bomb_actions_executed": "temporal_trap_bombs",
        "temporal_pair_completions": "temporal_trap_pair_completions",
    }
    for canonical, alias in aliases.items():
        metrics[canonical] = diagnostics.get(alias, 0)
    for key in ("temporal_pair_plans_started", "temporal_policy_action_matches"):
        metrics[key] = diagnostics.get(key, 0)
    steps = metrics["temporal_steps_supervised"]
    metrics["temporal_policy_agreement_rate"] = (
        metrics["temporal_policy_action_matches"] / steps if steps else 0.0
    )
    groups = {
        "safety": ("danger", "ring", "unsafe"),
        "stale": ("stale_position", "placement", "inactive"),
        "masked": ("mask",),
        "mismatch": ("mismatch",),
    }
    total_aborts = 0
    for group, reasons in groups.items():
        count = sum(diagnostics.get("temporal_plan_aborts_" + reason, 0)
                    for reason in reasons)
        metrics["temporal_plan_aborts_" + group] = count
        total_aborts += count
    metrics["temporal_plan_aborts"] = total_aborts
    plans = metrics["temporal_plans_started"]
    for key in ("predicted_refuge_reduction", "predicted_trap_completion_ticks"):
        metrics[key] = diagnostics.get(key, 0.0) / plans if plans else 0.0
    return metrics


def get_attack_action(env: BombermanFFAEnv, agent_id: int, danger, mask,
                      attack_lookahead: int = 2, min_pressure_gain: float = 0.15,
                      trap_sequence_depth: int = 2,
                      sequence_pressure_gain: float = 0.35,
                      temporal_trap_horizon: int = 8,
                      temporal_trap_max_bombs: int = 2,
                      interception_horizon: int = 4,
                      diagnostics: dict | None = None,
                      use_temporal_planner: bool = True):
    """Prefer bomb positions that restrict timed escape, then upgrades or opening routes."""
    agent = env.agents[agent_id]
    r, c = agent.row, agent.col
    if (r, c) in env.flames:
        return None
    if danger[r, c] == 1:
        escape = _bfs_safe_cell(env, r, c, danger, action_mask=mask)
        return escape if escape is not None and mask[escape] else None
    ring_action = _ring_safety_action(env, agent_id, danger, mask)
    if ring_action is not None:
        return ring_action
    temporal_plan = find_temporal_trap_plan(
        env,
        agent_id,
        danger,
        mask,
        TemporalTrapConfig(
            horizon=temporal_trap_horizon,
            max_bombs=temporal_trap_max_bombs,
            min_pressure_gain=sequence_pressure_gain,
        ),
        diagnostics=diagnostics,
    ) if use_temporal_planner else None
    if temporal_plan is not None:
        if diagnostics is not None:
            diagnostics["source"] = "temporal"
            diagnostics["temporal_trap_labels"] = diagnostics.get(
                "temporal_trap_labels", 0
            ) + 1
        return temporal_plan.first_action
    interception = find_interception_action(
        env, agent_id, danger, mask, horizon=interception_horizon
    )
    if interception is not None:
        if diagnostics is not None:
            diagnostics["source"] = "interception"
            diagnostics["interception_labels"] = diagnostics.get(
                "interception_labels", 0
            ) + 1
        return interception
    close = any(a.alive and a.agent_id != agent_id and
                abs(a.row - r) + abs(a.col - c) <= agent.blast_range + 1
                for a in env.agents)
    if close:
        sequence_action = _sequence_attack_action(
            env, agent_id, danger, mask, attack_lookahead,
            trap_sequence_depth, sequence_pressure_gain)
        if sequence_action is not None:
            return sequence_action
        pressure_action = _pressure_attack_action(
            env, agent_id, danger, mask, attack_lookahead, min_pressure_gain)
        if pressure_action is not None:
            return pressure_action
    blast = env._blast_cells(r, c, agent.blast_range)
    if mask[ACTION_BOMB] and any(
        a.alive and a.agent_id != agent_id and (a.row, a.col) in blast
        for a in env.agents
    ):
        return ACTION_BOMB
    move = _bfs_toward_item(env, agent_id, danger)
    if move is not None and mask[move]:
        return move
    move = _bfs_toward_opponent(env, agent_id, r, c, danger, action_mask=mask)
    if move is not None and mask[move]:
        return move
    if mask[ACTION_BOMB] and any(env.grid[pos] == WOODEN_WALL for pos in blast):
        return ACTION_BOMB
    move = _bfs_toward_wood(env, r, c, danger)
    return move if move is not None and mask[move] else None


def _ring_safety_action(env: BombermanFFAEnv, agent_id: int, danger, mask):
    """Move off the next closing ring during its warning window."""
    next_closure = env._next_sudden_death_closure()
    if next_closure is None:
        return None
    remaining, closing_cells = next_closure
    agent = env.agents[agent_id]
    if (remaining < 1 or remaining > env.sudden_death_warning_steps
            or (agent.row, agent.col) not in closing_cells):
        return None

    occupied = {(a.row, a.col) for a in env.agents
                if a.alive and a.agent_id != agent_id}
    visited = {(agent.row, agent.col)}
    queue = deque()
    for action, (dr, dc) in _MOVE_DELTA.items():
        position = (agent.row + dr, agent.col + dc)
        if (not mask[action] or position in occupied or danger[position]
                or not env._is_walkable(*position)):
            continue
        visited.add(position)
        queue.append((position, 1, action))

    while queue:
        position, depth, first_action = queue.popleft()
        if position not in closing_cells:
            return first_action
        if depth >= remaining:
            continue
        for dr, dc in _MOVE_DELTA.values():
            next_position = (position[0] + dr, position[1] + dc)
            if (next_position in visited or next_position in occupied
                    or danger[next_position] or not env._is_walkable(*next_position)):
                continue
            visited.add(next_position)
            queue.append((next_position, depth + 1, first_action))
    return None


def _supervision_losses(escape, combat, navigation):
    """Separate means keep rare combat labels from being diluted by navigation."""
    return tuple(torch.stack(group).mean() if group else 0.0
                 for group in (escape, combat, navigation))


def _combat_reward(reward: float, agent_id: int, info, episode_kills: int):
    """Reward owned eliminations; a token bomb cannot unlock a passive win bonus."""
    kills = info["kills"]
    if agent_id in info.get("self_blast_deaths", []) or (agent_id, agent_id) in kills:
        reward -= 1.5  # -1 from env, -2.5 total for suicide.
    if info["alive"][agent_id]:
        reward += 3.0 * sum(k == agent_id and v != agent_id for k, v in kills)
        if (info["episode_done"] and sum(info["alive"]) == 1 and episode_kills == 0
                and not info.get("sudden_death_deaths")):
            reward -= 10.0
    return reward


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
    """Preserve old policies: zero new input weights; interpolate different board sizes."""
    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(sd, dict) or not all(isinstance(k, str) and torch.is_tensor(v)
                                          for k, v in sd.items()):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    curr_sd = net.state_dict()
    missing = set(curr_sd) - set(sd)
    extra = set(sd) - set(curr_sd)
    if missing or extra:
        raise ValueError(f"Checkpoint keys do not match model (missing={sorted(missing)}, extra={sorted(extra)})")
    for k, v in sd.items():
        if v.shape == curr_sd[k].shape:
            curr_sd[k].copy_(v)
        elif (k == "conv.0.weight" and v.ndim == curr_sd[k].ndim == 4
              and v.shape[0] == curr_sd[k].shape[0]
              and v.shape[2:] == curr_sd[k].shape[2:]
              and v.shape[1] in (8, 12)
              and v.shape[1] < curr_sd[k].shape[1]):
            curr_sd[k].zero_()
            curr_sd[k][:, :v.shape[1]].copy_(v)
            print(f"Expanded observation channels {v.shape[1]} -> {curr_sd[k].shape[1]}")
        elif k == "trunk_fc.0.weight" and v.ndim == curr_sd[k].ndim == 2:
            old_flat = v.shape[1]
            new_flat = curr_sd[k].shape[1]
            old_s = int(np.round(np.sqrt(old_flat / 64)))
            new_s = int(np.round(np.sqrt(new_flat / 64)))
            if v.shape[0] != curr_sd[k].shape[0] or old_s * old_s * 64 != old_flat:
                raise ValueError(f"Unsupported shape for {k}: {tuple(v.shape)} -> {tuple(curr_sd[k].shape)}")
            w_4d = v.view(256, 64, old_s, old_s)
            w_interp = F.interpolate(w_4d, size=(new_s, new_s), mode="bilinear", align_corners=False)
            w_interp = w_interp * ((old_s * old_s) / (new_s * new_s))
            curr_sd[k].copy_(w_interp.reshape(256, new_flat))
            print(f"Interpolated {k} from board {old_s}x{old_s} to {new_s}x{new_s}")
        else:
            raise ValueError(f"Unsupported shape for {k}: {tuple(v.shape)} -> {tuple(curr_sd[k].shape)}")
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
                  phase: str = "", num_agents: int = 2):
    """Save a snapshot of the network weights."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if phase == "E":
        prefix = f"snapshot_phaseE_{num_agents}a_"
    else:
        prefix = "snapshot_"
    path = os.path.join(checkpoint_dir, f"{prefix}ep{episode:06d}.pt")
    torch.save(net.state_dict(), path)
    return path


# ─────────────────────── main training function ───────────────────────────

def train(
    phase: str = "C",
    episodes: int = 750,
    board_size: int = 9,
    num_agents: int = 2,
    gamma: float = 0.99,
    lr: float = 0.000025,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0075,
    wall_destroy_bonus: float = 0.15,
    max_steps: int = 500,
    log_every: int = 50,
    snapshot_every: int = 200,
    checkpoint_dir: str = "checkpoints",
    resume_from: str = "",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    on_step=None,
    on_episode=None,
    stop_event=None,
    run_report: bool = True,
    seed: int | None = None,
    validation_every: int = 250,
    validation_episodes: int = 48,
    validation_seed: int = 6000,
    on_validation=None,
    sudden_death_probability: float = 0.0,
    sudden_death_start: int = 350,
    sudden_death_interval: int = 20,
    sudden_death_warning_steps: int = 40,
    attack_lookahead: int = 2,
    min_pressure_gain: float = 0.15,
    trap_sequence_depth: int = 2,
    sequence_pressure_gain: float = 0.35,
    temporal_trap_horizon: int = 8,
    temporal_trap_max_bombs: int = 2,
    interception_horizon: int = 4,
    ring_safety_coef: float = 0.6,
    late_game_start: int = 350,
    late_game_attack_multiplier: float = 1.75,
    escape_loss_coef: float = 0.5,
    combat_loss_coef: float = 0.9,
    navigation_loss_coef: float = 0.4,
    target_win_rate: float | None = None,
    target_active_win_rate: float | None = None,
    target_draw_rate: float | None = None,
    target_sudden_death_death_rate: float | None = None,
    max_suicide_rate: float | None = None,
):
    """
    Unified training loop for phases C, D, E.

    Phase C: Rotate the learning corner against RuleBasedAgent.
             Reward shaping: +wall_destroy_bonus per wall destroyed.
    Phase D: Rotate the learning corner against a random opponent from the pool
             (includes RuleBasedAgent + saved snapshots).
    Phase E: Same as D with num_agents=2, 3 or 4 and rotating learning corners.

    Optional GUI hooks: on_step(env, episode, learning_id, actions, opponents)
    and on_episode(metrics). A set stop_event discards the partial episode,
    saves completed updates and returns. run_report=False skips final tournaments.
    """
    phase = phase.upper()
    assert phase in ("C", "D", "E"), f"Phase must be C, D, or E, got {phase}"
    if episodes < 1 or max_steps < 1 or log_every < 1 or snapshot_every < 1:
        raise ValueError("episodes, max_steps, log_every and snapshot_every must be positive")
    if validation_every < 0 or validation_episodes < num_agents or validation_episodes % num_agents:
        raise ValueError("validation_every must be nonnegative; validation_episodes must divide evenly across corners")
    if not 0.0 <= sudden_death_probability <= 1.0:
        raise ValueError("sudden_death_probability must be between 0 and 1")
    if attack_lookahead < 1:
        raise ValueError("attack_lookahead must be at least 1")
    if not 0.0 <= min_pressure_gain <= 1.0:
        raise ValueError("min_pressure_gain must be between 0 and 1")
    if trap_sequence_depth < 1:
        raise ValueError("trap_sequence_depth must be at least 1")
    if sequence_pressure_gain < 0.0:
        raise ValueError("sequence_pressure_gain must be non-negative")
    if temporal_trap_horizon < 1:
        raise ValueError("temporal trap horizon must be positive")
    if temporal_trap_max_bombs != 2:
        raise ValueError("temporal trap max bombs must equal 2")
    if interception_horizon < 1:
        raise ValueError("interception horizon must be positive")
    if late_game_start < 0 or late_game_attack_multiplier <= 0:
        raise ValueError("late-game configuration must be nonnegative with a positive multiplier")
    if min(escape_loss_coef, combat_loss_coef, navigation_loss_coef, ring_safety_coef) < 0:
        raise ValueError("supervision loss coefficients must be nonnegative")
    target_values = (
        target_win_rate,
        target_active_win_rate,
        target_draw_rate,
        target_sudden_death_death_rate,
        max_suicide_rate,
    )
    if any(value is not None and not 0.0 <= value <= 100.0 for value in target_values):
        raise ValueError("validation rates must be between 0 and 100")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    curriculum_rng = np.random.default_rng(seed)

    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                          max_steps=max_steps, seed=seed,
                          sudden_death=False,
                          sudden_death_start=sudden_death_start,
                          sudden_death_interval=sudden_death_interval,
                          sudden_death_warning_steps=sudden_death_warning_steps)
    net = ActorCriticNet(board_size=board_size,
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
    best_active_wr = 0.0
    best_episode = 0
    best_score = (-float("inf"),)
    best_state_dict = None   # will hold state_dict of the best model seen
    last_validation = None
    suicide_limit = max_suicide_rate
    validation_enabled = validation_every > 0 and episodes >= validation_every
    target_mode = any(value is not None for value in (
        target_win_rate,
        target_active_win_rate,
        target_draw_rate,
        target_sudden_death_death_rate,
    ))

    def validate_candidate(episode):
        nonlocal best_wr, best_active_wr, best_episode, best_score, best_state_dict, last_validation, suicide_limit
        from evaluate import evaluate_seeded
        training_mode = net.training
        # Rank checkpoints with the same CPU policy execution used by the GUI.
        policy = SnapshotOpponent(deepcopy(net).to("cpu"), "cpu")
        print(f"[Validation] ep {episode} | {validation_episodes} games, all corners, seed {validation_seed}", flush=True)
        def progress(done, total):
            if on_validation is not None:
                on_validation({"episode": episode, "completed": done, "total": total})
        try:
            result = evaluate_seeded(lambda e, obs, aid: policy.select_action(obs[aid], e, aid),
                                     board_size, num_agents, validation_episodes, validation_seed,
                                     max_steps, stop_event, progress,
                                     sudden_death=sudden_death_probability > 0.0,
                                     sudden_death_start=sudden_death_start,
                                     sudden_death_interval=sudden_death_interval,
                                     sudden_death_warning_steps=sudden_death_warning_steps)
        finally:
            net.train(training_mode)
        if result is None or (stop_event is not None and stop_event.is_set()):
            return
        last_validation = {k: v for k, v in result.items() if k != "games"}
        last_validation["device"] = "cpu"
        if target_mode:
            targets = (
                ("win_rate", target_win_rate, 1.0),
                ("active_win_rate", target_active_win_rate, 1.0),
                ("draw_rate", target_draw_rate, -1.0),
                ("sudden_death_death_rate", target_sudden_death_death_rate, -1.0),
            )
            margins = []
            achieved = 0
            for metric, target, direction in targets:
                if target is None:
                    continue
                margin = direction * (result[metric] - target) / max(abs(target), 1.0)
                margins.append(margin)
                achieved += int(margin >= 0.0)
            score = (achieved, min(margins), result["active_win_rate"],
                     -result["draw_rate"], -result["sudden_death_death_rate"],
                     result["win_rate"])
        else:
            score = (result["active_win_rate"], -result["suicide_rate"],
                     -result["draw_rate"], result["avg_kills"], result["win_rate"])
            if suicide_limit is None:
                suicide_limit = result["suicide_rate"]
        safe_candidate = suicide_limit is None or result["suicide_rate"] <= suicide_limit
        selected = safe_candidate and score > best_score
        last_validation["suicide_limit"] = suicide_limit
        os.makedirs(checkpoint_dir, exist_ok=True)
        tag = f"phaseE_{num_agents}a" if phase == "E" else f"phase{phase}"
        if selected:
            best_score, best_episode = score, episode
            best_wr, best_active_wr = result["win_rate"], result["active_win_rate"]
            best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}
            torch.save(best_state_dict, os.path.join(checkpoint_dir, f"best_{tag}.pt"))
            if phase == "E":
                torch.save(best_state_dict, os.path.join(checkpoint_dir, "best_phaseE.pt"))
            with open(os.path.join(checkpoint_dir, "best_validation.json"), "w", encoding="utf-8") as handle:
                json.dump({"episode": episode, **last_validation}, handle, indent=2)
        with open(os.path.join(checkpoint_dir, "validation_metrics.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"episode": episode, "selected": selected, **last_validation}) + "\n")
        print(f"[Validation] ep {episode} | ActiveWR {result['active_win_rate']:.1f}% | "
              f"Suicide {result['suicide_rate']:.1f}% | Draw {result['draw_rate']:.1f}% | best ep {best_episode}", flush=True)
        if not safe_candidate and suicide_limit is not None:
            print(f"[Validation] kept best: suicide exceeds warm-start limit {suicide_limit:.1f}%", flush=True)
        if on_validation is not None:
            on_validation({"episode": episode, "result": last_validation, "best_episode": best_episode})

    if validation_enabled:
        validate_candidate(0)  # Keep the warm-start if subsequent updates fail validation.

    # ── Logging buffers ──
    reward_history = []
    win_history = []        # 1 = learning agent is the sole survivor
    bombs_history = []      # bombs placed by learning agent
    walls_history = []      # walls destroyed by learning agent's bombs
    kills_history = []
    active_win_history = []
    suicide_history = []
    combat_history = []
    temporal_history = deque(maxlen=log_every)

    for ep in range(1, episodes + 1):
        if stop_event is not None and stop_event.is_set():
            break
        if sudden_death_probability == 0.0:
            env.sudden_death = False
        elif sudden_death_probability == 1.0:
            env.sudden_death = True
        else:
            env.sudden_death = bool(curriculum_rng.random() < sudden_death_probability)
        sudden_death_enabled = env.sudden_death
        obs_list = env.reset()
        alive_mask = [True] * num_agents

        learning_aid = (ep - 1) % num_agents

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

        if on_step is not None:
            on_step(env, ep, learning_aid, [ACTION_STOP] * num_agents, opponents)

        # ── Per-agent trajectory buffers (learning agent) ──
        log_probs_buf = []
        values_buf = []
        rewards_buf = []
        entropies_buf = []
        aux_losses_buf = []
        combat_losses_buf = []
        attack_losses_buf = []
        ring_losses_buf = []
        bombs_this_ep = 0
        walls_this_ep = 0
        suicide_this_ep = False
        items_this_ep = 0
        attack_opportunities = attack_bombs = attack_labels = missing_attack_labels = 0
        combat_labels = combat_bombs = 0
        followup_labels = followup_bombs = 0
        temporal_diagnostics = {
            "temporal_trap_labels": 0,
            "temporal_trap_plans": 0,
            "temporal_trap_bombs": 0,
            "temporal_trap_pair_completions": 0,
            "temporal_pair_plans_started": 0,
            "temporal_policy_action_matches": 0,
            "temporal_plan_aborts_mismatch": 0,
            "temporal_plan_aborts_mask": 0,
            "temporal_plan_aborts_danger": 0,
            "temporal_plan_aborts_ring": 0,
            "temporal_plan_aborts_stale_position": 0,
            "temporal_plan_aborts_placement": 0,
            "temporal_plan_aborts_inactive": 0,
            "temporal_plan_aborts_unsafe": 0,
            "temporal_trap_rejected_unsafe": 0,
            "temporal_trap_rejected_duplicate": 0,
            "interception_labels": 0,
            "predicted_refuge_reduction": 0.0,
            "predicted_trap_completion_ticks": 0.0,
        }
        temporal_follower = None
        temporal_config = TemporalTrapConfig(
            horizon=temporal_trap_horizon,
            max_bombs=temporal_trap_max_bombs,
            min_pressure_gain=sequence_pressure_gain,
        )
        sudden_death_deaths_this_ep = 0
        learner_sudden_death = False

        for t in range(max_steps):
            if stop_event is not None and stop_event.is_set():
                break
            # ── Select actions ──
            actions = [0] * num_agents

            # Learning agent: forward through network with action masking.
            if alive_mask[learning_aid]:
                obs_t = obs_to_tensor(obs_list[learning_aid], device)
                logits, value = net(obs_t)
                mask = get_action_mask(env, learning_aid)
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
                m_logits = logits.masked_fill(~mask_t, -1e9)
                dist = torch.distributions.Categorical(logits=m_logits)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t)
                entropy = dist.entropy()
                actions[learning_aid] = action_t.item()
                # Auxiliary escape supervision when in danger
                r_l, c_l = env.agents[learning_aid].row, env.agents[learning_aid].col
                d_map = _build_danger_map(env)
                temporal_diagnostics.pop("source", None)
                temporal_label, temporal_follower = _select_temporal_label(
                    env, learning_aid, d_map, mask, temporal_follower,
                    temporal_config, temporal_diagnostics,
                )
                urgent_escape = (d_map[r_l, c_l] == 1
                                 or (r_l, c_l) in env.flames)
                ring_action = (None if urgent_escape else
                               _ring_safety_action(env, learning_aid, d_map, mask))
                if urgent_escape:
                    esc_act = _bfs_safe_cell(env, r_l, c_l, d_map, action_mask=mask)
                    if esc_act is not None and mask[esc_act]:
                        aux_losses_buf.append(F.cross_entropy(
                            m_logits, torch.tensor([esc_act], device=device)))
                elif ring_action is not None:
                    ring_losses_buf.append(F.cross_entropy(
                        m_logits, torch.tensor([ring_action], device=device)))
                elif d_map[r_l, c_l] > 0 and temporal_label is None:
                    if mask[ACTION_BOMB]:
                        combat_losses_buf.append(F.cross_entropy(
                            m_logits, torch.tensor([ACTION_BOMB], device=device)))
                        followup_labels += 1
                        followup_bombs += int(actions[learning_aid] == ACTION_BOMB)
                        combat_labels += 1
                        combat_bombs += int(actions[learning_aid] == ACTION_BOMB)
                    else:
                        esc_act = _bfs_safe_cell(env, r_l, c_l, d_map, action_mask=mask)
                        if esc_act is not None and mask[esc_act]:
                            aux_losses_buf.append(
                                F.cross_entropy(m_logits, torch.tensor([esc_act], device=device))
                            )
                else:
                    attack_act = temporal_label
                    if attack_act is None:
                        attack_act = get_attack_action(
                            env, learning_aid, d_map, mask,
                            attack_lookahead=attack_lookahead,
                            min_pressure_gain=min_pressure_gain,
                            trap_sequence_depth=trap_sequence_depth,
                            sequence_pressure_gain=sequence_pressure_gain,
                            temporal_trap_horizon=temporal_trap_horizon,
                            temporal_trap_max_bombs=temporal_trap_max_bombs,
                            interception_horizon=interception_horizon,
                            diagnostics=temporal_diagnostics,
                            use_temporal_planner=False)
                    opportunity = mask[ACTION_BOMB] and any(
                        a.alive and a.agent_id != learning_aid and (a.row, a.col) in
                        env._blast_cells(r_l, c_l, env.agents[learning_aid].blast_range)
                        for a in env.agents)
                    attack_opportunities += int(opportunity)
                    attack_bombs += int(opportunity and actions[learning_aid] == ACTION_BOMB)
                    attack_labels += int(attack_act is not None)
                    missing_attack_labels += int(attack_act is None)
                    if attack_act is not None:
                        target_loss = F.cross_entropy(
                            m_logits, torch.tensor([attack_act], device=device))
                        sequence_action = _sequence_attack_action(
                            env, learning_aid, d_map, mask, attack_lookahead,
                            trap_sequence_depth, sequence_pressure_gain)
                        pressure_action = _pressure_attack_action(
                            env, learning_aid, d_map, mask,
                            attack_lookahead, min_pressure_gain)
                        source = temporal_diagnostics.get("source")
                        combat = source == "temporal"
                        if source != "interception":
                            combat |= attack_act in (sequence_action, pressure_action) and any(
                                action is not None
                                for action in (sequence_action, pressure_action))
                        if t >= late_game_start:
                            target_loss = target_loss * late_game_attack_multiplier
                        combat_labels += int(combat)
                        combat_bombs += int(combat and actions[learning_aid] == ACTION_BOMB)
                        (combat_losses_buf if combat else attack_losses_buf).append(target_loss)
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

            # ── Step environment ──
            obs_list, rewards, dones, info = env.step(actions)
            temporal_follower = _observe_temporal_transition(
                env, learning_aid, actions[learning_aid], info, temporal_follower,
            )

            # ── Reward shaping (anti-bomb-shyness & anti-kamikaze) ──
            shaped_reward = _combat_reward(rewards[learning_aid], learning_aid, info,
                                           env.agents[learning_aid].kills)
            bombs_this_ep += learning_aid in info["bombs_placed"]
            suicide_this_ep |= learning_aid in info.get("self_blast_deaths", []) or (learning_aid, learning_aid) in info["kills"]
            items_this_ep += len(info["items_collected"].get(learning_aid, []))
            sudden_death_deaths = info.get("sudden_death_deaths", [])
            sudden_death_deaths_this_ep += len(sudden_death_deaths)
            learner_sudden_death |= learning_aid in sudden_death_deaths

            # Wall destruction bonus
            wd = info.get("walls_destroyed", {})
            if learning_aid in wd:
                walls_this_ep += wd[learning_aid]
            if learning_aid in wd and info["alive"][learning_aid]:
                shaped_reward += wall_destroy_bonus * wd[learning_aid]

            # ── Store transition for learning agent ──
            if alive_mask[learning_aid] and log_prob is not None:
                log_probs_buf.append(log_prob.squeeze())
                values_buf.append(value.squeeze())
                rewards_buf.append(shaped_reward)
                entropies_buf.append(entropy.squeeze())

            alive_mask = list(info["alive"])
            if on_step is not None:
                on_step(env, ep, learning_aid, actions, opponents)
            if info["episode_done"]:
                break

        # A cancelled partial episode must not update the model or its metrics.
        if stop_event is not None and stop_event.is_set():
            break

        # ── Compute loss (Monte-Carlo returns + advantage) ──
        escape_loss, combat_loss, navigation_loss = _supervision_losses(
            aux_losses_buf, combat_losses_buf, attack_losses_buf)
        ring_safety_loss = (torch.stack(ring_losses_buf).mean()
                            if ring_losses_buf else 0.0)
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
            loss = (loss + escape_loss_coef * escape_loss
                    + combat_loss_coef * combat_loss
                    + navigation_loss_coef * navigation_loss
                    + ring_safety_coef * ring_safety_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
        else:
            loss = torch.zeros(())
            policy_loss = value_loss = loss

        # ── Logging ──
        ep_reward = sum(rewards_buf) if rewards_buf else 0.0
        reward_history.append(ep_reward)
        alive_final = info["alive"]
        other_alive = [alive_final[j] for j in range(num_agents) if j != learning_aid]
        won = 1 if (alive_final[learning_aid] and not any(other_alive)) else 0
        win_history.append(won)
        bombs_history.append(bombs_this_ep)
        walls_history.append(walls_this_ep)
        kills_this_ep = env.agents[learning_aid].kills
        active_win = int(won and kills_this_ep > 0)
        sudden_death_win = int(won and not active_win and sudden_death_deaths_this_ep > 0)
        passive_win = int(won and not active_win and not sudden_death_win)
        draw = int(sum(alive_final) != 1)
        kills_history.append(kills_this_ep)
        active_win_history.append(active_win)
        suicide_history.append(int(suicide_this_ep))
        combat_history.append((combat_bombs, combat_labels))
        temporal_history.append({key: value for key, value in temporal_diagnostics.items()
                                 if key != "source"})

        if validation_enabled and (ep % validation_every == 0 or ep == episodes):
            validate_candidate(ep)

        # ── Save snapshot (Phase D/E) ──
        if phase in ("D", "E") and ep % snapshot_every == 0:
            path = save_snapshot(net, checkpoint_dir, ep, phase=phase, num_agents=num_agents)
            # Add to live pool
            snap_net = ActorCriticNet(board_size=board_size).to(device)
            snap_net.load_state_dict(net.state_dict())
            snap_net.eval()
            snapshot_pool.append(SnapshotOpponent(snap_net, device))

        # ── LR schedule step ──
        scheduler.step()

        if on_episode is not None:
            temporal_metrics = _temporal_metrics(temporal_diagnostics)
            on_episode({"episode": ep, "reward": ep_reward, "win": won,
                        "bombs": bombs_this_ep, "walls": walls_this_ep,
                         "kills": kills_this_ep, "active_win": active_win,
                         "suicide": int(suicide_this_ep),
                         "items": items_this_ep, "learner_steps": len(rewards_buf),
                         "learning_aid": learning_aid, "draw": draw,
                         "sudden_death_enabled": sudden_death_enabled,
                         "sudden_death_death": int(learner_sudden_death),
                         "sudden_death_win": sudden_death_win,
                         "passive_win": passive_win,
                         "sudden_death_deaths": sudden_death_deaths_this_ep,
                         "sudden_death_wins": sudden_death_win,
                         "active_wins": active_win, "passive_wins": passive_win,
                         "draws": draw,
                        "attack_opportunities": attack_opportunities, "attack_bombs": attack_bombs,
                        "attack_labels": attack_labels, "missing_attack_labels": missing_attack_labels,
                        "combat_labels": combat_labels, "combat_bombs": combat_bombs,
                        "followup_labels": followup_labels, "followup_bombs": followup_bombs,
                        **temporal_metrics,
                        "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                        "escape_loss": float(escape_loss.detach()) if aux_losses_buf else 0.0,
                        "combat_loss": float(combat_loss.detach()) if combat_losses_buf else 0.0,
                        "navigation_loss": float(navigation_loss.detach()) if attack_losses_buf else 0.0,
                        "ring_safety_loss": float(ring_safety_loss.detach()) if ring_losses_buf else 0.0,
                        "steps": info["step"], "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0], "best_wr": best_wr,
                        "best_active_wr": best_active_wr, "best_episode": best_episode,
                        "validation": last_validation})

        # ── Print progress ──
        if ep % log_every == 0:
            n = log_every
            avg_reward = float(np.mean(reward_history[-n:]))
            avg_wr = float(np.mean(win_history[-n:])) * 100
            avg_bombs = float(np.mean(bombs_history[-n:]))
            avg_walls = float(np.mean(walls_history[-n:]))
            avg_kills = float(np.mean(kills_history[-n:]))
            active_wr = float(np.mean(active_win_history[-n:])) * 100
            cur_lr = scheduler.get_last_lr()[0]
            # Show best WR so far
            best_tag = f" | best validation active: {best_active_wr:.0f}%" if best_state_dict is not None else ""
            combat_taken, combat_total = map(sum, zip(*combat_history[-n:]))
            print(f"[{phase}] ep {ep:5d} | "
                  f"WR: {avg_wr:5.1f}% | "
                  f"active WR: {active_wr:5.1f}% | kills: {avg_kills:.2f} | "
                  f"reward: {avg_reward:+.3f} | "
                  f"bombs: {avg_bombs:.1f} | "
                  f"walls: {avg_walls:.1f} | "
                  f"combat bombs: {combat_taken}/{combat_total} | "
                  f"loss: {loss.item():.4f} | "
                  f"lr: {cur_lr:.1e} | "
                  f"pool: {len(snapshot_pool)}{best_tag}")
            temporal_totals = {
                key: sum(episode[key] for episode in temporal_history)
                for key in temporal_history[0]
            }
            temporal = _temporal_metrics(temporal_totals)
            print(f"[Temporal] ep {ep} | plans: {temporal['temporal_plans_started']} "
                  f"(pairs: {temporal['temporal_pair_plans_started']}) | "
                  f"steps: {temporal['temporal_steps_supervised']} | "
                  f"matches: {temporal['temporal_policy_action_matches']} "
                  f"({temporal['temporal_policy_agreement_rate']:.1%}) | "
                  f"bombs: {temporal['temporal_bomb_actions_executed']} | "
                  f"pair completions: {temporal['temporal_pair_completions']} | "
                  f"aborts: {temporal['temporal_plan_aborts']} "
                  f"(safety: {temporal['temporal_plan_aborts_safety']}, "
                  f"stale: {temporal['temporal_plan_aborts_stale']}, "
                  f"masked: {temporal['temporal_plan_aborts_masked']}, "
                  f"mismatch: {temporal['temporal_plan_aborts_mismatch']})")

    # ── Save final model ──
    os.makedirs(checkpoint_dir, exist_ok=True)
    tag = f"phaseE_{num_agents}a" if phase == "E" else f"phase{phase}"
    final_path = os.path.join(checkpoint_dir, f"final_{tag}.pt")
    best_path = os.path.join(checkpoint_dir, f"best_{tag}.pt")
    torch.save(net.state_dict(), final_path)
    torch.save(best_state_dict if best_state_dict is not None else net.state_dict(), best_path)
    if phase == "E":
        torch.save(net.state_dict(), os.path.join(checkpoint_dir, "final_phaseE.pt"))
        torch.save(best_state_dict if best_state_dict is not None else net.state_dict(),
                   os.path.join(checkpoint_dir, "best_phaseE.pt"))
    print(f"\nSaved final model to {final_path}\nSaved best model to {best_path} (best episode: {best_episode})")
    if phase == "C" and best_state_dict is not None:
        net.load_state_dict(best_state_dict)  # use best for gate checks, after saving final

    # ── Final report & gate checks ──
    if run_report and win_history and not (stop_event is not None and stop_event.is_set()):
        _print_report(phase, len(win_history), board_size, num_agents,
                      win_history, reward_history, bombs_history, walls_history,
                      kills_history, active_win_history,
                      snapshot_pool, env, net, device, checkpoint_dir)

    return net, reward_history, win_history


def _print_report(phase, episodes, board_size, num_agents,
                  win_history, reward_history, bombs_history, walls_history,
                  kills_history, active_win_history,
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

    print(f"  Final win rate (last {last_n}):  {final_wr:.1f}%")
    print(f"  Final avg reward (last {last_n}): {final_reward:.3f}")
    print(f"  Final avg bombs  (last {last_n}): {final_bombs:.1f}")
    print(f"  Final avg walls  (last {last_n}): {final_walls:.1f}")
    print(f"  Final avg kills  (last {last_n}): {np.mean(kills_history[-last_n:]):.2f}")
    print(f"  Active win rate  (last {last_n}): {np.mean(active_win_history[-last_n:]) * 100:.1f}%")

    print(f"\n--- Gate Checks (Phase {phase}) ---")
    combat_ok = any(kills_history[-last_n:]) and any(active_win_history[-last_n:])
    print("  Owned eliminations and active wins: " + ("[PASS]" if combat_ok else "[FAIL]"))

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
        obs_list, rewards, dones, info = env.step(actions)
        bombs_placed += len(info["bombs_placed"])
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
    kills = sum(a.kills for a in env.agents)
    result = "[PASS]" if bombs_placed > 0 and kills > 0 else "[INCONCLUSIVE] (no owned elimination observed)"
    print(f"  Demo combat: {bombs_placed} bombs, {kills} owned kills {result}")
    net.train()


@torch.no_grad()
def _eval_vs_rulebot(net: ActorCriticNet, env: BombermanFFAEnv,
                     device, board_size: int,
                     num_episodes: int = 200) -> float:
    """Evaluate current net vs RuleBasedAgent, return win rate %."""
    net.eval()
    rule_bot = RuleBasedOpponent(1)
    wins = 0
    for _ in range(num_episodes):
        obs_list = env.reset()
        for t in range(env.max_steps):
            # Agent 0: trained model
            obs_t = obs_to_tensor(obs_list[0], device)
            logits, _ = net(obs_t)
            mask0 = get_action_mask(env, 0)
            mask_t0 = torch.tensor(mask0, dtype=torch.bool, device=device)
            m_logits0 = logits.masked_fill(~mask_t0, -1e9)
            dist = torch.distributions.Categorical(logits=m_logits0)
            a0 = dist.sample().item()
            # Agent 1: RuleBasedAgent
            a1 = rule_bot.select_action_env(env)
            obs_list, _, _, info = env.step([a0, a1])
            if info["episode_done"]:
                break
        if info["alive"][0] and not info["alive"][1]:
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
    parser.add_argument("--episodes", type=int, default=750)
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.000025)
    parser.add_argument("--entropy_coef", type=float, default=0.0075)
    parser.add_argument("--wall_destroy_bonus", type=float, default=0.15,
                        help="Reward bonus per wall destroyed (anti-bomb-shyness)")
    parser.add_argument("--attack_lookahead", type=int, default=2)
    parser.add_argument("--min_pressure_gain", type=float, default=0.15)
    parser.add_argument("--trap_sequence_depth", type=int, default=2)
    parser.add_argument("--sequence_pressure_gain", type=float, default=0.35)
    parser.add_argument("--temporal_trap_horizon", type=int, default=8)
    parser.add_argument("--temporal_trap_max_bombs", type=int, default=2)
    parser.add_argument("--interception_horizon", type=int, default=4)
    parser.add_argument("--ring_safety_coef", type=float, default=0.6)
    parser.add_argument("--late_game_start", type=int, default=350)
    parser.add_argument("--late_game_attack_multiplier", type=float, default=1.75)
    parser.add_argument("--escape_loss_coef", type=float, default=0.5)
    parser.add_argument("--combat_loss_coef", type=float, default=0.9)
    parser.add_argument("--navigation_loss_coef", type=float, default=0.4)
    parser.add_argument("--target_win_rate", type=float, default=None)
    parser.add_argument("--target_active_win_rate", type=float, default=None)
    parser.add_argument("--target_draw_rate", type=float, default=None)
    parser.add_argument("--target_sudden_death_death_rate", type=float, default=None)
    parser.add_argument("--max_suicide_rate", type=float, default=None)
    parser.add_argument("--snapshot_every", type=int, default=200,
                        help="Save snapshot every N episodes (Phase D/E)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="Reproducible training seed")
    parser.add_argument("--validation_every", type=int, default=250)
    parser.add_argument("--validation_episodes", type=int, default=48)
    parser.add_argument("--validation_seed", type=int, default=6000)
    parser.add_argument("--sudden_death_probability", type=float, default=0.0)
    parser.add_argument("--sudden_death_start", type=int, default=350)
    parser.add_argument("--sudden_death_interval", type=int, default=20)
    parser.add_argument("--sudden_death_warning_steps", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    with open(os.path.join(args.checkpoint_dir, "metrics.jsonl"), "a", encoding="utf-8") as metrics_file:
        def write_metrics(metrics):
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()

        train(
            phase=args.phase,
            episodes=args.episodes,
            board_size=args.board_size,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            wall_destroy_bonus=args.wall_destroy_bonus,
            snapshot_every=args.snapshot_every,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
            log_every=args.log_every,
            seed=args.seed,
            validation_every=args.validation_every,
            validation_episodes=args.validation_episodes,
            validation_seed=args.validation_seed,
            sudden_death_probability=args.sudden_death_probability,
            sudden_death_start=args.sudden_death_start,
            sudden_death_interval=args.sudden_death_interval,
            sudden_death_warning_steps=args.sudden_death_warning_steps,
            attack_lookahead=args.attack_lookahead,
            min_pressure_gain=args.min_pressure_gain,
            trap_sequence_depth=args.trap_sequence_depth,
            sequence_pressure_gain=args.sequence_pressure_gain,
            temporal_trap_horizon=args.temporal_trap_horizon,
            temporal_trap_max_bombs=args.temporal_trap_max_bombs,
            interception_horizon=args.interception_horizon,
            ring_safety_coef=args.ring_safety_coef,
            late_game_start=args.late_game_start,
            late_game_attack_multiplier=args.late_game_attack_multiplier,
            escape_loss_coef=args.escape_loss_coef,
            combat_loss_coef=args.combat_loss_coef,
            navigation_loss_coef=args.navigation_loss_coef,
            target_win_rate=args.target_win_rate,
            target_active_win_rate=args.target_active_win_rate,
            target_draw_rate=args.target_draw_rate,
            target_sudden_death_death_rate=args.target_sudden_death_death_rate,
            max_suicide_rate=args.max_suicide_rate,
            on_episode=write_metrics,
        )
