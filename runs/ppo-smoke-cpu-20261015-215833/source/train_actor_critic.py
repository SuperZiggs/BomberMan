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
import math
from collections import Counter
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bomberman_env import BombermanFFAEnv, NUM_ACTIONS, OBS_CHANNELS, ACTION_BOMB, ACTION_STOP, WOODEN_WALL, ITEM_UPGRADE_REWARD, _MOVE_DELTA
from rule_based_bot import (
    RuleBasedAgent,
    _build_danger_map,
    _safe_action_mask,
    _bfs_safe_cell,
    _bfs_toward_opponent,
    _bfs_toward_wood,
    _bfs_toward_item,
    _bomb_escape_pressure,
)
from tactical_teacher import get_two_bomb_plan
from combat_move_teacher import certify_combat_move
from training_state import atomic_save, load_training_state, source_hashes, runtime_state, capture_rng, restore_rng


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

def normalize_policy_advantages(advantages: torch.Tensor):
    """Normalize one learner trajectory; retain its signal if variance is absent."""
    raw = advantages.detach()
    if not torch.isfinite(raw).all().item():
        raise FloatingPointError("Non-finite policy advantage")
    if raw.numel() < 2 or raw.std(unbiased=False).item() < 1e-8:
        return raw, True
    return (raw - raw.mean()) / raw.std(unbiased=False), False


def _tensor_stats(values):
    values = values.detach()
    return {"mean": values.mean().item(), "std": values.std(unbiased=False).item(),
            "min": values.min().item(), "max": values.max().item()}


@torch.no_grad()
def gae_advantages(rewards, values, next_values, bootstrap_allowed, continue_segment,
                   gamma=0.99, gae_lambda=0.95):
    """Detached GAE; a cut can bootstrap V(next) without crossing into another segment."""
    floats = (rewards, values, next_values)
    tensors = (*floats, bootstrap_allowed, continue_segment)
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be finite and in [0, 1]")
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("GAE inputs must be tensors")
    if rewards.ndim != 1 or not rewards.numel() or any(t.shape != rewards.shape for t in tensors):
        raise ValueError("GAE requires nonempty vectors of identical shape")
    if any(t.device != rewards.device for t in tensors):
        raise ValueError("GAE inputs must share one device")
    if not rewards.is_floating_point() or any(t.dtype != rewards.dtype for t in floats):
        raise ValueError("GAE rewards/values must share a floating dtype")
    if bootstrap_allowed.dtype != torch.bool or continue_segment.dtype != torch.bool:
        raise ValueError("GAE boundary masks must be boolean")
    if continue_segment[-1].item() or (continue_segment & ~bootstrap_allowed).any().item():
        raise ValueError("GAE cannot continue past a terminal or the collected segment")
    if not all(torch.isfinite(t).all().item() for t in floats):
        raise FloatingPointError("Non-finite GAE input")
    deltas = rewards + gamma * torch.where(bootstrap_allowed, next_values, 0) - values
    advantages = torch.empty_like(deltas)
    carry = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(rewards.numel())):
        # Multiply floats first: bool * Python float would round gamma to the default dtype.
        carry = deltas[t] + gamma * gae_lambda * carry * continue_segment[t]
        advantages[t] = carry
    if not torch.isfinite(advantages).all().item():
        raise FloatingPointError("Non-finite GAE advantage")
    return advantages, deltas


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


def get_attack_action(env: BombermanFFAEnv, agent_id: int, danger, mask):
    """Prefer bomb positions that restrict timed escape, then upgrades or opening routes."""
    agent = env.agents[agent_id]
    r, c = agent.row, agent.col
    if danger[r, c] > 0:
        return ACTION_BOMB if mask[ACTION_BOMB] else None
    close = any(a.alive and a.agent_id != agent_id and
                abs(a.row - r) + abs(a.col - c) <= agent.blast_range + 1
                for a in env.agents)
    if close:
        base = env._forecast_blasts(min_horizon=env.bomb_timer + env.flame_duration - 1)
        here = _bomb_escape_pressure(env, agent_id, r, c, base) if mask[ACTION_BOMB] else 0.0
        best, move = here, None
        occupied = {(a.row, a.col) for a in env.agents if a.alive and a.agent_id != agent_id}
        for action, (dr, dc) in _MOVE_DELTA.items():
            pos = (r + dr, c + dc)
            if not mask[action] or pos in occupied or danger[pos]:
                continue
            pressure = _bomb_escape_pressure(env, agent_id, *pos, base)
            if pressure > best + 1e-6:
                best, move = pressure, action
        if move is not None:
            return move
        if here > 0:
            return ACTION_BOMB
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


def _supervision_losses(escape, combat, navigation):
    """Separate means keep rare combat labels from being diluted by navigation."""
    return tuple(torch.stack(group).mean() if group else 0.0
                 for group in (escape, combat, navigation))


def freeze_rollout_step(obs, action, mask, old_log_prob, old_value, old_probs,
                        reward, terminal, boundary, teacher_group=None, teacher_actions=()):
    """Keep the sampled policy and labels independent of later env/model changes."""
    if (obs.ndim != 3 or mask.ndim != 1 or mask.dtype != torch.bool
            or old_probs.shape != mask.shape or not mask.any().item()
            or old_log_prob.numel() != 1 or old_value.numel() != 1):
        raise ValueError('Invalid rollout tensor shapes or empty action mask')
    if (not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(mask)
            or not mask[action].item()):
        raise ValueError('Sampled action must be allowed by the stored mask')
    tensors = (obs, old_log_prob, old_value, old_probs)
    if any(not t.is_floating_point() or t.dtype != old_probs.dtype or t.device != mask.device
           or not torch.isfinite(t).all().item() for t in tensors):
        raise ValueError('Rollout tensors must be finite and on the same device')
    if (not math.isfinite(reward) or (old_probs < 0).any().item()
            or old_probs[~mask].count_nonzero().item()
            or not torch.isclose(old_probs.sum(), old_probs.new_tensor(1.), atol=1e-6).item()
            or old_probs[action].item() <= 0
            or not torch.isclose(old_probs[action].log(), old_log_prob.squeeze(), atol=1e-5).item()):
        raise ValueError('Stored probabilities, action log probability or reward disagree')
    if (not isinstance(terminal, bool) or
            (boundary not in ('learner_death', 'time_limit', 'match_end') if terminal else boundary is not None)):
        raise ValueError('Terminal transition requires its actual boundary')
    if (teacher_group not in (None, 'escape', 'combat', 'navigation')
            or bool(teacher_actions) != (teacher_group is not None)
            or len(set(teacher_actions)) != len(teacher_actions)
            or any(not isinstance(a, int) or isinstance(a, bool) or not 0 <= a < len(mask)
                   or not mask[a].item() for a in teacher_actions)):
        raise ValueError('Teacher targets must belong to one group and the stored mask')
    target = old_probs.new_zeros(old_probs.shape)
    if teacher_actions:
        target[list(teacher_actions)] = 1. / len(teacher_actions)
    return {'obs': obs.detach().clone(), 'action': action, 'mask': mask.detach().clone(),
            'old_log_prob': old_log_prob.detach().clone().squeeze(),
            'old_value': old_value.detach().clone().squeeze(), 'old_probs': old_probs.detach().clone(),
            'reward': float(reward), 'terminal': terminal, 'boundary': boundary,
            'teacher_group': teacher_group, 'teacher_target': target.detach()}


def ratio_rollout_losses(net, steps, advantages, returns, clip_epsilon=None):
    """Full-episode ratio objective with fixed targets; optional D1 PPO clipping."""
    if clip_epsilon is not None and (not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1):
        raise ValueError('clip_epsilon must be in (0, 1)')
    if (not steps or advantages.shape != (len(steps),) or returns.shape != advantages.shape
            or any(s['terminal'] for s in steps[:-1]) or not steps[-1]['terminal']):
        raise ValueError('D0 requires one complete learner trajectory and matching targets')
    if not torch.isfinite(advantages).all().item() or not torch.isfinite(returns).all().item():
        raise FloatingPointError('Non-finite fixed rollout targets')
    mask = torch.stack([s['mask'] for s in steps])
    logits, values = net(torch.stack([s['obs'] for s in steps]))
    dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))
    actions = torch.tensor([s['action'] for s in steps], device=mask.device)
    old_log_probs = torch.stack([s['old_log_prob'] for s in steps])
    ratio = (dist.log_prob(actions) - old_log_probs).exp()
    entropies = dist.entropy()
    surrogate = ratio * advantages.detach()
    if clip_epsilon is not None:
        surrogate = torch.minimum(surrogate, ratio.clamp(1-clip_epsilon, 1+clip_epsilon) * advantages.detach())
    result = {'policy_loss': -surrogate.mean(), 'ratios': ratio.detach(),
              'value_loss': F.mse_loss(values.reshape(-1), returns.detach()),
              'entropy': entropies.mean(), 'entropies': entropies}
    log_probs = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
    counts = {}
    for group in ('escape', 'combat', 'navigation'):
        indices = [i for i, s in enumerate(steps) if s['teacher_group'] == group]
        counts[group] = len(indices)
        result[group + '_loss'] = -(torch.stack([steps[i]['teacher_target'] for i in indices])
            * log_probs[indices]).sum(-1).mean() if indices else values.new_zeros(())
    old_probs = torch.stack([s['old_probs'] for s in steps])
    old_logs = old_probs.clamp_min(torch.finfo(old_probs.dtype).tiny).log()
    kl = (old_probs * (old_logs - dist.logits)).sum(-1)
    if not all(torch.isfinite(result[k]).all().item() for k in
               ('policy_loss', 'value_loss', 'entropy', 'escape_loss', 'combat_loss', 'navigation_loss')):
        raise FloatingPointError('Non-finite ratio rollout loss')
    result['diagnostics'] = {'objective': 'ppo' if clip_epsilon is not None else 'ratio', 'samples': len(steps), 'epochs': 1,
        'clipped': clip_epsilon is not None, 'ratio_mean': ratio.detach().mean().item(),
        'ratio_min': ratio.detach().min().item(), 'ratio_max': ratio.detach().max().item(),
        'ratio_max_abs_error_from_one': (ratio.detach() - 1).abs().max().item(),
        'exact_kl_before_update': kl.detach().mean().item(), 'label_counts': counts,
        'terminal_transitions': sum(s['terminal'] for s in steps), 'last_boundary': steps[-1]['boundary']}
    return result


def update_ratio_rollout(net, optimizer, steps, advantages, returns, *, epochs, clipped,
                         clip_epsilon, target_kl, value_coef, entropy_coef, stop_event=None):
    """Reuse one fixed rollout; cancellation rolls back every update in this episode."""
    if (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1
            or not 0 < clip_epsilon < 1 or not math.isfinite(target_kl) or target_kl < 0):
        raise ValueError('Invalid epoch count, clipping epsilon or KL target')
    rollback = (deepcopy(net.state_dict()), deepcopy(optimizer.state_dict())) if stop_event is not None else None
    history = []
    def cancelled():
        if stop_event is None or not stop_event.is_set():
            return False
        net.load_state_dict(rollback[0])
        optimizer.load_state_dict(rollback[1])
        optimizer.zero_grad(set_to_none=True)
        return True
    for epoch in range(1, epochs+1):
        if cancelled():
            return None
        replay = ratio_rollout_losses(net, steps, advantages, returns, clip_epsilon if clipped else None)
        loss = replay['policy_loss'] + value_coef*replay['value_loss'] - entropy_coef*replay['entropy']
        loss = loss + .5*replay['escape_loss'] + .5*replay['combat_loss'] + .2*replay['navigation_loss']
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite multi-epoch training loss')
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.)
        if not torch.isfinite(grad_norm).item():
            raise FloatingPointError('Non-finite multi-epoch training gradient')
        optimizer.step()
        if cancelled():
            return None
        with torch.no_grad():
            post = ratio_rollout_losses(net, steps, advantages, returns, clip_epsilon if clipped else None)
        if cancelled():
            return None
        diagnostics = dict(replay['diagnostics'])
        ratios = replay['ratios']
        diagnostics.update(epoch=epoch, ratio_p95=ratios.quantile(.95).item(),
            clip_fraction=((ratios-1).abs() > clip_epsilon).float().mean().item(),
            entropy=replay['entropy'].detach().item(), value_loss=replay['value_loss'].detach().item(),
            policy_loss=replay['policy_loss'].detach().item(), loss=loss.detach().item(),
            grad_norm_before_clip=grad_norm.item(),
            exact_kl_after_update=post['diagnostics']['exact_kl_before_update'])
        history.append(diagnostics)
        exceeded = bool(target_kl and diagnostics['exact_kl_after_update'] > 1.5*target_kl)
        if exceeded:
            break
    diagnostics = {**replay['diagnostics'], 'objective': 'ppo' if clipped else 'ratio',
        'clipped': clipped, 'epochs': len(history), 'epochs_requested': epochs,
        'clip_epsilon': clip_epsilon, 'target_kl': target_kl, 'kl_threshold': 1.5*target_kl if target_kl else None,
        'guard_exceeded': exceeded, 'early_stop': exceeded and len(history) < epochs, 'epoch_metrics': history}
    return {**replay, 'diagnostics': diagnostics, 'grad_norm': grad_norm}


def _combat_reward(reward: float, agent_id: int, info, episode_kills: int):
    """Reward owned eliminations; a token bomb cannot unlock a passive win bonus."""
    kills = info["kills"]
    if agent_id in info.get("self_blast_deaths", []) or (agent_id, agent_id) in kills:
        reward -= 1.5  # -1 from env, -2.5 total for suicide.
    if info["alive"][agent_id]:
        reward += 3.0 * sum(k == agent_id and v != agent_id for k, v in kills)
        if info["episode_done"] and sum(info["alive"]) == 1 and episode_kills == 0:
            reward -= 10.0
    return reward


def _shaped_reward(reward, agent_id, info, episode_kills, wall_destroy_bonus=0.15):
    value = _combat_reward(reward, agent_id, info, episode_kills)
    walls = info.get("walls_destroyed", {})
    if agent_id in walls and info["alive"][agent_id]:
        value += wall_destroy_bonus * walls[agent_id]
    return value


def _reward_components(reward, agent_id, info, episode_kills, was_alive,
                       wall_destroy_bonus=0.15):
    """Explain the existing reward; never absorb a mismatch into a residual term."""
    alive = info["alive"][agent_id]
    owned = sum(k == agent_id and v != agent_id for k, v in info["kills"])
    sole = alive and info["episode_done"] and sum(info["alive"]) == 1
    parts = {
        "time": -0.001 if alive else 0.0,
        "item": ITEM_UPGRADE_REWARD * len(info.get("items_collected", {}).get(agent_id, [])),
        "wall": wall_destroy_bonus * info.get("walls_destroyed", {}).get(agent_id, 0) if alive else 0.0,
        "owned_kill": 4.0 * owned if alive else 0.0,
        "active_win": 10.0 if sole and episode_kills > 0 else 0.0,
        "passive_win_env": 10.0 if sole and episode_kills == 0 else 0.0,
        "passive_win_cancellation": -10.0 if sole and episode_kills == 0 else 0.0,
        "death": -1.0 if was_alive and not alive else 0.0,
        "self_blast": -1.5 if agent_id in info.get("self_blast_deaths", []) or
            (agent_id, agent_id) in info["kills"] else 0.0,
    }
    env_total = (parts["time"] + parts["item"] + (float(owned) if alive else 0.0) +
                 parts["active_win"] + parts["passive_win_env"] + parts["death"])
    if not math.isclose(env_total, reward, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Environment reward does not match its recorded events")
    actual = _shaped_reward(reward, agent_id, info, episode_kills, wall_destroy_bonus)
    if not math.isfinite(actual) or not math.isclose(sum(parts.values()), actual, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Reward components do not match the learner's shaped reward")
    return parts


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
    if sd.get('format') == 'bomberman_phase_c_training_v1':
        raise ValueError('Full training state: use resume_training_state; for inference select best/final weights')
    curr_sd = net.state_dict()
    for k, v in sd.items():
        if k in curr_sd:
            if v.shape == curr_sd[k].shape:
                curr_sd[k].copy_(v)
            elif k == "conv.0.weight" and v.shape[1] < curr_sd[k].shape[1]:
                curr_sd[k].zero_()
                curr_sd[k][:, :v.shape[1]].copy_(v)
                print(f"Expanded observation channels {v.shape[1]} -> {curr_sd[k].shape[1]}")
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
    episodes: int = 3000,
    board_size: int = 9,
    num_agents: int = 2,
    gamma: float = 0.99,
    lr: float = 3e-4,
    value_coef: float = 0.5,
    entropy_coef: float = 0.015,
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
    attack_teacher: str = "local",
    normalize_advantage: bool = False,
    advantage_diagnostics: bool = False,
    advantage_estimator: str = "mc",
    gae_lambda: float = 0.95,
    reward_diagnostics: bool = False,
    save_training_state: bool = False,
    resume_training_state: str = "",
    checkpoint_context=None,
    policy_objective: str = "ac",
    update_epochs: int = 1,
    clip_epsilon: float = 0.2,
    target_kl: float = 0.0,
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
    save_training_state = save_training_state or bool(resume_training_state)
    if save_training_state and phase != "C":
        raise ValueError("Full training-state resume currently supports Phase C only")
    resume_config = {key: value for key, value in locals().copy().items() if key in (
        'phase', 'episodes', 'board_size', 'num_agents', 'gamma', 'lr', 'value_coef', 'entropy_coef',
        'wall_destroy_bonus', 'max_steps', 'log_every', 'snapshot_every', 'resume_from', 'device', 'seed',
        'validation_every', 'validation_episodes', 'validation_seed', 'attack_teacher',
        'normalize_advantage', 'advantage_diagnostics', 'advantage_estimator', 'gae_lambda', 'reward_diagnostics',
        'policy_objective', 'update_epochs', 'clip_epsilon', 'target_kl')}
    restored = load_training_state(resume_training_state) if resume_training_state else None
    provenance = {'config': resume_config, 'sources': source_hashes(), 'runtime': runtime_state()} if save_training_state else None
    if restored is not None and restored['provenance'] != provenance:
        raise ValueError("Training config, source, device or runtime changed; exact resume refused")
    assert phase in ("C", "D", "E"), f"Phase must be C, D, or E, got {phase}"
    if episodes < 1 or max_steps < 1 or log_every < 1 or snapshot_every < 1:
        raise ValueError("episodes, max_steps, log_every and snapshot_every must be positive")
    if attack_teacher not in ("local", "two_bomb", "local_combat_move"):
        raise ValueError("attack_teacher must be local, two_bomb or local_combat_move")
    if policy_objective not in ('ac', 'ratio', 'ppo'):
        raise ValueError('policy_objective must be ac/ratio/ppo')
    if (not isinstance(update_epochs, int) or isinstance(update_epochs, bool) or update_epochs < 1
            or not 0 < clip_epsilon < 1 or not math.isfinite(target_kl) or target_kl < 0
            or (policy_objective == 'ac' and (update_epochs != 1 or target_kl != 0))):
        raise ValueError('Invalid epoch count, clipping epsilon or KL target for this objective')
    multi_update = policy_objective == 'ppo' or update_epochs > 1 or target_kl > 0
    if advantage_estimator not in ("mc", "gae") or not 0 <= gae_lambda <= 1:
        raise ValueError("advantage_estimator must be mc/gae; gae_lambda must be finite and in [0, 1]")
    if advantage_estimator == "gae" and not 0 <= gamma <= 1:
        raise ValueError("GAE gamma must be finite and in [0, 1]")
    if validation_every < 0 or validation_episodes < num_agents or validation_episodes % num_agents:
        raise ValueError("validation_every must be nonnegative; validation_episodes must divide evenly across corners")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    advantage_diagnostics = (advantage_diagnostics or normalize_advantage
                             or advantage_estimator == "gae" or policy_objective != 'ac')

    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                          max_steps=max_steps, seed=seed)
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
    suicide_limit = None
    validation_enabled = validation_every > 0 and episodes >= validation_every

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
                                     max_steps, stop_event, progress)
        finally:
            net.train(training_mode)
        if result is None or (stop_event is not None and stop_event.is_set()):
            return
        last_validation = {k: v for k, v in result.items() if k != "games"}
        last_validation["device"] = "cpu"
        score = (result["active_win_rate"], -result["suicide_rate"], -result["draw_rate"],
                 result["avg_kills"], result["win_rate"])
        if suicide_limit is None:
            suicide_limit = result["suicide_rate"]
        safe_candidate = result["suicide_rate"] <= suicide_limit
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
        if not safe_candidate:
            print(f"[Validation] kept best: suicide exceeds warm-start limit {suicide_limit:.1f}%", flush=True)
        if on_validation is not None:
            on_validation({"episode": episode, "result": last_validation, "best_episode": best_episode})

    # ── Logging buffers ──
    reward_history = []
    win_history = []        # 1 = learning agent is the sole survivor
    bombs_history = []      # bombs placed by learning agent
    walls_history = []      # walls destroyed by learning agent's bombs
    kills_history = []
    active_win_history = []
    suicide_history = []
    combat_history = []
    histories = {'reward': reward_history, 'win': win_history, 'bombs': bombs_history,
                 'walls': walls_history, 'kills': kills_history, 'active_win': active_win_history,
                 'suicide': suicide_history, 'combat': combat_history}
    state_path = os.path.join(checkpoint_dir, 'training_state.pt')

    def restore_state(state):
        nonlocal best_wr, best_active_wr, best_episode, best_score, best_state_dict, last_validation, suicide_limit
        net.load_state_dict(state['model'])
        net.train(state['training_mode'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        for key, values in histories.items():
            values[:] = state['histories'][key]
        (best_wr, best_active_wr, best_episode, best_score, best_state_dict,
         last_validation, suicide_limit) = state['selection']
        if best_state_dict is not None:
            best_state_dict = {key: value.to(device) for key, value in best_state_dict.items()}
        restore_rng(state['rng'], env)

    def save_state(episode):
        if save_training_state:
            atomic_save({'format': 'bomberman_phase_c_training_v1', 'episode': episode,
                'provenance': provenance, 'model': net.state_dict(), 'training_mode': net.training,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'histories': histories, 'selection': (best_wr, best_active_wr, best_episode, best_score,
                    best_state_dict, last_validation, suicide_limit),
                'context': json.loads(json.dumps(checkpoint_context())) if checkpoint_context is not None else None,
                'rng': capture_rng(env)}, state_path)

    start_episode = restored['episode'] if restored is not None else 0
    if restored is not None:
        restore_state(restored)
    else:
        save_state(0)
    if validation_enabled and last_validation is None:
        validate_candidate(0)  # Keep the warm-start if subsequent updates fail validation.
        if save_training_state and (last_validation is None or (stop_event is not None and stop_event.is_set())):
            restore_state(load_training_state(state_path))
        else:
            save_state(0)

    for ep in range(start_episode + 1, episodes + 1):
        if stop_event is not None and stop_event.is_set():
            break
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
        reward_totals = Counter()
        reward_max_error = 0.0
        entropies_buf = []
        mask_sizes_buf = []
        terminal_buf = []
        last_boundary = None
        aux_losses_buf = []
        combat_losses_buf = []
        attack_losses_buf = []
        rollout = [] if policy_objective != 'ac' else None
        bombs_this_ep = 0
        walls_this_ep = 0
        suicide_this_ep = False
        items_this_ep = 0
        attack_opportunities = attack_bombs = attack_labels = missing_attack_labels = 0
        combat_labels = combat_bombs = 0
        followup_labels = followup_bombs = 0
        teacher_stats = Counter(queries=0, plans=0, accepted=0, two_bomb_labels=0,
                                pressure=0.0, milliseconds=0.0)
        teacher_rejected = Counter()
        move_guidance = Counter(labels=0, agreements=0, bomb_instead_of_move=0,
                                target_probability_sum=0.0, entropy_sum=0.0,
                                queries=0, accepted=0, accepted_agreements=0,
                                pressure_gain_sum=0.0, certificate_steps=0, milliseconds=0.0)
        move_rejected = Counter()

        for t in range(max_steps):
            if stop_event is not None and stop_event.is_set():
                break
            # ── Select actions ──
            actions = [0] * num_agents

            # Learning agent: forward through network with action masking.
            if alive_mask[learning_aid]:
                obs_t = obs_to_tensor(obs_list[learning_aid], device)
                with torch.set_grad_enabled(policy_objective == 'ac'):
                    logits, value = net(obs_t)
                mask = get_action_mask(env, learning_aid)
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
                m_logits = logits.masked_fill(~mask_t, -1e9)
                dist = torch.distributions.Categorical(logits=m_logits)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t)
                entropy = dist.entropy()
                if advantage_diagnostics:
                    mask_sizes_buf.append(sum(mask))
                actions[learning_aid] = action_t.item()
                teacher_group, teacher_actions = None, ()
                # Auxiliary escape supervision when in danger
                r_l, c_l = env.agents[learning_aid].row, env.agents[learning_aid].col
                d_map = _build_danger_map(env)
                if d_map[r_l, c_l] == 0:
                    opportunity = mask[ACTION_BOMB] and any(
                        a.alive and a.agent_id != learning_aid and (a.row, a.col) in
                        env._blast_cells(r_l, c_l, env.agents[learning_aid].blast_range)
                        for a in env.agents)
                    attack_opportunities += int(opportunity)
                    attack_bombs += int(opportunity and actions[learning_aid] == ACTION_BOMB)
                plan = None
                if attack_teacher == "two_bomb":
                    plan, diagnostics = get_two_bomb_plan(env, learning_aid, mask)
                    teacher_stats["queries"] += 1
                    teacher_rejected.update(diagnostics.pop("rejected"))
                    teacher_stats.update(diagnostics)
                if plan is not None:
                    # Uniform soft target for equally good first actions, combat weight 0.5.
                    combat_losses_buf.append(-F.log_softmax(m_logits, dim=-1)[0, plan["first_actions"]].mean())
                    teacher_group, teacher_actions = 'combat', tuple(plan['first_actions'])
                    combat_labels += 1
                    combat_bombs += int(ACTION_BOMB in plan["first_actions"] and actions[learning_aid] == ACTION_BOMB)
                    attack_labels += 1
                elif d_map[r_l, c_l] > 0:
                    if mask[ACTION_BOMB]:
                        combat_losses_buf.append(F.cross_entropy(
                            m_logits, torch.tensor([ACTION_BOMB], device=device)))
                        teacher_group, teacher_actions = 'combat', (ACTION_BOMB,)
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
                            teacher_group, teacher_actions = 'escape', (esc_act,)
                else:
                    attack_act = get_attack_action(env, learning_aid, d_map, mask)
                    attack_labels += int(attack_act is not None)
                    missing_attack_labels += int(attack_act is None)
                    if attack_act is not None:
                        target_loss = F.cross_entropy(m_logits, torch.tensor([attack_act], device=device))
                        combat = (attack_act == ACTION_BOMB and
                                  _bomb_escape_pressure(env, learning_aid, r_l, c_l) > 0)
                        if attack_act in _MOVE_DELTA:
                            move_guidance.update(labels=1, agreements=int(actions[learning_aid] == attack_act),
                                bomb_instead_of_move=int(actions[learning_aid] == ACTION_BOMB),
                                target_probability_sum=float(dist.probs[0, attack_act].detach()),
                                entropy_sum=float(entropy.detach()))
                            if attack_teacher == "local_combat_move":
                                certificate = certify_combat_move(env, learning_aid, attack_act, mask)
                                accepted = certificate['accepted']
                                move_guidance.update(queries=1, accepted=int(accepted),
                                    accepted_agreements=int(accepted and actions[learning_aid] == attack_act),
                                    pressure_gain_sum=certificate['pressure_gain'] if accepted else 0.0,
                                    certificate_steps=len(certificate['path']) if accepted else 0,
                                    milliseconds=certificate['milliseconds'])
                                if not accepted:
                                    move_rejected[certificate['reason']] += 1
                                combat = accepted
                        combat_labels += int(combat)
                        combat_bombs += int(combat and attack_act == ACTION_BOMB and actions[learning_aid] == ACTION_BOMB)
                        (combat_losses_buf if combat else attack_losses_buf).append(target_loss)
                        teacher_group, teacher_actions = ('combat' if combat else 'navigation'), (attack_act,)
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

            # ── Reward shaping (anti-bomb-shyness & anti-kamikaze) ──
            shaped_reward = _shaped_reward(rewards[learning_aid], learning_aid, info,
                                           env.agents[learning_aid].kills, wall_destroy_bonus)
            bombs_this_ep += learning_aid in info["bombs_placed"]
            suicide_this_ep |= learning_aid in info.get("self_blast_deaths", []) or (learning_aid, learning_aid) in info["kills"]
            items_this_ep += len(info["items_collected"].get(learning_aid, []))

            # Wall destruction bonus
            wd = info.get("walls_destroyed", {})
            if learning_aid in wd:
                walls_this_ep += wd[learning_aid]

            # ── Store transition for learning agent ──
            if alive_mask[learning_aid] and log_prob is not None:
                if reward_diagnostics:
                    parts = _reward_components(rewards[learning_aid], learning_aid, info,
                        env.agents[learning_aid].kills, True, wall_destroy_bonus)
                    reward_totals.update(parts)
                    reward_max_error = max(reward_max_error, abs(sum(parts.values()) - shaped_reward))
                log_probs_buf.append(log_prob.squeeze())
                values_buf.append(value.squeeze())
                rewards_buf.append(shaped_reward)
                entropies_buf.append(entropy.squeeze())
                if advantage_estimator == "gae" or rollout is not None:
                    terminal = (dones[learning_aid] or not info["alive"][learning_aid]
                                or info["episode_done"] or t + 1 == max_steps)
                    terminal_buf.append(bool(terminal))
                    if terminal:
                        last_boundary = ("learner_death" if not info["alive"][learning_aid] else
                            "time_limit" if env.step_count >= env.max_steps or t + 1 == max_steps else "match_end")
                if rollout is not None:
                    rollout.append(freeze_rollout_step(obs_t.squeeze(0), actions[learning_aid], mask_t,
                        log_prob, value, dist.probs.squeeze(0), shaped_reward, bool(terminal),
                        last_boundary if terminal else None, teacher_group, teacher_actions))

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
        advantage_metrics = {}
        ratio_metrics = {}
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

            mc_advantages = returns_t - values_t.detach()
            advantages = mc_advantages
            if advantage_estimator == "gae":
                bootstrap = ~torch.tensor(terminal_buf, dtype=torch.bool, device=device)
                continuation = bootstrap.clone()
                continuation[-1] = False
                next_values = torch.cat((values_t.detach()[1:], values_t.new_zeros(1)))
                advantages, deltas = gae_advantages(
                    torch.tensor(rewards_buf, dtype=values_t.dtype, device=device), values_t,
                    next_values, bootstrap, continuation, gamma, gae_lambda)
            raw_advantages = advantages
            normalization_skipped = False
            if normalize_advantage:
                advantages, normalization_skipped = normalize_policy_advantages(advantages)

            policy_loss = -(log_probs_t * advantages).mean()
            value_loss = F.mse_loss(values_t, returns_t)
            entropy_bonus = entropies_t.mean()
            ent_coef_now = entropy_coef * max(0.2, 1.0 - 0.8 * ep / episodes)
            if rollout is not None:
                if multi_update:
                    replay = update_ratio_rollout(net, optimizer, rollout, advantages, returns_t,
                        epochs=update_epochs, clipped=policy_objective == 'ppo', clip_epsilon=clip_epsilon,
                        target_kl=target_kl, value_coef=value_coef, entropy_coef=ent_coef_now, stop_event=stop_event)
                    if replay is None:
                        if save_training_state:
                            restore_state(load_training_state(state_path))
                        break
                else:
                    replay = ratio_rollout_losses(net, rollout, advantages, returns_t)
                policy_loss, value_loss, entropy_bonus = (replay[k] for k in ('policy_loss', 'value_loss', 'entropy'))
                escape_loss, combat_loss, navigation_loss = (replay[k + '_loss'] for k in ('escape', 'combat', 'navigation'))
                entropies_t, ratio_metrics = replay['entropies'], replay['diagnostics']

            # Linearly anneal entropy coefficient: full at start, 20% at end
            loss = policy_loss + value_coef * value_loss - ent_coef_now * entropy_bonus
            loss = loss + 0.5 * escape_loss + 0.5 * combat_loss + 0.2 * navigation_loss
            if advantage_diagnostics and not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training loss")

            if multi_update:
                grad_norm = replay['grad_norm']
            else:
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            if advantage_diagnostics:
                if not torch.isfinite(grad_norm).item():
                    raise FloatingPointError("Non-finite training gradient")
                return_var = returns_t.var(unbiased=False).item()
                advantage_metrics = {
                    "normalize_enabled": normalize_advantage, "normalization_skipped": normalization_skipped,
                    "samples": len(rewards_buf), "raw": _tensor_stats(raw_advantages),
                    "policy": _tensor_stats(advantages), "returns": _tensor_stats(returns_t),
                    "values": _tensor_stats(values_t), "grad_norm_before_clip": grad_norm.item(),
                    "gradient_clipped": grad_norm.item() > 5.0,
                    "explained_variance": 1 - mc_advantages.var(unbiased=False).item() / return_var
                        if return_var >= 1e-8 else None,
                    "entropy_mean": entropy_bonus.item(),
                    "entropy_by_mask_size": {str(size): {"samples": mask_sizes_buf.count(size),
                        "mean": entropies_t.detach()[torch.tensor([n == size for n in mask_sizes_buf],
                            device=device)].mean().item()} for size in sorted(set(mask_sizes_buf))},
                    **({"estimator": "gae", "gae_lambda": gae_lambda, "mc_raw": _tensor_stats(mc_advantages),
                        "td_delta": _tensor_stats(deltas), "last_boundary": last_boundary,
                        "terminal_transitions": sum(terminal_buf),
                        "bootstrap_transitions": int(bootstrap.sum().item()),
                        "continuations": int(continuation.sum().item())}
                       if advantage_estimator == "gae" else {})}
            if not multi_update:
                optimizer.step()
        else:
            loss = torch.zeros(())
            policy_loss = value_loss = loss

        # ── Logging ──
        ep_reward = sum(rewards_buf) if rewards_buf else 0.0
        if reward_diagnostics and not math.isclose(sum(reward_totals.values()), ep_reward, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Episode reward components do not match the learner trajectory")
        reward_history.append(ep_reward)
        alive_final = info["alive"]
        other_alive = [alive_final[j] for j in range(num_agents) if j != learning_aid]
        won = 1 if (alive_final[learning_aid] and not any(other_alive)) else 0
        win_history.append(won)
        bombs_history.append(bombs_this_ep)
        walls_history.append(walls_this_ep)
        kills_this_ep = env.agents[learning_aid].kills
        active_win = int(won and kills_this_ep > 0)
        kills_history.append(kills_this_ep)
        active_win_history.append(active_win)
        suicide_history.append(int(suicide_this_ep))
        combat_history.append((combat_bombs, combat_labels))

        if validation_enabled and (ep % validation_every == 0 or ep == episodes):
            validate_candidate(ep)
            if save_training_state and stop_event is not None and stop_event.is_set():
                restore_state(load_training_state(state_path))
                break

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
            on_episode({"episode": ep, "reward": ep_reward, "win": won,
                        "bombs": bombs_this_ep, "walls": walls_this_ep,
                        "kills": kills_this_ep, "active_win": active_win,
                        "suicide": int(suicide_this_ep),
                        "items": items_this_ep, "learner_steps": len(rewards_buf),
                        "learning_aid": learning_aid, "draw": int(sum(alive_final) != 1),
                        "attack_opportunities": attack_opportunities, "attack_bombs": attack_bombs,
                        "attack_labels": attack_labels, "missing_attack_labels": missing_attack_labels,
                        "combat_labels": combat_labels, "combat_bombs": combat_bombs,
                        "followup_labels": followup_labels, "followup_bombs": followup_bombs,
                        "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                        "escape_loss": float(escape_loss.detach()) if aux_losses_buf else 0.0,
                        "combat_loss": float(combat_loss.detach()) if combat_losses_buf else 0.0,
                        "navigation_loss": float(navigation_loss.detach()) if attack_losses_buf else 0.0,
                        "steps": info["step"], "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0], "best_wr": best_wr,
                        "best_active_wr": best_active_wr, "best_episode": best_episode,
                        "validation": last_validation,
                        "attack_teacher": attack_teacher,
                        "teacher": {**teacher_stats, "rejected": dict(teacher_rejected)},
                        "move_guidance": {**move_guidance, "rejected": dict(move_rejected)},
                        **({"reward_components": dict(reward_totals),
                            "reward_component_max_error": reward_max_error} if reward_diagnostics else {}),
                        **({'ratio_update': ratio_metrics} if rollout is not None else {}),
                        **({"advantage": advantage_metrics} if advantage_diagnostics else {})})

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
        save_state(ep)

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
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wall_destroy_bonus", type=float, default=0.15,
                        help="Reward bonus per wall destroyed (anti-bomb-shyness)")
    parser.add_argument("--snapshot_every", type=int, default=200,
                        help="Save snapshot every N episodes (Phase D/E)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Weights-only warm start; Adam/scheduler/RNG start fresh")
    parser.add_argument("--save_training_state", action="store_true",
                        help="Atomically save Phase C Adam/scheduler/RNG at completed episode boundaries")
    parser.add_argument("--resume_training_state", default="",
                        help="Exact Phase C continuation; requires identical training config/source/runtime")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="Reproducible training seed")
    parser.add_argument("--validation_every", type=int, default=250)
    parser.add_argument("--validation_episodes", type=int, default=48)
    parser.add_argument("--validation_seed", type=int, default=6000)
    parser.add_argument("--attack_teacher", choices=["local", "two_bomb", "local_combat_move"], default="local",
                        help="Attack labels only; the CNN still samples every action through its mask")
    parser.add_argument("--normalize_advantage", action="store_true",
                        help="Normalize policy advantage per learner trajectory; keep MC critic targets")
    parser.add_argument("--advantage_diagnostics", action="store_true")
    parser.add_argument("--advantage_estimator", choices=("mc", "gae"), default="mc",
                        help="Actor advantage only; the critic always uses Monte-Carlo returns")
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--reward_diagnostics", action="store_true",
                        help="Log reward components without changing learner rewards")
    parser.add_argument('--policy_objective', choices=('ac', 'ratio', 'ppo'), default='ac',
                        help='Stored-rollout ratio objective, optionally PPO clipped')
    parser.add_argument('--update_epochs', type=int, default=1)
    parser.add_argument('--clip_epsilon', type=float, default=.2)
    parser.add_argument('--target_kl', type=float, default=0., help='Stop remaining epochs above 1.5 times this KL; 0 disables')
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
            wall_destroy_bonus=args.wall_destroy_bonus,
            snapshot_every=args.snapshot_every,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
            log_every=args.log_every,
            seed=args.seed,
            validation_every=args.validation_every,
            validation_episodes=args.validation_episodes,
            validation_seed=args.validation_seed,
            attack_teacher=args.attack_teacher,
            normalize_advantage=args.normalize_advantage,
            advantage_diagnostics=args.advantage_diagnostics,
            advantage_estimator=args.advantage_estimator,
            gae_lambda=args.gae_lambda,
            reward_diagnostics=args.reward_diagnostics,
            save_training_state=args.save_training_state,
            resume_training_state=args.resume_training_state,
            policy_objective=args.policy_objective,
            update_epochs=args.update_epochs,
            clip_epsilon=args.clip_epsilon,
            target_kl=args.target_kl,
            on_episode=write_metrics,
        )
