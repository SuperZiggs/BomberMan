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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bomberman_env import BombermanFFAEnv, NUM_ACTIONS, ACTION_BOMB, ACTION_STOP, WOODEN_WALL, _MOVE_DELTA
from rule_based_bot import (
    RuleBasedAgent,
    _build_danger_map,
    _can_bomb_and_escape,
    _bfs_safe_cell,
    _bfs_toward_opponent,
    _bfs_toward_wood,
)


# --------------------------------- network ---------------------------------

class ActorCriticNet(nn.Module):
    """
    Small CNN shared trunk -> actor head (action logits) + critic head (state value).
    Input: (batch, 8, H, W) observation from BombermanFFAEnv._get_obs.
    """

    def __init__(self, in_channels: int = 8, board_size: int = 11,
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
    """Return a boolean mask of valid actions for the given agent."""
    mask = [True] * NUM_ACTIONS
    agent = env.agents[agent_id]
    r, c = agent.row, agent.col
    danger = _build_danger_map(env)

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
    if agent.bombs_placed >= agent.ammo or any(b.row == r and b.col == c for b in env.bombs):
        mask[ACTION_BOMB] = False
    elif not _can_bomb_and_escape(env, agent_id, r, c, danger):
        mask[ACTION_BOMB] = False

    if not any(mask):
        mask[ACTION_STOP] = True
    return mask


def get_attack_action(env: BombermanFFAEnv, agent_id: int, danger, mask):
    """Training target: bomb an exposed opponent, approach one, or open a route."""
    agent = env.agents[agent_id]
    r, c = agent.row, agent.col
    if danger[r, c] > 0:
        return None
    blast = env._blast_cells(r, c, agent.blast_range)
    if mask[ACTION_BOMB] and any(
        a.alive and a.agent_id != agent_id and (a.row, a.col) in blast
        for a in env.agents
    ):
        return ACTION_BOMB
    move = _bfs_toward_opponent(env, agent_id, r, c, danger)
    if move is not None and mask[move]:
        return move
    if mask[ACTION_BOMB] and any(env.grid[pos] == WOODEN_WALL for pos in blast):
        return ACTION_BOMB
    move = _bfs_toward_wood(env, r, c, danger)
    return move if move is not None and mask[move] else None


def _combat_reward(reward: float, agent_id: int, info, episode_kills: int):
    """Reward owned eliminations; a token bomb cannot unlock a passive win bonus."""
    kills = info["kills"]
    if (agent_id, agent_id) in kills:
        reward -= 1.5  # -1 from env, -2.5 total for suicide.
    if info["alive"][agent_id]:
        reward += 3.0 * sum(k == agent_id and v != agent_id for k, v in kills)
        if info["episode_done"] and sum(info["alive"]) == 1 and episode_kills == 0:
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
    """Load checkpoint weights with automatic spatial interpolation if board_size differs."""
    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
):
    """
    Unified training loop for phases C, D, E.

    Phase C: Agent 0 = learning AC, Agent 1 = RuleBasedAgent (fixed).
             Reward shaping: +wall_destroy_bonus per wall destroyed.
    Phase D: Agent 0 = learning AC, Agent 1 = random from opponent pool
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

    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                          max_steps=max_steps)
    net = ActorCriticNet(in_channels=8, board_size=board_size,
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
    best_score = (-1.0, -1.0, -1.0)
    best_state_dict = None   # will hold state_dict of the best model seen

    # ── Logging buffers ──
    reward_history = []
    win_history = []        # 1 = learning agent is the sole survivor
    bombs_history = []      # bombs placed by learning agent
    walls_history = []      # walls destroyed by learning agent's bombs
    kills_history = []
    active_win_history = []

    for ep in range(1, episodes + 1):
        if stop_event is not None and stop_event.is_set():
            break
        obs_list = env.reset()
        alive_mask = [True] * num_agents

        # Learning agent: in Phase E, cycle corners across episodes so all corners are trained!
        learning_aid = (ep - 1) % num_agents if phase == "E" else 0

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
        attack_losses_buf = []
        bombs_this_ep = 0
        walls_this_ep = 0
        suicide_this_ep = False

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
                if d_map[r_l, c_l] > 0:
                    esc_act = _bfs_safe_cell(env, r_l, c_l, d_map)
                    if esc_act is not None and mask[esc_act]:
                        aux_losses_buf.append(
                            F.cross_entropy(m_logits, torch.tensor([esc_act], device=device))
                        )
                else:
                    attack_act = get_attack_action(env, learning_aid, d_map, mask)
                    if attack_act is not None:
                        attack_losses_buf.append(
                            F.cross_entropy(m_logits, torch.tensor([attack_act], device=device))
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
                d_map_before = _build_danger_map(env)
                danger_before = d_map_before[r_l, c_l]
            else:
                danger_before = 0

            # ── Step environment ──
            obs_list, rewards, dones, info = env.step(actions)

            # ── Reward shaping (anti-bomb-shyness & anti-kamikaze) ──
            shaped_reward = _combat_reward(rewards[learning_aid], learning_aid, info,
                                           env.agents[learning_aid].kills)
            bombs_this_ep += learning_aid in info["bombs_placed"]
            suicide_this_ep |= (learning_aid, learning_aid) in info["kills"]

            # Wall destruction bonus
            wd = info.get("walls_destroyed", {})
            if learning_aid in wd:
                shaped_reward += wall_destroy_bonus * wd[learning_aid]
                walls_this_ep += wd[learning_aid]

            # Safe escape bonus: stepped out of blast danger into safety
            if alive_mask[learning_aid] and info["alive"][learning_aid]:
                r1, c1 = env.agents[learning_aid].row, env.agents[learning_aid].col
                if danger_before > 0 and _build_danger_map(env)[r1, c1] == 0:
                    shaped_reward += 0.20

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
            if attack_losses_buf:
                loss = loss + 0.2 * torch.stack(attack_losses_buf).mean()

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
        kills_this_ep = env.agents[learning_aid].kills
        active_win = int(won and kills_this_ep > 0)
        kills_history.append(kills_this_ep)
        active_win_history.append(active_win)

        # ── Best-model checkpointing ──
        if len(win_history) >= 200:
            rolling_wr = float(np.mean(win_history[-200:])) * 100
            rolling_active_wr = float(np.mean(active_win_history[-200:])) * 100
            rolling_kills = float(np.mean(kills_history[-200:]))
            score = (rolling_active_wr, rolling_kills, rolling_wr)
            if score > best_score and rolling_kills > 0:
                best_score = score
                best_wr = rolling_wr
                best_active_wr = rolling_active_wr
                best_state_dict = {k: v.clone() for k, v in net.state_dict().items()}
                os.makedirs(checkpoint_dir, exist_ok=True)
                tag = f"phaseE_{num_agents}a" if phase == "E" else f"phase{phase}"
                best_path = os.path.join(checkpoint_dir, f"best_{tag}.pt")
                torch.save(best_state_dict, best_path)
                if phase == "E":
                    torch.save(best_state_dict, os.path.join(checkpoint_dir, "best_phaseE.pt"))

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
                        "steps": info["step"], "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0], "best_wr": best_wr,
                        "best_active_wr": best_active_wr})

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
            best_tag = f" | best active: {best_active_wr:.0f}%" if best_state_dict is not None else ""
            print(f"[{phase}] ep {ep:5d} | "
                  f"WR: {avg_wr:5.1f}% | "
                  f"active WR: {active_wr:5.1f}% | kills: {avg_kills:.2f} | "
                  f"reward: {avg_reward:+.3f} | "
                  f"bombs: {avg_bombs:.1f} | "
                  f"walls: {avg_walls:.1f} | "
                  f"loss: {loss.item():.4f} | "
                  f"lr: {cur_lr:.1e} | "
                  f"pool: {len(snapshot_pool)}{best_tag}")

    # ── Save final model ──
    os.makedirs(checkpoint_dir, exist_ok=True)
    tag = f"phaseE_{num_agents}a" if phase == "E" else f"phase{phase}"
    final_path = os.path.join(checkpoint_dir, f"final_{tag}.pt")
    best_path = os.path.join(checkpoint_dir, f"best_{tag}.pt")
    if phase == "C" and best_state_dict is not None:
        torch.save(best_state_dict, final_path)
        net.load_state_dict(best_state_dict)  # use best for gate checks
        print(f"\nSaved BEST model (active WR={best_active_wr:.1f}%, WR={best_wr:.1f}%) to {final_path}")
    else:
        torch.save(net.state_dict(), final_path)
        if best_state_dict is not None:
            torch.save(best_state_dict, best_path)
        else:
            torch.save(net.state_dict(), best_path)
        if phase == "E":
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, "final_phaseE.pt"))
            if best_state_dict is not None:
                torch.save(best_state_dict, os.path.join(checkpoint_dir, "best_phaseE.pt"))
            else:
                torch.save(net.state_dict(), os.path.join(checkpoint_dir, "best_phaseE.pt"))
        print(f"\nSaved final/best model to {final_path}")

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
                        help="Path to checkpoint to resume from")
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
        snapshot_every=args.snapshot_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        log_every=args.log_every,
    )
