"""
train_qlearning_baseline.py

Giai đoạn B — Q-learning với linear function approximation trên bàn cờ nhỏ
(board_size=7, num_agents=2).

Agent Q-learning train chống lại RuleBasedAgent.  Feature construction thủ công
từ 8 kênh đầu của observation (đúng tinh thần bài 3.5 — Feature Construction for
Linear Methods):

  - Khoảng cách Manhattan tới tường gỗ gần nhất
  - Khoảng cách Manhattan tới bom gần nhất
  - Khoảng cách Manhattan tới đối thủ
  - Số ô an toàn (không bom/lửa/tường) trong bán kính 2
  - Còn ammo hay không (bool → 0/1)
  - Đang ở trong vùng nguy hiểm hay không
  - Số tường gỗ xung quanh (bán kính 2)
  - Số ô lửa xung quanh (bán kính 2)
  - Có power-up gần không

Trọng số Q(s,a) = w_a · φ(s), cập nhật theo TD error (semi-gradient Q-learning).
ε-greedy exploration, ε decay dần.

Log win rate vs RuleBasedAgent mỗi 100 episode.
"""

from __future__ import annotations

import argparse
import numpy as np
from collections import deque

from bomberman_env import (
    BombermanFFAEnv,
    NUM_ACTIONS,
    ACTION_STOP,
    ACTION_BOMB,
    EMPTY,
    RIGID_WALL,
    WOODEN_WALL,
)
from rule_based_bot import RuleBasedAgent


# ─────────────────────── feature extraction ───────────────────────────────

def _manhattan_to_nearest(obs_channel: np.ndarray, r: int, c: int) -> float:
    """
    Tìm khoảng cách Manhattan nhỏ nhất từ (r,c) tới ô có giá trị > 0
    trong obs_channel.  Trả về giá trị chuẩn hóa (/ board_size*2).
    Nếu không tìm thấy → trả về 1.0 (xa nhất).
    """
    h, w = obs_channel.shape
    positions = np.argwhere(obs_channel > 0)
    if len(positions) == 0:
        return 1.0
    dists = np.abs(positions[:, 0] - r) + np.abs(positions[:, 1] - c)
    return float(np.min(dists)) / (h + w)


def _count_in_radius(obs_channel: np.ndarray, r: int, c: int,
                     radius: int = 2) -> float:
    """Đếm số ô có giá trị > 0 trong bán kính Manhattan, chuẩn hóa."""
    h, w = obs_channel.shape
    count = 0
    total = 0
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if abs(dr) + abs(dc) > radius:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                total += 1
                if obs_channel[rr, cc] > 0:
                    count += 1
    return count / max(total, 1)


def extract_features(obs: np.ndarray, env: BombermanFFAEnv,
                     agent_id: int) -> np.ndarray:
    """
    Trich xuat feature vector tu 8 kenh dau cua observation cho agent.
    Tra ve np.ndarray shape (num_features,).
    """
    agent = env.agents[agent_id]
    r, c = agent.row, agent.col
    h, w = obs.shape[1], obs.shape[2]

    # Cac kenh obs: 0=rigid, 1=wood, 2=bomb, 3=flame, 4=ammo_pu, 5=range_pu, 6=self, 7=opp
    features = []

    # 1. Khoang cach Manhattan toi tuong go gan nhat
    features.append(_manhattan_to_nearest(obs[1], r, c))

    # 2. Khoang cach Manhattan toi bom gan nhat
    features.append(_manhattan_to_nearest(obs[2], r, c))

    # 3. Khoang cach Manhattan toi doi thu gan nhat
    features.append(_manhattan_to_nearest(obs[7], r, c))

    # 4. So o an toan (empty, khong bom, khong lua) trong ban kinh 2
    safe_map = np.ones((h, w), dtype=np.float32)
    safe_map -= obs[0]  # tru rigid
    safe_map -= obs[1]  # tru wood
    safe_map[obs[2] > 0] = 0  # tru bomb
    safe_map[obs[3] > 0] = 0  # tru flame
    safe_map = np.clip(safe_map, 0, 1)
    features.append(_count_in_radius(safe_map, r, c, radius=2))

    # 5. Con ammo hay khong
    features.append(1.0 if agent.bombs_placed < agent.ammo else 0.0)

    # 6. Dang o trong vung nguy hiem (bom/lua)
    in_danger = 1.0 if (obs[2, r, c] > 0 or obs[3, r, c] > 0) else 0.0
    features.append(in_danger)

    # 7. So tuong go xung quanh (ban kinh 2) - chuan hoa
    features.append(_count_in_radius(obs[1], r, c, radius=2))

    # 8. So o lua xung quanh (ban kinh 2) - chuan hoa
    features.append(_count_in_radius(obs[3], r, c, radius=2))

    # 9. Khoang cach toi power-up gan nhat
    powerup_map = obs[4] + obs[5]
    features.append(_manhattan_to_nearest(powerup_map, r, c))

    # 10. So o ke (4 huong) la empty (co the di chuyen) - do linh hoat
    adj_empty = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            if obs[0, rr, cc] == 0 and obs[1, rr, cc] == 0 and obs[2, rr, cc] == 0:
                adj_empty += 1
    features.append(adj_empty / 4.0)

    # 11. So tuong go ke truc tiep (4 huong) - co hoi dat bom
    adj_wood = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and obs[1, rr, cc] > 0:
            adj_wood += 1
    features.append(adj_wood / 4.0)

    # 12. Khoang cach toi trung tam ban co (chuan hoa)
    center_r, center_c = h // 2, w // 2
    dist_center = (abs(r - center_r) + abs(c - center_c)) / (h + w)
    features.append(dist_center)

    # 13. Muc do nguy hiem bom (timer thap = nguy hiem hon)
    bomb_danger = 0.0
    if obs[2, r, c] > 0:
        bomb_danger = 1.0  # dang dung tren bom
    else:
        # Kiem tra cac o ke co bom khong
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and obs[2, rr, cc] > 0:
                bomb_danger = max(bomb_danger, 1.0 - obs[2, rr, cc])  # timer thap -> danger cao
    features.append(bomb_danger)

    # 14. Co duong thoat khong (so o an toan ke can)
    escape_routes = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            if safe_map[rr, cc] > 0:
                escape_routes += 1
    features.append(escape_routes / 4.0)

    # 15. So bom hien tai tren ban co
    features.append(float(np.sum(obs[2] > 0)) / (h * w))

    # 16. Bias
    features.append(1.0)

    return np.array(features, dtype=np.float32)


NUM_FEATURES = 16  # khop voi so feature o tren


# ─────────────────────── Q-learning agent ─────────────────────────────────

class LinearQAgent:
    """
    Q-learning agent với linear function approximation.
    Q(s, a) = w_a · φ(s)  cho mỗi action a.
    """

    def __init__(self, num_features: int = NUM_FEATURES,
                 num_actions: int = NUM_ACTIONS,
                 lr: float = 0.01,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.9995):
        self.num_features = num_features
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Trọng số: (num_actions, num_features)
        self.weights = np.zeros((num_actions, num_features), dtype=np.float64)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        """Trả về Q(s, a) cho tất cả action."""
        return self.weights @ features  # shape (num_actions,)

    def select_action(self, features: np.ndarray) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        q = self.q_values(features)
        return int(np.argmax(q))

    def update(self, features: np.ndarray, action: int, reward: float,
               next_features: np.ndarray, done: bool):
        """Semi-gradient TD update: w_a += lr * δ * φ(s)."""
        q_current = self.weights[action] @ features
        if done:
            td_target = reward
        else:
            q_next = np.max(self.q_values(next_features))
            td_target = reward + self.gamma * q_next
        td_error = td_target - q_current
        self.weights[action] += self.lr * td_error * features

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ─────────────────────── training loop ────────────────────────────────────

def train_qlearning(
    episodes: int = 5000,
    board_size: int = 7,
    max_steps: int = 200,
    lr: float = 0.005,
    gamma: float = 0.99,
    log_every: int = 100,
    seed: int = 42,
):
    """
    Train LinearQAgent (agent 0) vs RuleBasedAgent (agent 1) trên bàn nhỏ.
    """
    env = BombermanFFAEnv(board_size=board_size, num_agents=2,
                          max_steps=max_steps, seed=seed)
    q_agent = LinearQAgent(num_features=NUM_FEATURES, lr=lr, gamma=gamma)
    rule_bot = RuleBasedAgent(agent_id=1)

    win_history = []          # 1 = Q-agent thắng, 0 = thua/hòa
    reward_history = []       # tổng reward mỗi episode
    bombs_history = []        # số bom Q-agent đặt mỗi episode

    for ep in range(1, episodes + 1):
        obs_list = env.reset()
        features = extract_features(obs_list[0], env, agent_id=0)
        ep_reward = 0.0
        bombs_placed = 0

        for t in range(max_steps):
            # Q-agent chọn action
            action_q = q_agent.select_action(features)
            if action_q == ACTION_BOMB:
                bombs_placed += 1

            # Rule-based bot chọn action
            action_rb = rule_bot.select_action(env)

            actions = [action_q, action_rb]
            obs_list, rewards, dones, info = env.step(actions)

            reward_q = rewards[0]

            # Reward shaping: wall destruction bonus (chong bomb shyness)
            wd = info.get("walls_destroyed", {})
            if 0 in wd:
                reward_q += 0.1 * wd[0]

            ep_reward += reward_q

            if info["episode_done"] or dones[0]:
                q_agent.update(features, action_q, reward_q, features, done=True)
                break
            else:
                next_features = extract_features(obs_list[0], env, agent_id=0)
                q_agent.update(features, action_q, reward_q, next_features, done=False)
                features = next_features

        q_agent.decay_epsilon()

        # Ghi nhận kết quả
        alive = info["alive"]
        won = 1 if (alive[0] and not alive[1]) else 0
        win_history.append(won)
        reward_history.append(ep_reward)
        bombs_history.append(bombs_placed)

        if ep % log_every == 0:
            recent_wr = np.mean(win_history[-log_every:]) * 100
            recent_reward = np.mean(reward_history[-log_every:])
            recent_bombs = np.mean(bombs_history[-log_every:])
            print(f"Episode {ep:5d} | "
                  f"WinRate(last {log_every}): {recent_wr:5.1f}% | "
                  f"AvgReward: {recent_reward:+.3f} | "
                  f"AvgBombs: {recent_bombs:.1f} | "
                  f"eps: {q_agent.epsilon:.3f}")

    # ── Final report ──
    print(f"\n{'=' * 60}")
    print(f"Q-learning Linear FA — Training Complete ({episodes} episodes)")
    print(f"  Board: {board_size}x{board_size}, 2 agents (Q vs RuleBot)")
    print(f"{'=' * 60}")

    # Win rate over last 500 episodes
    last_n = min(500, len(win_history))
    final_wr = np.mean(win_history[-last_n:]) * 100
    print(f"  Final win rate (last {last_n}): {final_wr:.1f}%")

    # Gate check
    print(f"\n--- Gate Checks ---")
    # Check if learning curve is trending upward
    if len(win_history) >= 200:
        first_quarter = np.mean(win_history[:len(win_history)//4]) * 100
        last_quarter = np.mean(win_history[-len(win_history)//4:]) * 100
        trend_up = last_quarter > first_quarter
        print(f"  Learning trend: first 25% WR={first_quarter:.1f}%, "
              f"last 25% WR={last_quarter:.1f}% ",
              end="")
        print("[PASS] (upward trend)" if trend_up else "[FAIL] (no clear improvement)")
    print(f"  Final WR >= 50%: {final_wr:.1f}% ",
          end="")
    print("[PASS]" if final_wr >= 50 else "[FAIL]")

    return q_agent, win_history, reward_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--board_size", type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_qlearning(
        episodes=args.episodes,
        board_size=args.board_size,
        max_steps=args.max_steps,
        lr=args.lr,
        seed=args.seed,
    )
