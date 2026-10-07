"""
evaluate.py

Giai đoạn F — Đánh giá & so sánh tất cả phương pháp.

Chức năng:
1. So sánh learning curves (reward theo episode) giữa:
   - Random baseline
   - RuleBasedAgent
   - Q-learning linear (Phase B)
   - Actor-Critic 1v1 (Phase C)
   - Actor-Critic self-play FFA (Phase E)

2. Round-robin tournament giữa các agent.

3. Ablation: có/không reward shaping chống bomb shyness.

4. Xuất biểu đồ (matplotlib) và số liệu.

Usage:
    python evaluate.py --mode tournament --board_size 11 --num_agents 4
    python evaluate.py --mode curves
    python evaluate.py --mode ablation
"""

from __future__ import annotations

import argparse
import os
import json
import numpy as np

from bomberman_env import BombermanFFAEnv, NUM_ACTIONS, ACTION_BOMB
from rule_based_bot import RuleBasedAgent

# Lazy import torch — only needed when evaluating AC models.
_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    pass


# ────────────────────── random agent baseline ─────────────────────────────

class RandomAgent:
    """Chọn action ngẫu nhiên."""
    def select_action(self, *args, **kwargs) -> int:
        return int(np.random.randint(NUM_ACTIONS))


# ────────────────────── evaluation helpers ────────────────────────────────

def run_match(env: BombermanFFAEnv, agent_fns: list, num_episodes: int = 100,
              verbose: bool = False):
    """
    Chạy num_episodes, mỗi episode gọi agent_fns[i](env, obs_list, agent_id)
    để lấy action.

    Returns dict với thống kê.
    """
    num_agents = env.num_agents
    wins = [0] * num_agents
    draws = 0
    total_rewards = [0.0] * num_agents
    total_steps = 0
    total_bombs = [0] * num_agents
    total_kills = [0] * num_agents

    for ep in range(num_episodes):
        obs_list = env.reset()
        ep_rewards = [0.0] * num_agents
        ep_bombs = [0] * num_agents

        for t in range(env.max_steps):
            actions = []
            for i in range(num_agents):
                if not env.agents[i].alive:
                    actions.append(0)
                    continue
                a = agent_fns[i](env, obs_list, i)
                if a == ACTION_BOMB:
                    ep_bombs[i] += 1
                actions.append(a)

            obs_list, rewards, dones, info = env.step(actions)
            for i in range(num_agents):
                ep_rewards[i] += rewards[i]

            if info["episode_done"]:
                break

        total_steps += info["step"]
        for i in range(num_agents):
            total_rewards[i] += ep_rewards[i]
            total_bombs[i] += ep_bombs[i]
            total_kills[i] += env.agents[i].kills

        alive = info["alive"]
        survivors = [i for i, a in enumerate(alive) if a]
        if len(survivors) == 1:
            wins[survivors[0]] += 1
        else:
            draws += 1

    stats = {
        "num_episodes": num_episodes,
        "wins": wins,
        "draws": draws,
        "win_rates": [w / num_episodes * 100 for w in wins],
        "avg_reward": [r / num_episodes for r in total_rewards],
        "avg_steps": total_steps / num_episodes,
        "avg_bombs": [b / num_episodes for b in total_bombs],
        "avg_kills": [k / num_episodes for k in total_kills],
    }

    if verbose:
        print(f"\n{'─' * 50}")
        print(f"Match Results ({num_episodes} episodes)")
        print(f"{'─' * 50}")
        for i in range(num_agents):
            print(f"  Agent {i}: WR={stats['win_rates'][i]:.1f}%, "
                  f"Reward={stats['avg_reward'][i]:.2f}, "
                  f"Bombs={stats['avg_bombs'][i]:.1f}, "
                  f"Kills={stats['avg_kills'][i]:.1f}")
        print(f"  Draws: {draws} ({draws/num_episodes*100:.1f}%)")
        print(f"  Avg episode length: {stats['avg_steps']:.1f}")

    return stats


# ────────────────────── agent function factories ──────────────────────────

def make_random_fn():
    """Create a random agent function."""
    def fn(env, obs_list, agent_id):
        return int(np.random.randint(NUM_ACTIONS))
    return fn


def make_rule_based_fn(agent_id: int):
    """Create a rule-based agent function."""
    bot = RuleBasedAgent(agent_id)
    def fn(env, obs_list, aid):
        bot.agent_id = aid
        return bot.select_action(env)
    return fn


def make_ac_fn(model_path: str, board_size: int, device="cpu"):
    """Create an Actor-Critic agent function from a saved checkpoint."""
    if not _torch_available:
        raise RuntimeError("PyTorch not installed, cannot load AC model.")

    from train_actor_critic import ActorCriticNet, obs_to_tensor

    net = ActorCriticNet(board_size=board_size).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device,
                                   weights_only=True))
    net.eval()

    @torch.no_grad()
    def fn(env, obs_list, agent_id):
        obs_t = obs_to_tensor(obs_list[agent_id], device)
        logits, _ = net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()
    return fn


# ────────────────────── tournament mode ───────────────────────────────────

def tournament(board_size: int = 11, num_agents: int = 2,
               num_episodes: int = 100, ac_model_path: str = "",
               device: str = "cpu"):
    """
    Round-robin tournament giữa Random, RuleBot, và AC (nếu có model).
    """
    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                          max_steps=300)

    agents = {
        "Random": make_random_fn(),
        "RuleBot": make_rule_based_fn(0),
    }
    if ac_model_path and os.path.exists(ac_model_path):
        agents["ActorCritic"] = make_ac_fn(ac_model_path, board_size, device)

    agent_names = list(agents.keys())
    print(f"\n{'=' * 60}")
    print(f"Tournament: {agent_names}")
    print(f"  Board: {board_size}x{board_size}, {num_agents} agents")
    print(f"  Episodes per matchup: {num_episodes}")
    print(f"{'=' * 60}")

    if num_agents == 2:
        # Pairwise matchups.
        results = {}
        for i, name_a in enumerate(agent_names):
            for j, name_b in enumerate(agent_names):
                if i >= j:
                    continue
                print(f"\n  {name_a} vs {name_b}:")
                agent_fns = [agents[name_a], agents[name_b]]
                stats = run_match(env, agent_fns, num_episodes, verbose=True)
                results[f"{name_a}_vs_{name_b}"] = stats
        return results
    else:
        # FFA with all same type or mixed.
        print(f"\n  All RuleBot FFA ({num_agents} agents):")
        fns = [make_rule_based_fn(i) for i in range(num_agents)]
        stats = run_match(env, fns, num_episodes, verbose=True)
        return {"all_rulebots": stats}


# ────────────────────── learning curves (from saved logs) ─────────────────

def plot_curves(log_files: dict[str, str], output_path: str = "learning_curves.png"):
    """
    Plot learning curves from saved log files.
    Each log file is a JSON with {"rewards": [...], "win_rates": [...]}.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, cannot plot curves.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, path in log_files.items():
        if not os.path.exists(path):
            print(f"  Skipping {label}: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)

        if "rewards" in data:
            rewards = data["rewards"]
            # Smooth with rolling average
            window = max(50, len(rewards) // 20)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(smoothed, label=label, alpha=0.8)

        if "win_rates" in data:
            wrs = data["win_rates"]
            window = max(50, len(wrs) // 20)
            smoothed = np.convolve(wrs, np.ones(window)/window, mode='valid')
            ax2.plot(smoothed, label=label, alpha=0.8)

    ax1.set_title("Average Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Win Rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning curves to {output_path}")
    plt.close()


# ────────────────────── random baseline evaluation ────────────────────────

def eval_random_baseline(board_size: int = 11, num_agents: int = 2,
                         num_episodes: int = 100):
    """Evaluate random agents as a baseline."""
    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                          max_steps=300)
    fns = [make_random_fn() for _ in range(num_agents)]
    print(f"\nRandom Baseline ({num_agents} random agents, {num_episodes} episodes):")
    stats = run_match(env, fns, num_episodes, verbose=True)
    return stats


def eval_rulebots(board_size: int = 11, num_agents: int = 2,
                  num_episodes: int = 100):
    """Evaluate rule-based bots."""
    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                          max_steps=300)
    fns = [make_rule_based_fn(i) for i in range(num_agents)]
    print(f"\nRuleBot Baseline ({num_agents} rule-bots, {num_episodes} episodes):")
    stats = run_match(env, fns, num_episodes, verbose=True)
    return stats


# ────────────────────── ablation study ────────────────────────────────────

def ablation_bomb_shyness(board_size: int = 9, episodes: int = 1000,
                          device: str = "cpu"):
    """
    So sánh training có/không reward shaping chống bomb shyness.
    Chạy 2 lần training Phase C, 1 với wall_destroy_bonus=0.08, 1 với =0.
    So sánh bombs/episode và win rate.
    """
    if not _torch_available:
        print("PyTorch not installed, cannot run ablation.")
        return

    from train_actor_critic import train as ac_train

    print(f"\n{'=' * 60}")
    print("Ablation Study: Bomb Shyness Reward Shaping")
    print(f"{'=' * 60}")

    # With reward shaping
    print("\n--- WITH reward shaping (wall_destroy_bonus=0.08) ---")
    _, rewards_with, wins_with = ac_train(
        phase="C", episodes=episodes, board_size=board_size,
        wall_destroy_bonus=0.08, log_every=200, device=device,
        checkpoint_dir="checkpoints_ablation_with"
    )

    # Without reward shaping
    print("\n--- WITHOUT reward shaping (wall_destroy_bonus=0.0) ---")
    _, rewards_without, wins_without = ac_train(
        phase="C", episodes=episodes, board_size=board_size,
        wall_destroy_bonus=0.0, log_every=200, device=device,
        checkpoint_dir="checkpoints_ablation_without"
    )

    # Compare
    last_n = min(200, episodes)
    wr_with = np.mean(wins_with[-last_n:]) * 100
    wr_without = np.mean(wins_without[-last_n:]) * 100
    print(f"\n--- Ablation Results ---")
    print(f"  With shaping:    WR={wr_with:.1f}%")
    print(f"  Without shaping: WR={wr_without:.1f}%")
    print(f"  Difference: {wr_with - wr_without:+.1f}%")


# ────────────────────── CLI ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase F — Evaluation & comparison")
    parser.add_argument("--mode", type=str, default="tournament",
                        choices=["tournament", "random", "rulebots",
                                 "curves", "ablation"],
                        help="Evaluation mode")
    parser.add_argument("--board_size", type=int, default=11)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--ac_model", type=str, default="",
                        help="Path to Actor-Critic model checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.mode == "tournament":
        tournament(args.board_size, args.num_agents, args.num_episodes,
                   args.ac_model, args.device)
    elif args.mode == "random":
        eval_random_baseline(args.board_size, args.num_agents,
                             args.num_episodes)
    elif args.mode == "rulebots":
        eval_rulebots(args.board_size, args.num_agents, args.num_episodes)
    elif args.mode == "curves":
        # Example: provide log files
        logs = {
            "Q-learning": "logs/qlearning.json",
            "AC Phase C": "logs/phase_c.json",
            "AC Phase E": "logs/phase_e.json",
        }
        plot_curves(logs)
    elif args.mode == "ablation":
        ablation_bomb_shyness(board_size=args.board_size, device=args.device)
