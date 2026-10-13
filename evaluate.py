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

def classify_agent_death(info: dict, agent_id: int, was_alive: bool) -> str | None:
    """Classify a newly dead agent using environment attribution."""
    if not was_alive or info["alive"][agent_id]:
        return None
    if agent_id in info.get("sudden_death_kills", []):
        return "sudden_death"
    kills = info.get("kills", [])
    if any(killer == agent_id and victim == agent_id for killer, victim in kills):
        return "self_bomb"
    if any(killer != agent_id and victim == agent_id for killer, victim in kills):
        return "opponent_bomb"
    return "other"


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
    total_offensive_bombs = [0] * num_agents
    total_pressure_score = [0] * num_agents
    sudden_death_deaths = [0] * num_agents
    self_bomb_deaths = [0] * num_agents
    opponent_bomb_deaths = [0] * num_agents
    other_deaths = [0] * num_agents
    chain_bomb_self_deaths = [0] * num_agents

    for ep in range(num_episodes):
        obs_list = env.reset()
        ep_rewards = [0.0] * num_agents
        ep_bombs = [0] * num_agents

        for t in range(env.max_steps):
            alive_before = [agent.alive for agent in env.agents]
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
            for owner in info.get("offensive_bomb_owners", []):
                total_offensive_bombs[owner] += 1
            for owner, pressure_score in info.get("bomb_pressure_scores", []):
                total_pressure_score[owner] += pressure_score
            for agent_id in range(num_agents):
                death_cause = classify_agent_death(info, agent_id, alive_before[agent_id])
                if death_cause == "sudden_death":
                    sudden_death_deaths[agent_id] += 1
                elif death_cause == "self_bomb":
                    self_bomb_deaths[agent_id] += 1
                    if any(
                        outcome.get("chain_reaction", False)
                        and outcome.get("owner") == agent_id
                        and agent_id in outcome.get("victims", [])
                        for outcome in info.get("bomb_outcomes", [])
                    ):
                        chain_bomb_self_deaths[agent_id] += 1
                elif death_cause == "opponent_bomb":
                    opponent_bomb_deaths[agent_id] += 1
                elif death_cause == "other":
                    other_deaths[agent_id] += 1

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
        "avg_offensive_bombs": [b / num_episodes for b in total_offensive_bombs],
        "avg_pressure_score": [score / num_episodes for score in total_pressure_score],
        "sudden_death_deaths": sudden_death_deaths,
        "self_bomb_deaths": self_bomb_deaths,
        "opponent_bomb_deaths": opponent_bomb_deaths,
        "other_deaths": other_deaths,
        "chain_bomb_self_deaths": chain_bomb_self_deaths,
    }

    if verbose:
        print(f"\n{'-' * 50}")
        print(f"Match Results ({num_episodes} episodes)")
        print(f"{'-' * 50}")
        for i in range(num_agents):
            print(f"  Agent {i}: WR={stats['win_rates'][i]:.1f}%, "
                  f"Reward={stats['avg_reward'][i]:.2f}, "
                  f"Bombs={stats['avg_bombs'][i]:.1f}, "
                  f"Offensive={stats['avg_offensive_bombs'][i]:.1f}, "
                  f"Pressure={stats['avg_pressure_score'][i]:.1f}, "
                  f"Kills={stats['avg_kills'][i]:.1f}, "
                  f"Deaths(sd/self/opp/other)="
                  f"{stats['sudden_death_deaths'][i]}/"
                  f"{stats['self_bomb_deaths'][i]}/"
                  f"{stats['opponent_bomb_deaths'][i]}/"
                  f"{stats['other_deaths'][i]}, "
                  f"Chain-self={stats['chain_bomb_self_deaths'][i]}")
        print(f"  Draws: {draws} ({draws/num_episodes*100:.1f}%)")
        print(f"  Avg episode length: {stats['avg_steps']:.1f}")

    return stats


def combine_side_swapped_stats(first_side: dict, second_side: dict) -> dict:
    """Combine A-vs-B and B-vs-A statistics into an agent-order invariant result."""
    first_episodes = first_side["num_episodes"]
    second_episodes = second_side["num_episodes"]
    total_episodes = first_episodes + second_episodes

    def average(metric: str, first_index: int, second_index: int) -> float:
        return (
            first_side[metric][first_index] * first_episodes
            + second_side[metric][second_index] * second_episodes
        ) / max(total_episodes, 1)

    wins = [
        first_side["wins"][0] + second_side["wins"][1],
        first_side["wins"][1] + second_side["wins"][0],
    ]
    return {
        "num_episodes": total_episodes,
        "wins": wins,
        "draws": first_side["draws"] + second_side["draws"],
        "win_rates": [win / max(total_episodes, 1) * 100 for win in wins],
        "avg_reward": [average("avg_reward", 0, 1), average("avg_reward", 1, 0)],
        "avg_bombs": [average("avg_bombs", 0, 1), average("avg_bombs", 1, 0)],
        "avg_kills": [average("avg_kills", 0, 1), average("avg_kills", 1, 0)],
        "avg_offensive_bombs": [
            average("avg_offensive_bombs", 0, 1),
            average("avg_offensive_bombs", 1, 0),
        ],
        "avg_pressure_score": [
            average("avg_pressure_score", 0, 1),
            average("avg_pressure_score", 1, 0),
        ],
        "sudden_death_deaths": [
            first_side["sudden_death_deaths"][0] + second_side["sudden_death_deaths"][1],
            first_side["sudden_death_deaths"][1] + second_side["sudden_death_deaths"][0],
        ],
        "self_bomb_deaths": [
            first_side["self_bomb_deaths"][0] + second_side["self_bomb_deaths"][1],
            first_side["self_bomb_deaths"][1] + second_side["self_bomb_deaths"][0],
        ],
        "opponent_bomb_deaths": [
            first_side["opponent_bomb_deaths"][0] + second_side["opponent_bomb_deaths"][1],
            first_side["opponent_bomb_deaths"][1] + second_side["opponent_bomb_deaths"][0],
        ],
        "other_deaths": [
            first_side["other_deaths"][0] + second_side["other_deaths"][1],
            first_side["other_deaths"][1] + second_side["other_deaths"][0],
        ],
        "chain_bomb_self_deaths": [
            first_side["chain_bomb_self_deaths"][0] + second_side["chain_bomb_self_deaths"][1],
            first_side["chain_bomb_self_deaths"][1] + second_side["chain_bomb_self_deaths"][0],
        ],
    }


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

    from train_actor_critic import (
        ActorCriticNet,
        get_action_mask,
        obs_to_tensor,
        smart_load_weights,
    )

    net = ActorCriticNet(board_size=board_size).to(device)
    smart_load_weights(net, model_path, device)
    net.eval()

    @torch.no_grad()
    def fn(env, obs_list, agent_id):
        obs_t = obs_to_tensor(obs_list[agent_id], device)
        logits, _ = net(obs_t)
        mask = get_action_mask(env, agent_id)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        dist = torch.distributions.Categorical(logits=logits.masked_fill(~mask_t, -1e9))
        return dist.sample().item()
    return fn


# ────────────────────── tournament mode ───────────────────────────────────

def tournament(board_size: int = 11, num_agents: int = 2,
               num_episodes: int = 100, ac_model_path: str = "",
               device: str = "cpu", sudden_death: bool = False,
               sudden_death_start: int = 200,
               sudden_death_interval: int = 20):
    """
    Round-robin tournament giữa Random, RuleBot, và AC (nếu có model).
    """
    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                          max_steps=300, sudden_death=sudden_death,
                          sudden_death_start=sudden_death_start,
                          sudden_death_interval=sudden_death_interval)

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
    print(f"  Sudden death: {'on' if sudden_death else 'off'}")
    print(f"{'=' * 60}")

    if num_agents == 2:
        # Pairwise matchups, with every agent evaluated from both spawn sides.
        results = {}
        for i, name_a in enumerate(agent_names):
            for j, name_b in enumerate(agent_names):
                if i >= j:
                    continue
                print(f"\n  {name_a} vs {name_b} (sides swapped):")
                first_side = run_match(
                    env, [agents[name_a], agents[name_b]], num_episodes, verbose=False
                )
                second_side = run_match(
                    env, [agents[name_b], agents[name_a]], num_episodes, verbose=False
                )
                stats = combine_side_swapped_stats(first_side, second_side)
                for agent_index, name in enumerate((name_a, name_b)):
                    print(f"  {name}: WR={stats['win_rates'][agent_index]:.1f}%, "
                          f"Reward={stats['avg_reward'][agent_index]:.2f}, "
                          f"Bombs={stats['avg_bombs'][agent_index]:.1f}, "
                          f"Offensive={stats['avg_offensive_bombs'][agent_index]:.1f}, "
                          f"Pressure={stats['avg_pressure_score'][agent_index]:.1f}, "
                          f"Kills={stats['avg_kills'][agent_index]:.1f}, "
                          f"Deaths(sd/self/opp/other)="
                          f"{stats['sudden_death_deaths'][agent_index]}/"
                          f"{stats['self_bomb_deaths'][agent_index]}/"
                          f"{stats['opponent_bomb_deaths'][agent_index]}/"
                          f"{stats['other_deaths'][agent_index]}, "
                          f"Chain-self={stats['chain_bomb_self_deaths'][agent_index]}")
                print(f"  Draws: {stats['draws']} ({stats['draws']/stats['num_episodes']*100:.1f}%)")
                results[f"{name_a}_vs_{name_b}"] = stats
        return results
    else:
        # FFA benchmark: heuristic bots, then the learned shared policy when supplied.
        print(f"\n  All RuleBot FFA ({num_agents} agents):")
        fns = [make_rule_based_fn(i) for i in range(num_agents)]
        results = {"all_rulebots": run_match(env, fns, num_episodes, verbose=True)}
        if "ActorCritic" in agents:
            print(f"\n  All ActorCritic FFA ({num_agents} agents):")
            results["all_actor_critic"] = run_match(
                env, [agents["ActorCritic"]] * num_agents, num_episodes, verbose=True
            )
        return results


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
    parser.add_argument("--sudden_death", action="store_true")
    parser.add_argument("--sudden_death_start", type=int, default=200)
    parser.add_argument("--sudden_death_interval", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "tournament":
        tournament(args.board_size, args.num_agents, args.num_episodes,
                   args.ac_model, args.device, args.sudden_death,
                   args.sudden_death_start, args.sudden_death_interval)
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
