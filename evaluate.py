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
from copy import deepcopy
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

EVALUATION_SCHEMA_VERSION = 1


def _checkpoint_identity(path: str) -> dict:
    resolved = os.path.normcase(os.path.abspath(os.path.realpath(path)))
    stat = os.stat(resolved)
    return {
        "path": resolved,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_write_json(path: str, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2)
    target = os.path.abspath(path)
    temporary = f"{target}.tmp"
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _validate_ordered_games(games: list[dict], config: dict) -> None:
    requested = config["num_episodes"]
    if len(games) > requested:
        raise ValueError("completed game count exceeds requested count")

    required_fields = {
        "seed", "aid", "win", "active_win", "sudden_death_win",
        "passive_win", "suicide", "sudden_death_death", "draw",
        "kills", "bombs", "items", "steps",
    }
    seen = set()
    for index, game in enumerate(games):
        if not isinstance(game, dict):
            raise ValueError(f"game record at index {index} must be an object")
        missing = required_fields.difference(game)
        if missing:
            raise ValueError(f"game record at index {index} is missing fields: {sorted(missing)}")
        identity = (game.get("seed"), game.get("aid"))
        if identity in seen:
            raise ValueError("game records must be unique; duplicate record found")
        seen.add(identity)
        expected = (
            config["seed"] + index // config["num_agents"],
            index % config["num_agents"],
        )
        if identity != expected:
            raise ValueError(
                f"game records are out of order at index {index}: "
                f"expected seed/aid {expected}, got {identity}"
            )


def _load_partial_games(path: str, expected: dict) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("schema_version") != expected["schema_version"]:
        raise ValueError("partial schema version mismatch")
    if payload.get("checkpoint") != expected["checkpoint"]:
        raise ValueError("partial checkpoint identity mismatch")
    if payload.get("config") != expected["config"]:
        raise ValueError("partial evaluation config mismatch")
    if payload.get("requested_count") != expected["requested_count"]:
        raise ValueError("partial requested count mismatch")

    games = payload.get("games")
    if not isinstance(games, list):
        raise ValueError("partial games must be a list")
    if payload.get("completed_count") != len(games):
        raise ValueError("partial completed count mismatch")
    _validate_ordered_games(games, expected["config"])
    return deepcopy(games)


def _partial_path_for(output_path: str) -> str:
    root, extension = os.path.splitext(output_path)
    return f"{root}.partial{extension or '.json'}"


def _partial_payload(expected: dict, games: list[dict]) -> dict:
    return {
        **expected,
        "games": deepcopy(games),
        "completed_count": len(games),
    }

def run_match(env: BombermanFFAEnv, agent_fns: list, num_episodes: int = 100,
              verbose: bool = False):
    """
    Chạy num_episodes, mỗi episode gọi agent_fns[i](env, obs_list, agent_id)
    để lấy action.

    Returns dict với thống kê.
    """
    num_agents = env.num_agents
    wins = [0] * num_agents
    active_wins = [0] * num_agents
    draws = 0
    total_rewards = [0.0] * num_agents
    total_steps = 0
    total_bombs = [0] * num_agents
    total_kills = [0] * num_agents
    total_items = [0] * num_agents
    suicides = [0] * num_agents
    sudden_death_wins = [0] * num_agents
    sudden_death_deaths = [0] * num_agents
    passive_wins = [0] * num_agents

    for ep in range(num_episodes):
        obs_list = env.reset()
        ep_rewards = [0.0] * num_agents
        ep_bombs = [0] * num_agents
        ep_sudden_death_deaths = set()

        for t in range(env.max_steps):
            actions = []
            for i in range(num_agents):
                if not env.agents[i].alive:
                    actions.append(0)
                    continue
                a = agent_fns[i](env, obs_list, i)
                actions.append(a)

            obs_list, rewards, dones, info = env.step(actions)
            for i in info["bombs_placed"]:
                ep_bombs[i] += 1
            own_deaths = set(info.get("self_blast_deaths", []))
            own_deaths.update(v for k, v in info["kills"] if k == v)
            for victim in own_deaths:
                suicides[victim] += 1
            ep_sudden_death_deaths.update(info.get("sudden_death_deaths", []))
            for aid, items in info["items_collected"].items():
                total_items[aid] += len(items)
            for i in range(num_agents):
                ep_rewards[i] += rewards[i]

            if info["episode_done"]:
                break

        total_steps += info["step"]
        for i in range(num_agents):
            total_rewards[i] += ep_rewards[i]
            total_bombs[i] += ep_bombs[i]
            total_kills[i] += env.agents[i].kills
        for victim in ep_sudden_death_deaths:
            sudden_death_deaths[victim] += 1

        alive = info["alive"]
        survivors = [i for i, a in enumerate(alive) if a]
        if len(survivors) == 1:
            wins[survivors[0]] += 1
            if env.agents[survivors[0]].kills > 0:
                active_wins[survivors[0]] += 1
            elif any(victim != survivors[0] for victim in ep_sudden_death_deaths):
                sudden_death_wins[survivors[0]] += 1
            else:
                passive_wins[survivors[0]] += 1
        else:
            draws += 1

    stats = {
        "num_episodes": num_episodes,
        "wins": wins,
        "active_wins": active_wins,
        "sudden_death_wins": sudden_death_wins,
        "passive_wins": passive_wins,
        "sudden_death_deaths": sudden_death_deaths,
        "draws": draws,
        "win_rates": [w / num_episodes * 100 for w in wins],
        "active_win_rates": [w / num_episodes * 100 for w in active_wins],
        "sudden_death_win_rates": [w / num_episodes * 100 for w in sudden_death_wins],
        "passive_win_rates": [w / num_episodes * 100 for w in passive_wins],
        "suicide_rates": [s / num_episodes * 100 for s in suicides],
        "sudden_death_death_rates": [d / num_episodes * 100 for d in sudden_death_deaths],
        "avg_reward": [r / num_episodes for r in total_rewards],
        "avg_steps": total_steps / num_episodes,
        "avg_bombs": [b / num_episodes for b in total_bombs],
        "avg_kills": [k / num_episodes for k in total_kills],
        "avg_items": [n / num_episodes for n in total_items],
    }

    if verbose:
        print(f"\n{'-' * 50}")
        print(f"Match Results ({num_episodes} episodes)")
        print(f"{'-' * 50}")
        for i in range(num_agents):
            print(f"  Agent {i}: WR={stats['win_rates'][i]:.1f}%, "
                  f"Reward={stats['avg_reward'][i]:.2f}, "
                  f"Bombs={stats['avg_bombs'][i]:.1f}, "
                  f"Kills={stats['avg_kills'][i]:.2f}, "
                  f"ActiveWR={stats['active_win_rates'][i]:.1f}%, "
                  f"Suicide={stats['suicide_rates'][i]:.1f}%, "
                  f"Items={stats['avg_items'][i]:.2f}")
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

    from train_actor_critic import ActorCriticNet, SnapshotOpponent, smart_load_weights

    net = ActorCriticNet(board_size=board_size).to(device)
    smart_load_weights(net, model_path, device)
    policy = SnapshotOpponent(net, device)

    @torch.no_grad()
    def fn(env, obs_list, agent_id):
        return policy.select_action(obs_list[agent_id], env, agent_id)
    return fn


def evaluate_seeded(agent_fn, board_size=9, num_agents=2, num_episodes=100,
                    seed=7000, max_steps=500, stop_event=None, on_progress=None,
                    sudden_death=False, sudden_death_start=350,
                    sudden_death_interval=20, sudden_death_warning_steps=40,
                    initial_games=None, on_checkpoint=None):
    """Evaluate a fixed policy at every corner of each seeded board, preserving RNGs."""
    if num_episodes < num_agents or num_episodes % num_agents:
        raise ValueError("num_episodes must be positive and divisible by num_agents")
    if not _torch_available:
        raise RuntimeError("PyTorch is required for reproducible policy evaluation")
    config = {
        "board_size": board_size,
        "num_agents": num_agents,
        "num_episodes": num_episodes,
        "seed": seed,
        "max_steps": max_steps,
        "sudden_death": sudden_death,
        "sudden_death_start": sudden_death_start,
        "sudden_death_interval": sudden_death_interval,
        "sudden_death_warning_steps": sudden_death_warning_steps,
    }
    games = deepcopy(initial_games or [])
    _validate_ordered_games(games, config)
    numpy_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            for ep in range(len(games), num_episodes):
                if on_progress:
                    on_progress(ep, num_episodes)
                if stop_event is not None and stop_event.is_set():
                    return None
                board_seed, aid = seed + ep // num_agents, ep % num_agents
                np.random.seed(board_seed)
                torch.manual_seed(board_seed)
                env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                                      max_steps=max_steps, seed=board_seed,
                                      sudden_death=sudden_death,
                                      sudden_death_start=sudden_death_start,
                                      sudden_death_interval=sudden_death_interval,
                                      sudden_death_warning_steps=sudden_death_warning_steps)
                fns = [make_rule_based_fn(i) for i in range(num_agents)]
                fns[aid] = agent_fn
                stats = run_match(env, fns, num_episodes=1)
                sudden_death_win = stats.get("sudden_death_wins", [0] * num_agents)[aid]
                passive_win = stats.get(
                    "passive_wins",
                    [win - active - sd for win, active, sd in zip(
                        stats["wins"], stats["active_wins"],
                        stats.get("sudden_death_wins", [0] * num_agents))],
                )[aid]
                games.append({"seed": board_seed, "aid": aid,
                              "win": stats["wins"][aid], "active_win": stats["active_wins"][aid],
                              "sudden_death_win": sudden_death_win,
                              "passive_win": passive_win,
                              "suicide": stats["suicide_rates"][aid] / 100,
                              "sudden_death_death": stats.get(
                                  "sudden_death_death_rates", [0] * num_agents)[aid] / 100,
                              "draw": stats["draws"], "kills": stats["avg_kills"][aid],
                              "bombs": stats["avg_bombs"][aid], "items": stats["avg_items"][aid],
                              "steps": stats["avg_steps"]})
                if on_checkpoint:
                    on_checkpoint(deepcopy(games))
            if on_progress:
                on_progress(num_episodes, num_episodes)
    finally:
        np.random.set_state(numpy_state)

    def summarize(rows):
        rates = {f"{key}_rate": sum(g[key] for g in rows) / len(rows) * 100
                 for key in ("win", "active_win", "sudden_death_win", "passive_win",
                             "suicide", "sudden_death_death", "draw")}
        averages = {f"avg_{key}": sum(g[key] for g in rows) / len(rows)
                    for key in ("kills", "bombs", "items", "steps")}
        return {"num_episodes": len(rows), **rates, **averages}

    return {"seed": seed, "board_size": board_size, "num_agents": num_agents,
            "max_steps": max_steps, "sudden_death": sudden_death,
            "sudden_death_start": sudden_death_start,
            "sudden_death_interval": sudden_death_interval,
            "sudden_death_warning_steps": sudden_death_warning_steps,
            **summarize(games),
            "by_corner": {str(aid): summarize([g for g in games if g["aid"] == aid])
                          for aid in range(num_agents)}, "games": games}


def run_seeded_evaluation_to_file(
        agent_fn, checkpoint_path: str, output_path: str, board_size=9,
        num_agents=2, num_episodes=100, seed=7000, max_steps=500,
        sudden_death=False, sudden_death_start=350, sudden_death_interval=20,
        sudden_death_warning_steps=40, partial_every=10,
        resume_partial=False, device=None):
    if partial_every <= 0:
        raise ValueError("partial_every must be positive")

    evaluation_config = {
        "board_size": board_size,
        "num_agents": num_agents,
        "num_episodes": num_episodes,
        "seed": seed,
        "max_steps": max_steps,
        "sudden_death": sudden_death,
        "sudden_death_start": sudden_death_start,
        "sudden_death_interval": sudden_death_interval,
        "sudden_death_warning_steps": sudden_death_warning_steps,
    }
    artifact_config = dict(evaluation_config)
    if device is not None:
        artifact_config["device"] = device

    expected = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "checkpoint": _checkpoint_identity(checkpoint_path),
        "config": artifact_config,
        "requested_count": num_episodes,
    }
    partial_path = _partial_path_for(output_path)
    initial_games = []
    if os.path.exists(partial_path):
        if not resume_partial:
            raise ValueError(
                f"partial evaluation exists; use resume_partial: {partial_path}"
            )
        initial_games = _load_partial_games(partial_path, expected)

    latest_games = deepcopy(initial_games)
    last_persisted_count = len(initial_games)

    def persist(games):
        nonlocal latest_games, last_persisted_count
        latest_games = deepcopy(games)
        completed = len(latest_games)
        if completed == num_episodes or completed % partial_every == 0:
            _atomic_write_json(partial_path, _partial_payload(expected, latest_games))
            last_persisted_count = completed
            print(f"Evaluation progress: {completed}/{num_episodes}", flush=True)

    try:
        result = evaluate_seeded(
            agent_fn,
            **evaluation_config,
            initial_games=initial_games,
            on_checkpoint=persist,
        )
    except BaseException:
        if latest_games and len(latest_games) != last_persisted_count:
            _atomic_write_json(partial_path, _partial_payload(expected, latest_games))
        raise

    if not isinstance(result, dict):
        raise ValueError("evaluation did not return a complete result")
    games = result.get("games")
    if not isinstance(games, list):
        raise ValueError("evaluation result games must be a list")
    _validate_ordered_games(games, evaluation_config)
    if len(games) != num_episodes or result.get("num_episodes") != num_episodes:
        raise ValueError("evaluation result is incomplete for requested count")
    for key, value in evaluation_config.items():
        if key == "num_episodes":
            continue
        if result.get(key) != value:
            raise ValueError(f"evaluation result config mismatch: {key}")

    _atomic_write_json(output_path, result)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"Saved evaluation to {os.path.abspath(output_path)}", flush=True)
    return result


# ────────────────────── tournament mode ───────────────────────────────────

def tournament(board_size: int = 11, num_agents: int = 2,
               num_episodes: int = 100, ac_model_path: str = "",
               device: str = "cpu", max_steps: int = 500,
               sudden_death: bool = False, sudden_death_start: int = 350,
               sudden_death_interval: int = 20,
               sudden_death_warning_steps: int = 40):
    """
    Round-robin tournament giữa Random, RuleBot, và AC (nếu có model).
    """
    env = BombermanFFAEnv(board_size=board_size, num_agents=num_agents,
                          max_steps=max_steps, sudden_death=sudden_death,
                          sudden_death_start=sudden_death_start,
                          sudden_death_interval=sudden_death_interval,
                          sudden_death_warning_steps=sudden_death_warning_steps)

    agents = {
        "Random": make_random_fn(),
        "RuleBot": make_rule_based_fn(0),
    }
    if ac_model_path:
        if not os.path.isfile(ac_model_path):
            raise FileNotFoundError(f"Actor-Critic checkpoint not found: {ac_model_path}")
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
        results = {"all_rulebots": stats}
        if "ActorCritic" in agents:
            print(f"\n  ActorCritic vs {num_agents - 1} RuleBots:")
            fns[0] = agents["ActorCritic"]
            results["ac_vs_rulebots"] = run_match(env, fns, num_episodes, verbose=True)
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
                        choices=["tournament", "seeded", "random", "rulebots",
                                 "curves", "ablation"],
                        help="Evaluation mode")
    parser.add_argument("--board_size", type=int, default=11)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--ac_model", type=str, default="",
                        help="Path to Actor-Critic model checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--sudden_death", action="store_true")
    parser.add_argument("--sudden_death_start", type=int, default=350)
    parser.add_argument("--sudden_death_interval", type=int, default=20)
    parser.add_argument("--sudden_death_warning_steps", type=int, default=40)
    parser.add_argument("--output", type=str, default="", help="Save seeded evaluation as JSON")
    parser.add_argument("--resume_partial", action="store_true",
                        help="Resume a compatible partial seeded evaluation")
    parser.add_argument("--partial_every", type=int, default=10,
                        help="Persist and print seeded progress every N games")
    args = parser.parse_args()

    if args.mode == "tournament":
        tournament(args.board_size, args.num_agents, args.num_episodes,
                   args.ac_model, args.device, args.max_steps, args.sudden_death,
                   args.sudden_death_start, args.sudden_death_interval,
                   args.sudden_death_warning_steps)
    elif args.mode == "seeded":
        policy = make_ac_fn(args.ac_model, args.board_size, args.device)
        if args.output:
            result = run_seeded_evaluation_to_file(
                policy,
                checkpoint_path=args.ac_model,
                output_path=args.output,
                board_size=args.board_size,
                num_agents=args.num_agents,
                num_episodes=args.num_episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                sudden_death=args.sudden_death,
                sudden_death_start=args.sudden_death_start,
                sudden_death_interval=args.sudden_death_interval,
                sudden_death_warning_steps=args.sudden_death_warning_steps,
                partial_every=args.partial_every,
                resume_partial=args.resume_partial,
                device=args.device,
            )
        else:
            result = evaluate_seeded(
                policy,
                args.board_size,
                args.num_agents,
                args.num_episodes,
                args.seed,
                args.max_steps,
                sudden_death=args.sudden_death,
                sudden_death_start=args.sudden_death_start,
                sudden_death_interval=args.sudden_death_interval,
                sudden_death_warning_steps=args.sudden_death_warning_steps,
            )
        print(json.dumps({k: v for k, v in result.items() if k != "games"}, indent=2))
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
