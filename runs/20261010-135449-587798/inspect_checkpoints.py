"""Read-only, seeded evaluation matching the earlier 100-game benchmark."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from bomberman_env import BombermanFFAEnv
from evaluate import make_ac_fn, make_rule_based_fn, run_match

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", choices=["best", "final"])
args = parser.parse_args()
torch.set_num_threads(1)
folder = Path(__file__).resolve().parent
checkpoint = folder / f"{args.checkpoint}_phaseC.pt"
policy = make_ac_fn(str(checkpoint), 9, "cpu")
games, traces = [], []
for ep in range(100):
    seed, aid = 7000 + ep, ep % 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed)
    recent = deque(maxlen=12)

    def learner(env, obs, agent_id):
        action = policy(env, obs, agent_id)
        a = env.agents[agent_id]
        recent.append({"step": env.step_count + 1, "pos": [a.row, a.col],
                       "action": action, "agents": [[b.row, b.col] for b in env.agents],
                       "bombs": [[b.row, b.col, b.owner, b.timer, b.blast_range] for b in env.bombs],
                       "flames": [[r, c, timer] for (r, c), timer in env.flames.items()]})
        return action

    fns = [None, None]
    fns[aid], fns[1-aid] = learner, make_rule_based_fn(1-aid)
    stats = run_match(env, fns, num_episodes=1)
    game = {"seed": seed, "aid": aid, "win": stats["wins"][aid],
            "active_win": stats["active_wins"][aid], "draw": stats["draws"],
            "suicide": int(stats["suicide_rates"][aid] / 100),
            "kills": stats["avg_kills"][aid], "bombs": stats["avg_bombs"][aid],
            "items": stats["avg_items"][aid], "steps": stats["avg_steps"],
            "alive": env.agents[aid].alive, "ammo": env.agents[aid].ammo,
            "range": env.agents[aid].blast_range}
    assert game["active_win"] <= game["win"] and game["suicide"] <= 1
    assert sum(stats["wins"]) + game["draw"] == 1
    games.append(game)
    if game["suicide"] and len(traces) < 2:
        traces.append({"seed": seed, "aid": aid, "grid": env.grid.tolist(),
                       "trace": list(recent), "final_pos": [env.agents[aid].row, env.agents[aid].col]})
    if (ep + 1) % 25 == 0:
        print(args.checkpoint, ep + 1, "games", flush=True)

keys = ["win", "active_win", "draw", "suicide", "kills", "bombs", "items", "steps"]
def summarize(rows):
    return {"n": len(rows), **{k: sum(g[k] for g in rows) / len(rows) for k in keys},
            "kills_while_dead": sum(g["kills"] > 0 and not g["alive"] for g in rows),
            "max_ammo_games": sum(g["ammo"] == 3 for g in rows),
            "max_range_games": sum(g["range"] == 4 for g in rows)}

result = {"checkpoint": str(checkpoint), "protocol": {"board_size": 9, "num_agents": 2,
          "max_steps": 500, "seeds": [7000, 7099], "policy": "sampled with shared action mask",
          "opponent": "current RuleBot", "device": "cpu", "threads": 1},
          "summary": summarize(games), "by_corner": {str(aid): summarize([g for g in games if g["aid"] == aid]) for aid in (0, 1)},
          "games": games, "suicide_traces": traces}
output = folder / f"evaluation_{args.checkpoint}.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"checkpoint": args.checkpoint, "summary": result["summary"],
                  "by_corner": result["by_corner"]}), flush=True)
