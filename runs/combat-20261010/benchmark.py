import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bomberman_env import BombermanFFAEnv
from evaluate import run_match, make_rule_based_fn
from visualize import load_agent

torch.set_num_threads(1)
model = sys.argv[1]
episodes_per_corner = int(sys.argv[2]) if len(sys.argv) > 2 else 10
policy = load_agent(ROOT / model, 11)
rows = []
for aid in range(4):
    torch.manual_seed(100 + aid)
    np.random.seed(100 + aid)
    env = BombermanFFAEnv(board_size=11, num_agents=4, max_steps=500, seed=100 + aid)
    fns = [make_rule_based_fn(i) for i in range(4)]
    fns[aid] = lambda env, obs, i: policy.select_action(obs[i], env, i)
    stats = run_match(env, fns, episodes_per_corner)
    row = {"corner": aid, "win_rate": stats["win_rates"][aid],
           "bombs": stats["avg_bombs"][aid], "kills": stats["avg_kills"][aid],
           "steps": stats["avg_steps"]}
    if "active_win_rates" in stats:
        row["active_win_rate"] = stats["active_win_rates"][aid]
        row["suicide_rate"] = stats["suicide_rates"][aid]
    rows.append(row)
    print(json.dumps(row), flush=True)
result = {"model": model, "episodes": 4 * episodes_per_corner, "max_steps": 500,
          "seeds": [100, 101, 102, 103], "corners": rows,
          "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "corner"}}
output = ROOT / "runs" / "combat-20261010" / (Path(model).stem + "_eval.json")
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result["mean"]), flush=True)
