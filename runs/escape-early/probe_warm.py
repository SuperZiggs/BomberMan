"""Check the same escape change on warm-start, using development seeds only."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import train_actor_critic as ac
from bomberman_env import ACTION_STOP, _MOVE_DELTA
from evaluate import evaluate_seeded, make_ac_fn

spec = importlib.util.spec_from_file_location('before_escape', Path(__file__).parent / 'before/rule_based_bot.py')
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
torch.set_num_threads(1)

def early(env, aid):
    mask = before._safe_action_mask(env, aid)
    a = env.agents[aid]
    if a.alive and mask[ACTION_STOP] and not any(mask[i] for i in _MOVE_DELTA) and env._danger_map()[a.row, a.col] > 1:
        alternative = before._safe_action_mask(env, aid, avoid_contests=False)
        if any(alternative[i] for i in _MOVE_DELTA):
            return alternative
    return mask

ac._safe_action_mask = early
result = evaluate_seeded(make_ac_fn(str(ROOT / 'runs/combat-20261012/combat_bc_phaseC.pt'), 9, 'cpu'),
                        board_size=9, num_agents=2, num_episodes=200, seed=15000, max_steps=500,
                        on_progress=lambda n, total: print('Warm probe', n, '/', total, flush=True) if n % 50 == 0 else None)
(Path(__file__).parent / 'probe_warm_15000.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ('games', 'by_corner')}), flush=True)
