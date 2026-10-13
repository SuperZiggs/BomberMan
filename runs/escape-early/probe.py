"""Development test of leaving a contested exit before the final damage tick."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import train_actor_critic as ac
from bomberman_env import ACTION_STOP, ACTION_BOMB, _MOVE_DELTA
from evaluate import evaluate_seeded, make_ac_fn

torch.set_num_threads(1)
original=ac._safe_action_mask
counts={'unlocked':0}
def early(env,aid):
    mask=original(env,aid)
    a=env.agents[aid]
    if a.alive and mask[ACTION_STOP] and not any(mask[i] for i in _MOVE_DELTA) and env._danger_map()[a.row,a.col]>1:
        alternative=original(env,aid,avoid_contests=False)
        if any(alternative[i] for i in _MOVE_DELTA):
            counts['unlocked']+=1
            return alternative
    return mask

ac._safe_action_mask=early
path=ROOT/'runs/pressure-C-next/final_phaseC.pt'
result=evaluate_seeded(make_ac_fn(str(path),9,'cpu'),board_size=9,num_agents=2,
                       num_episodes=200,seed=15000,max_steps=500,
                       on_progress=lambda n,total:print('Probe',n,'/',total,flush=True) if n%50==0 else None)
result['early_escape_counts']=counts
(Path(__file__).parent/'probe_final_15000.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('games','by_corner')}),flush=True)
