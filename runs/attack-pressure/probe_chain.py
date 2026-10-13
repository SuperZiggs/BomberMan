"""Allow only a safe follow-up with zero remaining opponent positions in the forecast."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import train_actor_critic as ac
from bomberman_env import ACTION_BOMB
from rule_based_bot import _can_bomb_and_escape, _bomb_escape_pressure
from evaluate import evaluate_seeded, make_ac_fn

torch.set_num_threads(1)
original=ac._safe_action_mask
counts={'checks':0,'unlocked':0}
def followup(env,aid):
    mask=original(env,aid)
    a=env.agents[aid]
    if a.alive and not mask[ACTION_BOMB] and env._danger_map()[a.row,a.col]>0:
        counts['checks']+=1
        if _can_bomb_and_escape(env,aid,a.row,a.col,None) and _bomb_escape_pressure(env,aid,a.row,a.col)==1.0:
            mask[ACTION_BOMB]=True
            counts['unlocked']+=1
    return mask

ac._safe_action_mask=followup
policy=make_ac_fn(str(ROOT/'runs/combat-20261012/combat_bc_phaseC.pt'),9,'cpu')
result=evaluate_seeded(policy,board_size=9,num_agents=2,num_episodes=100,seed=6300,max_steps=500)
result['mask_probe_counts']=counts
(Path(__file__).parent/'chain_probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('games','by_corner')}),flush=True)
