"""Test a timed follow-up attack independently; never modify the running pilot."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import train_actor_critic as ac
from bomberman_env import ACTION_BOMB
from evaluate import make_ac_fn,evaluate_seeded
from rule_based_bot import _can_bomb_and_escape

torch.set_num_threads(1)
original=ac.get_action_mask
counts=dict(added=0,chosen=0)
def mask(env,aid):
    result=original(env,aid)
    a=env.agents[aid]
    if not result[ACTION_BOMB] and a.alive and env._danger_map()[a.row,a.col]>0:
        ray=env._blast_cells(a.row,a.col,a.blast_range)
        if any(b.alive and b.agent_id!=aid and (b.row,b.col) in ray for b in env.agents):
            if _can_bomb_and_escape(env,aid,a.row,a.col,None):
                result[ACTION_BOMB]=True
                counts['added']+=1
    return result
ac.get_action_mask=mask
policy=make_ac_fn(str(ROOT/'runs/20261010-135449-587798/best_phaseC.pt'),9,'cpu')
def act(env,obs,aid):
    action=policy(env,obs,aid)
    if action==ACTION_BOMB and env._danger_map()[env.agents[aid].row,env.agents[aid].col]>0:
        counts['chosen']+=1
    return action
def progress(done,total):
    if done%25==0: print('Follow-up probe',done,'/',total,flush=True)
result=evaluate_seeded(act,num_episodes=100,seed=9000,on_progress=progress)
assert len(result['games'])==100
result['followup']=counts
(Path(__file__).parent/'followup_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('games','by_corner')}),flush=True)
