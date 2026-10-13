"""Replay both policies on final's failure cases without changing policy RNG or weights."""
from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bomberman_env import BombermanFFAEnv, Bomb, ACTION_BOMB, ACTION_STOP, _MOVE_DELTA
from evaluate import make_ac_fn, make_rule_based_fn
from train_actor_critic import get_action_mask
from rule_based_bot import _survival_map

torch.set_num_threads(1)
folder=Path(__file__).parent
cases=json.loads((folder/'diagnostic_summary_15000.json').read_text())['final_failure_cases']
models={'final':folder/'final_phaseC.pt','warm_start':ROOT/'runs/combat-20261012/combat_bc_phaseC.pt'}
expected={name:{(g['seed'],g['aid']):g for g in json.loads(path.read_text())['games']}
          for name,path in [('final',folder/'test_final_15000.json'),('warm_start',ROOT/'runs/test_before_15000.json')]}
hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in models.items()}
output={'games':[],'summary':{},'model_sha256':hashes}
for name,path in models.items():
    policy=make_ac_fn(str(path),9,'cpu')
    total=Counter()
    for idx,case in enumerate(cases):
        seed,aid=case['seed'],case['aid']
        np.random.seed(seed);torch.manual_seed(seed)
        env=BombermanFFAEnv(board_size=9,num_agents=2,max_steps=500,seed=seed)
        obs=env.reset();bot=make_rule_based_fn(1-aid)
        tail=deque(maxlen=16);events=[];counts=Counter();suicide=False;bombs=0
        first_doom=None
        for tick in range(500):
            a=env.agents[aid]; start=(a.row,a.col)
            mask=get_action_mask(env,aid)
            safe,blocked,burns=_survival_map(env,*start,avoid_contests=False)
            alive_routes=bool(safe[0,start[0],start[1]])
            if not alive_routes and first_doom is None:first_doom=tick+1
            actions=[0,0]
            for i in range(2):actions[i]=policy(env,obs,i) if i==aid else bot(env,obs,i)
            own_solo_safe=None
            if actions[aid]==ACTION_BOMB:
                placement=Bomb(*start,aid,env.bomb_timer,a.blast_range)
                solo,_,_=_survival_map(env,*start,placement,safety_margin=2,reserve_exits=True)
                own_solo_safe=bool(solo[0,start[0],start[1]])
                counts['bombs_solo_unsafe']+=int(not own_solo_safe)
            record={'step':tick+1,'agents':[[x.row,x.col] for x in env.agents],
                    'range':[x.blast_range for x in env.agents],'ammo':[x.ammo for x in env.agents],
                    'actions':actions,'mask':mask,'danger':int(env._danger_map()[start]),
                    'known_route':alive_routes,'solo_safe':own_solo_safe,
                    'bombs':[[b.row,b.col,b.owner,b.timer,b.blast_range] for b in env.bombs]}
            before=deepcopy(env)
            obs,_,_,info=env.step(actions)
            placed=aid in info['bombs_placed'];bombs+=int(placed)
            suicide |= aid in info.get('self_blast_deaths',[]) or (aid,aid) in info['kills']
            record.update(after=[[x.row,x.col] for x in env.agents],kills=info['kills'],self_deaths=info.get('self_blast_deaths',[]))
            counts['steps']+=1
            if (a.row,a.col)==start and actions[aid] in _MOVE_DELTA:
                counts['blocked_moves']+=1
                record['blocked_move']=True
            if placed or record.get('blocked_move') or not alive_routes:
                record['grid']=before.grid.tolist()
                events.append(record)
            tail.append(record)
            if info['episode_done']:break
        win=int(env.agents[aid].alive and not env.agents[1-aid].alive)
        actual={'win':win,'active_win':int(win and env.agents[aid].kills>0), 'suicide':int(suicide),
                'draw':int(sum(info['alive'])!=1),'kills':env.agents[aid].kills,'bombs':bombs,'steps':info['step']}
        wanted=expected[name][seed,aid]
        assert all(actual[k]==wanted[k] for k in actual),(name,seed,aid,actual,wanted)
        row={'model':name,'seed':seed,'aid':aid,'result':actual,'counts':dict(counts),
             'first_no_known_route':first_doom,'tail':list(tail),'events':events,'final_grid':env.grid.tolist()}
        output['games'].append(row);total.update(counts)
        print(name,idx+1,'/',len(cases),seed,aid,actual,'no_route',first_doom,dict(counts),flush=True)
    output['summary'][name]=dict(total)
assert all(hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name] for name,path in models.items())
(folder/'replay_failures_15000.json').write_text(json.dumps(output,indent=2))
print('SUMMARY',json.dumps(output['summary']),flush=True)
