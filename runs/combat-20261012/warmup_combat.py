"""A bounded, balanced teacher warm-up candidate on diagnostic boards only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
folder=Path(__file__).parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from bomberman_env import BombermanFFAEnv,ACTION_BOMB
from evaluate import evaluate_seeded
from train_actor_critic import ActorCriticNet,SnapshotOpponent,smart_load_weights,get_action_mask,get_attack_action

torch.set_num_threads(1)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
old_rule=load('warmup_before_rule',folder/'before/rule_based_bot.py')
old_ac=load('warmup_before_ac',folder/'before/train_actor_critic.py')
old_ac._safe_action_mask=old_rule._safe_action_mask
checkpoint=ROOT/'runs/20261010-135449-587798/best_phaseC.pt'
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
net=ActorCriticNet(board_size=9)
smart_load_weights(net,str(checkpoint),'cpu')
old_policy=old_ac.SnapshotOpponent(net,'cpu')
diagnostics=json.loads((folder/'diagnostics.json').read_text())
groups=[[],[]]
start=time.monotonic()
for idx,game in enumerate(diagnostics['games']):
    seed,aid=game['seed'],game['aid']
    np.random.seed(seed);torch.manual_seed(seed)
    env=BombermanFFAEnv(board_size=9,num_agents=2,max_steps=500,seed=seed)
    obs=env.reset()
    bot=old_rule.RuleBasedAgent(1-aid)
    for step in range(500):
        mask=get_action_mask(env,aid)
        target=get_attack_action(env,aid,env._danger_map(),mask)
        if target is not None and sum(mask)>1:
            assert mask[target]
            a=env.agents[aid]
            direct=target==ACTION_BOMB and (env.agents[1-aid].row,env.agents[1-aid].col) in env._blast_cells(a.row,a.col,a.blast_range)
            groups[0 if direct else 1].append((obs[aid].copy(),mask,target))
        actions=[0,0]
        for i in (0,1):
            actions[i]=old_policy.select_action(obs[i],env,i) if i==aid else bot.select_action(env)
        obs,rewards,dones,info=env.step(actions)
        if info['episode_done']:break
    assert info['step']==game['result']['steps'],(seed,aid,info['step'])
    assert env.agents[aid].kills==game['result']['kills']
    if (idx+1)%10==0:print('Teacher boards',idx+1,'/ 30',list(map(len,groups)),flush=True)
assert min(map(len,groups))>100
torch.manual_seed(2026101201)
tensors=[]
for rows in groups:
    tensors.append((torch.tensor(np.stack([r[0] for r in rows])),
                    torch.tensor([r[1] for r in rows],dtype=torch.bool),
                    torch.tensor([r[2] for r in rows],dtype=torch.long)))
optimizer=torch.optim.Adam(net.parameters(),lr=5e-5)
net.train()
losses=[]
for update in range(200):
    data=[]
    for x,m,y in tensors:
        index=torch.randint(len(x),(16,))
        data.append((x[index],m[index],y[index]))
    obs=torch.cat([d[0] for d in data]);mask=torch.cat([d[1] for d in data]);target=torch.cat([d[2] for d in data])
    logits,_=net(obs)
    per_example=F.cross_entropy(logits.masked_fill(~mask,-1e9),target,reduction='none')
    loss=0.5*per_example[:16].mean()+0.2*per_example[16:].mean()
    assert torch.isfinite(loss)
    optimizer.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
    optimizer.step();losses.append(float(loss.detach()))
path=folder/'combat_bc_phaseC.pt'
torch.save(net.state_dict(),path)
policy=SnapshotOpponent(net,'cpu')
print('Warm-up validation, 100 games seed 6100',flush=True)
result=evaluate_seeded(lambda e,obs,aid:policy.select_action(obs[aid],e,aid),num_episodes=100,seed=6100)
result.update(source='balanced teacher warm-up',checkpoint=str(path),updates=200,lr=5e-5,
              train_seed=2026101201,combat_labels=len(groups[0]),navigation_labels=len(groups[1]),
              initial_loss=losses[0],final_loss=losses[-1],elapsed_seconds=time.monotonic()-start,
              model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
(folder/'warmup_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('games','by_corner')}),flush=True)
