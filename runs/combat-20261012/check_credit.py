"""Regression checks for damage attribution; later reused against production."""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent/'fair_runtime'))
from bomberman_env import BombermanFFAEnv,Bomb,ACTION_BOMB,ACTION_STOP,ACTION_DOWN,ACTION_LEFT,ACTION_UP
from train_actor_critic import _combat_reward

for learner in (0,1):
    env=BombermanFFAEnv(board_size=9,num_agents=2,wooden_wall_density=0);env.reset()
    env.agents[learner].row,env.agents[learner].col=1,3
    env.agents[learner].blast_range=4
    env.agents[1-learner].row,env.agents[1-learner].col=1,1
    _,_,_,info=env.step([ACTION_BOMB,ACTION_BOMB])
    for action in [ACTION_DOWN,ACTION_DOWN,ACTION_LEFT]+[ACTION_STOP]*4:
        acts=[ACTION_STOP,ACTION_STOP];acts[learner]=action
        _,reward,_,info=env.step(acts)
        if info['episode_done']:break
    assert env.agents[learner].alive
    assert info['kills']==[(learner,1-learner)]
    assert info['self_blast_deaths']==[1-learner]
    assert env.agents[learner].kills==1
    assert abs(_combat_reward(reward[learner],learner,info,1)-13.999)<1e-6

env=BombermanFFAEnv(board_size=9,num_agents=2,wooden_wall_density=0);env.reset()
for a,pos in zip(env.agents,[(1,1),(1,3)]):
    a.row,a.col=pos;a.blast_range=4
    env.bombs.append(Bomb(*pos,a.agent_id,1,4));a.bombs_placed=1
_,rewards,_,info=env.step([ACTION_STOP]*2)
assert info['kills']==[(1,0),(0,1)]
assert info['self_blast_deaths']==[0,1]
assert all(_combat_reward(rewards[i],i,info,1)==-2.5 for i in (0,1))

env=BombermanFFAEnv(board_size=9,num_agents=3,wooden_wall_density=0);env.reset()
for a,pos in zip(env.agents,[(1,7),(7,7),(4,3)]):a.row,a.col=pos
env.bombs.append(Bomb(3,1,0,1,2));env.agents[0].bombs_placed=1
env.step([ACTION_STOP]*3)
env.bombs.append(Bomb(3,5,1,1,2));env.agents[1].bombs_placed=1
env.step([ACTION_STOP]*3)
assert env._flame_sources[(3,3)]=={1:1}
_,_,_,info=env.step([ACTION_STOP,ACTION_STOP,ACTION_UP])
assert info['kills']==[(1,2)],info
assert env._flame_sources=={} and env._flame_owners=={}
print('PASS: mirrored credit, own contribution penalties and refreshed flame expiry')
