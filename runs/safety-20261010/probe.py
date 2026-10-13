"""Seeded checkpoint comparison; preserves traces and original implementation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--variant', default='before')
p.add_argument('--opponent', choices=['legacy', 'current'], default='legacy')
p.add_argument('--episodes', type=int, default=100)
p.add_argument('--checkpoint', default='runs/20261010-125814-296523/best_phaseC.pt')
p.add_argument('--range-cap', type=int, choices=[4,5], default=4)
args = p.parse_args()
sys.path.insert(0, str(HERE / 'before' if args.variant == 'before' else ROOT))
import numpy as np
import torch
from bomberman_env import BombermanFFAEnv
from rule_based_bot import RuleBasedAgent, _build_danger_map
from train_actor_critic import ActorCriticNet, SnapshotOpponent, smart_load_weights, get_action_mask
if args.variant != 'before':
    import bomberman_env
    import rule_based_bot
    bomberman_env.MAX_BLAST_RANGE = args.range_cap
    rule_based_bot.MAX_BLAST_RANGE = args.range_cap
torch.set_num_threads(1)
if args.opponent == 'legacy':
    spec = importlib.util.spec_from_file_location('legacy_rule', HERE / 'before/rule_based_bot.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    RuleBasedAgent = legacy.RuleBasedAgent
net = ActorCriticNet(board_size=9)
smart_load_weights(net, str(ROOT / args.checkpoint), 'cpu')
policy = SnapshotOpponent(net, 'cpu')
episodes, traces = [], []
for ep in range(args.episodes):
    seed = 7000 + ep
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed)
    obs = env.reset()
    aid = ep % 2
    bot = RuleBasedAgent(1-aid)
    bombs = suicide = items = 0
    trace = []
    for t in range(500):
        actions = [0, 0]
        actions[aid] = policy.select_action(obs[aid], env, aid)
        actions[1-aid] = bot.select_action(env)
        a = env.agents[aid]
        trace.append({'step':t+1, 'pos':[a.row,a.col], 'action':actions[aid],
                      'agents':[[b.row,b.col] for b in env.agents], 'actions':actions[:],
                      'mask':get_action_mask(env,aid),
                      'bombs':[[b.row,b.col,b.owner,b.timer,b.blast_range] for b in env.bombs],
                      'danger':int(_build_danger_map(env)[a.row,a.col])})
        obs, rewards, dones, info = env.step(actions)
        bombs += aid in info['bombs_placed']
        suicide |= (aid,aid) in info['kills']
        items += len(info.get('items_collected', {}).get(aid, []))
        if info['episode_done']:
            break
    won = int(env.agents[aid].alive and not env.agents[1-aid].alive)
    kills = env.agents[aid].kills
    episodes.append({'seed':seed, 'aid':aid, 'win':won, 'active_win':int(won and kills>0),
                     'suicide':int(suicide), 'kills':kills,'bombs':bombs,'items':items,'steps':info['step']})
    if suicide and len(traces)<2:
        traces.append({'seed':seed, 'aid':aid, 'grid':env.grid.tolist(), 'trace':trace})
summary = {k:sum(e[k] for e in episodes)/len(episodes) for k in ['win','active_win','suicide','kills','bombs','items','steps']}
out = {'checkpoint':args.checkpoint, 'episodes':len(episodes), 'variant':args.variant, 'range_cap':args.range_cap,
       'opponent':args.opponent, 'summary':summary,'games':episodes,'suicide_traces':traces}
suffix = '-range5' if args.range_cap == 5 else ''
path = HERE / f'{args.variant}-{args.opponent}-{Path(args.checkpoint).stem}{suffix}.json'
path.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(path.name, json.dumps(summary), flush=True)
for trace in traces:
    print('suicide trace', trace['seed'], trace['trace'][-8:], flush=True)
