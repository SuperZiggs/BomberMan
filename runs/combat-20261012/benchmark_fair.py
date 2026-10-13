"""Compare policies under the same final damage-credit convention."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
folder=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('--runtime',choices=['before','current'],required=True)
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--seed',type=int,default=12000)
args=parser.parse_args()
sys.path.insert(0,str(folder/'fair_runtime'))
import torch
from evaluate import evaluate_seeded,make_ac_fn
from train_actor_critic import ActorCriticNet,smart_load_weights,obs_to_tensor
torch.set_num_threads(1)
checkpoint=Path(args.checkpoint).resolve()
if args.runtime=='current':
    policy=make_ac_fn(str(checkpoint),9,'cpu')
else:
    spec=importlib.util.spec_from_file_location('fair_before_rule',folder/'before/rule_based_bot.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    net=ActorCriticNet(board_size=9);smart_load_weights(net,str(checkpoint),'cpu');net.eval()
    @torch.no_grad()
    def policy(env,obs,aid):
        logits,_=net(obs_to_tensor(obs[aid],'cpu'))
        mask=torch.tensor(old._safe_action_mask(env,aid),dtype=torch.bool)
        return torch.distributions.Categorical(logits=logits.masked_fill(~mask,-1e9)).sample().item()
result=evaluate_seeded(policy,num_episodes=100,seed=args.seed)
result.update(checkpoint=str(checkpoint),runtime=args.runtime,
              credit='opponent contributor; own-blast involvement recorded separately',
              model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Fair test saved',args.runtime,flush=True)
