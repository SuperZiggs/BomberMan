"""Fixed-policy test on fresh boards; before mode uses today's frozen runtime."""
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
parser.add_argument('--seed',type=int,default=11000)
parser.add_argument('--episodes',type=int,default=100)
args=parser.parse_args()
sys.path.insert(0,str(folder/'before') if args.runtime=='before' else str(ROOT))
import torch
from evaluate import make_ac_fn,evaluate_seeded
torch.set_num_threads(1)
checkpoint=Path(args.checkpoint).resolve()
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
def progress(done,total):
    if done%25==0:
        print(args.runtime,done,'/',total,flush=True)
result=evaluate_seeded(make_ac_fn(str(checkpoint),9,'cpu'),num_episodes=args.episodes,
                       seed=args.seed,on_progress=progress)
assert len(result['games'])==args.episodes
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
result.update(checkpoint=str(checkpoint),runtime=args.runtime,model_sha256=digest)
Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('games','by_corner')}),flush=True)
