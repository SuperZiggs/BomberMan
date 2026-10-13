"""Rerank existing candidates with fair credit; do not use final-test results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
folder=Path(__file__).parent
parser=argparse.ArgumentParser()
parser.add_argument('candidate',choices=['original','bc','final'])
args=parser.parse_args()
pilot=folder/'pilot-500'
if args.candidate=='final':
    deadline=time.monotonic()+10800
    while not (pilot/'training_summary.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Pilot did not finish')
        time.sleep(1)
paths={'original':ROOT/'runs/20261010-135449-587798/best_phaseC.pt',
       'bc':folder/'combat_bc_phaseC.pt','final':pilot/'final_phaseC.pt'}
path=paths[args.candidate]
sys.path.insert(0,str(folder/'fair_runtime'))
import torch
from evaluate import evaluate_seeded,make_ac_fn
torch.set_num_threads(1)
result=evaluate_seeded(make_ac_fn(str(path),9,'cpu'),num_episodes=100,seed=6100)
result.update(candidate=args.candidate,checkpoint=str(path),device='cpu',
              credit='opponent contributor; own-blast involvement recorded separately',
              model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(folder/f'fair_validation_{args.candidate}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('FAIR VALIDATION',args.candidate,json.dumps({k:v for k,v in result.items() if k not in ('games','by_corner')}),flush=True)
