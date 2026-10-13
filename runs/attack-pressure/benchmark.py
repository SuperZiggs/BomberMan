"""Paired final test of identical weights with the previous and final safety filters."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import train_actor_critic as ac
from evaluate import evaluate_seeded, make_ac_fn

torch.set_num_threads(1)
folder=Path(__file__).parent
source=ROOT/'runs/combat-20261012/combat_bc_phaseC.pt'
digest=hashlib.sha256(source.read_bytes()).hexdigest()
code_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
             for name in ['bomberman_env.py','rule_based_bot.py','train_actor_critic.py','evaluate.py']}
spec=importlib.util.spec_from_file_location('control_rule',folder/'before/rule_based_bot.py')
control=importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
final_filter=ac._safe_action_mask
results={}
start=time.monotonic()
try:
    for name,action_filter in [('before',control._safe_action_mask),('after',final_filter)]:
        ac._safe_action_mask=action_filter
        policy=make_ac_fn(str(source),9,'cpu')
        def progress(done,total):
            if done%50==0:
                print(name,done,'/',total,flush=True)
        result=evaluate_seeded(policy,board_size=9,num_agents=2,num_episodes=200,
                               seed=14000,max_steps=500,on_progress=progress)
        results[name]=result
        (folder/f'test_{name}_14000.json').write_text(json.dumps(result,indent=2))
        print(name,json.dumps({k:v for k,v in result.items() if k not in ('games','by_corner')}),flush=True)
finally:
    ac._safe_action_mask=final_filter
assert [(g['seed'],g['aid']) for g in results['before']['games']]==[(g['seed'],g['aid']) for g in results['after']['games']]
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in code_hashes.items())
(folder/'benchmark_runtime.json').write_text(json.dumps({'source_sha256':digest,'code_sha256':code_hashes,
                                                       'elapsed_seconds':time.monotonic()-start},indent=2))
