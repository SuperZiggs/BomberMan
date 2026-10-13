"""Evaluate the learned final checkpoint; preserve best and all existing test files."""
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from evaluate import evaluate_seeded, make_ac_fn

torch.set_num_threads(1)
folder=Path(__file__).parent
paths=[ROOT/'runs/combat-20261012/combat_bc_phaseC.pt',folder/'best_phaseC.pt',folder/'final_phaseC.pt']
hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
baseline=json.loads((ROOT/'runs/test_before_15000.json').read_text())
assert (baseline['seed'],baseline['board_size'],baseline['num_agents'],baseline['max_steps'],baseline['num_episodes']) == (15000,9,2,500,200)
calibration=evaluate_seeded(make_ac_fn(str(paths[0]),9,'cpu'),board_size=9,num_agents=2,
                            num_episodes=8,seed=15000,max_steps=500)
assert calibration['games']==baseline['games'][:8], 'Baseline replay differs; rerun both policies on the same runtime'
policy=make_ac_fn(str(paths[2]),9,'cpu')
def progress(done,total):
    if done%50==0:print('Final test',done,'/',total,flush=True)
result=evaluate_seeded(policy,board_size=9,num_agents=2,num_episodes=200,
                       seed=15000,max_steps=500,on_progress=progress)
assert [(g['seed'],g['aid']) for g in result['games']]==[(g['seed'],g['aid']) for g in baseline['games']]
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in hashes.items())
(folder/'test_final_15000.json').write_text(json.dumps(result,indent=2))
rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
assert len(rows)==500
assert all(all(math.isfinite(row[k]) for k in ['loss','policy_loss','value_loss','escape_loss','combat_loss','navigation_loss']) for row in rows)
summary={name:{k:v for k,v in data.items() if k!='games'} for name,data in [('warm_start',baseline),('final',result)]}
summary['model_sha256']=hashes
summary['baseline_calibration_games']=8
(folder/'comparison_final_15000.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
