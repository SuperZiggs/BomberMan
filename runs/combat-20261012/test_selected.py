"""Test the validation-selected checkpoint only after all 500 updates finish."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
folder=Path(__file__).parent
pilot=folder/'pilot-500'
deadline=time.monotonic()+10800
while not (pilot/'training_summary.json').exists():
    if time.monotonic()>deadline:
        raise TimeoutError('The requested pilot has not finished')
    time.sleep(1)
sys.path.insert(0,str(ROOT))
import torch
from evaluate import make_ac_fn,evaluate_seeded
torch.set_num_threads(1)
hashes=json.loads((pilot/'code_hashes.json').read_text())
for name,expected in hashes.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
selection=json.loads((pilot/'best_validation.json').read_text())
assert selection['device']=='cpu' and selection['seed']==6100
checkpoint=pilot/'best_phaseC.pt'
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
result=evaluate_seeded(make_ac_fn(str(checkpoint),9,'cpu'),num_episodes=100,seed=11000)
assert len(result['games'])==100
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
result.update(checkpoint=str(checkpoint),runtime='current',model_sha256=digest,
              selected_episode=selection['episode'],runtime_hashes=hashes)
(folder/'test_best.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Final test saved for selected episode',selection['episode'],flush=True)
