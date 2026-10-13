"""Wait for validation selection, then test the chosen model on untouched boards."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
folder = Path(__file__).parent / 'pilot-500'
deadline = time.monotonic() + 1800
while True:
    try:
        selection = json.loads((folder/'best_validation.json').read_text())
        if selection.get('source') == 'post-training CPU validation':
            break
    except (OSError,json.JSONDecodeError):
        pass
    if time.monotonic() >= deadline:
        raise TimeoutError('Checkpoint selection did not finish')
    time.sleep(1)
import torch
from evaluate import make_ac_fn,evaluate_seeded
torch.set_num_threads(1)
checkpoint = folder/'best_phaseC.pt'
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
def progress(done,total):
    if done % 25 == 0:
        print('Final test',done,'/',total,flush=True)
result = evaluate_seeded(make_ac_fn(str(checkpoint),9,'cpu'),num_episodes=100,seed=9000,on_progress=progress)
assert len(result['games']) == 100
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
result.update(checkpoint=str(checkpoint),model_sha256=digest,selected_episode=selection['episode'])
(folder.parent/'test_best.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['games','by_corner']}),flush=True)
