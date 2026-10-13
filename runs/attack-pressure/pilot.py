"""Bounded CLI pilot; writes only a new run directory and preserves source weights."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
source = ROOT/'runs/combat-20261012/combat_bc_phaseC.pt'
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
code_names = ['bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'evaluate.py']
code_hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in code_names}
output = ROOT/'runs/attack-pressure/pilot-50'
assert not (output/'final_phaseC.pt').exists(), 'Use a new directory rather than overwrite a prior pilot'
sys.argv = ['train_actor_critic.py', '--phase','C','--episodes','50','--board_size','9',
            '--num_agents','2','--max_steps','500','--lr','0.0001','--wall_destroy_bonus','0.15',
            '--seed','2026101202','--resume_from',str(source),'--validation_every','50',
            '--validation_episodes','100','--validation_seed','6300','--log_every','10',
            '--checkpoint_dir',str(output)]
start = time.monotonic()
runpy.run_path(str(ROOT/'train_actor_critic.py'),run_name='__main__')
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
           for name,digest in code_hashes.items())
rows=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
assert len(rows)==50
assert [r['learning_aid'] for r in rows].count(0)==25
assert all(0<=r['combat_bombs']<=r['combat_labels']<=r['attack_labels'] for r in rows)
metadata={'elapsed_seconds':time.monotonic()-start,'source_sha256':source_hash,'code_sha256':code_hashes}
(output/'runtime.json').write_text(json.dumps(metadata,indent=2))
print('Pilot completed',json.dumps(metadata),flush=True)
