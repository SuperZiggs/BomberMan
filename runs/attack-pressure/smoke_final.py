"""Exercise final CLI training and metrics without replacing an existing run."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
torch.set_num_threads(1)
source=ROOT/'runs/combat-20261012/combat_bc_phaseC.pt'
digest=hashlib.sha256(source.read_bytes()).hexdigest()
out=Path(__file__).parent/'smoke-final-10'
assert not (out/'metrics.jsonl').exists()
sys.argv=['train_actor_critic.py','--phase','C','--episodes','10','--board_size','9','--num_agents','2',
          '--max_steps','500','--lr','0.0001','--wall_destroy_bonus','0.15','--seed','2026101202',
          '--resume_from',str(source),'--checkpoint_dir',str(out),'--log_every','5']
runpy.run_path(str(ROOT/'train_actor_critic.py'),run_name='__main__')
rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
assert len(rows)==10
assert [r['learning_aid'] for r in rows].count(0)==5
assert all(0<=r['followup_bombs']<=r['followup_labels'] for r in rows)
assert all(0<=r['combat_bombs']<=r['combat_labels']<=r['attack_labels']+r['followup_labels'] for r in rows)
assert all(all(torch.isfinite(torch.tensor(r[key])) for key in ['loss','policy_loss','value_loss',
                                                              'escape_loss','combat_loss','navigation_loss']) for r in rows)
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
print('Final training smoke: 10 episodes, followup labels',sum(r['followup_labels'] for r in rows),flush=True)
