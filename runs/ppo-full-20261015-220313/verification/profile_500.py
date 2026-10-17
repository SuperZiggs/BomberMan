"""Worst-length D1 update smoke on a static 500-step fixture; not a performance evaluation."""
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from audit_t2 import digest,dump
from bomberman_env import BombermanFFAEnv
from benchmark_t3 import observations
import train_actor_critic as trainer
from training_state import load_training_state

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
device='cuda' if torch.cuda.is_available() else 'cpu'
base=Path(__file__).resolve().parent/'profile500'
base.mkdir(exist_ok=False)
B=ROOT/'runs/pressure-C-next/final_phaseC.pt'
before=digest(B)
results={}
for objective in ('ratio','ppo'):
    env=BombermanFFAEnv(board_size=9,num_agents=2,max_steps=500,seed=33)
    env.reset()
    folder=base/objective
    rows=[]
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=perf_counter()
    with patch.object(trainer,'BombermanFFAEnv',return_value=env), \
         patch.object(env,'reset',side_effect=lambda:observations(env)), \
         patch.object(trainer.RuleBasedOpponent,'select_action_env',return_value=0), \
         patch('torch.distributions.Categorical.sample',side_effect=lambda:torch.tensor([0],device=device)):
        trainer.train(episodes=1,board_size=9,max_steps=500,device=device,seed=33,lr=1e-4,
            checkpoint_dir=str(folder),resume_from=str(B),policy_objective=objective,update_epochs=4,
            target_kl=0,save_training_state=True,validation_every=0,run_report=False,on_episode=rows.append)
    elapsed=perf_counter()-start
    state=load_training_state(folder/'training_state.pt')
    assert len(rows)==1 and rows[0]['learner_steps']==500 and rows[0]['ratio_update']['epochs']==4
    assert rows[0]['ratio_update']['last_boundary']=='time_limit'
    assert {int(v['step']) for v in state['optimizer']['state'].values()}=={4}
    assert all(torch.isfinite(v).all().item() for v in state['model'].values())
    dump(folder/'metrics.json',rows[0])
    results[objective]={'device':device,'steps':500,'epochs':4,'adam_steps':4,'seconds':elapsed,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,'finite_weights':True}
assert digest(B)==before
dump(base/'summary.json',{'status':'complete','full_protocol':False,'modes':results,
    'baseline_unchanged':True,'limits':'Static STOP fixture, KL disabled to force 4 epochs; not a policy or full-game throughput estimate.'})
print(json.dumps(results,indent=2))
