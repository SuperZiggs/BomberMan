"""Short reproducible CPU warm-start; not a completed curriculum."""
import json
from functools import partial
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from bomberman_env import BombermanFFAEnv
from train_actor_critic import train

seed = 20261010
torch.set_num_threads(1)
torch.manual_seed(seed)
np.random.seed(seed)
folder = Path(__file__).resolve().parent / 'pilot-final'
folder.mkdir(exist_ok=True)
cfg = dict(phase='C', episodes=40, board_size=9, num_agents=2, max_steps=500,
           lr=1e-4, device='cuda', checkpoint_dir=str(folder), log_every=10,
           resume_from=str(ROOT / 'runs/20261010-125814-296523/best_phaseC.pt'), run_report=False)
(folder/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
with (folder/'metrics.jsonl').open('w', encoding='utf-8') as log:
    def record(data):
        log.write(json.dumps(data)+'\n')
        log.flush()
    with patch('train_actor_critic.BombermanFFAEnv', partial(BombermanFFAEnv, seed=seed)):
        train(**cfg, on_episode=record)
