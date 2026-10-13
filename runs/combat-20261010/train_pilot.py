import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train_actor_critic import train

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)
directory = Path(__file__).resolve().parent
config = dict(phase="E", episodes=300, board_size=11, num_agents=4,
              max_steps=500, snapshot_every=1000, log_every=25,
              resume_from=str(ROOT / "checkpoints" / "best_phaseE_4a.pt"),
              checkpoint_dir=str(directory), device="cpu", run_report=False)
(directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
with (directory / "metrics.jsonl").open("w", encoding="utf-8") as log:
    def record(row):
        log.write(json.dumps(row) + "\n")
        log.flush()
    train(**config, on_episode=record)
