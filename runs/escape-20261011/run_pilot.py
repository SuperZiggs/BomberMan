"""Authorized 500-episode pilot; preserves the original checkpoint and all metrics."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from train_actor_critic import train
torch.set_num_threads(1)
folder = Path(__file__).parent / 'pilot-500'
folder.mkdir(exist_ok=False)
config = dict(phase='C', episodes=500, board_size=9, num_agents=2, max_steps=500,
              lr=1e-4, checkpoint_dir=str(folder),
              resume_from=str(ROOT / 'runs/20261010-135449-587798/best_phaseC.pt'),
              device='cuda' if torch.cuda.is_available() else 'cpu',
              log_every=25, run_report=False, seed=20261011,
              validation_every=250, validation_episodes=48, validation_seed=6000)
(folder / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
(folder / 'code_hashes.json').write_text(json.dumps({name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in ['bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'evaluate.py']}, indent=2), encoding='utf-8')
start = time.monotonic()
with (folder / 'metrics.jsonl').open('w', encoding='utf-8') as handle:
    def log(data):
        handle.write(json.dumps(data) + '\n')
        handle.flush()
    train(**config, on_episode=log)
rows = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines()]
assert len(rows) == 500 and [r['episode'] for r in rows] == list(range(1,501))
assert [r['learning_aid'] for r in rows] == [i % 2 for i in range(500)]
summary = {k: sum(r[k] for r in rows)/len(rows)
           for k in ['win','active_win','suicide','draw','kills','bombs','items','learner_steps']}
summary['elapsed_seconds'] = time.monotonic() - start
(folder / 'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('PILOT COMPLETE', json.dumps(summary), flush=True)
