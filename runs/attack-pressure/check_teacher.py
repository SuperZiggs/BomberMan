"""Check the production teacher on the same development cases as the prototype."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate import evaluate_seeded
from probe import teacher

torch.set_num_threads(1)
folder = Path(__file__).parent
baseline = json.loads((folder/'teacher_probe.json').read_text())['original_teacher']
candidate = evaluate_seeded(teacher(False), board_size=9, num_agents=2,
                            num_episodes=40, seed=6300, max_steps=500)
assert [(g['seed'],g['aid']) for g in candidate['games']] == [(g['seed'],g['aid']) for g in baseline['games']]
(folder/'teacher_production.json').write_text(json.dumps({'original_teacher':baseline,'production_teacher':candidate},indent=2))
print(json.dumps({k:v for k,v in candidate.items() if k not in ('games','by_corner')}),flush=True)
