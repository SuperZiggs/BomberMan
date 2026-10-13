"""Compare a checkpoint with the saved or current runtime on identical seeded boards."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', choices=['before', 'current'], default='current')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--seed', type=int, default=8000)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent / 'before' if args.runtime == 'before' else ROOT))
import torch
torch.set_num_threads(1)
if args.runtime == 'before':
    spec = importlib.util.spec_from_file_location('seeded_evaluator', ROOT / 'evaluate.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
else:
    import evaluate as evaluator
policy = evaluator.make_ac_fn(str(ROOT / args.checkpoint), 9, 'cpu')
def progress(done, total):
    if done % 25 == 0:
        print(args.runtime, done, '/', total, flush=True)
result = evaluator.evaluate_seeded(policy, num_episodes=args.episodes, seed=args.seed, on_progress=progress)
assert len(result['games']) == args.episodes
assert all(result['by_corner'][str(i)]['num_episodes'] == args.episodes // 2 for i in range(2))
result.update(checkpoint=args.checkpoint, runtime=args.runtime)
(ROOT / args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ['games', 'by_corner']}), flush=True)
