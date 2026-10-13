"""Fixed independent comparisons, same boards/corners/RNG and unchanged weights."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import evaluate as ev
import train_actor_critic as ac

folder = Path(__file__).parent
spec = importlib.util.spec_from_file_location('before_escape', folder / 'before/rule_based_bot.py')
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
current_mask = ac._safe_action_mask
torch.set_num_threads(1)
paths = {'final': ROOT / 'runs/pressure-C-next/final_phaseC.pt',
         'warm': ROOT / 'runs/combat-20261012/combat_bc_phaseC.pt'}
hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
code_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in ('rule_based_bot.py', 'train_actor_critic.py', 'bomberman_env.py', 'evaluate.py')}
protocol = {'seed': 16000, 'board_size': 9, 'num_agents': 2, 'num_episodes': 200, 'max_steps': 500,
            'device': 'cpu', 'torch_threads': 1,
            'candidate': 'final', 'selection_seed': 15000,
            'selection': 'Final after escape fix: ActiveWR 30%, self 1%; warm after: 23%, self 0.5%.',
            'model_sha256': hashes, 'code_sha256': code_hashes,
            'runs': ['final_before', 'final_after', 'warm_after']}
assert not (folder / 'comparison_16000.json').exists(), 'Do not overwrite a completed benchmark'
(folder / 'protocol_16000.json').write_text(json.dumps(protocol, indent=2))
results = {}
elapsed = {}
for label, checkpoint, mask in [('final_before', 'final', before._safe_action_mask),
                                ('final_after', 'final', current_mask),
                                ('warm_after', 'warm', current_mask)]:
    destination = folder / ('test_' + label + '_16000.json')
    assert not destination.exists(), destination
    ac._safe_action_mask = mask
    start = time.perf_counter()
    result = ev.evaluate_seeded(ev.make_ac_fn(str(paths[checkpoint]), 9, 'cpu'),
                               board_size=9, num_agents=2, num_episodes=200, seed=16000, max_steps=500,
                               on_progress=lambda n, total: print(label, n, '/', total, flush=True) if n % 50 == 0 else None)
    elapsed[label] = time.perf_counter() - start
    destination.write_text(json.dumps(result, indent=2))
    results[label] = result
    print(label, json.dumps({k: v for k, v in result.items() if k not in ('games', 'by_corner')}), flush=True)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name] for name, path in paths.items())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in code_hashes.items())

ac._safe_action_mask = current_mask
keys = [[(g['seed'], g['aid']) for g in r['games']] for r in results.values()]
assert all(key == keys[0] for key in keys)
old, new = results['final_before']['games'], results['final_after']['games']
paired = {key: {'gained': sum(a[key] < b[key] for a, b in zip(old, new)),
                'lost': sum(a[key] > b[key] for a, b in zip(old, new))}
          for key in ('win', 'active_win', 'suicide', 'draw')}
summary = {name: {k: v for k, v in result.items() if k != 'games'} for name, result in results.items()}
summary.update(protocol=protocol, elapsed_seconds=elapsed, paired_final=paired)
(folder / 'comparison_16000.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
