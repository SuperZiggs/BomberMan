"""One registered, inference-only arm using the existing seeded evaluator."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from time import monotonic

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parent.parent
sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_registration():
    registered = json.loads((OUTPUT / 'registration.json').read_text(encoding='utf-8'))
    config = registered['config']
    if digest(config) != registered['config_sha256']:
        raise ValueError('Config hash mismatch')
    for name, expected in registered['sources_normalized_lf_sha256'].items():
        live = OUTPUT / name if name == Path(__file__).name else ROOT / name
        for path in (live, OUTPUT / 'source' / name):
            actual = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            if actual != expected:
                raise ValueError(f'Source hash mismatch: {path}')
    for name, path in config['checkpoint_paths'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != config['checkpoint_sha256'][name]:
            raise ValueError(f'Checkpoint changed: {name}')
    return registered


def main():
    label = sys.argv[1]
    registered = verify_registration()
    config = registered['config']
    if label not in config['checkpoint_paths']:
        raise ValueError('Unknown evaluation arm')
    marker = json.loads((OUTPUT / 'test_opened.json').read_text(encoding='utf-8'))
    if marker['config_sha256'] != registered['config_sha256'] or marker['seed'] != config['seed']:
        raise ValueError('Opened marker mismatch')
    target = OUTPUT / f'evaluation-{label}.json'
    if target.exists():
        raise FileExistsError('Completed arm exists; never replay it')
    lock = OUTPUT / f'arm-{label}.lock.json'
    lock_text = json.dumps(dict(pid=os.getpid(), model=label,
                               started_utc=datetime.now(timezone.utc).isoformat()))
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(lock_text)
    try:
        import numpy as np
        import torch
        from evaluate import make_ac_fn, evaluate_seeded
        from progress import bar
        from run_sd53_pilot import _atomic_json
        if (sys.version, np.__version__, torch.__version__) != tuple(registered['runtime'][k] for k in ('python', 'numpy', 'torch')):
            raise ValueError('Runtime changed')
        torch.set_num_threads(1)
        started = monotonic()
        policy = make_ac_fn(config['checkpoint_paths'][label], 9, 'cpu')
        with bar(config['games_per_model'], f'{label} 9x9 vs RuleBot') as shown:
            def advance(value, total):
                shown.update(value - shown.n)
            result = evaluate_seeded(policy, board_size=9, num_agents=2,
                                     num_episodes=config['games_per_model'], seed=config['seed'],
                                     max_steps=500, sudden_death=True,
                                     sudden_death_start=475, sudden_death_interval=25,
                                     sudden_death_warning_steps=40, on_progress=advance)
        expected = [(seed, aid) for seed in range(config['seed'], config['seed'] + config['boards']) for aid in (0, 1)]
        if [(g['seed'], g['aid']) for g in result['games']] != expected:
            raise ValueError('Incomplete seed/corner coverage')
        verify_registration()
        result.update(complete=True, model=label, config_sha256=registered['config_sha256'],
                      checkpoint_sha256=config['checkpoint_sha256'][label], seconds=monotonic()-started,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(target, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'games'}, indent=2), flush=True)
    finally:
        if lock.is_file() and lock.read_text(encoding='utf-8') == lock_text:
            lock.unlink()


if __name__ == '__main__':
    main()
