"""Finish the fixed 200-episode comparison, without the pilot's time limit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic

import torch

from evaluate import evaluate_seeded, make_ac_fn
from pilot_two_bomb import complete_validation, paired_active_bootstrap, save_json
from train_actor_critic import train


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'runs/two-bomb-pilot-20261014'
CORE_FILES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py',
              'tactical_teacher.py', 'evaluate.py')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_inputs():
    manifest = json.loads((REFERENCE / 'config.json').read_text(encoding='utf-8'))
    checkpoint = Path(manifest['training']['resume_from'])
    if digest(checkpoint) != manifest['warm_start_sha256']:
        raise ValueError('Original checkpoint changed; cannot reuse this control run')
    for filename in CORE_FILES:
        if digest(ROOT / filename) != manifest['source_sha256'][filename]:
            raise ValueError(f'{filename} changed; rerun both teachers for a fair comparison')
    local_config = json.loads((REFERENCE / 'local/config.json').read_text(encoding='utf-8'))
    if local_config != {**manifest['training'], 'attack_teacher': 'local'}:
        raise ValueError('Control training configuration differs')
    metrics = [json.loads(line) for line in (REFERENCE / 'local/metrics.jsonl').read_text().splitlines()]
    validations = [json.loads(line) for line in (REFERENCE / 'local/validation_metrics.jsonl').read_text().splitlines()]
    if [row['episode'] for row in metrics] != list(range(1, 201)) or not complete_validation(validations):
        raise ValueError('Control must have 200 episodes and complete 0/100/200 validations')
    local = REFERENCE / 'local/best_phaseC.pt'
    recorded = json.loads((REFERENCE / 'evaluation_local_25000.json').read_text())
    if digest(local) != recorded['checkpoint_sha256']:
        raise ValueError('Control checkpoint changed')
    return manifest, checkpoint, local


def run(output):
    torch.set_num_threads(1)
    manifest, checkpoint, local = reference_inputs()
    config = manifest['training'].copy()
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('The control used CUDA; activate the torch environment first')
    output.mkdir(parents=True, exist_ok=False)
    directory = output / 'two_bomb'
    directory.mkdir()
    source = output / 'source'
    source.mkdir()
    for filename in (*CORE_FILES, 'pilot_two_bomb.py', 'run_two_bomb_complete.py'):
        shutil.copy2(ROOT / filename, source / filename)
    config.update(attack_teacher='two_bomb', checkpoint_dir=str(directory))
    save_json(directory / 'config.json', config)
    save_json(output / 'config.json', {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'training': config,
        'control_run': str(REFERENCE), 'warm_start_sha256': digest(checkpoint),
        'control_sha256': digest(local), 'source_sha256': {f: digest(ROOT / f) for f in CORE_FILES},
        'evaluation': {'episodes': 400, 'seed': 26000, 'last_seed': 26199, 'device': 'cpu'},
        'time_limit': None, 'torch': torch.__version__,
        'hardware': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
    })
    started = monotonic()
    print(f'Output: {output}\nReusing completed local control; training two_bomb from the original for 200 episodes.', flush=True)
    with (directory / 'metrics.jsonl').open('w', encoding='utf-8') as handle:
        def metrics(row):
            handle.write(json.dumps(row) + '\n')
            handle.flush()
        _, rewards, _ = train(**config, on_episode=metrics)
    rows = [json.loads(line) for line in (directory / 'validation_metrics.jsonl').read_text().splitlines()]
    if len(rewards) != 200 or not complete_validation(rows):
        raise RuntimeError('Training incomplete; do not interpret it as a completed comparison')
    reference_inputs()  # Reject concurrent edits before evaluating the reused control.
    results = {}
    for name, model in [('original', checkpoint), ('local', local),
                        ('two_bomb', directory / 'best_phaseC.pt')]:
        policy = make_ac_fn(str(model), 9, 'cpu')
        def progress(done, total):
            if done % 50 == 0:
                print(f'[Evaluation] {name}: {done}/{total}', flush=True)
        result = evaluate_seeded(policy, 9, 2, 400, 26000, 500, on_progress=progress)
        if result is None or len(result['games']) != 400:
            raise RuntimeError(f'Incomplete evaluation: {name}')
        result.update(checkpoint=str(model.resolve()), checkpoint_sha256=digest(model),
                      passive_win_rate=result['win_rate'] - result['active_win_rate'])
        for corner in result['by_corner'].values():
            corner['passive_win_rate'] = corner['win_rate'] - corner['active_win_rate']
        results[name] = result
        save_json(output / f'evaluation_{name}_26000.json', result)
    comparisons = {name: paired_active_bootstrap(results['two_bomb'], results[name])
                   for name in ('original', 'local')}
    eligible = (all(c['difference_pp'] >= 5 and c['ci95_pp'][0] > 0 for c in comparisons.values())
                and all(results['two_bomb']['suicide_rate'] <= results[name]['suicide_rate']
                        for name in ('original', 'local')))
    summary = {'training_complete': True, 'completed_episodes': 200, 'evaluation_complete': True,
               'comparisons': comparisons, 'eligible_for_replication': eligible,
               'decision': 'replicate_before_replacing' if eligible else 'retain_original',
               'best_episode': json.loads((directory / 'best_validation.json').read_text())['episode'],
               'elapsed_seconds': monotonic() - started,
               'models': {name: {k: v for k, v in result.items() if k != 'games'}
                          for name, result in results.items()}}
    save_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Send summary.json, config.json, two_bomb/metrics.jsonl, two_bomb/validation_metrics.jsonl, and the three evaluation JSON files from {output}', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / ('two-bomb-complete-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    run(args.output.resolve())
