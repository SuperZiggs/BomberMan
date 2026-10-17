"""T4-B0 technical smoke only: MC, GAE(1), GAE(.95); never evaluates or promotes a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
from unittest.mock import patch

import torch

from audit_t2 import digest, dump
import train_actor_critic as trainer

ROOT = Path(__file__).resolve().parent
B = ROOT / 'runs/pressure-C-next/final_phaseC.pt'
B_SHA = '93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60'
SOURCE_FILES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'tactical_teacher.py',
                'combat_move_teacher.py', 'audit_t2.py', 'run_gae_b0.py', 'test_gae.py')


def run(output, device='cpu'):
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device is unavailable')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if digest(B) != B_SHA:
        raise ValueError('Warm-start B changed')
    torch.set_num_threads(1)
    hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    output.mkdir(parents=True)
    (output / 'source').mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, output / 'source' / name)
    config = {'registered_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'T4-B0',
        'full_protocol': False, 'device': device, 'torch': torch.__version__,
        'hardware': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        'warm_start': str(B), 'warm_start_sha256': B_SHA, 'source_sha256': hashes,
        'episodes_per_mode': 20, 'max_steps': 20, 'train_seed': 2026101503,
        'normalization': False, 'critic_target': 'MC', 'teacher': 'local',
        'evaluation': None, 'decision': 'retain_original'}
    dump(output / 'config.json', config)
    summaries, reason = {}, None
    started = perf_counter()
    try:
        for name, estimator, lam in (('mc', 'mc', 1.), ('gae1', 'gae', 1.), ('gae095', 'gae', .95)):
            folder = output / name
            folder.mkdir()
            cfg = dict(episodes=20, phase='C', board_size=9, num_agents=2, max_steps=20,
                lr=1e-4, gamma=.99, seed=2026101503, resume_from=str(B), device=device,
                checkpoint_dir=str(folder), run_report=False, validation_every=0, log_every=10,
                attack_teacher='local', normalize_advantage=False, advantage_diagnostics=True,
                advantage_estimator=estimator, gae_lambda=lam)
            dump(folder / 'config.json', cfg)
            optimizers, rows = [], []
            real_adam = trainer.optim.Adam

            def adam(*args, **kwargs):
                value = real_adam(*args, **kwargs)
                optimizers.append(value)
                return value

            with (folder / 'metrics.jsonl').open('w', encoding='utf-8') as metrics:
                def record(row):
                    steps = int(next(iter(optimizers[0].state.values()))['step'].item())
                    row = {**row, 'optimizer_steps': steps}
                    rows.append(row)
                    metrics.write(json.dumps(row) + '\n')
                    metrics.flush()
                with patch.object(trainer.optim, 'Adam', side_effect=adam):
                    net, rewards, _ = trainer.train(**cfg, on_episode=record)
            assert len(rows) == len(rewards) == 20 and len(optimizers) == 1
            assert [r['optimizer_steps'] for r in rows] == list(range(1, 21))
            assert all(torch.isfinite(v).all().item() for v in net.state_dict().values())
            for row in rows:
                d = row['advantage']
                assert not d['normalize_enabled'] and d['samples'] == row['learner_steps']
                if estimator == 'gae':
                    assert d['estimator'] == 'gae' and d['gae_lambda'] == lam
                    assert d['terminal_transitions'] == 1 and d['last_boundary'] is not None
                    assert d['bootstrap_transitions'] == d['continuations'] == d['samples'] - 1
            summaries[name] = {'episodes': len(rows), 'optimizer_instances': len(optimizers),
                'optimizer_steps': rows[-1]['optimizer_steps'], 'learner_steps': sum(r['learner_steps'] for r in rows),
                'finite_weights': True, 'last_boundaries': {k: sum(r['advantage'].get('last_boundary') == k for r in rows)
                    for k in ('time_limit', 'match_end', 'learner_death')} if estimator == 'gae' else None,
                'checkpoints': {kind: digest(folder / f'{kind}_phaseC.pt') for kind in ('best', 'final')}}
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        unchanged = digest(B) == B_SHA and hashes == {name: digest(ROOT / name) for name in SOURCE_FILES}
        summary = {'status': 'complete' if reason is None and len(summaries) == 3 and unchanged else 'incomplete',
            'full_protocol': False, 'reason': reason, 'hashes_unchanged': unchanged, 'modes': summaries,
            'decision': 'retain_original', 'elapsed_seconds': perf_counter() - started,
            'limits': 'Technical smoke only; no B1 pilot, held-out evaluation, win-rate claim or checkpoint replacement'}
        dump(output / 'summary.json', summary)
        shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / ('gae-b0-smoke-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.device)
