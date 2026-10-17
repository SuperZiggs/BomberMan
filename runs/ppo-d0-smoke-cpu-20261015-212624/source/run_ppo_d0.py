"""T4-D0 technical smoke: AC versus one unclipped ratio update, no held-out evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from audit_t2 import digest, dump
import train_actor_critic as trainer
from training_state import load_training_state, run_lock, runtime_state, source_hashes

ROOT = Path(__file__).resolve().parent
B = ROOT / 'runs/pressure-C-next/final_phaseC.pt'
B_SHA = '93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60'
SOURCE_FILES = (*source_hashes().keys(), 'run_ppo_d0.py', 'test_ppo_d0.py', 'audit_t2.py')


def run(output, device='cpu'):
    output = Path(output).resolve()
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested device is unavailable')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if digest(B) != B_SHA:
        raise ValueError('Warm-start B changed')
    with run_lock(output):
        return _run(output, device)


def _run(output, device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    hashes = {name: digest(ROOT/name) for name in SOURCE_FILES}
    output.mkdir(parents=True)
    (output/'source').mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(ROOT/name, output/'source'/name)
    dump(output/'config.json', {'stage': 'T4-D0', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'full_protocol': False, 'runtime': runtime_state(), 'device': device, 'source_sha256': hashes,
        'warm_start': str(B), 'warm_start_sha256': B_SHA, 'episodes_per_mode': 20, 'max_steps': 20,
        'train_seed': 2026101506, 'estimator': 'mc', 'normalize_advantage': False, 'critic_target': 'MC',
        'teacher': 'local', 'wall_destroy_bonus': .15, 'epochs': 1, 'clipping': False, 'evaluation': None})
    started, modes, first, reason = perf_counter(), {}, {}, None
    try:
        for objective in ('ac', 'ratio'):
            folder = output/objective
            folder.mkdir()
            cfg = dict(episodes=20, phase='C', board_size=9, num_agents=2, max_steps=20, lr=1e-4,
                gamma=.99, seed=2026101506, resume_from=str(B), device=device, checkpoint_dir=str(folder),
                run_report=False, validation_every=0, log_every=10, policy_objective=objective,
                attack_teacher='local', advantage_estimator='mc', normalize_advantage=False,
                advantage_diagnostics=True, reward_diagnostics=True, save_training_state=True)
            dump(folder/'config.json', cfg)
            rows, actions, optimizers, gradients = [], [], [], []
            real_adam, real_clip = trainer.optim.Adam, torch.nn.utils.clip_grad_norm_
            def adam(*args, **kwargs):
                opt = real_adam(*args, **kwargs)
                optimizers.append(opt)
                return opt
            def clip(parameters, *args, **kwargs):
                parameters = list(parameters)
                if not gradients:
                    gradients.extend(p.grad.detach().cpu().clone() if p.grad is not None else None for p in parameters)
                return real_clip(parameters, *args, **kwargs)
            def frame(env, ep, aid, sampled, opponents):
                if ep == 1 and env.step_count:
                    actions.append([aid, list(sampled)])
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            random.seed(2026101506)
            with (folder/'metrics.jsonl').open('w', encoding='utf-8') as metrics:
                def record(row):
                    opt = optimizers[0]
                    updates = {int(state['step'].item()) for state in opt.state.values()}
                    assert updates == {row['episode']}
                    row = {**row, 'optimizer_steps': row['episode']}
                    rows.append(row)
                    metrics.write(json.dumps(row)+'\n')
                    metrics.flush()
                    if row['episode'] == 1:
                        first[objective] = {'actions': actions, 'row': row, 'gradients': gradients,
                            'parameters': [p.detach().cpu().clone() for group in opt.param_groups for p in group['params']],
                            'moments': [{k: v.detach().cpu().clone() for k, v in state.items()} for state in opt.state.values()]}
                with patch.object(trainer.optim, 'Adam', side_effect=adam), \
                     patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=clip):
                    net, rewards, _ = trainer.train(**cfg, on_step=frame, on_episode=record)
            state = load_training_state(folder/'training_state.pt')
            assert len(rows) == len(rewards) == state['episode'] == 20 and len(optimizers) == 1
            assert state['scheduler']['last_epoch'] == 20
            final = torch.load(folder/'final_phaseC.pt', map_location='cpu', weights_only=True)
            assert all(torch.equal(final[k], v.cpu()) and torch.isfinite(v).all().item() for k, v in state['model'].items())
            for row in rows:
                assert row['attack_teacher'] == 'local' and not row['advantage']['normalize_enabled']
                assert row['reward_component_max_error'] <= 1e-10
                if objective == 'ratio':
                    d = row['ratio_update']
                    assert d['samples'] == row['learner_steps'] and d['epochs'] == 1 and not d['clipped']
                    assert d['terminal_transitions'] == 1 and d['ratio_max_abs_error_from_one'] < 1e-4
            modes[objective] = {'episodes': len(rows), 'optimizer_steps': rows[-1]['optimizer_steps'],
                'learner_steps': sum(r['learner_steps'] for r in rows), 'finite_weights': True,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                'training_state_sha256': digest(folder/'training_state.pt'),
                'checkpoints': {kind: digest(folder/f'{kind}_phaseC.pt') for kind in ('best', 'final')}}
            print(f'[D0 {objective}] 20 episodes / 20 Adam updates complete', flush=True)
        ac, ratio = first['ac'], first['ratio']
        assert ac['actions'] == ratio['actions'] and ac['row']['reward'] == ratio['row']['reward']
        loss_deltas = {k: abs(ac['row'][k] - ratio['row'][k])
                       for k in ('value_loss', 'escape_loss', 'combat_loss', 'navigation_loss')}
        assert max(loss_deltas.values()) <= 2e-5
        for a, b in zip(ac['gradients'], ratio['gradients']):
            assert (a is None) == (b is None)
            if a is not None:
                torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-6)
        for a, b in zip(ac['parameters'], ratio['parameters']):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-5)
        for a, b in zip(ac['moments'], ratio['moments']):
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(a[key], b[key], rtol=2e-4, atol=3e-7)
        dump(output/'first_update_parity.json', {'actions_and_rewards_identical': True,
            'auxiliary_and_value_loss_abs_delta': loss_deltas,
            'gradient_max_abs_delta': max((a-b).abs().max().item() for a, b in zip(ac['gradients'],ratio['gradients']) if a is not None),
            'parameter_max_abs_delta': max((a-b).abs().max().item() for a, b in zip(ac['parameters'],ratio['parameters'])),
            'ratio': ratio['row']['ratio_update'],
            'note': 'Policy loss scalars differ; compare gradients. Float32/Adam parity uses registered tolerances.'})
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        unchanged = digest(B) == B_SHA and hashes == {name: digest(ROOT/name) for name in SOURCE_FILES}
        summary = {'status': 'complete' if reason is None and len(modes) == 2 and unchanged else 'incomplete',
            'full_protocol': False, 'reason': reason, 'hashes_unchanged': unchanged, 'modes': modes,
            'decision': 'retain_original', 'elapsed_seconds': perf_counter()-started,
            'limits': 'D0 technical smoke only; no PPO clipping, repeated epochs, held-out games or win-rate claim.'}
        dump(output/'summary.json', summary)
        shutil.make_archive(str(output)+'-send', 'zip', root_dir=output)
        print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs'/('ppo-d0-smoke-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    if run(args.output_dir, args.device)['status'] != 'complete':
        raise SystemExit(1)
