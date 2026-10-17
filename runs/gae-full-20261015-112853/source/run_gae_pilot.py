"""T4-B1: paired MC/GAE(.95) actor advantage, with a 150-minute maximum budget."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter

import torch

from audit_t2 import digest, dump
from pilot_two_bomb import stage_timer
from run_combat_move_pilot import (ROOT, WARM_START, B_SHA256, BaseNet, SOURCE_FILES as SHARED_FILES,
    fresh_test_check, train_arm, pair_ready, evaluate_models, comparison_gate, probe_summary)
from run_t3_curriculum import skill_probe
import train_actor_critic as trainer

TRAIN_SEED = 2026101504
TEST_SEED = 35000
ARMS = {'control': 'mc', 'gae': 'gae'}
SOURCE_FILES = (*SHARED_FILES, 'run_gae_pilot.py', 'test_gae_pilot.py', 'test_gae.py')


def preflight(stop):
    if stop.is_set():
        return None
    rewards = torch.tensor([1., 2., 3.], dtype=torch.float64)
    values = torch.tensor([.5, .25, .75], dtype=torch.float64)
    bootstrap = torch.tensor([True, True, False])
    next_values = torch.tensor([.25, .75, 999.], dtype=torch.float64)
    for lam in (0., .95, 1.):
        advantage, delta = trainer.gae_advantages(rewards, values, next_values,
            bootstrap, bootstrap, .99, lam)
        assert torch.isfinite(advantage).all() and not advantage.requires_grad
        if lam == 0.:
            torch.testing.assert_close(advantage, delta)
        elif lam == 1.:
            returns = torch.tensor([1 + .99 * (2 + .99 * 3), 2 + .99 * 3, 3], dtype=torch.float64)
            torch.testing.assert_close(advantage, returns - values)
    return {'passed': True, 'cases': 3, 'gae_lambda': .95, 'normalization': False,
            'critic_target': 'unchanged MC returns', 'timeout': 'terminal value zero'}


def run(output, device='cpu', smoke=False, budget_minutes=150):
    if not 0 < budget_minutes <= 150:
        raise ValueError('Budget must be in (0, 150] minutes')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested training device is unavailable')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if digest(WARM_START) != B_SHA256:
        raise ValueError('Warm-start B changed')
    if not smoke:
        fresh_test_check(TEST_SEED)
    torch.set_num_threads(1)
    sources = {name: digest(ROOT / name) for name in SOURCE_FILES}
    output.mkdir(parents=True)
    (output / 'source').mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, output / 'source' / name)
    config = {'registered_utc': datetime.now(timezone.utc).isoformat(), 'experiment': 'T4-B1',
        'full_protocol': not smoke, 'warm_start': str(WARM_START), 'warm_start_sha256': B_SHA256,
        'source_sha256': sources, 'arms': ARMS, 'train_seed': TRAIN_SEED,
        'episodes_per_arm': 20 if smoke else 200, 'phase': 'C', 'board_size': 9, 'num_agents': 2,
        'max_steps': 20 if smoke else 500, 'lr': 1e-4, 'gamma': 0.99, 'train_device': device,
        'attack_teacher': 'local', 'critic_target': 'MC returns', 'advantage_scope': 'learner trajectory',
        'normalization': False, 'gae_lambda': .95, 'timeout': 'terminal value zero',
        'validation_seed': 21000, 'validation_episodes': 2 if smoke else 100,
        'validation_every': 10 if smoke else 100,
        'evaluation': {'seed': 39850 if smoke else TEST_SEED, 'games_per_model': 2 if smoke else 400,
                       'last_board_seed': 39850 if smoke else TEST_SEED + 199, 'device': 'cpu', 'threads': 1},
        'budget_minutes': {'overall': budget_minutes, 'preflight': 15, 'training': 85, 'evaluation': 50},
        'only_treatment_change': 'Actor MC versus GAE(.95); leave critic MC targets, normalization off and auxiliary losses unchanged',
        'gate': {'gain_vs_both_pp': 5, 'ci95_lower_above': 0, 'suicide_not_above_either': True},
        'torch': torch.__version__, 'hardware': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}
    dump(output / 'config.json', config)
    (ROOT / 'runs' / ('gae-smoke-last-path.txt' if smoke else 'gae-last-path.txt')).write_text(str(output), encoding='utf-8')
    started = perf_counter()
    arms, results, comparisons, selected, reason, eligible = {}, {}, {}, {}, None, False
    timer = None

    def phase(seconds):
        return stage_timer(min(seconds, max(0, budget_minutes * 60 - (perf_counter() - started))))

    try:
        stop, timer = phase(15 * 60)
        profile = preflight(stop)
        dump(output / 'preflight.json', profile)
        if profile is None or not profile['passed']:
            reason = 'preflight_incomplete_or_failed'
        else:
            net = BaseNet(board_size=9)
            trainer.smart_load_weights(net, str(WARM_START), 'cpu')
            baseline = skill_probe(net, 1, stop, smoke=False)
            if baseline is None:
                reason = 'baseline_skill_incomplete'
            else:
                dump(output / 'baseline_skill.json', probe_summary(baseline))
                timer.cancel()
                stop, timer = phase(85 * 60)
                for arm, estimator in ARMS.items():
                    if stop.is_set():
                        break
                    arms[arm] = train_arm(output, arm, device, smoke, stop, baseline,
                        attack_teacher='local', normalize_advantage=False, advantage_estimator=estimator,
                        gae_lambda=.95, train_seed=TRAIN_SEED,
                        smoke_episodes=20, smoke_steps=20)
                    if not (arms[arm]['training_complete'] and arms[arm]['validation_complete'] and arms[arm]['skill_probes_complete']):
                        break
                if not pair_ready(arms, ARMS):
                    reason = 'training_validation_or_skill_incomplete'
                else:
                    assert sources == {name: digest(ROOT / name) for name in SOURCE_FILES}
                    assert digest(WARM_START) == B_SHA256
                    selected = {'original': {'path': str(WARM_START), 'sha256': B_SHA256},
                                **{arm: arms[arm]['checkpoints']['best'] for arm in ARMS}}
                    assert all(digest(Path(r['path'])) == r['sha256'] for r in selected.values())
                    dump(output / 'selected_models.json', selected)
                    dump(output / 'test_opened.json', {'seed': 39850 if smoke else TEST_SEED,
                         'num_episodes': 2 if smoke else 400, 'num_agents': 2})
                    timer.cancel()
                    stop, timer = phase(50 * 60)
                    results = evaluate_models(output, stop, smoke, arm_names=ARMS,
                                              test_seed=TEST_SEED, smoke_seed=39850)
                    if len(results) != 3:
                        reason = 'evaluation_incomplete'
                    elif not smoke:
                        comparisons, eligible = comparison_gate(results, 'gae')
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        if timer is not None:
            timer.cancel()
        try:
            checkpoint_ok = all(digest(Path(r['path'])) == r['sha256'] for r in selected.values())
            unchanged = digest(WARM_START) == B_SHA256 and sources == {
                name: digest(ROOT / name) if (ROOT / name).exists() else None for name in SOURCE_FILES}
        except FileNotFoundError:
            checkpoint_ok, unchanged = False, False
        unchanged &= checkpoint_ok
        if not unchanged:
            reason, eligible = 'source_or_warm_start_changed', False
        complete = reason is None and pair_ready(arms, ARMS) and len(results) == 3 and unchanged
        summary = {'status': 'complete' if complete else 'incomplete', 'full_protocol': not smoke,
            'reason': reason, 'arms': arms, 'evaluation_complete': len(results) == 3,
            'hashes_unchanged': unchanged, 'selected_checkpoints_unchanged': checkpoint_ok,
            'comparisons': comparisons, 'eligible_for_replication': complete and not smoke and eligible,
            'decision': 'replicate_before_replacing' if complete and not smoke and eligible else 'retain_original',
            'elapsed_seconds': perf_counter() - started,
            'models': {name: {k: v for k, v in r.items() if k != 'games'} for name, r in results.items()}}
        dump(output / 'summary.json', summary)
        shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--budget-minutes', type=float, default=150)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / ('gae-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.device, args.smoke, args.budget_minutes)
