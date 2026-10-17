"""SD3: B trained under SD2 rules, without vs with a ring observation channel."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from audit_t2 import digest, dump
from pilot_two_bomb import paired_active_bootstrap, stage_timer
from run_combat_move_pilot import (ROOT, WARM_START, B_SHA256, BaseNet, SOURCE_FILES as SHARED_FILES,
                                   fresh_test_check, train_arm, pair_ready, probe_summary)
from run_t3_curriculum import skill_probe
from run_sudden_death_experiment import evaluate_case, sudden_death_runtime
import train_actor_critic as trainer
from training_state import runtime_state, run_lock

SD2 = {'sudden_death_start': 400, 'sudden_death_interval': 40,
       'sudden_death_warning_steps': 40, 'sudden_death_max_ring': 2}
# Smoke runs 20 steps, so it uses short timings to exercise closures in training and evaluation.
SMOKE_SD = {'sudden_death_start': 8, 'sudden_death_interval': 4,
            'sudden_death_warning_steps': 6, 'sudden_death_max_ring': 2}
ARMS = {'sd_12ch': False, 'sd_13ch': True}
TRAIN_SEED = 2026101601
TEST_SEED = 40000
SMOKE_SEED = 39840
SOURCE_FILES = (*SHARED_FILES, 'sudden_death_experiment.py', 'run_sudden_death_experiment.py',
                'run_sd_training_pilot.py', 'test_sd_training_pilot.py', 'training_state.py')
METRICS = ('win', 'active_win', 'draw', 'suicide')


def gate(results, candidate):
    comparison = paired_active_bootstrap(results[candidate], results['original'])
    eligible = (comparison['difference_pp'] >= 5 and comparison['ci95_pp'][0] > 0
                and results[candidate]['suicide_rate'] <= results['original']['suicide_rate'])
    return comparison, bool(eligible)


def paired(results, left, right):
    out = {}
    for metric in METRICS:
        wrapped = [{**results[name], 'games': [{**g, 'active_win': g[metric]} for g in results[name]['games']]}
                   for name in (left, right)]
        out[metric] = paired_active_bootstrap(*wrapped)
    return out


def evaluate(output, models, stop, smoke, timings):
    seed, games, steps = (SMOKE_SEED, 2, 20) if smoke else (TEST_SEED, 400, 500)
    results = {}
    for name, path in models.items():
        stored = output / f'evaluation_{name}_{seed}.json'
        if stored.exists():
            cached = json.loads(stored.read_text(encoding='utf-8'))
            if (cached['seed'], cached['num_episodes'], cached['checkpoint_sha256']) != (seed, games, digest(path)):
                raise ValueError('Stored evaluation differs from the locked model/protocol')
            results[name] = cached
            continue

        def progress(done, total, name=name):
            if done % 25 == 0:
                print(f'[Evaluation] {name}: {done}/{total}', flush=True)
        result = evaluate_case('both_aware', games=games, seed=seed, max_steps=steps, timings=timings,
                               progress=progress, stop=stop, checkpoint=path, ring_channel=True)
        if result is None:
            break
        result.update(checkpoint=str(path), checkpoint_sha256=digest(path))
        dump(stored, result)
        results[name] = result
    return results


def run(output, device='cpu', smoke=False, budget_minutes=150, resume=False):
    if not 0 < budget_minutes <= 150:
        raise ValueError('Budget must be in (0, 150] minutes')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested training device is unavailable')
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f'Refusing to overwrite {output}')
    if resume and not (output / 'config.json').exists():
        raise ValueError('No registered run at the requested resume directory')
    with run_lock(output):
        return _run(output, device, smoke, budget_minutes, resume)


def _run(output, device, smoke, budget_minutes, resume):
    if digest(WARM_START) != B_SHA256:
        raise ValueError('Warm-start B changed')
    if not smoke and not (output / 'test_opened.json').exists():
        fresh_test_check(TEST_SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sources = {name: digest(ROOT / name) for name in SOURCE_FILES}
    timings = SMOKE_SD if smoke else SD2
    config = {'experiment': 'SD3', 'full_protocol': not smoke, 'warm_start': str(WARM_START),
        'warm_start_sha256': B_SHA256, 'source_sha256': sources, 'arms': ARMS, 'sudden_death': timings,
        'train_seed': TRAIN_SEED, 'episodes_per_arm': 20 if smoke else 200, 'phase': 'C',
        'board_size': 9, 'num_agents': 2, 'max_steps': 20 if smoke else 500, 'lr': 1e-4,
        'train_device': device, 'attack_teacher': 'local', 'advantage_estimator': 'mc',
        'opponent': 'RuleBot, sudden-death aware', 'validation_seed': 21000,
        'validation_episodes': 2 if smoke else 100, 'validation_every': 10 if smoke else 100,
        'evaluation': {'seed': SMOKE_SEED if smoke else TEST_SEED, 'games_per_model': 2 if smoke else 400,
                       'case': 'both_aware', 'ring_channel_env': True, 'device': 'cpu', 'threads': 1},
        'budget_minutes': {'overall': budget_minutes, 'training': 100, 'evaluation': 45},
        'gate': {'gain_vs_B_pp': 5, 'ci95_lower_above': 0, 'suicide_not_above_B': True},
        'runtime': runtime_state(), 'torch': torch.__version__,
        'hardware': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}
    if resume:
        stored = json.loads((output / 'config.json').read_text(encoding='utf-8'))
        if any(stored.get(key) != value for key, value in config.items()):
            raise ValueError('Resume config/source/runtime differs from the registered run')
        config = stored
    else:
        output.mkdir(parents=True)
        (output / 'source').mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(ROOT / name, output / 'source' / name)
        config['registered_utc'] = datetime.now(timezone.utc).isoformat()
        dump(output / 'config.json', config)
    deadline = datetime.fromisoformat(config['registered_utc']) + timedelta(minutes=budget_minutes)
    if resume and deadline <= datetime.now(timezone.utc):
        raise ValueError('Registered deadline expired; register a separately budgeted continuation')
    started = perf_counter()
    arms, results, comparisons, reason, eligible, timer = {}, {}, {}, None, False, None

    def phase(minutes):
        left = (deadline - datetime.now(timezone.utc)).total_seconds()
        return stage_timer(max(0, min(minutes * 60, left)))

    try:
        stop, timer = phase(100)
        for arm, ring in ARMS.items():
            if stop.is_set():
                break
            saved = output / arm / 'summary.json'
            if resume and saved.exists():
                cached = json.loads(saved.read_text(encoding='utf-8'))
                if pair_ready({arm: cached}, (arm,)):
                    if any(digest(Path(r['path'])) != r['sha256'] for r in cached['checkpoints'].values()):
                        raise ValueError('Completed arm checkpoint changed')
                    arms[arm] = cached
                    continue
            with sudden_death_runtime(timings, ring_channel=ring):
                net = BaseNet(in_channels=12 + int(ring), board_size=9)
                trainer.smart_load_weights(net, str(WARM_START), 'cpu')
                baseline = skill_probe(net, 1, stop, smoke=False)
                if baseline is None:
                    break
                dump(output / f'baseline_skill_{arm}.json', probe_summary(baseline))
                arms[arm] = train_arm(output, arm, device, smoke, stop, baseline,
                    attack_teacher='local', normalize_advantage=False, advantage_estimator='mc',
                    contact_diagnostics=True, train_seed=TRAIN_SEED, save_training_state=True,
                    resume=resume and (output / arm).exists(), smoke_episodes=20, smoke_steps=20)
            if not (arms[arm]['training_complete'] and arms[arm]['validation_complete']
                    and arms[arm]['skill_probes_complete']):
                break
        if not pair_ready(arms, ARMS):
            reason = 'training_validation_or_skill_incomplete'
        else:
            models = {'original': WARM_START,
                      **{arm: Path(arms[arm]['checkpoints']['best']['path']) for arm in ARMS}}
            selected = {name: {'path': str(path), 'sha256': digest(path)} for name, path in models.items()}
            if (output / 'selected_models.json').exists():
                if json.loads((output / 'selected_models.json').read_text(encoding='utf-8')) != selected:
                    raise ValueError('Test models are locked; selecting another checkpoint is forbidden')
            else:
                dump(output / 'selected_models.json', selected)
                dump(output / 'test_opened.json', {'seed': SMOKE_SEED if smoke else TEST_SEED,
                     'num_episodes': 2 if smoke else 400, 'num_agents': 2})
            timer.cancel()
            stop, timer = phase(45)
            results = evaluate(output, models, stop, smoke, timings)
            if len(results) != 3:
                reason = 'evaluation_incomplete'
            elif not smoke:
                for arm in ARMS:
                    comparison, ok = gate(results, arm)
                    comparisons[arm] = {'gate': comparison, 'eligible': ok,
                                        'vs_original': paired(results, arm, 'original')}
                comparisons['sd_13ch_vs_sd_12ch'] = paired(results, 'sd_13ch', 'sd_12ch')
                eligible = any(comparisons[arm]['eligible'] for arm in ARMS)
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        if timer is not None:
            timer.cancel()
        unchanged = digest(WARM_START) == B_SHA256 and sources == {
            name: digest(ROOT / name) if (ROOT / name).exists() else None for name in SOURCE_FILES}
        if not unchanged:
            reason, eligible = 'source_or_warm_start_changed', False
        complete = reason is None and pair_ready(arms, ARMS) and len(results) == 3
        summary = {'status': 'complete' if complete else 'incomplete', 'full_protocol': not smoke,
            'reason': reason, 'arms': arms, 'hashes_unchanged': unchanged, 'comparisons': comparisons,
            'eligible_for_replication': complete and not smoke and eligible,
            'decision': 'replicate_before_replacing' if complete and not smoke and eligible else 'retain_original',
            'elapsed_seconds_this_process': perf_counter() - started, 'resumed': resume,
            'models': {name: {k: v for k, v in r.items() if k != 'games'} for name, r in results.items()}}
        dump(output / 'summary.json', summary)
        shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(json.dumps({k: v for k, v in summary.items() if k not in ('arms', 'models')}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--budget-minutes', type=float, default=150)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'runs' / ('sd3-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.device, args.smoke, args.budget_minutes, args.resume)
