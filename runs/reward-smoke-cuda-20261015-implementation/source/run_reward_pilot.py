"""T4-R1: paired wall reward .15/.075, with a 150-minute maximum budget."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import math
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

TRAIN_SEED = 2026101505
TEST_SEED = 36000
ARMS = {'control': .15, 'half_wall': .075}
SOURCE_FILES = (*SHARED_FILES, 'run_reward_pilot.py', 'test_reward_pilot.py', 'test_reward_audit.py')
R0 = ROOT / 'runs/r0-full-20261015-153908'
PREFLIGHT_INPUTS = tuple(R0 / name for name in ('summary.json', 'config.json', 'steps_original.jsonl.gz'))


def preflight(stop):
    """Reprice the fixed R0 B action tapes; policy, physics and other terms stay fixed."""
    config = json.loads((R0 / 'config.json').read_text(encoding='utf-8-sig'))
    summary = json.loads((R0 / 'summary.json').read_text(encoding='utf-8-sig'))
    assert summary['status'] == 'complete' and summary['full_protocol'] and summary['games_completed'] == 300
    for name in ('train_actor_critic.py', 'bomberman_env.py', 'rule_based_bot.py'):
        assert digest(ROOT / name) == config['source_sha256'][name], name
    steps, eligible_walls, max_error, kills_by_game = 0, 0, 0., {}
    for line in gzip.open(R0 / 'steps_original.jsonl.gz', 'rt', encoding='utf-8'):
        if stop.is_set():
            return None
        row = json.loads(line)
        aid, info = row['aid'], row['info']
        info['kills'] = [tuple(pair) for pair in info['kills']]
        for name in ('items_collected', 'walls_destroyed'):
            info[name] = {int(key): value for key, value in info[name].items()}
        key = (row['seed'], aid)
        kills_by_game[key] = kills_by_game.get(key, 0) + sum(k == aid and v != aid for k, v in info['kills'])
        kills = kills_by_game[key]
        parts = [trainer._reward_components(row['env_reward'], aid, info, kills, True, coefficient)
                 for coefficient in ARMS.values()]
        count = info['walls_destroyed'].get(aid, 0) if info['alive'][aid] else 0
        delta = (ARMS['half_wall'] - ARMS['control']) * count
        assert all(math.isclose(parts[0][k], parts[1][k], rel_tol=0, abs_tol=1e-10)
                   for k in parts[0] if k != 'wall')
        error = abs((sum(parts[1].values()) - sum(parts[0].values())) - delta)
        assert error <= 1e-10
        assert math.isclose(sum(parts[0].values()), row['shaped_reward'], rel_tol=0, abs_tol=1e-10)
        steps += 1
        eligible_walls += count
        max_error = max(max_error, error)
    assert steps == 39338 and eligible_walls > 0
    return {'passed': True, 'transitions': steps, 'eligible_walls': eligible_walls,
            'max_delta_error': max_error, 'only_changed_component': 'wall'}


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
    input_hashes = {str(path): digest(path) for path in PREFLIGHT_INPUTS}
    sources = {name: digest(ROOT / name) for name in SOURCE_FILES}
    output.mkdir(parents=True)
    (output / 'source').mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, output / 'source' / name)
    config = {'registered_utc': datetime.now(timezone.utc).isoformat(), 'experiment': 'T4-R1',
        'full_protocol': not smoke, 'warm_start': str(WARM_START), 'warm_start_sha256': B_SHA256,
        'source_sha256': sources, 'preflight_input_sha256': input_hashes, 'arms': ARMS, 'train_seed': TRAIN_SEED,
        'episodes_per_arm': 20 if smoke else 200, 'phase': 'C', 'board_size': 9, 'num_agents': 2,
        'max_steps': 20 if smoke else 500, 'lr': 1e-4, 'gamma': 0.99, 'train_device': device,
        'attack_teacher': 'local', 'critic_target': 'MC returns', 'advantage_scope': 'learner trajectory',
        'normalization': False, 'advantage_estimator': 'mc', 'reward_diagnostics': True,
        'contact_diagnostics': True, 'timeout': 'terminal value zero',
        'validation_seed': 21000, 'validation_episodes': 2 if smoke else 100,
        'validation_every': 10 if smoke else 100,
        'evaluation': {'seed': 39860 if smoke else TEST_SEED, 'games_per_model': 2 if smoke else 400,
                       'last_board_seed': 39860 if smoke else TEST_SEED + 199, 'device': 'cpu', 'threads': 1},
        'budget_minutes': {'overall': budget_minutes, 'preflight': 15, 'training': 100, 'evaluation': 50},
        'only_treatment_change': 'Wall bonus .15 versus .075; keep item/kill/win/death/time, MC targets, raw advantage, mask and local labels unchanged',
        'gate': {'gain_vs_both_pp': 5, 'ci95_lower_above': 0, 'suicide_not_above_either': True},
        'torch': torch.__version__, 'hardware': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}
    dump(output / 'config.json', config)
    (ROOT / 'runs' / ('reward-smoke-last-path.txt' if smoke else 'reward-last-path.txt')).write_text(str(output), encoding='utf-8')
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
                stop, timer = phase(100 * 60)
                for arm, coefficient in ARMS.items():
                    if stop.is_set():
                        break
                    arms[arm] = train_arm(output, arm, device, smoke, stop, baseline,
                        attack_teacher='local', normalize_advantage=False, advantage_estimator='mc',
                        wall_destroy_bonus=coefficient, reward_diagnostics=True, contact_diagnostics=True,
                        train_seed=TRAIN_SEED,
                        smoke_episodes=20, smoke_steps=20)
                    if not (arms[arm]['training_complete'] and arms[arm]['validation_complete'] and arms[arm]['skill_probes_complete']):
                        break
                if not pair_ready(arms, ARMS):
                    reason = 'training_validation_or_skill_incomplete'
                else:
                    assert sources == {name: digest(ROOT / name) for name in SOURCE_FILES}
                    assert digest(WARM_START) == B_SHA256
                    assert all(digest(Path(path)) == sha for path, sha in input_hashes.items())
                    selected = {'original': {'path': str(WARM_START), 'sha256': B_SHA256},
                                **{arm: arms[arm]['checkpoints']['best'] for arm in ARMS}}
                    assert all(digest(Path(r['path'])) == r['sha256'] for r in selected.values())
                    dump(output / 'selected_models.json', selected)
                    dump(output / 'test_opened.json', {'seed': 39860 if smoke else TEST_SEED,
                         'num_episodes': 2 if smoke else 400, 'num_agents': 2})
                    timer.cancel()
                    stop, timer = phase(50 * 60)
                    results = evaluate_models(output, stop, smoke, arm_names=ARMS,
                                              test_seed=TEST_SEED, smoke_seed=39860, contact_diagnostics=True)
                    if len(results) != 3:
                        reason = 'evaluation_incomplete'
                    elif not smoke:
                        comparisons, eligible = comparison_gate(results, 'half_wall')
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
            unchanged &= all(digest(Path(path)) == sha for path, sha in input_hashes.items())
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
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / ('reward-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.device, args.smoke, args.budget_minutes)
