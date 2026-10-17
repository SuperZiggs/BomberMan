"""Reproducible two-hour Phase C pilot; keep the original unless every gate passes."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import threading
from time import monotonic

import numpy as np
import torch

from bomberman_env import BombermanFFAEnv
from evaluate import evaluate_seeded, make_ac_fn, make_rule_based_fn
from tactical_teacher import get_two_bomb_plan
from train_actor_critic import get_action_mask, train


def paired_active_bootstrap(candidate, reference, samples=10000, seed=20261014):
    """Percentile CI in percentage points, resampling whole board seeds with both corners."""
    def boards(result):
        grouped = {}
        for game in result['games']:
            corners = grouped.setdefault(game['seed'], {})
            if game['aid'] in corners:
                raise ValueError('Duplicate board/corner')
            corners[game['aid']] = game['active_win']
        if not grouped or any(set(corners) != {0, 1} for corners in grouped.values()):
            raise ValueError('Each paired board must contain A0 and A1')
        return grouped

    a, b = boards(candidate), boards(reference)
    if a.keys() != b.keys():
        raise ValueError('Candidate and reference must use identical board seeds')
    differences = np.array([np.mean([a[s][i] - b[s][i] for i in (0, 1)]) for s in sorted(a)])
    rng = np.random.default_rng(seed)
    draws = differences[rng.integers(0, len(differences), size=(samples, len(differences)))].mean(axis=1) * 100
    low, high = np.quantile(draws, [0.025, 0.975])
    return {'difference_pp': float(differences.mean() * 100), 'ci95_pp': [float(low), float(high)],
            'board_pairs': len(differences), 'resamples': samples, 'bootstrap_seed': seed}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def complete_validation(rows):
    return ([r.get('episode') for r in rows] == [0, 100, 200]
            and all(r.get('num_episodes') == 100 and r.get('seed') == 21000 for r in rows))


def stage_timer(seconds):
    stop = threading.Event()
    timer = threading.Timer(max(0, seconds), stop.set)
    timer.daemon = True
    timer.start()
    return stop, timer


def preflight(checkpoint, stop):
    policy = make_ac_fn(str(checkpoint), 9, 'cpu')
    rows = []
    totals, reasons = Counter(), Counter()
    for seed, aid in [(19000, 0), (19003, 0), (19000, 1), (19002, 1)]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed)
        obs = env.reset()
        opponent = make_rule_based_fn(1 - aid)
        sampled = 0
        for step in range(500):
            if stop.is_set():
                return None
            agent, enemy = env.agents[aid], env.agents[1 - aid]
            distance = abs(agent.row - enemy.row) + abs(agent.col - enemy.col)
            if step % 10 == 0 and distance <= agent.blast_range + 2 and sampled < 50:
                mask = get_action_mask(env, aid)
                plan, stats = get_two_bomb_plan(env, aid, mask)
                assert stats['plans'] <= 64
                if plan:
                    assert all(mask[a] for a in plan['first_actions'])
                row = {'seed': seed, 'aid': aid, 'step': step, 'ammo': agent.ammo,
                       'active_bombs': agent.bombs_placed, **stats}
                if plan and sampled < 4:
                    row['example'] = {**plan, 'bombs': [(t, asdict(b)) for t, b in plan['bombs']]}
                rows.append(row)
                totals.update({k: v for k, v in stats.items() if k != 'rejected'})
                reasons.update(stats['rejected'])
                sampled += 1
            actions = [0, 0]
            actions[aid] = policy(env, obs, aid)
            actions[1 - aid] = opponent(env, obs, 1 - aid)
            obs, _, _, info = env.step(actions)
            if info['episode_done']:
                break
        print(f'[Preflight] seed {seed} A{aid}: {sampled} combat states, {env.step_count} replay steps', flush=True)
    times = [r['milliseconds'] for r in rows]
    return {'queries': len(rows), **totals, 'rejected': dict(reasons),
            'mean_ms': float(np.mean(times)) if times else 0,
            'p50_ms': float(np.percentile(times, 50)) if times else 0,
            'p95_ms': float(np.percentile(times, 95)) if times else 0,
            'max_ms': max(times, default=0), 'states': rows}


def final_evaluation(output, checkpoint, seconds=40 * 60):
    """Evaluate available best checkpoints even after a truncated training stage."""
    results = {}
    stop, timer = stage_timer(seconds)
    try:
        for name, model in [('original', checkpoint), ('local', output / 'local/best_phaseC.pt'),
                            ('two_bomb', output / 'two_bomb/best_phaseC.pt')]:
            if stop.is_set() or not model.is_file():
                break
            policy = make_ac_fn(str(model), 9, 'cpu')
            def progress(done, total):
                if done % 50 == 0:
                    print(f'[Evaluation] {name}: {done}/{total}', flush=True)
            result = evaluate_seeded(policy, 9, 2, 400, 25000, 500, stop, progress)
            if result is None:
                break
            result['checkpoint'] = str(model.resolve())
            result['checkpoint_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
            result['passive_win_rate'] = result['win_rate'] - result['active_win_rate']
            for corner in result['by_corner'].values():
                corner['passive_win_rate'] = corner['win_rate'] - corner['active_win_rate']
            results[name] = result
            save_json(output / f'evaluation_{name}_25000.json', result)
    finally:
        timer.cancel()
    return results


def run_pilot(output, checkpoint):
    torch.set_num_threads(1)
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    files = ['bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py',
             'tactical_teacher.py', 'evaluate.py', 'pilot_two_bomb.py', 'test_tactical_teacher.py',
             'test_combat.py', 'test_visualize.py']
    hashes = {}
    for filename in files:
        original = Path(__file__).parent / filename
        shutil.copy2(original, source / filename)
        hashes[filename] = hashlib.sha256(original.read_bytes()).hexdigest()
    config = {'phase': 'C', 'episodes': 200, 'board_size': 9, 'num_agents': 2,
              'max_steps': 500, 'lr': 1e-4, 'seed': 2026101303, 'resume_from': str(checkpoint),
              'validation_every': 100, 'validation_episodes': 100, 'validation_seed': 21000,
              'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'run_report': False,
              'log_every': 10}
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'training': config,
                'evaluation': {'episodes': 400, 'seed': 25000, 'last_seed': 25199, 'device': 'cpu'},
                'budget_minutes': {'preflight': 15, 'training': 65, 'evaluation': 40},
                'warm_start_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'source_sha256': hashes, 'torch': torch.__version__,
                'hardware': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}
    save_json(output / 'config.json', manifest)
    started = monotonic()
    status = {'state': 'preflight', 'promote': False, 'retain': str(checkpoint),
              'completed_episodes': {}, 'validations_complete': {}}
    save_json(output / 'status.json', status)
    stop, timer = stage_timer(15 * 60)
    try:
        profile = preflight(checkpoint, stop)
    finally:
        timer.cancel()
    if profile is None or not profile['queries']:
        status.update(state='inconclusive', reason='Preflight incomplete or no combat states')
        save_json(output / 'status.json', status)
        return status
    save_json(output / 'preflight.json', profile)
    print(f"[Preflight] {profile['queries']} queries: mean {profile['mean_ms']:.1f} ms, p95 {profile['p95_ms']:.1f} ms", flush=True)

    stop, timer = stage_timer(min(65 * 60, 120 * 60 - (monotonic() - started)))
    training_start = monotonic()
    try:
        for teacher in ('local', 'two_bomb'):
            if stop.is_set():
                break
            directory = output / teacher
            directory.mkdir()
            status['state'] = f'training_{teacher}'
            save_json(output / 'status.json', status)
            save_json(directory / 'config.json', {**config, 'attack_teacher': teacher})
            with (directory / 'metrics.jsonl').open('w', encoding='utf-8') as handle:
                def metrics(row):
                    handle.write(json.dumps(row) + '\n')
                    handle.flush()
                _, rewards, _ = train(**config, attack_teacher=teacher, checkpoint_dir=str(directory),
                                      stop_event=stop, on_episode=metrics)
            status['completed_episodes'][teacher] = len(rewards)
            validation_path = directory / 'validation_metrics.jsonl'
            rows = [json.loads(line) for line in validation_path.read_text().splitlines()] if validation_path.exists() else []
            status['validations_complete'][teacher] = complete_validation(rows)
            save_json(output / 'status.json', status)
            if not complete_validation(rows):
                stop.set()
    finally:
        timer.cancel()
    status['training_seconds'] = monotonic() - training_start
    training_complete = (status['completed_episodes'] == {'local': 200, 'two_bomb': 200}
                         and status['validations_complete'] == {'local': True, 'two_bomb': True})
    status['training_complete'] = training_complete
    status['state'] = 'evaluation'
    save_json(output / 'status.json', status)
    results = final_evaluation(output, checkpoint, min(40 * 60, 120 * 60 - (monotonic() - started)))
    if len(results) != 3:
        status.update(state='inconclusive', reason='All three 400-game evaluations did not finish within budget')
    else:
        candidate = results['two_bomb']
        comparisons = {name: paired_active_bootstrap(candidate, results[name]) for name in ('original', 'local')}
        promote = training_complete and all(c['difference_pp'] >= 5 and c['ci95_pp'][0] > 0 for c in comparisons.values())
        promote &= all(candidate['suicide_rate'] <= results[name]['suicide_rate'] for name in ('original', 'local'))
        status.update(state='complete' if training_complete else 'inconclusive', promote=bool(promote), comparisons=comparisons,
                      decision='eligible_for_replication' if promote else 'retain_original')
        if promote:
            status['candidate'] = str((output / 'two_bomb/best_phaseC.pt').resolve())
    if not training_complete:
        status['state'] = 'inconclusive'
        status['reason'] = 'Both runs and all 0/100/200 validations did not finish within 65 minutes'
    status['elapsed_seconds'] = monotonic() - started
    save_json(output / 'status.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs') / ('two-bomb-pilot-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/pressure-C-next/final_phaseC.pt'))
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.output, args.checkpoint), indent=2), flush=True)
