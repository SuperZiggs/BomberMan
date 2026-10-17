"""Paired Phase C pilot: local labels versus certified MOVE combat guidance."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import inspect
import json
from pathlib import Path
import pickle
import shutil
from time import perf_counter
from unittest.mock import patch
import time

import numpy as np
import torch

from audit_t2 import digest, dump
from audit_t3_approach import fingerprint, local_label, read_selected
from benchmark_t3 import make_env, scenarios, summarize_games
from bomberman_env import _MOVE_DELTA
from combat_move_teacher import certify_combat_move
from evaluate import evaluate_seeded, make_ac_fn, observe_contact
from pilot_two_bomb import paired_active_bootstrap, stage_timer
from run_t3_curriculum import model_digest, skill_probe
import train_actor_critic as trainer
from training_state import load_training_state, atomic_save, source_hashes, runtime_state

ROOT = Path(__file__).resolve().parent
WARM_START = ROOT / 'runs/pressure-C-next/final_phaseC.pt'
B_SHA256 = '93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60'
TRAIN_SEED = 2026101501
TEST_SEED = 33000
ARMS = {'control': 'local', 'combat_move': 'local_combat_move'}
SOURCE_FILES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py',
    'tactical_teacher.py', 'combat_move_teacher.py', 'evaluate.py', 'visualize.py',
    'pilot_two_bomb.py', 'audit_t2.py', 'audit_l2_candidates.py', 'audit_t3_approach.py',
    'benchmark_t3.py', 'run_t3_curriculum.py', 'run_combat_move_pilot.py',
    'test_combat_move_teacher.py', 'test_combat_move_pilot.py')
BaseNet, BaseAdam = trainer.ActorCriticNet, trainer.optim.Adam


def fresh_test_check(test_seed=TEST_SEED):
    for path in [*(ROOT / 'runs').rglob('evaluation*.json'), *(ROOT / 'runs').rglob('test_opened.json')]:
        result = json.loads(path.read_text(encoding='utf-8-sig'))
        seed, games, agents = result.get('seed'), result.get('num_episodes'), result.get('num_agents')
        if (isinstance(seed, int) and isinstance(games, int) and isinstance(agents, int)
                and agents > 0 and seed <= test_seed + 199 and seed + games // agents - 1 >= test_seed):
            raise ValueError(f'Test{test_seed} already opened in {path}; register a new protocol, do not rerun to select another model')


def preflight(stop, smoke):
    """Fixed, already-viewed C1 corpus; no CNN or fresh test-board evaluation."""
    path = ROOT / 'runs/t3-curriculum-full-20261014-154703/curriculum/skill_validation_games.jsonl.gz'
    registry = {c['id']: c for c in scenarios(('validation',))}
    rows, seen, reasons = [], set(), Counter()
    for game in read_selected(path):
        if game['group'] != 'approach':
            continue
        env, aid = make_env(registry[game['scenario_id']], game['seed']), game['aid']
        for record in game['tape']:
            if stop.is_set():
                return None
            mask = trainer.get_action_mask(env, aid)
            assert mask == record['mask']
            label, kind, _ = local_label(env, aid, mask)
            key = pickle.dumps((env.grid, env.agents, env.bombs, env.flames, env.powerups))
            if kind == 'navigation' and label in _MOVE_DELTA and key not in seen:
                seen.add(key)
                before = fingerprint(env)
                proof = certify_combat_move(env, aid, label, mask)
                assert fingerprint(env) == before
                reasons[proof['reason']] += 1
                rows.append({'scenario_id': game['scenario_id'], 'episode': game['episode'],
                             'seed': game['seed'], 'tick': record['tick'], 'aid': aid, 'label': label, **proof})
            env.step(record['actions'])
            if len(rows) >= (32 if smoke else 256):
                break
        if len(rows) >= (32 if smoke else 256):
            break
    times = [r['milliseconds'] for r in rows]
    result = {'queries': len(rows), 'accepted': sum(r['accepted'] for r in rows),
        'reason_counts': dict(reasons), 'p95_ms': float(np.percentile(times, 95)) if times else None,
        'mean_ms': float(np.mean(times)) if times else None, 'max_ms': max(times, default=0),
        'corpus_sha256': digest(path), 'query_limit': 32 if smoke else 256, 'records': rows}
    result['passed'] = bool(result['accepted'] and result['p95_ms'] <= 50)
    return result


def probe_summary(result):
    return {**{k: v for k, v in result.items() if k != 'records'},
            'groups': {group: summarize_games([g for g in result['records'] if g['group'] == group])
                       for group in ('escape', 'approach')}}


def train_arm(output, arm, device, smoke, stop, baseline, *, attack_teacher=None,
              normalize_advantage=None, advantage_estimator=None, gae_lambda=.95,
              train_seed=TRAIN_SEED, smoke_episodes=4, smoke_steps=8,
              wall_destroy_bonus=.15, reward_diagnostics=False, contact_diagnostics=False,
              save_training_state=False, resume=False):
    directory = output / arm
    directory.mkdir(exist_ok=resume)
    config = {'phase': 'C', 'episodes': smoke_episodes if smoke else 200, 'board_size': 9, 'num_agents': 2,
        'max_steps': smoke_steps if smoke else 500, 'lr': 1e-4, 'gamma': 0.99, 'value_coef': 0.5,
        'entropy_coef': 0.015, 'wall_destroy_bonus': wall_destroy_bonus, 'seed': train_seed,
        'resume_from': str(WARM_START), 'checkpoint_dir': str(directory), 'device': device,
        'attack_teacher': attack_teacher if attack_teacher is not None else ARMS[arm], 'run_report': False, 'log_every': 2 if smoke else 10,
        'validation_every': smoke_episodes // 2 if smoke else 100, 'validation_episodes': 2 if smoke else 100,
        'validation_seed': 21000, 'snapshot_every': 200}
    if normalize_advantage is not None:
        config.update(normalize_advantage=normalize_advantage, advantage_diagnostics=True)
    if advantage_estimator is not None:
        config.update(advantage_estimator=advantage_estimator, gae_lambda=gae_lambda,
                      advantage_diagnostics=True)
    if reward_diagnostics:
        config['reward_diagnostics'] = True
    if save_training_state:
        config['save_training_state'] = True
    state_path = directory / 'training_state.pt'
    restored = load_training_state(state_path) if resume and state_path.exists() else None
    if resume and (directory / 'config.json').exists():
        if json.loads((directory / 'config.json').read_text()) != config:
            raise ValueError('Arm config changed; resume refused')
    if restored is not None:
        expected = {key: config.get(key, inspect.signature(trainer.train).parameters[key].default)
                    for key in restored['provenance']['config']}
        if restored['provenance'] != {'config': expected, 'sources': source_hashes(), 'runtime': runtime_state()}:
            raise ValueError('Training config/source/runtime changed; log recovery refused')
    dump(directory / 'config.json', config)
    networks, optimizers, rows, probes = [], [], [], []
    games = []
    started = perf_counter()
    contact, current_episode = {}, None
    elapsed_offset = 0.
    if restored is not None:
        context = restored['context']
        if not isinstance(context, dict) or not {'rows', 'probes', 'games', 'contact', 'current_episode', 'elapsed', 'validation_rows'} <= context.keys():
            raise ValueError('Training state lacks the runner log context')
        rows, probes, games = context['rows'], context['probes'], context['games']
        contact, current_episode, elapsed_offset = context['contact'], context['current_episode'], context['elapsed']
        if len(rows) != restored['episode']:
            raise ValueError('Checkpoint and committed metrics differ')
    if resume:
        recovery = directory / 'recovery' / str(time.time_ns())
        recovery.mkdir(parents=True)
        for name in ('metrics.jsonl', 'skill_metrics.jsonl', 'skill_games.jsonl.gz',
                     'validation_metrics.jsonl', 'best_validation.json', 'best_phaseC.pt', 'final_phaseC.pt'):
            if (directory / name).exists():
                shutil.copy2(directory / name, recovery / name)
        if restored is None and (directory / 'metrics.jsonl').exists() and (directory / 'metrics.jsonl').stat().st_size:
            raise ValueError('Metrics exist without an atomic training state; start a new output')
        if restored is not None and restored['selection'][5] is not None:
            vals = context['validation_rows']
        else:
            vals = []
        (directory / 'validation_metrics.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in vals), encoding='utf-8')
        if restored is not None and restored['selection'][4] is not None:
            atomic_save(restored['selection'][4], directory / 'best_phaseC.pt')
            best_row = next(v for v in vals if v['episode'] == restored['selection'][2])
            dump(directory / 'best_validation.json', {key: value for key, value in best_row.items() if key != 'selected'})

    def on_step(env, episode, aid, actions, opponents):
        nonlocal current_episode
        if current_episode != episode:
            contact.clear()
            current_episode = episode
        observe_contact(env, aid, contact)

    def net(*args, **kwargs):
        value = BaseNet(*args, **kwargs)
        networks.append(value)
        return value

    def adam(*args, **kwargs):
        value = BaseAdam(*args, **kwargs)
        optimizers.append(value)
        return value

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with (directory / 'metrics.jsonl').open('w', encoding='utf-8') as metrics, \
         (directory / 'skill_metrics.jsonl').open('w', encoding='utf-8') as skills, \
         gzip.open(directory / 'skill_games.jsonl.gz', 'wt', encoding='utf-8') as tapes:
        for row in rows:
            metrics.write(json.dumps(row) + '\n')
        for probe in probes:
            skills.write(json.dumps(probe) + '\n')
        for game in games:
            tapes.write(json.dumps(game) + '\n')
        metrics.flush()
        skills.flush()
        tapes.flush()
        def on_episode(row):
            state = optimizers[0].state
            updates = int(next(iter(state.values()))['step'].item()) if state else 0
            row = {**row, 'optimizer_steps': updates, 'elapsed_seconds': elapsed_offset + perf_counter() - started}
            if contact_diagnostics:
                row['contact'] = dict(contact)
            rows.append(row)
            metrics.write(json.dumps(row) + '\n')
            metrics.flush()

        def on_validation(row):
            if 'completed' in row and row['completed'] % 25 == 0:
                print(f'[{arm} validation] ep{row["episode"]}: {row["completed"]}/{row["total"]}', flush=True)
            if 'result' not in row or stop.is_set():
                return
            if row['episode'] == 0:
                assert model_digest(networks[0]) == baseline['model_state_sha256']
                result = baseline
            else:
                result = skill_probe(networks[0], 1, stop, smoke=False)
            if result is None:
                return
            record = {'episode': row['episode'], **probe_summary(result)}
            probes.append(record)
            skills.write(json.dumps(record) + '\n')
            skills.flush()
            for game in result['records']:
                game_row = {'episode': row['episode'], **game}
                games.append(game_row)
                tapes.write(json.dumps(game_row) + '\n')
            tapes.flush()
            print(f'[{arm} skill] ep{row["episode"]}: approach '
                  f'{record["groups"]["approach"]["owned_kill_success"]["percent"]:.1f}%', flush=True)

        with patch.object(trainer, 'ActorCriticNet', side_effect=net), \
             patch.object(trainer.optim, 'Adam', side_effect=adam):
            hooks = {'on_step': on_step} if contact_diagnostics else {}
            if save_training_state:
                def context():
                    validation = directory / 'validation_metrics.jsonl'
                    return {'rows': rows, 'probes': probes, 'games': games, 'contact': contact,
                            'current_episode': current_episode, 'elapsed': elapsed_offset + perf_counter() - started,
                            'validation_rows': [json.loads(line) for line in validation.read_text().splitlines()] if validation.exists() else []}
                hooks.update(checkpoint_context=context)
            if restored is not None:
                hooks['resume_training_state'] = str(state_path)
            _, rewards, _ = trainer.train(**config, stop_event=stop, on_episode=on_episode,
                                         on_validation=on_validation, **hooks)
    assert len(networks) == len(optimizers) == 1 and len(rows) == len(rewards)
    validation = directory / 'validation_metrics.jsonl'
    vals = [json.loads(line) for line in validation.read_text().splitlines()] if validation.exists() else []
    expected = list(range(0, config['episodes'] + 1, config['validation_every']))
    guidance, rejected = Counter(), Counter()
    for row in rows:
        guidance.update({k: v for k, v in row['move_guidance'].items() if k != 'rejected'})
        rejected.update(row['move_guidance']['rejected'])
    result = {'completed_episodes': len(rows), 'expected_episodes': config['episodes'],
        'training_complete': len(rows) == config['episodes'],
        'validation_complete': [v['episode'] for v in vals] == expected and all(
            v['seed'] == 21000 and v['num_episodes'] == config['validation_episodes'] for v in vals),
        'skill_probes_complete': [p['episode'] for p in probes] == expected,
        'optimizer_instances': len(optimizers), 'optimizer_steps': rows[-1]['optimizer_steps'] if rows else 0,
        'learner_steps': sum(r['learner_steps'] for r in rows),
        'guidance': {**guidance, 'rejected': dict(rejected)},
        'best_episode': json.loads((directory / 'best_validation.json').read_text())['episode'] if
            (directory / 'best_validation.json').exists() else None,
        'elapsed_seconds': elapsed_offset + perf_counter() - started,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        'checkpoints': {name: {'path': str(directory / f'{name}_phaseC.pt'),
                              'sha256': digest(directory / f'{name}_phaseC.pt')} for name in ('best', 'final')}}
    if save_training_state:
        result['training_state'] = {'path': str(state_path), 'sha256': digest(state_path),
                                    'episode': load_training_state(state_path)['episode']}
    dump(directory / 'summary.json', result)
    return result


def pair_ready(arms, arm_names=ARMS):
    return set(arms) == set(arm_names) and all(a['training_complete'] and a['validation_complete']
        and a['skill_probes_complete'] and a['completed_episodes'] == a['expected_episodes']
        and a['optimizer_instances'] == 1 and a['optimizer_steps'] == a['completed_episodes']
        for a in arms.values())


def evaluate_models(output, stop, smoke, *, arm_names=ARMS, test_seed=TEST_SEED, smoke_seed=39830,
                    contact_diagnostics=False, resume=False):
    results = {}
    seed, count, steps = (smoke_seed, 2, 8) if smoke else (test_seed, 400, 500)
    paths = {'original': WARM_START, **{arm: output / arm / 'best_phaseC.pt' for arm in arm_names}}
    for name, path in paths.items():
        previous = output / f'evaluation_{name}_{seed}.json'
        if resume and previous.exists():
            cached = json.loads(previous.read_text())
            if (cached['seed'], cached['num_episodes'], cached['max_steps'], cached['checkpoint_sha256']) != (seed, count, steps, digest(path)):
                raise ValueError('Stored evaluation differs from the locked model/protocol')
            if len(cached['games']) != count or {(g['seed'], g['aid']) for g in cached['games']} != {
                    (seed + i, aid) for i in range(count // 2) for aid in (0, 1)}:
                raise ValueError('Stored evaluation is incomplete')
            results[name] = cached
            continue
        def progress(done, total):
            if done % 25 == 0:
                print(f'[Evaluation] {name}: {done}/{total}', flush=True)
        diagnostics = {'contact_diagnostics': True} if contact_diagnostics else {}
        result = evaluate_seeded(make_ac_fn(str(path), 9, 'cpu'), 9, 2, count, seed, steps,
                                 stop_event=stop, on_progress=progress, **diagnostics)
        if result is None:
            break
        assert len(result['games']) == count
        assert {(g['seed'], g['aid']) for g in result['games']} == {
            (seed + i, aid) for i in range(count // 2) for aid in (0, 1)}
        result.update(checkpoint=str(path), checkpoint_sha256=digest(path),
                      passive_win_rate=result['win_rate'] - result['active_win_rate'])
        for corner in result['by_corner'].values():
            corner['passive_win_rate'] = corner['win_rate'] - corner['active_win_rate']
        results[name] = result
        dump(output / f'evaluation_{name}_{seed}.json', result)
    return results


def comparison_gate(results, candidate='combat_move'):
    comparisons = {name: paired_active_bootstrap(results[candidate], results[name])
                   for name in ('original', 'control')}
    eligible = all(c['difference_pp'] >= 5 and c['ci95_pp'][0] > 0 for c in comparisons.values())
    eligible &= all(results[candidate]['suicide_rate'] <= results[name]['suicide_rate']
                    for name in ('original', 'control'))
    return comparisons, bool(eligible)


def run(output, device='cpu', smoke=False, budget_minutes=120):
    if budget_minutes <= 0 or budget_minutes > 120:
        raise ValueError('Budget must be in (0, 120] minutes')
    if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Requested training device is unavailable')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if digest(WARM_START) != B_SHA256:
        raise ValueError('Warm-start B changed')
    if not smoke:
        fresh_test_check()
    torch.set_num_threads(1)
    sources = {name: digest(ROOT / name) for name in SOURCE_FILES}
    output.mkdir(parents=True)
    (output / 'source').mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, output / 'source' / name)
    config = {'registered_utc': datetime.now(timezone.utc).isoformat(), 'full_protocol': not smoke,
        'warm_start': str(WARM_START), 'warm_start_sha256': B_SHA256, 'source_sha256': sources,
        'arms': ARMS, 'train_seed': TRAIN_SEED, 'episodes_per_arm': 4 if smoke else 200,
        'phase': 'C', 'board_size': 9, 'num_agents': 2, 'lr': 1e-4, 'train_device': device,
        'max_steps': 8 if smoke else 500, 'validation_seed': 21000,
        'validation_episodes': 2 if smoke else 100, 'validation_every': 2 if smoke else 100,
        'evaluation': {'seed': 39830 if smoke else TEST_SEED, 'games_per_model': 2 if smoke else 400,
                       'last_board_seed': 39830 if smoke else TEST_SEED + 199, 'device': 'cpu', 'threads': 1},
        'preflight': {'p95_ms_at_most': 50, 'must_accept_a_move': True, 'corpus': 'already-viewed C1 validation'},
        'budget_minutes': {'overall': budget_minutes, 'preflight': 15, 'training': 65, 'evaluation': 40},
        'only_treatment_change': 'Certified local MOVE targets join combat mean (0.5) instead of navigation mean (0.2)',
        'gate': {'gain_vs_both_pp': 5, 'ci95_lower_above': 0, 'suicide_not_above_either': True},
        'torch': torch.__version__, 'hardware': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}
    dump(output / 'config.json', config)
    (ROOT / 'runs' / ('combat-move-smoke-last-path.txt' if smoke else 'combat-move-last-path.txt')).write_text(str(output), encoding='utf-8')
    started = perf_counter()
    arms, results, comparisons, selected, reason, eligible = {}, {}, {}, {}, None, False
    timer = None

    def phase(seconds):
        return stage_timer(min(seconds, max(0, budget_minutes * 60 - (perf_counter() - started))))

    try:
        stop, timer = phase(15 * 60)
        profile = preflight(stop, smoke)
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
                stop, timer = phase(65 * 60)
                for arm in ARMS:
                    if stop.is_set():
                        break
                    arms[arm] = train_arm(output, arm, device, smoke, stop, baseline)
                    if not (arms[arm]['training_complete'] and arms[arm]['validation_complete'] and arms[arm]['skill_probes_complete']):
                        break
                if not pair_ready(arms):
                    reason = 'training_validation_or_skill_incomplete'
                else:
                    assert sources == {name: digest(ROOT / name) for name in SOURCE_FILES}
                    assert digest(WARM_START) == B_SHA256
                    selected = {'original': {'path': str(WARM_START), 'sha256': B_SHA256},
                                **{arm: arms[arm]['checkpoints']['best'] for arm in ARMS}}
                    assert all(digest(Path(record['path'])) == record['sha256'] for record in selected.values())
                    dump(output / 'selected_models.json', selected)
                    timer.cancel()
                    stop, timer = phase(40 * 60)
                    results = evaluate_models(output, stop, smoke)
                    if len(results) != 3:
                        reason = 'evaluation_incomplete'
                    elif not smoke:
                        comparisons, eligible = comparison_gate(results)
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        if timer is not None:
            timer.cancel()
        try:
            checkpoints_unchanged = all(digest(Path(record['path'])) == record['sha256'] for record in selected.values())
            unchanged = digest(WARM_START) == B_SHA256 and sources == {
                name: digest(ROOT / name) if (ROOT / name).exists() else None for name in SOURCE_FILES}
        except FileNotFoundError:
            checkpoints_unchanged, unchanged = False, False
        unchanged &= checkpoints_unchanged
        if not unchanged:
            reason, eligible = 'source_or_warm_start_changed', False
        complete = reason is None and pair_ready(arms) and len(results) == 3 and unchanged
        summary = {'status': 'complete' if complete else 'incomplete', 'full_protocol': not smoke,
            'reason': reason, 'arms': arms, 'evaluation_complete': len(results) == 3,
            'hashes_unchanged': unchanged, 'selected_checkpoints_unchanged': checkpoints_unchanged, 'comparisons': comparisons,
            'eligible_for_replication': complete and not smoke and eligible,
            'decision': 'replicate_before_replacing' if complete and not smoke and eligible else 'retain_original',
            'elapsed_seconds': perf_counter() - started,
            'models': {name: {k: v for k, v in result.items() if k != 'games'} for name, result in results.items()}}
        dump(output / 'summary.json', summary)
        shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(json.dumps(summary, indent=2), flush=True)
        print(f'Send {output}-send.zip and the console log. Keep B until a separate decision.', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--budget-minutes', type=float, default=120)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / ('combat-move-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.device, args.smoke, args.budget_minutes)
