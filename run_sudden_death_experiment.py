"""Paired frozen-checkpoint test of Loc_update rings and agent awareness."""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import threading
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

import evaluate as evaluation
import rule_based_bot as bot_module
import train_actor_critic as trainer
from pilot_two_bomb import paired_active_bootstrap
from sudden_death_experiment import SuddenDeathEnv, sudden_death_mask

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'runs/pressure-C-next/final_phaseC.pt'
CHECKPOINT_SHA = '93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60'
REMOTE_COMMIT = 'afbf6f9483a05ddddbd042cb716b545018bf7bde'
CASES = {'off': (False, False, False), 'rings_only': (True, False, False),
         'learner_aware': (True, True, False), 'both_aware': (True, True, True)}
SOURCES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'evaluate.py',
           'training_state.py', 'tactical_teacher.py', 'combat_move_teacher.py',
           'pilot_two_bomb.py', 'sudden_death_experiment.py', 'run_sudden_death_experiment.py',
           'test_sudden_death_experiment.py', 'test_sudden_death_runner.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def summarize_extra(games):
    n = len(games)
    return {'learner_deaths': dict(Counter(g['learner_death'] for g in games)),
            'opponent_deaths': dict(Counter(g['opponent_death'] for g in games)),
            'endings': dict(Counter(g['ending'] for g in games)),
            'wins_opponent_sd': sum(g['win'] and g['opponent_death'] == 'sudden_death' for g in games),
            'both_sd_draws': sum(g['both_sd_draw'] for g in games),
            'reached_sd': sum(g['reached_sd'] for g in games),
            'mask': {side: dict(sum((Counter(g['mask'][side]) for g in games), Counter()))
                     for side in ('learner', 'opponent')},
            'games': n}


def evaluate_case(case, *, games, seed, max_steps, timings, trace_path=None, progress=None, stop=None,
                  checkpoint=CHECKPOINT, ring_channel=False, learner_margin=0):
    enabled, learner_aware, opponent_aware = CASES[case]
    extras, traces = [], []
    diagnostics = {'learner': {}, 'opponent': {}}
    original_match, original_mask = evaluation.run_match, trainer.get_action_mask
    original_bot_mask = bot_module._safe_action_mask

    class ObservedEnv(SuddenDeathEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, sudden_death=enabled, ring_channel=ring_channel, **timings)

        def reset(self):
            self.actions = []
            return super().reset()

        def step(self, actions):
            self.actions.append(list(actions))
            return super().step(actions)

    def cnn_mask(env, aid):
        base = original_mask(env, aid)
        return (sudden_death_mask(env, aid, base, diagnostics['learner'], margin=learner_margin)
                if learner_aware else base)

    def rule_mask(env, aid, *args, **kwargs):
        base = original_bot_mask(env, aid, *args, **kwargs)
        return sudden_death_mask(env, aid, base, diagnostics['opponent']) if opponent_aware else base

    def observed_match(env, fns, *args, **kwargs):
        ep = len(extras)
        aid, enemy = ep % 2, 1 - ep % 2
        deaths = ['alive', 'alive']
        diagnostics['learner'].clear()
        diagnostics['opponent'].clear()
        old_observer = kwargs.pop('on_step', None)
        final = None
        def observe(current, info):
            nonlocal final
            if old_observer:
                old_observer(current, info)
            if info is None:
                return
            final = info
            for victim in range(2):
                if not info['alive'][victim] and deaths[victim] == 'alive':
                    if victim in info.get('sudden_death_kills', ()):
                        deaths[victim] = 'sudden_death'
                    elif victim in info['self_blast_deaths']:
                        deaths[victim] = 'self_bomb'
                    elif any(v == victim and k != victim for k, v in info['kills']):
                        deaths[victim] = 'opponent_bomb'
                    else:
                        deaths[victim] = 'other'
        stats = original_match(env, fns, *args, on_step=observe, **kwargs)
        alive = sum(final['alive'])
        detail = {'learner_death': deaths[aid], 'opponent_death': deaths[enemy],
                  'ending': 'sole_survivor' if alive == 1 else ('all_dead' if alive == 0 else 'timeout'),
                  'both_sd_draw': alive == 0 and set(final.get('sudden_death_kills', ())) == {0, 1},
                  'reached_sd': enabled and env.step_count >= env.sudden_death_start,
                  'final_ring': env._sudden_death_ring, 'mask': deepcopy(diagnostics)}
        extras.append(detail)
        traces.append({'seed': seed + ep // 2, 'aid': aid, 'actions': env.actions,
                       'detail': detail, 'final_alive': final['alive'],
                       'kills': [a.kills for a in env.agents],
                       'bombs': stats['avg_bombs'], 'items': stats['avg_items'],
                       'suicides': [rate / 100 for rate in stats['suicide_rates']]})
        return stats

    with torch.random.fork_rng(devices=[]), \
         patch.object(evaluation, 'BombermanFFAEnv', ObservedEnv), \
         patch.object(evaluation, 'run_match', side_effect=observed_match), \
         patch.object(trainer, 'get_action_mask', side_effect=cnn_mask), \
         patch.object(bot_module, '_safe_action_mask', side_effect=rule_mask):
        policy = evaluation.make_ac_fn(str(checkpoint), 9, 'cpu')
        result = evaluation.evaluate_seeded(policy, 9, 2, games, seed, max_steps,
                                             stop, progress, contact_diagnostics=True)
    if result is None:
        return None
    for game, detail in zip(result['games'], extras):
        game.update(detail)
    result['passive_win_rate'] = result['win_rate'] - result['active_win_rate']
    result['sudden_death'] = summarize_extra(result['games'])
    for aid in (0, 1):
        rows = [g for g in result['games'] if g['aid'] == aid]
        corner = result['by_corner'][str(aid)]
        corner['passive_win_rate'] = corner['win_rate'] - corner['active_win_rate']
        corner['sudden_death'] = summarize_extra(rows)
    if trace_path:
        with gzip.open(trace_path, 'wt', encoding='utf-8') as f:
            for trace in traces:
                f.write(json.dumps(trace) + '\n')
    return result


@contextmanager
def sudden_death_runtime(timings, ring_channel=False):
    """Run trainer/evaluator/skill probes under SD rules with SD-aware learner and RuleBot."""
    import benchmark_t3
    base_mask, base_bot_mask = trainer.get_action_mask, bot_module._safe_action_mask

    class TrainEnv(SuddenDeathEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, sudden_death=True, ring_channel=ring_channel, **timings)

    class ProbeEnv(SuddenDeathEnv):
        # Skill-probe fixtures keep their rules; only the observation width follows the arm.
        def __init__(self, **kwargs):
            super().__init__(**kwargs, ring_channel=ring_channel)

    def aware(base):
        def mask(env, aid, *args, **kwargs):
            plain = base(env, aid, *args, **kwargs)
            return sudden_death_mask(env, aid, plain) if hasattr(env, '_closure_ticks') else plain
        return mask

    with patch.object(trainer, 'BombermanFFAEnv', TrainEnv),          patch.object(evaluation, 'BombermanFFAEnv', TrainEnv),          patch.object(benchmark_t3, 'BombermanFFAEnv', ProbeEnv),          patch.object(trainer, 'get_action_mask', aware(base_mask)),          patch.object(bot_module, '_safe_action_mask', aware(base_bot_mask)):
        yield


def comparisons(results):
    pairs = {'rings_vs_off': ('rings_only', 'off'),
             'learner_mask_vs_rings': ('learner_aware', 'rings_only'),
             'opponent_mask_vs_learner': ('both_aware', 'learner_aware'),
             'both_aware_vs_off': ('both_aware', 'off')}
    output = {}
    for label, (left, right) in pairs.items():
        output[label] = {}
        for metric in ('win', 'active_win', 'draw', 'suicide'):
            wrapped = [{**r, 'games': [{**g, 'active_win': g[metric]} for g in r['games']]}
                       for r in (results[left], results[right])]
            output[label][metric] = paired_active_bootstrap(*wrapped)
    return output


def run(output, smoke=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Baseline checkpoint changed')
    torch.set_num_threads(1)
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    config = {'registered_utc': datetime.now(timezone.utc).isoformat(), 'experiment': 'SD1-rules',
              'full_protocol': not smoke, 'cases': CASES, 'checkpoint': str(CHECKPOINT),
              'checkpoint_sha256': CHECKPOINT_SHA, 'source_sha256': hashes,
              'reference_commit': REMOTE_COMMIT, 'board_size': 9, 'num_agents': 2,
              'games_per_case': 4 if smoke else 400, 'seed': 39890 if smoke else 38000,
              'max_steps': 40 if smoke else 500, 'device': 'cpu', 'threads': 1,
              'timings': {'sudden_death_start': 20 if smoke else 200,
                          'sudden_death_interval': 5 if smoke else 20,
                          'sudden_death_warning_steps': 8 if smoke else 40},
              'observation': 'unchanged 12 channels; no additional warning channel',
              'optimizer_steps': 0, 'budget_minutes': 90,
              'torch_version': torch.__version__, 'numpy_version': np.__version__}
    output.mkdir(parents=True)
    source = output / 'source'
    source.mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / name, source / name)
    reference = ROOT / '_comparison/BomberMan-Loc_update/bomberman_env.py'
    config['reference_env_sha256'] = digest(reference)
    shutil.copy2(reference, source / 'Loc_update_bomberman_env.py')
    dump(output / 'config.json', config)
    results, error = {}, None
    stop = threading.Event()
    timer = threading.Timer(90 * 60, stop.set)
    timer.daemon = True
    timer.start()
    started = perf_counter()
    try:
        for case in CASES:
            begin = perf_counter()
            def progress(done, total):
                if done % 25 == 0 or done == total:
                    print(f'[{case}] {done}/{total}', flush=True)
            result = evaluate_case(case, games=config['games_per_case'], seed=config['seed'],
                max_steps=config['max_steps'], timings=config['timings'],
                trace_path=output / f'replays_{case}.jsonl.gz', progress=progress, stop=stop)
            if result is None:
                error = 'budget_exhausted'
                break
            result['seconds'] = perf_counter() - begin
            results[case] = result
            dump(output / f'evaluation_{case}.json', result)
            print(f"[{case}] WR={result['win_rate']:.2f} ActiveWR={result['active_win_rate']:.2f} "
                  f"Draw={result['draw_rate']:.2f} Self={result['suicide_rate']:.2f}", flush=True)
    except KeyboardInterrupt:
        error = 'interrupted'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        timer.cancel()
        unchanged = hashes == {name: digest(ROOT / name) for name in SOURCES} and digest(CHECKPOINT) == CHECKPOINT_SHA
        complete = error is None and len(results) == len(CASES) and unchanged
        summary = {'status': 'complete' if complete else 'incomplete', 'full_protocol': not smoke,
                   'hashes_unchanged': unchanged, 'optimizer_steps': 0, 'error': error,
                   'elapsed_seconds': perf_counter() - started,
                   'results': {case: {k: v for k, v in r.items() if k != 'games'} for case, r in results.items()},
                   'comparisons': comparisons(results) if complete and not smoke else {},
                   'decision': 'retain_checkpoint_B; evaluate_rule_variant_only'}
        dump(output / 'summary.json', summary)
        shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if run(args.output_dir, args.smoke)['status'] != 'complete':
        raise SystemExit(1)
