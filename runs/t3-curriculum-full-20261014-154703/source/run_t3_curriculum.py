"""Paired T3.2 curriculum pilot, using the existing trainer without changing its core."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import signal
import sys
import threading
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

import train_actor_critic as trainer
from audit_t2 import digest, dump, opponent_action
from benchmark_t3 import (make_env, observations, scenarios, reference_plan, certify,
                          genuine_combo, play_case, summarize_games)
from evaluate import make_ac_fn, evaluate_seeded
from pilot_two_bomb import paired_active_bootstrap
from rule_based_bot import RuleBasedAgent

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'runs/t3-full-20261014-implementation'
WARM_START = ROOT / 'runs/pressure-C-next/final_phaseC.pt'
TRAIN_SEED = 2026101303
SKILL_SEEDS = (2026101401, 2026101402)
STAGES = {1: 'C1_static', 2: 'C2_evader', 3: 'C3_rule_mix', 4: 'C4_transfer'}
CORE_FILES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py',
              'tactical_teacher.py', 'evaluate.py')
SOURCE_FILES = (*CORE_FILES, 'visualize.py', 'pilot_two_bomb.py', 'audit_t2.py',
                'audit_l2_candidates.py', 'benchmark_t3.py', 'run_t3_curriculum.py', 'test_t3_curriculum.py')
BaseEnv, BaseNet = trainer.BombermanFFAEnv, trainer.ActorCriticNet
BaseAdam = trainer.optim.Adam


def skill_cases(stage):
    groups = ('escape', 'approach') if stage == 1 else ('combo',)
    return [c for c in scenarios(('validation',)) if c['group'] in groups]


def model_digest(net):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def skill_probe(net, stage, stop, smoke=False):
    """Probe current weights on CPU, preserving CPU/CUDA/NumPy training RNGs."""
    rng = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    games = []
    try:
        with torch.random.fork_rng(devices=devices):
            policy = trainer.SnapshotOpponent(deepcopy(net).to('cpu'), 'cpu')
            weight_hash = model_digest(policy.net)
            cases = skill_cases(stage)
            if smoke:
                cases = cases[:4]
            kind = {1: 'static', 2: 'evader', 3: 'rule'}[stage]
            for case in cases:
                for seed in SKILL_SEEDS[:1] if smoke else SKILL_SEEDS:
                    if stop.is_set():
                        return None
                    games.append(play_case(case, lambda e, obs, aid: policy.select_action(obs[aid], e, aid),
                                           'current', kind, seed, 12 if smoke else 80))
    finally:
        np.random.set_state(rng)
    return {'stage': STAGES[stage], 'opponent': kind, 'model_state_sha256': weight_hash,
            **summarize_games(games), 'records': games}


class ScriptedOpponent:
    def __init__(self, kind, aid):
        self.kind, self.bot = kind, RuleBasedAgent(aid)

    def select_action(self, obs, env, aid):
        action = opponent_action(env, aid, self.kind, self.bot)
        assert self.kind != 'evader' or action != 5
        return action


class Curriculum:
    def __init__(self, arm, stop, guards, smoke=False):
        self.arm, self.stop, self.guards, self.smoke = arm, stop, guards, smoke
        self.stage, self.stage_start, self.episode, self.streak = 1, 1, 0, 0
        self.events, self.failure, self.setup = [], None, {}
        self.rng = np.random.default_rng(TRAIN_SEED)
        self.train_cases = [c for c in scenarios(('train',)) if c['group'] in ('escape', 'approach', 'combo')]

    def is_set(self):
        return self.stop.is_set() or self.failure is not None

    def reset(self, env):
        self.episode += 1
        aid = (self.episode - 1) % 2
        stage = self.stage if self.arm == 'curriculum' else 4
        kind = {1: 'static', 2: 'evader', 3: 'rule', 4: 'rule'}[stage]
        group = ('escape', 'approach') if stage == 1 else ('combo',)
        random_board = (stage == 4 or (stage in (1, 3) and self.rng.random() < 0.5))
        self.setup = {'stage': STAGES[stage] if self.arm == 'curriculum' else 'control_rule',
                      'opponent': kind, 'learning_aid': aid, 'scenario_id': None,
                      'layout': 'random', 'stage_age': self.episode - self.stage_start + 1}
        if random_board:
            env.wooden_wall_density = 0.25 if stage == 1 else 0.7
            env.powerup_prob = 0.3
            obs = BaseEnv.reset(env)
        else:
            pool = [c for c in self.train_cases if c['aid'] == aid and c['group'] in group]
            case = pool[int(self.rng.integers(len(pool)))]
            fresh = make_env(case, TRAIN_SEED + self.episode)
            fresh.max_steps = env.max_steps
            env.__dict__.clear()
            env.__dict__.update(fresh.__dict__)
            obs = observations(env)
            self.setup.update(scenario_id=case['id'], layout='scenario')
        self.setup.update(wood_density=env.wooden_wall_density, powerup_prob=env.powerup_prob,
                          ammo=env.agents[aid].ammo, blast_range=env.agents[aid].blast_range)
        return obs

    def on_step(self, env, episode, aid, actions, opponents):
        if env.step_count == 0:
            assert episode == self.episode and aid == self.setup['learning_aid']
            kind = self.setup['opponent']
            if kind != 'rule':
                opponents[1 - aid] = ScriptedOpponent(kind, 1 - aid)

    def probe_due(self, episode):
        if self.arm != 'curriculum' or self.stage == 4:
            return False
        age = episode - self.stage_start + 1
        return age % 2 == 0 if self.smoke else age >= 100 and age % 50 == 0

    def record_probe(self, episode, result):
        age = episode - self.stage_start + 1
        if result is None:
            return False
        if self.smoke:
            passed = True  # Wiring test only; never eligible for model selection.
        else:
            passed = (result['games'] == len(skill_cases(self.stage)) * 2
                      and result['owned_kill_success']['percent'] >= 60
                      and result['suicide']['percent'] <= self.guards[self.stage])
        self.streak = self.streak + 1 if passed else 0
        if self.smoke or self.streak >= 2:
            self.events.append({'episode_completed': episode, 'next_episode': episode + 1,
                                'from': STAGES[self.stage], 'to': STAGES[self.stage + 1], 'age': age})
            self.stage += 1
            self.stage_start, self.streak = episode + 1, 0
        elif age >= 200:
            self.failure = {'stage': STAGES[self.stage], 'episode': episode,
                            'reason': 'skill_gate_not_met_within_200_episodes'}
        return passed


def train_arm(arm, directory, guards, stop, device, smoke):
    directory.mkdir()
    controller = Curriculum(arm, stop, guards, smoke)
    networks, optimizers, rows = [], [], []

    class EpisodeEnv(BaseEnv):
        def reset(self):
            return controller.reset(self)

    def create_net(*args, **kwargs):
        net = BaseNet(*args, **kwargs)
        networks.append(net)
        return net

    def create_optimizer(*args, **kwargs):
        optimizer = BaseAdam(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    config = {'phase': 'C', 'episodes': 6 if smoke else 600, 'board_size': 9, 'num_agents': 2,
              'max_steps': 8 if smoke else 500, 'lr': 1e-4, 'gamma': 0.99, 'value_coef': 0.5,
              'entropy_coef': 0.015, 'wall_destroy_bonus': 0.15, 'seed': TRAIN_SEED,
              'resume_from': str(WARM_START), 'checkpoint_dir': str(directory), 'device': device,
              'attack_teacher': 'local', 'run_report': False, 'log_every': 2 if smoke else 50,
              'validation_every': 2 if smoke else 100, 'validation_episodes': 2 if smoke else 100,
              'validation_seed': 21000, 'snapshot_every': 200}
    dump(directory / 'config.json', config)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with (directory / 'metrics.jsonl').open('w', encoding='utf-8') as metrics, \
         (directory / 'skill_validation_metrics.jsonl').open('w', encoding='utf-8') as skills, \
         gzip.open(directory / 'skill_validation_games.jsonl.gz', 'wt', encoding='utf-8') as tapes:
        def on_episode(row):
            state = optimizers[0].state
            updates = int(next(iter(state.values()))['step'].item()) if state else 0
            row = {**row, **controller.setup, 'optimizer_steps': updates,
                   'elapsed_seconds': perf_counter() - started}
            rows.append(row)
            metrics.write(json.dumps(row) + '\n')
            metrics.flush()
            if arm == 'control' and not smoke and row['episode'] >= 200 and row['episode'] % 50 == 0:
                torch.save(networks[0].state_dict(), directory / f'budget_ep{row["episode"]}.pt')
            if controller.probe_due(row['episode']) and not stop.is_set():
                stage = controller.stage
                result = skill_probe(networks[0], stage, stop, smoke)
                passed = controller.record_probe(row['episode'], result)
                record = {'episode': row['episode'], 'stage_age': row['stage_age'],
                          'stage': STAGES[stage], 'complete': result is not None,
                          'passed': passed, 'smoke_forced_transition': smoke,
                          'guard_suicide_percent': guards[stage],
                          'result': {k: v for k, v in result.items() if k != 'records'} if result else None}
                skills.write(json.dumps(record) + '\n')
                skills.flush()
                if result:
                    for game in result['records']:
                        tapes.write(json.dumps({'episode': row['episode'], 'stage': STAGES[stage], **game}) + '\n')
                    tapes.flush()
                dump(directory / 'stage_events.json', {'events': controller.events, 'failure': controller.failure})
                print(f'[{arm} skill] ep{row["episode"]} {STAGES[stage]} pass={passed} next={STAGES[controller.stage]}', flush=True)
            dump(directory / 'status.json', {'status': 'running', 'episode': row['episode'],
                 'stage': row['stage'], 'next_stage': STAGES[controller.stage],
                 'stage_age': row['stage_age'], 'stage_failure': controller.failure})

        def progress(row):
            if 'completed' in row and row['completed'] % 25 == 0:
                print(f'[{arm} Phase C validation] ep{row["episode"]} {row["completed"]}/{row["total"]}', flush=True)

        # CLI-only adapter scope. Never run concurrent train() calls in this process.
        with patch.object(trainer, 'BombermanFFAEnv', EpisodeEnv), \
             patch.object(trainer, 'ActorCriticNet', side_effect=create_net), \
             patch.object(trainer.optim, 'Adam', side_effect=create_optimizer):
            _, rewards, _ = trainer.train(**config, stop_event=controller, on_step=controller.on_step,
                                          on_episode=on_episode, on_validation=progress)
    assert len(networks) == len(optimizers) == 1 and len(rewards) == len(rows)
    expected_val = list(range(0, config['episodes'] + 1, config['validation_every']))
    validations = [json.loads(line) for line in (directory / 'validation_metrics.jsonl').read_text().splitlines()] \
                  if (directory / 'validation_metrics.jsonl').exists() else []
    validation_complete = ([r['episode'] for r in validations] == expected_val and
        all(r['num_episodes'] == config['validation_episodes'] and r['seed'] == 21000 for r in validations))
    result = {'arm': arm, 'completed_episodes': len(rows), 'expected_episodes': config['episodes'],
              'training_complete': len(rows) == config['episodes'], 'validation_complete': validation_complete,
              'curriculum_stages_complete': arm == 'control' or controller.stage == 4,
              'stage_failure': controller.failure, 'stage_events': controller.events,
              'stage_episode_counts': dict(Counter(r['stage'] for r in rows)),
              'learner_steps': sum(r['learner_steps'] for r in rows), 'optimizer_instances': len(optimizers),
              'optimizer_steps': rows[-1]['optimizer_steps'] if rows else 0,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
              'elapsed_seconds': perf_counter() - started}
    dump(directory / 'stage_events.json', {'events': controller.events, 'failure': controller.failure})
    dump(directory / 'summary.json', result)
    dump(directory / 'status.json', {'status': 'complete' if result['training_complete'] and validation_complete
         and result['curriculum_stages_complete'] else 'incomplete', **result})
    return result


def preflight(probe, stop):
    cfg = json.loads((REFERENCE / 'config.json').read_text())
    registry = json.loads((REFERENCE / 'scenario_registry.json').read_text())
    assert registry['test_policy_results_locked']
    assert scenarios(('train', 'validation', 'test')) == registry['cases']
    assert digest(WARM_START) == cfg['checkpoint_sha256']['original']
    assert all(digest(ROOT / n) == cfg['source_sha256'][n] for n in CORE_FILES)
    cases = [c for c in registry['cases'] if c['split'] == 'train' and c['group'] in ('escape', 'approach', 'combo')]
    for case in cases:
        if stop.is_set():
            return None
        env = make_env(case)
        plan, quality = reference_plan(env, case)
        assert certify(env, case['aid'], plan)['full_mask_pass']
        assert case['group'] != 'combo' or genuine_combo(quality)
    baseline = {}
    for stage in (1, 2, 3):
        result = probe(stage)
        if result is None:
            return None
        baseline[stage] = {k: v for k, v in result.items() if k != 'records'}
        print(f'[Baseline skill] {STAGES[stage]}: {result["owned_kill_success"]["percent"]:.1f}% owned kill', flush=True)
    return baseline


def full_pair_ready(arms):
    return (set(arms) == {'control', 'curriculum'} and all(a['training_complete'] and a['validation_complete']
            and a['completed_episodes'] == a['expected_episodes'] == 600
            and a['curriculum_stages_complete'] for a in arms.values()))


def evaluate_models(output, stop, smoke):
    results = {}
    seed, count = (29800, 2) if smoke else (31000, 400)
    paths = {'original': WARM_START, **{a: output / a / 'best_phaseC.pt' for a in ('control', 'curriculum')}}
    for name, path in paths.items():
        policy = make_ac_fn(str(path), 9, 'cpu')
        def progress(done, total):
            if done % 25 == 0:
                print(f'[Final random-map evaluation] {name} {done}/{total}', flush=True)
        result = evaluate_seeded(policy, 9, 2, count, seed, 8 if smoke else 500, stop_event=stop, on_progress=progress)
        if result is None:
            return results, False
        result.update(checkpoint=str(path), checkpoint_sha256=digest(path),
                      passive_win_rate=result['win_rate'] - result['active_win_rate'])
        for corner in result['by_corner'].values():
            corner['passive_win_rate'] = corner['win_rate'] - corner['active_win_rate']
        results[name] = result
        dump(output / f'evaluation_{name}_{seed}.json', result)
    return results, True


def heldout_skills(output, stop):
    summaries = {}
    cases = scenarios(('test',))
    paths = {'original': WARM_START, **{a: output / a / 'best_phaseC.pt' for a in ('control', 'curriculum')}}
    with gzip.open(output / 'heldout_skill_games.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for model, path in paths.items():
            policy, games = make_ac_fn(str(path), 9, 'cpu'), []
            for case in cases:
                for kind in ('static', 'evader', 'rule'):
                    for seed in SKILL_SEEDS:
                        if stop.is_set():
                            return summaries, False
                        game = play_case(case, policy, model, kind, seed, 80)
                        games.append(game)
                        stream.write(json.dumps(game) + '\n')
                stream.flush()
                if len(games) % 60 == 0:
                    print(f'[Held-out skills] {model}: {len(games)}/{len(cases) * 6}', flush=True)
            summaries[model] = {group: {kind: summarize_games([g for g in games if
                g['group'] == group and g['opponent'] == kind]) for kind in ('static', 'evader', 'rule')}
                for group in sorted({c['group'] for c in cases})}
            dump(output / 'heldout_skill_summary.json', summaries)
            print(f'[Held-out skills] {model}: {len(games)} games', flush=True)
    return summaries, True


def matched_budget_diagnostic(output, episode, stop):
    """Known validation boards only; this cannot unlock test boards or select a model."""
    results = {}
    paths = {'control_budget': output / 'control' / f'budget_ep{episode}.pt',
             'curriculum_final': output / 'curriculum/final_phaseC.pt'}
    for name, path in paths.items():
        print(f'[Matched-budget diagnostic] {name} ep{episode}: 100 validation games', flush=True)
        policy = make_ac_fn(str(path), 9, 'cpu')
        def progress(done, total):
            if done % 25 == 0:
                print(f'[Matched-budget diagnostic] {name} {done}/{total}', flush=True)
        result = evaluate_seeded(policy, 9, 2, 100, 21000, 500, stop_event=stop, on_progress=progress)
        if result is None:
            return False
        result.update(checkpoint=str(path), checkpoint_sha256=digest(path),
                      budget_episode=episode, diagnostic_only=True,
                      passive_win_rate=result['win_rate'] - result['active_win_rate'])
        for corner in result['by_corner'].values():
            corner['passive_win_rate'] = corner['win_rate'] - corner['active_win_rate']
        results[name] = result
        dump(output / 'matched_budget_diagnostic_21000.json', results)
    return True


def run(output, smoke=False, device=None, seconds=0):
    output = Path(output).resolve()
    if output.exists() or Path(str(output) + '-send.zip').exists():
        raise FileExistsError('Choose a new output folder; runs cannot be resumed or overwritten')
    if seconds < 0:
        raise ValueError('seconds must be nonnegative')
    device = device or ('cpu' if smoke else 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the torch environment or explicitly --device cpu')
    sources = {name: digest(ROOT / name) for name in SOURCE_FILES}
    original_hash = digest(WARM_START)
    output.mkdir(parents=True)
    (output / 'source').mkdir()
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, output / 'source' / name)
    config = {'started_utc': datetime.now(timezone.utc).isoformat(), 'smoke': smoke,
              'full_protocol': not smoke, 'episodes_per_arm': 6 if smoke else 600,
              'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                          'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None},
              'train_seed': TRAIN_SEED, 'device': device, 'teacher': 'local', 'max_steps': 8 if smoke else 500,
              'lr': 1e-4, 'skill_min_episodes': 100, 'skill_max_episodes': 200, 'skill_probe_every': 50,
              'skill_success_percent': 60, 'skill_consecutive_passes': 2, 'skill_seeds': SKILL_SEEDS,
              'warm_start': str(WARM_START), 'warm_start_sha256': original_hash,
              'source_sha256': sources, 'registry_sha256': digest(REFERENCE / 'scenario_registry.json'),
              'seconds_limit': seconds or None, 'arm_order': ['control', 'curriculum'],
              'evaluation_seed': 29800 if smoke else 31000, 'evaluation_games_per_model': 2 if smoke else 400,
              'phaseC_validation': {'every': 2 if smoke else 100, 'games': 2 if smoke else 100, 'seed': 21000},
              'curriculum': {'C1': '50% certified train escape/approach, 50% random density0.25; static',
                             'C2': 'certified train combo ammo2/range2-3; evader never bombs',
                             'C3': '50% train combo, 50% normal Phase C random maps; RuleBot',
                             'C4': 'normal Phase C RuleBot consolidation to episode600 after C3 passes'},
              'control': 'normal Phase C random maps/RuleBot for all600 episodes; same LR/entropy schedule',
              'heldout_family': None if smoke else 'wood_branch; fixed final skill evaluation only',
              'early_stage_failure_diagnostic': 'equal-episode final weights on validation21000; never test31000',
              'automatic_model_replacement': False}
    dump(output / 'config.json', config)
    dump(output / 'scenario_registry.json', json.loads((REFERENCE / 'scenario_registry.json').read_text()))
    started, stop, arms, results, complete_eval, complete_skills = perf_counter(), threading.Event(), {}, {}, False, False
    diagnostic_complete = False
    timer = threading.Timer(seconds, stop.set) if seconds else None
    if timer:
        timer.daemon = True
        timer.start()
    old_signal = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    torch.set_num_threads(1)
    print(f'Output: {output}\nMode: {"smoke" if smoke else "full"}; device={device}; source trainer unchanged', flush=True)
    dump(output / 'status.json', {'status': 'running', 'phase': 'preflight'})
    try:
        net = BaseNet(board_size=9)
        trainer.smart_load_weights(net, str(WARM_START), 'cpu')
        baseline = preflight(lambda stage: skill_probe(net, stage, stop, smoke), stop)
        if baseline:
            dump(output / 'baseline_skill.json', baseline)
            guards = {stage: result['suicide']['percent'] for stage, result in baseline.items()}
            for arm in ('control', 'curriculum'):
                if stop.is_set():
                    break
                dump(output / 'status.json', {'status': 'running', 'phase': 'train_' + arm, 'arms': arms})
                arms[arm] = train_arm(arm, output / arm, guards, stop, device, smoke)
                if not arms[arm]['training_complete'] or not arms[arm]['validation_complete']:
                    break
        ready = (len(arms) == 2 and all(a['training_complete'] and a['validation_complete'] and
                 a['curriculum_stages_complete'] for a in arms.values())) if smoke else full_pair_ready(arms)
        failure = arms.get('curriculum', {}).get('stage_failure')
        if failure and not smoke and not stop.is_set() and arms['control']['training_complete']:
            dump(output / 'status.json', {'status': 'running', 'phase': 'matched_budget_diagnostic', 'arms': arms})
            diagnostic_complete = matched_budget_diagnostic(output, failure['episode'], stop)
        if ready and not stop.is_set():
            dump(output / 'status.json', {'status': 'running', 'phase': 'random_map_evaluation', 'arms': arms})
            results, complete_eval = evaluate_models(output, stop, smoke)
            if complete_eval and not smoke:
                dump(output / 'status.json', {'status': 'running', 'phase': 'heldout_skills', 'arms': arms})
                _, complete_skills = heldout_skills(output, stop)
        assert original_hash == digest(WARM_START)
        assert sources == {name: digest(ROOT / name) for name in SOURCE_FILES}
        assert config['registry_sha256'] == digest(REFERENCE / 'scenario_registry.json')
        complete = ready and complete_eval and (smoke or complete_skills) and not stop.is_set()
        comparisons = {n: paired_active_bootstrap(results['curriculum'], results[n]) for n in ('original', 'control')} \
                      if complete and not smoke else {}
        eligible = (bool(comparisons) and all(c['difference_pp'] >= 5 and c['ci95_pp'][0] > 0 for c in comparisons.values())
                    and all(results['curriculum']['suicide_rate'] <= results[n]['suicide_rate'] for n in ('original', 'control')))
        summary = {'status': 'complete' if complete else 'incomplete', 'full_protocol': not smoke,
                   'reason': 'smoke_complete' if complete and smoke else 'protocol_complete' if complete else
                             'stage_gate_failed' if failure else 'cancelled_or_budget_exhausted' if stop.is_set()
                             else 'protocol_incomplete',
                   'arms': arms, 'random_map_evaluation_complete': complete_eval,
                   'heldout_skills_complete': complete_skills, 'comparisons': comparisons,
                   'matched_budget_diagnostic_complete': diagnostic_complete,
                   'eligible_for_replication': bool(eligible), 'hashes_unchanged': True,
                   'decision': 'replicate_before_replacing' if eligible else 'retain_original',
                   'elapsed_seconds': perf_counter() - started,
                   'models': {n: {k: v for k, v in r.items() if k != 'games'} for n, r in results.items()}}
        dump(output / 'summary.json', summary)
        dump(output / 'status.json', {k: summary[k] for k in ('status', 'reason', 'arms', 'elapsed_seconds')})
        archive = shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(f'{summary["status"].upper()}: {archive}\nDecision: {summary["decision"]}', flush=True)
        return summary
    except BaseException as exc:
        dump(output / 'status.json', {'status': 'incomplete', 'reason': f'{type(exc).__name__}: {exc}', 'arms': arms})
        raise
    finally:
        signal.signal(signal.SIGINT, old_signal)
        if timer:
            timer.cancel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / ('t3-curriculum-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--seconds', type=int, default=0, help='Whole-run deadline; 0 means no time limit')
    args = parser.parse_args()
    run(args.output, args.smoke, args.device, args.seconds)
