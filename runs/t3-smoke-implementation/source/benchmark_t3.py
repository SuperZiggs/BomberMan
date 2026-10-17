"""T3.1: certify skill scenarios, audit the teacher, then evaluate frozen CNNs."""
import argparse
from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
import shutil
from time import perf_counter

import numpy as np
import torch

from audit_l2_candidates import collect_candidates
from audit_t2 import check_plan, digest, dump, escape_suffix, reachable_comparison, serial_plan
from bomberman_env import (BombermanFFAEnv, Bomb, RIGID_WALL, WOODEN_WALL,
                           ACTION_STOP, ACTION_BOMB, _MOVE_DELTA)
from evaluate import make_ac_fn
from rule_based_bot import RuleBasedAgent
from audit_t2 import opponent_action
from train_actor_critic import get_action_mask, get_attack_action

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = {'original': ROOT / 'runs/pressure-C-next/final_phaseC.pt',
               'two_bomb_final': ROOT / 'runs/two-bomb-complete-20261014-031626/two_bomb/final_phaseC.pt'}
FAMILIES = {'fork': 'train', 'long_retreat': 'validation', 'wood_branch': 'test'}
# Same geometry as the existing combo_arena fixture, embedded in a 9x9 board.
FORK_CELLS = [(3, 3), (3, 4), (3, 5), (2, 3), (2, 4), (2, 5),
              (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (3, 2)]
SOURCE_FILES = ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py',
                'tactical_teacher.py', 'evaluate.py', 'visualize.py', 'audit_t2.py',
                'audit_l2_candidates.py', 'benchmark_t3.py', 'test_benchmark_t3.py')


def transform(pos, rotation, mirror):
    r, c = pos
    if mirror:
        c = 8 - c
    for _ in range(rotation):
        r, c = c, 8 - r
    return r, c


def transform_action(action, rotation, mirror):
    if action not in _MOVE_DELTA:
        return action
    origin = transform((4, 4), rotation, mirror)
    dr, dc = _MOVE_DELTA[action]
    dest = transform((4 + dr, 4 + dc), rotation, mirror)
    delta = dest[0] - origin[0], dest[1] - origin[1]
    return next(a for a, d in _MOVE_DELTA.items() if d == delta)


def scenarios(splits=('train', 'validation'), smoke=False):
    result = []
    for family, split in FAMILIES.items():
        if split not in splits:
            continue
        bases = [('escape', 1, 2, 'single_trap', (3, 3), [5]),
                 ('approach', 1, 2, 'single_trap', (1, 1), [4, 4, 2, 2, 5]),
                 ('evader', 1, 2, 'pressure', (3, 3), [5]),
                 ('combo', 2, 2, 'combo', (3, 3), [5, 1, 5]),
                 ('combo', 2, 3, 'combo', (3, 3), [5, 1, 5]),
                 ('ammo_control', 1, 2, 'illegal_capacity', (3, 3), [5, 1, 5])]
        if family == 'fork':
            bases.append(('ammo_control', 1, 2, 'delayed_capacity', (3, 3), [5, 1, 1, 3, 0, 0, 0, 0, 5]))
        for group, ammo, blast_range, expected, start, prefix in bases:
            for rotation in range(1 if smoke else 4):
                for mirror in (False,) if smoke else (False, True):
                    for aid in (0,) if smoke else (0, 1):
                        case_id = f'{family}-{group}-a{ammo}-r{blast_range}-{expected}-rot{rotation}-m{int(mirror)}-A{aid}'
                        result.append({'id': case_id, 'family': family, 'split': split, 'group': group,
                            'ammo': ammo, 'range': blast_range, 'expected': expected,
                            'start': list(start), 'prefix': prefix, 'rotation': rotation, 'mirror': mirror, 'aid': aid})
    return result


def make_env(case, seed=2026101401):
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500,
                         wooden_wall_density=0, powerup_prob=0, seed=seed)
    env.reset()
    env.grid[:] = RIGID_WALL
    cells = list(FORK_CELLS)
    if case['group'] in ('escape', 'approach'):
        cells = [pos for pos in cells if pos not in ((2, 4), (2, 5))]
    if case['family'] == 'long_retreat':
        cells += [(4, 1), (5, 1), (5, 2), (5, 3)]
    elif case['family'] == 'wood_branch':
        cells += [(4, 3), (5, 3)]
    if case['expected'] == 'delayed_capacity':
        cells += [(1, 4)]
    rotation, mirror, aid = case['rotation'], case['mirror'], case['aid']
    for pos in cells:
        env.grid[transform(pos, rotation, mirror)] = 0
    if case['family'] == 'wood_branch':
        env.grid[transform((4, 3), rotation, mirror)] = WOODEN_WALL
        pos = transform((5, 3), rotation, mirror)
        env.flames[pos], env._flame_owners[pos], env._flame_sources[pos] = 2, 1 - aid, {1 - aid: 2}
    if case['expected'] == 'delayed_capacity':
        env.grid[transform((1, 4), rotation, mirror)] = WOODEN_WALL
    env.agents[aid].row, env.agents[aid].col = transform(case['start'], rotation, mirror)
    enemy_start = (2, 5) if case['expected'] == 'delayed_capacity' else (3, 5)
    env.agents[1 - aid].row, env.agents[1 - aid].col = transform(enemy_start, rotation, mirror)
    env.agents[aid].ammo, env.agents[aid].blast_range = case['ammo'], case['range']
    validate_env(env)
    return env


def validate_env(env):
    if env.grid.shape != (9, 9) or set(np.unique(env.grid)) - {0, 1, 2}:
        raise ValueError('Invalid 9x9 grid')
    if not all(np.all(side == RIGID_WALL) for side in (env.grid[0], env.grid[-1], env.grid[:, 0], env.grid[:, -1])):
        raise ValueError('Open border')
    positions = []
    for aid, agent in enumerate(env.agents):
        pos = agent.row, agent.col
        if (agent.agent_id != aid or env.grid[pos] != 0 or pos in env.flames
                or not 1 <= agent.ammo <= 3 or not 1 <= agent.blast_range <= 4
                or agent.bombs_placed != sum(b.owner == aid for b in env.bombs)
                or agent.bombs_placed > agent.ammo):
            raise ValueError('Invalid agent/capacity state')
        positions.append(pos)
    if len(set(positions)) != len(positions):
        raise ValueError('Overlapping agents')
    if any(b.timer < 1 or env.grid[b.row, b.col] != 0 for b in env.bombs):
        raise ValueError('Invalid existing bomb')
    if any(env.grid[pos] != 0 or duration < 1 for pos, duration in env.flames.items()):
        raise ValueError('Invalid fire')
    if any(obs.shape != (12, 9, 9) for obs in observations(env)):
        raise ValueError('Unexpected observation channels')


def observations(env):
    danger = env._danger_map()
    return [env._get_obs(aid, danger) for aid in (0, 1)]


def reference_plan(env, case):
    aid = case['aid']
    path = [transform_action(a, case['rotation'], case['mirror']) for a in case['prefix']]
    pos, schedule = (env.agents[aid].row, env.agents[aid].col), []
    for delay, action in enumerate(path):
        if action == ACTION_BOMB:
            schedule.append((delay, Bomb(*pos, aid, env.bomb_timer, env.agents[aid].blast_range)))
        elif action in _MOVE_DELTA:
            dr, dc = _MOVE_DELTA[action]
            pos = pos[0] + dr, pos[1] + dc
    burns, _ = env._forecast_blasts(scheduled_bombs=schedule)
    horizon = max(len(path), int(np.flatnonzero(burns.any(axis=(1, 2)))[-1]) + 1)
    plan = {'path': path, 'bombs': schedule, 'horizon': horizon,
            'pressure': 0.0, 'trapped': 0, 'before': [], 'after': []}
    quality = reachable_comparison(env, aid, plan)
    before, after = quality['survivors']['baseline'], quality['survivors']['combo']
    plan.update(before=before, after=after,
                pressure=max((1 - a / b for b, a in zip(before, after) if b > 0), default=0.0),
                trapped=sum(b > 0 and a == 0 for b, a in zip(before, after)))
    return plan, quality


def genuine_combo(quality):
    q = quality['survivors']
    return ('first_only' in q and isinstance(q['first_only'], list)
            and isinstance(q['second_only'], list)
            and any(b > 0 and c == 0 and f > 0 and s > 0 for b, c, f, s in
                    zip(q['baseline'], q['combo'], q['first_only'], q['second_only'])))


def step_checked(env, aid, action):
    mask = get_action_mask(env, aid)
    if not mask[action]:
        return None, 'mask_blocked'
    before = env.agents[aid].row, env.agents[aid].col
    actions = [0, 0]
    actions[aid] = action
    _, _, _, info = env.step(actions)
    if action == ACTION_BOMB and aid not in info['bombs_placed']:
        return info, 'placement_failed'
    if action in _MOVE_DELTA:
        dr, dc = _MOVE_DELTA[action]
        if (env.agents[aid].row, env.agents[aid].col) != (before[0] + dr, before[1] + dc):
            return info, 'movement_failed'
    return info, 'complete' if env.agents[aid].alive else 'learner_died'


def masked_escape(env, aid, ticks):
    physical = escape_suffix(env, aid, ticks)
    if physical is not None:
        branch = deepcopy(env)
        if all(step_checked(branch, aid, a)[1] == 'complete' for a in physical):
            return physical, 'complete', 0
    # shortcut: bound certificate search; exhaustion is unknown, never a no-solution claim.
    queue, seen, expanded = deque([(deepcopy(env), [])]), set(), 0
    while queue:
        branch, path = queue.popleft()
        if len(path) == ticks:
            return path, 'complete', expanded
        state = (len(path), pickle.dumps(branch.__dict__))
        if state in seen:
            continue
        seen.add(state)
        expanded += 1
        if expanded > 2048:
            return None, 'search_limit', expanded
        mask = get_action_mask(branch, aid)
        for action in range(ACTION_BOMB):
            if mask[action]:
                child = deepcopy(branch)
                if step_checked(child, aid, action)[1] == 'complete':
                    queue.append((child, path + [action]))
    return None, 'no_masked_escape', expanded


def certify(env, aid, plan):
    before = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
    started, branch, tape, post_terminal = perf_counter(), deepcopy(env), [], False
    result = {'status': 'complete', 'prefix_pass': False, 'full_mask_pass': False,
              'full_horizon_alive': None, 'blocked_delay': None, 'blocked_action': None,
              'suffix_search_nodes': 0}
    for delay, action in enumerate(plan['path']):
        if branch._episode_done:
            result.update(status='terminal_before_prefix', blocked_delay=delay, blocked_action=action)
            break
        info, status = step_checked(branch, aid, action)
        if info is not None:
            tape.append(action)
        if status != 'complete':
            result.update(status=status, blocked_delay=delay, blocked_action=action)
            break
        if action == ACTION_BOMB:
            expected = next(b for d, b in plan['bombs'] if d == delay)
            agent = branch.agents[aid]
            if (agent.row, agent.col, agent.blast_range) != (expected.row, expected.col, expected.blast_range):
                result.update(status='schedule_changed', blocked_delay=delay, blocked_action=action)
                break
    else:
        result['prefix_pass'] = True
        suffix, status, nodes = masked_escape(branch, aid, max(0, plan['horizon'] - len(tape)))
        result.update(status=status, suffix_search_nodes=nodes)
        if suffix is not None:
            for action in suffix:
                post_terminal |= branch._episode_done
                assert step_checked(branch, aid, action)[1] == 'complete'
            tape += suffix
            result.update(full_mask_pass=True, full_horizon_alive=branch.agents[aid].alive)
    result.update(action_tape=tape, learner_alive=branch.agents[aid].alive,
                  owned_kills=branch.agents[aid].kills - env.agents[aid].kills,
                  post_terminal_physics=post_terminal,
                  milliseconds=(perf_counter() - started) * 1000)
    assert pickle.dumps(env.__dict__) == before[0] and pickle.dumps(np.random.get_state()) == before[1]
    assert torch.equal(torch.get_rng_state(), before[2])
    return result


def tape_mask_check(env, aid, check):
    branch = deepcopy(env)
    for tick, actions in enumerate(check['action_tape']):
        if not get_action_mask(branch, aid)[actions[aid]]:
            return {'all_tape_actions_masked': False, 'first_blocked_delay': tick}
        branch.step(actions)
    return {'all_tape_actions_masked': True, 'first_blocked_delay': None}


def audit_case(case, seed):
    env, aid = make_env(case, seed), case['aid']
    initial = pickle.dumps(env.__dict__)
    row = {'scenario_id': case['id'], 'family': case['family'], 'split': case['split'],
           'group': case['group'], 'expected': case['expected'], 'aid': aid}
    try:
        reference, quality = reference_plan(env, case)
    except ValueError as exc:
        row.update(reference_status='illegal_capacity', reference_error=str(exc), reference=None)
        assert case['expected'] == 'illegal_capacity'
    else:
        proof = certify(env, aid, reference)
        row.update(reference_status=proof['status'], reference=serial_plan(reference, 0),
                   reference_quality=quality, reference_certificate=proof,
                   reference_genuine_combo=genuine_combo(quality))
        stress = {}
        rng = np.random.get_state()
        try:
            for index, kind in enumerate(('evader', 'rule')):
                np.random.seed(seed + index)
                checked = check_plan(env, aid, reference, kind, True)
                stress[kind] = {**checked, **tape_mask_check(env, aid, checked)}
        finally:
            np.random.set_state(rng)
        row['reference_stress'] = stress
    mask = get_action_mask(env, aid)
    original, stats, candidates = collect_candidates(env, aid, mask)
    row.update(teacher_stats=stats, scored_paths=len(candidates),
               local_first_action=get_attack_action(env, aid, env._danger_map(), mask), old_best=None)
    if original:
        row['old_best'] = {'plan': serial_plan(original, 0), 'certificate': certify(env, aid, original),
                           'quality': reachable_comparison(env, aid, original)}
    row['masked_genuine_combo_candidate'], row['combo_candidates_checked'] = None, 0
    if case['expected'] == 'combo':
        for candidate in candidates:
            plan = candidate['plan']
            if len(plan['bombs']) != 2 or not plan['trapped']:
                continue
            row['combo_candidates_checked'] += 1
            proof = certify(env, aid, plan)
            if proof['full_mask_pass']:
                quality = reachable_comparison(env, aid, plan)
                if genuine_combo(quality):
                    row['masked_genuine_combo_candidate'] = {'plan': serial_plan(plan, 0),
                                                            'certificate': proof, 'quality': quality}
                    break
    assert pickle.dumps(env.__dict__) == initial
    return row


def play_case(case, policy, model, kind, seed, horizon):
    env, aid, tape = make_env(case, seed), case['aid'], []
    bot = RuleBasedAgent(1 - aid)
    np.random.seed(seed)
    torch.manual_seed(seed)
    suicide, placements, failed_moves = 0, [], 0
    for tick in range(horizon):
        mask = get_action_mask(env, aid)
        action = int(policy(env, observations(env), aid))
        assert mask[action], 'CNN chose an action outside the production mask'
        actions = [0, 0]
        actions[aid], actions[1 - aid] = action, opponent_action(env, 1 - aid, kind, bot)
        before = env.agents[aid].row, env.agents[aid].col
        _, _, _, info = env.step(actions)
        if aid in info['bombs_placed']:
            placements.append({'tick': tick, 'position': list(before)})
        if action in _MOVE_DELTA:
            dr, dc = _MOVE_DELTA[action]
            failed_moves += (env.agents[aid].row, env.agents[aid].col) != (before[0] + dr, before[1] + dc)
        suicide |= aid in info['self_blast_deaths'] or (aid, aid) in info['kills']
        tape.append({'tick': tick, 'actions': actions, 'mask': list(map(bool, mask)),
                     'bombs_placed': info['bombs_placed'], 'kills': info['kills'],
                     'self_blast_deaths': info['self_blast_deaths'], 'alive': info['alive'],
                     'positions': [[a.row, a.col] for a in env.agents]})
        if info['episode_done']:
            break
    agent, enemy = env.agents[aid], env.agents[1 - aid]
    active = agent.alive and not enemy.alive and agent.kills > 0
    return {'scenario_id': case['id'], 'family': case['family'], 'split': case['split'],
            'group': case['group'], 'expected': case['expected'], 'aid': aid, 'model': model,
            'opponent': kind, 'seed': seed, 'owned_kill_success': bool(active),
            'suicide': bool(suicide), 'kills': agent.kills, 'learner_alive': agent.alive,
            'passive_win': agent.alive and not enemy.alive and agent.kills == 0,
            'actual_draw': bool(env._episode_done and sum(a.alive for a in env.agents) != 1),
            'skill_horizon_truncated': not env._episode_done, 'steps': env.step_count,
            'placements': placements, 'two_or_more_placements': len(placements) >= 2,
            'failed_moves': int(failed_moves), 'tape': tape}


def summarize_games(games):
    return {'games': len(games), **{key: {'count': sum(bool(g[key]) for g in games),
            'percent': 100 * sum(bool(g[key]) for g in games) / len(games) if games else None}
            for key in ('owned_kill_success', 'suicide', 'two_or_more_placements', 'passive_win',
                        'actual_draw', 'skill_horizon_truncated')},
            'mean_steps': float(np.mean([g['steps'] for g in games])) if games else None,
            'mean_kills': float(np.mean([g['kills'] for g in games])) if games else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--splits', nargs='+', choices=tuple(FAMILIES.values()), default=['train', 'validation'])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=80)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 10 or not 10 <= args.horizon <= 500 or args.seconds < 1:
        parser.error('repeats 1..10, horizon 10..500, seconds positive')
    output = (args.output or ROOT / ('runs/t3-scenarios-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    if output.exists() or Path(str(output) + '-send.zip').exists():
        raise FileExistsError('Choose a new output folder')
    sources = {name: digest(ROOT / name) for name in SOURCE_FILES}
    checkpoints = {name: digest(path) for name, path in CHECKPOINTS.items()}
    cases = scenarios(args.splits, args.smoke)
    assert cases and len({c['id'] for c in cases}) == len(cases)
    output.mkdir(parents=True)
    (output / 'source').mkdir()
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, output / 'source' / name)
    config = {'started_utc': datetime.now(timezone.utc).isoformat(), 'splits': args.splits,
              'family_splits': FAMILIES, 'test_split_run': 'test' in args.splits,
              'smoke': args.smoke, 'repeats': args.repeats, 'skill_horizon': args.horizon,
              'env_max_steps': 500, 'board_size': 9, 'num_agents': 2,
              'rollout_seeds': [2026101401 + r for r in range(args.repeats)],
              'opponents': ['static', 'evader', 'rule'], 'scenarios': len(cases),
              'expected_games': len(cases) * len(CHECKPOINTS) * 3 * args.repeats,
              'seconds': args.seconds, 'source_sha256': sources, 'checkpoint_sha256': checkpoints,
              'checkpoints': {n: str(p) for n, p in CHECKPOINTS.items()},
              'policy': 'existing make_ac_fn / SnapshotOpponent: categorical sampling through production mask',
              'teacher_selects_cnn_action': False, 'certificate_opponent': 'static STOP',
              'escape_search_bound': 2048, 'promotion_allowed': False}
    dump(output / 'config.json', config)
    dump(output / 'scenarios.json', cases)
    dump(output / 'status.json', {'status': 'running', 'cases_audited': 0, 'games_completed': 0})
    started, audits, games = perf_counter(), [], []
    deadline = started + args.seconds
    torch.set_num_threads(1)
    rng = np.random.get_state()
    print(f'Output: {output}\nCases: {len(cases)}; games: {config["expected_games"]}; test_run={config["test_split_run"]}', flush=True)
    try:
        with torch.random.fork_rng(devices=[]):
            for case in cases:
                if perf_counter() >= deadline:
                    break
                audits.append(audit_case(case, config['rollout_seeds'][0]))
                if len(audits) % 16 == 0 or len(audits) == len(cases):
                    dump(output / 'status.json', {'status': 'running', 'cases_audited': len(audits), 'games_completed': 0})
                    print(f'[Certificates/teacher {len(audits)}/{len(cases)}]', flush=True)
            dump(output / 'certificates_teacher.json', audits)
            if len(audits) == len(cases):
                policies = {name: make_ac_fn(str(path), 9, 'cpu') for name, path in CHECKPOINTS.items()}
                with gzip.open(output / 'games.jsonl.gz', 'wt', encoding='utf-8') as stream:
                    for case in cases:
                        if perf_counter() >= deadline:
                            break
                        for kind in config['opponents']:
                            for seed in config['rollout_seeds']:
                                for model, policy in policies.items():
                                    game = play_case(case, policy, model, kind, seed, args.horizon)
                                    games.append(game)
                                    stream.write(json.dumps(game) + '\n')
                        stream.flush()
                        dump(output / 'status.json', {'status': 'running', 'cases_audited': len(audits),
                              'games_completed': len(games), 'elapsed_seconds': perf_counter() - started})
                        print(f'[{len(games)}/{config["expected_games"]}] {case["id"]}', flush=True)
        assert sources == {name: digest(ROOT / name) for name in SOURCE_FILES}
        assert checkpoints == {name: digest(path) for name, path in CHECKPOINTS.items()}
        positive = [a for a in audits if a['expected'] == 'combo']
        gate = {'positive_combo_cases': len(positive),
                'reference_combo_full_mask_pass': sum(a.get('reference_genuine_combo', False) and
                    a['reference_certificate']['full_mask_pass'] for a in positive),
                'teacher_old_best_genuine_full_mask_pass': sum(bool(a['old_best'] and
                    a['old_best']['certificate']['full_mask_pass'] and genuine_combo(a['old_best']['quality'])) for a in positive),
                'teacher_any_genuine_full_mask_pass': sum(a['masked_genuine_combo_candidate'] is not None for a in positive)}
        complete = len(audits) == len(cases) and len(games) == config['expected_games']
        summary = {'status': 'complete' if complete else 'incomplete', 'hashes_unchanged': True,
                   'cases_audited': len(audits), 'games_completed': len(games), 'expected_games': config['expected_games'],
                   'elapsed_seconds': perf_counter() - started, 'gate': gate,
                   'reference_outcomes': dict(Counter(a['reference_status'] for a in audits)),
                   'by_model': {model: {group: {kind: summarize_games([g for g in games if
                       (g['model'], g['group'], g['opponent']) == (model, group, kind)]) for kind in config['opponents']}
                       for group in sorted({c['group'] for c in cases})} for model in CHECKPOINTS},
                   'by_corner': {model: {str(aid): summarize_games([g for g in games if g['model'] == model and g['aid'] == aid])
                                 for aid in (0, 1)} for model in CHECKPOINTS},
                   'decision': 'review_required_keep_original',
                   'limits': ['Synthetic skill benchmark, not random-map ActiveWR or checkpoint promotion.',
                             'Rotations/reflections/roles/RNG repeats are dependent; base layout family is the unit.',
                             'Related small layout families; test family must stay locked until a registered comparison.',
                             'Certificate is static and mask-valid through the hazard horizon; stress futures are sampled.',
                             '80-step skill limit is external truncation; env rules/reward/max_steps500 stay unchanged.']}
        dump(output / 'summary.json', summary)
        dump(output / 'status.json', {k: summary[k] for k in ('status', 'cases_audited', 'games_completed', 'elapsed_seconds')})
        archive = shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(f'{summary["status"].upper()}: {archive}\nGate: {json.dumps(gate)}', flush=True)
        return output
    except BaseException as exc:
        dump(output / 'certificates_teacher.json', audits)
        dump(output / 'status.json', {'status': 'incomplete', 'reason': f'{type(exc).__name__}: {exc}',
                                     'cases_audited': len(audits), 'games_completed': len(games)})
        raise
    finally:
        np.random.set_state(rng)


if __name__ == '__main__':
    main()
