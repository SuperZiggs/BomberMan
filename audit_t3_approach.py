"""Offline local-label audit on already-viewed C1 tapes; never opens a test split."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import torch

from audit_t2 import digest, dump
from benchmark_t3 import make_env, observations, scenarios
from bomberman_env import ACTION_BOMB, _MOVE_DELTA
from rule_based_bot import _bfs_safe_cell, _build_danger_map, _bomb_escape_pressure
from train_actor_critic import (ActorCriticNet, get_action_mask, get_attack_action_v1 as get_attack_action,
                                obs_to_tensor, smart_load_weights)

ROOT = Path(__file__).resolve().parent
T3 = ROOT / 'runs/t3-curriculum-full-20261014-154703'
T31 = ROOT / 'runs/t3-full-20261014-implementation'
CONFIRMATION = ROOT / 'runs/control-confirmation-20261014-201557'


def fingerprint(env):
    return (pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state()),
            torch.get_rng_state().numpy().tobytes())


def local_label(env, aid, mask):
    # Mirror the frozen trainer's local dispatch, including its three loss groups.
    danger = _build_danger_map(env)
    agent = env.agents[aid]
    if danger[agent.row, agent.col] > 0:
        if mask[ACTION_BOMB]:
            return ACTION_BOMB, 'combat', 0.5
        action = _bfs_safe_cell(env, agent.row, agent.col, danger, action_mask=mask)
        return (action, 'escape', 0.5) if action is not None and mask[action] else (None, 'missing', 0)
    action = get_attack_action(env, aid, danger, mask)
    if action is None:
        return None, 'missing', 0
    combat = action == ACTION_BOMB and _bomb_escape_pressure(env, aid, agent.row, agent.col) > 0
    return action, 'combat' if combat else 'navigation', 0.5 if combat else 0.2


def firing_distance(env, aid, position):
    """Static geometric distance, not a survival certificate or attack score."""
    enemy = {(a.row, a.col) for a in env.agents if a.alive and a.agent_id != aid}
    seen, queue = {position}, deque([(position, 0)])
    while queue:
        pos, distance = queue.popleft()
        if enemy.intersection(env._blast_cells(*pos, env.agents[aid].blast_range)):
            return distance
        for dr, dc in _MOVE_DELTA.values():
            dest = pos[0] + dr, pos[1] + dc
            if (dest not in seen and dest not in enemy and 0 <= dest[0] < env.board_size
                    and 0 <= dest[1] < env.board_size and env.grid[dest] == 0):
                seen.add(dest)
                queue.append((dest, distance + 1))
    return None


def label_frame(env, aid, action, tick):
    before = fingerprint(env)
    mask = get_action_mask(env, aid)
    label, kind, weight = local_label(env, aid, mask)
    assert fingerprint(env) == before, 'Teacher or diagnostic mutated state/RNG'
    assert mask[action], 'Saved CNN action is outside current mask'
    assert label is None or mask[label], 'Teacher label is outside mask'
    pos = env.agents[aid].row, env.agents[aid].col
    danger = _build_danger_map(env)
    distance = firing_distance(env, aid, pos)
    progress_moves = []
    if kind == 'navigation' and not env.bombs and not env.flames and distance is not None:
        for move, (dr, dc) in _MOVE_DELTA.items():
            if mask[move]:
                next_distance = firing_distance(env, aid, (pos[0] + dr, pos[1] + dc))
                if next_distance is not None and next_distance < distance:
                    progress_moves.append(move)
    target_distance = distance
    if label in _MOVE_DELTA:
        dr, dc = _MOVE_DELTA[label]
        target_distance = firing_distance(env, aid, (pos[0] + dr, pos[1] + dc))
    return {'tick': tick, 'position': list(pos), 'action': action, 'label': label,
            'kind': kind, 'weight': weight, 'agrees': label == action if label is not None else None,
            'mask': mask, 'firing_distance': distance, 'label_firing_distance': target_distance,
            'danger_at_self': int(danger[pos]), 'clean_state': not env.bombs and not env.flames,
            'geometric_progress_moves': progress_moves}


def read_selected(path, baseline=False):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            game = json.loads(line)
            if (game['split'] != 'validation' or game['group'] not in ('escape', 'approach')
                    or game['opponent'] != 'static'):
                continue
            if baseline and game['model'] != 'original':
                continue
            yield game


def replay_labels(game, case):
    env, frames = make_env(case, game['seed']), []
    aid = case['aid']
    for row in game['tape']:
        assert row['tick'] == env.step_count
        frame = label_frame(env, aid, row['actions'][aid], row['tick'])
        assert frame['mask'] == row['mask']
        _, _, _, info = env.step(row['actions'])
        assert [list((a.row, a.col)) for a in env.agents] == row['positions']
        for key in ('bombs_placed', 'alive', 'self_blast_deaths'):
            assert info[key] == row[key]
        assert [list(k) for k in info['kills']] == [list(k) for k in row['kills']]
        frames.append(frame)
    success = env.agents[aid].alive and env.agents[aid].kills > 0 and not env.agents[1 - aid].alive
    assert bool(success) == game['owned_kill_success'] and len(frames) == game['steps']
    return frames


def frame_summary(frames):
    labeled = [r for r in frames if r['label'] is not None]
    navigation = [r for r in frames if r['kind'] == 'navigation' and r['label'] in _MOVE_DELTA]
    geometric = [r for r in navigation if r['firing_distance'] is not None and r['label_firing_distance'] is not None]
    positions = [tuple(r['position']) for r in frames]
    by_distance = {}
    for row in navigation:
        if not row['clean_state']:
            continue
        stats = by_distance.setdefault(str(row['firing_distance']), Counter())
        stats.update({'frames': 1, 'teacher_agreement': int(row['agrees']),
                      'any_progress_move': int(row['action'] in row['geometric_progress_moves']),
                      'multiple_progress_moves': int(len(row['geometric_progress_moves']) > 1)})
    return {'frames': len(frames), 'labels': len(labeled), 'missing': len(frames) - len(labeled),
            'groups': dict(Counter(r['kind'] for r in frames)),
            'agreements': sum(r['agrees'] is True for r in labeled),
            'agreement_percent': 100 * sum(r['agrees'] is True for r in labeled) / max(1, len(labeled)),
            'navigation_move_labels': len(navigation),
            'navigation_move_agreements': sum(r['agrees'] is True for r in navigation),
            'navigation_geometric_reducing': sum(r['label_firing_distance'] < r['firing_distance'] for r in geometric),
            'navigation_geometric_not_reducing': sum(r['label_firing_distance'] >= r['firing_distance'] for r in geometric),
            'bomb_labels': sum(r['label'] == ACTION_BOMB for r in frames),
            'bomb_label_agreements': sum(r['label'] == ACTION_BOMB and r['agrees'] for r in frames),
            'unique_positions': len(set(positions)), 'revisited_frames': len(positions) - len(set(positions)),
            'clean_navigation_by_distance': {key: dict(value) for key, value in by_distance.items()}}


def teacher_rollout(case, seed):
    env, tape, missing = make_env(case, seed), [], 0
    aid = case['aid']
    for tick in range(80):
        before = fingerprint(env)
        mask = get_action_mask(env, aid)
        action, kind, _ = local_label(env, aid, mask)
        assert fingerprint(env) == before
        if action is None:
            missing += 1
            # Explicit fallback, reported separately from teacher-provided actions.
            action = next(a for a, allowed in enumerate(mask) if allowed)
        assert mask[action]
        agent = env.agents[aid]
        position = [agent.row, agent.col]
        clean = not env.bombs and not env.flames
        actions = [0, 0]
        actions[aid] = action
        _, _, _, info = env.step(actions)
        tape.append({'tick': tick, 'actions': actions, 'kind': kind, 'mask': mask,
                     'position_before': position, 'clean_state': clean})
        if info['episode_done']:
            break
    return {'scenario_id': case['id'], 'group': case['group'], 'seed': seed, 'steps': len(tape),
            'owned_kill_success': env.agents[aid].alive and env.agents[aid].kills > 0,
            'learner_alive': env.agents[aid].alive, 'missing_labels': missing, 'tape': tape}


def teacher_probabilities(cases, rollouts):
    result = {}
    checkpoints = {'original': ROOT / 'runs/pressure-C-next/final_phaseC.pt',
                   'curriculum_final200': T3 / 'curriculum/final_phaseC.pt'}
    for name, path in checkpoints.items():
        net = ActorCriticNet(board_size=9)
        smart_load_weights(net, str(path), 'cpu')
        net.eval()
        rows = []
        with torch.no_grad():
            for case in cases:
                if case['group'] != 'approach':
                    continue
                env, aid = make_env(case), case['aid']
                reference = next(r for r in rollouts if r['scenario_id'] == case['id'])
                for row in reference['tape']:
                    if row['kind'] == 'missing':
                        break
                    mask = get_action_mask(env, aid)
                    label, kind, _ = local_label(env, aid, mask)
                    logits, _ = net(obs_to_tensor(observations(env)[aid], 'cpu'))
                    probabilities = logits[0].masked_fill(~torch.tensor(mask), -1e9).softmax(0).tolist()
                    agent = env.agents[aid]
                    rows.append({'scenario_id': case['id'], 'tick': row['tick'], 'label': label, 'kind': kind,
                        'firing_distance': firing_distance(env, aid, (agent.row, agent.col)),
                        'probabilities': probabilities, 'teacher_probability': probabilities[label]})
                    env.step(row['actions'])
        result[name] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    torch.set_num_threads(1)
    start = perf_counter()
    protocol = json.loads((CONFIRMATION / 'protocol.json').read_text(encoding='utf-8-sig'))
    for name, expected in protocol['source_sha256'].items():
        assert digest(ROOT / name) == expected, f'Frozen source changed: {name}'
    inputs = [T31 / 'games.jsonl.gz', T3 / 'curriculum/skill_validation_games.jsonl.gz',
              CONFIRMATION / 'shared_suicide_replays.json',
              ROOT / 'runs/pressure-C-next/final_phaseC.pt', T3 / 'curriculum/final_phaseC.pt']
    hashes = {str(p): digest(p) for p in inputs}
    assert hashes[str(inputs[3])] == protocol['checkpoints']['original']['sha256']
    assert hashes[str(inputs[4])] == 'f5f6135dc5ca940e6efe2c6ed84c95a840c88b8f50348a9d0051f130e7d40945'
    out.mkdir(parents=True)
    cases = [c for c in scenarios(('validation',)) if c['group'] in ('escape', 'approach')]
    registry = {c['id']: c for c in cases}
    buckets, per_game = {}, []
    with gzip.open(out / 'label_frames.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for path, baseline in ((inputs[0], True), (inputs[1], False)):
            count, seen = 0, set()
            for game in read_selected(path, baseline):
                identity = (game['scenario_id'], game['seed'], 0 if baseline else game['episode'])
                assert identity not in seen, ('duplicate game', identity)
                seen.add(identity)
                cohort = 'original' if baseline else f"curriculum_ep{game['episode']}"
                frames = replay_labels(game, registry[game['scenario_id']])
                key = f"{cohort}/{game['group']}/{'success' if game['owned_kill_success'] else 'failure'}"
                buckets.setdefault(key, []).extend(frames)
                record = {'cohort': cohort, 'scenario_id': game['scenario_id'], 'seed': game['seed'],
                          'group': game['group'], 'owned_kill_success': game['owned_kill_success'],
                          **frame_summary(frames)}
                per_game.append(record)
                for frame in frames:
                    stream.write(json.dumps({**record, 'frame': frame}) + '\n')
                count += 1
            assert count == (64 if baseline else 192), (path, count)
            assert seen == {(case_id, seed, episode) for case_id in registry
                            for seed in (2026101401, 2026101402)
                            for episode in ((0,) if baseline else (100, 150, 200))}
            print(f'Replayed {count} C1 games from {path.name}; labels/state/RNG/masks verified', flush=True)
    closed_loop = [teacher_rollout(case, seed) for case in cases for seed in (2026101401, 2026101402)]
    dump(out / 'teacher_rollouts.json', closed_loop)
    dump(out / 'teacher_path_probabilities.json', teacher_probabilities(cases, closed_loop))
    from counterbomb_fixtures import certify_fixtures, replay_seen_counterbombs
    fixture_results = certify_fixtures()
    replay_results = replay_seen_counterbombs(json.loads(inputs[2].read_text()))
    dump(out / 'counterbomb_fixtures.json', fixture_results)
    dump(out / 'seen_counterbombs.json', replay_results)
    assert all(digest(Path(p)) == expected for p, expected in hashes.items())
    assert all(digest(ROOT / p) == expected for p, expected in protocol['source_sha256'].items())
    summary = {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Offline diagnostic on previously viewed validation and two shared suicide tapes; no train or fresh test',
        'input_sha256': hashes, 'source_sha256': protocol['source_sha256'],
        'audit_sha256': {p: digest(ROOT / p) for p in ('audit_t3_approach.py', 'counterbomb_fixtures.py', 'test_t3_approach.py')},
        'hashes_unchanged': True, 'saved_games': len(per_game), 'replayed_frames': sum(g['frames'] for g in per_game),
        'groups': {k: frame_summary(v) for k, v in buckets.items()}, 'per_game': per_game,
        'teacher_closed_loop': {group: {'games': sum(r['group'] == group for r in closed_loop),
            'successes': sum(r['group'] == group and r['owned_kill_success'] for r in closed_loop),
            'missing_labels': sum(r['missing_labels'] for r in closed_loop if r['group'] == group)} for group in ('escape', 'approach')},
        'counterbomb_fixtures': len(fixture_results), 'seen_counterbombs': len(replay_results),
        'seconds': perf_counter() - start, 'retain_original': True}
    dump(out / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'saved_games', 'replayed_frames', 'teacher_closed_loop',
                                               'counterbomb_fixtures', 'seen_counterbombs', 'seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
