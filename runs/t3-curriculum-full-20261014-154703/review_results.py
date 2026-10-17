"""Audit this T3.2 run without training or opening either held-out test set."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from math import isclose
from pathlib import Path
import sys
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


cfg, saved = read(OUT / 'config.json'), read(OUT / 'summary.json')
with zipfile.ZipFile(str(OUT) + '-send.zip') as archive:
    assert archive.testzip() is None
    entries = [i for i in archive.infolist() if not i.is_dir()]
    for item in entries:
        local = (OUT / item.filename).resolve()
        assert local.is_relative_to(OUT)
        with archive.open(item) as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == sha(local), item.filename
for name, expected in cfg['source_sha256'].items():
    assert sha(OUT / 'source' / name) == sha(ROOT / name) == expected, name
assert sha(Path(cfg['warm_start'])) == cfg['warm_start_sha256']
reference_registry = ROOT / 'runs/t3-full-20261014-implementation/scenario_registry.json'
assert sha(reference_registry) == cfg['registry_sha256']
assert read(OUT / 'scenario_registry.json') == read(reference_registry)
assert saved['status'] == 'incomplete' and saved['reason'] == 'stage_gate_failed'
assert not saved['random_map_evaluation_complete'] and not saved['heldout_skills_complete']
assert not saved['eligible_for_replication'] and saved['decision'] == 'retain_original'
assert not list(OUT.glob('evaluation_*_31000.json')) and not list(OUT.glob('heldout_*'))
print(f'ZIP {len(entries)} files, source/warm-start hashes, incomplete protocol: PASS', flush=True)

sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from benchmark_t3 import make_env, get_action_mask, played_pairs, summarize_games, transform
from pilot_two_bomb import paired_active_bootstrap
from run_t3_curriculum import model_digest, skill_cases
from train_actor_critic import ActorCriticNet

torch.set_num_threads(1)
registry = {c['id']: c for c in read(OUT / 'scenario_registry.json')['cases']}
rows, validations, training, arm_configs = {}, {}, {}, []
for arm, count in (('control', 600), ('curriculum', 200)):
    directory = OUT / arm
    acfg = read(directory / 'config.json')
    acfg.pop('checkpoint_dir')
    arm_configs.append(acfg)
    rows[arm], validations[arm] = lines(directory / 'metrics.jsonl'), lines(directory / 'validation_metrics.jsonl')
    rr, vv, ss = rows[arm], validations[arm], saved['arms'][arm]
    assert len(rr) == ss['completed_episodes'] == count
    assert [r['episode'] for r in rr] == [r['optimizer_steps'] for r in rr] == list(range(1, count + 1))
    assert all(r['learning_aid'] == (r['episode'] - 1) % 2 for r in rr)
    assert sum(r['learner_steps'] for r in rr) == ss['learner_steps']
    assert all(r['attack_teacher'] == 'local' for r in rr)
    assert all(rr[i]['lr'] > rr[i + 1]['lr'] for i in range(count - 1))
    assert [v['episode'] for v in vv] == list(range(0, count + 1, 100))
    assert all(v['num_episodes'] == 100 and v['seed'] == 21000 for v in vv)
    best = read(directory / 'best_validation.json')
    assert best['episode'] == (400 if arm == 'control' else 0)
    training[arm] = {'episodes': count, 'learner_steps': ss['learner_steps'], 'optimizer_steps': ss['optimizer_steps'],
                    'stage_counts': dict(Counter(r['stage'] for r in rr)), 'best_episode': best['episode'],
                    'validation': [{k: v[k] for k in ('episode', 'active_win_rate', 'draw_rate', 'suicide_rate', 'selected')}
                                   for v in vv], 'layouts': {}}
    for layout in sorted({r['layout'] for r in rr}):
        part = [r for r in rr if r['layout'] == layout]
        if layout == 'scenario':
            assert all(registry[r['scenario_id']]['split'] == 'train' and registry[r['scenario_id']]['family'] == 'fork'
                       and registry[r['scenario_id']]['group'] in ('escape', 'approach') for r in part)
        else:
            assert all(r['scenario_id'] is None for r in part)
        assert all(r['opponent'] == ('rule' if arm == 'control' else 'static') for r in part)
        training[arm]['layouts'][layout] = {'episodes': len(part), 'active_wins': sum(r['active_win'] for r in part),
            'suicides': sum(r['suicide'] for r in part), 'mean_steps': float(np.mean([r['learner_steps'] for r in part])),
            'mean_reward': float(np.mean([r['reward'] for r in part])), 'initial_ammo': dict(Counter(r['ammo'] for r in part))}
assert arm_configs[0] == arm_configs[1]
assert validations['control'][0] == validations['curriculum'][0]
assert saved['arms']['curriculum']['stage_events'] == []
print('800 episode rows, optimizer/LR/roles/layouts, ten Phase C validations: PASS', flush=True)

def load_state(path):
    return torch.load(path, map_location='cpu', weights_only=True)

original = load_state(Path(cfg['warm_start']))
states = {name: load_state(OUT / path) for name, path in {
    'curriculum_best': 'curriculum/best_phaseC.pt', 'curriculum_final': 'curriculum/final_phaseC.pt',
    'control_best': 'control/best_phaseC.pt', 'control_final': 'control/final_phaseC.pt',
    **{f'control_ep{ep}': f'control/budget_ep{ep}.pt' for ep in range(200, 601, 50)}}.items()}
assert all(torch.equal(original[k], states['curriculum_best'][k]) for k in original)
assert all(torch.equal(states['control_final'][k], states['control_ep600'][k]) for k in original)
tensor_changes = {name: sum(not torch.equal(original[k], value[k]) for k in original) for name, value in states.items()}
assert all(tensor_changes[name] > 0 for name in ('curriculum_final', 'control_best', 'control_final'))

probes = lines(OUT / 'curriculum/skill_validation_metrics.jsonl')
assert [p['episode'] for p in probes] == [100, 150, 200]
assert [p['passed'] for p in probes] == [True, False, True]
assert all(p['complete'] and not p['smoke_forced_transition'] and p['stage'] == 'C1_static' for p in probes)
net = ActorCriticNet(board_size=9)
net.load_state_dict(states['curriculum_final'])
assert model_digest(net) == probes[-1]['result']['model_state_sha256']

with gzip.open(OUT / 'curriculum/skill_validation_games.jsonl.gz', 'rt', encoding='utf-8') as stream:
    games = [json.loads(line) for line in stream]
expected = {(ep, c['id'], seed) for ep in (100, 150, 200) for c in skill_cases(1) for seed in cfg['skill_seeds']}
actual = [(g['episode'], g['scenario_id'], g['seed']) for g in games]
assert len(games) == len(set(actual)) == len(expected) == 192 and set(actual) == expected
frames = 0
for index, game in enumerate(games):
    case, aid = registry[game['scenario_id']], game['aid']
    assert case['split'] == 'validation' and game['opponent'] == 'static'
    env = make_env(case, game['seed'])
    placements, placement_states, suicide, failed_moves = [], [], False, 0
    for tick, row in enumerate(game['tape']):
        assert not env._episode_done and row['tick'] == tick and row['actions'][1 - aid] == 0
        mask = list(map(bool, get_action_mask(env, aid)))
        assert row['mask'] == mask and mask[row['actions'][aid]]
        agent, action = env.agents[aid], row['actions'][aid]
        before = agent.row, agent.col
        branch = deepcopy(env) if action == 5 else None
        _, _, _, info = env.step(row['actions'])
        if aid in info['bombs_placed']:
            placement_states.append(branch)
            placements.append({'tick': tick, 'position': list(before), 'range': agent.blast_range})
        suicide |= aid in info['self_blast_deaths'] or (aid, aid) in info['kills']
        if action in (1, 2, 3, 4):
            dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
            failed_moves += (agent.row, agent.col) != (before[0] + dr, before[1] + dc)
        assert row['bombs_placed'] == info['bombs_placed'] and row['kills'] == [list(k) for k in info['kills']]
        assert row['self_blast_deaths'] == info['self_blast_deaths'] and row['alive'] == info['alive']
        assert row['positions'] == [[a.row, a.col] for a in env.agents]
        frames += 1
    agent, enemy = env.agents[aid], env.agents[1 - aid]
    derived = {'owned_kill_success': bool(agent.alive and not enemy.alive and agent.kills > 0),
        'suicide': bool(suicide), 'kills': agent.kills, 'learner_alive': agent.alive,
        'passive_win': agent.alive and not enemy.alive and agent.kills == 0,
        'actual_draw': bool(env._episode_done and sum(a.alive for a in env.agents) != 1),
        'skill_horizon_truncated': not env._episode_done, 'steps': env.step_count, 'placements': placements,
        'two_or_more_placements': len(placements) >= 2, 'failed_moves': int(failed_moves)}
    assert all(game[k] == value for k, value in derived.items())
    pairs = played_pairs(placement_states, placements, aid)
    assert game['pairs'] == pairs and game['has_genuine_combo_pair'] == any(p['genuine_combo'] for p in pairs)
    if (index + 1) % 64 == 0:
        print(f'Rebuilt all outcomes and masks: {index + 1}/192 games', flush=True)
for probe in probes:
    subset = [g for g in games if g['episode'] == probe['episode']]
    assert all(probe['result'][k] == v for k, v in summarize_games(subset).items())

diagnostic = read(OUT / 'matched_budget_diagnostic_21000.json')
assert set(diagnostic) == {'control_budget', 'curriculum_final'}
for name, result in diagnostic.items():
    assert result['num_episodes'] == len(result['games']) == 100 and result['seed'] == 21000
    assert {(g['seed'], g['aid']) for g in result['games']} == {(s, a) for s in range(21000, 21050) for a in (0, 1)}
    assert result['budget_episode'] == 200 and result['diagnostic_only']
    assert result['checkpoint_sha256'] == sha(Path(result['checkpoint']))
    ref = validations['control' if name == 'control_budget' else 'curriculum'][-1 if name == 'curriculum_final' else 2]
    for corner, subset in [('overall', result['games']), *[(str(a), [g for g in result['games'] if g['aid'] == a]) for a in (0, 1)]]:
        recomputed = {f'{k}_rate': 100 * sum(g[k] for g in subset) / len(subset) for k in ('win', 'active_win', 'suicide', 'draw')}
        recomputed.update({f'avg_{k}': sum(g[k] for g in subset) / len(subset) for k in ('kills', 'bombs', 'items', 'steps')})
        target = result if corner == 'overall' else result['by_corner'][corner]
        validation = ref if corner == 'overall' else ref['by_corner'][corner]
        assert all(isclose(value, target[k]) and isclose(value, validation[k]) for k, value in recomputed.items())
comparison = paired_active_bootstrap(diagnostic['curriculum_final'], diagnostic['control_budget'])
comparison['interpretation'] = 'Exploratory validation21000; not the registered 400-game held-out test'
skill_by_episode = {str(ep): {group: summarize_games([g for g in games if g['episode'] == ep and g['group'] == group])
                             for group in ('escape', 'approach')} for ep in (100, 150, 200)}
previous = ROOT / 'runs/t3-full-20261014-implementation'
assert read(previous / 'config.json')['checkpoint_sha256']['original'] == cfg['warm_start_sha256']
with gzip.open(previous / 'games.jsonl.gz', 'rt', encoding='utf-8') as stream:
    baseline_games = []
    for line in stream:
        game = json.loads(line)
        if (game['model'] == 'original' and game['opponent'] == 'static' and game['split'] == 'validation'
                and game['group'] in ('escape', 'approach') and game['seed'] in cfg['skill_seeds']):
            baseline_games.append(game)
assert len(baseline_games) == 64
assert {(g['scenario_id'], g['seed']) for g in baseline_games} == {(cid, seed) for ep, cid, seed in expected}
baseline_saved = read(OUT / 'baseline_skill.json')['1']
assert all(baseline_saved[k] == v for k, v in summarize_games(baseline_games).items())
skill_by_episode['B'] = {group: summarize_games([g for g in baseline_games if g['group'] == group])
                         for group in ('escape', 'approach')}
approach_failures = {}
for label, selected in [('B', baseline_games), *[(str(ep), [g for g in games if g['episode'] == ep]) for ep in (100, 150, 200)]]:
    failed = [g for g in selected if g['group'] == 'approach' and not g['owned_kill_success']]
    reached, never_bombed, min_distances, first_actions, stops, steps = 0, 0, [], Counter(), 0, 0
    for game in failed:
        case, aid = registry[game['scenario_id']], game['aid']
        initial = make_env(case, game['seed'])
        a, enemy = initial.agents[aid], initial.agents[1 - aid]
        positions = [(a.row, a.col)] + [tuple(row['positions'][aid]) for row in game['tape']]
        reached += transform((3, 3), case['rotation'], case['mirror']) in positions
        never_bombed += not game['placements']
        min_distances.append(min(abs(r - enemy.row) + abs(c - enemy.col) for r, c in positions))
        first_actions[game['tape'][0]['actions'][aid]] += 1
        stops += sum(row['actions'][aid] == 0 for row in game['tape'])
        steps += len(game['tape'])
    approach_failures[label] = {'games': len(failed), 'skill_truncations': sum(g['skill_horizon_truncated'] for g in failed),
        'never_bombed': never_bombed, 'reached_reference_bomb_cell': reached,
        'closest_manhattan_distance_counts': dict(Counter(min_distances)), 'first_action_counts': dict(first_actions),
        'stop_actions_percent': 100 * stops / steps if steps else None}
result = {'zip_files_verified': len(entries), 'source_and_warm_hashes_verified': True,
    'training': training, 'checkpoint_tensor_changes_vs_B': tensor_changes, 'skill_probes': probes,
    'skill_games_rebuilt': len(games), 'all_mask_frames_rebuilt': frames, 'skill_by_episode_group': skill_by_episode,
    'baseline64_matches_prior_T3_1_games': True, 'approach_failure_behavior': approach_failures,
    'skill_by_episode_corner': {str(ep): {str(a): summarize_games([g for g in games if g['episode'] == ep and g['aid'] == a])
                                      for a in (0, 1)} for ep in (100, 150, 200)},
    'matched_budget_diagnostic': {name: {k: v for k, v in data.items() if k != 'games'} for name, data in diagnostic.items()},
    'matched_budget_activeWR_difference': comparison, 'full_test_unopened': True, 'decision': 'retain_original'}
(OUT / 'analysis_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'{frames} frames/all masks; skill outcomes/pairs; 200 diagnostic games/rates/corners: PASS', flush=True)
print(json.dumps({'baseline_group_success': {g: s['owned_kill_success'] for g,s in skill_by_episode['B'].items()},
    'approach_failures': approach_failures, 'paired_difference': comparison,
    'training_layouts': {a: v['layouts'] for a, v in training.items()}}, indent=2))
