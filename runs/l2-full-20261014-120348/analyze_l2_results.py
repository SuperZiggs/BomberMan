"""Verify the submitted L2.0 corpus; leave original audit files and checkpoints intact."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np
import torch

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(OUTPUT / 'source'))
from audit_l2_candidates import collect_candidates, plan_key, summarize_states
from audit_t2 import check_plan, digest, reachable_comparison, serial_plan
from bomberman_env import BombermanFFAEnv
from train_actor_critic import get_action_mask


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def usable(candidate):
    check = candidate['check']
    return (check['prefix_status'] == 'complete' and check['full_horizon_alive'] is True
            and not check['range_changed'])


def choose(candidates):
    if not candidates:
        return None
    score = max(tuple(c['score']) for c in candidates)
    tied = [c for c in candidates if tuple(c['score']) == score]
    plan = tied[0]['plan']
    return {'score': list(score), 'first_actions': sorted({c['plan']['path'][0] for c in tied}),
            'two_bomb': len(plan['bombs']) == 2, 'trapped': plan['trapped'],
            'pressure': plan['pressure'], 'candidate_index': tied[0]['candidate_index'],
            'tied_paths': len(tied)}


def without_timing(value):
    return {k: v for k, v in value.items() if k not in ('elapsed_ms', 'milliseconds')}


def main():
    torch.set_num_threads(1)
    cfg, saved = read_json(OUTPUT / 'config.json'), read_json(OUTPUT / 'summary.json')
    source = Path(cfg['input'])
    t2cfg = read_json(source / 'config.json')
    assert saved['status'] == 'complete' and saved['full_protocol'] and saved['hashes_unchanged']
    assert (cfg['boards'], cfg['states_per_game'], cfg['seconds']) == (10, 4, 900)
    assert cfg['source_checkpoint_sha256'] == t2cfg['sha256']
    with zipfile.ZipFile(str(OUTPUT) + '-send.zip') as archive:
        assert archive.testzip() is None
        for item in archive.infolist():
            if not item.is_dir():
                with archive.open(item) as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest(OUTPUT / item.filename)
    for name, expected in cfg['source_checkpoint_sha256'].items():
        path = ROOT / name if name.endswith('.py') else Path(t2cfg['checkpoints'][name])
        assert digest(path) == expected, name
        if name.endswith('.py'):
            assert digest(OUTPUT / 'source' / name) == expected, name
    for name, key in (('audit_l2_candidates.py', 'runner_sha256'), ('test_l2_candidates.py', 'tests_sha256')):
        assert digest(ROOT / name) == digest(OUTPUT / 'source' / name) == cfg[key]
    with zipfile.ZipFile(str(source) + '-send.zip') as archive:
        for name, expected in cfg['input_sha256'].items():
            assert digest(source / name) == expected
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
    states = list(rows(OUTPUT / 'candidates.jsonl.gz'))
    assert len(states) == saved['states_checked'] == 149 and all(s['complete'] for s in states)
    lookup = {(s['game_id'], s['tick']): s for s in states}
    assert len(lookup) == len(states)
    for s in states:
        candidates = s['candidates']
        assert s['scored_paths'] == s['checked_paths'] == len(candidates)
        assert s['teacher_stats']['plans'] <= 64
        good = [c for c in candidates if usable(c)]
        assert s['viable_paths'] == len(good)
        assert s['best_viable'] == choose(good)
        assert s['best_viable_two_bomb'] == choose([c for c in good if len(c['plan']['bombs']) == 2])
        for c in candidates:
            p = c['plan']
            assert s['mask'][p['path'][0]]
            live = [(b, a) for b, a in zip(p['before'], p['after']) if b > 0]
            assert live and p['pressure'] == max(1 - a / b for b, a in live) > 0
            assert p['trapped'] == sum(a == 0 for _, a in live)
            assert c['score'] == [p['trapped'], round(p['pressure'], 10), -len(p['bombs']),
                                  -sum(a in (1, 2, 3, 4) for a in p['path']), -len(p['path'])]
            assert [b['delay'] for b in p['bombs']] == [i for i, a in enumerate(p['path']) if a == 5]
        for k, top in s['top_k'].items():
            selected = choose([c for c in good if c['schedule_rank'] <= int(k)])
            assert top['selected'] == selected
            best = s['best_viable']
            assert top['same_best_score'] == bool(best and selected and best['score'] == selected['score'])
            assert top['same_first_actions'] == bool(best and selected and best['first_actions'] == selected['first_actions'])
    assert summarize_states(states) == saved['overall']
    assert {m: summarize_states([s for s in states if s['model'] == m]) for m in cfg['models']} == saved['by_model']
    print('ZIP/input/source/checkpoints, selections, scores and all summary fields: PASS', flush=True)

    targets = {(m, seed, aid) for m in cfg['models'] for seed in range(cfg['seed'], cfg['last_seed'] + 1)
               for aid in cfg['corners']}
    counts, coverage, reviewed, qualities = Counter(), {}, set(), []
    current, env, expected_tick, replay_frames, branch_count = None, None, 0, 0, 0
    for row in rows(source / 'trace.jsonl.gz'):
        key = row['model'], row['seed'], row['aid']
        if key not in targets:
            continue
        if current != key:
            assert current is None
            current, expected_tick = key, 0
            env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=row['seed'])
            env.reset()
        assert row['tick'] == expected_tick
        assert hashlib.sha256(pickle.dumps(env.__dict__)).hexdigest() == row['state_id']
        eligible = (row['tick'] in cfg['sample_ticks'] and row['first_actions']
                    and counts[key] < cfg['states_per_game'])
        assert bool(eligible) == ((row['game_id'], row['tick']) in lookup)
        if eligible:
            s = lookup[row['game_id'], row['tick']]
            assert s['state_id'] == row['state_id'] and s['mask'] == row['mask']
            assert s['old_first_actions'] == row['first_actions']
            before = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
            mask = get_action_mask(env, row['aid'])
            original, stats, candidates = collect_candidates(env, row['aid'], mask)
            assert without_timing(stats) == without_timing(s['teacher_stats'])
            groups = defaultdict(list)
            for c in candidates:
                groups[plan_key(c['plan'])[0]].append(c)
            assert len(groups) == s['scored_schedules']
            flat = [(rank, c) for rank, paths in enumerate(groups.values(), 1) for c in paths]
            assert len(flat) == len(s['candidates'])
            for index, ((rank, fresh), c) in enumerate(zip(flat, s['candidates'])):
                assert c['candidate_index'] == index and c['schedule_rank'] == rank
                assert c['plan'] == serial_plan(fresh['plan'], row['tick']) and c['score'] == list(fresh['score'])
                assert without_timing(check_plan(env, row['aid'], fresh['plan'], 'static', True)) == without_timing(c['check'])
                branch_count += 1
            original_serial = serial_plan(original, row['tick'])
            representative = next(c for c in s['candidates'] if c['plan'] == original_serial)
            assert s['old_representative_viable'] == usable(representative)
            assert s['old_best_any_tied_viable'] == any(usable(c) and c['score'] == list(candidates[0]['score']) for c in s['candidates'])
            rng = np.random.get_state()
            try:
                np.random.seed(s['stress_seed'])
                if s['best_viable']:
                    plan = flat[s['best_viable']['candidate_index']][1]['plan']
                    for kind in ('evader', 'rule'):
                        assert without_timing(check_plan(env, row['aid'], plan, kind, True)) == without_timing(s['selected_stress_checks'][kind])
                        branch_count += 1
            finally:
                np.random.set_state(rng)
            seen_schedules = set()
            for (_, fresh), c in zip(flat, s['candidates']):
                schedule = plan_key(fresh['plan'])[0]
                if usable(c) and len(schedule) == 2 and schedule not in seen_schedules:
                    seen_schedules.add(schedule)
                    quality = reachable_comparison(env, row['aid'], fresh['plan'])
                    q = quality['survivors']
                    genuine = any(b > 0 and a == 0 and f > 0 and t > 0
                                  for b, a, f, t in zip(q['baseline'], q['combo'], q['first_only'], q['second_only']))
                    assert not genuine and all(a > 0 for a in q['combo'])
                    qualities.append({'game_id': row['game_id'], 'tick': row['tick'],
                                      'candidate_index': c['candidate_index'], 'reachability': quality})
            assert pickle.dumps(env.__dict__) == before[0] and pickle.dumps(np.random.get_state()) == before[1]
            assert torch.equal(torch.get_rng_state(), before[2])
            counts[key] += 1
            reviewed.add((row['game_id'], row['tick']))
        env.step(row['actions'])
        expected_tick += 1
        replay_frames += 1
        if row['info']['episode_done']:
            coverage[key] = {'model': key[0], 'seed': key[1], 'aid': key[2],
                             'states': counts[key], 'replayed_complete': True}
            print(f'Rebuilt {row["game_id"]}: {expected_tick} frames, {counts[key]} sampled states', flush=True)
            current, env = None, None
            if set(coverage) == targets:
                break
    assert reviewed == set(lookup) and set(coverage) == targets
    assert [coverage[k] for k in sorted(coverage)] == saved['games_sampling']
    assert saved['games_replayed'] == saved['expected_games'] == len(targets) == 40
    selected = [s for s in states if s['best_viable']]
    rescued = [s for s in selected if not s['old_representative_viable']]
    diagnostics = {
        'viable_paths_by_bomb_count': dict(Counter(len(c['plan']['bombs']) for s in states for c in s['candidates'] if usable(c))),
        'trapped_scored_paths': sum(c['plan']['trapped'] > 0 for s in states for c in s['candidates']),
        'selected_two_stress': {kind: {
            'checks': sum(s['best_viable']['two_bomb'] for s in selected),
            'prefix_complete_and_survived': sum(s['best_viable']['two_bomb'] and
                s['selected_stress_checks'][kind]['prefix_status'] == 'complete' and
                s['selected_stress_checks'][kind]['full_horizon_alive'] is True for s in selected)
            } for kind in ('evader', 'rule')},
        'rescued_same_first_actions': sum(s['old_first_actions'] == s['best_viable']['first_actions'] for s in rescued),
        'rescued_no_first_action_overlap': sum(not set(s['old_first_actions']) & set(s['best_viable']['first_actions']) for s in rescued),
        'top_k_selected_two': {k: sum(bool(s['top_k'][k]['selected'] and s['top_k'][k]['selected']['two_bomb'])
                                    for s in states) for k in ('1', '4', '8')},
        'rule_known_suffix_deaths': [{'game_id': s['game_id'], 'tick': s['tick']}
            for s in selected if s['selected_stress_checks']['rule']['prefix_status'] == 'complete'
            and s['selected_stress_checks']['rule']['full_horizon_alive'] is False]}
    result = {'analysis_script_sha256': digest(Path(__file__)), 'all_zip_members_and_hashes_verified': True,
              'all_summary_fields_recomputed': True, 'games_rebuilt': len(targets), 'frames_rebuilt': replay_frames,
              'sampled_states_rebuilt': len(reviewed), 'candidate_and_stress_branches_recomputed': branch_count,
              'viable_two_bomb_schedule_checks': qualities, 'diagnostics': diagnostics,
              'overall': saved['overall'], 'by_model': saved['by_model'],
              'decision': 'retain_original_prioritize_T3_1_benchmark_defer_L2_training',
              'limits': ['Ten previously reviewed board seeds; frames/candidates are dependent observations.',
                        'Static mask pass uses physical escape suffix, without future mask enforcement in suffix.',
                        'No new training or model win-rate comparison; stress rollout uses prescribed prefix/escape.',
                        'Search covers only original scored candidates within <=64 schedules.']}
    (OUTPUT / 'analysis_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'frames': replay_frames, 'branches': branch_count, 'diagnostics': diagnostics}, indent=2))
    print('All saved game states/candidates/guards/stress and viable two-bomb reachability: PASS', flush=True)


if __name__ == '__main__':
    main()
