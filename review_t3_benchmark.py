"""Recompute a completed T3 benchmark from its snapshot and saved action tapes."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    output = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(output / 'source'))
    from benchmark_t3 import (make_env, reference_plan, certify, genuine_combo, summarize_games,
                              played_pairs, collect_candidates, serial_plan, get_action_mask, digest)
    torch.set_num_threads(1)
    cfg, saved = read(output / 'config.json'), read(output / 'summary.json')
    assert saved['status'] == 'complete' and saved['hashes_unchanged']
    assert not cfg['teacher_selects_cnn_action'] and not cfg['promotion_allowed']
    with zipfile.ZipFile(str(output) + '-send.zip') as archive:
        assert archive.testzip() is None
        for item in archive.infolist():
            if not item.is_dir():
                with archive.open(item) as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest(output / item.filename)
    for name, expected in cfg['source_sha256'].items():
        assert digest(output / 'source' / name) == expected
    for name, expected in cfg['checkpoint_sha256'].items():
        assert digest(Path(cfg['checkpoints'][name])) == expected
    cases = {c['id']: c for c in read(output / 'scenarios.json')}
    audits = {a['scenario_id']: a for a in read(output / 'certificates_teacher.json')}
    assert len(cases) == cfg['scenarios'] == saved['cases_audited'] and cases.keys() == audits.keys()
    fresh_teacher, proofs = {}, 0
    for cid, case in cases.items():
        env, aid = make_env(case, cfg['rollout_seeds'][0]), case['aid']
        audit = audits[cid]
        if case['expected'] != 'illegal_capacity':
            plan, quality = reference_plan(env, case)
            assert serial_plan(plan, 0) == audit['reference']
            assert quality == audit['reference_quality']
            proof = certify(env, aid, plan)
            assert {k: v for k, v in proof.items() if k != 'milliseconds'} == {
                k: v for k, v in audit['reference_certificate'].items() if k != 'milliseconds'}
            proofs += 1
        else:
            try:
                reference_plan(env, case)
            except ValueError:
                assert audit['reference_status'] == 'illegal_capacity'
            else:
                raise AssertionError('Ammo-one overlapping schedule accepted')
        old, stats, candidates = collect_candidates(env, aid, get_action_mask(env, aid))
        assert stats['plans'] <= 64 and len(candidates) == audit['scored_paths']
        assert {k: v for k, v in stats.items() if k != 'milliseconds'} == {
            k: v for k, v in audit['teacher_stats'].items() if k != 'milliseconds'}
        if old:
            assert serial_plan(old, 0) == audit['old_best']['plan']
        else:
            assert audit['old_best'] is None
        fresh_teacher[cid] = {'first_actions': old['first_actions'] if old else [],
                              'plans': stats['plans'], 'scored_paths': len(candidates)}
    print(f'ZIP, hashes, {len(cases)} teacher states and {proofs} legal reference certificates: PASS', flush=True)
    with gzip.open(output / 'games.jsonl.gz', 'rt', encoding='utf-8') as stream:
        games = [json.loads(line) for line in stream]
    expected = {(cid, model, kind, seed) for cid in cases for model in cfg['checkpoints']
                for kind in cfg['opponents'] for seed in cfg['rollout_seeds']}
    actual = [(g['scenario_id'], g['model'], g['opponent'], g['seed']) for g in games]
    assert len(games) == saved['games_completed'] == saved['expected_games'] == cfg['expected_games']
    assert len(set(actual)) == len(actual) and set(actual) == expected
    mask_groups, frames, sample_masks = set(), 0, 0
    for index, game in enumerate(games):
        case, aid = cases[game['scenario_id']], game['aid']
        env = make_env(case, game['seed'])
        placements, states, suicide, failed_moves = [], [], False, 0
        group = case['group'], game['opponent']
        check_masks = group not in mask_groups
        for tick, row in enumerate(game['tape']):
            assert not env._episode_done and row['tick'] == tick
            actions = row['actions']
            assert row['mask'][actions[aid]]
            if check_masks:
                assert row['mask'] == list(map(bool, get_action_mask(env, aid)))
                sample_masks += 1
            agent = env.agents[aid]
            before = agent.row, agent.col
            branch = deepcopy(env) if actions[aid] == 5 else None
            _, _, _, info = env.step(actions)
            if aid in info['bombs_placed']:
                states.append(branch)
                placements.append({'tick': tick, 'position': list(before), 'range': agent.blast_range})
            suicide |= aid in info['self_blast_deaths'] or (aid, aid) in info['kills']
            if actions[aid] in (1, 2, 3, 4):
                delta = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[actions[aid]]
                failed_moves += (agent.row, agent.col) != (before[0] + delta[0], before[1] + delta[1])
            assert row['bombs_placed'] == info['bombs_placed'] and row['kills'] == [list(k) for k in info['kills']]
            assert row['self_blast_deaths'] == info['self_blast_deaths'] and row['alive'] == info['alive']
            assert row['positions'] == [[a.row, a.col] for a in env.agents]
            frames += 1
        mask_groups.add(group)
        agent, enemy = env.agents[aid], env.agents[1 - aid]
        result = {'owned_kill_success': bool(agent.alive and not enemy.alive and agent.kills > 0),
                  'suicide': bool(suicide), 'kills': agent.kills, 'learner_alive': agent.alive,
                  'passive_win': agent.alive and not enemy.alive and agent.kills == 0,
                  'actual_draw': bool(env._episode_done and sum(a.alive for a in env.agents) != 1),
                  'skill_horizon_truncated': not env._episode_done, 'steps': env.step_count,
                  'placements': placements, 'two_or_more_placements': len(placements) >= 2,
                  'failed_moves': int(failed_moves)}
        assert all(game[k] == value for k, value in result.items())
        pairs = played_pairs(states, placements, aid)
        assert game['pairs'] == pairs
        assert game['has_genuine_combo_pair'] == any(p['genuine_combo'] for p in pairs)
        if (index + 1) % 240 == 0:
            print(f'Rebuilt {index + 1}/{len(games)} games', flush=True)
    groups = sorted({c['group'] for c in cases.values()})
    by_model = {model: {group: {kind: summarize_games([g for g in games if
        (g['model'], g['group'], g['opponent']) == (model, group, kind)]) for kind in cfg['opponents']}
        for group in groups} for model in cfg['checkpoints']}
    by_corner = {model: {str(aid): summarize_games([g for g in games if g['model'] == model and g['aid'] == aid])
                        for aid in (0, 1)} for model in cfg['checkpoints']}
    assert by_model == saved['by_model'] and by_corner == saved['by_corner']
    combo_metrics = {}
    for split in cfg['splits']:
        combo_metrics[split] = {}
        for kind in cfg['opponents']:
            selected = [g for g in games if g['split'] == split and g['expected'] == 'combo' and g['opponent'] == kind]
            def prefix_matches(g):
                path = audits[g['scenario_id']]['reference']['path']
                return [r['actions'][g['aid']] for r in g['tape'][:len(path)]] == path
            combo_metrics[split][kind] = {**summarize_games(selected),
                'reference_prefix_exact': sum(prefix_matches(g) for g in selected),
                'teacher_initial_first_action_agreement': sum(g['tape'][0]['actions'][g['aid']] in
                    fresh_teacher[g['scenario_id']]['first_actions'] for g in selected)}
    result = {'zip_and_hashes_verified': True, 'all_game_outcomes_and_pairs_recomputed': True,
              'reference_certificates_recomputed': proofs, 'teacher_states_recomputed': len(cases),
              'games_rebuilt': len(games), 'frames_rebuilt': frames,
              'sample_mask_games': len(mask_groups), 'sample_mask_frames': sample_masks,
              'teacher_first_actions': fresh_teacher, 'combo_by_split_opponent': combo_metrics,
              'combo_by_family_opponent': {family: {kind: summarize_games([g for g in games if
                  g['family'] == family and g['expected'] == 'combo' and g['opponent'] == kind])
                  for kind in cfg['opponents']} for family in cfg['family_splits'] if family in {c['family'] for c in cases.values()}},
              'by_model': by_model, 'by_corner': by_corner,
              'reference_outcomes': dict(Counter(a['reference_status'] for a in audits.values())),
              'decision': 'review_required_keep_original'}
    (output / 'analysis_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'All outcomes/placements/pair forecasts/summary, {frames} frames: PASS', flush=True)
    print(json.dumps({'combo_by_split_opponent': combo_metrics, 'by_corner': by_corner}, indent=2))


if __name__ == '__main__':
    main()
