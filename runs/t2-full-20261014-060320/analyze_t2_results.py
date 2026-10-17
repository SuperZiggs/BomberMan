"""Recompute this saved T2 audit; no training, new matches or checkpoint selection."""
from collections import Counter, defaultdict
from copy import deepcopy
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
from audit_t2 import diagnostic_row, summarize
from bomberman_env import BombermanFFAEnv, ACTION_BOMB
from pilot_two_bomb import paired_active_bootstrap
from rule_based_bot import _can_bomb_and_escape, _bomb_escape_pressure
from train_actor_critic import get_action_mask


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(n, d):
    return {'numerator': n, 'denominator': d, 'percent': 100 * n / d if d else None}


def verify_archive(config):
    with zipfile.ZipFile(str(OUTPUT) + '-send.zip') as archive:
        assert archive.testzip() is None
        for item in archive.infolist():
            if item.is_dir():
                continue
            with archive.open(item) as stream:
                archived = hashlib.file_digest(stream, 'sha256').hexdigest()
            with (OUTPUT / item.filename).open('rb') as stream:
                assert archived == hashlib.file_digest(stream, 'sha256').hexdigest(), item.filename
    source_checks, checkpoint_checks = {}, {}
    for name, expected in config['sha256'].items():
        if name.endswith('.py'):
            source_checks[name] = sha(OUTPUT / 'source' / name) == expected
        else:
            checkpoint_checks[name] = sha(Path(config['checkpoints'][name])) == expected
    assert all(source_checks.values()) and all(checkpoint_checks.values())
    return {'original_zip_crc_and_all_members_match': True,
            'snapshot_source_hashes': source_checks, 'current_checkpoint_hashes': checkpoint_checks}


def review_case(case):
    """Rebuild exact states from the saved real-game tape, then replay selected clone tapes."""
    aid, seed = case['result']['aid'], case['result']['seed']
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed)
    env.reset()
    examples, first_missing, missing_mask = [], None, None
    starts, churn_unknown, kills = [], [], []
    for row in case['timeline']:
        assert sha_state(env) == row['state_id'], (case['game_id'], row['tick'])
        for c in row['checks']:
            if c['opponent'] != 'static' or not c['enforce_mask'] or c['prefix_status'] != 'mask_blocked':
                continue
            paired = next(q for q in row['checks'] if q['opponent'] == 'static' and not q['enforce_mask'])
            if paired['prefix_status'] != 'complete':
                continue
            stage = 'second_bomb' if c['blocked_action'] == ACTION_BOMB and c['first_bomb'] else (
                    'first_bomb' if c['blocked_action'] == ACTION_BOMB else 'wait_or_move')
            if any(q['stage'] == stage for q in examples):
                continue
            clone = deepcopy(env)
            for actions in c['action_tape']:
                clone.step(actions)
            mask = get_action_mask(clone, aid)
            assert list(map(bool, mask)) == c['mask'] and not mask[c['blocked_action']]
            agent = clone.agents[aid]
            burning = bool(clone._forecast_blasts()[0][:, agent.row, agent.col].any())
            entry = {'stage': stage, 'plan_tick': row['tick'], 'blocked_delay': c['blocked_tick'],
                     'blocked_absolute_tick': row['tick'] + c['blocked_tick'], 'action': c['blocked_action'],
                     'position': [agent.row, agent.col], 'ammo': agent.ammo, 'active_bombs': agent.bombs_placed,
                     'mask': mask, 'known_fire_on_current_tile': burning, 'plan_pressure': c['plan']['pressure'],
                     'plan_bombs': c['plan']['bombs'], 'physical_horizon_alive': paired['full_horizon_alive']}
            if c['blocked_action'] == ACTION_BOMB:
                blast = clone._blast_cells(agent.row, agent.col, agent.blast_range)
                entry.update(single_bomb_pressure=_bomb_escape_pressure(clone, aid, agent.row, agent.col),
                             robust_escape=_can_bomb_and_escape(clone, aid, agent.row, agent.col, None),
                             wood_on_blast=any(clone.grid[pos] == 2 for pos in blast))
            examples.append(entry)
        if row['completed_plan']:
            p = row['completed_plan']
            starts.append(p)
            if p['two_bomb'] and p['first_bomb'] and not p['second_bomb'] and first_missing is None:
                first_missing = p
            if p['churn_reasons'].get('unexplained_by_logged_events'):
                churn_unknown.append(p['start_tick'])
        if row['info']['kills']:
            kills.append({'tick': row['tick'], 'kills': row['info']['kills']})
        env.step(row['actions'])
        assert [[a.row, a.col, a.alive] for a in env.agents] == [
            [a['row'], a['col'], a['alive']] for a in row['after']['agents']]
    if first_missing:
        start = case['timeline'][first_missing['start_tick']]
        plan = next(p for p in start['tied_plans'] if p['path'][0] == start['actions'][aid])
        target_tick = start['tick'] + plan['bombs'][1]['delay']
        if target_tick < len(case['timeline']):
            target = case['timeline'][target_tick]
            missing_mask = {'plan_start': start['tick'], 'first_bomb_correct': True,
                            'second_expected_tick': target_tick, 'expected_position': [plan['bombs'][1]['row'], plan['bombs'][1]['col']],
                            'actual_position': [target['before']['agents'][aid]['row'], target['before']['agents'][aid]['col']],
                            'actual_action': target['actions'][aid], 'actual_mask': target['mask'],
                            'placed_bomb': aid in target['info']['bombs_placed']}
    return {'game_id': case['game_id'], 'result': case['result'], 'frames_rebuilt': len(case['timeline']),
            'plan_windows': len(starts), 'correct_two_bomb_placements': sum(p['two_bomb'] and p['complete_placements'] for p in starts),
            'static_mask_examples': examples, 'first_missing_second_example': missing_mask,
            'churn_unknown_start_ticks': churn_unknown[:3], 'kill_events': kills}


def sha_state(env):
    return hashlib.sha256(pickle.dumps(env.__dict__)).hexdigest()


def main():
    torch.set_num_threads(1)
    config, saved, status = (read_json(OUTPUT / name) for name in ('config.json', 'summary.json', 'status.json'))
    assert status['status'] == saved['status'] == 'complete' and saved['full_protocol']
    provenance = verify_archive(config)
    print('Archive, snapshot and checkpoint hashes: PASS', flush=True)
    games = list(rows(OUTPUT / 'games.jsonl'))
    starts = list(rows(OUTPUT / 'plan_starts.jsonl'))
    checks = list(rows(OUTPUT / 'plan_checks.jsonl.gz'))
    by_game = {g['game_id']: g for g in games}
    assert len(by_game) == len(games) == 400
    expected = {(s, a) for s in range(28000, 28100) for a in (0, 1)}
    for model in saved['models']:
        assert {(g['seed'], g['aid']) for g in games if g['model'] == model} == expected
    by_state = defaultdict(list)
    for c in checks:
        by_state[(c['game_id'], c['tick'])].append(c)
    cases = read_json(OUTPUT / 'replay_cases.json')
    assert len({c['game_id'] for c in cases}) == len(cases) == 10
    rare = []
    for c in checks:
        if c['enforce_mask'] or c['opponent'] != 'static' or len(c['plan']['bombs']) != 2:
            continue
        q = c['reachability']['survivors']
        if (isinstance(q.get('first_only'), list) and isinstance(q.get('second_only'), list)
                and any(b > 0 and a == 0 and f > 0 and s > 0 for b, a, f, s in
                        zip(q['baseline'], q['combo'], q['first_only'], q['second_only']))):
            rare.append(c)
    positive = next(p for p in starts if p['model'] == 'two_bomb_final'
                    and p['two_bomb'] and p['complete_placements'])
    extra_ids = {c['game_id'] for c in rare} | {positive['game_id']}
    extra_tapes = {gid: [] for gid in extra_ids - {c['game_id'] for c in cases}}
    physical = {(c['game_id'], c['tick'], c['opponent']): c for c in checks if not c['enforce_mask']}
    gaps = defaultdict(Counter)
    for c in checks:
        if not c['enforce_mask']:
            continue
        a = physical[c['game_id'], c['tick'], c['opponent']]
        v = gaps[c['model'], c['opponent']]
        v['sampled_states'] += 1
        if c['prefix_status'] == 'mask_blocked':
            assert c['blocked_tick'] > 0, 'First label must already pass the same mask'
        if a['prefix_status'] == 'complete':
            v['physical_prefix_complete'] += 1
            v['masked_prefix_complete'] += c['prefix_status'] == 'complete'
            if c['prefix_status'] == 'complete':
                v['masked_two_bomb_complete'] += len(c['plan']['bombs']) == 2
                v['masked_pressure_at_least_0_5'] += c['plan']['pressure'] >= 0.5
                v['masked_trap'] += c['plan']['trapped'] > 0
            if c['prefix_status'] == 'mask_blocked':
                v['mask_blocked'] += 1
                v['blocked_BOMB'] += c['blocked_action'] == ACTION_BOMB
                v['blocked_second_BOMB'] += c['blocked_action'] == ACTION_BOMB and c['first_bomb']
                v['blocked_first_BOMB'] += c['blocked_action'] == ACTION_BOMB and not c['first_bomb']
                v['blocked_STOP'] += c['blocked_action'] == 0
                v['physical_full_horizon_alive'] += a['full_horizon_alive'] is True
    diagnostics, subgroups = defaultdict(list), defaultdict(Counter)
    recomputed_games = {}
    last_tick, n_trace, sampled_states = {}, 0, set()
    completed = {}
    for row in rows(OUTPUT / 'trace.jsonl.gz'):
        model, aid, game = row['model'], row['aid'], row['game_id']
        if game in extra_tapes:
            extra_tapes[game].append(row)
        assert row['tick'] == last_tick.get(game, -1) + 1
        last_tick[game] = row['tick']
        assert row['before']['step'] == row['tick'] and row['after']['step'] == row['tick'] + 1
        assert row['stats']['plans'] <= 64 and row['mask'][row['actions'][aid]]
        assert all(row['mask'][a] for a in row['first_actions'])
        assert row['stats']['accepted'] == bool(row['first_actions'])
        assert row['checks'] == by_state.get((game, row['tick']), [])
        if row['checks']:
            assert len(row['checks']) == 6 and row['tick'] % 10 == 0
            sampled_states.add((game, row['tick']))
        d = diagnostic_row(row)
        diagnostics[model].append(d)
        if d['labels']:
            for group in ('all', *(['mask_more_than_one'] if d['mask_support'] > 1 else []),
                          *(['informative_K_lt_M'] if len(d['labels']) < d['mask_support'] else [])):
                v = subgroups[model, group]
                v['labels'] += 1
                v['agreements'] += d['agreement']
                v['mass_sum'] += d['mass']
                v['chance_sum'] += d['uniform_chance']
        g = recomputed_games.setdefault(game, {'seed': row['seed'], 'aid': aid, 'bombs': 0, 'items': 0, 'suicide': 0})
        g['bombs'] += aid in row['info']['bombs_placed']
        g['items'] += len(row['info']['items_collected'].get(str(aid), []))
        g['suicide'] |= int(aid in row['info'].get('self_blast_deaths', []) or [aid, aid] in row['info']['kills'])
        if row['completed_plan']:
            p = row['completed_plan']
            assert p['plan_id'] not in completed
            completed[p['plan_id']] = p
        if row['info']['episode_done']:
            agents = row['after']['agents']
            win = int(agents[aid]['alive'] and not agents[1 - aid]['alive'])
            g.update(win=win, active_win=int(win and agents[aid]['kills'] > 0),
                     kills=agents[aid]['kills'], draw=int(sum(a['alive'] for a in agents) != 1), steps=row['after']['step'])
            assert g == {k: v for k, v in by_game[game].items() if k not in ('model', 'game_id')}, game
        n_trace += 1
    assert len(completed) == len(starts) and all(completed[p['plan_id']] == p for p in starts)
    assert n_trace == 158293 and len(recomputed_games) == 400
    assert len(sampled_states) * 6 == len(checks)
    windows = defaultdict(list)
    for p in starts:
        windows[p['game_id']].append(p)
    for group in windows.values():
        group.sort(key=lambda p: p['start_tick'])
        assert all(a['end_tick'] <= b['start_tick'] for a, b in zip(group, group[1:]))
    print(f'Trace: {n_trace} frames; games, windows, masks and six paired branches: PASS', flush=True)
    verified = {}
    for model, entry in saved['models'].items():
        gs, ps, cs = ([r for r in group if r['model'] == model] for group in (games, starts, checks))
        result = {'overall': summarize(gs, diagnostics[model], ps, cs),
                  'by_corner': {str(a): summarize([g for g in gs if g['aid'] == a],
                    [d for d in diagnostics[model] if d['aid'] == a], [p for p in ps if p['aid'] == a],
                    [c for c in cs if c['aid'] == a]) for a in (0, 1)}}
        assert json.loads(json.dumps(result)) == entry, model
        verified[model] = result
    difference = paired_active_bootstrap({'games': [g for g in games if g['model'] == 'two_bomb_final']},
                                        {'games': [g for g in games if g['model'] == 'original']})
    assert difference == saved['final_minus_original_activewr']
    signal = {}
    for (model, group), v in subgroups.items():
        signal.setdefault(model, {})[group] = {**pct(v['agreements'], v['labels']),
            'mean_mass': v['mass_sum'] / v['labels'], 'mean_uniform_K_over_M': v['chance_sum'] / v['labels']}
    gap_results = {}
    for (model, kind), v in gaps.items():
        gap_results.setdefault(model, {})[kind] = {**v,
            'blocked_given_physical_complete': pct(v['mask_blocked'], v['physical_prefix_complete'])}
    for gid, tape in sorted(extra_tapes.items()):
        cases.append({'game_id': gid, 'result': by_game[gid], 'timeline': tape})
    reviews = [review_case(case) for case in cases]
    (OUTPUT / 'T2_replay_review.json').write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding='utf-8')
    rare_findings = []
    for c in rare:
        timeline = next(case['timeline'] for case in cases if case['game_id'] == c['game_id'])
        row = timeline[c['tick']]
        branches = {f"{q['opponent']}_{q['enforce_mask']}": {
            k: q[k] for k in ('prefix_status', 'full_horizon_alive', 'blocked_action', 'blocked_tick')}
            for q in row['checks']}
        rare_findings.append({'game_id': c['game_id'], 'tick': c['tick'], 'plan': c['plan'],
            'reachability': c['reachability'], 'actual_first_action': row['actions'][c['aid']],
            'first_actions': row['first_actions'], 'masked_probabilities': row['probabilities'],
            'branches': branches, 'actual_prefix_actions': [r['actions'][c['aid']]
                for r in timeline[c['tick']:c['tick'] + len(c['plan']['path'])]]})
    result = {'analysis_script_sha256': sha(Path(__file__)), 'provenance': provenance,
              'trace_rows': n_trace, 'games': len(games), 'plan_windows': len(starts),
              'sampled_states': len(sampled_states), 'clone_branches': len(checks), 'all_summary_fields_recomputed': True,
              'difference_final_minus_original': difference, 'agreement_signal': signal, 'future_mask_gaps': gap_results,
              'original_replay_cases_reviewed': 10, 'additional_control_games': sorted(extra_tapes),
              'replay_games_rebuilt': len(reviews), 'replay_frames_rebuilt': sum(c['frames_rebuilt'] for c in reviews),
              'rare_genuine_combo_forecasts': rare_findings, 'positive_placement_control': positive,
              'models': verified, 'decision': 'retain_original_prioritize_L2_prefix_mask_guard',
              'limitations': ['Exploratory audit; board seed is the sampling unit, not frame or branch.',
                  'Fixed final policies; these are not the historical training trajectories.',
                  'Single canonical tied path; exact-placement completion is a descriptive measure.',
                  'Post-terminal physics and unknown survival remain separate from actual CNN game outcome.',
                  'Churn events are proxies, not evidence of causation; no optimizer diagnosis from agreement alone.']}
    (OUTPUT / 'analysis_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'All summary fields, paired CI and {len(reviews)} replay tapes: PASS', flush=True)
    print(json.dumps({'difference': difference, 'agreement_signal': signal, 'future_mask_gaps': gap_results}, indent=2))


if __name__ == '__main__':
    main()
