"""L2.0 offline: compare all scored candidates with future-mask guards on saved T2 states."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
from time import perf_counter
import zipfile

import numpy as np
import torch

from audit_t2 import (AUDIT_TEACHER, check_plan, digest, distribution, dump,
                      reachable_comparison, serial_plan)
from bomberman_env import BombermanFFAEnv
from train_actor_critic import get_action_mask

ROOT = Path(__file__).resolve().parent
T2 = ROOT / 'runs/t2-full-20261014-060320'
SAMPLE_TICKS = (50, 150, 250, 350, 450)
TOP_K = (1, 4, 8)


def plan_key(plan):
    schedule = tuple((d, b.row, b.col, b.owner, b.timer, b.blast_range) for d, b in plan['bombs'])
    return schedule, tuple(plan['path'])


def collect_candidates(env, aid, mask):
    candidates = []
    def record(score, plan):
        candidates.append({'score': score, 'plan': plan})
    original, stats = AUDIT_TEACHER(env, aid, mask, audit_record=record)
    candidates.sort(key=lambda c: c['score'], reverse=True)
    assert stats['plans'] <= 64
    assert len({plan_key(c['plan'])[0] for c in candidates}) <= stats['plans']
    assert len({plan_key(c['plan']) for c in candidates}) == len(candidates)
    if original:
        assert serial_plan(original, 0) == serial_plan(candidates[0]['plan'], 0)
        labels = sorted({c['plan']['path'][0] for c in candidates if c['score'] == candidates[0]['score']})
        assert labels == original['first_actions'] and all(mask[a] for a in labels)
    else:
        assert not candidates
    return original, stats, candidates


def viable(check):
    return (check['prefix_status'] == 'complete' and check['full_horizon_alive'] is True
            and not check['range_changed'])


def selection(candidates):
    if not candidates:
        return None
    score = max(c['score'] for c in candidates)
    tied = [c for c in candidates if c['score'] == score]
    return {'score': list(score), 'first_actions': sorted({c['plan']['path'][0] for c in tied}),
            'two_bomb': len(tied[0]['plan']['bombs']) == 2,
            'trapped': tied[0]['plan']['trapped'], 'pressure': tied[0]['plan']['pressure'],
            'candidate_index': tied[0]['candidate_index'], 'tied_paths': len(tied)}


def inspect_state(env, aid, tick, deadline, stress_seed=0):
    fingerprint = hashlib.sha256(pickle.dumps(env.__dict__)).hexdigest()
    np_state, torch_state = pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
    started = perf_counter()
    mask = get_action_mask(env, aid)
    original, stats, candidates = collect_candidates(env, aid, mask)
    teacher_ms = (perf_counter() - started) * 1000
    # Group by schedule so top-K keeps every scored path for each included schedule.
    schedules = {}
    for candidate in candidates:
        schedules.setdefault(plan_key(candidate['plan'])[0], []).append(candidate)
    evaluated, schedule_ms, complete = [], [], True
    guard_start = perf_counter()
    for rank, paths in enumerate(schedules.values(), 1):
        for candidate in paths:
            if perf_counter() >= deadline:
                complete = False
                break
            check = check_plan(env, aid, candidate['plan'], 'static', True)
            candidate.update(candidate_index=len(evaluated), schedule_rank=rank, check=check)
            evaluated.append(candidate)
        if not complete:
            break
        schedule_ms.append((perf_counter() - guard_start) * 1000)
    guard_ms = (perf_counter() - guard_start) * 1000
    accepted = [c for c in evaluated if viable(c['check'])]
    best = selection(accepted) if complete else None
    best_two = selection([c for c in accepted if len(c['plan']['bombs']) == 2]) if complete else None
    old_score = candidates[0]['score'] if candidates else None
    old_key = plan_key(original) if original else None
    old_representative = next((c for c in evaluated if plan_key(c['plan']) == old_key), None)
    top = {}
    if complete:
        for k in TOP_K:
            limited = [c for c in accepted if c['schedule_rank'] <= k]
            chosen = selection(limited)
            top[str(k)] = {'selected': chosen,
                          'same_best_score': bool(best and chosen and best['score'] == chosen['score']),
                          'same_first_actions': bool(best and chosen and best['first_actions'] == chosen['first_actions']),
                          'guard_ms': schedule_ms[min(k, len(schedule_ms)) - 1] if schedule_ms else 0.0}
    genuine, qualities, quality_records = [], {}, []
    for c in accepted:
        if len(c['plan']['bombs']) != 2 or not c['plan']['trapped']:
            continue
        schedule = plan_key(c['plan'])[0]
        if schedule not in qualities:
            qualities[schedule] = reachable_comparison(env, aid, c['plan'])
            quality_records.append({'candidate_index': c['candidate_index'], 'reachability': qualities[schedule]})
        q = qualities[schedule]['survivors']
        if (isinstance(q['first_only'], list) and isinstance(q['second_only'], list)
                and any(b > 0 and a == 0 and f > 0 and s > 0
                        for b, a, f, s in zip(q['baseline'], q['combo'], q['first_only'], q['second_only']))):
            genuine.append(c['candidate_index'])
    stress = {}
    if complete and best:
        rng_before = np.random.get_state()
        try:
            np.random.seed(stress_seed)
            selected = evaluated[best['candidate_index']]['plan']
            for kind in ('evader', 'rule'):
                if perf_counter() >= deadline:
                    break
                stress[kind] = check_plan(env, aid, selected, kind, True)
        finally:
            np.random.set_state(rng_before)
    assert hashlib.sha256(pickle.dumps(env.__dict__)).hexdigest() == fingerprint
    assert pickle.dumps(np.random.get_state()) == np_state and torch.equal(torch_state, torch.get_rng_state())
    return {'tick': tick, 'state_id': fingerprint, 'mask': list(map(bool, mask)), 'complete': complete,
            'teacher_stats': stats, 'scored_schedules': len(schedules), 'scored_paths': len(candidates),
            'checked_paths': len(evaluated), 'viable_paths': len(accepted),
            'old_representative_viable': bool(old_representative and viable(old_representative['check'])),
            'old_first_actions': original['first_actions'] if original else [],
            'old_best_any_tied_viable': any(c['score'] == old_score for c in accepted),
            'best_viable': best, 'best_viable_two_bomb': best_two, 'top_k': top,
            'genuine_combo_candidate_indices': genuine,
            'trap_quality_checks': quality_records, 'stress_seed': stress_seed, 'selected_stress_checks': stress,
            'teacher_ms': teacher_ms, 'guard_ms': guard_ms, 'state_total_ms': (perf_counter() - started) * 1000,
            'candidates': [{'candidate_index': c['candidate_index'], 'schedule_rank': c['schedule_rank'],
                'score': list(c['score']), 'plan': serial_plan(c['plan'], tick), 'check': c['check']}
                for c in evaluated]}


def fraction(n, d):
    return {'numerator': n, 'denominator': d, 'percent': 100 * n / d if d else None}


def summarize_states(states):
    completed = [s for s in states if s['complete']]
    with_viable = [s for s in completed if s['best_viable']]
    original_invalid = [s for s in completed if not s['old_representative_viable']]
    return {'states': len(states), 'fully_checked_states': len(completed),
        'old_representative_viable': fraction(sum(s['old_representative_viable'] for s in completed), len(completed)),
        'old_best_any_tied_viable': fraction(sum(s['old_best_any_tied_viable'] for s in completed), len(completed)),
        'any_viable_candidate': fraction(len(with_viable), len(completed)),
        'rescued_original_invalid': fraction(sum(s['best_viable'] is not None for s in original_invalid), len(original_invalid)),
        'any_viable_two_bomb': fraction(sum(s['best_viable_two_bomb'] is not None for s in completed), len(completed)),
        'selected_best_is_two_bomb': fraction(sum(s['best_viable']['two_bomb'] for s in with_viable), len(completed)),
        'any_genuine_combo_trap': fraction(sum(bool(s['genuine_combo_candidate_indices']) for s in completed), len(completed)),
        'scored_paths': sum(s['scored_paths'] for s in states), 'checked_paths': sum(s['checked_paths'] for s in states),
        'guard_outcomes': dict(Counter(c['check']['prefix_status'] for s in states for c in s['candidates'])),
        'range_changed_paths': sum(c['check']['range_changed'] for s in states for c in s['candidates']),
        'teacher_ms': distribution([s['teacher_ms'] for s in completed]),
        'guard_ms': distribution([s['guard_ms'] for s in completed]),
        'state_total_ms': distribution([s['state_total_ms'] for s in completed]),
        'selected_pressure': distribution([s['best_viable']['pressure'] for s in with_viable]),
        'selected_stress': {kind: {
            'checks': sum(kind in s['selected_stress_checks'] for s in with_viable),
            'eligible_states': len(with_viable),
            'prefix_pass': fraction(sum(s['selected_stress_checks'][kind]['prefix_status'] == 'complete'
                for s in with_viable if kind in s['selected_stress_checks']),
                sum(kind in s['selected_stress_checks'] for s in with_viable)),
            'full_horizon_alive': fraction(sum(s['selected_stress_checks'][kind]['full_horizon_alive'] is True
                for s in with_viable if kind in s['selected_stress_checks']),
                sum(s['selected_stress_checks'][kind]['full_horizon_alive'] is not None
                    for s in with_viable if kind in s['selected_stress_checks']))} for kind in ('evader', 'rule')},
        'top_k': {str(k): {
            'any_viable': fraction(sum(s['top_k'][str(k)]['selected'] is not None for s in completed), len(completed)),
            'best_score_recall': fraction(sum(s['top_k'][str(k)]['same_best_score'] for s in with_viable), len(with_viable)),
            'first_actions_exact_given_viable': fraction(sum(s['top_k'][str(k)]['same_best_score']
                and s['top_k'][str(k)]['same_first_actions'] for s in with_viable), len(with_viable)),
            'guard_ms': distribution([s['top_k'][str(k)]['guard_ms'] for s in completed])} for k in TOP_K}}


def write_report(output, summary):
    lines = ['# L2.0 candidate audit', '', f"Status: {summary['status']}; full_protocol={summary['full_protocol']}.", '',
             '| Model | States checked | Old representative pass | Any candidate pass | Any two-bomb pass | Chosen best two-bomb | Genuine combo trap |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for model, entry in summary['by_model'].items():
        def value(key):
            r = entry[key]
            return f"{r['numerator']}/{r['denominator']}"
        lines.append(f"| {model} | {entry['fully_checked_states']} | {value('old_representative_viable')} | "
            f"{value('any_viable_candidate')} | {value('any_viable_two_bomb')} | "
            f"{value('selected_best_is_two_bomb')} | {value('any_genuine_combo_trap')} |")
    lines += ['', '## Dùng để quyết định', '',
        '- Nếu alternatives khả thi/hữu ích đủ đa dạng: cân nhắc L2.1 chọn trong các candidate đã qua guard; xem coverage và chi phí top-K trước.',
        '- Nếu có nhiều candidate nhưng chọn best vẫn là một bom, chưa thể gọi guard là cải tiến phối hợp hai bom.',
        '- Nếu combo qua mask/trap vẫn rất hiếm: ưu tiên T3.1 benchmark tình huống trước train trên map ngẫu nhiên dài.',
        '- K=1/4/8 là số schedules, mỗi schedule giữ tất cả scored paths. Best-score recall chưa bảo đảm first-actions exact; xem cả hai.',
        '- Status incomplete hoặc corpus thiếu thì giữ kết quả để chẩn đoán, chưa kết luận toàn bộ corpus.', '',
        '## Phạm vi', '', *['- ' + x for x in summary['limitations']]]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=T2)
    parser.add_argument('--boards', type=int, default=10)
    parser.add_argument('--states-per-game', type=int, default=4)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.boards <= 100 or not 1 <= args.states_per_game <= 4 or args.seconds < 1:
        parser.error('boards 1..100; states-per-game 1..4; seconds positive')
    torch.set_num_threads(1)
    source = args.input.resolve()
    cfg = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    status = json.loads((source / 'status.json').read_text(encoding='utf-8'))
    assert status['status'] == 'complete' and cfg['full_protocol'] and args.boards <= cfg['boards_per_model']
    assert (cfg['board_size'], cfg['num_agents'], cfg['max_steps']) == (9, 2, 500)
    inputs = {'config.json': digest(source / 'config.json'), 'trace.jsonl.gz': digest(source / 'trace.jsonl.gz')}
    with zipfile.ZipFile(str(source) + '-send.zip') as archive:
        for name, expected in inputs.items():
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
    hashes = {name: digest(ROOT / name) if name.endswith('.py') else digest(Path(cfg['checkpoints'][name]))
              for name in cfg['sha256']}
    assert hashes == cfg['sha256'], 'Current sources/checkpoints differ from the T2 snapshot'
    output = (args.output or ROOT / ('runs/l2-candidates-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    if Path(str(output) + '-send.zip').exists():
        raise FileExistsError('Send ZIP already exists; choose new output')
    output.mkdir(parents=True, exist_ok=False)
    started, states, game_counts, seen_games = perf_counter(), [], Counter(), set()
    deadline = started + args.seconds
    models = ('original', 'two_bomb_final')
    targets = {(model, seed, aid) for model in models
               for seed in range(cfg['seed'], cfg['seed'] + args.boards) for aid in (0, 1)}
    config = {'started_utc': datetime.now(timezone.utc).isoformat(), 'input': str(source), 'output': str(output),
              'input_sha256': inputs, 'source_checkpoint_sha256': hashes,
              'runner_sha256': digest(Path(__file__)), 'tests_sha256': digest(ROOT / 'test_l2_candidates.py'),
              'boards': args.boards, 'seed': cfg['seed'], 'last_seed': cfg['seed'] + args.boards - 1,
              'models': models, 'corners': [0, 1], 'sample_ticks': SAMPLE_TICKS,
              'states_per_game': args.states_per_game, 'seconds': args.seconds, 'top_k_schedules': TOP_K,
              'sample_rule': 'first states_per_game ticks from sample_ticks with a T2 label; no outcome filtering',
              'opponent': 'static STOP; production robust mask at each prefix action; forecast escape suffix',
              'selected_stress_rng': 'evader/rule, same seed*1000+aid*500+tick; not historical T2 branch RNG',
              'candidate_scope': 'all scored paths passing original physics, positive pressure and first mask within <=64 schedules',
              'full_protocol': args.boards == 10 and args.states_per_game == 4}
    dump(output / 'config.json', config)
    (output / 'source').mkdir()
    for name in (*[n for n in hashes if n.endswith('.py')], 'audit_l2_candidates.py', 'test_l2_candidates.py'):
        shutil.copyfile(ROOT / name, output / 'source' / name)
    dump(output / 'status.json', {'status': 'running', 'states_checked': 0, 'games_replayed': 0})
    print(f'Output: {output}\nCorpus: {len(targets)} games; <= {len(targets)*args.states_per_game} states; budget {args.seconds}s.', flush=True)
    try:
        with gzip.open(source / 'trace.jsonl.gz', 'rt', encoding='utf-8') as trace, \
             gzip.open(output / 'candidates.jsonl.gz', 'wt', encoding='utf-8') as records:
            current, env, expected_tick = None, None, 0
            for line in trace:
                if perf_counter() >= deadline:
                    break
                row = json.loads(line)
                key = row['model'], row['seed'], row['aid']
                if key not in targets:
                    continue
                if current != key:
                    assert current is None, 'Selected games must be contiguous'
                    current, expected_tick = key, 0
                    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=row['seed'])
                    env.reset()
                assert row['tick'] == expected_tick
                eligible = (row['tick'] in SAMPLE_TICKS and row['first_actions']
                            and game_counts[key] < args.states_per_game)
                if eligible:
                    result = inspect_state(env, row['aid'], row['tick'], deadline,
                                           row['seed']*1000 + row['aid']*500 + row['tick'])
                    assert result['state_id'] == row['state_id']
                    assert result['mask'] == row['mask']
                    assert result['teacher_stats']['plans'] == row['stats']['plans']
                    assert result['old_first_actions'] == row['first_actions']
                    result.update(model=row['model'], seed=row['seed'], aid=row['aid'], game_id=row['game_id'])
                    states.append(result)
                    records.write(json.dumps(result, ensure_ascii=False) + '\n')
                    records.flush()
                    game_counts[key] += 1
                    dump(output / 'status.json', {'status': 'running', 'states_checked': len(states),
                        'games_replayed': len(seen_games), 'elapsed_seconds': perf_counter() - started})
                    print(f"[{len(states)} states] {row['game_id']} t{row['tick']}: "
                          f"paths={result['checked_paths']}/{result['scored_paths']} viable={result['viable_paths']} "
                          f"two={result['best_viable_two_bomb'] is not None} ms={result['state_total_ms']:.0f}", flush=True)
                    if not result['complete']:
                        break
                env.step(row['actions'])
                expected_tick += 1
                if row['info']['episode_done']:
                    seen_games.add(key)
                    current, env = None, None
                    if seen_games == targets:
                        break
        complete = seen_games == targets and all(s['complete'] for s in states)
        after = {name: digest(ROOT / name) if name.endswith('.py') else digest(Path(cfg['checkpoints'][name])) for name in hashes}
        assert hashes == after and all(digest(source / name) == value for name, value in inputs.items())
        assert digest(Path(__file__)) == config['runner_sha256']
        assert digest(ROOT / 'test_l2_candidates.py') == config['tests_sha256']
        summary = {'status': 'complete' if complete else 'incomplete',
                   'reason': 'corpus_finished' if complete else 'budget_exhausted',
                   'full_protocol': config['full_protocol'], 'decision': 'review_pending_keep_original',
                   'games_replayed': len(seen_games), 'expected_games': len(targets),
                   'states_checked': len(states), 'maximum_states': len(targets)*args.states_per_game,
                   'hashes_unchanged': True, 'elapsed_seconds': perf_counter() - started,
                   'games_sampling': [{'model': m, 'seed': s, 'aid': a, 'states': game_counts[m, s, a],
                       'replayed_complete': (m, s, a) in seen_games} for m, s, a in sorted(targets)],
                   'overall': summarize_states(states), 'by_model': {m: summarize_states([s for s in states if s['model'] == m]) for m in models},
                   'limitations': ['Saved fixed-policy T2 corpus; diagnostic only, not new training/test for model promotion.',
                       'Only scored positive-pressure candidates within original <=64 schedules, not exhaustive game planning.',
                       'Static opponent; robust mask still models counter-bombs. Passing it does not guarantee future opponent reactions.',
                       'Viable requires prefix mask/physics pass and actual forecast suffix survival; post-terminal physics is not a game win.',
                       'Top-K measured through all paths in K schedules; timing includes clone/suffix overhead, not a production guard benchmark.',
                       'Frame/branch samples are correlated within board seeds; counts here are descriptive, not independent trials.']}
        dump(output / 'summary.json', summary)
        write_report(output, summary)
        dump(output / 'status.json', {k: summary[k] for k in ('status', 'reason', 'games_replayed', 'expected_games', 'states_checked', 'elapsed_seconds')})
        archive = shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(f"{summary['status'].upper()}. Send: {archive}", flush=True)
        return output
    except BaseException as exc:
        dump(output / 'status.json', {'status': 'incomplete', 'reason': f'{type(exc).__name__}: {exc}',
              'states_checked': len(states), 'games_replayed': len(seen_games), 'elapsed_seconds': perf_counter() - started})
        print(f'INCOMPLETE. Keep folder/log: {output}', flush=True)
        raise


if __name__ == '__main__':
    main()
