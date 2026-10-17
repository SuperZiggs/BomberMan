"""Audit the completed run's saved results without playing or training new games."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(ROOT))
from pilot_two_bomb import paired_active_bootstrap, complete_validation, save_json
from run_two_bomb_complete import reference_inputs


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def training_summary(metrics):
    totals, rejected = Counter(), Counter()
    for row in metrics:
        totals.update({k: v for k, v in row['teacher'].items() if k != 'rejected'})
        rejected.update(row['teacher']['rejected'])
    windows = []
    for start in range(0, len(metrics), 50):
        window = metrics[start:start + 50]
        windows.append({'episodes': [start + 1, start + len(window)],
                        **{k: float(np.mean([r[k] for r in window])) for k in
                           ('bombs', 'walls', 'items', 'steps', 'reward', 'policy_loss',
                            'value_loss', 'combat_loss', 'escape_loss', 'navigation_loss', 'loss')},
                        **{k + '_rate': float(np.mean([r[k] for r in window]) * 100)
                           for k in ('win', 'active_win', 'suicide', 'draw')}})
    peak = max(metrics, key=lambda r: r['loss'])
    return {'episodes': len(metrics), 'learner_steps': sum(r['learner_steps'] for r in metrics),
            'teacher': {**totals, 'rejected': dict(rejected),
                        'accepted_rate': totals['accepted'] / totals['queries'] * 100 if totals['queries'] else None,
                        'two_bomb_share_of_labels': totals['two_bomb_labels'] / totals['accepted'] * 100 if totals['accepted'] else None,
                        'mean_pressure': totals['pressure'] / totals['accepted'] if totals['accepted'] else None,
                        'mean_ms': totals['milliseconds'] / totals['queries'] if totals['queries'] else None},
            'windows_50': windows,
            'peak_loss': {k: peak[k] for k in ('episode', 'loss', 'policy_loss', 'value_loss',
                                             'combat_loss', 'escape_loss', 'navigation_loss')}}


def main():
    config, summary = read_json(OUTPUT / 'config.json'), read_json(OUTPUT / 'summary.json')
    expected_pairs = {(seed, aid) for seed in range(26000, 26200) for aid in (0, 1)}
    results = {}
    for name in ('original', 'local', 'two_bomb'):
        result = read_json(OUTPUT / f'evaluation_{name}_26000.json')
        assert finite(result)
        games = result['games']
        assert len(games) == 400 and {(g['seed'], g['aid']) for g in games} == expected_pairs
        assert (result['seed'], result['board_size'], result['num_agents'], result['max_steps']) == (26000, 9, 2, 500)
        assert result['num_episodes'] == 400
        for field in ('win', 'active_win', 'suicide', 'draw'):
            assert all(g[field] in (0, 1) for g in games)
            assert math.isclose(result[field + '_rate'], np.mean([g[field] for g in games]) * 100, abs_tol=1e-10)
        for field in ('kills', 'bombs', 'items', 'steps'):
            assert math.isclose(result['avg_' + field], np.mean([g[field] for g in games]), abs_tol=1e-10)
        for aid in (0, 1):
            corner = result['by_corner'][str(aid)]
            corner_games = [g for g in games if g['aid'] == aid]
            assert corner['num_episodes'] == len(corner_games) == 200
            for field in ('win', 'active_win', 'suicide', 'draw'):
                assert math.isclose(corner[field + '_rate'], np.mean([g[field] for g in corner_games]) * 100, abs_tol=1e-10)
            for field in ('kills', 'bombs', 'items', 'steps'):
                assert math.isclose(corner['avg_' + field], np.mean([g[field] for g in corner_games]), abs_tol=1e-10)
        assert {k: v for k, v in result.items() if k != 'games'} == summary['models'][name]
        assert digest(Path(result['checkpoint'])) == result['checkpoint_sha256']
        results[name] = result
    original_path = ROOT / 'runs/pressure-C-next/final_phaseC.pt'
    assert digest(original_path) == config['warm_start_sha256']
    reference_inputs()
    assert digest(Path(results['local']['checkpoint'])) == config['control_sha256']
    source_checks = {f: digest(ROOT / f) == sha and digest(OUTPUT / 'source' / f) == sha
                     for f, sha in config['source_sha256'].items()}
    assert all(source_checks.values())
    comparisons = {name: paired_active_bootstrap(results['two_bomb'], results[name])
                   for name in ('original', 'local')}
    assert comparisons == summary['comparisons']
    original = torch.load(original_path, map_location='cpu', weights_only=True)
    best = torch.load(OUTPUT / 'two_bomb/best_phaseC.pt', map_location='cpu', weights_only=True)
    final = torch.load(OUTPUT / 'two_bomb/final_phaseC.pt', map_location='cpu', weights_only=True)
    best_equal = original.keys() == best.keys() and all(torch.equal(original[k], best[k]) for k in original)
    final_changed = sum(not torch.equal(original[k], final[k]) for k in original)
    metrics = rows(OUTPUT / 'two_bomb/metrics.jsonl')
    validations = rows(OUTPUT / 'two_bomb/validation_metrics.jsonl')
    assert finite(metrics) and finite(validations)
    assert [r['episode'] for r in metrics] == list(range(1, 201))
    assert all(r['learning_aid'] == (r['episode'] - 1) % 2 for r in metrics)
    assert complete_validation(validations)
    assert summary['best_episode'] == 0 and best_equal
    assert summary['training_complete'] and summary['evaluation_complete'] and summary['completed_episodes'] == 200
    eligible = (all(c['difference_pp'] >= 5 and c['ci95_pp'][0] > 0 for c in comparisons.values())
                and all(results['two_bomb']['suicide_rate'] <= results[name]['suicide_rate']
                        for name in ('original', 'local')))
    assert summary['eligible_for_replication'] == eligible == False
    assert summary['decision'] == 'retain_original'
    report = {'source_checks': source_checks, 'original_checkpoint_unchanged': True,
              'evaluations_verified': {name: {'games': 400, 'board_pairs': 200,
                                             'counts': {k: sum(g[k] for g in r['games'])
                                                        for k in ('win', 'active_win', 'suicide', 'draw', 'kills')}}
                                       for name, r in results.items()},
              'best_weights_equal_original': best_equal, 'final_changed_tensors': final_changed,
              'original_and_two_bomb_game_records_equal': results['original']['games'] == results['two_bomb']['games'],
              'comparisons_recomputed': comparisons,
              'local_minus_original': paired_active_bootstrap(results['local'], results['original']),
              'training': training_summary(metrics),
              'control_training': training_summary(rows(ROOT / 'runs/two-bomb-pilot-20261014/local/metrics.jsonl')),
              'decision': 'retain_original', 't1_execution': 'complete', 'two_bomb_improvement_gate': 'FAIL',
              'limitations': ['One training seed', 'Final ep200 has validation only; final held-out not evaluated',
                              'No per-state teacher agreement, plan completion or full-trap trace in current metrics']}
    save_json(OUTPUT / 'analysis_results.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
