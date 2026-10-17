"""Recompute the frozen control confirmation; never train or evaluate another board."""
from collections import Counter
import hashlib
import json
from math import isclose
from pathlib import Path
import sys
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


protocol, saved = read(OUT / 'protocol.json'), read(OUT / 'summary.json')
registration = ROOT / 'runs/t3-curriculum-full-20261014-154703/control_confirmation_protocol.json'
assert sha(OUT / 'protocol.json') == sha(registration)
with zipfile.ZipFile(str(OUT) + '-send.zip') as archive:
    assert archive.testzip() is None
    files = [i for i in archive.infolist() if not i.is_dir()]
    for item in files:
        relative = item.filename.removeprefix(OUT.name + '/')
        local = (OUT / relative).resolve()
        assert local.is_relative_to(OUT)
        with archive.open(item) as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == sha(local), item.filename
for name, expected in protocol['source_sha256'].items():
    assert sha(ROOT / name) == expected, name
for name, checkpoint in protocol['checkpoints'].items():
    assert sha(Path(checkpoint['path'])) == checkpoint['sha256'], name
assert saved['status'] == 'complete' and saved['confirmation_only'] and saved['hashes_unchanged']
assert not saved['curriculum_protocol_completed'] and not saved['automatic_model_replacement']
print(f'ZIP {len(files)} files, frozen protocol/source/checkpoints: PASS', flush=True)

sys.path.insert(0, str(ROOT))
from pilot_two_bomb import paired_active_bootstrap

expected = {(s, a) for s in range(32000, 32200) for a in (0, 1)}
results, counts = {}, {}
for name in ('original', 'control_best'):
    data = read(OUT / f'evaluation_{name}_32000.json')
    assert (data['num_episodes'], data['seed'], data['board_size'], data['num_agents'], data['max_steps']) == (400, 32000, 9, 2, 500)
    keys = [(g['seed'], g['aid']) for g in data['games']]
    assert len(keys) == len(set(keys)) == len(expected) == 400 and set(keys) == expected
    for game in data['games']:
        assert all(game[k] in (0, 1) for k in ('win', 'active_win', 'suicide', 'draw'))
        assert game['active_win'] == int(game['win'] and game['kills'] > 0)
        assert not (game['win'] and (game['draw'] or game['suicide']))
        assert 1 <= game['steps'] <= 500
    counts[name] = {}
    for corner, games in [('overall', data['games']), *[(str(a), [g for g in data['games'] if g['aid'] == a]) for a in (0, 1)]]:
        target = data if corner == 'overall' else data['by_corner'][corner]
        mirror = saved['models'][name] if corner == 'overall' else saved['models'][name]['by_corner'][corner]
        assert target['num_episodes'] == mirror['num_episodes'] == len(games)
        rates = {f'{k}_rate': 100 * sum(g[k] for g in games) / len(games) for k in ('win', 'active_win', 'suicide', 'draw')}
        rates.update({f'avg_{k}': sum(g[k] for g in games) / len(games) for k in ('kills', 'bombs', 'items', 'steps')})
        for key, value in rates.items():
            assert isclose(value, target[key]) and isclose(value, mirror[key]), (name, corner, key)
        assert isclose(mirror['passive_win_rate'], rates['win_rate'] - rates['active_win_rate'])
        counts[name][corner] = {'games': len(games),
            **{k: sum(g[k] for g in games) for k in ('win', 'active_win', 'suicide', 'draw', 'kills')},
            'passive_wins': sum(g['win'] - g['active_win'] for g in games)}
    results[name] = data

comparison = paired_active_bootstrap(results['control_best'], results['original'],
                                    samples=protocol['bootstrap']['samples'], seed=protocol['bootstrap']['seed'])
assert comparison == saved['comparison']
checks = {'minimum_gain': comparison['difference_pp'] >= protocol['gate']['minimum_activeWR_gain_pp'],
          'positive_ci_lower': comparison['ci95_pp'][0] > protocol['gate']['ci95_difference_lower_above'],
          'observed_suicide_not_higher': results['control_best']['suicide_rate'] <= results['original']['suicide_rate']}
passed = all(checks.values())
assert passed == saved['gate_passed']
assert saved['decision'] == ('candidate_for_replication' if passed else 'retain_original')
original = {(g['seed'], g['aid']): g for g in results['original']['games']}
candidate = {(g['seed'], g['aid']): g for g in results['control_best']['games']}
transitions = Counter((original[k]['active_win'], candidate[k]['active_win']) for k in sorted(expected))
diagnostics = {name: {'suicide_cases': [{'seed': g['seed'], 'aid': g['aid'], 'steps': g['steps']}
                 for g in data['games'] if g['suicide']], 'activeWR_by_corner':
                 {a: data['by_corner'][a]['active_win_rate'] for a in ('0', '1')}} for name, data in results.items()}
report = {'zip_files_verified': len(files), 'source_and_checkpoint_hashes_verified': True,
          'protocol_matches_registration': True, 'game_rates_averages_and_corners_recomputed': True,
          'games_recomputed': 800, 'independent_resampling_unit': '200 board seeds with both A0/A1',
          'comparison': comparison, 'gate_checks': checks, 'gate_passed': passed, 'counts': counts,
          'active_win_transitions_B_to_control': {f'{a}->{b}': n for (a, b), n in sorted(transitions.items())},
          'diagnostics': diagnostics, 'decision': saved['decision'], 'no_new_model_or_board_evaluation': True}
(OUT / 'analysis_results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('800 games/rates/averages/corners, paired bootstrap200 boards, registered gates: PASS', flush=True)
print(json.dumps(report, indent=2))
