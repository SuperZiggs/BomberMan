"""Recompute report tables from the completed offline audit."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from audit_t2 import digest, dump

summary = json.loads((OUT / 'summary.json').read_text())
assert summary['status'] == 'complete' and summary['hashes_unchanged']
for path, expected in summary['input_sha256'].items():
    assert digest(Path(path)) == expected
for path, expected in {**summary['source_sha256'], **summary['audit_sha256']}.items():
    assert digest(ROOT / path) == expected
with gzip.open(OUT / 'label_frames.jsonl.gz', 'rt') as stream:
    rows = [json.loads(line) for line in stream]
assert len(rows) == summary['replayed_frames'] == 8224
assert all(r['frame']['label'] is None or r['frame']['mask'][r['frame']['label']] for r in rows)
table = {}
for cohort in ('original', 'curriculum_ep100', 'curriculum_ep150', 'curriculum_ep200'):
    games = [g for g in summary['per_game'] if g['cohort'] == cohort and g['group'] == 'approach']
    nav = [r['frame'] for r in rows if r['cohort'] == cohort and r['group'] == 'approach'
           and r['frame']['clean_state'] and r['frame']['kind'] == 'navigation']
    near = [f for f in nav if f['firing_distance'] == 1]
    assert len(games) == 32 and all(f['label'] in f['geometric_progress_moves'] for f in nav)
    assert all(len(f['geometric_progress_moves']) == 1 for f in near)
    table[cohort] = {'games': len(games), 'owned_kill_successes': sum(g['owned_kill_success'] for g in games),
        'clean_navigation_frames': len(nav), 'teacher_agreements': sum(f['agrees'] for f in nav),
        'any_progress_moves': sum(f['action'] in f['geometric_progress_moves'] for f in nav),
        'one_cell_away_frames': len(near), 'one_cell_away_progress_moves': sum(f['agrees'] for f in near),
        'one_cell_away_premature_bombs': sum(f['action'] == 5 for f in near)}
common = {}
for model, states in json.loads((OUT / 'teacher_path_probabilities.json').read_text()).items():
    common[model] = {}
    for tick in range(5):
        chosen = [r for r in states if r['tick'] == tick]
        values = [r['teacher_probability'] for r in chosen]
        assert len(chosen) == 16
        common[model][str(tick)] = {'states': len(chosen), 'kind': chosen[0]['kind'],
            'distance': chosen[0]['firing_distance'], 'mean_probability': sum(values) / len(values),
            'min_probability': min(values), 'max_probability': max(values)}
fixtures = json.loads((OUT / 'counterbomb_fixtures.json').read_text())
assert len(fixtures) == 96
test_log = ROOT / 'runs/t3-approach-tests-20261015-final.log'
assert 'Ran 119 tests' in test_log.read_text(encoding='utf-8-sig') and '\nOK' in test_log.read_text(encoding='utf-8-sig')
result = {'approach': table, 'common_teacher_path_probabilities': common,
          'fixture_counts': dict(Counter(r['variant'] for r in fixtures)),
          'test_log_sha256': digest(test_log), 'source_and_inputs_still_match': True,
          'note': 'Descriptive frame ratios; rotated roles/repeated seeds are dependent. No new train/test.'}
dump(OUT / 'analysis_results.json', result)
print(json.dumps(result, indent=2))
