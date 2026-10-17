from pathlib import Path
import json, math, zipfile
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit_t2 import digest
results = {}
for device in ('cpu', 'cuda'):
    folder = ROOT / f'runs/reward-smoke-{device}-20261015-implementation'
    cfg = json.loads((folder / 'config.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    assert summary['status'] == 'complete' and not summary['full_protocol']
    assert summary['hashes_unchanged'] and not summary['eligible_for_replication']
    assert summary['decision'] == 'retain_original'
    for name, sha in cfg['source_sha256'].items():
        assert digest(ROOT / name) == digest(folder / 'source' / name) == sha
    configs = {}
    for arm, coefficient in cfg['arms'].items():
        sub = folder / arm
        configs[arm] = json.loads((sub / 'config.json').read_text())
        assert configs[arm]['wall_destroy_bonus'] == coefficient
        assert summary['arms'][arm]['completed_episodes'] == summary['arms'][arm]['optimizer_steps'] == 20
        rows = [json.loads(line) for line in (sub / 'metrics.jsonl').read_text().splitlines()]
        assert len(rows) == 20
        for row in rows:
            assert math.isclose(sum(row['reward_components'].values()), row['reward'], rel_tol=0, abs_tol=1e-10)
            assert row['reward_component_max_error'] <= 1e-10
            assert row['contact']['min_distance'] is not None
            rewarded = row['reward_components'].get('wall', 0) / coefficient
            assert math.isclose(rewarded, round(rewarded), rel_tol=0, abs_tol=1e-9)
            assert 0 <= round(rewarded) <= row['walls']
    assert {k:v for k,v in configs['control'].items() if k not in ('wall_destroy_bonus','checkpoint_dir')} == {
           k:v for k,v in configs['half_wall'].items() if k not in ('wall_destroy_bonus','checkpoint_dir')}
    for name in ('original', 'control', 'half_wall'):
        data = json.loads((folder / f'evaluation_{name}_39860.json').read_text())
        assert len(data['games']) == 2 and 'contact' in data
        assert all('walls_rewarded' in r and 'contact' in r for r in data['games'])
    with zipfile.ZipFile(str(folder) + '-send.zip') as z:
        assert z.testzip() is None
        for rec in z.infolist():
            if not rec.is_dir(): assert z.read(rec) == (folder / rec.filename).read_bytes()
    results[device] = {'status':'reviewed', 'elapsed_seconds':summary['elapsed_seconds'],
        'episodes_per_arm':20, 'optimizer_steps_per_arm':20,
        'peak_cuda_bytes':summary['arms']['half_wall']['peak_cuda_bytes']}
(ROOT / 'runs/r1-implementation-20261015/smoke_review.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
