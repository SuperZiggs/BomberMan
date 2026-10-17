"""Independent B0 smoke archive, source, checkpoint and metric checks."""
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audit_t2 import digest, dump

review = {}
for device in ('cpu', 'cuda'):
    out = ROOT / f'runs/gae-b0-smoke-{device}-20261015-implementation'
    config = json.loads((out / 'config.json').read_text())
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['hashes_unchanged']
    assert not summary['full_protocol'] and summary['decision'] == 'retain_original'
    assert not list(out.glob('evaluation*.json')) and not (out / 'test_opened.json').exists()
    assert digest(Path(config['warm_start'])) == config['warm_start_sha256']
    for name, expected in config['source_sha256'].items():
        assert digest(ROOT / name) == digest(out / 'source' / name) == expected
    with zipfile.ZipFile(str(out) + '-send.zip') as archive:
        assert archive.testzip() is None
        for entry in archive.infolist():
            if not entry.is_dir():
                assert archive.read(entry) == (out / entry.filename).read_bytes()
        archive_files = sum(not e.is_dir() for e in archive.infolist())
    modes = {}
    for mode, estimator, lam in (('mc', 'mc', 1.), ('gae1', 'gae', 1.), ('gae095', 'gae', .95)):
        folder = out / mode
        cfg = json.loads((folder / 'config.json').read_text())
        assert (cfg['episodes'], cfg['max_steps'], cfg['seed'], cfg['attack_teacher']) == (20, 20, 2026101503, 'local')
        assert cfg['advantage_estimator'] == estimator and cfg['gae_lambda'] == lam
        assert not cfg['normalize_advantage'] and cfg['validation_every'] == 0 and not cfg['run_report']
        rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        assert [r['episode'] for r in rows] == [r['optimizer_steps'] for r in rows] == list(range(1, 21))
        assert [r['learning_aid'] for r in rows] == [i % 2 for i in range(20)]
        for row in rows:
            d = row['advantage']
            assert 1 <= d['samples'] == row['learner_steps'] <= 20 and d['normalize_enabled'] is False
            for key in ('raw', 'policy', 'returns', 'values'):
                assert all(math.isfinite(v) for v in d[key].values())
            assert math.isclose(row['value_loss'], d.get('mc_raw', d['raw'])['mean'] ** 2 +
                                d.get('mc_raw', d['raw'])['std'] ** 2, rel_tol=1e-5, abs_tol=1e-5)
            if estimator == 'gae':
                assert d['estimator'] == 'gae' and d['gae_lambda'] == lam and d['terminal_transitions'] == 1
                assert d['bootstrap_transitions'] == d['continuations'] == d['samples'] - 1
        for kind in ('best', 'final'):
            assert digest(folder / f'{kind}_phaseC.pt') == summary['modes'][mode]['checkpoints'][kind]
        assert summary['modes'][mode]['learner_steps'] == sum(r['learner_steps'] for r in rows)
        modes[mode] = summary['modes'][mode]
    review[device] = {'archive_files': archive_files, 'elapsed_seconds': summary['elapsed_seconds'],
                      'hashes_and_metrics_pass': True, 'modes': modes}
before = ROOT / 'runs/gae-b0-implementation-20261015/before/train_actor_critic.py'
assert digest(before) == digest(ROOT / 'runs/advantage-full-20261015-030430/source/train_actor_critic.py')
dump(Path(__file__).parent / 'smoke_review.json', review)
print(json.dumps(review, indent=2))
