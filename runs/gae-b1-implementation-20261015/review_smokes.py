from pathlib import Path
from collections import Counter
import json, math, zipfile, sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from audit_t2 import digest, dump
reports = {}
for device in ('cpu', 'cuda'):
    folder = ROOT / f'runs/gae-b1-smoke-{device}-20261015-implementation'
    cfg = json.loads((folder/'config.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    assert summary['status'] == 'complete' and not summary['full_protocol']
    assert summary['hashes_unchanged'] and not summary['eligible_for_replication']
    assert summary['decision'] == 'retain_original'
    assert digest(Path(cfg['warm_start'])) == cfg['warm_start_sha256']
    for name, expected in cfg['source_sha256'].items():
        assert digest(ROOT/name) == digest(folder/'source'/name) == expected
    configs, counts = {}, {}
    for arm, estimator in cfg['arms'].items():
        configs[arm] = json.loads((folder/arm/'config.json').read_text())
        rows = [json.loads(line) for line in (folder/arm/'metrics.jsonl').read_text().splitlines()]
        assert len(rows) == 20 and [r['optimizer_steps'] for r in rows] == list(range(1,21))
        assert summary['arms'][arm]['optimizer_instances'] == 1
        for r in rows:
            d = r['advantage']
            assert not d['normalize_enabled'] and d['raw'] == d['policy']
            assert math.isfinite(r['loss']) and math.isfinite(d['grad_norm_before_clip'])
            if estimator == 'gae':
                assert d['estimator'] == 'gae' and d['gae_lambda'] == .95
                assert d['terminal_transitions'] == 1
                assert d['bootstrap_transitions'] == d['continuations'] == d['samples']-1
        for name, checkpoint in summary['arms'][arm]['checkpoints'].items():
            assert digest(Path(checkpoint['path'])) == checkpoint['sha256']
            weights = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
            assert all(torch.isfinite(v).all() for v in weights.values())
        counts[arm] = {'learner_steps': sum(r['learner_steps'] for r in rows),
            'boundaries': dict(Counter(r['advantage'].get('last_boundary') for r in rows)) if estimator=='gae' else None}
    assert {k:v for k,v in configs['control'].items() if k not in ('checkpoint_dir','advantage_estimator')} == {k:v for k,v in configs['gae'].items() if k not in ('checkpoint_dir','advantage_estimator')}
    with zipfile.ZipFile(str(folder)+'-send.zip') as z:
        assert z.testzip() is None
        for item in z.infolist():
            if not item.is_dir():
                assert z.read(item) == (folder/item.filename).read_bytes()
        count = sum(not item.is_dir() for item in z.infolist())
    reports[device] = {'passed': True, 'raw_files': count, 'seconds': summary['elapsed_seconds'], 'arms':counts}
dump(Path(__file__).with_name('smoke_review.json'), reports)
print(json.dumps(reports, indent=2))
