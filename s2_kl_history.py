"""Hash-bound indexes for three inspected legacy seed artifacts.

Evidence: runs/s2-kl-launch-preflight-20261023/history-audit.json.
Unknown files retain the ordinary fail-closed inventory path. These indexes
describe exact historical bytes, never a general exemption for null seeds.
"""
from pathlib import Path

from phase_d_s2_protocol import sha


INDEX = {
    'runs/sd6-diagnostics-6600/overrides.json': {
        'sha256': '09f2b9a62205a22aebc880bd523529d1abe2489a642556bf1a2bba94767af4be',
        # The generator runs 20 games at each of 6600/6610/6620/6630.
        # Cover sequential AND corner-paired numbering conservatively. The
        # original null rows remain unassigned; no per-row seed is invented.
        'ranges': ((6600, 6649),),
        'evidence': {
            'runs/sd6-diagnostics-6600/override_probe.py':
                'a3c9c532faf965747a4a67d45a6c4f5b22b9bfa2b7db38fa3bcc4039f6c70d17',
        },
    },
    'runs/t2-full-20261014-060320/replay_cases.json': {
        'sha256': 'c0be883584d68b5351fa1885f8ce98b998352345fdf2f1d092ea3eb0e4d05e78',
        'ranges': ((28000, 28003),),
        'evidence': {},
    },
    'runs/t2-smoke-20261014-060320/replay_cases.json': {
        'sha256': 'cd49441f14e02c275235ab28929841ea7d6dadfd92a31dc7a17c7378366984fb',
        'ranges': ((28000, 28003),),
        'evidence': {},
    },
}


def verified_history(path, root):
    """Return a verified bounded index, or None for ordinary metadata."""
    path, root = Path(path), Path(root)
    relative = path.relative_to(root).as_posix()
    record = INDEX.get(relative)
    if record is None:
        return None
    sources = {}
    for relative, expected in {relative: record['sha256'], **record['evidence']}.items():
        evidence = root / relative
        if not evidence.is_file() or sha(evidence) != expected:
            raise ValueError('Unverified historical seed index/evidence: ' + str(evidence))
        sources[str(evidence)] = expected
    return dict(ranges=[[a, b, str(path)] for a, b in record['ranges']], sources=sources)
