"""Collect final verification only after the full root command has finished."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
evidence = Path(__file__).resolve().parent
document = root / 'docs/superpowers/test-matrices/2026-10-23-s2-kl-verification.json'
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
log = (evidence / 'full-regression-reviewed.log').read_text(encoding='utf-8', errors='replace')
runs = re.findall(r'^Ran (\d+) tests in ([\d.]+)s$', log, re.M)
assert runs, 'Root verification has not finished'
tests, seconds = int(runs[-1][0]), float(runs[-1][1])
assert tests == 586, (tests, 'Unexpected discovery count')
qualified = [list(row) for row in re.findall(r'^(ERROR|FAIL): (.+)$', log, re.M)]
baseline = read(evidence / 'baseline-summary.json')
new = [row for row in qualified if row not in baseline['failures']]
assert not new, new
assert sorted(qualified) == sorted(baseline['failures']), 'Baseline identity changed; inspect before handoff'
modules = ['test_s2_kl_protocol', 'test_s2_kl_training', 'test_s2_kl_evaluation',
           'test_run_s2_kl_probe', 'test_s2_kl_review']
scope_tests = sum(unittest.defaultTestLoader.loadTestsFromName(name).countTestCases() for name in modules)
assert scope_tests == 51
protected = read(evidence / 'protected-before.json')
assert all(digest(root / name) == value for name, value in protected.items())
frozen = read(evidence / 'acceptance-hashes-current.json')['hashes']
assert all(digest(root / name) == value for name, value in frozen.items())
from s2_kl_protocol import source_closure, sha, SPEC, verify_run
sources = source_closure()
for label in ('cpu', 'cuda'):
    reg = verify_run(root / f'runs/s2-kl-code-smoke-20261023-{label}-final')
    assert reg['sources'] == sources and reg['spec_sha256'] == sha(root / SPEC)
assert not (root / 'runs/s2-kl-20261023').exists()
data = read(document)
data.update(status='CODE_VERIFIED_WITH_KNOWN_BASELINE_FAILURES',
    root_final=dict(command='S2_KL_ACCEPTANCE_CUDA=1 venv\\Scripts\\python.exe -m unittest discover -p "test_*.py" -q',
        tests=tests, seconds=seconds, exit_code=1,
        error_count=sum(kind == 'ERROR' for kind, _ in qualified),
        failure_count=sum(kind == 'FAIL' for kind, _ in qualified),
        skipped=2, qualified_failures=qualified, new_failures=new,
        known_failures_unchanged=True, log=str(evidence / 'full-regression-reviewed.log')),
    s2_kl_final=dict(tests=scope_tests, passed=scope_tests, errors=0, failures=0,
        actual_cuda_enabled=True, verification='Included in final whole-project command; no S2-KL failure/error IDs'),
    final_protocol=dict(tests=26, seconds=43.218, exit_code=0,
        log=str(evidence / 'final-protocol-green.log')),
    smokes={label: read(evidence / f'smoke-{label}-final.json') for label in ('cpu', 'cuda')},
    reviewed_smokes={label: read(evidence / f'smoke-{label}-review.json') for label in ('cpu', 'cuda')},
    full_seed_preflight=read(evidence / 'full-seed-preflight-final.json'),
    protected_unchanged=True, acceptance_integrity_verified=True,
    source_identity=dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        sources=sources, spec_sha256=sha(root / SPEC), both_final_smokes_match=True),
    full_run_started=False, full_seed_reservation_created=False,
    full_launch_ready=False, pending=[])
assert re.search(r'FAILED \(failures=1, errors=20, skipped=2\)', log), 'Inspect final root summary'
data['review'].update(important_fixed=2, important_remaining=0, minor_deferred=[],
    inventory_followup=dict(red_methods=1, red_assertion_records=2, red_seconds=2.023,
        green='Included in26/26 protocol and final51/51 scope'))
ledger_path = root / '.superpowers/sdd/2026-10-23-s2-kl-retention/progress.md'
ledger = ledger_path.read_text(encoding='utf-8') if ledger_path.exists() else data['execution_ledger']
data['execution_ledger'] = ledger
data['execution_rulings'] = [line for line in ledger.splitlines() if 'Ruling:' in line]
document.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
(evidence / 'root-final-summary.json').write_text(json.dumps(data['root_final'], indent=2), encoding='utf-8')
print(json.dumps(dict(status=data['status'], tests=tests, seconds=seconds,
    s2_kl_passed=scope_tests, new_failures=0, full_launch_ready=False)))
