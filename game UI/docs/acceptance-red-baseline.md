# Independent RED baseline

Python baseline executed 2026-10-22 before production implementation was authorized to start.

```text
Ran 19 tests in 0.111s
FAILED (failures=19)
```

All failures were `AssertionError` identifying absent required `bridge.app` or `bridge.snapshots`; no dependency, syntax or unittest discovery errors occurred. Existing NumPy engine import succeeded in the two snapshot tests. No game session or training worker existed or ran.

Command from `E:\Bomberman`:

```powershell
$env:PYTHONPATH = 'E:\Bomberman\game UI\.python-deps;E:\Bomberman\game UI;E:\Bomberman'
& 'E:\Bomberman\venv\Scripts\python.exe' -m unittest discover -s 'game UI/tests/python' -p 'test_*.py' -v
```

Suite: `tests/python/test_acceptance.py`; matrix: `docs/acceptance-matrix.md`. Runtime output is also recorded by the Codex tool transcript. Browser baseline is recorded separately after browser tooling is ready; this Python baseline does not claim browser RED coverage.

Author preimplementation correction: strengthened the copied-frame case with nonempty flame/pickup assertions, and engine replay with actual per-tick or cumulative reward equality. This adds assertions; no requirement or acceptance expectation was relaxed.

## Browser readiness RED

Playwright discovered all 9 tests in 3 files and launched installed Chrome successfully. A sandboxed first run was **environment-blocked** by `net::ERR_NETWORK_ACCESS_DENIED`; it is not feature RED evidence.

The approved unsandboxed rerun executed all 9 tests. All 9 failed the explicit assertion `Runnable loopback browser application must answer GET /; start the documented server`, with `net::ERR_CONNECTION_REFUSED` at `http://127.0.0.1:8000/`. This is **frontend readiness RED only**: browser launch and test discovery are proven, while downstream journey/visual assertions remain pending the running frontend. No missing tooling was mislabeled as a behavior failure. Browser raw report is preserved at `tests/_artifacts/browser-red-readiness.json`.

Command from `E:\Bomberman\game UI`:

```powershell
& '.\node_modules\.bin\playwright.cmd' test --config tests/browser/playwright.config.ts
```

Browser source was then independently corrected for requirements-only assumptions listed in `acceptance-author-corrections.md`; unchanged readiness assertions remain the recorded baseline. The revised 9-test suite passed discovery. No new production source was read.
