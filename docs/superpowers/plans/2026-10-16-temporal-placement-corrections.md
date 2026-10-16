# Temporal placement corrections

The independent acceptance matrix and RED tests are in `docs/superpowers/test-matrices/2026-10-16-temporal-placement-regressions.md` and `test_temporal_trap_regressions.py`. Preserve existing run directories and all unrelated working-tree changes.

1. Give proposed bombs an explicit placement tick. Activate each proposed bomb in the forecast only on that tick; its normal timer begins then. Keep actual environment bombs active immediately.
2. Retain reachable intermediate path nodes so the first placement may follow movement. Check the complete two-placement action timeline against burns, blockers, sudden-death closure, and a time-indexed escape margin.
3. Allow a caller-selected fresh output directory in the temporal launcher while keeping default behavior, no-overwrite protection, and evaluation-only recovery.
4. Turn the independent RED suite GREEN, run focused and full tests with `TEMP`/`TMP` set to the writable project cache, compile and diff checks, then request an independent code review. Fix any Critical or Important findings before a fresh PowerShell training launch.

Do not commit, push, delete/reset changes, or overwrite old run outputs. The final training command must use a new output directory.
