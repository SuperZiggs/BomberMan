# S2-KL publication

Branch: `Loc/s2-kl`. Includes S2-KL code and all new, non-ignored files in
`runs/`, as requested. Existing unrelated working-tree documents are preserved.

- [S2-KL run and raw report](../s2-kl-20261023/report.json)
- [Full analysis in Vietnamese](../s2-kl-20261023-analysis-20261023/analysis_vi.md)
- [Raw reconstruction and archive verification](../s2-kl-20261023-analysis-20261023/verification.json)
- [Launch and regression evidence](../s2-kl-launch-preflight-20261023/launch-preflight.json)
- [Named baseline test failures](../s2-kl-launch-preflight-20261023/root-summary.json)
- [Packaged result](../s2-kl-20261023-send.zip)

Both training arms finished500episodes; validation finished1400games.
Scientific result: `RB_RETENTION_FAIL`. Keep original S2 as the default.
There is no held-out test stage in this pilot.

The existing full-project regression had20errors and1failure, with no new
qualified failure IDs in the595-test launch verification; all60S2-KL tests
passed at that pre-launch verification. See the evidence above for exact scope.

New checkpoints and ZIPs use Git LFS. Experiment snapshots retain their exact
bytes, and registered live sources have explicit checkout line endings so
the published experiment hashes remain meaningful.
