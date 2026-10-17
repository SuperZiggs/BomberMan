# D1 evidence

The raw `-send.zip` is unchanged. This review archive adds independent results, report, console, tests, smoke checks and document snapshots.

241 regression tests passed. A final metadata-only fix made per-epoch objective/clipped flags agree with PPO; 25 focused metadata/compatibility tests then passed. Counts overlap and are not 266 distinct tests. Model/update arithmetic was unchanged by that metadata fix.

D1 tests include sign-sensitive clipped surrogate, fixed targets and group means, common post-update KL stopping, nonfinite gradients, cancellation restoring previous Adam/model, and exact PPO resume in new Python processes on CPU/CUDA. The full trial started fresh, then P4 resumed from episode36/Adam144 after the user freed VRAM. U4 completed files stayed byte-identical. The original deadlines were retained; both console logs and resume audit are included.

The 500-step fixture forces STOP and disables KL stopping to exercise all four updates; it is a memory/finite-value check, not a full-game throughput or policy evaluation. The first mode also includes cold runtime initialization costs and may overlap a finishing compatibility test.

All full-state checkpoints are for the registered Phase C config/source/runtime/device. Resume preserves original deadlines. Best/final weights are for inference; full state is for training recovery. A completed pilot needs no resume.

Only one train seed was used. CI resamples paired board seeds with both corners, not independent games or independent training runs. No automatic model replacement occurred.
