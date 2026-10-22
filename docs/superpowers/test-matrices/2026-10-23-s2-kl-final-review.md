# S2-KL independent final review

Reviewer: fresh independent context, `gpt-6-astra`, read-only, no subagents or heavy test rerun. Immutable range `ed7c7493318061dc8f70921fea3ad319c0814a53..2dbcdd3da4acaa29a56df4567a7f98a69ac85e19`. Verdict: **with fixes**,0Critical/2Important(P2)/0Minor.

Reviewer checked all five Review Focus rows: nested UI/history and registry concurrency; frozen S2 reference before learner restore; locked samples/ties/held-out; micro paired board bootstrap; raw projection/partial shards/fixed endpoint. Reuse of protected trainer, strict input identities, exact CPU/CUDA continuation and gate math were accepted. The four execution rulings were judged justified; independent JSONL correction strengthens acceptance.

## Important findings and disposition

1. `s2_kl_protocol.py`, `verify_run`: requested directory was not bound to `registration.output/experiment_id`. A copied run could pass verification under the original reservation while acquiring a different directory lock. Reviewer reproduced with in-memory redirected reads. Required fix: canonical output/experiment binding; permit only the exact registered `.registration-pending` location with `allow_pending`.
2. `s2_kl_ledger.py`, `validate_run`: missing/partial diagnostics allowed RB games and a diagnostic-dependent validation projection; later valid diagnostics made package/validation resume reject the projection as drift. Reviewer reproduced with existing smoke raw rows and in-memory writes. Required fix: enforce all diagnostics before any validation games/projection, matching the guide's stage order.

Implementer regressions before production fix: `FinalReviewTests.test_copied_output_cannot_reuse_original_experiment_identity` covers full and smoke copies; `FinalReviewTests.test_validation_requires_all_diagnostics_before_any_game_or_projection` covers missing and partial diagnostics, asserts zero dispatched games and no validation projection. Extended RED:2methods,4assertion failure records,0errors,13.875s. Log `runs/s2-kl-code-verification-20261023/final-review-red-extended.log`. One fix pass binds canonical output/experiment identity (exact pending sibling only) and requires the full diagnostic SHA set before validation. GREEN:2/2methods,14.803s,all four cases PASS, in `final-review-green.log`. Final50test/CUDA/root/smoke evidence is recorded in the main verification. No second review seat.

## Declined to judge and executor rulings

- Historical D-S2 data publication `80bafa7`: separate concurrent workflow, not created/pushed by this executor; do not infer authorization or scientific validity. Preserve its history; scoped attributes have no identified S2-KL runtime effect. Cost: future whole-branch integration needs separate payload/scope assessment.
- Existing UI/PDE/journal changes and baseline failures: preserve user work and compare known qualified failures under the approved plan. Cost: unrelated historical failures remain and can conceal unrelated defects.
- Null historical seeds and oversized replay metadata: retain fail-closed full registration until a verified historical mapping exists, without deleting/skipping old logs. Cost: full launch remains unavailable until that prerequisite is resolved.
- Full scientific effectiveness/generalization/long-horizon throughput: make bounded software/resume claims only; no full launch, default-model replacement or full ETA from20step smoke. Cost: the later pilot may fail or take longer than expected.
- Component-gradient instrumentation and inherited reward-decomposition limits: preserve the approved one-variable pilot and unchanged trainer, with advantage diagnostics on/reward diagnostics off. Cost: limited causal attribution/observability; excluded helpers are not thereby proven correct.

No Minor findings deferred. Verification limits: reviewer read supplied test evidence but did not run heavy suites; final root regression was still running when inspected. Two defects were reproduced without filesystem writes. Implementer owns the regression-driven fix and final suite comparison.
