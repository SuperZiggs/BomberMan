# Independent acceptance matrix: S2-KL retention pilot

Author read only the frozen requirements packet `docs/superpowers/requirements/2026-10-23-s2-kl-retention.md`, existing source and existing tests. No implementation plan or proposed implementation was read. Tests use public exports from `run_s2_kl_probe`; missing APIs fail assertions before fixture work. Production code is outside this author's ownership.

Command: `E:\Bomberman\venv\Scripts\python.exe -m unittest test_s2_kl_protocol test_s2_kl_training test_s2_kl_evaluation test_run_s2_kl_probe -q`.

| Requirement / risk | Level and scenario | Observable result | Tests |
|---|---|---|---|
| Public contracts; frozen one-variable protocol (§3–6) | Unit, full/smoke scalar config, bounds and schedules | 500 versus 4 episodes; KL0/1; same paired board/corner schedule; no held-out stage | protocol ConfigurationTests; training ScheduleTests |
| Input provenance (§3,6,7) | Integration, copy S2 + SD8A metadata + pinned sample manifest and all referenced shards | Rehash genuine inputs, source closure, canonical registration, no symlinks/junctions; drift rejected | protocol RegistrationTests |
| Seed separation and cross-root inventory (§5) | Integration, synthetic historical JSON/JSONL/registry and nested game UI metadata | Conflict/unresolved history rejects before publication; smoke uses only 790000–790003 and no full reservation | protocol SeedSafetyTests |
| Registry atomicity and same D-S2 lock (§5–6) | Integration, full registration only in isolated input fixture; concurrent lock and crash injection | Other entries unchanged; PENDING/publish/COMPLETE under original lock; same identity resume only; no duplicate reservation | protocol TransactionTests |
| Runtime/device/source immutability (§6) | Integration, runtime projection/source/checkpoint/manifest mutation; unavailable CUDA mock | Reject drift, no fallback, source snapshot used | protocol RegistrationTests; CLI journey |
| Learner KL direction/mask/reference (§3) | Existing trainer regression plus real bounded arm execution | learner||S2, legal mask, mean alive steps, frozen reference, beta0 disabled, rebuilt S2 reference after resume | training KLMathTests; ExactResumeTests |
| Initial state and state0 identity (§3,6) | Integration, interrupt around atomic episode commits | Real optimizer/scheduler/RNG/cursor persisted; committed episodes preserved; state loss fails closed | training ExactResumeTests |
| Exact CPU / actual CUDA resume (§6,10) | Real 4×20 smoke for each arm; interrupt before/after state commit | Uninterrupted versus resumed tensors/optimizer/scheduler/RNG/history equal; no duplicated metrics | training ExactResumeTests; opt-in CudaExactResumeTests |
| Diagnostics neutrality (§6,10) | Integration, inference between stop and resume; diagnostic on/off shared trainer smoke | No weights/RNG drift or loss change from observer | training ExactResumeTests |
| Locked TRAIN samples (§7) | Unit/integration, actual 1024+1024 manifest; malformed row/count/shape/held-out/hash | Exact rows only; ties uniform legal finite maxima; all tie invalid; reject rather than resample | evaluation SamplingTests |
| Clustered CI (§7–8) | Unit, unequal cluster synthetic paired rows and independent numerical oracle | State-micro difference; whole-board paired resample; linear quantiles; identity/range reject | evaluation ClusterTests |
| Fixed endpoint and strict gates (§8–9) | Unit, all priorities and adjacent floating-point boundaries | Endpoint500 gates; checkpoint100/250 do not select best; full conditions emitted; smoke SMOKE_ONLY | evaluation GateTests; ReportTests |
| Offline raw projections (§7,10) | Integration, real model forward on copied locked rows; alter cached summary/raw identity | Finite raw rows, required provenance/digest, R/Q recompute; tamper rejected | evaluation DiagnosticsTests; CLI PackagingTests |
| RB two-corner ledger and resume (§8,10) | End-to-end bounded CPU CLI register/train/diagnose/validate/package; corrupt ledger | Complete board tasks only, no duplicate metrics, worker source/runtime identity checked | CLI JourneyTests |
| Incomplete packaging / scientific failure (§9–10) | CLI/integration package before missing work; smoke end-to-end | INCOMPLETE ZIP of available new output only; no implicit work, no false PASS, complete smoke SMOKE_ONLY; exit0 stage success | CLI PackagingTests; JourneyTests |
| CLI surfaces and exit contracts (§10) | Black-box help, invalid order/flags, KeyboardInterrupt main | register/train/diagnose/validate/package only, frozen budgets, invalid1, interrupted130 | CLI ContractTests |

## Fixture and payload conventions

Fixtures copy only S2, SD8A selected/test-opened/registration/collection-summary metadata, the exact pinned manifest, and its referenced collection shards. No copies of full `runs/`, mutable links, production reservation, original checkpoint writes, or real full training/validation seeds are permitted. Full registration tests reserve only inside their disposable fixture root and never train or evaluate those ranges. CPU smoke executes actual shared trainer and evaluator. CUDA exact resume is enabled by `S2_KL_ACCEPTANCE_CUDA=1` and fails rather than skipping when explicitly enabled without a working CUDA device.

Where the packet names a dict but leaves its inner fields unspecified, the acceptance input convention is explicit: gate summary `{R,Q,AW,SU}` in percent; comparison `anchor_minus_control.R` contains the public cluster-comparison result; loaded samples have `{decision:{obs,mask,scores,targets,boards,ids},anchor:{obs,mask,boards,ids}}`; diagnostics expose `raw.decision`/`raw.anchor` and summary `R`/`Q`. `build_report` receives diagnostics keyed by model SHA and a flat list of standard D-S2 RB records. Conditions may be booleans or dictionaries containing `pass`. Model-manifest digest may be `digest` or `model_manifest_digest`. Endpoint aliases are always `baseline/control500/anchor500`, including smoke endpoint4. Existing raw trainer state keys and `commit_hook(stage,episode)` conventions are used. These names do not constrain private helpers. The frozen result payload must still contain all packet-mandated fields.

## Practical bounds and residual risk

Actual smoke uses four episodes × twenty steps and two validation boards. It demonstrates integration, exact continuation, raw-ledger handling and artifact integrity; it does not establish scientific quality, full-run throughput, long-horizon training stability, multi-seed uncertainty or held-out generalization. Actual CUDA is an explicit bounded acceptance command, not a simulated device. Source/runtime execution is checked through registered worker provenance and a tampered registered snapshot rejection, without editing the live repository. OS-level process crash after fsync is approximated by injected failures immediately before and after atomic state commit and registration publication; power-loss durability remains dependent on filesystem guarantees.

## Baseline evidence

Frozen requirements SHA256: `4f984cde4152626bb2be3cd427fd8d0a65fc1134f31dcdf2a5d17c63a98791b2` (includes public payload clarification agreed with implementer).

Recorded command: `E:\Bomberman\venv\Scripts\python.exe -m unittest test_s2_kl_protocol test_s2_kl_training test_s2_kl_evaluation test_run_s2_kl_probe -q`.

RED evidence: `runs/s2-kl-code-verification-20261023/acceptance-red.log`, SHA256 `20425a926a0ed9cde979d004673507e75633163d5fb6c62cea0fc421dfb34d1e`. Process exit1; 40 test methods, 39 assertion failure records including subtests, zero errors, one opt-in CUDA skip. All failures are `Missing S2-KL public runner run_s2_kl_probe.py`, before fixture setup. Two existing-trainer math/observer regressions pass; they are explicitly regression coverage and are not represented as RED. The two genuine regression tests were also run separately with `-m unittest test_s2_kl_training.KLMathTests -v`: 2/2 PASS.

Fixture preflight separately copied and rehashed the genuine S2, pinned manifest, and all sixty referenced shards successfully. Actual S2 SHA `d222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814`; manifest SHA `9b36b8e07ed8f65175e5c9fbf53a18eca16ae791f0073d19730b28e75529e333`. Collection shards total 3,371,312 bytes. Temporary fixtures were removed, original runs/checkpoints/registry were not modified. No production edit, journal edit, staging or commit was made by the test author.

The synthetic full-report test creates seven shape-valid checkpoint descriptors without training, uses genuine locked sample IDs/boards, supplies 1024+1024 finite raw offline rows/model and 200 synthetic RB corner rows/model, and proves a failing500 endpoint cannot be replaced by perfect100/250 metrics. Missing endpoint, missing diagnostics, missing/duplicate RB corners, short offline rows, or cached-summary tampering cannot create completion/PASS. These synthetic rates are hand-built acceptance data, not experimental results.

Frozen test SHA256:

| File | SHA256 |
|---|---|
| `test_s2_kl_protocol.py` | `4b82443a8b2d25d7c63d306d09b33b35b977395ed955042fc714c0922a50ca16` |
| `test_s2_kl_training.py` | `32289cb78534ad911476b44a818e109f156c611f2637f550570d4c69eda15395` |
| `test_s2_kl_evaluation.py` | `9ea2d6b396ac7293c4ff50e279e83a27c634baa131d647f362e54e814b9171f3` |
| `test_run_s2_kl_probe.py` | `1b994ea4ab4bdb3a212603c1e42daa7f66ccdca54ec5d16fd19b59ae1f5ae190` |
| `test_s2_kl_support.py` | `aa196acdcf29c5c4e01af6b2817a51c78d1243b55e3a4d29cd061170cfa53a61` |

The suite is handed off unchanged. Its runtime integration assertions will execute after implementation exists. Any subsequent test correction requires author review with frozen-requirements evidence or a demonstrated technical invalidity, preserving the prior assertion and reason.

## Independent-author fixture correction: valid JSONL overlap case

After the original acceptance suite passed, a fixture-integrity audit proved a technical invalidity in `SeedSafetyTests.test_nested_ui_history_raw_ledger_mechanism_and_reservation_conflicts_fail_closed`: the `runs/legacy/raw.jsonl` case was emitted through `write_json`, which uses `indent=2`. Its eight-line pretty JSON object was valid JSON but invalid JSONL. Parsing its first line raised `JSONDecodeError: Expecting property name enclosed in double quotes`; registration rejected `Unresolved seed history` due to JSON parsing, so the original broad rejection assertion did not demonstrate the intended board710012 overlap.

The independent author corrected only that JSONL fixture to one complete JSON object per line, with a trailing newline, and strengthened only that case's expected reason to `(?i)overlap`. Other fixtures and assertions remain unchanged. A separate proof parsed the corrected single line as `{'records': [{'board': 710012, 'corner': 0}]}` and observed registration rejection `Seed overlap with historical boards`, with no output published. This correction removes an unrelated setup rejection and strengthens the existing frozen seed-isolation requirement; it does not relax a production acceptance criterion or create a new RED claim.

The original test is preserved in existing commit `58e5321:test_s2_kl_protocol.py`. Original SHA256: `4b82443a8b2d25d7c63d306d09b33b35b977395ed955042fc714c0922a50ca16`. Corrected SHA256: `6e829dbb0c7d2c4af4bbc64c440e7a9b6ac65a3e099e7739d975e4a968aa9abd`. The original freeze table above intentionally remains historical evidence.

Validation command: `E:\Bomberman\venv\Scripts\python.exe -m unittest test_s2_kl_protocol.SeedSafetyTests.test_nested_ui_history_raw_ledger_mechanism_and_reservation_conflicts_fail_closed -v`. Result: exit0, 1 test passed in 1.997 seconds, no errors/failures. The scalar710012 overlap behavior was already implemented; this is fixture-integrity verification and regression coverage. No production code, other tests, original experiment artifacts, staging, commit or journal was changed for this correction.
