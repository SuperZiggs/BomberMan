# S2-KL Retention Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for Native execution, or superpowers:subagent-driven-development if the user changes the execution method. Steps use checkbox (`- [ ]`) syntax for tracking. Independent acceptance-test authorship remains required with either method.

**Goal:** Triển khai pilot KL0/KL1 từ S2, 500 episode/arm cùng RuleBot, GPU/full-state resume và quyết định giữ kỹ năng theo endpoint 500 đã duyệt.

**Architecture:** Runner/context mới tái dùng nguyên training loop, anchor loss và full-state hiện có; bốn helper mới tách provenance, train, metric và ledger. Offline chấm sample SD8A đã khóa; validation chỉ RuleBot, mỗi task một board/hai góc. Stage chạy snapshot source và chỉ công bố report từ raw artifact đủ identity/count.

**Tech Stack:** Python/PyTorch/NumPy/unittest hiện có, stdlib pathlib/json/hashlib/ast/subprocess/multiprocessing/zipfile và `progress.py`; không thêm dependency.

**Spec:** `docs/superpowers/specs/2026-10-23-s2-kl-retention-design.md`, người dùng duyệt bản `59350ac` ngày 23/10/2026.

**Status:** APPROVED_IMPLEMENTATION — người dùng duyệt và yêu cầu triển khai ngày23/10/2026. Giữ phương thức Native: implementer hiện tại thực hiện các task, acceptance có tác giả độc lập và cuối lượt có reviewer độc lập. Theo dõi execution trong ledger của plan; full pilot chỉ khởi chạy khi có yêu cầu riêng sau code-ready.

## Global Constraints

- S2 duy nhất: `runs/sd8a-20261020/train/epoch_16.pt`, SHA256 `d222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814`; 13 kênh/9×9, strict tensor shape.
- `control` và `anchor`: 500 episode/arm, chỉ RuleBot; `anchor_coef=0.0/1.0`, cùng weights S2/Adam mới/RNG 2026102301, cùng máy/runtime; train tuần tự CUDA:0, evaluation CPU 1 thread và tối đa 2 worker.
- Phase D, 2 agent, max_steps=500; E475/25/40, xác suất1.0. AC/MC/normalize_advantage=false; lr=2.5e-5/gamma=0.99/value_coef=0.5/entropy_coef=0.0075, cosine T_max=500/eta_min=lr×0.05, entropy theo 500 episode.
- Teacher local/v2, attack_lookahead=2/min_pressure_gain=0.15/trap_sequence_depth=2/sequence_pressure_gain=0.35; escape/combat/navigation/ring=0.5/0.9/0.4/0.6; late_game_start=350/multiplier=1.75/wall=0.15; search/group/PPO tắt, validation_every=0/run_report=false.
- KL train mean theo learner steps còn sống, cùng mask, chiều learner||S2. Reference dựng từ S2 trước restore learner, eval/no_grad/ngoài optimizer; hash cố định. Control không giả có KL đã tính.
- Candidate 0/100/250/500, chỉ 500 quyết định; không chọn best100/250. advantage_diagnostics=true, reward_diagnostics=false; không sửa lỗi reward-helper hoặc công thức trainer trong pilot.
- Train board 710000–710249, 250 board×2 góc/arm; validation 720000–720099, 200 trận/model, tối đa 7 SHA=1400 trận. Trần 1000 train episode/500000 learner steps và 700000 validation environment steps.
- Smoke riêng 790000–790003, không mở/reserve dải thật. Full registration rà cả `runs/` và `game UI/runs`, metadata/raw board IDs/registry; unknown/giao dải thì fail closed, không tự đổi seed.
- Offline 1024 decision+1024 anchor TRAIN states từ manifest SHA `9b36b8e07ed8f65175e5c9fbf53a18eca16ae791f0073d19730b28e75529e333`; giữ file/row/shard SHA và loại 150 board held-out gốc. Không resample sample manifest hoặc dùng các state này làm gradient/replay.
- Baseline đo lại cùng runtime; CI 10000 resample/seed20261023. Retention bootstrap cluster board, micro-weight theo state; RB mean hai góc/board. Không làm tròn trước gate; CI lower>0 strict.
- Gate endpoint500 và thứ tự status đúng spec §9: INCOMPLETE → INCONCLUSIVE_NO_DRIFT → RB_RETENTION_FAIL → RETENTION_NOT_PRESERVED → NO_CLEAR_ANCHOR_BENEFIT → RETENTION_PILOT_PASS. R(control)<=R(baseline)−10 pp; R/Q(anchor)>=baseline−5 pp; R(anchor)−R(control)>=10 pp và CI lower>0; RB AW(anchor)>=baseline−2 pp, suicide<=baseline+1 pp.
- Không stage test, không PHASE_D_PASS, không tự đổi model mặc địnhS2. Giữ output/registry reservation D-S2 cũ630000/640000, source và công thức cũ và checkpoint chỉ đọc; không sửa hash cũ để hợp thức hóa drift.
- File/state/shard atomic và một writer; frozen CLI/worker; log append/progress console. Code-ready khác experiment-started; chỉ user yêu cầu khởi chạy mới mở full job, không post-launch monitoring.

## Review Focus

| Failure mode | Observable acceptance | Owner |
|---|---|---|
| UI nested metadata hoặc concurrent registry writer | Phân loại seedRNG đúng; unknown/offset overlap bị từ chối, pending recovery không lấy chỗ của lượt khác | T1 |
| Anchor đổi thành resumed learner hoặc bị optimizer cập nhật | Full-state resume cho kết quả liên tục; reference vẫnS2/no_grad/hash đúng | T2 |
| Sample/NPZ row/held-out board/mask/nhãn tie bị thay | Từ chối identity sai, không đổi sample; đúng tập action đồng hạng và chỉ1024+1024 state TRAIN | T3 |
| Bootstrap vô tình macro các board có số state không đều | Điểm micro/paired CI tính đúng; RB luôn đủ hai góc; biên CI0 không pass | T3 |
| Summary cũ, shard dở hoặc checkpoint100 đẹp hơn500 | Không chọn lại best; stage/package tái dựng raw, thiếu endpoint hoặc projection drift không tạo pass | T4 |

## File map và quyết định tái dùng

| File | Vai trò |
|---|---|
| Create `s2_kl_protocol.py` | Config/constants, identity inputs, source closure, inventory hai root, registration/reservation/runtime verification |
| Create `s2_kl_training.py` | LịchRB/corner, context API của trainer, candidate và committed-state projections |
| Create `s2_kl_evaluation.py` | Sample loader/forward retention, RB task factory, cluster CI và gate/report math |
| Create `s2_kl_ledger.py` | Stage locks, model manifest, atomic offline/board shards, projections và ZIP |
| Create `run_s2_kl_probe.py` | CLI và dispatch frozen sources; export public acceptance seams dưới đây |
| Create `S2_KL_GUIDE.md` | LệnhCMD/register/train/resume/diagnose/validate/package và diễn giải status |
| Create `test_s2_kl_protocol.py`, `test_s2_kl_training.py`, `test_s2_kl_evaluation.py`, `test_run_s2_kl_probe.py` | Acceptance độc lập theo requirements, không đổi để khớp implementation |
| Create requirements/matrix/verification dưới `docs/superpowers/` | Requirement IDs, frozen test hashes, RED/GREEN/regression/smoke evidence |
| Modify spec/plan/RESULTS/IMPROVEMENTS/PLAN_SUDDEN_DEATH/PLAN_BOMBERMAN | Trạng thái và kết quả thực tế; không ghi kết quả khoa học từ smoke |

Giữ nguyên `train_actor_critic.py`, `training_state.py`, engine/RB/mask và bốn module/runner D-S2. Tái dùng stdlib helpers `phase_d_s2_protocol.{atomic_json,canonical_digest,configure_runtime,runtime,inventory_seeds,read_json,sha,validate_task_payload}`; `training_state.{run_lock,atomic_save,load_training_state}`; evaluator `phase_d_s2_evaluation.{load_registered_model,evaluate_board,summarize}`. Không gọi config/verify/selection/package D-S2 cho registration mới.

`source_closure()` cũ cố định entrypoint D-S2: helper mới phải walk AST từ `run_s2_kl_probe.py` và các import động thực tế bằng stdlib, không sửa closure cũ. Schema inner board payload giữ `d-s2-board-result-v1` để dùng evaluator/validator đã kiểm; task_id và registration_digest thuộc S2-KL mới nên không tái dùng shard giữa thí nghiệm.

## Public contracts và artifact schema

Các public functions sau được CLI export bằng import/lazy wrapper; test-author nhận contract hành vi này, không nhận thiết kế helper nội bộ.

| Owner | Interface |
|---|---|
| protocol | `config_for(smoke=False)->dict`; `seed_manifest(smoke=False)->dict`; `register(output,input_root,*,device='cuda:0',workers=2,smoke=False,resume=False)->dict`; `verify_run(output,*,allow_pending=False)->dict` |
| training | `build_train_schedule(arm,*,smoke=False)->list[dict]`; `train_arm(output,arm,*,resume=False,stop_event=None,commit_hook=None)->dict`; `candidates_for_run(output,reg)->list[dict]` |
| evaluation | `load_samples(reg)->dict`; `decision_targets(scores,mask)->(targets,valid)`; `diagnose_model(model,reg)->dict`; `validation_tasks(reg,models)->list[dict]`; `cluster_comparison(left,right,*,resamples=10000,seed=20261023)->dict`; `decide_gate(baseline,control,anchor,comparisons,*,complete,smoke=False)->dict`; `build_report(reg,models,diagnostics,rb_rows)->dict` |
| ledger/CLI | `diagnose_run(output,*,resume=False)->dict`; `validate_run(output,*,resume=False)->dict`; `package_run(output)->dict`; `main(argv=None)->int` |

`model` descriptor: `{id,arm,episode,checkpoint,sha256}`; baselineS2 arm=null/episode0. `candidates_for_run` chỉ công bố inventory khi cả hai arm hoàn tất, state/completion/candidate SHA khớp. `model_manifest.json` giữ aliases, các SHA duy nhất và endpoints `{baseline,control500,anchor500}`, có canonical digest; đây không phải selection chọn best.

Registration format `s2-kl-registration-v1`: experiment_id=`S2-KL:<resolved-output>`, mode/config/input_root/source_root/device/runtime/sources/spec_sha256/seed_manifest/checkpoints.S2/offline_manifest+shard hashes/registration_digest. Registry full dùng nguyên file `runs/phase_d_s2_seed_registry.json` và nguyên `run_lock(registry)` của D-S2; khóa này bảo vệ reserve PENDING → publish output → COMPLETE. Kiểm lại inventory hai root dưới lock; resume chỉ cùng identity. Mode smoke chỉ audit dải smoke trong output, không ghi reservation full, cho phép lặp lại smoke cùng protocol.

`config_for` trả `{train: common_kwargs, anchor_coef: {control:0.0,anchor:1.0}, checkpoint_episodes:[0,100,250,500], endpoint_episode:500, workers:2}`; train không chứa hệ số KL, train_arm chèn giá trị theo arm. Smoke thay episodes=4/max_steps=20, candidates0/2/4/endpoint4, train790000–790001/validation790002–790003; workers chỉ1hoặc2 và cùng beta0/1/manifest đầy đủ. Smoke report chỉ `SMOKE_ONLY`, không quyết định scientific gates. `train_arm` trả `{arm,episode,target,complete}`.

Offline payload format `s2-kl-diagnostics-v1`: model_sha256/registration_digest/sample_manifest_sha256/diagnostic_digest; raw decision rows `{id,board,hit,mass,ce,kl_s2_to_model,bomb_probability,value}`, raw anchor rows tương tự nhưng không có hit/mass/ce. `id` từ relative shard+row, 1024 row mỗi loại, finite numeric; cached summary phải khớp raw. R=100×mean(hit),Q=100×mean(mass). Decision target uniform trên legal finite-score maxima; all-tie/không legal không tạo nhãn hữu ích. Locked sample sai số lượng/row/shape thì từ chối, không lấy mẫu bù.

`cluster_comparison` nhận paired rows `{id,board,value}` có value0..1, ID duy nhất và hai phía trùng ID/board. Trả `{difference_pp,ci95_pp,board_clusters,samples,resamples,bootstrap_seed}`. Resample board giữ cả cụm, tỷ lệ trên tổng số state được lấy, quantile linear2.5/97.5%; RB chuyển hai binary corner rows thành cùng schema. Trọng sốmicro với kích thước cluster khác nhau được giữ.

`report.json`: status/complete/mode, endpoints/counts/summary/comparisons/conditions/limitations, registration_digest/model_manifest_digest; display rounding tách khỏi gate. `complete=true` chỉ khi đủ model diagnostics và RB raw ledger; package khi chưa đủ được phép xuất INCOMPLETE và ZIP hiện có, không chạy việc còn thiếu. CLI exit 0 khi stage hoàn tất dù kết quả khoa học không đạt, exit 1 khi invalid/drift, exit 130 khi KeyboardInterrupt; không có subcommand test.

## Task 0: Khóa acceptance độc lập trước production

**Files:** Create `docs/superpowers/requirements/2026-10-23-s2-kl-retention.md`, `docs/superpowers/test-matrices/2026-10-23-s2-kl-retention.md`, bốn test file trong map.

**Interfaces:** Consumes spec/public contracts; produces requirements packet/hash, matrix, executable acceptance và RED evidence. Không tạo production skeleton để làm test pass.

- [ ] **Step 1 — chuẩn bị baseline và packet:** xác minh checkout/input fixture thực tế, ghi source/test baseline và `git status`; packet chỉ chứa hành vi/giá trị/public contract/artifact/non-goals, không internal plan/code. Fixture copy đúng S2, metadata SD8A, sample manifest và shard cần thiết có SHA; không copy/junction mutable cả runs hoặc tạo RED do thiếu fixture. Chạy baseline root `venv\Scripts\python.exe -m unittest discover -p "test_*.py" -q`, lưu tên lỗi hiện hữu; không giả định 20 error/1 failure trước đây vẫn đúng.
- [ ] **Step 2 — tác giả độc lập:** theo independent-test-first, giao agent context mới chỉ packet/current code/conventions/commands; không gửi implementation plan/history. Agent viết matrix rồi acceptance cho các task và Review Focus. Implementer không tự thay acceptance để xanh.
- [ ] **Step 3 — RED hợp lệ:** chạy `venv\Scripts\python.exe -m unittest test_s2_kl_protocol test_s2_kl_training test_s2_kl_evaluation test_run_s2_kl_probe -q`; thiếu public behavior phải thành assertion rõ ràng, không phải syntax/dependency/fixture failures. Behavior đã có ở trainer được ghi regression, không giả làm RED.
- [ ] **Step 4 — đóng băng:** commit packet/matrix/tests và SHA từng file; các task sau dùng nguyên suite. Việc sửa test chỉ do tác giả/reviewer chứng minh sai requirements, giữ lý do và phiên bản cũ. Commit `test: freeze S2 KL acceptance contracts`.

## Task 1: Protocol, inputs, seed audit và registration

**Files:** Create `s2_kl_protocol.py`, register/public wrapper của `run_s2_kl_probe.py`; Test `test_s2_kl_protocol.py`, CLI registration trong `test_run_s2_kl_probe.py`.

**Interfaces:** Consumes stdlib/shared helpers và locked inputs; produces config/seed_manifest/register/verify_run theo contract, source snapshot và registry entry mới. Mọi scalar/hash trong Global Constraints là input đã chốt.

- [ ] **Step 1 — RED requirements:** kiểm hai arm chỉ khác beta; S2/13 channels và manifest/shard SHA đúng; đủ source import gián tiếp/động; từ chối CUDA thiếu, metadata không rõ, giao dải qua offset hoặc UI root; kiểm crash lúc đăng ký và hai writer cùng registry. Assertion tiêu biểu:
  ```python
  self.assertEqual(config_for()['anchor_coef'], {'control': 0.0, 'anchor': 1.0})
  self.assertEqual(seed_manifest()['ranges']['train'], {'start':710000,'end_inclusive':710249})
  with self.assertRaisesRegex(ValueError, 'Seed overlap'):
      register(output, inputs_with_ui_board_overlap, device='cpu')
  ```
- [ ] **Step 2 — chạy RED:** `venv\Scripts\python.exe -m unittest test_s2_kl_protocol -q`, ghi các assertion thiếu behavior của T1; không mở seed thật.
- [ ] **Step 3 — implement config/verify:** lưu S2 trong `checkpoints.S2` để trainer dùng provenance hiện có; copy source/sample manifest vào output, khóa collection shards bằng SHA; kiểm live source, snapshot, spec, runtime, inputs và registry. Scan `inventory_seeds(input_root)` và `inventory_seeds(input_root/'game UI')`, gắn prefix nguồn để không trùng path; phân biệt training RNG có phase/episodes với board seed. Metadata không rõ vẫn phải từ chối.
- [ ] **Step 4 — implement registration atomic:** khóa output rồi shared registry, lưu identity pending trước copy, publish atomic và COMPLETE; phục hồi mọi cửa sổ crash mà không reset artifact/reservation của lượt khác. Smoke chỉ audit dải 790000–790003, không ghi reservation full 710000/720000. Task này chỉ hoàn thiện CLI register; stage còn thiếu trả lỗi rõ.
- [ ] **Step 5 — GREEN và commit:** chạy lại suite T1, xác minh registry/checkpoint gốc không đổi; fixture hoạt động dưới root tạm. Commit `feat: register frozen S2 KL experiments`.

## Task 2: Train context, KL reference và GPU/full-state resume

**Files:** Create `s2_kl_training.py`; Test `test_s2_kl_training.py`.

**Interfaces:** Consumes registration T1 và `train_actor_critic.train`; produces train_arm/schedule/candidates và context `S2KLRun(output,arm,registration,commit_hook=None)` với `prepare_episode(ep,env,device)`, `snapshot_completed(ep,net)`, `record_episode(row)`, `state_dict()`, `load_state_dict(state)`, `after_commit(ep)`.

- [ ] **Step 1 — RED requirements:** lịch đúng 500 episode/góc/RB; cùng weights S2, chỉ beta khác; reference S2 đông cứng, ngoài optimizer; KL đúng mask, mean và chiều; control beta 0 giữ hành vi cũ. Kiểm crash trước state 0, partial episode, trước/sau commit, mất state sau artifact committed; reference sau resume vẫn là S2. Bật/tắt diagnostics phải cho model/Adam/RNG khớp, không chỉ loss giống nhau.
- [ ] **Step 2 — chạy RED:** `venv\Scripts\python.exe -m unittest test_s2_kl_training -q`; toán KL đã pass trên trainer hiện có là regression; context/resume mới phải có RED.
- [ ] **Step 3 — implement context:** prepare_episode đặt `env.rng` trước đúng một reset của trainer, trả learning_aid/opponents `{other:'rule'}`/audit. Context lưu arm/registration/S2/beta/cursor/candidates/metrics; candidate 0 sao đúng bytes S2, candidate sau ghi weights CPU atomic. record_episode gắn board/corner/optimizer_steps; after_commit xuất JSONL từ state đã commit.
- [ ] **Step 4 — implement train_arm:** khóa output và dùng T1 verify; kwargs đúng config, save_training_state=true, resume_from=S2 và phase_d_run=context, format `bomberman_phase_d_training_v1`. Tái dựng metric/candidate projection từ state khi resume; từ chối artifact committed thiếu state. Completion chứa episode/registration_digest/state_sha256; candidates_for_run kiểm đủ cả hai arm và SHA trước khi công bố endpoint. Giữ nguyên trainer/anchor math.
- [ ] **Step 5 — GREEN CPU/CUDA và commit:** cả hai arm chạy 4 episode/max20, so liên tục với dừng/resume bằng smoke/callback: model/Adam/scheduler/RNG/context, reference hash, metric không lặp. Định nghĩa opt-in `S2_KL_ACCEPTANCE_CUDA=1`; chạy `set S2_KL_ACCEPTANCE_CUDA=1` rồi `venv\Scripts\python.exe -m unittest test_s2_kl_training -q` trong CMD. Thiếu GPU không đánh dấu CUDA-ready. Commit `feat: train and resume S2 KL arms`.

## Task 3: Offline retention, RB tasks và decision math

**Files:** Create `s2_kl_evaluation.py`; Test `test_s2_kl_evaluation.py`.

**Interfaces:** Consumes manifest T1/model descriptors T2; produces load_samples/decision_targets/diagnose_model/validation_tasks/cluster_comparison/decide_gate/build_report theo bảng public contracts.

- [ ] **Step 1 — RED deterministic fixtures:** kiểm nhãn legal maxima đồng hạng, masked NaN/illegal, all-tie, sai row/shape/shard/held-out board; load đúng sample đã khóa. Cluster không đều phải giữ micro: board A có 5 state delta+1, board B có 1 state delta−1 thì point delta là 66.6667 pp. Thiếu/trùng ID, board khác giữa hai phía, RB thiếu góc hoặc nonfinite phải bị từ chối. Kiểm sáu điều kiện, thứ tự status, endpoint500, SMOKE_ONLY, biên 5/10/2/1 pp và CI0/nextafter.
  ```python
  self.assertAlmostEqual(cluster_comparison(left, right)['difference_pp'], 100*4/6)
  self.assertFalse(gate_with_anchor_gain_ci_lower(0.0)['conditions']['ANCHOR_BENEFIT'])
  self.assertEqual(no_control_drift_report()['status'], 'INCONCLUSIVE_NO_DRIFT')
  ```
  Hai helper fixture trong assertions cuối do test author định nghĩa bằng public decide_gate, không thêm production API.
- [ ] **Step 2 — chạy RED:** `venv\Scripts\python.exe -m unittest test_s2_kl_evaluation -q`; đối chiếu giá trị tính tay và bootstrap seed, không dùng code dự định làm oracle.
- [ ] **Step 3 — implement offline bất biến:** đọc từng NPZ shard, chỉ lấy row trong manifest, materialize đúng 1024+1024 quan sát; CPU batch128/inference_mode, không env/optimizer/search. Lưu raw từng state và summaries/KL ghi rõ chiều; kiểm sample identity/SHA/count. Load model strict bằng helper đã kiểm; đo lại baseline, không hardcode 97.36/91.83.
- [ ] **Step 4 — implement RB/CI/gates:** task ID là canonical digest của stage/modelSHA/RBSHA/board; corners0/1 và max_steps theo registration, dedup SHA giữ alias. Tái dùng evaluate_board/summarize; CI giữ micro theo state và mean hai góc RB, số liệu phải finite. Chỉ endpoint500 quyết định; xuất đủ conditions theo thứ tự spec. build_report tính R/Q/RBAW/SU từ raw; diagnostics100/250 không đổi endpoint.
- [ ] **Step 5 — GREEN và commit:** chạy lại T3; forward fixture xác nhận weights/RNG không bị cập nhật, synthetic report không có PHASE_D_PASS. Commit `feat: measure S2 retention and pre-registered gates`.

## Task 4: Atomic stages, frozen CLI và package có provenance

**Files:** Create `s2_kl_ledger.py`; Complete `run_s2_kl_probe.py`; Test `test_run_s2_kl_probe.py`, bổ sung unit hồi phục riêng nếu tìm lỗi integration.

**Interfaces:** Consumes verification T1/candidates T2/raw metrics T3; produces diagnose_run/validate_run/package_run/main và model_manifest bất biến. `_frozen_job(task,destination)` là worker entry trong snapshot, validation Pool spawn dùng progress parent S2-KL.

- [ ] **Step 1 — RED journey/fault cases:** kiểm hai arm hoàn tất→diagnose→validate→package; resume diagnostic/board đã ngắt; projection/model manifest sai, shard từ lượt khác, thiếu/trùng góc, hai writer và thiếu state. Từ chối report PASS không có raw; thiếu endpoint500 không chọn100/250. Package incomplete không tự chạy model/game. CLI `test` phải bị argparse từ chối.
- [ ] **Step 2 — chạy RED:** `venv\Scripts\python.exe -m unittest test_run_s2_kl_probe -q`; fake evaluator chỉ serial trong test, không có production flag bypass.
- [ ] **Step 3 — implement ledgers:** model_manifest lấy từ completed states; stage khóa identity trước khi ghi shard. Diagnostic/modelSHA và board/taskID có digest/schema. Resume tái dùng shard hợp lệ, replay shard dở; từ chối source/runtime drift trước job mới. Diagnose/validate cần cả hai arm hoàn tất; package còn thiếu xuất INCOMPLETE. Projection đã có khác kết quả tính lại phải bị từ chối.
- [ ] **Step 4 — complete frozen CLI/package:** parser có register/train/diagnose/validate/package và --resume đúng stage; verify live trước subprocess snapshot, worker thread/runtime đúng registration. Exit code/progress đúng contract; ZIP dùng temp/atomic replace và kiểm symlink/path closure. Chỉ gói output mới, không collection NPZ/lượt cũ; không chọn best hoặc tạo test-open marker.
- [ ] **Step 5 — GREEN và commit:** bốn file acceptance T0 phải qua và giữ SHA; synthetic cases kiểm mọi status qua CLI/report. Commit `feat: ship resumable S2 KL stages and packaging`.

## Task 5: Integration, regression, smoke và hướng dẫn bàn giao

**Files:** Create `S2_KL_GUIDE.md`, `docs/superpowers/test-matrices/2026-10-23-s2-kl-verification.md` vàJSON; cập nhật journals/trackers/module map khi file thật tồn tại. Code fix nếu cần qua test hẹp mới, không đổi acceptance.

**Interfaces:** Consumes tấtcảstage CLI/evidence; produces bằng chứng code-ready, hướng dẫn CMD và ZIP smoke dùng chẩn đoán; chưa tạo/reserve full run.

- [ ] **Step 1 — integration evidence:** giữ RED/GREEN T0–T4 và acceptance hashes; chạy suite S2-KL cùng legacy `test_training_state test_phase_d_s2_protocol test_phase_d_s2_training test_phase_d_s2_evaluation test_run_phase_d_s2 test_run_sd8` bằng unittest. Legacy CUDA dùng cờ D_S2_ACCEPTANCE_CUDA, không nhầm S2_KL_ACCEPTANCE_CUDA. Không mở lại held-out qua test fixture.
- [ ] **Step 2 — full regression một lần:** `venv\Scripts\python.exe -m unittest discover -p "test_*.py" -q`; so tên lỗi với baseline mới T0, phân biệt lỗi hiện hữu và mới phát sinh. Không báo toàn suite PASS khi còn fail hoặc sửa lỗi lịch sử ngoài scope.
- [ ] **Step 3 — actual smoke:** output mới `runs/s2-kl-code-smoke-20261023-cpu` và `...-cuda`: register --smoke → train control → train anchor → diagnose → validate → package. Dải smoke audit-only, report SMOKE_ONLY, raw count khớp, worker frozen/1 thread; CPU+CUDA resume cả hai arm trên source cuối. Kiểm interruption và CON progress khi redirect; chưa launch full job.
- [ ] **Step 4 — perf/guide:** lưu wall time/peak RAM từng stage và worker; dự phóng riêng train/diagnose/validate, nêu giới hạn smoke20 bước, không hứa ETA full. GUIDE dùng Python/path CMD tuyệt đối, log append/errorlevel, resume cùng output/máy, stage order/status và input chỉ đọc. Launcher full chỉ tạo khi user yêu cầu start, tránh chạy full khi test guide.
- [ ] **Step 5 — review/commit/handoff:** `git diff --check`, xác minh SHA D-S2/SD8A/registry gốc không đổi; reviewer độc lập cuối theo Native workflow. Sửa finding có evidence/regression và chạy lại check bị ảnh hưởng. Chỉ commit file mới/đúng scope, giữ README/UI/untracked PDE của user. Bàn giao code-ready và CMD; chưa đặt chỗ 710000/720000 hoặc mở terminal train.

## Coverage và execution handoff

| Spec | Task |
|---|---|
| §§1–3 câu hỏi, inputs, arm/KL/reference | T0/T1/T2 |
| §4 config/schedule/diagnostics | T1/T2/T5 |
| §§5–6 seed/registry/runtime/frozen/resume | T1/T2/T4/T5 |
| §7 offline khóa trước/ties/retention/clusterCI | T1/T3/T4 |
| §§8–9 RB ledger/endpoint500/gates/limits | T3/T4/T5 |
| §§10–11 stages/package/acceptance/approval | T0/T4/T5 vàGlobal Constraints |

Dependencies: T0 → T1 → T2 → T3 → T4 → T5. Tác giả acceptance độc lập và reviewer cuối chỉ được dispatch khi bắt đầu execution đã duyệt, không trong lượt viết plan. Review không thay requirements/tests để làm suite xanh.

Self-review: đủ spec§§1–11, năm Review Focus đều có owner/test, public function/schema/seed/endpoint nhất quán; smoke chỉ SMOKE_ONLY, registry lock và reference sau resume đã chốt. Không thêm biến can thiệp hoặc mở test. Native được giữ theo lựa chọn trước; cần user review plan trước implementation. Sau đó dùng executing-plans, independent-test-first/TDD và review theo workflow; pilot full là bước user yêu cầu start riêng sau code-ready.
