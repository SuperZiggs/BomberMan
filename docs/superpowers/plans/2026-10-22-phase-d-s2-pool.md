# D-S2 Controlled Opponent Pool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. The user must review this plan and choose the execution method first. Independent acceptance-test authorship remains required with either method.

**Goal:** Tạo runner D-S2 có hai nhánh đối chứng/pool, resume đầy đủ và đánh giá chọn model theo CNN, đúng spec đã duyệt; bàn giao lệnh CMD trước lượt chạy thật.

**Architecture:** Giữ training math trong `train_actor_critic.train`; thêm một context tùy chọn chỉ dành cho D-S2 để điều khiển seed/góc/đối thủ, snapshot và state. Tách protocol/provenance, điều khiển train và đánh giá thành ba module; CLI chỉ điều phối các stage trên source đã đóng băng. Không sửa evaluator PDE/PDE2 hoặc sao chép training loop.

**Tech Stack:** Python hiện có, PyTorch, NumPy, unittest, pathlib/json/hashlib/ast/subprocess/zipfile, `progress.py`; không thêm dependency.

**Spec:** `docs/superpowers/specs/2026-10-22-phase-d-s2-pool-design.md` — người dùng duyệt ngày 22/10/2026 sau commit tài liệu `2b16eb5`.

**Status:** IMPLEMENTED_VERIFIED — người dùng duyệt Native ngày 22/10/2026; acceptance độc lập, CPU/CUDA resume, smoke, review và fix pass đã hoàn tất. Full suite535test giữ20error/1failure cùng baseline; không có lỗi mới. Chưa đặt chỗ hoặc mở experiment full. Bằng chứng: `docs/superpowers/test-matrices/2026-10-22-phase-d-s2-verification.md`; bàn giao `PHASE_D_S2_GUIDE.md`.

## Global Constraints

- Hai nhánh `control` và `pool`, mỗi nhánh 2.000 episode, cùng S2 SHA trong spec §3; Adam/scheduler mới, cùng seed RNG 2026102201, cùng máy/runtime; train tuần tự, evaluation CPU.
- 9×9, 2 agent, max_steps 500; lịch E 475/25/40, sudden_death_probability=1.0; sampled policy qua mask chung; opponents frozen/eval/no_grad.
- Gamma 0.99, lr 0.000025, value_coef 0.5, entropy_coef 0.0075; cosine T_max=2000, eta_min=lr×0.05; giữ lịch entropy hiện có.
- `attack_teacher=local`, `teacher_version=v2`; attack_lookahead=2, min_pressure_gain=0.15, trap_sequence_depth=2, sequence_pressure_gain=0.35; escape/combat/navigation/ring coefficients=0.5/0.9/0.4/0.6; late_game_start=350, late_game_attack_multiplier=1.75; wall_destroy_bonus=0.15.
- Objective AC, MC, normalize_advantage=False; search_teacher=False, group_rl=False, anchor_coef=0; không đổi engine/RB/reward/observation/mask/network hoặc training math.
- `control` chỉ RB; `pool` block 20 episode với 10 cặp: RB/G/F/S/S2/RB/recent/recent/recent/recent; 400/800/800 episode RB/fixed/recent. Cùng board và opponent trong hai góc của một cặp; recent chỉ lấy hai snapshot pool đã commit trước block, fallback S2.
- Candidate episodes 0/500/1000/1500/2000; snapshots mỗi 200; validation nội bộ và báo gate cũ tắt trong D-S2. B chỉ được đánh giá sau khi test mở; chỉ B được zero-pad 12→13 channels, mọi shape khác phải đúng 9×9.
- Dải board full: train 610000–610999, validation 620000–620039, test reference 630000–630099, test old snapshots 640000–640099. Smoke riêng 690000–690003; không ghi global full registry hoặc mở seed full khi unit/smoke.
- Validation 80 trận/model/đối thủ G/F/S/S2/RB; test 200 trận/cặp của S2/control/pool với G/F/S/S2/B/RB; thêm pool-vs-mọi snapshot cũ hơn selected episode. Budget evaluation tối đa 3600+3600+1800=9000 trận; dedupe SHA giữ alias.
- Selection/guard/open-test/pilot/Phase-D gates giữ nguyên spec §§7–9; các sai số số học dùng 1e-12 cho tie-break; gate CI lower > 0 là strict. Macro CNN chỉ G/F/S/S2; bootstrap theo board 10000 lần, seed 20261014.
- Registry kiểm cả khoảng, source closure gồm import gián tiếp/untracked, worker chạy snapshot source; state/shard atomic, một writer, refuse drift. S2 và mọi output PDE/SD8/Phase D cũ là input chỉ đọc.
- Số liệu seed cũ không thay baseline mới; không mở lại test cũ để hồi quy. Không train full hoặc mở test full trong implementation nếu người dùng mới chỉ duyệt code/plan.

## Review Focus

| Failure mode | Hành vi phải được kiểm | Task |
|---|---|---|
| Metadata lịch sử thiếu, seed gốc khác nhưng offset giao nhau hoặc registration bị ngắt | Refuse giao/unknown; không công bố register xong khi output chưa đủ; reservation dang dở giữ fail-closed và chỉ resume đúng transaction | 1 |
| Crash sau ghi snapshot hoặc giữa hai góc | State cũ vẫn authoritative; không dùng snapshot orphan, đổi đối thủ giữa cặp hoặc nhân đôi metric | 2–3 |
| S2-vs-S2, các alias cùng SHA và B 12 channels | Giữ vai learner/góc; evaluate một lần/SHA; chỉ B được pad, không nội suy bàn | 4 |
| Model tốt trước RB nhưng yếu trước CNN; bảng đối thủ thiếu | Selection dựa macro CNN đủ bốn đối thủ; fallback episode 0; không mở test khi thiếu/không đạt validation | 4–5 |
| Source import gián tiếp đổi, selected bị sửa sau test-open, hoặc writer thứ hai | Refuse trước khi tạo trận mới; ledger sai không được reuse; summary không được coi là complete | 1, 5 |

## File structure và contracts

| File | Trách nhiệm |
|---|---|
| Create `phase_d_s2_protocol.py` | Config chuẩn, checkpoint/seed audit, registration/source snapshot, atomic JSON và xác minh run |
| Create `phase_d_s2_training.py` | `PhaseDRun`, lịch đối thủ, snapshot/candidate manifests, metric projection và wrapper train |
| Modify `train_actor_critic.py` | Một keyword `phase_d_run=None`, tích hợp các hook và full-state D có guard; đường cũ giữ nguyên |
| Modify `training_state.py` | Load định dạng D theo opt-in; format C mặc định không thay |
| Create `phase_d_s2_evaluation.py` | Board tasks, strict weights, raw metrics, selection/CI/gates và test manifests |
| Create `run_phase_d_s2.py` | CLI, stage prerequisites, frozen-source bootstrap, ledger/progress, package |
| Create `test_phase_d_s2_protocol.py`, `test_phase_d_s2_training.py`, `test_phase_d_s2_evaluation.py`, `test_run_phase_d_s2.py` | Acceptance do tác giả test độc lập viết trước production; bổ sung unit tests cùng file sau nếu cần |
| Create `PHASE_D_S2_GUIDE.md` | Lệnh CMD, resume, thời gian/RAM đo ở smoke và diễn giải trạng thái |
| Update `CLAUDE.md`, `RESULTS.md`, `IMPROVEMENTS.md`, `PLAN_SUDDEN_DEATH.md` | Map module và bằng chứng/tiến độ; chỉ cập nhật map sau khi module thật tồn tại |

### Public CLI contract để đóng băng vào requirements packet

- Stages: `register`, `train --arm control|pool`, `validate`, `test`, `package`; mọi stage nhận `--output PATH`.
- `register`: `--input-root PATH` mặc định repo root, `--device cpu|cuda|cuda:0`, `--workers 1..5` mặc định 2, `--smoke`, `--resume` cho transaction registration bị ngắt. Checkpoint/lịch sử trong input root chỉ đọc; ngoại lệ duy nhất là registry mới `<input-root>/runs/phase_d_s2_seed_registry.json`, dùng chung giữa các checkout. Không tự chép artifact cũ vào Git.
- `train`: `--arm` bắt buộc, `--resume` nếu tiếp tục. Device/config lấy từ registration, không có override episodes/LR/seed của run đã khóa.
- `validate` và `test`: `--resume` tiếp tục cùng ledger/manifest; evaluation luôn CPU. `package` tạo ZIP/status hiện có, không tự chạy các stage còn thiếu.
- Stage thành công hoặc kết quả khoa học không đạt gate: exit 0 kèm status chính xác. Input/provenance/ledger sai, stage trái thứ tự, mở test mới lần hai hoặc run incomplete ở stage đòi complete: exit khác 0, không phát sinh trận mới.
- `--resume` của test với cùng selected/manifest là tiếp tục một lần mở; đã xong thì verify/reuse và trả report cũ. `test` không có `--resume` sau khi test-open phải bị từ chối.
- Không có `--force-test`, `--ignore-hash`, `--reset-seed`, tự tăng episode hoặc cho phép đổi selection sau test-open.
- Smoke dùng namespace/output riêng: 4 episode/arm, max_steps=20, candidate 0/2/4, snapshot_every=2; train boards 690000–690001, validation 690002, test 690003. Không công bố PASS full từ smoke. Đường test thành công được kiểm bằng fixture có gate biết trước; smoke thực có thể kết thúc VALIDATION_STOP và vẫn package đầy đủ.

### Public artifact schemas để đưa vào requirements packet

- JSON dùng UTF-8, reject NaN/Infinity. Digest metadata = SHA-256 của JSON `sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False`; bỏ trường digest của chính object khi tính. Hash source/checkpoint là raw bytes; không tự chuẩn hóa LF/CRLF trong lúc verify.
- `registration.json`: `format='d-s2-registration-v1'`, `experiment_id`, `mode='full'|'smoke'`, `input_root`, `config`, `checkpoints`, `sources`, `spec_sha256`, `runtime`, `git`, `seed_manifest`, `registration_digest`. `config` có `train` (kwargs scalar trainer), `pool_slots`, `candidate_episodes`, `validation_games`, `test_games`, `workers`; paths/device/arm được resolve từ registration, không nhận override ngầm.
- Checkpoint descriptor: `id`, `checkpoint`, `sha256`, `episode` (null cho mốc lịch sử), `role`. `checkpoints` ánh xạ S2/G/F/S/B vào descriptor. RB descriptor có checkpoint=null và SHA source RB đã đăng ký; không giả lập nó thành file weights.
- `seed_manifest.json`: `mode`, `ranges` với keys `train`, `validation`, `test_reference`, `test_snapshots`; mỗi khoảng có `start`, `end_inclusive`; `inventory_digest`. Full ranges đúng Global Constraints, smoke test_reference/test_snapshots cùng dùng 690003 trong namespace smoke.
- `selected.json`: `format='d-s2-selection-v1'`, `registration_digest`, `models` (S2/control/pool), `validation_summary`, `open_test`, `test_manifest`, `selection_digest`. Test manifest chỉ chứa identity/model/opponent/board/corners, chưa gắn selection_digest nên không tạo vòng hash. Không có selected hoàn chỉnh trước khi mọi validation task đủ.
- Task descriptor: `id`, `stage`, `learner`, `opponent`, `board`, `corners=[0,1]`, `max_steps`, `registration_digest`, `selection_digest` (null ở validation, được gắn sau khi selection khóa ở test). ID hash identity stage/learner SHA/opponent ID+SHA/board, không phụ thuộc alias learner; giữ alias map để tái sử dụng SHA mà không lẫn vai.
- Task payload: `format='d-s2-board-result-v1'`, `id`, `registration_digest`, `selection_digest`, `records`. Đúng hai records, mỗi record gồm `learner_sha256`, `opponent_id`, `opponent_sha256`, `board`, `corner`, metric keys spec §9 và `opponent_outcome` để kiểm winner/draw. Metric lưu tên singular `win`, `active_win`, `draw`, `passive_win`, `sudden_death_win`, `suicide`, `sudden_death_death`, `kills`, `bombs`, `steps`; aggregate dùng phần trăm.
- `test_opened.json`: write-once `registration_digest`, `selection_digest`, `test_manifest_digest`, thời điểm mở. `report.json` luôn có `status`, `complete`, `actual_games`, `expected_games`, `summary`, `comparisons`, `pilot_gate`, `phase_d_gate`, `limitations`; không đặt complete=true khi thiếu task hoặc công bố gate full từ smoke.

### Interfaces giữa các task

Không có type ở module này được định nghĩa lại ở module khác; checkpoint/task descriptor là dict JSON-compatible, chứa SHA và role rõ ràng.

| Module | Interface tạo ra |
|---|---|
| protocol | `canonical_config(smoke: bool = False) -> dict`; `inventory_seeds(input_root: Path) -> dict`; `register(output: Path, input_root: Path, device: str, workers: int, smoke: bool = False, resume: bool = False) -> dict`; `verify_run(output: Path) -> dict`; `atomic_json(path: Path, payload: dict, *, overwrite: bool = False) -> None` |
| training | `PhaseDRun(output: Path, arm: str, registration: dict)`; `prepare_episode(ep: int, env, device: str) -> dict`; `snapshot_completed(ep: int, net) -> None`; `record_episode(row: dict) -> dict`; `state_dict() -> dict`; `load_state_dict(state: dict) -> None`; `after_commit(ep: int) -> None`; `train_arm(output: Path, arm: str, resume: bool = False) -> None` |
| training_state | `load_training_state(path, *, expected_format: str = 'bomberman_phase_c_training_v1') -> dict`; format D = `bomberman_phase_d_training_v1` |
| evaluation | `validation_tasks(output: Path) -> list[dict]`; `test_tasks(output: Path, selection: dict) -> list[dict]`; `evaluate_board(task: dict) -> dict`; `validate_task(task: dict, payload: dict) -> list[dict]`; `select_models(rows: list[dict], candidates: list[dict]) -> dict`; `macro_board_bootstrap(left: list[dict], right: list[dict], opponents: tuple[str, ...]) -> dict`; `build_report(rows: list[dict], selection: dict) -> dict` |
| CLI | `main(argv: list[str] | None = None) -> int`; `run_tasks(output: Path, stage: str, tasks: list[dict], resume: bool) -> list[dict]`; `package_run(output: Path) -> Path` |

`prepare_episode` đặt `env.rng` theo seed board **trước duy nhất một lần `env.reset()` do trainer gọi**, trả `learning_aid`, `opponents` và `audit`; opponents dùng `"rule"` hoặc `SnapshotOpponent` hiện có. Tránh import vòng: trainer không import module D-S2 ở đường mặc định; context được runner truyền vào. `record_episode` trả row đã gắn audit và giữ nó trong context trước state commit; `after_commit` xuất projection metric/manifest từ state đã commit.

## Task 0: Acceptance độc lập trước production edits

**Files:** create `docs/superpowers/requirements/2026-10-22-phase-d-s2.md`, `docs/superpowers/test-matrices/2026-10-22-phase-d-s2.md`, bốn test file trong file map.

**Consumes:** written spec đã duyệt, CLI contract phía trên, source hiện tại và quy ước unittest. **Produces:** requirements packet chỉ chứa hành vi/giá trị/ngưỡng/CLI/artifacts/non-goals, test matrix, acceptance suite và RED evidence.

- [x] Trước RED, kiểm checkout/worktree và fixture dependencies thực tế. Nếu checkout cách ly thiếu source/test PDE đang untracked hoặc fixture lịch sử cần cho regression, chỉ sao chép đúng dependency đã đọc với SHA ghi lại, inputs giữ read-only; không chép cả `runs/`, tạo junction mutable chung, tự stage artifact cũ hoặc dùng lỗi thiếu fixture làm RED.
- [x] Đóng băng requirements packet từ spec + public CLI/artifact schemas; ghi hash và approval reference. Không đưa các interface helper nội bộ, thiết kế hook hoặc thuật toán dự định trong plan vào packet.
- [x] Dùng `independent-test-first`: giao tác giả test riêng với context mới, chỉ packet + code hiện tại + test conventions; không truyền lịch sử chat, implementation plan hoặc code dự định. Native implementation vẫn giữ bước này.
- [x] Tác giả test ánh xạ từng mục acceptance spec §11 và năm Review Focus vào test tự động, gồm lỗi/boundary/integration. Mỗi task phía dưới phải có RED tương ứng trước khi implement; thiếu tầng test cần nêu residual risk và xử lý theo workflow đã duyệt.
- [x] Run `venv\Scripts\python.exe -m unittest test_phase_d_s2_protocol test_phase_d_s2_training test_phase_d_s2_evaluation test_run_phase_d_s2 -q`; lưu lỗi do thiếu behavior, không chấp nhận lỗi syntax/dependency/fixture làm RED. Không chạy full seed.
- [x] Lưu nguyên suite/matrix/RED evidence, commit riêng. Implementer không giảm assertion, xóa hoặc viết lại acceptance để làm xanh; chỉ tác giả test xác minh và sửa test thật sự sai với reason/version lưu lại.

## Task 1: Registration, seed audit và source bất biến

**Files:** create `phase_d_s2_protocol.py`; test `test_phase_d_s2_protocol.py`.

**Consumes:** spec §§3–6 và CLI contract. **Produces:** năm interface protocol trong bảng; `registration.json`, `seed_manifest.json`, `runtime.json`, `sources/`, registry full `runs/phase_d_s2_seed_registry.json`.

- [x] RED: kiểm overlap chỉ xuất hiện sau offset, unknown legacy metadata, hai register đồng thời, failure trước atomic replace hoặc sau reserve/trước publish output, checkpoint/hash/shape/source import gián tiếp sai. Assertions: exit khác 0 khi chưa complete; không có completed registration một phần; metadata đã commit không đổi; pending reservation giữ chỗ và chỉ đúng register --resume mới hoàn tất được.
- [x] Implement config theo Global Constraints. Inventory đọc schema metadata/raw-board/JSONL được nhận diện, ghi nguồn và khoảng/IDs; phân biệt RNG seed với board seed, gom intervals. Với file lớn đọc theo record/chunk phù hợp schema, không nạp toàn bộ lịch sử replay vào RAM. Schema/seed mapping chưa xác minh trả `unresolved`; không tự giả định an toàn. Full register chỉ tiếp tục khi hết unresolved và không giao khoảng.
- [x] Kiểm SHA/checkpoint 9×9 từ bảng spec; hash B được xác minh lúc register nhưng không load policy B vào train/validation. Chuẩn bị source/metadata trong thư mục sibling tạm gắn transaction ID; lock registry dùng chung, kiểm lại inventory/overlap và atomic-reserve bốn dải với output path/registration digest. Chỉ publish output bằng atomic rename khi file đã đủ. Crash sau reserve giữ reservation PENDING, không cho experiment khác dùng; register --resume chỉ loại khỏi overlap đúng reservation cùng transaction/digest, xác minh temp hoặc output đã publish/config/runtime rồi hoàn tất publish và đánh dấu COMPLETE. Stage train đòi cả output lẫn reservation COMPLETE; full register mới không chấp nhận output có sẵn. Smoke chỉ local transaction/manifest trong output riêng, không ghi global registry.
- [x] Snapshot source closure: các module mới, trainer/state/env/RB/evaluate/teachers/progress/kill_search/group_rl và mọi import local gián tiếp. Dùng AST inventory local imports, bao gồm import trong function; explicit-manifest cho dynamic imports. Copy raw bytes rồi xác minh lại hash; thiếu local dependency thì refuse. Lưu spec snapshot/hash; journal không thuộc source closure.
- [x] Register runtime trước sampling; cấu hình runner deterministic, cuDNN benchmark=False, TF32=False, threads=1; CUDA profile dùng CUBLAS_WORKSPACE_CONFIG=:4096:8 trước torch import. Ghi đúng flags/device/phần cứng; unavailable CUDA báo lỗi trước reserve. `verify_run` so config/spec/source/model/runtime, không sửa hash. Git HEAD/dirty inventory là evidence; journal commit không đổi source closure không tự làm invalid run.
- [x] GREEN: chạy `venv\Scripts\python.exe -m unittest test_phase_d_s2_protocol -q`, xác nhận mọi test task này qua và suites còn thiếu behavior vẫn RED hợp lệ. Cập nhật RESULTS/IMPROVEMENTS/tracker; commit protocol và evidence.

## Task 2: Lịch pool, snapshot/candidate và context có thể khôi phục

**Files:** create `phase_d_s2_training.py`; test `test_phase_d_s2_training.py`.

**Consumes:** registration xác minh từ Task 1, `SnapshotOpponent` hiện có. **Produces:** `PhaseDRun` và các method trong bảng interface; chưa nhân bản training loop.

- [x] RED: kiểm 2.000 episode cho tỷ lệ 400 RB/800 fixed/800 recent, 200 episode từng G/F/S/S2; corners 0/1 dùng cùng board/opponent; cursor không đổi giữa cặp. Kiểm recent tại block mới sau ep200, manifest mất/SHA sai, snapshot của control hoặc tương lai, orphan sau crash, tensor opponent bị cập nhật.
- [x] Implement `prepare_episode`: board 610000+(ep−1)//2, corner=(ep−1)%2, slot=((ep−1)//2)%10; block=(ep−1)//20. Đóng băng recent membership tại đầu block, round-robin qua hai snapshot pool mới nhất, cursor và lựa chọn cặp được lưu; fallback S2 khi rỗng. Dùng manifest, không glob toàn bộ file có trên đĩa để quyết định pool.
- [x] Snapshot/candidate ghi atomic weights CPU sau completed episode: snapshot mỗi 200, candidate mỗi 500 và ep0. Manifest có episode/path/SHA/role; không xuất `best_phaseD.pt` trước validation. Opponent net eval + requires_grad=False; chỉ model learner đi vào optimizer. Chưa restore/publish candidate/snapshot chưa có trong committed state.
- [x] `record_episode` thêm seed/corner/opponent SHA/block/slot/learner steps/optimizer steps, giữ authoritative rows trong context. `state_dict` chỉ JSON-compatible metadata/rows/manifest, không pickle object policy. `load_state_dict` xác minh mọi referenced artifact trước materialize opponents; `after_commit` atomic rewrite projection metrics/manifests để sửa torn logs, không nhân đôi hàng.
- [x] GREEN: chạy `venv\Scripts\python.exe -m unittest test_phase_d_s2_training -q` theo nhóm context đã được tác giả test tách; các bài resume tích hợp chưa implement vẫn phải ghi RED riêng. Cập nhật nhật ký và commit context.

## Task 3: Tích hợp trainer và full-state Phase D

**Files:** modify `train_actor_critic.py:691` (signature/guard, resume/episode/snapshot/commit/final branches), `training_state.py:53`; test `test_phase_d_s2_training.py`, regression `test_training_state.py`.

**Consumes:** PhaseDRun Task 2, training math hiện tại. **Produces:** `train(..., phase_d_run=None)` và `load_training_state(..., expected_format=...)`, wrapper `train_arm` hoạt động end-to-end.

- [x] RED: continuous vs interrupted/resumed phải khớp actions/model/Adam/scheduler/RNG/history/manifest/corner, gồm crash giữa cặp, trước/sau snapshot/state atomic replace và giữa episode. Kiểm wrapper luôn bắt đầu cùng S2/optimizer mới; default `phase='D', save_training_state=True` không có context vẫn bị từ chối như trước.
- [x] Opt-in chỉ khi phase D + đúng context; Phase C/legacy D/E giữ guard/format/selection/đường code cũ. Format C loader mặc định không nhận format D; format D có model/optimizer/scheduler/RNG/histories/context/provenance, validation đủ schema + completed episode. Dùng `expected_format` ở mọi đường rollback/restore của D-S2.
- [x] Tích hợp prepare trước env.reset, dùng corner/opponents trả về; không chạy lựa chọn ngẫu nhiên pool cũ cho D-S2. Sau update/metric: tạo snapshot/candidate dự kiến, scheduler.step, lưu row/context + state atomically, rồi `after_commit` và progress/callback. Snapshot orphan không được publish. Partial episode restore state mới nhất; cả Python random và numpy/torch/CUDA được seed/restore trong wrapper. Không thay code loss/returns/mask/teacher.
- [x] `train_arm` khóa output, xác minh nguồn/runtime, resume kiểm state và reconcile projection; arm mới chỉ tạo trong registration hợp lệ, không auto-resume từ inference file. D-S2 tắt internal validation/run_report và legacy final-as-best; chỉ `completed.json` khi đủ registered episodes. CPU/GPU cache frozen opponents có giới hạn, không giữ mọi snapshot trên GPU.
- [x] GREEN: `venv\Scripts\python.exe -m unittest test_phase_d_s2_training test_training_state -q`; chạy CUDA resume trên GPU thật nếu CUDA dùng cho lượt full, ghi rõ unavailable/skip. Acceptance legacy có mệnh đề guard chỉ cần cập nhật khi tác giả xác minh nó thật sự trái contract; không giảm assertion của default path. Cập nhật nhật ký và commit integration.

## Task 4: Board evaluation, selection, CI và hai tầng gate

**Files:** create `phase_d_s2_evaluation.py`; test `test_phase_d_s2_evaluation.py`. Reuse read-only `evaluate.run_match`, `make_rule_based_fn`, `make_ac_fn`; không dựa vào `evaluate_pair_seeded` vì branch hiện tại không có function đó.

**Consumes:** verified registration, arm completion/candidate manifests. **Produces:** các interface evaluation trong bảng; đủ schemas cho validation/selection/report và task manifest.

- [x] RED: bộ fixture có macro delta/CI biết trước, thiếu corner/đối thủ, duplicate learner row, cùng SHA khác role, winners/draw sai; model chỉ tốt trước RB; boundary guard/tie-break/open gate. Kiểm B không được gọi trong validation, B 12→13 giữ logits; mismatch khác không được nội suy.
- [x] Mỗi task là một matchup-board đủ hai góc; dùng đúng public artifact schemas. `evaluate_board` seed/restore Python/numpy/torch RNG riêng cho từng góc bằng board seed, dựng fresh env mỗi góc, run_match một trận, giữ sampled mask và CPU threads=1. Row đủ các metric singular trong schema; S2-vs-S2 dùng role learner/opponent riêng.
- [x] `validate_task` kiểm identity/hash/seed/đủ góc/count/metric binary/steps≤registered max, active/passive/SD-win phân hoạch win và nhất quán kết quả hai ghế. `validation_tasks` chỉ chạy khi hai arm complete, 9 SHA tối đa × 5 đối thủ × 40 board. Không tính missing opponent bằng 0 hoặc bỏ qua bảng thiếu.
- [x] `select_models` dùng đúng guard/tie tuple/episode0 tại spec §7; khóa cả hai selection và test manifest, chỉ OPEN_TEST nếu pool ep>0, delta pool−S2≥3pp, pool≥control và mọi guard đạt. `test_tasks` ba learner × 6 đối thủ × 100 board; old snapshots chỉ pool episode<selected, loại SHA selected, dedupe giữ alias; empty old set không đạt Phase D.
- [x] `macro_board_bootstrap` đòi đúng bốn opponents G/F/S/S2 và hai corners cho mọi board, mean corners/opponents trước resample; return difference_pp/ci95_lower_pp/ci95_upper_pp/board_pairs/resamples/bootstrap_seed. RB/B dùng paired board CI riêng. `build_report` áp dụng nguyên spec §9: +5pp/CI>0 trước S2, CI>0 trước control, draw≤0/suicide≤+1pp, RB Active≥−2pp/suicide≤+1pp; Phase D thêm WR≥60% từng pool/snapshot và RB WR≥S2. Không dùng conditional WR hoặc suy ra tương đương từ CI chứa 0.
- [x] GREEN: `venv\Scripts\python.exe -m unittest test_phase_d_s2_evaluation -q`; yêu cầu tests ở đúng biên +5/0/+1/−2/60 và one-ulp phía không đạt, guard point estimates tách strict CI. Cập nhật nhật ký và commit evaluation.

## Task 5: CLI, frozen workers, ledger, progress và package

**Files:** create `run_phase_d_s2.py`; test `test_run_phase_d_s2.py`; reuse `progress.py` và `training_state.run_lock/atomic_save` không sửa nếu đủ.

**Consumes:** protocol/train/evaluation contracts. **Produces:** CLI trong public contract, `run_tasks`, `package_run`; full workflow nhưng chưa tự mở seed full.

- [x] RED: CLI reject stage sai thứ tự/arm sai/workers ngoài1..5/override config; selected bị đổi sau test-open, hash nguồn gián tiếp sai, ledger thiếu/trùng/sai một góc, second writer, mở test mới lần hai. Interrupted ledger resume không gọi evaluator cho shard đã valid; expected tasks − reused = số task chạy mới.
- [x] Parent xác minh live closure trước stage, bootstrap vào `sources/` đã đóng băng; worker chỉ import/run từ snapshot. Không ghi model/history cũ. Validation/test lock output; `test_opened.json` write-once khóa selected SHA + task manifest + registration digest trước trận đầu. Run/resume dùng danh sách expected IDs, validate mỗi shard trước reuse, từ chối extra/invalid IDs; chỉ publish final summary sau đủ task.
- [x] `run_tasks` dùng `parallel_map` và per-matchup-board atomic payload; bar khởi tạo số trận đã reuse, worker ticks đúng số trận mới. Train bar chỉ tính episode commit. Trên Ctrl+C dừng an toàn, state/task cũ giữ nguyên; resume cùng manifest idempotent. Không chạy validate/test từ train hoặc package ngầm.
- [x] `package_run` ZIP registration/spec/source/runtime, checkpoints/state/manifests/log/raw ledger và status; thiếu stage trả INCOMPLETE rõ ràng. VALIDATION_STOP không tạo test_opened. Ngăn đọc file ngoài input/output closure do manifest sai path; ZIP chỉ gói artifact trong output đã xác minh, không gom mọi `runs/`.
- [x] GREEN: `venv\Scripts\python.exe -m unittest test_run_phase_d_s2 -q`; kiểm subprocess Windows spawn, source import path và CMD redirect `> log 2>&1`. Successful test/resume fixture phải giữ nguyên gate, không thêm bypass full. Cập nhật nhật ký và commit CLI.

## Task 6: Kiểm chứng tích hợp và bàn giao CMD

**Files:** create `PHASE_D_S2_GUIDE.md`; update `CLAUDE.md` architecture map và ba nhật ký/tracker; test bốn acceptance file + regression suites.

**Consumes:** CLI đã qua tests. **Produces:** bằng chứng kiểm chứng, smoke package/status, hướng dẫn chạy từng stage và báo giới hạn; không có output full D-S2.

- [x] Chạy toàn bộ acceptance: `venv\Scripts\python.exe -m unittest test_phase_d_s2_protocol test_phase_d_s2_training test_phase_d_s2_evaluation test_run_phase_d_s2 -q`; xác nhận RED→GREEN bằng suite nguyên bản, không chỉ unit tests mới do implementer tự thêm.
- [x] Chạy regression có liên quan: `venv\Scripts\python.exe -m unittest test_training_state test_run_sd8 test_pde_eval test_pde_evaluator_contracts -q`; chỉ fixture/unit, không chạy lại seed held-out. Sau targeted pass, chạy một lần `venv\Scripts\python.exe -m unittest discover -p "test_*.py" -q`; nếu fail, phân biệt lỗi sẵn có bằng evidence cùng baseline/source, không gọi full suite PASS khi còn fail.
- [x] Smoke CLI trong output mới riêng: register --smoke --device cpu → train control → train pool → validate → package; nếu validation đủ gate smoke thì thêm test → test --resume → package, nếu không thì xác minh VALIDATION_STOP và test bị từ chối. Kiểm success path và interrupted test bằng fixtures gate biết trước. Đo thời gian/RAM, ghi dự báo theo từng stage; không tạo/reserve dải full.
- [x] CUDA smoke/resume trên GPU thật nếu máy sẽ train CUDA; thiếu GPU thì ghi giới hạn và chưa đánh dấu sẵn sàng CUDA. Kiểm default runtime của launcher/venv thực tế, deterministic/CUBLAS profile và progress trong CMD redirect; không khẳng định số liệu suy từ CPU là CUDA đã kiểm.
- [x] Viết GUIDE với lệnh CMD tuyệt đối dựa `E:\Bomberman\venv\Scripts\python.exe`, cách chỉ input-root khi chạy checkout khác, register/train control/train pool/resume/validate/test/package, exit/status, budget 4000 ep/9000 eval tối đa và giữ spec/source bất biến sau register. Cập nhật module map chỉ khi file thật tồn tại; tracker tách code verified khỏi full train not started.
- [x] `git diff --check`, kiểm diff chỉ các file D-S2/đường opt-in/docs; review toàn bộ thay đổi theo execution workflow đã chọn, sửa lỗi có bằng chứng rồi chạy lại checks bị ảnh hưởng. Commit kết quả và báo RED/GREEN/regression/CUDA/smoke evidence cùng lệnh bàn giao. Không tự launch training full, đăng ký seed full hoặc theo dõi job khi chưa được yêu cầu.

## Coverage và handoff

| Spec | Owner |
|---|---|
| §§1–2 scope, evidence, no search/FFA/full-run | Global Constraints, Task 0, Task 6 |
| §3 SHA/role/channel compatibility | Task 1, Task 4 |
| §§4–5 train config, paired schedule, frozen opponents | Task 2–3 |
| §6 ranges, audit, source/runtime freeze | Task 1, Task 5 |
| §7 validation selection/open gate | Task 4–5 |
| §§8–9 budgets, metrics, CI, pilot/Phase-D statuses | Task 4–6 |
| §10 artifact/resume/progress/CMD/package | Task 1–3, Task 5–6 |
| §11 acceptance và legacy regressions | Task 0, Task 3–6 |
| §12 approval/stage tracking | Task 0, Task 6 và tracker gốc |

Dependencies: 0 → 1 → 2 → 3 → 4 → 5 → 6. Chỉ task đã có RED tương ứng mới được implement. Không tạo nhánh triển khai, spawn test-author/implementer hoặc sửa code khi plan còn PLAN_REVIEW.

Đề xuất execution: **Native** vì các thay đổi nối nhau qua cùng trainer/state và cần giữ backward compatibility; tác giả acceptance độc lập và review cuối vẫn được giữ theo workflow. Subagent-driven là lựa chọn khác nếu người dùng muốn review từng task. Cần người dùng review plan và chọn cách thực hiện trước production edits; việc duyệt plan không đồng nghĩa yêu cầu mở lượt train full.

Execution note: checkbox thể hiện công việc đã thực hiện; full-suite có baseline failure/error đã được đối chiếu và ghi riêng, không phải tuyên bố toàn repo xanh. Rulings, review/fix và giới hạn xem verification.md. Không mở seed full hoặc thay checkpoint mặc định S2.
