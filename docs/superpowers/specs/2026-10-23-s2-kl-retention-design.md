# S2-KL — pilot ngắn kiểm khả năng giữ kỹ năng S2

Ngày: 23/10/2026. Trạng thái: **APPROVED_DESIGN** — người dùng duyệt bản spec trong commit59350ac ngày23/10/2026.

Người dùng đã duyệt pilot ngắn và implementation plan ngày23/10/2026: so sánh đối chứng với KL neo vào S2, cùng đấu RuleBot và giữ các yếu tố khác giống nhau. Runner/context/metric/ledger đã triển khai, acceptance40/40 gồm CUDA thật đã pass; kiểm chứng tích hợp/regression/smoke/review được ghi riêng trong `docs/superpowers/test-matrices/2026-10-23-s2-kl-verification.md`. Chưa đăng ký output full, đặt chỗ seed full hoặc khởi chạy pilot500episode/arm.

## 1. Câu hỏi và bằng chứng

**Câu hỏi:** với cùng ngân sách500episode, KL neo S2 có giữ được các quyết định đã chưng cất, đồng thời giữ ActiveWR trước RuleBot, tốt hơn nhánh không neo hay không?

D-S2 cũ tại `runs/phaseD-s2-20261022` kết thúc VALIDATION_STOP. Sau2000episode, RB ActiveWR S2/control/pool là83,75/65,00/57,50%; CNN ActiveWR6,25/6,25/7,19%. Trên1024decision states TRAIN cũ, argmax thuộc tập nước tốt của search giảm97,36→75,59(control)/73,14%(pool). Suy giảm đã hiện diện ở checkpoint500. Nguồn: `runs/phaseD-s2-20261022-analysis-20261023/ANALYSIS.md` và JSON/CSV kèm theo.

Control100%RB cũng giảm kỹ năng; vì vậy ưu tiên kiểm khả năng giữ policy trước khi đổi tỷ lệ đối thủ. Critic/shared trunk và teacher local là các giả thuyết còn lại, chưa được ablation. Engine đã loại bonus+10 khi đối thủ chết do vòng; không dùng giả thuyết thưởng thắng do vòng để thiết kế lượt này.

## 2. Lựa chọn phương án

| Phương án | Giá trị | Giới hạn |
|---|---|---|
| **Hai nhánh RB: KL0 và KL1 — chọn** | Một biến can thiệp, tận dụng đường `anchor_coef` hiện có, tách khả năng giữ policy | Chỉ thăm dò một hệ số KL và một train seed |
| Train pool tiếp, tăng tỷ lệRB | Đánh trực tiếp vào lịch đối thủ | Không giải thích được control100%RB đã giảm; thay thêm biến trước khi giải quyết retention |
| Replay nhãn distillation hoặc critic warm-up | Có thể nhắm trực tiếp vào điểm yếu còn lại | Thêm cơ chế và biến mới; dành cho ablation sau nếu KL rollout không giữ được kỹ năng |

Không quét hệ số, chọn hệ số theo validation hoặc tự kéo dài train trong pilot. KL1 là mốc thăm dò cố định, chưa phải hệ số tối ưu. Kết quả âm không đủ kết luận mọi dạng KL đều không hiệu quả.

## 3. Model và hai nhánh

Warm-start/reference chung: `E:\Bomberman\runs\sd8a-20261020\train\epoch_16.pt`.

```text
S2 SHA256 d222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814
```

SHA được xác minh lại lúc soạn spec; registration phải xác minh lại với identity SD8A đã chọn. Observation13kênh, board9×9; strict tensor shape, không nội suy.

| Arm | Episode | Đối thủ | `anchor_coef` |
|---|---:|---|---:|
| `control` | 500 | RuleBot | 0,0 |
| `anchor` | 500 | RuleBot | 1,0 |

Hai arm bắt đầu từ cùng weightsS2, Adam và scheduler mới; không lấy optimizer hoặc model sau train từ D-S2. Train tuần tự trên cùng GPU/máy/runtime. Chỉ khác hệ số KL; diagnostics và phương pháp đánh giá giống nhau.

Với nhánh anchor, dùng đúng KL đang có trong trainer:

\[
L_{\rm KL}=\beta\frac1T\sum_{t=1}^{T}
D_{\rm KL}\bigl(\pi_\theta(\cdot\mid s_t,m_t)\;\Vert\;\pi_{S2}(\cdot\mid s_t,m_t)\bigr),\qquad\beta=1.
\]

Hai phân phối dùng cùng safety mask tại trạng thái learner; chỉ tính các bước learner còn sống. Reference là bản sao S2 eval/no-gradient, nằm ngoài optimizer và giữ nguyên hash. Không dùng đối thủ làm reference. Khi resume, dựng reference lại từ S2 đã khóa **trước khi** khôi phục weights learner; reference vẫn là S2, không phải checkpoint đang resume.

KL train có chiều learner||S2; chỉ số offline KL(S2||learner) ở mục7 có chiều ngược và phải ghi tên đầy đủ, không gộp thành một số.

## 4. Cấu hình chung

| Thành phần | Giá trị |
|---|---|
| Phase / bàn / agents / max_steps | D / 9×9 / 2 / 500 |
| Policy | Actor-Critic hiện tại, MC, lấy mẫu qua safety mask; normalize_advantage=false |
| Adam / gamma / value_coef | lr2,5e-5 / 0,99 / 0,5 |
| LR schedule | CosineAnnealingLR T_max500, eta_min=lr×0,05 |
| Entropy |0,0075, lịch hiện có theo tổng500episode |
| Sudden death | Bật xác suất1; start475, interval25, warning40 |
| Teacher | local, versionv2; lookahead2; min_pressure_gain0,15; trap_sequence_depth2; sequence_pressure_gain0,35 |
| Auxiliary losses | escape0,5; combat0,9; navigation0,4; ring_safety0,6 |
| Cuối trận / wall bonus | late_game_start350; multiplier1,75; wall0,15 |
| Search teacher / group RL / PPO | Tắt |
| Nội bộ validation/report cũ | validation_every0; run_report=false |
| Checkpoint chẩn đoán | episode0,100,250,500 |
| RNG train |2026102301 cho cả hai arm |
| Diagnostics | advantage_diagnostics=true; reward_diagnostics=false |

Lịch LR/entropy500episode giống nhau giữa hai arm nhưng khác lịch2000episode của D-S2. Đây không phải tái lập chính xác500episode đầu của lượt cũ; mọi kết luận về KL dựa trên đối chứng mới cùng lịch500, không lấy control cũ làm đối chứng số học.

Giữ reward/teacher/mask/kiến trúc trong source đăng ký. Helper reward decomposition có thiếu ngoại lệSD; flag đó tắt và việc sửa helper nằm ngoài pilot này. Không đồng thời thêm critic warm-up, đóng băng trunk, đổi reward hoặc replay nhãn.

Mỗi cặp episode dùng cùng board và hai góc: với j=0..249, seedboard=710000+j; episode2j+1 learner góc0, episode2j+2 góc1. Cùng lịch này cho hai arm; không coi hai episode train là cặp thống kê độc lập vì policy cập nhật giữa chúng.

## 5. Ngân sách và seed

| Mục đích | Dải đề xuất | Số lượng |
|---|---|---|
| Train |710000–710249 |250board×2góc×2arm=1000episode |
| Validation RB |720000–720099 |100board×2góc/model; tối đa7SHA model=1400trận |
| Smoke tách biệt |790000–790003 | Cấu hình nhỏ riêng, không tiêu thụ dải train/validation thật |
| Offline retention | Collection TRAIN cũ của SD8A theo manifest khóa tại mục7 |1024decision+1024anchor quan sát/model |

Trần train500000learner steps; trần validation700000environment steps; tối đa7model×2048quan sát được chấm offline. BaselineS2, hai arm episode0 cùng SHA được đánh giá một lần và giữ alias. Không có stage held-out test trong pilot này; các dải630000/640000 của D-S2 cũ giữ nguyên reservation và chưa mở.

Rà read-only ngày23/10/2026 bằng inventory hiện có trên `runs/`:797nguồn/97965range,0unresolved, chưa thấy giao ba dải đề xuất với lịch sử hoặc reservation hiện có. Đây là kiểm tra tại thời điểm soạn, **chưa đặt chỗ**, và scanner cũ không bao quát `game UI/runs` mới. Preflight phải rà thêm metadata/board IDs trong `game UI/runs`, rồi rà lại toàn bộ lịch sử, raw ledger, registry và dải cơ chế trước registration; không coi sự vắng mặt trong registry là đủ chứng minh sạch. Metadata không ánh xạ được phải giải quyết trước chạy.

Đặt chỗ atomically bằng namespace experiment mới trong seed registry, giữ nguyên entry D-S2. Dải smoke tách rõ trong registry/audit. Không tự chọn dải khác nếu va chạm; cập nhật thiết kế trước khi mở dữ liệu.

## 6. Runtime, registration và resume

- Dự kiến Python `E:\Bomberman\venv\Scripts\python.exe`, train CUDA:0, validation/offline CPU1thread/worker, tối đa2worker validation. Runtime quan sát lúc soạn là torch2.11.0+cu128, numpy2.4.6, RTX4060Laptop; registration lúc chạy là nguồn chính thức. CUDA thiếu thì báo lỗi, không âm thầm đổi device.
- Khóa config/spec/S2/reference/RuleBot/mask/engine/trainer/helper sources, source dependency closure, runtime/thread/deterministic/TF32, seed manifest, offline sample manifest và collection shard SHA. Worker thực thi source snapshot đã đăng ký; không chỉ chạy live source.
- Cơ chế mới ở runner/context riêng, tái dùng training loop, anchor loss và full-state hiện có; không cần thay công thức trainer để chạy ablation. Giữ source D-S2 cũ còn xác minh được, không viết lại hash registration cũ để hợp thức hóa thay đổi.
- Full-state có model/Adam/scheduler, Python/numpy/torch/CUDA/env RNG, episode/corner/board cursor, histories, candidate manifest và identityS2/KL/config. State0 cũng phải có identity. Một writer/output; file tạm+flush/fsync+atomic replace.
- Resume từ cuối episode đã commit, replay phần chưa commit; referenceS2 dựng đúng như mục3. Không reset arm khi có artifact committed mà state mất. Evaluation resume tái dùng shard hợp lệ đủ hai góc; không lặp metric. Không chuyển máy/runtime trong cùng registration.
- Console có progress qua `progress.py`; log append khi resume. Lượt train dài mở CMD riêng, không tạo monitor hoặc polling sau launch nếu người dùng không yêu cầu.

## 7. Offline retention được khóa trước train

Reference sample manifest hiện có:

```text
runs/phaseD-s2-20261022-analysis-20261023/offline_sample_manifest.json
SHA256 9b36b8e07ed8f65175e5c9fbf53a18eca16ae791f0073d19730b28e75529e333
```

Manifest chỉ rõ từng file/row và shard SHA, lấy mẫu seed20261023. Decision1024state trên722board, anchor1024state trên667board. Loại150board thuộc held-out split SD8A gốc (seed530,10%). Registration copy và khóa manifest này; từ chối nếu collection/shard/identity thay đổi, không lấy mẫu lại sau khi xem kết quả.

Mỗi checkpoint0/100/250/500 chạy forward-only cùng batch CPU/mask đã lưu. Đo:

1. **Decision retention R:** phần trăm argmax hợp lệ thuộc tập action có search-score cao nhất; nhận mọi action đồng hạng, loại trạng thái mọi action đồng điểm theo logic SD8A.
2. **Decision probability Q:** trung bình tổng xác suất model dành cho tập action tốt đó, tính theo%.
3. Diagnostics: soft-target CE, KL(S2||model) ở decision/anchor, argmax thay đổi soS2, xác suấtBOMB, value mean. Không dùng value mean làm critic calibration nếu thiếu target.

Baseline S2 phải đo lại trong runtime mới. Số97,36%(R) và91,83%(Q) từ phân tích trước chỉ là tham chiếu, không cắm cứng thay kết quả baseline của lượt mới.

Đây là dữ liệu TRAIN cũ đã được xem; dùng đo giữ kỹ năng, không dùng làm bằng chứng tổng quát hoặc gọi là held-out. Trong hai arm, dữ liệu này chỉ được chấm offline; không dùng làm replay/gradient/nhãn train trong ablation.

CI retention: cluster bootstrap10000lần, seed20261023, resample board với replacement và giữ toàn bộ state thuộc board đã chọn; tính tỷ lệ trên tổng số state trong mỗi bootstrap sample. So sánh model dùng cùng các board resample. Giữ trọng số micro theo state cho R/Q; không vô tình đổi sang macro theo board. Đây không phải CI qua nhiều training seed.

## 8. Validation RuleBot và checkpoint cuối

Sau khi cả hai arm đủ500episode, đo baselineS2 và tất cả checkpoint100/250/500 của hai arm,200trận/model (100board×2góc), cùng dải720000–720099 và sampling/mask/engine đăng ký. Mỗi task một board chứa đủ hai góc; policyRNG seed theo board giống evaluator D-S2. Baseline và model đều đo lại trong cùng runtimeCPU.

Xuất WR/ActiveWR/hòa/tự nổ/thắng thụ động/thắng do vòng/chết do vòng, kills/bombs/steps, theo góc và tổng. CI delta RB dùng paired bootstrap10000lần, seed20261023, mean hai góc/board rồi resample100board chung giữa model.

**Chỉ episode500 quyết định pilot**. Checkpoint100/250 chỉ mô tả diễn biến, không chọn best, không thay thế endpoint nếu500không đạt. Các phép so checkpoint giữa lượt là diagnostics và phải ghi hậu kiểm/không dùng gate. Không dùng kết quả validation để đổi beta, ngân sách hoặc train lại trong cùng output.

## 9. Luật quyết định đã chốt trước

Đặt B=baselineS2, C=control500, A=anchor500. R vàQ lấy từ decision states; AW vàSU là ActiveWR/tự nổ validationRB, đều đơn vị%.

Điều kiện kiểm riêng, luôn xuất đủ mọi kết quả:

| Mã | Điều kiện |
|---|---|
| DRIFT_REPRODUCED | R(C) <= R(B)−10pp |
| DECISION_RETAINED | R(A) >= R(B)−5pp |
| PROBABILITY_RETAINED | Q(A) >= Q(B)−5pp |
| ANCHOR_BENEFIT | R(A)−R(C) >=10pp **và** CI95 lower của delta >0 |
| RB_ACTIVE_RETAINED | AW(A) >= AW(B)−2pp |
| RB_SUICIDE_RETAINED | SU(A) <= SU(B)+1pp |

Các mức10pp/5pp là tiêu chí nghiên cứu mới để tách mức suy giảm khoảng20–24pp đã quan sát, chưa phải ngưỡng tối ưu hay chuẩn tổng quát. RB guard giữ mức điểm tương ứng D-S2; CI RB phải xuất đầy đủ nhưng không thêm ngưỡng CI gate sau khi xem dữ liệu.

Phân loại theo thứ tự sau; các gate khác vẫn xuất kể cả khi status ưu tiên đã xác định:

1. Artifact thiếu/drift, số trận không đủ hoặc offline chưa đủ → `INCOMPLETE`, không kết luận chất lượng.
2. `DRIFT_REPRODUCED=false` → `INCONCLUSIVE_NO_DRIFT`: đối chứng ngắn không tái hiện mức mất kỹ năng đã định; không tuyên bố KL giúp hoặc không giúp.
3. Bất kỳ guardRB không đạt → `RB_RETENTION_FAIL`.
4. R hoặc Q của anchor không giữ được → `RETENTION_NOT_PRESERVED`.
5. `ANCHOR_BENEFIT=false` → `NO_CLEAR_ANCHOR_BENEFIT`.
6. Tất cả điều kiện đạt → `RETENTION_PILOT_PASS`.

Không status nào là PHASE_D_PASS hoặc tự thay model mặc địnhS2. Pass chỉ cho phép đề xuất một lượt pool mới có spec riêng. Fail/inconclusive đóng gói và đọc kết quả, không tự tăng beta, đổi schedule, chọn100/250 hoặc kéo dài train. Một seed và dữ liệu retention cũ giới hạn kết luận ngay cả khi pass.

## 10. Artifact, stages và kiểm chứng trước chạy

Output full dự kiến `runs/s2-kl-20261023`, namespaceS2-KL riêng. CLI `run_s2_kl_probe.py` gồm `register`, `train --arm control|anchor [--resume]`, `diagnose [--resume]`, `validate [--resume]`, `package`; lệnhCMD ở `S2_KL_GUIDE.md`. Không có stage test. Luồng đầy đủ: register → traincontrol → trainanchor → diagnose → validate → package.

Artifact tối thiểu:

```text
registration.json, runtime.json, seed_manifest.json, offline_sample_manifest.json,
sources/, arms/control/, arms/anchor/,
  initializing.json, training_state.pt, completed.json,
  checkpoints/{0,100,250,500}.pt, metrics.jsonl,
diagnostics/<model-sha>.json,
validation/tasks/<task-id>.json, validation.json,
report.json, decision_report.md, logs/, <output>-send.zip
```

Model SHA trùng giữ alias trong manifest/report, không đánh lại. Các projection/report phải tái dựng từ raw data và từ chối drift; package không tự train/evaluate hoặc đổi status chưa đủ dữ liệu thành pass. Package đọc selection cố định endpoint500, không thực hiện chọn best. ZIP chỉ đóng output mới, không chép collection gốc hoặc artifact của thí nghiệm khác vào.

Train metrics bổ sung các trường đã có trong trainer: `anchor_kl` ở armanchor, policy/value/teacher loss, advantage/returns/value statistics, explained variance, entropy và grad_norm trước clip. Armcontrol ghi rõ anchor tắt; không giả làm có KLloss nếu chưa tính. Không suy gradient từng thành phần từ globalgrad norm hoặc độ lớn loss. Log per-component gradients chưa có và không thuộc phép thử một biến này.

Trước lượt thật cần acceptance theo thiết kế, integration/smoke CPU+CUDA, exact-resume cả hai arm (bao gồm reference anchor), và kiểm quan sát diagnostics không đổi weights/RNG/loss so khi không ghi log. Các kiểm chứng quan trọng:

- KLbeta0 giữ đường control; KLbeta1 đúng chiều/mask/mean và không backprop reference; anchor hash luôn bằngS2 sau resume.
- Cùng initialweights/board/corner/RNG policy trước khi hai nhánh khác gradientKL; checkpoint0baseline dedup đúng.
- State đủ optimizer/scheduler/RNG/cursor, không mất/lặp episode khi ngắt giữa update/atomic commit; CUDA resume kiểm trên CUDA thật.
- Sampling manifest/split/nhãn tie/CI cluster và biên gate có dữ liệu tổng hợp tính tay; endpoint cố định500, không chọn checkpoint trung gian.
- Registration/source/runtime/seed/drift/writer lock, crash recovery, ledger đủ hai góc và package không tạo pass giả.
- Smoke tách dải riêng; ghi walltime/RAM và dự phóng từng stage trước registrationfull, không dùng trận max20bước để hứa ETA500bước.

## 11. Ranh giới phê duyệt

Người dùng đã duyệt spec cụ thể sau commit59350ac, rồi duyệt plan ed7c749 và yêu cầu triển khai ngày23/10/2026. Giữ cách triển khai Native, acceptance có tác giả độc lập và reviewer cuối. Chỉ chạy lượt full mới khi implementation/kiểm chứng xong và có yêu cầu khởi chạy trong phiên làm việc; duyệt triển khai code không tự mở terminal train full.

Không sửa hoặc ghi đè D-S2/SD8A đã có. Giữ mặc địnhS2 trong suốt pilot và sau mọi kết quả cho đến khi có quyết định riêng dựa trên thí nghiệm phù hợp.
