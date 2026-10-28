# Plan và tracker: hướng Sudden death (SD)

File này chỉ để **theo dõi công việc** (đánh dấu `[x]` khi xong, có bằng chứng). Số liệu, bảng kết quả và quyết định ghi trong `RESULTS.md`; thay đổi code/phương pháp ghi trong `IMPROVEMENTS.md`; quy ước bền vững ghi trong `CLAUDE.md`. Tracker tổng của dự án: `PLAN_BOMBERMAN.md`.

- [x] S3-MAP11-COMPAT (24/10/2026): xác nhận checkpoint S3 9×9 nạp/forward được trên 11×11 bằng `smart_load_weights`; chỉ kiểm tra tương thích, chưa train hoặc đo sức chơi. RESULTS `S3-MAP11-COMPAT`, IMPROVEMENTS §90.
- [x] S3-MAP11-STEPS (24/10/2026): giải thích quy trình runner riêng → baseline/thầy → dữ liệu mới → train/validation/test; cấu hình minh họa là đề xuất, chưa triển khai hoặc chạy. RESULTS `S3-MAP11-COMPAT`, IMPROVEMENTS §90.
- [x] S3-MAP11-STEP1 (24/10/2026): tạo runner/guide riêng 11×11 từ S3; 35 test mới + 27 regression liên quan PASS, smoke 2 worker PASS. Toàn repo486: 8 lỗi fixture cũ, 1 skip. Chưa chạy baseline/collect/train thật hoặc đăng ký seed test. RESULTS `S3-MAP11-STEP1`, IMPROVEMENTS §91.
- [x] S3-MAP11-AUTO-SETUP (24/10/2026): người dùng cho phép bước2–5; init `runs/s3-map11-20261024-auto-v1`, launch baseline, heartbeat45phút ACTIVE. RESULTS `S3-MAP11-AUTO`, IMPROVEMENTS §92.
- [x] S3-MAP11-STEP2: baseline S3/S3+search3 trên11×11,100trận/agent — DONE/PASS; ActiveWR50→98%,+48pp CI95[38,58], suicide0→0%. RESULTS `S3-MAP11-BASELINE`, IMPROVEMENTS §93.
- [x] S3-MAP11-STEP3: collect200trận seed812000 DONE;4shard,746decision/1310anchor, split90/10bàn hợp lệ. RESULTS `S3-MAP11-COLLECT`, IMPROVEMENTS §94.
- [x] S3-MAP11-STEP4: chưng cất16epoch từ S3/CUDA DONE; đủ4candidate4/8/12/16, metrics hữu hạn/critic_head giữ nguyên. RESULTS `S3-MAP11-TRAIN`, IMPROVEMENTS §95.
- [x] S3-MAP11-STEP5: DONE/PASS, epoch16 test820000 một lần: ActiveWR87.25% vs46.25%,+41pp CI95[34.5,47.25], tự nổ1.25% vs.75%(+.5pp≤1). Đủ400trận/model, giữ S3 gốc, gỡ heartbeat khi chốt. RESULTS `S3-MAP11-TEST`, IMPROVEMENTS §97, run `FINAL_REPORT.md`.
- [x] S3-11-PUBLISH (25/10/2026): push nhánh mới `Loc/S3-11` thành công,355file tại commit nội dung5eb6b67;5checkpointLFS/38.28MB đã xác nhận.67test S3-map11/CNN+34bridge+3UIunit và UIbuild PASS; remote/upstream đúng, workingtree sạch sau upload. Ghi xác nhận bằng commit tài liệu cùng nhánh; không merge main. RESULTS `GIT-LOC-S3-11-PREP`, IMPROVEMENTS §102.
- [x] S3-MAP11-CNN-HOWTO (25/10/2026): giải thích evaluator CNN11×11 từ opponent_fn; runner cũ cố định9×9. Đề xuất pool S3/S2/G, so S3-11 với S3 nội suy, chưa viết/chạy benchmark hoặc mở seed mới. RESULTS `S3-MAP11-CNN-HOWTO`, IMPROVEMENTS §98.
- [x] S3-MAP11-CNN-RUNNER (25/10/2026): runner/guide/spec/test riêng;32/32targeted + native2worker24game PASS, review integrity hoàn tất. Suite rộng518test:32test mới PASS,8lỗi fixture cũ đã tái hiện trước code/1skip, không có assertion failure. Registration/source/model giữ nguyên. RESULTS `S3-MAP11-CNN-LAUNCH`, IMPROVEMENTS §100.
- [x] S3-MAP11-CNN-EVAL (25/10/2026): đủ2400/2400trận/120shard/6cặp, hash/raw/report xác thực,22.90phút đến summary. S3-11 vs S3 macroActive8.58vs4.75%,+3.83pp[1.92,5.83]; WR23.00vs21.67%, hòa64.33vs63.08%, tự nổ1.58vs1.33%. Gate không đạt ngưỡng+5pp; state completed_no_qualified_gain, không train/promote, kết thúc theo dõi và gỡ lịch45phút. RESULTS `S3-MAP11-CNN-FINAL`, IMPROVEMENTS §101, run `FINAL_REPORT.md`.

Trạng thái: **DONE** = xong (kể cả gate FAIL); **READY** = làm được ngay; **WAIT** = chờ người dùng chạy/kết quả; **COND** = chỉ mở khi điều kiện ghi kèm thỏa; **DEFERRED** = để sau.

Cập nhật: 24/10/2026. Checkpoint CNN đang giữ: G `runs/sd75-20261018/full/best_phaseC.pt` (từ SD7.6; trước đó B tới SD4.6, F cho SD5). Agent mạnh nhất: G + `kill_search.search_action3` (SD8B, test 49000). Seed test tiếp theo: 58000 (57000: SD9; PDE2 đã dùng 51000–56099; 50000: PDE; 48000: SD8A). Agent chính (người dùng chốt chính thức 24/10, sau SD9.3): S3. CNN thuần mạnh nhất: S3 `runs/sd9-20261023/train/epoch_12.pt` (SD9, test 57000; trước đó S2 `runs/sd8a-20261020/train/epoch_16.pt`). Lịch SD dùng để đánh giá từ SD5: lịch E 475/25/40 của engine (SD1–SD4 dùng SD2 + biên mask learner k=15).

Số liệu trong checkbox là tóm tắt. Trừ khi ghi khác, mỗi con số là 400 trận/agent (200 bàn × 2 góc), RuleBot, 9×9, 2 agent, max_steps 500, policy lấy mẫu, CI95 bootstrap ghép cặp theo bàn. Thông số đầy đủ (episode, train seed, máy, torch, số tiến trình) nằm ở dòng **Thông số** của mục tương ứng trong `RESULTS.md` và `IMPROVEMENTS.md` (quy định 20/10/2026).

| ID | Việc | Trạng thái | Kết quả chính (chi tiết ở RESULTS.md) |
|---|---|---|---|
| SD1 | Kiểm chứng SD gốc (start200/interval20, đóng hết vòng) | DONE | WR +4,25 nhưng ActiveWR −15,25; giữ tắt cấu hình gốc |
| SD2 | Tune timing + `max_ring` | DONE | start400/interval40/warning40/max_ring2; WR 39,5→59,25 chủ yếu thắng thụ động |
| SD3 | Train B dưới SD2 ± kênh vòng | DONE, gate FAIL | ActiveWR −2,25 cả hai nhánh; kênh vòng +0; giữ B |
| SD4 | Giảm learner chết vì vòng, rồi biến thời gian sống thêm thành thắng chủ động | SD4.0/4.1/4.2a/4.6 DONE. SD4.1 PASS (WR +10.75, ActiveWR +0.75); SD4.2a không đạt; SD4.6: search cả trận +5.75 ActiveWR (đạt), search cuối trận +0. Cấu hình chốt: SD2 + k=15 | — |
| XC | Đối chiếu nhánh bạn `codex/phasec-sd-best-fallback` | DONE | F hơn B +7.5 ActiveWR (gộp, lịch E, CI dưới > 0); số 51.0 họ báo không tái lập được (lấy mẫu torch) |
| SD5 | Từ F: công thức của bạn + chưng cất kill-search, 3000 ep/lượt | Merge DONE; SD5.0 ĐẠT (search +22.5 ActiveWR trên F); SD5.1 gate FAIL (search − F −2.75 ActiveWR, control − F −6.5), giữ F | — |

## SD1 — kiểm chứng SD gốc

- [x] Runner `run_sudden_death_experiment.py`, 4 case (off/rings_only/learner_aware/both_aware), 400 trận/case, seed 38000.
- [x] Chẩn đoán: vòng cuối giết cả hai (tâm 9×9 là cột cứng), start sớm cắt các trận thắng chủ động muộn.
- [x] Sửa test fixture đi vào gỗ.
- [x] Ghi RESULTS.md (SD1/SD2) và IMPROVEMENTS.md §23.

## SD2 — tune timing

- [x] Thêm `sudden_death_max_ring` + test.
- [x] `tune_sudden_death.py` (resume, 6 worker, cấm seed ≥ 38000); quét `runs/sd-tune-v1-closure` (30 cấu hình) và `runs/sd-tune-v2-maxring` (49/49).
- [x] Chốt luật chọn trước; `confirm_sudden_death.py` xác nhận một lần seed 39000 (`runs/sd-confirm-39000`).
- [x] Ghi RESULTS.md và IMPROVEMENTS.md §23.

## SD3 — train dưới SD2 ± kênh vòng

- [x] Spec `docs/superpowers/specs/2026-10-16-sd-training-ring-channel-design.md`, plan `docs/superpowers/plans/2026-10-16-sd-training-ring-channel.md`.
- [x] Kênh vòng `ring_channel`, trainer/evaluator đọc số kênh theo checkpoint, `smart_load_weights` từ chối checkpoint rộng hơn mạng.
- [x] `run_sd_training_pilot.py` + test; 269 test PASS; smoke CPU/CUDA complete; review cuối.
- [x] Full `runs/sd3-full` (người dùng chạy), seed 40000: gate FAIL, giữ B.
- [x] Chẩn đoán chết vì vòng từ dữ liệu test (RESULTS.md "SD3 — chẩn đoán chết vì vòng").
- [x] Ghi RESULTS.md và IMPROVEMENTS.md §24–§26.
- [ ] (Minor, hoãn) Cache đánh giá chưa khóa max_steps/timings; training state chưa ghi timing SD; chưa có unit test đường resume; `SOURCE_FILES` thiếu 2 file test.

## SD4 — giảm chết vì vòng (đề xuất, chưa cài)

Giả thuyết cần kiểm: mask learner chỉ đòi "còn đường thoát", nên learner nán lại vòng ngoài sát hạn chót và bị chặn trong vài bước cuối (B: 121/400 trận chết vì vòng, ~3 lần mask giao rỗng/trận; RuleBot chỉ 47/400).

### SD4.0 — kiểm chứng giả thuyết bằng replay (DONE)

- [x] Script chẩn đoán trên validation 21000 (không đụng seed test): với mỗi trận learner chết vì vòng, ghi theo bước trong cửa sổ cảnh báo: learner có ở ô vòng không, khoảng cách tới vùng an toàn, số hành động mask cho phép, bước giao rỗng đầu tiên, thứ chặn đường (đối thủ, bom mới, gỗ).
- [x] Kết luận giả thuyết đúng/sai, ghi RESULTS.md + IMPROVEMENTS.md. Nếu sai, sửa lại SD4.1 trước khi làm. → Đúng một phần: nán lại sát giờ phổ biến; ca chết do đường bị cắt lúc còn 2–4 bước bởi bom của mình (48/112) và thân đối thủ (41/112). Giữ SD4.1 như đã chốt (`runs/sd4-diag-21000`, IMPROVEMENTS §28).

### SD4.1 — mask có biên an toàn cho learner (DONE, gate PASS)

- [x] Chốt trước với người dùng (đã duyệt 16/10/2026, trước khi chạy SD4.0): tập k, luật chọn k, gate: k ∈ {0, 5, 10, 15, 20}; chọn k có ActiveWR validation cao nhất trong các k giảm chết-vì-vòng, hòa thì k nhỏ hơn; gate test: Δ chết-vì-vòng có CI95 < 0, ActiveWR CI95 lower ≥ −3pp, tự nổ không tăng quá 1pp.
- [x] TDD: tham số biên `k` (learner bị coi như ô vòng đóng sớm hơn k bước); k=0 phải giống hệt SD2 (test parity); RuleBot không đổi. → `sudden_death_mask(..., margin=k)` + `evaluate_case(learner_margin=k)`, runner `run_sd_margin.py`; parity k=0 khớp bản SD3 kể cả bộ đếm mask.
- [x] Quét k trên validation 21000, 400 trận/k (≤ 6 tiến trình). → `runs/sd4-margin-sweep-21000`, chọn k=15.
- [x] Test một lần seed 41000, 400 trận: B + SD2 vs B + SD2 + biên k đã chọn. → `runs/sd4-margin-test-41000`: gate PASS; chết vì vòng 116→76, WR +10.75, ActiveWR +0.75 (CI chứa 0).
- [x] Ghi RESULTS.md, IMPROVEMENTS.md, cập nhật seed đã dùng trong CLAUDE.md.

### SD4.2a — probe cuối trận: teacher có hạ được RuleBot trong vùng co không? (DONE, không đạt)

Insight (validation 21000, k=15): 250/400 trận kéo tới bước 400; chỉ 19 trong số đó learner thắng chủ động (8%), trong khi 120/150 trận kết thúc trước 400 là thắng chủ động. ActiveWR gần như do giai đoạn trước 400 quyết định; SD và biên k chỉ đổi kết cục thụ động. Thoát vòng đã được mask k=15 lo, nên teacher **thoát vòng** (SD4.2 cũ) không còn là đòn bẩy; cần biết giai đoạn cuối có thể hạ được RuleBot hay không.

- [x] Probe `probe_sd_endgame.py`: cùng trận/RNG tới bước 360 (đầu cửa sổ cảnh báo vòng 1), sau đó learner do một bộ điều khiển khác: B (đối chứng), B + teacher tấn công local, B + two_bomb rồi local, RuleBot. SD2 + biên k=15, validation 21000, 400 trận/nhánh.
- [x] Luật quyết định (chốt trước khi chạy): nếu một nhánh teacher tăng ActiveWR ≥ +5pp so với B trên validation, mở SD4.2b (train B học teacher tấn công chỉ ở giai đoạn cuối, hoặc train bắt đầu từ trạng thái bước ~360). Nếu không nhánh nào đạt, kết luận giai đoạn cuối khó khai thác với đối thủ RuleBot; dừng nhánh SD hướng ActiveWR và quay về kỹ năng tấn công đầu trận.
  → KHÔNG ĐẠT: ActiveWR local −4.25, two_bomb −3.5, RuleBot −2.25 so B (`runs/sd4-endgame-probe-21000`). B là bộ hạ đối thủ tốt nhất ở giai đoạn cuối.
- [x] Ghi RESULTS.md + IMPROVEMENTS.md (§30).

### SD4.2b — train học teacher tấn công giai đoạn cuối (ĐÓNG: SD4.2a không đạt)

- [ ] Spec + plan, TDD, smoke, full (người dùng chạy), seed test 42000.

### SD4.2 (cũ, thay bằng SD4.2a/b) — teacher thoát vòng từ RuleBot (READY: điều kiện đã thỏa — SD4.1 giảm chết vòng nhưng ActiveWR không tăng; cần chốt thiết kế với người dùng)

- [ ] Thiết kế: loss imitation phụ theo hành động RuleBot biết vòng, chỉ trong cửa sổ cảnh báo, dùng hạ tầng teacher phụ có sẵn.
- [ ] Spec + plan, TDD, smoke, full (người dùng chạy), seed test mới.

### SD4.3 — curriculum timing SD + GAE (ĐÓNG cùng SD4.2b; chỉ mở lại nếu có bộ điều khiển giai đoạn cuối tốt hơn B)

- [ ] Khi train, vòng đến sớm rồi lùi dần về start400 để có nhiều tín hiệu vòng hơn; actor dùng GAE (đã có).
- [ ] Ghép vào lượt train SD4.2 như một nhánh riêng, không bật chung trong cùng một nhánh.

### SD4.5 — cấm/hạn chế đặt bom trong ô vòng gần hạn chót (COND: sau SD4.1; từ chẩn đoán SD4.0)

- [ ] Chỉ mở nếu SD4.1 xong mà bom của chính learner vẫn là nguyên nhân chính của chết vì vòng (cần chạy lại `diagnose_sd_ring_deaths.py` với k=15 trên validation để biết).

### SD4.6 — probe kill-search có mô hình đối thủ (DONE: search@0 đạt, search@360 không đạt)

- [x] `kill_search.py` + test: với mỗi hành động mask cho phép, mô phỏng 12 bước (đối thủ = RuleBot thật, learner sau bước đầu chỉ né); giữ hành động B trừ khi có hành động điểm cao hơn hẳn (hạ được đối thủ / tránh chết). Chỉ chạy khi đối thủ trong Manhattan ≤ 6. Nhánh `search` trong `probe_sd_endgame.py`.
- [x] Probe validation 21000, 400 trận/nhánh, SD2 + k=15: B; search từ bước 360 (giai đoạn cuối); search từ bước 0 (cả trận). Lượt đầu bị dừng vì thiếu RAM; người dùng chạy lại trong cmd (`runs/sd4-search-probe-21000`).
- [x] Luật quyết định (chốt trước khi chạy): search@360 tăng ActiveWR ≥ +5pp so B → giai đoạn cuối khai thác được bằng lookahead, mở thiết kế chưng cất search vào CNN cho giai đoạn cuối. search@0 tăng ActiveWR ≥ +5pp → mở hướng mới (chưng cất search kiểu expert iteration cho cả trận, ngoài phạm vi SD). Dưới ngưỡng → lookahead ngắn với mô hình RuleBot không đủ; dừng.
  → search@360 −0.25 [−2.0, 1.5] (không đạt); search@0 **+5.75 [2.75, 8.75]** (đạt) → mở hướng chưng cất search cả trận (SD5, chờ người dùng duyệt thiết kế).
- [x] Ghi RESULTS.md + IMPROVEMENTS.md (§33). Lưu ý giới hạn: search dùng mô hình đối thủ hoàn hảo (chính code RuleBot), là cận trên chứ chưa phải agent học được.

### SD4.4 — reward shaping dạng potential (DEFERRED)

- [ ] Chỉ mở nếu SD4.1–SD4.3 đều không giảm chết vì vòng; các biến thể reward trước đều FAIL gate.

## XC — đối chiếu nhánh `codex/phasec-sd-best-fallback` (DONE)

- [x] Phân tích tĩnh: cùng B, cùng engine/RuleBot (trừ vá vòng), cùng định nghĩa ActiveWR, không teacher khi đánh giá, seed không dùng lại; khác: ~3000 ep train, teacher tấn công mạnh hơn, lịch vòng 475/25 (RESULTS.md "XC").
- [x] Chạy `crosscheck_friend_branch.py` sau khi SD4.6 xong (E1 tái lập, E2 B cùng seed, E3/E4 seed 60000). → `runs/crosscheck-friend-a6650a0`.
- [x] Kết luận + ghi RESULTS.md/IMPROVEMENTS.md (§33): F hơn B thật (+7.5 ActiveWR gộp), ít hơn số họ báo. Nếu model của họ hơn B thật, cân nhắc dùng làm điểm xuất phát mới (cần quyết định với người dùng).

## SD5 — từ F: công thức của bạn + chưng cất kill-search (người dùng chọn hướng a, 17/10/2026)

Quy ước SD5: điểm xuất phát và đối chứng là F (`runs/phaseC-sd-best-fallback/best_phaseC.pt`); mỗi lượt train 3000 episode; luật chính lịch E (SD engine 475/25/40, bật 75% episode khi train); báo thêm không SD. Validation 21000, test một lần seed 42000.

- [x] Merge nhánh bạn vào `danh/sd5-friend-merge`, giữ tương đương (RESULTS.md "MERGE", IMPROVEMENTS.md §34).
- [x] Duyệt thiết kế SD5 với người dùng (người dùng duyệt 17/10/2026, gồm ngưỡng SD5.0 và gate SD5.1).
- [x] SD5.0a — `probe_sd5_search.py` + `test_probe_sd5_search.py` (3 test), smoke 8 trận/nhánh: F 1.1 s/trận, search 6.6 s/trận.
- [x] SD5.0 — probe validation (không train, người dùng chạy `runs/sd5-search-probe-21000`): F vs F + kill-search cả trận dưới lịch E, 400 trận; kèm chẩn đoán trận hòa của F (chạm mặt, bước gặp đầu). Luật (chốt): mở SD5.1 nhánh chưng cất nếu search tăng ActiveWR ≥ +3pp với CI dưới > 0; nếu không, chỉ chạy nhánh đối chứng và chuyển sang đòn bẩy trận hòa (SD5.2).
  → ĐẠT: ActiveWR 47.0 → 69.5 (+22.5 [18.25, 26.75]); hòa −5.75; mọi trận đều chạm mặt (hòa do không kết liễu được, không do né).
- [x] SD5.1a — cài `search_teacher` (nhóm loss riêng, IMPROVEMENTS §36), sửa resume SD, runner `run_sd5_pilot.py` + test; smoke CLI chạy được; đo 2.8 s/ep (control), ~8–9.5 s/ep (search). Có thanh progress (`progress.py`, IMPROVEMENTS §37).
- [x] SD5.1 — (DONE, gate FAIL; RESULTS SD5.1 kết quả: search − F −2.75 [−9.75, 4.0], control − F −6.5 [−12.75, 0.0], search − control +3.75 [−2.75, 10.25]; giữ F; nhánh search train trên máy khác, torch 2.14.1, code giống hệt) train 2 nhánh × 3000 ep từ F, cùng train seed: `control` (công thức của bạn) và `search` (thêm nhãn kill-search ở nhóm combat khi search tìm được nước tốt hơn hẳn). Validation mỗi 500 ep, 400 trận. Gate đề xuất (nhánh search): ActiveWR ≥ +5pp so F, CI dưới > 0, không kém control, tự nổ không tăng quá 1pp.
- [x] SD5.3a — đo mật độ khoảnh khắc quyết định (20 trận seed 6600: 2.2 bước có thông tin/trận, 1.1 lần search sửa nước) và viết spec `docs/superpowers/specs/2026-10-18-sd5-3-dense-search-distillation-design.md`.
- [x] SD5.3 — (ĐẠT, bạn chạy: S − F ActiveWR +16.25 [+9.75, +22.5] test 43000; Claude review + phân tích offline, RESULTS SD5.3 review) chưng cất kill-search dày: F + search thu 3000 trận, fine-tune giám sát + KL neo, chọn 1 trong 3 ứng viên trên validation, test một lần seed 43000. Gate: ActiveWR ≥ +5pp so F, CI dưới > 0, tự nổ ≤ +1pp.
- [ ] SD5.2 — (COND) đòn bẩy trận hòa: ưu tiên lại bàn F hay hòa/thua khi train (kiểu PLR), chỉ mở sau SD5.0/SD5.1.

## SD6 — sửa điểm mù của F (đề xuất 18/10/2026, chạy song song SD5.3 của bạn)

Insight rút ra từ SD5–SD6 được gom ở `INSIGHTS.md`.

- [x] SD6.0 — chẩn đoán (RESULTS SD6.0): nhiệt độ/argmax bị bác bỏ; điểm mù (a) nước chặn đường thoát sau khi đặt bom (F gán xác suất ~0.001), (b) rời vành ngoài trước bước 475.
- [x] SD6.1a — người dùng duyệt (18/10/2026); cài `cutoff_teacher.trap_actions` + `probe_sd6_cutoff.py` (test: `test_cutoff_teacher` 3, `test_probe_sd6_cutoff` 4; fixture chặn đường được kill-search xác nhận); kèm hồ sơ độ khớp với search trên seed 15000, 400 trận.
- [x] SD6.1 — (DONE, KHÔNG ĐẠT; RESULTS SD6.1 và SD6.1b: bản đầu −2.25 ActiveWR, khớp 0/85; bản sửa −1.75 [−4.25, 0.75], khớp 56/85, WR +3.75 [1.75, 6.0] và hòa −2.75 nhưng đều là thắng thụ động) probe trên validation: F + luật "escape có xét đối thủ" lúc chơi (ưu tiên lối thoát an toàn làm đối thủ hết đường sống). Luật đề xuất: mở SD6.2 nếu ActiveWR ≥ +3pp với CI dưới > 0.
- [ ] SD6.2 — (KHÔNG MỞ: SD6.1 không đạt) train 3000 ep từ F, công thức của bạn, thay nhãn escape `exits[0]` bằng nhãn escape có xét đối thủ (không có lối chặn thì đích mềm đều trên mọi lối thoát an toàn). So với F, test một lần seed 44000 (43000 để cho SD5.3).
- [ ] SD6.3 — (COND, phụ, cho WR/hòa) biên an toàn với vòng của engine cho mask learner (giống SD4.1 k=15), probe trên validation.

## SD7 — RL với advantage theo nhóm từ các khoảnh khắc quyết định (đề xuất 18/10/2026)

Ý tưởng: phần thưởng đã khớp với ActiveWR, nhưng tín hiệu quá thưa (khoảng 0.5 kill và khoảng 2 bước quyết định mỗi trận). Từ cùng một trạng thái quyết định, cho policy chơi lại K lần ngắn; advantage của mỗi lần = kết quả − trung bình nhóm. Không cần teacher hay value function.

- [x] SD7.0 — spike (RESULTS SD7.0): ở trạng thái search nói có nước giết, F giết được 60%, và 86% trạng thái cho kết quả lúc được lúc không → có tín hiệu để học. Tiêu chí rẻ không lọc bằng search vô dụng.
- [x] SD7.1a — người dùng yêu cầu viết và chạy (18/10/2026); spec `docs/superpowers/specs/2026-10-18-sd7-group-advantage-design.md`; cài `group_rl.py`, trainer `group_rl`, `run_sd7_pilot.py` (12 test mới, suite 381 OK); đo 4.5 s/ep.
- [x] SD7.1 — (DONE) train nhánh group 3000 ep (`runs/sd7-20261018`); best ep 1000, validation ActiveWR 47.75 vs F 47.0; không có xu hướng tăng (RESULTS SD7.1–SD7.3).
- [x] SD7.2 — (KHÔNG ĐẠT: +2.05pp [−0.34, +4.62]) kiểm H1 (`mechanism`, seed 6650–6699): tỉ lệ giết group − F ≥ +10pp, CI dưới > 0.
- [x] SD7.3 — (FAIL: group − F ActiveWR +0.25 [−5.75, +6.25]; group − control −3.25; giữ F) test một lần seed 44000; gate: group − F ActiveWR ≥ +5pp, CI dưới > 0, group − control ≥ 0, tự nổ ≤ +1pp.
- [x] SD7.4 — (ĐẠT: group − control +7.12pp [+4.86, +9.72]; RESULTS SD7.4) probe cơ chế, tín hiệu dày hơn (coef 4, K 16): control và group 1000 ep từ F, `runs/sd74-20261018`; H1' trên seed 6700: group − control ≥ +5pp, CI dưới > 0. Spec `docs/superpowers/specs/2026-10-18-sd7-4-dense-group-probe-design.md`.
- [x] SD7.5 — (FAIL: full − F ActiveWR +3.75 [−2.5, +10.0]; full − control +0.75; best ep 1000; gate giữ F, nhưng người dùng chọn SD7.5 best = G làm model tốt nhất 19/10/2026; RESULTS SD7.5) train full 3000 ep (`run_sd74_probe.py train --arm full`, thư mục `runs/sd75-<thời gian>`), test một lần seed 45000, gate như SD7 (full − F ActiveWR ≥ +5pp, CI dưới > 0, full − control SD5.1 ≥ 0, tự nổ ≤ +1pp).
- [x] SD7.6a — (DONE, RESULTS SD7.6a) chẩn đoán `diagnose_sd75.py` seed 6800: điểm nghẽn là khoảnh khắc "đặt bom ngay" 1 bước (28/37 cửa sổ thất bại, P(bom) F 0.58); 38/57 trận không thắng chủ động không có cơ hội giết; mẫu SD7.5 rơi vào trạng thái đã chắc thắng.
- [x] SD7.6-code — (DONE, IMPROVEMENTS §50) `group_rl.narrow`/`kill_actions`, buffer + advantage bước đầu trong trainer, `run_sd76_pilot.py`; spec `docs/superpowers/specs/2026-10-19-sd7-6-bomb-moment-buffer-design.md`.
- [x] SD7.6-probe — (ĐẠT theo chữ: tỉ lệ giết +18.43pp [+14.58, +22.28], validation −0.5; nhưng kills/trận và ActiveWR không đổi, bom/trận +20%, hòa +6; RESULTS SD7.6-probe) control và buffer 1000 ep từ G (`runs/sd76-<thời gian>`), `mechanism` seed 6850; luật H1‴: buffer − control tỉ lệ giết ≥ +10pp, CI dưới > 0, chết ≤ +1pp, validation buffer − control ≥ −2pp.
- [ ] SD7.6-full — (KHÔNG CHẠY: người dùng chọn sửa tác dụng phụ trước; seed 46000 chuyển cho SD7.7)
- [x] SD7.7-code — (DONE, IMPROVEMENTS §52) `anchor_coef` (KL về G) trong trainer, `run_sd77_pilot.py`; spec `docs/superpowers/specs/2026-10-19-sd7-7-kl-anchor-design.md`.
- [x] SD7.7-calibrate — (KHÔNG ĐẠT: không β nào đưa P(BOMB | hợp lệ) về ≤ 0.687; β 16 còn 0.829; P(bom) và P(nước giết) đi cùng nhau; RESULTS SD7.7-calibrate)
- [ ] SD7.7-probe — (KHÔNG MỞ theo spec) anchor 1000 ep, mechanism seed 6900, luật H1⁗.
- [ ] SD7.7-full — (KHÔNG MỞ) 3000 ep.
- [x] SD7.8 — (ĐẠT: ActiveWR +16.5 [+12.5, +20.5], WR 69.0 → 84.0, hòa −8, tự nổ −0.75; RESULTS SD7.8) G + kill-search lúc chơi, test một lần seed 46000; gate ActiveWR ≥ +5pp, CI dưới > 0, tự nổ ≤ +1pp. Spec `docs/superpowers/specs/2026-10-19-sd7-8-g-plus-search-design.md`. 3000 ep, test một lần seed 46000; gate full − G ActiveWR ≥ +5pp, CI dưới > 0, tự nổ ≤ +1pp.
- [x] SD7.9a — (KHÔNG MỞ TEST: S+search − G+search −0.75 [−6.75, +5.5]; S ≈ G+search về ActiveWR nhưng hòa nhiều hơn; RESULTS SD7.9a) S (SD5.3) vs G làm nền cho search, validation 21000. Spec `docs/superpowers/specs/2026-10-20-sd7-9-two-move-search-design.md`.
- [x] SD7.9b — (ĐẠT luật: +17.5 [+13.75, +21.25]; RESULTS SD7.9b) G + search hai nước (`kill_search.search_action2`), validation 21000.
- [x] SD7.9c — (ĐẠT: ActiveWR 61.0 → 82.0, +21.0 [+17.0, +25.25], WR 80.75 → 93.75, hòa −12.25, tự nổ 0; RESULTS SD7.9c) test một lần seed 47000, G+search2 vs G+search. Agent khuyến nghị mới: G + search hai nước.
- [x] SD8-code — (DONE, IMPROVEMENTS §58) `kill_search.search_action3`/`search_action_deep`, `run_sd8a_distill.py`, `run_sd8b_search3.py`, runner SD5.3 của bạn đưa vào nhánh; spec `docs/superpowers/specs/2026-10-20-sd8-distill-and-deep-search-design.md`, hướng dẫn `SD8_GUIDE.md`; smoke và đo tốc độ (search3 ~23 s/trận vs search2 ~12 s).
- [x] SD8A — (DONE, bạn chạy; Claude kiểm định 22/10: gate ĐẠT, S2 − G +35.0 [+29.0, +41.25], S2 − G+search2 −3.5 [−8.25, +1.25]; RESULTS "SD8A-review") chưng cất G+search2 vào CNN từ G: collect 3000 trận bàn 210000–211499, train 16 epoch, validation 21000 chọn epoch, test một lần seed 48000; gate S2 − G ActiveWR ≥ +5pp, CI dưới > 0, tự nổ ≤ +1pp; luật phụ thay S nếu S2 − S CI dưới > 0.
- [x] SD8B — (ĐẠT, RESULTS SD8B, IMPROVEMENTS §59) validation 21000 +9.25; test 49000 G+search3 ActiveWR 94.0 vs G+search2 85.75, +8.25 [+5.25, +11.5], tự nổ 0; **agent mạnh nhất mới: G + search ba nước**.
- [x] SD8B-off — (DONE, RESULTS SD8B-off, IMPROVEMENTS §60) ablation lịch off trên các bàn 49000: E − off ActiveWR search3 0.0 (399/400 trận giống hệt), search2 0.0 (WR +3.25, hòa −4.0); search3 − search2 ở lịch off +8.25 [+5.25, +11.25]. Sudden death chỉ phân xử ván dài, không ảnh hưởng ActiveWR lúc chơi. Kế hoạch đã chốt: G + search ba nước: validation 21000 (luật ≥ +2pp so với search2), rồi test một lần seed 49000 lịch E; gate ActiveWR ≥ +5pp, CI dưới > 0, tự nổ ≤ +1pp.
- [x] PDE — (DONE 20/10, RESULTS "PDE": E1 G+s3 ActiveWR 94.44, E2 86.25; đấu CNN 1v1 G+s3 WR 19–44, hòa 50–70%; luật Phase E không kích hoạt, luật Phase D kích hoạt vì WR < 60%, chưa train) agent mạnh nhất (G + search3) trong bối cảnh Phase D/E, không train: E2 11×11 4 agent, E1 9×9 3 agent (đấu RuleBot), D 1v1 đấu CNN đông cứng (G, F, S, B, Phase D cũ); seed 50000; spec `docs/superpowers/specs/2026-10-20-pde-check-design.md`, runner `run_pde_check.py`. Luật đề xuất train D/E chốt trước trong spec.
- [x] PDE2-spec — (DONE) spec `docs/superpowers/specs/2026-10-20-pde2-ffa-mix-design.md`: nhiều agent đánh nhau (FFA hỗn hợp 4 ghế G+s3/S2/G/RuleBot, self-play gương 4×S2, đối đầu 1v1), seed 51000, luật đề xuất train D/E chốt trước.
- [x] SDM — (DONE 21/10, luật: mô hình CNN giúp) kill-search mô phỏng đối thủ bằng CNN (oracle) vs RuleBot, G + search3 đấu S và G, validation 21000, 100 trận/nhánh; spec `docs/superpowers/specs/2026-10-20-sdm-search-cnn-opponent-model.md`, runner `probe_search_cnn_model.py`. Oracle − rulebot ActiveWR: vs G +14.0 [+3, +25], vs S +4.0 [−6, +15] (hòa với S không đổi); chậm ~2.4 lần. Xem RESULTS "SDM".
- [ ] ~~SDM-test~~ — (THAY BẰNG SDM2, người dùng đồng ý 22/10) không xác nhận oracle trên test vì oracle cần biết checkpoint đối thủ.
- [x] SDM2 — (DONE 22/10, KHÔNG GIỮ) search luôn đoán đối thủ bằng CNN G (`self`) vs RuleBot, đối thủ RuleBot/S/F, validation 21000, 100 trận/nhánh; spec `docs/superpowers/specs/2026-10-22-sdm2-search-self-cnn-model.md`. ActiveWR self − rulebot: RuleBot −33.0 [−41, −25], S +9.0 [−1, +20], F +12.0 [+3, +22]. Không dùng seed 52000. Xem RESULTS "SDM2".
- [x] PDE2 — (DONE 21/10, runner không có trong repo; Claude review report 22/10) bàn 51000–56099; FFA 9×9 G+s3 ≈ S2 (ActiveWR 9.0 vs 5.0, CI chứa 0) → khuyến nghị S2; luật đề xuất train Phase D và Phase E từ S2 (chỉ đề xuất). RESULTS "PDE2".
- [x] D-S2 (bạn) — (DONE 22–23/10, Claude review 23/10) Phase D từ S2, control RuleBot so với pool CNN, 2000 ep/nhánh: **VALIDATION_STOP**, mọi checkpoint trượt guard RuleBot (ActiveWR 83.75 → 65.0 / 57.5), ActiveWR trước CNN không tăng (6.25 → 6.25 / 7.19), chọn đúng nước search 97.4 → 75.6 / 73.1; test không mở. RESULTS "D-S2-review". Nhánh `origin/Loc_s2-kl-retention` chưa merge.
- [x] SD9.2 — (XONG 24/10, RESULTS "SD9.2": `structural_consider_rules`, P = 15.9 [9.9, 22.7]% trận hòa có kế hoạch giết theo oracle, theo RuleBot 90.3%; đối chứng 20/20; đề xuất thử đổi luật) (code xong 23/10) probe các trận hòa của S3 đấu S2/G/S3 (50 bàn × 2 góc, validation 21000): search3 đoán đối thủ bằng CNN thật có tìm được kế hoạch giết không; đối chứng = trận S3 thắng chủ động (≥ 50% mới hợp lệ); P ≥ 50% → hướng A, P ≤ 20% → xét đổi luật. Spec `docs/superpowers/specs/2026-10-23-sd9-2-draw-kill-probe.md`.
- [x] SD9.3 — (KHÔNG GIÚP 24/10, RESULTS "SD9.3": macro CNN X − E ActiveWR −1.42…−0.17, hòa −3.25…−0.92, tự nổ +1.2…+1.8; 80–87% trận hòa ở lịch sớm là cùng chết ở vòng 3; giữ E, không mở 58000) (code xong 24/10) probe lịch vòng sớm hơn (S400/S350/S300 so với E) cho S3 đấu S2/G/S3/RuleBot, 200 bàn × 2 góc, validation 21000; lịch "giúp" nếu macro CNN hòa ≤ −15pp và ActiveWR ≥ +5pp (CI đúng phía), tự nổ ≤ +1pp, RuleBot ActiveWR ≥ −3pp. Spec `docs/superpowers/specs/2026-10-24-sd9-3-sd-schedule-probe.md`.
- [x] SD10 — (DONE 28/10/2026, ĐẠT gate) chưng cất S3 + search3 đoán đúng đối thủ thành S4; `runs/sd10-20261026`, collect đủ 2000 trận/50 shard, train 16 epoch, validation chọn epoch 12. Test 58000–58199 đã dùng một lần, 4000 trận: macro CNN ActiveWR 24.4167 vs 10.4167%, +14.00pp CI95 [10.9167,17.0833], tự nổ -2.50pp, RuleBot ActiveWR +4.50pp; đạt toàn bộ gate. S4 `train/epoch_12.pt` SHA `90d1d0e7…62bece` đủ điều kiện thí nghiệm để thay S3, chưa đổi model phục vụ/UI. Báo cáo `runs/sd10-20261026/FINAL_REPORT.md`, xuất bản nhánh `Loc/SD10` theo yêu cầu.
- [x] S2-KL (bạn) — (RB_RETENTION_FAIL 23/10, RESULTS "S2-KL-review", nhánh `origin/Loc/s2-kl`: anchor-500 R 77.5 vs S2 97.4, ActiveWR 68 vs 84; anchor − control R +2.5pp; bỏ hướng B) (spec duyệt 23/10) 500 ep đấu RuleBot, KL neo về S2 β = 1 so với 0, đo việc giữ kỹ năng; bàn 710000–710249 / 720000–720099.
- [x] SD9.1 — (KHÔNG ĐẠT 23/10, RESULTS "SD9.1": macro CNN S3 − S2 ActiveWR +3.33 [+1.25, +5.50], hòa −4.08 [−8.17, −0.17]; S3 vẫn hòa 53–64% khi đấu CNN; không mở 58000; chọn A/B sau S2-KL) probe S3 so với S2 khi đấu CNN S2/G/S3 (200 bàn × 2 góc) và G + search3 (50 bàn × 2 góc), validation 21000; luật: "S3 bớt hòa rõ" nếu macro CNN hòa ≤ −15pp (CI trên < 0) và ActiveWR ≥ +5pp (CI dưới > 0). Spec `docs/superpowers/specs/2026-10-23-sd9-1-s3-vs-cnn-probe.md`.
- [ ] Phase D/E từ S3 — (COND) cần cơ chế giữ kỹ năng chưng cất (xem D-S2, S2-KL) trước khi spec.
- [x] SD9.0 — (DONE 23/10) probe S2 + search3 vs G + search3, validation 21000; ActiveWR S2+s3 − G+s3: RuleBot −5.0 [−11, +1], S2 −2.0 [−11, +7], G +7.0 [−2, +16]. Luật: thầy SD9 = S2 + search3; G + search3 vẫn là agent mạnh nhất. RESULTS "SD9.0".
- [x] SD9 — (DONE 23/10, **ĐẠT**) chưng cất S2 + search3 thành S3 (`runs/sd9-20261023/train/epoch_12.pt`, `4185375f…69ed7`) từ S2, test 57000: S3 92.25 vs S2 81.75 ActiveWR, +10.5 [+6.5, +14.5], tự nổ −0.25; S3 − G + search3 −1.75 [−5.0, +1.5]. S3 là CNN thuần mạnh nhất. RESULTS "SD9".

## Bảo trì workspace — 24/10/2026

- [x] Pull nhánh `danh/sd9-s3-distillation` tại `6b49216`, giữ nguyên `game UI`, lưu thay đổi local cũ trong stash; đủ 327 file LFS. Bằng chứng: RESULTS `SYNC-SD9-S3` và `.codex-recovery/pull-sd9-s3-20261024-035246/verification.json`.


## S3-11 giữ kỹ năng9×9 — 27/10/2026

- [x] S3-11-9X9-RULEBOT: DONE27/10/2026,800/800game/200bàn840000–840199×2góc,RuleBot9×9/max500/E475-25-40; S3-11 WR94.75/Active87.50/hòa3.75/tựnổ.25% vs S3 98.25/93.25/1/0%; Active−5.75ppCI[-9.75,-1.5]. Vẫn thắng nhiều nhưng kémS3gốc9×9; giữ model phục vụ, khôngtrain/push, kết thúc/gỡ heartbeat45phút. RESULTS `S3-11-9X9-RULEBOT-FINAL`, IMPROVEMENTS §104, run FINAL_REPORT.md.
