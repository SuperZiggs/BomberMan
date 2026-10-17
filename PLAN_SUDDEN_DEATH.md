# Plan và tracker: hướng Sudden death (SD)

File này chỉ để **theo dõi công việc** (đánh dấu `[x]` khi xong, có bằng chứng). Số liệu, bảng kết quả và quyết định ghi trong `RESULTS.md`; thay đổi code/phương pháp ghi trong `IMPROVEMENTS.md`; quy ước bền vững ghi trong `CLAUDE.md`. Tracker tổng của dự án: `PLAN_BOMBERMAN.md`.

Trạng thái: **DONE** = xong (kể cả gate FAIL); **READY** = làm được ngay; **WAIT** = chờ người dùng chạy/kết quả; **COND** = chỉ mở khi điều kiện ghi kèm thỏa; **DEFERRED** = để sau.

Cập nhật: 18/10/2026. Model đang giữ: B `runs/pressure-C-next/final_phaseC.pt` (đến SD4.6); F `runs/phaseC-sd-best-fallback/best_phaseC.pt` cho SD5. Seed test tiếp theo: 43000 (42000: SD5.1). Cấu hình SD hiện dùng: SD2 + biên mask learner k=15.

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
- [ ] SD5.2 — (COND) đòn bẩy trận hòa: ưu tiên lại bàn F hay hòa/thua khi train (kiểu PLR), chỉ mở sau SD5.0/SD5.1.
