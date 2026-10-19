# SD7.4 — Probe cơ chế: advantage theo nhóm với tín hiệu dày hơn

Ngày: 18/10/2026. Trạng thái: người dùng yêu cầu viết spec và chạy luôn. Mọi luật dưới đây được chốt **trước** khi train.
Liên quan: spec SD7 (`2026-10-18-sd7-group-advantage-design.md`), RESULTS SD7.1–SD7.3, `INSIGHTS.md`.

## 1. Lý do

- SD7 (K = 8, `group_coef` 1.0) không đạt: tỉ lệ giết trên trạng thái quyết định group − F = +2.05pp, nhưng control − F = +1.71pp, nên phần do cơ chế nhóm chỉ là group − control = +0.34pp [−1.03, +1.71].
- Metric của lượt SD7: trung bình 0.75 trạng thái/ep, 0.59 nhóm có chênh lệch/ep, 52 bước chơi lại/ep; chỉ 43/3000 ep chạm giới hạn 3 trạng thái. Tín hiệu nhóm rất thưa so với loss RL và teacher (khoảng 350 bước/ep).
- Giả thuyết: cơ chế đúng, nhưng tín hiệu quá yếu. Nếu tăng trọng số và số lần chơi lại thì group − control phải tách ra rõ ràng.

## 2. Giả thuyết và luật (chốt trước)

- **H1' (cơ chế, so với control cùng số episode):** sau 1000 ep từ F, tỉ lệ giết trên trạng thái quyết định của group − control ≥ **+5pp**, CI95 bootstrap theo trạng thái có cận dưới > 0, và tỉ lệ chết trong lúc chơi lại của group ≤ control + 1pp.
- Đo trên seed chẩn đoán **mới 6700–6749** (100 trận, F tự chơi, lịch E), giống hệt cách đo của SD7 (K = 8, H = 20, tối đa 3 trạng thái/trận, giãn cách 8; seed lần chơi lại theo (trạng thái, k), chung cho mọi model). Không dùng 6650 vì đã đọc kết quả SD7 trên đó.
- Báo cáo thêm group − F và control − F.
- **Nếu H1' đạt:** mở SD7.5 = train full 3000 ep từ F với config group của SD7.4 (validation 21000 chọn best như SD7), test **một lần** trên seed **45000**, gate giống SD7 (group − F ActiveWR ≥ +5pp, CI dưới > 0, group − control SD5.1 ≥ 0, tự nổ ≤ +1pp).
- **Nếu H1' không đạt:** dừng hướng SD7 (advantage theo nhóm qua loss phụ), ghi vào INSIGHTS là đã bác bỏ.
- Chỉ có một cấu hình group, không quét nhiều cấu hình rồi chọn cái tốt nhất.

## 3. Thiết kế

- Cả hai nhánh: config control SD5.1 (`runs/sd5-20261017/control/config.json`), từ F, train seed 2026101701, nhưng `episodes = 1000`, `validation_every = 0` (không validation, dùng `final_phaseC.pt`).
- **control74:** không có `group_*`.
- **group74:** `group_rl = True`, `group_coef = 4.0`, `group_k = 16`, `group_horizon = 20`, `group_max_states = 6`, `group_gap = 4`.
  - K 8 → 16: ước lượng trung bình nhóm ít nhiễu hơn, nhiều nhóm có chênh lệch hơn.
  - coef 1 → 4: tín hiệu nhóm không bị loss chính lấn át.
  - Giãn cách 8 → 4 và tối đa 6 trạng thái: thu thêm một ít trạng thái (giới hạn cũ hiếm khi chạm).
- Không đổi code trainer, `group_rl.py`, env, RuleBot, reward.
- Runner: `run_sd74_probe.py` (`train --arm control|group [--resume]`, `mechanism`), tái dùng `run_sd7_pilot.mechanism_shard` và `paired_states`.

## 4. Chi phí

- Smoke đo s/ep trước. Ước: control khoảng 2.8 s/ep (khoảng 47 phút), group khoảng 8–12 s/ep (khoảng 2.5–3.5 giờ). Hai nhánh chạy song song (2 tiến trình).
- Mechanism: khoảng 2–3 phút.

## 5. Giới hạn đã biết

- Một train seed; 1000 ep ngắn hơn lượt full.
- H1' là điều kiện cần cho ActiveWR, không phải điều kiện đủ (SD7: +2pp tỉ lệ giết chỉ tương ứng khoảng 1pp ActiveWR).
