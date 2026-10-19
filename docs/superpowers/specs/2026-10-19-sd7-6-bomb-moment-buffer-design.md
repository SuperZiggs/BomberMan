# SD7.6 — Group RL tập trung vào khoảnh khắc "đặt bom ngay" (buffer), từ G

Ngày: 19/10/2026. Người dùng duyệt hướng "buffer khoảnh khắc bom". Luật dưới đây chốt **trước** khi train.

## Bối cảnh (RESULTS SD7.6a)

- 31/37 cửa sổ giết thất bại của F chỉ dài 1 bước; ở 28 cửa sổ, nước giết duy nhất là đặt bom ngay, F gán P ≈ 0.58 (G 0.63).
- SD7.5 dồn mẫu vào trạng thái đã chắc thắng (79% lần chơi lại giết được, 52% nhóm có chênh lệch), bộ lọc rẻ loại các trạng thái chưa có bom cháy, và advantage được rải đều lên mọi bước của lần chơi lại.
- Trần của mọi cách sửa kết liễu khoảng +19pp ActiveWR (38/57 trận không thắng chủ động không có cơ hội giết).

## Thay đổi (chỉ khi bật, mặc định giữ nguyên SD7)

1. **Chọn trạng thái hẹp** (`group_select='narrow'`, `group_rl.narrow`): learner còn sống, BOMB hợp lệ trong mask, có đối thủ trong khoảng Manhattan ≤ 4; kill-search cho tập nước giết (điểm = 1) có 1..2 phần tử. Không cần bom cháy.
2. **Buffer** (`group_buffer=N`, N = 1000): trạng thái hẹp mới của episode được thêm vào buffer FIFO; mỗi episode chơi lại các trạng thái mới, rồi lấy thêm ngẫu nhiên (`group_rng`) từ buffer cho đủ `group_max_states` (6). Buffer được lưu trong training_state để resume chính xác.
3. **Advantage chỉ ở bước đầu** (`group_first_step=True`): loss = −mean(A · log π(a₀|s)) trên các lần chơi lại của nhóm có chênh lệch.
4. Còn lại như SD7.5: K 16, H 20, `group_coef` 4.0, `group_gap` 4, config control SD5.1, nhưng `resume_from` = **G**.

## Probe SD7.6 (1000 ep, không validation khi train)

- Hai nhánh từ G, train seed 2026101701: `control` (config control SD5.1, không group) và `buffer` (thêm các khóa trên).
- Cơ chế (seed chẩn đoán mới 6850–6899, 100 trận, G chơi, lịch E): thu trạng thái hẹp (tối đa 6/trận, giãn cách 4); G, control, buffer (final) mỗi model chơi lại K 8 × H 20 với cùng seed lấy mẫu; đo tỉ lệ giết và P(tập nước giết).
- Kiểm thêm trên validation 21000 (400 trận, lịch E): ActiveWR, số bom/trận của control và buffer final.
- **Luật đạt (H1‴):** buffer − control tỉ lệ giết ở trạng thái hẹp ≥ +10pp với CI95 dưới > 0 (bootstrap theo trạng thái); chết lúc chơi lại ≤ control + 1pp; và trên validation 21000 buffer − control ActiveWR ≥ −2pp (bootstrap theo cặp bàn, điểm ước lượng).
- Đạt → SD7.6 full. Không đạt → ghi lại, không thử lại trên cùng seed chẩn đoán.

## SD7.6 full (chỉ khi probe đạt)

- 3000 ep từ G với config nhánh buffer, validation 21000 mỗi 500 ep chọn best (như SD7.5).
- Test **một lần** seed **46000**, 400 trận/model, lịch E (gate) và off (báo cáo): G và full best.
- **Gate:** full − G ActiveWR ≥ +5pp, CI95 dưới > 0, tự nổ ≤ +1pp.

## Giới hạn đã biết

Một train seed; nhãn "nước giết" dùng search 12 tick với RuleBot hoàn hảo; trần ~+19pp.
