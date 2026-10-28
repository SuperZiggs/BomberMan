# SD10 — kết quả cuối, 28/10/2026

**Kết luận:** S4 tại epoch 12 đạt toàn bộ gate đã đăng ký trên test seed 58000. ActiveWR macro CNN tăng 14.00 điểm phần trăm so với S3, CI95 [10.92, 17.08]; tự nổ giảm 2.50 điểm phần trăm và ActiveWR trước RuleBot tăng 4.50 điểm phần trăm.

## Thiết kế và dữ liệu

**Thông số:** Codex và người dùng chạy tại `E:\Bomberman`, Windows, Python 3.11.9 / PyTorch 2.11.0+cu128 / NumPy 2.4.6. Chưng cất offline từ S3 `runs/sd9-20261023/train/epoch_12.pt` (SHA256 `4185375f…69ed7`), không phải train RL theo episode. Bàn 9×9, 2 agent, max_steps 500; lịch E = 475/25/40, bật ở 75% bàn collect và toàn bộ đánh giá. Thu 1000 bàn 214000–214999 × 2 góc = 2000 trận, chia đều 500 trận trước RuleBot/S2/G/S3; thầy S3 lấy mẫu + search3, mô hình đối thủ CNN dùng argmax đã mask, đối thủ thực tế lấy mẫu. Collect CPU, khởi đầu 5 workers rồi resume với 8; 50 shard × 40 trận. Train CUDA RTX 4060 Laptop GPU, Adam lr 1e-4, batch 256, 16 epoch, CE mục tiêu mềm + KL neo về S3 với lambda 1, held-out 10% bàn, RNG chia dữ liệu và minibatch 1010; torch train seed riêng không lưu lại. Validation 21000–21199, 5 model × 4 đối thủ × 400 trận = 8000 trận; chọn trong epoch 4/8/12/16 theo luật đã đăng ký. Test 58000–58199, S4/S3 × S2/G/S3/RuleBot/F × 400 trận = 4000 trận; validation và test CPU 5 workers, CNN thuần lấy mẫu + mask. Bootstrap ghép cặp theo 200 bàn, gồm cả 2 góc, 10000 lần, seed 20261014.

- Thu đủ 2000 trận, 7078 trạng thái quyết định và 32072 trạng thái neo; hoàn tất lúc 13:30 ngày 28/10. Có các lần ngắt/resume, không lưu tổng thời gian CPU thực tế.
- Train hoàn tất 16 epoch; checkpoint ứng viên 4/8/12/16. Dựa trên mốc stage và file, train khoảng 22 giây, validation khoảng 16 phút và test khoảng 9 phút; đây là thời gian wall-clock xấp xỉ, không phải profiler.
- Validation chọn epoch 12: ActiveWR macro CNN 26.25% so với S3 9.9167%, đủ điều kiện mở test. Test hoàn tất lúc 14:48 ngày 28/10.
- Checkpoint S4: `train/epoch_12.pt`; SHA256 `90d1d0e70c64f879e07f85c68c003969d082043b3cd8065314a3c35faf62bece`.

## Kết quả test 58000

| Chỉ số (%) | S3 | S4 | Delta S4 − S3 (điểm %) | CI95 delta |
|---|---:|---:|---:|---|
| ActiveWR macro CNN | 10.4167 | 24.4167 | +14.00 | [10.9167, 17.0833] |
| WR macro CNN | 27.0833 | 42.4167 | +15.3333 | [11.5833, 19.0833] |
| Hòa macro CNN | 56.6667 | 51.7500 | -4.9167 | [-9.0833, -0.7500] |
| Tự nổ macro CNN | 5.5000 | 3.0000 | -2.50 | [-4.3333, -0.7500] |
| ActiveWR trước RuleBot | 87.7500 | 92.2500 | +4.50 | [1.00, 8.00] |

Macro CNN là trung bình ngang nhau trên S2/G/S3. Đối thủ F không nằm trong gate.

| Đối thủ | ActiveWR S3 (%) | ActiveWR S4 (%) | Delta (điểm %) | CI95 delta |
|---|---:|---:|---:|---|
| S2 | 11.25 | 26.00 | +14.75 | [9.50, 20.00] |
| G | 10.25 | 25.00 | +14.75 | [10.00, 20.00] |
| S3 | 9.75 | 22.25 | +12.50 | [7.50, 17.50] |
| RuleBot | 87.75 | 92.25 | +4.50 | [1.00, 8.00] |
| F, chỉ mô tả | 15.00 | 30.00 | +15.00 | [9.25, 20.75] |

## Gate và quyết định

- ActiveWR macro CNN tăng ít nhất 5 điểm %: đạt (+14.00).
- Cận dưới CI95 của mức tăng lớn hơn 0: đạt (+10.9167).
- Tự nổ macro CNN không tăng quá 1 điểm %: đạt (-2.50).
- ActiveWR trước RuleBot không giảm quá 3 điểm %: đạt (+4.50).

S4 đạt điều kiện thí nghiệm để thay S3. Lượt này tạo checkpoint và báo cáo; không đổi cấu hình model phục vụ hoặc game UI. Seed test 58000–58199 đã dùng một lần, không chạy lại hoặc chọn checkpoint theo test.

## Bằng chứng và giới hạn

Nguồn gốc: `registration.json`, `collection_summary.json`, 50 file `collection/shard-*.npz`, `train/metrics.json`, `validation.json`, `selected.json`, `test_opened.json`, 10 file `evaluation_*_58000.json`, `summary.json`. Script điều phối và log ở thư mục cạnh bên `../sd10-20261026-control/`. ZIP gốc khi hoàn tất: `../sd10-20261026-send.zip`; báo cáo này được bổ sung sau khi tạo ZIP.

Kết luận giới hạn ở 9×9, 1v1, lịch E và một lượt chưng cất. S2/G/S3 xuất hiện cả trong collect và đánh giá, nhưng khác bàn; F là đối thủ ngoài collect, chỉ dùng mô tả. Hòa macro CNN vẫn 51.75%. Chưa xác nhận nhiều seed train, FFA hoặc map 11×11. Không so trực tiếp tỷ lệ của test này với test cũ khác seed/runtime.

Nhánh xuất bản theo yêu cầu: `Loc/SD10`. Các checkpoint `.pt` dùng Git LFS theo cấu hình sẵn có của repo.
