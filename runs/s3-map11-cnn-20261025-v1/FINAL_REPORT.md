# S3-11 đấu CNN — kết quả cuối, 25/10/2026

**Thông số:** Codex/Windows `E:\Bomberman`, Python3.11.9/torch2.11.0+cu128/NumPy2.4.6; S3-11 `runs/s3-map11-20261024-auto-v1/train/epoch_16.pt` SHA `17748471…a95f`, S3 `runs/sd9-20261023/train/epoch_12.pt` SHA `4185375f…69ed7`, S2 `runs/sd8a-20261020/train/epoch_16.pt` SHA `d222d81f…6814`, G `runs/sd75-20261018/full/best_phaseC.pt` SHA `bb0c775c…c7ac`; hai learner S3_11/S3 cùng pool S3/S2/G; seed830000–830199,200bàn×2góc=400trận/cặp,6cặp=2400trận; 11×11/13kênh/2agent/max500/E475-25-40; CNN CPU lấy mẫu+mask, không search/RuleBot/gradient/train; 5worker/1torchthread,10bàn/shard/120shard. Paired bootstrap theo200bàn/10000lần/seed20261014; macro trung bình đều3opponent ở mỗi(seed,corner) trước bootstrap. Từ01:39:46 đến lúc ghi summary02:02:40ngày25/10/2026, khoảng1373.79s (22.90phút); tổng thời gian shard6765.86s là tổng các worker, không phải wall-time. Một lần launch, không resume hoặc thêm trận.

**Gate: không đạt mức cải thiện đã chốt.** Macro ActiveWR tăng3.83pp, CI95[1.92,5.83] dương, nhưng point estimate chưa đạt ngưỡng≥5pp. Tự nổ tăng0.25pp, nằm trong giới hạn≤1pp theo point estimate đã đăng ký; CI tự nổ[-0.75,1.17] vẫn cho thấy bất định. WR tăng1.33pp với CI95[-1.83,4.50], chưa có bằng chứng chắc cho tăng WR. ActiveWR trước S3/S2/G lần lượt9.50/8.75/7.50%; hòa62.25/64.25/66.50%, phần lớn hết500bước. Không sửa ngưỡng, không mở thêm game để cứu gate, không train/promote.

## Macro trên pool chung

| Chỉ số (%) | S3-11 | S3 gốc chuyển map | Delta (pp) | CI95 delta (pp) |
|---|---:|---:|---:|---|
| WR | 23.00 | 21.67 | +1.33 | [-1.83, 4.50] |
| ActiveWR | 8.58 | 4.75 | +3.83 | [1.92, 5.83] |
| Hòa | 64.33 | 63.08 | +1.25 | [-2.50, 5.08] |
| Hòa hết bước | 61.58 | 62.17 | -0.58 | [-4.42, 3.33] |
| Thua | 12.67 | 15.25 | -2.58 | [-5.25, -0.00] |
| Tự nổ | 1.58 | 1.33 | +0.25 | [-0.75, 1.17] |
| Thắng thụ động | 0.75 | 0.67 | +0.08 | [-0.67, 0.75] |
| Thắng nhờ vòng | 13.67 | 16.25 | -2.58 | [-5.25, 0.08] |
| Chết vì vòng | 13.75 | 13.08 | +0.67 | [-2.08, 3.33] |

## Sáu cặp đối đầu

| Learner | CNN đối thủ | Trận | WR% | ActiveWR% | Hòa% | Hết bước% | Thua% | Tự nổ% | Thụ động% | Thắng vòng% | Chết vòng% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S3_11 | S3 | 400 | 26.25 | 9.50 | 62.25 | 60.50 | 11.50 | 1.50 | 0.50 | 16.25 | 13.75 |
| S3_11 | S2 | 400 | 19.75 | 8.75 | 64.25 | 62.00 | 16.00 | 2.00 | 1.50 | 9.50 | 15.25 |
| S3_11 | G | 400 | 23.00 | 7.50 | 66.50 | 62.25 | 10.50 | 1.25 | 0.25 | 15.25 | 12.25 |
| S3 | S3 | 400 | 21.25 | 5.75 | 57.50 | 57.00 | 21.25 | 2.50 | 0.50 | 15.00 | 18.00 |
| S3 | S2 | 400 | 17.50 | 4.25 | 68.25 | 67.00 | 14.25 | 1.00 | 0.50 | 12.75 | 12.25 |
| S3 | G | 400 | 26.25 | 4.25 | 63.50 | 62.50 | 10.25 | 0.50 | 1.00 | 21.00 | 9.00 |

## Paired delta theo đối thủ

| Đối thủ | Chỉ số | Delta S3-11 − S3 (pp) | CI95 (pp) |
|---|---|---:|---|
| S3 | WR | +5.00 | [-0.75, 11.00] |
| S3 | ActiveWR | +3.75 | [0.50, 7.00] |
| S3 | Hòa | +4.75 | [-3.75, 13.50] |
| S3 | Hòa hết bước | +3.50 | [-5.25, 12.25] |
| S3 | Thua | -9.75 | [-14.50, -5.00] |
| S3 | Tự nổ | -1.00 | [-3.00, 1.00] |
| S3 | Thắng thụ động | +0.00 | [-1.00, 1.00] |
| S3 | Thắng nhờ vòng | +1.25 | [-3.75, 6.00] |
| S3 | Chết vì vòng | -4.25 | [-9.50, 0.75] |
| S2 | WR | +2.25 | [-3.00, 7.50] |
| S2 | ActiveWR | +4.50 | [1.00, 8.00] |
| S2 | Hòa | -4.00 | [-10.00, 2.25] |
| S2 | Hòa hết bước | -5.00 | [-11.25, 1.25] |
| S2 | Thua | +1.75 | [-3.25, 6.75] |
| S2 | Tự nổ | +1.00 | [-0.75, 2.75] |
| S2 | Thắng thụ động | +1.00 | [-0.25, 2.50] |
| S2 | Thắng nhờ vòng | -3.25 | [-7.50, 1.00] |
| S2 | Chết vì vòng | +3.00 | [-1.50, 7.75] |
| G | WR | -3.25 | [-9.50, 2.75] |
| G | ActiveWR | +3.25 | [0.00, 6.50] |
| G | Hòa | +3.00 | [-3.75, 9.75] |
| G | Hòa hết bước | -0.25 | [-7.00, 6.50] |
| G | Thua | +0.25 | [-3.75, 4.25] |
| G | Tự nổ | +0.75 | [-0.50, 2.00] |
| G | Thắng thụ động | -0.75 | [-2.00, 0.25] |
| G | Thắng nhờ vòng | -5.75 | [-11.00, -0.50] |
| G | Chết vì vòng | +3.25 | [-1.00, 7.25] |

## Xác thực

Xác minh native: root/interpreter/children của run không còn, lock đã được runner gỡ, stderr0byte. API `probe_s3_map11_cnn.run(output,workers=5,resume=True)` kiểm lại report từ raw đầy đủ, chặn mọi lời gọi chơi game mới; summary không đổi SHA. Đủ120shard/6cặp×400trận, 200seed×2góc/cặp, marker830000 mở đúng một lần,18source/live+bundle/config/runtime/4checkpoint đúng provenance. Summary SHA256 `4773f66560e59880c1eef24628e9bdb2c8a7d7f75c7fe270ba4691a3eb13c665`; proof `completion-verification.json`.

## Giới hạn và quyết định

Kết luận áp dụng11×11/1v1 trên ba CNN9×9 được chuyển map bằng loader; chưa đại diện CNN train native11×11, FFA hoặc độ ổn định nhiều train seed. Không so chênh lệch với RuleBot như cùng một đối thủ. S3 gốc9×9 và model phục vụ giữ nguyên. Các CI theo từng chỉ số/đối thủ là CI riêng, không điều chỉnh nhiều so sánh. CI thua macro[-5.25,0.00] chạm0 sau làm tròn, không diễn giải epsilon âm như bằng chứng tách khỏi0.

Theo dõi45phút đã hoàn thành mục tiêu. State cuối `completed_no_qualified_gain`; heartbeat `s3-11-u-cnn-ki-m-tra-45-ph-t` đã được xóa, tool xác nhận `deleteStatus=deleted`, lưu tại `automation-delete-confirmation.json` và state. Không có bước huấn luyện kế tiếp được tự mở.
