# S3-11 — hoàn tất pilot 11×11, 24/10/2026

**Kết quả: PASS toàn bộ gate test đã chốt.** Đã hoàn tất bước2–5, không đổi source/config trong run hoặc ghi đè S3 gốc. Model được chọn cho thí nghiệm11×11 là `train/epoch_16.pt`, SHA256 `17748471874686388f1a9b373b0debdfe53ba04a9b1c821a9695efdd8825a95f`.

## Test cuối một lần

400trận/model =200bàn820000–820199×2góc; S3 sau nội suy lên11×11 so với S3-11 epoch16, cùng RuleBot, map11×11/13kênh/2agent/max500/E475-25-40; CNN thuần lấy mẫu+mask, không gọi search lúc đánh giá. Windows/CPU2worker thực tế/OMP-MKL1thread, torch2.11.0+cu128/NumPy2.4.6. Launch→summary322.358s (~5phút22giây), artifact hoàn tấtUTC14:46:18.636803. Paired bootstrap200cặp bàn/10000lần/seed20261014.

| Chỉ số | S3 nội suy, chưa học11×11 | S3-11 epoch16 | Chênh lệch S3-11−S3 |
|---|---:|---:|---:|
| WR | 59.25% | 90.75% | +31.50pp, CI95[25.75,37.25] |
| ActiveWR | 46.25% | 87.25% | +41.00pp, CI95[34.50,47.25] |
| Hòa | 30.75% | 6.00% | −24.75pp, CI95[−30.00,−19.50] |
| Tự nổ | 0.75% | 1.25% | +0.50pp, CI95[−1.00,2.00] |
| Chết vì SD | 6.75% | 1.50% | −5.25pp, không tính CI riêng |
| Thắng thụ động | 2.00% | 2.00% | 0pp |

Gate đã đăng ký: ActiveWR≥+5pp (**+41, PASS**), CI95 ActiveWR dưới>0 (**34.5, PASS**), tự nổ tăng≤1pp (**+0.5, PASS**). Guard tự nổ là chênh lệch điểm ước lượng, không phải điều kiện lên cận trên CI; không đổi luật sau khi đọc test.

Đã xác thực đủ400cặp seed/góc mỗi model, không trùng/thiếu; tính lại summary/paired bootstrap/gate từ raw games và khớp `summary.json`. Marker test, selection và hash model nhất quán; S3/source/config gốc còn đúng SHA. Test seed820000–820199 đã dùng, không mở lại.

## Các bước đã thực hiện

1. Baseline100trận/agent: S3 ActiveWR50%, S3+search3 98%; delta+48pp[38,58], tự nổ đều0%. Gate thu dữ liệu PASS; baseline2429.969s.
2. Collect200trận812000–812099×2góc, pE.75/off.25/neo.05:746decision/1310anchor, teacher ActiveWR98%. Split90/10bàn, decision687/59 và anchor1161/149; không giao nhau theo bàn. Launch→artifact2463.052s.
3. Chưng cất16epoch từ S3 trên CUDA RTX4060 Laptop8GB; Adam1e-4/batch256/KL1/RNG530/held-out10%bàn. Lưu epoch4/8/12/16, critic_head giữ nguyên, actor cập nhật; launch→metrics5.153s, không có timer riêng cho loop.
4. Validation400trận/model×5 trên21000–21199×2góc: S3 ActiveWR45%, candidate4/8/12/16 đạt78.25/82/86/87%. Epoch8 bị loại vì tự nổ1.25%>baseline+1pp; chọn epoch16 theo ActiveWR và guard tự nổ≤1pp. Launch→artifact870.340s.
5. Test đúng một lần trên tập mới, kết quả trong bảng trên. Không chọn lại epoch hoặc retrain sau test.

## Artifact và phạm vi kết luận

- Model: `E:\Bomberman\runs\s3-map11-20261024-auto-v1\train\epoch_16.pt`.
- Bằng chứng: `registration.json`, `baseline.json`,4shardcollection,`train/metrics.json`,`validation.json`,`selected.json`,`test_opened.json`,`summary.json`, các file `*_review.json` và stdout/stderr từng stage.
- Chỉ xác nhận lợi ích trên **11×11, 1v1 RuleBot**, một lượt train và một tập test200bàn. Chưa đánh giá FFA, đối thủ CNN, lịch SD khác hoặc khả năng giữ kỹ năng9×9 của checkpoint mới.
- Model11×11 được lưu riêng; S3 gốc9×9 và model đang phục vụ không đổi. Lịch kiểm tra45phút đã được gỡ bằng automation_update (deleteStatus=deleted) vì quy trình đã kết thúc; truy vấn native khi chốt không còn runner của run này.
