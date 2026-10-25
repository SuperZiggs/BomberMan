# S3 11×11 — yêu cầu bước 1

Người dùng đã yêu cầu thực hiện bước 1 của quy trình được trình bày trong chat: chuẩn bị runner riêng cho chưng cất S3 trên map 11×11. Phạm vi đã được xác nhận bằng yêu cầu “hãy làm từng bước 1”; không tự chạy baseline, collect hoặc train dài.

## Hợp đồng và điều kiện nghiệm thu

- File mới `run_s3_map11_distill.py`; không sửa source runner/engine/loader cũ hoặc checkpoint. Dùng 11×11, 13 kênh, 2 agent và RuleBot xuyên suốt thu dữ liệu, mạng thầy/mạng học, baseline, validation, test.
- S3 gốc: `runs/sd9-20261023/train/epoch_12.pt`, SHA-256 `4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7`. Nạp trên mạng 11×11; không ghi đè checkpoint. Thầy = S3 + search3; learner và các trạng thái neo xuất phát từ S3 sau chuyển map. Baseline là S3 trên 11×11.
- Làm rõ baseline theo quy trình người dùng đã chọn: stage `baseline` so sánh cả S3 thuần và S3+search3, mỗi agent chạy `baseline_games` trên cùng seed bàn và cả hai góc; smoke là 4 trận/agent, tổng 8 lượt env. Cả hai dùng lịch E; báo cáo lợi thế của thầy trước khi quyết định thu dữ liệu. Đây là bổ sung làm rõ phạm vi đã trình bày, không phải thay thuật toán để khớp test.
- CLI có các stage `init`, `baseline`, `collect`, `train`, `validate`, `test`. Import module, `--help` và `init` không chạy game hoặc cập nhật gradient. Chỉ gọi stage cụ thể mới làm công việc tương ứng.
- API công khai: `config_for(output, *, collect_first_seed, collect_boards, validation_seed, test_seed, smoke=False)`, `initialize(output, *, collect_first_seed, collect_boards=100, validation_seed=21000, test_seed, smoke=False)`, `load_net(path, device='cpu')`, `baseline(output, workers=5)`, `collect(output, workers=5, resume=False)`, `train(output, device='cpu')`, `validate(output, workers=5)`, `test(output, workers=5)`, `main(argv=None)`.
- `init` yêu cầu khai báo seed collect và test; không tự chiếm một seed test đã dùng. Các stage sau đọc cấu hình đã đăng ký. Từ chối output tồn tại, khoảng seed âm/rỗng, collect ít hơn 2 bàn, và giao nhau giữa tập collect/validation/test. Baseline/validation dùng cùng tập validation; test dùng tập riêng.
- Cấu hình ban đầu: full max_steps 500, E 475/25/40, xác suất E khi collect 0.75, neo 0.05; mặc định collect 100 bàn × 2 góc = 200 trận pilot, baseline 100 trận, validation/test 400 trận; Adam 1e-4, batch 256, 16 epoch, ứng viên 4/8/12/16, KL neo 1, held-out 10% theo bàn. Đây là cấu hình chạy trong tương lai, không phải kết quả đã kiểm định. Smoke riêng: 4 bàn collect, max_steps 12, 4 trận mỗi phép đo, batch 16, 2 epoch, ứng viên [2]; smoke không dùng seed test thật.
- Registration lưu cấu hình, SHA checkpoint, hash source và runtime; stage sau từ chối thay đổi nguồn/checkpoint/cấu hình. Collection có shard có metadata, từ chối shard hỏng và chỉ resume phần thiếu; không tự ghi đè artifact đã hoàn tất. Có progress cho thao tác dài; số worker 1–5.
- Train chỉ chưng cất nhãn quyết định và KL trên neo mới của 11×11; không RL, không augmentation cố định 9×9, không dùng lại logits/observation cũ 9×9. Giữ critic_head không cập nhật như SD9. Sai shape dữ liệu phải bị từ chối. Không tách trạng thái cùng bàn sang cả train và held-out.
- Validation chọn ActiveWR cao nhất, tự nổ không quá baseline S3 + 1pp, hòa điểm chọn epoch sớm hơn; không có ứng viên hơn baseline thì không mở test thật. Test một lần, so với baseline S3 trên 11×11; gate tham khảo SD9: ActiveWR +5pp, CI95 dưới > 0, tự nổ không quá +1pp. Smoke được ghi rõ không phải kết quả test thật.
- Hướng dẫn cmd trong file mới `S3_MAP11_GUIDE.md`; nói rõ chuẩn bị runner là bước 1, các lệnh công việc dài dành cho các bước sau. Ghi nhật ký và tracker theo AGENT.md.

## Không thuộc phạm vi

Không chạy thí nghiệm dài, không thay đổi lịch SD/reward/RuleBot, không thay model đang phục vụ, không merge/push/commit tự động, không tuyên bố model 11×11 mạnh hơn.

## Kiểm chứng

Một agent độc lập viết acceptance matrix và test trước production code, ghi RED vì module/chức năng chưa có. Có kiểm tra tích hợp ngắn dùng mạng/env thật trên 11×11 và dữ liệu tổng hợp cho train, không tiêu thụ test seed thật. Chạy suite mới, regression runner SD9/SD53/SD8, rồi suite unittest toàn repo; phân biệt lỗi có sẵn. Reviewer độc lập đọc phần thay đổi cuối cùng.
