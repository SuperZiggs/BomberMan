# S4-11 — hồ sơ yêu cầu và ma trận kiểm thử dự thảo

**Trạng thái:** yêu cầu trích từ thiết kế đã được người dùng duyệt. Bảng này do agent triển khai chuẩn bị, **chưa phải bộ test do tác giả độc lập viết**, chưa có test acceptance mới hoặc RED baseline. Mục đích là tài liệu hóa đầu vào/đầu ra và tình huống để tác giả test độc lập sử dụng, hoặc làm fallback nếu người dùng cho phép.

## Hợp đồng người dùng nhìn thấy

- Huấn luyện S4 9×9 sang model CNN thuần 11×11/13 kênh để đấu CNN 1v1, dùng RuleBot làm guard. S4 khởi tạo `runs/sd10-20261026/train/epoch_12.pt`, SHA `90d1d0e70c64f879e07f85c68c003969d082043b3cd8065314a3c35faf62bece`; không đổi file nguồn/checkpoint cũ hoặc thay model phục vụ.
- Pool collect/đánh giá chính: S2, G, S3-11; 333 bàn × 2 góc × 3 đối thủ = 1998 trận. RuleBot chỉ dùng pilot và đánh giá, không phải full collect. Pilot 160 trận độc lập; validation 8000 trận; test 7200 trận. Toàn bộ là 11×11/2 agent/max500; lịch E 475/25/40, xác suất E 0.75 trong pilot/collect và 1.0 trong validation/test.
- Thầy dùng S4+search3 với dự đoán argmax có mask của đúng CNN đối thủ; đối thủ thực tế và learner đánh giá lấy mẫu. Ghi state quyết định/anchor của map 11; train từ S4 nội suy, 16 epoch, lr 1e-4, batch 256, CE mềm + KL neo 1, held-out 10% theo bàn, seed 1010, checkpoint 4/8/12/16.
- Validation và gate chính theo thiết kế đã duyệt; RuleBot guard ActiveWR ≥ baseline −3 điểm %, tự nổ ≤ baseline +1 điểm %. Macro CNN gate ActiveWR ≥ baseline +5 điểm %, cận dưới CI95 >0, tự nổ ≤ baseline +1 điểm %. Bootstrap ghép cặp theo bàn, 10000 lần. Baseline là S4 nội suy trên cùng bàn/góc, S3-11 learner và hai đối thủ ngoài pool chỉ báo mô tả.
- Dự kiến các stage `init`, `pilot`, `collect`, `train`, `validate`, `test`; output mới có registration/source/checkpoint hash, progress, dữ liệu thô và báo cáo. Seed đề xuất pilot 850000–850009, collect 860000–860998, validation 870000–870199, test 880000–880199, smoke 900000–900003 chỉ được dùng sau kiểm tra ledger.
- Pilot phải có nhãn hợp lệ từng CNN và anchor; dừng nếu CI95 pilot của macro ActiveWR thầy−baseline có cận trên <0 hoặc phát sinh lỗi/thiếu dữ liệu. Pilot quá nhỏ để kết luận tổng quát. Chỉ mở test sau validation đủ điều kiện; không tự chạy lại test đã mở, kể cả bị gián đoạn.

## Ma trận dự thảo

| Yêu cầu | Mức | Tình huống | Kết quả quan sát được | Lệnh dự kiến |
|---|---|---|---|---|
| Cấu hình, số trận | Unit | Full và smoke | Đúng kích thước, số seed, số góc, tỷ lệ đối thủ và counts; smoke phân biệt full | `python -m unittest test_run_s4_map11_acceptance -q` |
| Seed tách biệt | Unit | Dải chồng nhau, âm/tràn/kiểu sai, marker hoặc ledger đã dùng | Từ chối trước tạo output hoặc mở game | Cùng lệnh |
| Provenance | Integration | Sửa source/config/checkpoint sau init | Stage từ chối trước game/train, dữ liệu cũ nguyên vẹn | Cùng lệnh |
| Lock/no-overwrite | Integration | Hai process, lock của process khác, output tồn tại | Chỉ một owner; từ chối, không xóa lock hay ghi đè | Cùng lệnh |
| Đúng đối thủ | Integration | 11×11 thật với từng CNN và RuleBot pilot | Dự đoán search và policy thực tế đúng model/kiểu hành động; full collect không có RuleBot | Cùng lệnh |
| Global hook | Unit/integration | Trận thành công và ném exception | Hook đối thủ cũ được khôi phục trong cả hai trường hợp | Cùng lệnh |
| Nhãn và mask | Unit | Score không chấm là NaN; score hợp lệ/logit không hữu hạn | Cho phép NaN đúng vị trí; loại dữ liệu không hợp lệ và báo lỗi | Cùng lệnh |
| Lưu/resume collect | Integration | Một shard đã hoàn tất, shard thiếu, shard sai hoặc trùng seed/góc | Giữ shard hợp lệ; chỉ chạy shard thiếu; từ chối shard sai, không overwrite | Cùng lệnh |
| Pilot | Unit/integration | Thiếu nhãn một CNN/anchor, CI upper âm, signal yếu nhưng CI chứa 0 | Dừng hai trường hợp đầu; không kết luận FAIL chỉ từ tín hiệu yếu | Cùng lệnh |
| Train | Integration | Dataset 11×11 nhỏ hợp lệ và thiếu mẫu split | Split không trùng bàn; tạo checkpoint/metrics; thiếu mẫu thì dừng | Cùng lệnh |
| Train an toàn | Unit | Khởi tạo checkpoint, optimizer và tính loss | S4 được nạp đúng; critic không đổi; RNG/loss/metrics hữu hạn | Cùng lệnh |
| Chọn checkpoint | Unit | Hòa điểm, guard RuleBot/CNN sát ngưỡng, không hơn baseline | Chọn epoch sớm khi hòa; vi phạm guard bị loại; không hơn thì `VALIDATION_STOP` | Cùng lệnh |
| Phân tích | Unit | Dữ liệu tổng hợp theo bàn/góc/đối thủ | Macro trung bình đều 3 CNN, paired bootstrap theo bàn; S3-11 và đối thủ giữ ngoài không đi vào gate | Cùng lệnh |
| Test một lần | Integration | Test trước validation, hash thay đổi, marker đã tồn tại hoặc test dở | Không mở/chơi trước điều kiện; không replay test đã mở hay ghi đè marker | Cùng lệnh |
| Gate biên | Unit | +5pp, CI dưới bằng 0, tự nổ +1pp, RuleBot −3pp | Áp dụng đúng so sánh ≥/≤ và CI dưới phải lớn hơn 0 | Cùng lệnh |
| Smoke native | Integration | Nhiều worker, seed smoke và output riêng | Chạy thật ngắn, dữ liệu có dấu smoke, không mở seed pilot/collect/validation/test thật | `python run_s4_map11_distill.py ... --smoke` |

Các tên test và đối tượng fixture cụ thể do tác giả test chọn dựa trên hợp đồng trên và quy ước unittest của repo. Lệnh chạy thực tế sẽ dùng `E:\Tools\python.exe`. Không mở game test chính thức để kiểm thử code.
