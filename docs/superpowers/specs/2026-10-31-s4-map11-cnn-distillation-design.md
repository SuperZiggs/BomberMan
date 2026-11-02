# Đề xuất S4-11: chưng cất đối thủ CNN trên map 11×11

**Trạng thái:** người dùng đã duyệt thiết kế ngày 31/10/2026. Chưa tạo runner, đăng ký run, mở seed, chơi trận hoặc huấn luyện theo thiết kế này. Tên S4-11 là tên tạm cho ứng viên 11×11, không phải checkpoint đã tồn tại.

## Mục tiêu và bằng chứng

Mục tiêu người dùng chọn: cải thiện S4 khi đấu **CNN 1v1 trên map 11×11**, đồng thời dùng **RuleBot để kiểm tra giữ kỹ năng**. Đối chứng chính là chính S4 9×9 được nội suy lên 11×11, trên cùng bàn và góc xuất phát; báo thêm S3-11 làm tham chiếu, không chọn epoch theo S3-11.

- S4 SD10 là `runs/sd10-20261026/train/epoch_12.pt`, SHA256 `90d1d0e70c64f879e07f85c68c003969d082043b3cd8065314a3c35faf62bece`. Kiểm tra ngày 31/10 cho thấy loader nội suy được trọng số 9→11 và forward input giả có đầu ra hữu hạn; chưa đánh giá sức chơi S4 trên 11×11.
- SD10 9×9 đạt +14pp ActiveWR macro CNN so với S3 trên test đã mở, nhưng kết quả này không chứng minh hiệu năng 11×11. Pilot S3-11 từng tăng +41pp trước RuleBot 11×11 so S3 nội suy, song benchmark CNN chỉ +3.83pp và không đạt gate +5pp; đó là S3, không phải S4.
- Dữ liệu collect SD10 chứa observation 13×9×9 nên không dùng trực tiếp làm observation 13×11×11. Cần thu nhãn mới trên map 11×11.

## Các hướng cân nhắc

1. **Khuyến nghị — chưng cất thầy S4 + search3 mô hình hóa CNN đối thủ.** Khi đấu CNN, thầy dự đoán đối thủ bằng chính checkpoint đó, như SD10; học trò bắt đầu từ S4 nội suy và có KL neo về S4. Phù hợp mục tiêu CNN, đổi lại thu nhãn tốn CPU và cần runner mới.
2. **Chưng cất thầy mô phỏng RuleBot như runner S3-11.** Kế thừa nhiều code và từng hiệu quả trước RuleBot, nhưng dữ liệu nhắm vào RuleBot nên không trực tiếp giải quyết mục tiêu CNN.
3. **RL tiếp với pool CNN.** Có thể khám phá chiến thuật mới nhưng các lượt Phase D từ S2 và S2-KL trước đây không giữ được kỹ năng; chưa có cơ chế giữ kỹ năng được xác minh. Chưa chọn hướng này.

## Phạm vi và đối thủ

- 11×11, 13 kênh, 2 agent, tối đa 500 bước, lịch E `475/25/40`; collect bật E theo xác suất 0.75 giống SD10; validation/test bật E. CNN thực tế lấy mẫu và mask; lúc dùng search, mô hình dự đoán CNN dùng argmax đã mask.
- **Pool collect và macro chính:** S2 `runs/sd8a-20261020/train/epoch_16.pt` (`d222d81f…6814`), G `runs/sd75-20261018/full/best_phaseC.pt` (`bb0c775c…c7ac`), S3-11 `runs/s3-map11-20261024-auto-v1/train/epoch_16.pt` (`17748471…a95f`). S2 và G được nội suy 9→11; S3-11 là checkpoint 11×11. Các checkpoint đóng băng.
- Collect chính chỉ đấu ba CNN, chia đều theo seed; **RuleBot dùng cho pilot và kiểm tra retention, không tạo nhãn collect chính**. KL neo giữ hành vi của S4 nội suy trên các trạng thái 11×11; nó không bảo đảm hiệu năng trước RuleBot hay trên 9×9, vì vậy phải đo riêng RuleBot.
- Test bổ sung S3 và S4 nội suy làm hai đối thủ ngoài pool collect, chỉ báo mô tả. Chúng cùng họ mô hình với S4 nên không được gọi là kiểm tra tổng quát độc lập. S3-11 cũng được đánh giá như một learner tham chiếu, chỉ báo mô tả. Không suy ra kết quả FFA hoặc map 9×9.
- Giữ S4 9×9 và các checkpoint/nguồn cũ; tạo model và output mới. Các thay đổi chưa commit của người dùng trong workspace phải được giữ nguyên.

## Phương án thực hiện đề xuất

### 1. Runner và kiểm tra kỹ thuật

Tạo runner riêng, dự kiến `run_s4_map11_distill.py`, kế thừa các hàm phù hợp từ SD10/S3-11; không sửa runner hoặc nguồn thí nghiệm đã khóa. Trong mỗi trận CNN, set `kill_search.OPPONENT_MODEL` thành model argmax đúng của đối thủ và khôi phục trong `finally`; agent đối thủ thật vẫn lấy mẫu. Pilot RuleBot dùng mô hình RuleBot. Lúc đánh giá học trò là CNN thuần, không search.

Trước run: khóa config, checkpoint SHA256, toàn bộ nguồn thực sự dùng và runtime; lưu bundle nguồn; guard đường dẫn output, lock chống chạy trùng, marker seed, lưu shard nguyên tử. Collect cho phép resume chỉ shard thiếu sau khi xác thực metadata/seed/góc/hash. Test tạo marker độc quyền trước trận đầu; nếu bị ngắt thì ghi incomplete và không tự chạy lại, vì không biết chắc các trận trong shard chưa lưu đã được chơi hay chưa. Không ghi đè test hay tự dùng lại seed. Stage dài có progress; CPU worker một luồng torch, mặc định 5 workers; train có thể dùng CUDA. Bài kiểm tra phải bao gồm môi trường 11×11 thật, đúng model đối thủ, khôi phục hook search, mask/shape, no-overwrite, source hash, resume collect, test gián đoạn, khóa và lựa chọn/gate. Smoke riêng không được dùng để kết luận.

### 2. Pilot độc lập trước collect lớn

Đề xuất 10 bàn × 2 góc × 4 đối thủ (S2/G/S3-11/RuleBot) × 2 learner (S4 nội suy và S4 nội suy + search3 tương ứng) = **160 trận**, trong đó 80 trận dùng search. Pilot dùng lịch E với xác suất 0.75 theo bàn như collect; hai learner cùng bàn/góc/lịch. Đo thời gian/trận theo đối thủ, số bước, số nhãn quyết định/neo và ActiveWR; giữ raw, thời gian thực và nguồn. Pilot chỉ để đo chi phí, kiểm tra nhãn và mô tả thầy: 10 bàn/đối thủ quá ít để kết luận thắng/thua hay điều chỉnh gate sau khi xem số liệu.

Chỉ tự tiếp tục collect lớn khi pilot hoàn tất đúng cấu hình, logits và điểm được chấm hữu hạn, có nhãn quyết định từ mỗi CNN và có trạng thái neo. Điểm `NaN` dành cho hành động không được search chấm là hợp lệ theo biểu diễn SD10, không được hiểu là lỗi số. Tính thêm CI95 bootstrap ghép cặp theo 10 bàn cho macro ActiveWR thầy − S4; nếu cận trên nhỏ hơn 0 thì dừng để người dùng xem xét thay vì tự thu lớn với thầy kém hơn ở pilot. Với mẫu nhỏ, điều kiện dừng này là cảnh báo vận hành, không đủ để kết luận tổng quát về thầy. Nếu sai cấu hình, thiếu nhãn ở một CNN, lỗi hoặc hết tài nguyên, dừng và báo người dùng; không gọi thiếu dữ liệu là kết quả PASS/FAIL về sức chơi. Nếu hợp lệ, ghi dự báo thời gian dựa trên số đo pilot rồi tiếp tục; không dùng trận pilot làm collect/validation/test.

### 3. Collect và train

- Thu **999 bàn × 2 góc = 1998 trận**, 333 bàn/666 trận cho mỗi CNN; 50 shard gồm 49 shard 20 bàn và shard cuối 19 bàn. Bản kế hoạch dùng 1998 thay vì 2000 để phân bố chính xác ba đối thủ. Khi ngắt, shard hoàn tất được giữ; shard chưa hoàn tất được thu lại.
- Lưu observation, mask, seed/góc/bước và điểm search ở trạng thái quyết định; lấy 5% trạng thái còn lại làm anchor cùng logits S4. Giữ dữ liệu theo quy trình SD10, không mở rộng nhãn bằng cách thay ngưỡng sau pilot.
- Train offline từ S4 nội suy: 16 epoch, Adam `1e-4`, batch 256, loss CE mục tiêu mềm + KL neo hệ số 1; held-out 10% theo bàn, seed chia/batch và RNG NumPy/PyTorch cố định 1010, ghi runtime và thiết lập deterministic, không hứa tái lập bit-for-bit giữa thiết bị/phiên bản. Lưu epoch 4/8/12/16, metrics và hash; critic giữ nguyên như runner chưng cất hiện có. Nếu split thiếu mẫu decision/anchor cần thiết, dừng thay vì tuyên bố train thành công.

### 4. Validation và test

**Validation:** S4 gốc nội suy + 4 ứng viên, trước 3 CNN collect và RuleBot, 200 bàn × 2 góc mỗi cặp = **8000 trận**. Ứng viên hợp lệ nếu so với S4 nội suy trên cùng bàn: ActiveWR RuleBot không giảm quá 3 điểm %, tự nổ macro CNN không tăng quá 1 điểm %, và tự nổ RuleBot không tăng quá 1 điểm %. Chọn ứng viên hợp lệ có ActiveWR macro CNN cao nhất; hòa chọn epoch sớm. Chỉ mở test nếu macro ActiveWR của ứng viên được chọn lớn hơn S4 nội suy trên validation; nếu không, `VALIDATION_STOP`.

**Test một lần:** S4-11 được chọn, S4 nội suy và S3-11 learner tham chiếu × (S2/G/S3-11/RuleBot/S3/S4) × 200 bàn × 2 góc = **7200 trận**. Các learner và đối thủ đều đóng băng, CNN thuần lấy mẫu, không search. Chỉ S4-11 so với S4 nội suy trên S2/G/S3-11 và RuleBot nằm trong gate; S3/S4 đối thủ ngoài pool và S3-11 learner chỉ báo mô tả.

**Gate đề xuất chốt trước test:**

1. ActiveWR macro CNN tăng ít nhất 5 điểm %, cận dưới CI95 lớn hơn 0.
2. Tự nổ macro CNN tăng không quá 1 điểm %.
3. ActiveWR trước RuleBot giảm không quá 3 điểm %.
4. Tự nổ trước RuleBot tăng không quá 1 điểm %.

Các ngưỡng là **đề xuất**, dựa theo protocol SD10; điều kiện 4 bổ sung để kiểm tra RuleBot chặt hơn. Macro là trung bình đều ba CNN tại mỗi cặp bàn/góc; bootstrap ghép cặp theo **200 bàn**, 10000 lần, seed bootstrap `20261014`. Báo thêm WR, hòa, hết bước, thắng thụ động, tự nổ và từng đối thủ. Gate PASS chỉ kết luận trong phạm vi protocol; không thay model phục vụ/UI tự động. FAIL giữ kết quả và checkpoint cũ, không chọn lại theo test hoặc mở test khác để cứu gate.

## Seed, chi phí và vận hành

Đề xuất seed chưa đặt chỗ: pilot `850000–850009`, collect `860000–860998`, validation `870000–870199`, test `880000–880199`, smoke `900000–900003`. Đã quét tham chiếu văn bản hiện có tại workspace trước khi viết đề xuất và chưa thấy các khối này; kiểm lại ledger, registration và test marker của các run/nhánh liên quan trước init, loại trừ đúng bản đề xuất và registration của chính run này khỏi kết quả dò trùng. Nếu phát hiện trùng thực sự, chọn dải mới và cập nhật config **trước** khi mở bất kỳ trận nào. Test chỉ được mở sau validation và chỉ một lần; pilot/collect/validation/test/smoke không trùng nhau.

Chi phí 11×11 chưa đo. Số đo SD10 9×9 trước đây khoảng 628 giây/trận CNN chỉ dựa trên ba trận CNN; nếu giả sử tốc độ đó áp dụng nguyên vẹn và chia lý tưởng cho 5 workers, collect 1998 trận sẽ khoảng **70 giờ**, nhưng đây chỉ là ví dụ tính toán, không phải ETA 11×11. Pilot sẽ đo lại và lưu dự báo; tổng wall-clock còn tùy CPU, nhiệt, tải máy và những lần ngắt. Sau khi thiết kế được duyệt, chạy pilot rồi tự tiếp tục nếu đủ điều kiện kỹ thuật ở trên; theo dõi mỗi giờ theo yêu cầu vận hành đã dùng trước đó, báo khi chuyển stage/lỗi/hoàn tất, dừng lịch khi xong.

Trước khi giao kết quả: kiểm source/checkpoint/seed, đủ shard và trận, tính lại macro/CI/gate từ raw, ghi báo cáo và cập nhật RESULTS/IMPROVEMENTS/PLAN theo `AGENT.md`. Không chạy lại test đã mở hoặc tự thay model phục vụ.
