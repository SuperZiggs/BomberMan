# SD10 (hướng A): chưng cất S3 + search3 đoán đối thủ đúng thành CNN thuần S4

Ngày 24/10/2026. Người dùng duyệt ("làm cả 3 việc của đề xuất") sau SD9.3. Mọi luật dưới đây được chốt trước khi chạy. Runner: `run_sd10_distill.py`; test: `test_run_sd10.py`; lệnh: `SD10_GUIDE.md`.

## Câu hỏi

Thầy của S3 (S2 + search3) luôn giả định đối thủ là RuleBot. Khi đấu CNN, giả định này sai: SD9.2 cho thấy đoán bằng RuleBot "thấy" kế hoạch giết ở 90.3% trận hòa, trong khi đoán bằng chính CNN của đối thủ chỉ thấy 15.9%. Vì vậy S3 học được bẫy kiểu RuleBot, mà những bẫy này không ăn CNN.

Câu hỏi: nếu dạy S3 bằng nhãn search **đoán đúng đối thủ**, S4 có thắng chủ động CNN nhiều hơn S3 mà vẫn giữ được sức mạnh khi đấu RuleBot không?

**Kỳ vọng ghi trước:** lợi ích tối đa nhỏ. Theo SD9.2, chỉ khoảng 16% trận hòa có cơ hội giết dưới mô hình đúng, nên mức tăng ActiveWR đấu CNN tối đa khoảng +10pp (lạc quan). Gate +5pp có thể không đạt.

## Thiết kế

- **Thu dữ liệu:**
  - Learner S3 (`4185375f…69ed7`, lấy mẫu) + `kill_search.search_action3`.
  - 1000 bàn 214000–214999 × 2 góc = 2000 trận. Dải bàn mới: SD9 dùng 212000–213499.
  - Đối thủ theo bàn, lần lượt RuleBot, S2, G, S3, mỗi loại 250 bàn (500 trận). Đối thủ CNN lấy mẫu.
  - **Mô hình đối thủ trong search:**
    - đấu CNN: argmax policy đã mask của chính CNN đó (`probe_search_cnn_model.cnn_model`, giống oracle SDM/SD9.2);
    - đấu RuleBot: RuleBot như trước.

    Mô hình chỉ dùng để tạo nhãn. Lúc chơi, S4 là CNN thuần, không cần biết đối thủ là ai.
  - Lịch E với xác suất 0.75 theo bàn, max_steps 500.
  - Trạng thái quyết định và neo giữ như SD9:
    - trạng thái quyết định = trạng thái mà điểm search của các nước khác nhau;
    - neo = 5% trạng thái còn lại, kèm logits của S3.
  - Shard 20 bàn. Mỗi shard có đủ bốn loại đối thủ, nên tải chia đều cho các tiến trình.
- **Train:** giống hệt SD9, chỉ đổi điểm xuất phát và đích neo sang S3.
  - Học trò xuất phát từ S3. Loss = CE mục tiêu mềm trên trạng thái quyết định + KL về S3 trên neo (λ = 1).
  - Adam 1e-4, batch 256, 16 epoch, ứng viên 4/8/12/16, held-out 10% bàn, seed chia 1010.
- **Validation:** seed 21000, 200 bàn × 2 góc = 400 trận cho mỗi cặp (model, đối thủ).
  - Model: S3 và 4 ứng viên.
  - Đối thủ: S2, G, S3, RuleBot.
  - Lịch E, max_steps 500. Policy lấy mẫu.
- **Test (một lần):** seed 58000, 200 bàn × 2 góc = 400 trận cho mỗi cặp.
  - Model: S4 và S3.
  - Đối thủ: S2, G, S3, RuleBot, và F (`0315a56f…1483`). F là đối thủ không có trong dữ liệu thu, chỉ dùng để mô tả khả năng tổng quát.
- **Thống kê:**
  - Bootstrap ghép cặp theo bàn, 10 000 lần, seed 20261014.
  - Macro CNN = trung bình S2/G/S3 theo từng (bàn, góc), giống SD9.1.

## Luật chốt trước

**Chọn ứng viên (validation):**
1. Một ứng viên hợp lệ nếu thỏa cả ba:
   - ActiveWR đấu RuleBot ≥ S3 − 3pp;
   - tự nổ macro CNN ≤ S3 + 1pp;
   - tự nổ đấu RuleBot ≤ S3 + 1pp.
2. Trong các ứng viên hợp lệ, chọn ứng viên có ActiveWR macro CNN cao nhất. Nếu bằng nhau, chọn epoch sớm hơn.
3. Chỉ mở test khi ứng viên được chọn có ActiveWR macro CNN **lớn hơn** S3 trên validation. Nếu không có ứng viên hợp lệ, hoặc ứng viên được chọn không hơn S3, thì `VALIDATION_STOP`: seed 58000 vẫn chưa dùng và S3 giữ nguyên.

**Gate (test 58000, S4 − S3):**
- ActiveWR macro CNN ≥ +5pp, và cận dưới CI95 > 0;
- tự nổ macro CNN ≤ +1pp;
- ActiveWR đấu RuleBot ≥ −3pp (ước lượng điểm).

**Quyết định:**
- ĐẠT: S4 thay S3 làm agent chính.
- FAIL: giữ S3 và kết luận hướng A không đủ, khép lại hướng giảm hòa khi đấu CNN.
- Hòa, WR, thắng thụ động và kết quả đấu F đều được báo, nhưng không nằm trong gate.

## Chi phí (đo 24/10, RESULTS "SD10-prep")

Đo bằng 1 trận đầy đủ cho mỗi loại đối thủ, bàn 214000–214003, góc 0, chạy 4 tiến trình song song trên máy người dùng.

| Đối thủ | Số bước | Giây/trận | Trạng thái quyết định |
|---|---|---|---|
| RuleBot | 75 | 32.9 | 3 |
| S2 | 500 | 651.2 | 0 |
| G | 475 | 628.9 | 2 |
| S3 | 500 | 603.0 | 6 |

- Một trận đấu CNN tốn khoảng 628 giây, tức chậm khoảng 19 lần so với trận đấu RuleBot. Lý do: trận dài gần hết 500 bước, và mỗi bước mô phỏng của search đều phải chạy CNN của đối thủ.
- Mỗi trận đấu CNN chỉ cho khoảng 2.7 trạng thái quyết định, khớp với SD9.2 (0.2% trạng thái có kế hoạch giết).
- Thiết kế đầy đủ (500 trận RuleBot + 1500 trận CNN) tốn khoảng **53 giờ** với 5 tiến trình và cho khoảng 4000 trạng thái quyết định khi đấu CNN. Muốn đạt mức 10 000 trạng thái như SD9 thì cần khoảng 130 giờ.
- Bản rút gọn (150 bàn CNN + 250 bàn RuleBot) tốn khoảng 11 giờ nhưng chỉ cho khoảng 800 trạng thái khi đấu CNN. INSIGHTS ghi rằng nhãn thưa không đổi được CNN, nên bản này nhiều khả năng FAIL.
- Số đo chỉ dựa trên 4 trận, nên sai số lớn.

**Trạng thái:** code và test xong, **chưa chạy**. Lượt thu dài 53 giờ cho mức lợi ích tối đa khoảng +10pp, nên chờ người dùng quyết định. Nếu chạy, giữ đúng thiết kế trên, không đổi số bàn sau khi đã mở thu.

## Giới hạn

- Đối thủ CNN trong validation và test (S2/G/S3) cũng là đối thủ lúc thu dữ liệu, chỉ khác bàn. F được thêm vào để kiểm tra khả năng tổng quát, nhưng chỉ mô tả.
- Mô hình đối thủ dùng argmax, còn đối thủ thật lấy mẫu, nên nhãn là "giết nếu đối thủ đi nước có khả năng nhất".
- Chỉ một train seed và một test seed.
