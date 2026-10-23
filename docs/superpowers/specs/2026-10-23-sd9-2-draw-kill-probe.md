# SD9.2: probe trong các trận hòa của S3 khi đấu CNN, có nước giết bị bỏ lỡ không

Ngày 23/10/2026. Người dùng yêu cầu ("làm một probe nhỏ trên seed validation, chạy search đoán đối thủ bằng CNN thật trên các trạng thái của các trận hòa, rồi đếm số trạng thái có nước giết"). Từ hôm nay S3 tạm là agent chính. Mọi luật dưới đây được chốt trước khi chạy lượt chính. Runner: `probe_draw_kills.py`. Probe không train.

## Câu hỏi

SD9.1 cho thấy S3 vẫn hòa 53–64% khi đấu CNN, và khoảng 60% số trận đi hết 500 bước. Cần biết đây là chuyện nào:

- **Bỏ lỡ cơ hội:** trong các trận hòa vẫn có những lúc tồn tại kế hoạch giết, nhưng S3 không đi. Khi đó hướng A (chưng cất search3, cho search đoán đối thủ bằng CNN) là đáng làm.
- **Do cấu trúc:** khi hai bên đều né tốt thì hầu như không có kế hoạch giết nào. Khi đó chỉ đổi luật mới giảm được hòa, ví dụ cho vòng đóng sớm hơn hoặc kéo dài trận. Lưu ý: hòa đổi thành thắng nhờ vòng là thắng thụ động, nên WR tăng mà ActiveWR có thể không tăng.

## Thiết kế

- **Trận đấu.** Learner S3 (`4185375f…69ed7`, lấy mẫu) đấu ba đối thủ CNN lấy mẫu: S2 (`d222d81f…6814`), G (`bb0c775c…c7ac`) và S3.
  - Seed validation 21000, 50 bàn (21000–21049) × 2 góc = 100 trận mỗi đối thủ, tổng 300 trận.
  - Luật chơi: 9×9 1v1, max_steps 500, lịch E 475/25/40.
  - Đây là đúng các trận của SD9.1 chơi lại: cùng bàn, cùng code, cùng torch nên kết quả phải giống hệt. Runner so lại với shard SD9.1 và báo số trận trùng.
- **Trạng thái được lưu.** Cứ mỗi 5 bước (`step % 5 == 0`), nếu cả hai còn sống và đối thủ cách trong `kill_search.REACH` = 6 ô Manhattan, chụp lại trạng thái. Chụp bằng `deepcopy`, không dùng RNG nên không làm lệch trận.
- **Kiểm tra nước giết.** Sau trận, với mọi trạng thái đã lưu của các trận **hòa**, chạy `kill_search._best_plan(env, aid, depth=3, HORIZON=12)`, tức cùng thuật toán của search3: tối đa 3 nước của learner, sau đó chỉ né. Mỗi trạng thái kiểm hai lần:
  - `oracle`: đối thủ được mô hình bằng chính CNN của nó (argmax trên policy đã mask, `probe_search_cnn_model.cnn_model`);
  - `rulebot`: đối thủ được mô hình bằng RuleBot, đúng giả định mà thầy của S3 dùng.
- **Đối chứng dương.** Làm cùng phép kiểm trên các trận S3 **thắng chủ động**. Ở những trận này chắc chắn đã có cú giết, nên search phải tìm ra kế hoạch giết ở phần lớn các trận.
- **Báo cáo.** Theo từng đối thủ và gộp cả ba:
  - số trận hòa;
  - số trạng thái;
  - % trạng thái có kế hoạch giết;
  - % trận hòa có ít nhất 1 trạng thái có kế hoạch giết, kèm CI95 bootstrap theo bàn (10 000 lần, seed 20261014);
  - chỉ số trên tính riêng phần trước bước 475;
  - đối chứng;
  - số giây mỗi trận.

## Luật chốt trước

Chỉ số chính **P** = % trận hòa (gộp ba đối thủ) có ít nhất một trạng thái mà `oracle` tìm được kế hoạch giết.

- **Điều kiện hợp lệ:** đối chứng `oracle` (% trận thắng chủ động có trạng thái có kế hoạch giết) phải ≥ 50%. Nếu dưới 50% → `invalid_control`, tức search không nhìn ra được cả những cú giết đã thật sự xảy ra, nên không kết luận gì.
- **P ≥ 50%** → `missed_kills_direction_A`: phần lớn trận hòa có cơ hội giết bị bỏ lỡ. Đề xuất viết spec hướng A.
- **P ≤ 20%** → `structural_consider_rules`: hòa chủ yếu do cấu trúc. Đề xuất viết spec thử đổi luật (lịch vòng hoặc max_steps), báo cả ActiveWR lẫn thắng thụ động.
- **20% < P < 50%** → `mixed`: báo số liệu, quyết định cùng kết quả S2-KL.
- Cột `rulebot` chỉ để mô tả: nó cho biết các bẫy theo kiểu RuleBot có "nhìn thấy" được trong trận đấu CNN hay không.
- Không dùng seed test nào.

## Chi phí

Đo bằng lượt chạy thử 5 bàn mỗi đối thủ (xem RESULTS "SD9.2-prep"). Số bàn chỉ chỉnh theo chi phí, không chỉnh theo kết quả.

## Lệnh (cmd)

```
D:\anaconda\envs\torch\python.exe probe_draw_kills.py --output runs/sd92-21000 --workers 5 > runs\sd92-21000-console.log 2>&1
REM Nếu bị ngắt:
D:\anaconda\envs\torch\python.exe probe_draw_kills.py --output runs/sd92-21000 --workers 5 --resume > runs\sd92-21000-resume-console.log 2>&1
```

## Giới hạn

- `oracle` dùng argmax, trong khi đối thủ thật lấy mẫu. Kế hoạch giết tìm được là giết khi đối thủ đi đúng nước dự đoán, không phải giết chắc chắn trước mọi nước đi. Vì vậy P là cận trên lạc quan cho "có thể giết". Ngược lại, search chỉ nhìn tối đa 3 nước rồi né, nên có thể bỏ sót kế hoạch dài hơn.
- Chỉ lấy mẫu mỗi 5 bước nên có thể sót trạng thái ngắn.
- Chỉ dùng seed validation, 300 trận.
