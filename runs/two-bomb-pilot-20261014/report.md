# Pilot giáo viên hai bom — 2026-10-14

**Giữ model gốc. Pilot chưa đủ để kết luận:** local hoàn tất 200/200 episode, two_bomb chỉ hoàn tất 146/200 trong ngân sách train chung 65 phút. Bài test cuối vẫn hoàn tất 400 trận mỗi model, tổng 1.200 trận. Tổng thời gian chạy ghi nhận 86.67 phút, dưới hai giờ.

Best two_bomb là warm-start ở episode 0, có trọng số giống hệt model gốc. Validation episode 100 của two_bomb đạt ActiveWR 18%, tự nổ 1%, hòa 77%, nên guard từ chối checkpoint đó. Final 146 episode vẫn được lưu riêng; chưa có validation episode 200 và không dùng final để thay best trong benchmark.

CNN tự chọn hành động qua safety mask; planner chỉ tạo nhãn khi train. Observation 12 kênh, reward, optimizer và các hệ số loss khác giữ nguyên. Flag trainer/CLI mới là `attack_teacher=local|two_bomb`, mặc định local. Forecast nhận bom đặt trễ và tính sức chứa trả lại sau nổ/chain; search tối đa 64 kế hoạch, hai bước trước bom đầu và ba bước sau đó. Nhãn ngang điểm được phân phối đều, các nhãn mới vào combat loss 0.5 và thay nhãn cũ trên cùng bước.

58 test pass: đối chiếu forecast với env.step, capacity một/hai, chain trả slot, gỗ mở tia, lửa còn sót, va chạm, fixture combo đóng hết refuge trong khi từng single vẫn còn refuge, state/RNG không đổi, nhãn trong mask, soft target không ép hành động CNN, bootstrap theo board và hủy validation/evaluation. `git diff --check` pass; SHA256 checkpoint gốc không đổi.

| Model dùng để test | Best episode | WR % | ActiveWR % | Tự nổ % | Hòa % | Kills/trận | Thắng thụ động % |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | gốc | 37.50 | 33.00 | 0.00 | 61.00 | 0.330 | 4.50 |
| local | 200 | 37.50 | 33.00 | 0.25 | 60.75 | 0.330 | 4.50 |
| two_bomb | 0 | 37.50 | 33.00 | 0.00 | 61.00 | 0.330 | 4.50 |

Mỗi model dùng seed bàn 25000–25199, mỗi bàn chạy CNN tại cả A0 và A1, 9×9/2 agents/max_steps 500. Opponent là RuleBot hiện tại, policy CNN chạy CPU qua mask, không gọi giáo viên mới.

| Model | Góc | ActiveWR % | Tự nổ % | Hòa % | Kills/trận | Thắng thụ động % |
|---|---|---:|---:|---:|---:|---:|
| original | A0 | 32.50 | 0.00 | 57.50 | 0.325 | 7.50 |
| original | A1 | 33.50 | 0.00 | 64.50 | 0.335 | 1.50 |
| local | A0 | 30.50 | 0.50 | 61.00 | 0.305 | 6.00 |
| local | A1 | 35.50 | 0.00 | 60.50 | 0.355 | 3.00 |
| two_bomb | A0 | 32.50 | 0.00 | 57.50 | 0.325 | 7.50 |
| two_bomb | A1 | 33.50 | 0.00 | 64.50 | 0.335 | 1.50 |

Bootstrap percentile 10.000 lần, seed 20261014, resample 200 **seed bàn** và giữ A0/A1 cùng nhau. Chênh lệch dưới đây là two_bomb trừ đối chứng, đơn vị điểm phần trăm.

| Đối chứng | Chênh ActiveWR | CI 95% | Tăng ≥5 điểm và cận dưới >0 |
|---|---:|---|---|
| original | +0.00 | [0.00, 0.00] | Không |
| local | +0.00 | [-6.00, 5.75] | Không |

Tỷ lệ tự nổ của best two_bomb không cao hơn hai đối chứng, nhưng không có tăng ActiveWR, hai CI không có cận dưới dương, và lượt train chưa đủ. CI [0,0] so với gốc phản ánh hai policy giống nhau; không phải bằng chứng giáo viên mới cải thiện CNN. Một seed train và một lượt bị cắt chưa kiểm chứng độ ổn định phương pháp.

| Teacher | Episode validation | ActiveWR % | Tự nổ % | Hòa % | Chọn best |
|---|---:|---:|---:|---:|---|
| local | 0 | 29.00 | 0.00 | 62.00 | Có |
| local | 100 | 33.00 | 0.00 | 60.00 | Có |
| local | 200 | 34.00 | 0.00 | 56.00 | Có |
| two_bomb | 0 | 29.00 | 0.00 | 62.00 | Có |
| two_bomb | 100 | 18.00 | 1.00 | 77.00 | Không |

Cả hai warm-start từ runs/pressure-C-next/final_phaseC.pt, phase C/9×9/2 agents, lr 1e-4, train seed 2026101303, max_steps 500. Validation 100 trận seed 21000 ở 0/100/200; two_bomb chỉ hoàn tất 0/100. Không giảm số episode đích hoặc nới guard để báo thành công.

Preflight dùng 147 trạng thái giao tranh từ bốn replay timeout seed 19000/19003/19002: 33.31 ms/query trung bình, p95 62.83 ms, tối đa 71.93 ms. Có 137 nhãn được nhận, 100 nhãn hai bom; chỉ 1 nhãn đóng hết refuge trên mẫu replay này (0 nhãn hai bom). Mẫu timeout này không đại diện toàn bộ train.

146 episode two_bomb hoàn tất có 59,261 query, 2,377,107 kế hoạch, 56,036 nhãn được nhận, 43,321 nhãn hai bom. Pressure trung bình trên nhãn được nhận 36.32%; thời gian planner trung bình 27.96 ms/query, cộng dồn 27.61 phút. Đây là nhãn dự kiến, không phải số combo CNN đã thực hiện hoặc tỷ lệ hạ đối thủ.

Bước tiếp theo hợp lý là giảm chi phí planner và kiểm tra chất lượng/khả năng thực hiện chuỗi nhãn qua các bước trước khi chạy lại đủ 200 episode mỗi lượt. Không kết luận nguyên nhân thất bại chỉ từ các số trên. GAE, PPO, curriculum mới và thay reward chưa thử. Forecast chỉ dựa trên bom đã biết; không bảo đảm thắng trước phản bom và va chạm tương lai.

Artifact: config.json; status.json; preflight.json; teacher_summary.json; audit.json; local/ và two_bomb/ (config, metrics, validation, best/final); evaluation_original_25000.json, evaluation_local_25000.json, evaluation_two_bomb_25000.json. source/ giữ source đúng lúc bắt đầu train. evaluation_source/ giữ bản runner/test được bổ sung kiểm tra hoàn tất và hủy bài test; code env/rule/trainer/teacher/evaluate không đổi trong toàn pilot. finish_evaluation.py hoàn tất phần đánh giá dự phòng sau khi runner đầu dừng vì hết thời gian train.
