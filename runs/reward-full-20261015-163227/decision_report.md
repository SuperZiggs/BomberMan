# T4-R1 — pilot bị ngắt, chưa đủ kết luận

**R1 IMPLEMENTED/TESTED; lượt full INCOMPLETE/REVIEWED. Giữ checkpoint B và wall bonus mặc định0,15.**

Runner đã triển khai đối chứng wall0,15 và0,075, cùng warm-startB, train seed2026101505, teacher local, MC raw, reward khác và mask giữ nguyên. Bộ regression200/200 PASS; smokeCPU/CUDA đầy đủ20episode/nhánh, source/checkpoint hashes qua. Preflight39.338transition R0 xác nhận delta chỉ bằng−0,075×tường đủ điều kiện. Logging contact giữ nguyên actions/kết quả/RNG so với evaluator trước R1.

Tiến trình kết thúc trong validation half-wall tại episode200; console cuối ghi17:14:07 ngày15/10/2026, Asia/Saigon. Không còn tiến trình Python khi kiểm tra lại. Runner không kịp chạy finally; summary này do post-interruption audit tạo, không phải một lượt full đã hoàn tất. Không có thời gian kết thúc chính xác nên elapsed_seconds để null.

| Nhánh | Episode/Adam update đã lưu log | Validation đầy đủ | Best validation ActiveWR | Final200 |
|---|---:|---|---|---|
| Control0,15 | 200/200 | 0/100/200 | 33%, episode200 | Có |
| Half-wall0,075 | 199/199 | 0/100 | 30%, episode100 | Thiếu |

Control validation: ActiveWR29%→22%→33%, suicide0% ở cả ba mốc; draw62%→68%→59%. Half-wall:29%→30%, suicide0%, draw62%→63%; mốc200 chưa có kết quả hoàn chỉnh. Validation dùng100game/50board đã xem21000–21049/cảA0-A1. Không dùng các điểm này để khẳng định half-wall tốt hơn hoặc kém hơn trên test.

Episode200 của half-wall đã bắt đầu validation sau update trong RAM, nhưng weights/update đó không được lưu đầy đủ; chỉ199episode/update có metrics phục hồi được. Không coi nhánh này đã hoàn tất protocol200. Checkpoint chỉ chứa weights, không có Adam/scheduler/RNG để resume chính xác; không warm-start từ best100 rồi chạy thêm1episode hoặc ghép với control cũ thành cặp đầy đủ.

**Chưa mở test36000–36199**, không có evaluation, bootstrapCI hoặc gate cải tiến. Không chấm best tạm thời để chọn winner. Không thay B hoặc coefficient mặc định.

Audit kiểm tra399episode logs, tổng component khớp reward, coefficient×tường đủ điều kiện, loss hữu hạn, source snapshot/current source, checkpoint và inputR0 hashes. Replay lại320skill games/6.966frames và tính lại summary từng mốc. Control gzip đầy đủ; half-wall gzip thiếu footer do bị ngắt. Đã đọc prefix deflate, chỉ nhận128JSON record hoàn chỉnh khớp hai skill probes; không sửa file gzip gốc. Có bản skill_games_recovered.jsonl riêng và recovery_manifest.json.

Để hoàn tất R1 cần một attempt mới: train lại cả hai nhánh từB trong thư mục mới, cùng protocol đã đăng ký, rồi400test/model. Dải36000–36199 vẫn chưa mở và guard của runner sẽ kiểm tra lại trước train. Không ghi đè thư mục này. Lệnh đầy đủ và điều kiện trong [REWARD_PILOT_GUIDE.md](/D:/fpt/subject/BomberMan/REWARD_PILOT_GUIDE.md).

Một seed train và validation đã xem chưa đủ kết luận ổn định. Bản này chỉ chứng minh implementation/checks đã qua và bảo toàn kết quả dở dang; R1 research chưa đóng.

Files: [summary.json](/D:/fpt/subject/BomberMan/runs/reward-full-20261015-163227/summary.json), [analysis_results.json](/D:/fpt/subject/BomberMan/runs/reward-full-20261015-163227/analysis_results.json), [config.json](/D:/fpt/subject/BomberMan/runs/reward-full-20261015-163227/config.json), [preflight.json](/D:/fpt/subject/BomberMan/runs/reward-full-20261015-163227/preflight.json).
