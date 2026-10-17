# T4-R1 — thử giảm riêng thưởng phá tường

**R1 DONE/REVIEWED.** **Gate FAIL, giữ checkpoint B và reward hiện tại.**

Đủ 200 episode và 200 Adam updates mỗi nhánh; validation ở0/100/200; 400 test games/model, seeds36000–36199/cảA0-A1. Runner hoàn tất trong 49.54 phút. Các model dùng chung200 seed bàn; không coi1.200 game là1.200 bàn độc lập.

Half-wall đạt 35.00% ActiveWR: chênh +0.50 điểm phần trăm so với B và -1.25 điểm so với control. Cả hai CI95% đều chứa 0; số tự nổ của half-wall là 2/400, so với 1/400 ở mỗi đối chứng. Validation tốt hơn chưa chuyển thành cải thiện đạt tiêu chí trên test.

## Thiết kế và kiểm tra

Hai nhánh warm-start từ B với Adam mới, train seed2026101505. Control wall bonus0,15; treatment0,075. Cùng teacher local, reward item/kill/win/death/time, mask, MC returns, advantage không chuẩn hóa, observation12 kênh, LR1e-4 và mọi loss khác. Đây là thay đổi reward dùng train; physics và checkpoint B không đổi.

Preflight định giá lại39.338 transition B của R0: delta chỉ bằng−0,075×số tường đủ điều kiện; các thành phần khác giữ nguyên. Bộ regression209/209 PASS và các guard bổ sung PASS (212 test riêng biệt), smoke CPU/CUDA đầy đủ20episode/nhánh. Full-state resume qua tiến trình mới CPU/CUDA deterministic khớp bit về actions, weights, Adam, scheduler và RNG. Eval/contact logging đối chiếu với evaluator trước R1: actions, kết quả và RNG giữ nguyên. Default caller cũ không bật logging mới.

## Best chọn bằng validation

| Nhánh | Episode | ActiveWR % | Tự nổ % | Hòa % | Được chọn |
| --- | --- | --- | --- | --- | --- |
| Control 0,15 | 0 | 29.00 | 0.00 | 62.00 | True |
| Control 0,15 | 100 | 22.00 | 0.00 | 68.00 | False |
| Control 0,15 | 200 | 33.00 | 0.00 | 59.00 | True |
| Half wall 0,075 | 0 | 29.00 | 0.00 | 62.00 | True |
| Half wall 0,075 | 100 | 29.00 | 0.00 | 64.00 | False |
| Half wall 0,075 | 200 | 36.00 | 0.00 | 55.00 | True |

Best được chọn và khóa hash trước khi mở test. Final200 chỉ phục vụ validation/skill diagnostics; không chấm thêm final trên test để đổi candidate sau khi xem kết quả.

## Test đã đăng ký trước

| Model | Best ep | ActiveWR % | Thắng thụ động % | Hòa % | Tự nổ % | Kills/game |
| --- | --- | --- | --- | --- | --- | --- |
| B | gốc | 34.50 | 5.75 | 58.00 | 0.25 | 0.34 |
| Control 0,15 | 200 | 36.25 | 6.25 | 55.75 | 0.25 | 0.36 |
| Half wall 0,075 | 200 | 35.00 | 6.75 | 56.25 | 0.50 | 0.35 |

| Half wall so với | Delta ActiveWR (pp) | CI95% (pp) | Số seed bàn |
| --- | --- | --- | --- |
| B | 0.50 | [-5.50; 6.50] | 200 |
| Control 0,15 | -1.25 | [-7.25; 5.00] | 200 |

CI percentile bootstrap10.000 lần theo cặp seed bàn, giữ cả hai góc trong mỗi cụm. Gate cần delta≥5pp so với cả B và control, hai cận dướiCI>0, tự nổ quan sát không cao hơn cả hai. Các điều kiện recompute: {'gain': False, 'ci': False, 'suicide': False}.

## Từng góc

| Model | Góc | ActiveWR % | Thắng thụ động % | Hòa % | Tự nổ % | Kills/game |
| --- | --- | --- | --- | --- | --- | --- |
| B | A0 | 32.00 | 9.00 | 56.00 | 0.50 | 0.32 |
| B | A1 | 37.00 | 2.50 | 60.00 | 0.00 | 0.37 |
| Control 0,15 | A0 | 36.00 | 10.00 | 51.00 | 0.50 | 0.36 |
| Control 0,15 | A1 | 36.50 | 2.50 | 60.50 | 0.00 | 0.36 |
| Half wall 0,075 | A0 | 36.50 | 9.00 | 51.00 | 1.00 | 0.36 |
| Half wall 0,075 | A1 | 33.50 | 4.50 | 61.50 | 0.00 | 0.34 |

## Mở đường và tiếp cận đối thủ trên test

| Model | Tường/game | Tường đủ bonus/game | Đã rời ô spawn % | Từng tới gần % | Bước đầu tới gần, có điều kiện |
| --- | --- | --- | --- | --- | --- |
| B | 8.97 | 8.96 | 100.00 | 99.75 | 19.73 |
| Control 0,15 | 9.06 | 9.06 | 100.00 | 99.50 | 19.89 |
| Half wall 0,075 | 8.91 | 8.90 | 100.00 | 99.50 | 19.53 |

Tới gần nghĩa là Manhattan≤3 tới một đối thủ còn sống. Phải đọc tỷ lệ tới gần cùng thời gian có điều kiện; không dùng0 cho các trận chưa tới gần. Rời ô spawn và Manhattan là proxy, chưa chứng minh đường đi an toàn hoặc chiến thuật trap. Tường/game là attribution; tường đủ bonus loại tick learner đã chết.

## Hành vi trong train

| Nhánh | Active wins/200 | Tự nổ/200 | Tường/ep | Tường đủ bonus/ep | Đã tới gần % | Bước đầu, có điều kiện |
| --- | --- | --- | --- | --- | --- | --- |
| Control 0,15 | 70 | 1 | 9.23 | 9.22 | 100.00 | 20.59 |
| Half wall 0,075 | 70 | 1 | 8.73 | 8.73 | 100.00 | 19.41 |

| Nhánh | Time reward mean | Wall mean | Item mean | Owned kill mean | Active win mean |
| --- | --- | --- | --- | --- | --- |
| Control 0,15 | -0.38 | 1.38 | 0.09 | 1.40 | 3.50 |
| Half wall 0,075 | -0.37 | 0.65 | 0.08 | 1.40 | 3.50 |

Mean trên toàn200episode, gồm0 khi sự kiện không xảy ra. Tổng component khớp reward từng episode; wall/coefficient là số nguyên và không vượt số tường attribution. Reward giữa hai nhánh khác hệ số nên không dùng reward train cao/thấp để chọn winner. Trajectory thay đổi sau cập nhật; không diễn giải chênh lệch reward quan sát trong train như phép đối chiếu cùng action tape.

## Kỹ năng trên tình huống validation cố định

| Nhánh | Episode | Escape owned-kill success % | Approach owned-kill success % |
| --- | --- | --- | --- |
| Control 0,15 | 0 | 100.00 | 87.50 |
| Control 0,15 | 100 | 100.00 | 100.00 |
| Control 0,15 | 200 | 100.00 | 93.75 |
| Half wall 0,075 | 0 | 100.00 | 87.50 |
| Half wall 0,075 | 100 | 100.00 | 93.75 |
| Half wall 0,075 | 200 | 100.00 | 100.00 |

Đây là bộ tình huống validation với đối thủ static đã dùng, giúp chẩn đoán kỹ năng; không thay thế full-game RuleBot test và không mở wood_branch/test31000.

## Review và quyết định

ZIP raw 53 files/CRC/bytes, source snapshot, checkpoint và input R0 hashes qua. Đã replay 384 skill games/8308 frames, tính lại validation selection, toàn bộ aggregate/từng góc, contact/walls và bootstrap/gate. Các best có cùng model-state hash với B phải cho kết quả từng game giống B. training_state.pt của mỗi nhánh khớp final/best weights, Adam200, scheduler200 và toàn bộ context/history; RNG Python/NumPy/env/Torch CPU/CUDA đầy đủ. 

Giữ hệ số mặc định0,15 và checkpoint B. Pilot chưa đạt tiêu chí thay model. Không tự chuyển sang wall bonus0 hoặc ghép sửa item/GAE/PPO. Bước tiếp theo cần dựa vào validation, kỹ năng và regression ở các bảng trên; không tiếp tục giảm reward chỉ vì reward train thấp hơn.

Giới hạn: một train seed, một mức treatment, test200 seed bàn và RuleBot hiện tại. CI mô tả chênh lệch policy được chọn trên seed bàn, không đo biến thiên giữa các lượt train. Không kết luận mọi mức giảm reward đều vô ích hoặc reward chắc chắn là nguyên nhân hòa.

Nguồn dữ liệu: [config.json](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/config.json), [summary.json](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/summary.json), [analysis_results.json](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/analysis_results.json), [preflight.json](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/preflight.json), [selected_models.json](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/selected_models.json).
