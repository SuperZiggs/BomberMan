# T3.1 — kết quả và quyết định, 14/10/2026

**Đã hoàn thành ba bước: chứng nhận tình huống combo, kiểm tra teacher và đo CNN gốc. Giữ B. Mở T3.2 curriculum đối thủ biết né/phản công làm thí nghiệm kế tiếp; chưa có lượt train curriculum trong T3.1.** Không cần sửa search để tìm lời giải trên các positive controls đã kiểm chứng, nhưng vấn đề nhãn qua future mask trên các trạng thái thật của T2/L2 vẫn còn.

## Phạm vi, code và bằng chứng

- [benchmark_t3.py](/D:/fpt/subject/BomberMan/benchmark_t3.py), [tests](/D:/fpt/subject/BomberMan/test_benchmark_t3.py), [review_t3_benchmark.py](/D:/fpt/subject/BomberMan/review_t3_benchmark.py) và [hướng dẫn/lệnh](/D:/fpt/subject/BomberMan/T3_GUIDE.md).
- Full:240 train/validation cases,8 orientations gồm xoay/phản chiếu, cả vaiA0/A1,3 opponents và2 RNG repeats; **1.440 games**,498,18 giây. Chỉ B được đánh giá đầy đủ; final200 chỉ xuất hiện trong smoke trước khi thu gọn scope.
- Env9×9/2 agents/max_steps500,12 kênh, reward/mask/evaluator categorical sampling giữ nguyên. Skill horizon80 là giới hạn quan sát ngoài env; chưa giải xong thì `skill_horizon_truncated`, không gọi là hòa500 bước.
- 89 tests qua ở source snapshot dùng chạy full;90 tests qua ở phiên bản hiện tại, gồm test mới cho prefix chết phải ghi survival=false. [Log mới](/D:/fpt/subject/BomberMan/runs/t3-tests-current.log) lưu riêng. Sửa này không đổi tình huống hoặc hành động trong corpus đã chạy. Code core và checkpoint giữ nguyên.
- CRC/SHA ZIP/source/checkpoint khớp; đã dựng lại đủ **1.440 action tapes/62.110 frames**, tính lại mọi outcome, placement, pair forecast và summary. Đã tạo lại240 teacher states/208 legal reference certificates;32 schedules ammo1 chồng bom bị từ chối đúng. 15 game/618 frames được đối chiếu lại mask; runner đã assert mọi action CNN nằm trong mask khi chạy.
- [Config](/D:/fpt/subject/BomberMan/runs/t3-full-20261014-implementation/config.json), [summary](/D:/fpt/subject/BomberMan/runs/t3-full-20261014-implementation/summary.json), [certificates/teacher](/D:/fpt/subject/BomberMan/runs/t3-full-20261014-implementation/certificates_teacher.json), [analysis](/D:/fpt/subject/BomberMan/runs/t3-full-20261014-implementation/analysis_results.json), [games/replay](/D:/fpt/subject/BomberMan/runs/t3-full-20261014-implementation/games.jsonl.gz).
- [Registry336 cases](/D:/fpt/subject/BomberMan/runs/t3-full-20261014-implementation/scenario_registry.json) lưu toàn bộ definitions. `wood_branch` là test family; physics/certificate đã unit-test, **CNN chưa được chấm trên test family**. Fork=train; long_retreat/long_fork=validation. Các biến thể cùng family không chia sang split khác.

## Bước1: combo có lời giải qua mask

96/96 positive combo cases trên train/validation có baseline còn đường sống, từng bom riêng còn đường sống, combo hết đường sống trên cùng horizon. Learner đặt đúng lịch/capacity và có cả prefix lẫn đường thoát qua production mask tới hết bom/lửa. Không tắt mask để làm tình huống đạt.

Ví dụ `fork-combo-a2-r2-combo-rot0-m0-A0`: trên common horizon11, baseline12 ô sống; bom đầu riêng12; bom sau riêng7; combo0. Chuỗi chứng nhận:

```text
t0 BOMB tại(3,3)
t1 UP tới(2,3)
t2 BOMB tại(2,3)
t3 UP → t4 LEFT → t5 LEFT, tới(1,1)
t6–8 STOP an toàn; bom/lửa đã hết
```

Đây là chứng nhận khả năng đóng tất cả đường thoát trong mô hình bom đã biết. Target đứng yên có thể chết bởi một bom dù nó còn refuge trong dự báo; owned kill trên static không tự chứng minh CNN học combo.

Controls gồm: một bom đủ trap, tiếp cận rồi hạ mục tiêu đứng yên, ammo1 không đặt hai bom đồng thời, ammo1 đặt bom sau ở delay8 khi slot trả lại, wood/fire/range trong test family và va chạm/mask/state-RNG trong tests. Reference riêng với prefix định trước, không dùng teacher làm đáp án gốc.

## Bước2: teacher

Teacher hiện tại tìm được **96/96** genuine combos có full prefix/escape qua mask; old representative cũng đạt96/96. Search vẫn ≤64 schedules/call và first actions tương đương đã được kiểm chứng/tái tạo, lưu trong analysis. Vì vậy không sửa search/architecture chỉ để giải các tình huống này.

Reference stress trên96 cases:48/96 complete, sống và mọi tape actions qua mask với evader;48/96 với RuleBot. Các branch còn lại thường bị chặn khi enemy di chuyển; đây là một kịch bản/RNG của reference, không phải mọi phản ứng đối thủ, cũng không phải win rate CNN. Static certificate không bảo đảm prefix vẫn thực thi được trong mọi trận đối thủ phản ứng.

T2/L2 lấy những trạng thái thực khác: không có trap trong6.666 scored paths ở L2. T3 chứng minh planner có positive controls và có thể tạo trap trên bố cục phù hợp; chưa xác định tỷ lệ tình huống như vậy trong train/map ngẫu nhiên, hoặc guard sẽ giữ được bao nhiêu nhãn tại các state khác.

## Bước3: CNN gốc

Chỉ số dưới đây là **owned kill và learner sống trong skill benchmark**, không phải ActiveWR Phase C trên map ngẫu nhiên. Mỗi hàng opponent có cùng cases/repeats trong nhóm tương ứng.

| Nhóm | Static | Evader không đặt bom | RuleBot |
|---|---:|---:|---:|
| Một bom rồi thoát | 64/64 =100% | 60/64 =93,75% | 60/64 =93,75% |
| Tiếp cận mục tiêu | 60/64 =93,75% | 4/64 =6,25% | 4/64 =6,25% |
| Có refuge, ammo1 | 48/64 =75% | 0/64 | 0/64 |
| Combo, ammo2 | 168/192 =87,50% | 92/192 =47,92% | 74/192 =38,54% |

**Không ghi nhận tự nổ trong1.440 games.** Tỷ lệ này là quan sát trên các layout tổng hợp và80 bước, không có nghĩa agent luôn an toàn trên500 bước/map khác. Nhóm ammo1 có refuge không được coi là positive certified combo; thành công0 trên nhóm này chưa chứng minh CNN không học được nhiệm vụ có lời giải.

| Bài combo (192 games/opponent) | Static | Evader | RuleBot |
|---|---:|---:|---:|
| Đặt ít nhất hai bom | 120 | 192 | 178 |
| Có pair thật sự đóng đường thoát theo known-bomb forecast | 80 | 86 | 70 |
| Thực hiện đúng reference prefix | 76 | 76 | 70 |
| Hết80 bước chưa xong | 24 | 100 | 94 |

Pair dùng vị trí/thời điểm hai placement liên tiếp của CNN, dự báo từ state trước bom đầu; các bom đối thủ đặt sau đó chưa được biết. Nó tách khỏi outcome và không chứng minh chỉ pair đó gây ra owned kill.

Trên combo validation: static114/128=89,06%; evader60/128=46,88%; RuleBot52/128=40,63%. Genuine pair lần lượt54/54/48 trên128 games. Train family có owned-kill success54/64,32/64,22/64. Pattern yếu trước enemy di chuyển có ở cả train và validation; không chỉ dựa vào một orientation.

A0/A1 mỗi vai720 games, kết quả giống nhau:353 owned-kill successes,18 passive wins,0 suicide,310 skill truncations;118 games có genuine pair. Vai và RNG repeats không phải samples độc lập; policy nhìn self/opponent channels tương đương nên không suy số game lớn là evidence chắc chắn.

## Quyết định kế tiếp

T3.1a–e đã đóng với bằng chứng. Hướng **T3.2** được chọn để kiểm chứng transfer từ giao tranh đơn giản sang enemy né/phản bom. T3.2 chưa có runner/training trong thay đổi này; không có checkpoint mới hoặc lệnh curriculum giả định.

- Giữ B làm warm-start, reward/mask/12 channels/optimizer/loss như hiện tại. Teacher `local` ở cả candidate và control là phương án ưu tiên để đo riêng curriculum; `two_bomb` trên state thật còn vấn đề future mask từT2/L2, không âm thầm trộn một sửa teacher vào cùng lượt.
- Pilot600 episodes đối chứng600 chỉ RuleBot, cùng seed2026101303/LR1e-4/max_steps500. C1 static → C2 evader → C3 phản bom/trộn RuleBot theo protocol trong PLAN; cùng training loop, không reset optimizer giữa stages.
- Chỉ train trên train family; không dùng wood_branch để tune. Tình huống C2 dạy combo phải có ammo/capacity và lời giải đã chứng nhận; nhóm ammo1 negative control không thay positive training task. Ghi distribution/layout/initial capacity để thấy curriculum khác random Phase C ở đâu.
- Các ngưỡng chuyển stage và dừng phải ghi trước chạy: ít nhất100 episodes, owned-kill≥60% và suicide không vượt guard trên hai skill validations liên tiếp, cap200/stage. Không đạt thì dừng, ghi curriculum incomplete; chưa so như một cặp600 hoàn tất.
- Giữ validation RuleBot100 trận/seed21000 để chọn best và guard như hiện tại; skill validation dùng riêng, không chọn best bằng việc nhớ layout. Log stage/success/steps/time và callback đúng observation sau reset.
- Trước full train: test stage boundaries, reset/obs theo role, evader không đặt bom, optimizer continuity, RNG validation, source/weights không đổi ngoài output, cancellation và budget. Sau đó chạy cặp đối chứng, chỉ mở test xác nhận mới nếu có tín hiệu; gate +5pp/CI/không tăng suicide trong PLAN giữ nguyên.

Không tăng episode map ngẫu nhiên hoặc đổi GAE/PPO/reward ngay sau benchmark này. Curriculum là giả thuyết tiếp theo, chưa được chứng minh sẽ tăng ActiveWR.

## Phạm vi và nguồn quy trình

Ba base families train/validation có hình học gần nhau; orientations/roles/seeds tạo dependent observations. Không dùng1.440 game như1.440 bản đồ độc lập, không đặt CI binomial lên tỷ lệ này để tuyên bố chắc chắn. Skill horizon80 và initial ammo2/range3–4 có thể tạo domain shift so Phase C; cần kiểm chứng transfer bằng bài random-map500 bước.

Skill [scientific-critical-thinking](C:/Users/ADMIN/.codex/skills/scientific-critical-thinking/SKILL.md) hỗ trợ tách reference feasibility, actual execution và outcome, giữ mẫu số/split/phạm vi kết luận. Nguồn quy trình: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065). Metadata đã kiểm tra trên arXiv ngày14/10/2026; bản hiện hành sửa02/09/2026. Đây là nguồn quy trình, không phải nghiên cứu Bomberman hay bằng chứng curriculum hiệu quả.
