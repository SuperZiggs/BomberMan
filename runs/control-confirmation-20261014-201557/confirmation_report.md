# Control best400 — confirmation hoàn tất, gate FAIL; giữ B

Ngày14/10/2026. Đủ800 games theo protocol đăng ký: B/control best400, mỗi model400 games,200 seed bàn32000–32199/cảA0-A1,9×9/2agents/max500,CPU1thread/RuleBot/mask hiện tại. **Control tăng nhẹ trên mẫu này nhưng chưa được xác nhận tốt hơn B theo gate. Giữ B.** Đây là xác nhận checkpoint control, không phải thành công của curriculum đã dừng C1.

## Audit dữ liệu

- ZIP7 files qua CRC và SHA256 đối chiếu với raw directory; protocol copy khớp byte với đăng ký trước test. Source/checkpoint hashes đúng; không thay model B hoặc source chạy inference.
- Đủ400 unique seed/corner pairs mỗi model, đúng32000–32199/cảA0-A1. Tái tính mọi rate/average/từng góc từ800 game rows; khớp hai evaluation JSON và summary.
- Bootstrap10.000 resamples/seed20261014, resampling theo **200 seed bàn có cả hai góc**; recompute trùng summary. Không coi400 games là400 đơn vị độc lập.
- Review chỉ thêm script/results/replay/report sau ZIP; giữ nguyên ZIP, protocol và các JSON/console raw. Không train hay mở test mới trong review. Test32000 đã đọc và dùng để diagnostic; chỉnh model/code sau này phải có test xác nhận mới. Curriculum test31000/wood_branch vẫn chưa mở.

Audit tái lập: [review_results.py](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/review_results.py), [analysis_results.json](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/analysis_results.json), [analysis_console.log](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/analysis_console.log).

## Kết quả trên test32000

| Chỉ số | B | Control best400 |
|---|---:|---:|
| Active wins /400 |131|140|
| ActiveWR |32,75%|35,00%|
| Win rate |38,00% (152)|40,25% (161)|
| Hòa |60,50% (242)|58,25% (233)|
| Tự nổ |0,50% (2)|0,50% (2)|
| Avg kills |0,3275|0,3500|
| Passive wins |5,25% (21)|5,25% (21)|
| Avg bombs |41,4525|40,6825|
| Avg steps |390,5475|379,6675|

| Góc /200 games | B ActiveWR | Control ActiveWR | B / control draw | B / control suicide |
|---|---:|---:|---:|---:|
| A0 |34,50%|35,00%|56% /57%|1% /1%|
| A1 |31,00%|35,00%|65% /59,50%|0% /0%|

Chênh lệch control−B = **+2,25pp**, bootstrap95%CI **[−3,25625;8,00]pp**.

| Gate đăng ký trước test | Kết quả |
|---|---|
| ActiveWR gain≥5pp |FAIL:2,25pp |
| CI95% difference lower>0 |FAIL:−3,25625pp |
| Observed suicide≤B |PASS:2/400 cả hai |

CI bao gồm0 và cả thay đổi âm; chưa đủ bằng chứng gain dương hoặc đáp ứng mục tiêu≥5pp. CI cũng không loại trừ cải thiện hữu ích; không diễn đạt thành “hai model chắc chắn ngang nhau”. Suicide bằng nhau trên mẫu quan sát không chứng minh true risk bằng nhau hoặc an toàn hơn.

Candidate có80 trận đổi từ B không active win sang active win, nhưng71 trận mất active win mà B có;60 cả hai active win,189 cả hai không active win. Net gain9/400. Tín hiệu validation21000 (42% vs29%,+13pp) chưa được xác nhận tương ứng trên test mới; chênh lệch giữa bộ validation/test không tự chứng minh nguyên nhân overfitting.

## Hai lần tự nổ chung: đã replay

Đã replay4 rollouts (hai seeds × hai frozen models),**188 frames**. Mọi action của learner qua mask, state/RNG không bị diagnostic thay đổi, tất cả game fields khớp evaluation; hai model có **cùng action tape** trong mỗi seed. Không gọi teacher để chọn hành động.

| Seed/A0 | Own bomb được đặt ở tick | Enemy bomb mới tại cửa(1,2) | Tick đầu không còn known-safe move | Death step |
|---|---:|---:|---:|---:|
|32080|41, tại(3,1)|46|47|49|
|32133|37, tại(3,1)|40|41|45|

Tick dùng index0, death step dùng count của env. Khi đặt own bomb, learner còn lối thoát theo forecast; enemy đang ở(2,3) trên bomb đang hoạt động. Enemy sau đó ra cửa(1,2), đặt bomb mới và chặn lối. Sau bomb đó, ngay cả diagnostic bỏ opponent occupancy vẫn không tìm được MOVE/STOP sống hết các known blasts; mask chuyển sang fallback sống tức thời, rồi learner chết bởi own flame. Đây là kết quả của hai replay, không phải guarantee trước mọi hành động đối thủ.

`_can_bomb_and_escape` hiện chỉ thêm counter-bomb tức thời tại vị trí hiện tại của enemy nếu có capacity và cell chưa chứa bomb. Nó chưa đánh giá enemy đi tới cửa khác rồi đặt bom trễ, gồm trường hợp capacity được trả lại trước own bomb nổ. Replay phù hợp với giới hạn này; chưa chứng minh một sửa guard cụ thể sẽ tốt trên mọi map hoặc không làm tăng hòa.

Files: [replay_shared_suicides.py](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/replay_shared_suicides.py), [shared_suicide_replays.json](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/shared_suicide_replays.json), [shared_suicide_summary.json](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/shared_suicide_summary.json).

## Quyết định và backlog tiếp theo

**Khép lại confirmation, gate FAIL; giữ B.** Không chọn final/checkpoint khác hoặc tăng số trận tùy ý sau khi xem test32000 để tìm kết quả qua gate. Chưa cần người dùng chạy thêm cùng checkpoint này.

Ưu tiên offline trước train mới:

1. Audit nhãn di chuyển approach trên tapes C1 đã biết: so local teacher với CNN, kiểm tra nhãn qua mask và đến được vị trí tấn công. T3.2 đã thấy approach giảm87,5%→37,5%; cần biết nguyên nhân label/execution trước curriculum mới.
2. Dựng fixtures positive/negative cho đường thoát một cửa bị counter-bomb đặt trễ, gồm enemy rời bomb cũ/ammo được trả lại. Kiểm tra guard đặt bomb ở bước sớm, không chỉ fallback lúc đã mất đường sống. Sửa mask nếu có thì là experiment runtime riêng, phải kiểm chứng không chặn quá nhiều bomb có lợi hoặc tăng hòa; không trộn vào cùng lượt đổi curriculum.
3. T3.2b còn là đề xuất: bỏ C1 bắt buộc khi B đã tốt C1; thử combo/evader cùng rehearsal PhaseC/RuleBot (giả thuyết75/25), log từng skill group và retention, control/protocol mới. Chưa implement/train trong review này; C2/C3 vẫn chưa kiểm chứng.

Một train seed chưa đánh giá độ ổn định phương pháp; không có candidate qua gate để đề nghị thay B. Các raw test scores không đổi sau diagnostic. Source hiện tại giữ nguyên; chưa chạy ablation/mask fix/GAE/PPO mới.

## Nguồn quy trình

Skill [scientific-critical-thinking](C:/Users/ADMIN/.codex/skills/scientific-critical-thinking/SKILL.md) hỗ trợ giữ gate đã đăng ký, unit seed bàn, tách descriptive gain khỏi xác nhận và tách diagnostic khỏi held-out test. Nguồn: Kassis,T., Agarwal,V., He,Y., Patel,D., & Brueckner,A.M.(2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065). Metadata arXiv kiểm tra14/10/2026, bản hiện hành sửa02/09/2026. Đây là nguồn quy trình, không phải bằng chứng Bomberman control/curriculum hiệu quả.
