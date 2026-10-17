# T3 — audit approach và counter-bomb đặt trễ, 15/10/2026

**Audit hoàn tất; giữ checkpoint B.** Teacher có đường tấn công hợp lệ trên C1 đã biết, còn CNN sau C1 thường quay lại hoặc đặt bom trước khi tới ô hữu hiệu. Hai replay tự nổ cũng có thể tái tạo bằng phản bom đặt trễ tại cửa thoát. Chưa sửa teacher/mask đang dùng, chưa train hoặc chấm bộ test mới.

## Phạm vi và kiểm chứng

- Dựng lại **256 games / 8.224 frames**: 64 baseline B từ T3.1 và 192 skill probes C1 ở ep100/150/200. Chỉ dùng validation `long_retreat`, static opponent, hai seed2026101401/02; không mở `wood_branch` hoặc31000.
- Mỗi frame khớp action mask, vị trí, placement, kills, self-blast và trạng thái sống của tape. Tái tạo đúng local dispatch của trainer đã đóng băng: navigation0,2; escape0,5; combat0,5. Mọi nhãn/action đều trong mask; teacher/diagnostic không đổi env hoặc NumPy/Torch CPU RNG.
- Chạy64 rollouts theo teacher, dùng hành động đầu tiên được mask cho phép khi không có nhãn; trong các trường hợp này fallback luôn STOP. Không coi rollout đó là CNN tự chơi hay chứng minh thắng trước enemy biết né.
- Đo xác suất của B và final200 trên **cùng trạng thái của đường teacher**, CPU1thread, lấy masked softmax, không sample action. Không có trọng số ep100/150 để đo logits hai mốc đó; dùng action tapes đã lưu.
- **119 tests PASS**, gồm9 tests mới. Runner đầy đủ mất35,85giây khi chạy cùng bộ test; đây không phải benchmark thời gian riêng của production planner.
- Hash toàn bộ12 source đã đóng băng, checkpoint B/final200 và các input tapes khớp, không bị sửa. Các file mới chỉ phục vụ audit/fixture/test và báo cáo.

Code: [audit_t3_approach.py](/D:/fpt/subject/BomberMan/audit_t3_approach.py), [counterbomb_fixtures.py](/D:/fpt/subject/BomberMan/counterbomb_fixtures.py), [test_t3_approach.py](/D:/fpt/subject/BomberMan/test_t3_approach.py).

## Approach: teacher có lời giải, CNN lệch ở đoạn gần mục tiêu

| Cohort | Owned kill /32 | Navigation frames sạch | Đi theo đúng nhãn | Còn1 bước tới ô bắn: đi tiếp /frames |
|---|---:|---:|---:|---:|
| B |28|640|324|32/70 (45,71%)|
| C1 ep100 |12|1.758|886|12/170 (7,06%)|
| C1 ep150 |4|2.228|1.114|4/162 (2,47%)|
| C1 ep200 |12|1.746|828|12/124 (9,68%)|

“Sạch” nghĩa là không có bom/lửa hiện tại; khoảng cách là BFS trên hình học tới ô có tia bom chạm enemy hiện tại, không phải chứng nhận sống hay trap. Tất cả navigation labels trong các frames sạch đều giảm khoảng cách đó. Những trạng thái còn1 bước chỉ có **một** move qua mask làm giảm khoảng cách, nên độ đồng ý thấp ở cột cuối không do teacher tùy ý chọn giữa hai đường cùng tiến.

Ở xa mục tiêu có thể có nhiều move tốt: B đi theo bất kỳ move giảm khoảng cách378/640, final200 là900/1.746. Vì vậy không dùng việc khác nhãn đơn lẻ để kết luận CNN chơi sai. Frame ratios còn chịu ảnh hưởng agent lặp lại cùng vị trí; không coi chúng là số mẫu độc lập hoặc gắn CI thắng thua vào đó.

Teacher closed loop: **approach32/32 và escape32/32** owned kill + learner sống. Mỗi tape có4 bước không nhãn, đều sau placement/đã tới refuge, fallback STOP hợp lệ. Không có thiếu nhãn trên đoạn tiếp cận ban đầu của những rollouts này. Teacher vì thế có thể giải C1 static qua mask; chưa có bằng chứng cần mở rộng search để chữa failure C1.

Đường teacher ở hướng gốc: `(1,1) → (2,1) → (3,1) → (3,2) → (3,3) → BOMB → thoát`. So sánh trên cùng16 cases (8 orientations ×2 roles, không lặp seed):

| Bước teacher | Khoảng cách tới ô bắn | Xác suất đúng nhãn, B | Xác suất đúng nhãn, final200 |
|---|---:|---:|---:|
| Move đầu |4|56,22%|79,74%|
| Move tiếp |3|54,68%|49,49%|
| Move tiếp |2|37,46%|9,78%|
| Move vào vị trí hữu hiệu |1|27,73%|26,30%|
| BOMB tại mục tiêu |0|94,69%|gần100%|

Không thể giải thích failure chỉ từ move đầu: xác suất đó tăng, nhưng đoạn giữa suy giảm. Trung bình bước cuối cũng che lệch hướng: ở rot0/m0, B chọn RIGHT tại `(3,2)` khoảng2,5%, final200 khoảng`7,7e-10`; final200 gần100% BOMB tại ô này. Bom range2 ở `(3,2)` chưa bắn tới enemy `(3,5)`. Mask cho phép đặt vì có thể tạo pressure lên vị trí lân cận; mask không xếp hạng hiệu quả tấn công. Khi thực sự tới `(3,3)`, cả hai mạng rất chịu đặt bom.

Các quan sát phù hợp với policy thiên về quay lại/đặt sớm và kỹ năng approach không được giữ sau C1. **Chưa chứng minh hệ số navigation0,2 hay optimizer là nguyên nhân**: treatment đổi cả layout/opponent và chỉ có một train seed. Teacher labels tại các saved states là chẩn đoán trên validation, không phải bản ghi gradients của từng training update.

## Counter-bomb: đã có fixtures và đối chứng

96 certificate checks =6 variants ×4 rotations ×mirror ×A0/A1:

| Variant | Checks | Kết quả được kiểm chứng |
|---|---:|---|
| `returned_capacity` |16|Enemy ammo1, đứng trên bom cũ; trả slot rồi đặt ở cửa tại delay5. Mask cho BOMB ban đầu; với route enemy cố định learner hết đường sống.|
| `occupied_tile` |16|Enemy ammo2, còn slot nhưng ô hiện tại đã có bom; rời ô và đặt ở cửa tại delay3. Cùng failure.|
| `no_counter` |16|Không đặt bom cửa; có đường qua mask sống tới hết bom/lửa.|
| `alternate_exit` |16|Vẫn có counter-bomb nhưng thêm lối thoát; learner có đường qua mask sống hết nguy hiểm.|
| `capacity_unavailable` |16|Slot chưa trả trước own hazard; actual attempt không đặt được, forecast từ chối nếu giả định đặt. Learner vẫn có đường sống.|
| `unreachable_door` |16|Enemy không tới được cửa; không coi bom đặt ở cửa là phản ứng khả thi. Learner vẫn có đường sống.|

Đối chiếu `_forecast_blasts` với `env.step` từng tick: blockers, free slots trước damage, damage trước flame decay, placement thực tế; gỗ bị phá và bom/lửa phải hết cuối certificate. Future bomb chưa chặn tile trước placement. Control survival được replay qua mask, không chỉ dựa vào BFS hình học. Các checks unsafe kỳ vọng tái tạo tự nổ; test PASS không có nghĩa mask đã được sửa an toàn hơn. Sau khi trận kết thúc, fixture có thể tiếp tục gọi step để kiểm tra physics tới hết lửa, như certificate T3.1; đó không phải thêm turns vào outcome game.

Hai states đã xem cũng khớp seed/tape:32080A0 ownBOMB tick41/counter delay5 và32133A0 tick37/delay3. Chỉ thêm known own bomb còn có timed path; thêm counter và occupancy enemy trước lúc đặt thì không còn. **Ở32080, bỏ occupancy enemy lại tìm được đường đi sớm qua cửa**: chỉ thêm scheduled bomb vào forecast sẽ bỏ sót vai trò đối thủ đứng chắn cửa. Vì thế fix tương lai phải xét route/occupancy khả thi cùng thời gian trả ammo, không chỉ cộng một quả bom tương lai.

Các kết luận no-path ở đây có điều kiện theo route/counter đã định. Chưa giải mọi phản ứng đối thủ hoặc khẳng định mọi lần BOMB trong state tương tự đều thua trước mọi policy.

## Quyết định và bước triển khai tiếp

1. **Đóng audit này; giữ B/local và không train nối C1.** Không nới stage gate, không chọn lại checkpoint bằng test32000.
2. **Ưu tiên một ablation về nhãn move tạo thế tấn công.** Teacher đã tìm được đường; thử chuyển riêng move có positive pressure tăng lên, next placement và escape qua mask trên clone thành combat guidance0,5. Các move chỉ đi tìm item/wood hoặc thiếu certificate vẫn navigation0,2. Giữ CNN tự sample action; không sửa nhãn/hướng đi/reward/mask/optimizer cùng lượt. Việc chuyển loss group còn đổi mean của nhóm combat: phải ghi rõ cả nhóm lẫn hệ số, không diễn đạt như chỉ nhân mọi navigation với2,5. Đây là giả thuyết cần pilot, chưa triển khai hoặc chứng minh hiệu quả.
3. Trước train ablation đó, kiểm tra coverage và certificate trên corpus đã biết; log move trước vị trí bắn, bom sớm, label agreement/entropy và từng nhóm skill. Pilot200ep/arm từ B, normal PhaseC/RuleBot, đối chứng local cùng seed/schedule; chọn best bằng validation, dùng test mới đã đăng ký để xác nhận. Giữ +5pp/paired CI lower>0/suicide guard. Không trộn C2/rehearsal/GAE/PPO vào pilot nhãn.
4. **Nhánh runtime counter-bomb riêng:** bounded opponent route search, capacity return/occupied tile và occupancy tới cửa. Các fixtures trên là regression gates, đặc biệt controls còn đường sống. Đo thêm tỷ lệ BOMB bị chặn, bombs/game, hòa, tự nổ và latency trên corpus đã biết trước test mới. Guard quá bảo thủ có thể tăng hòa; hai failures không đủ chọn guard mặc định.
5. **Curriculum C2 + rehearsal vẫn backlog riêng.** Bỏ C1 bắt buộc với B đã tốt, thử mix75% PhaseC/RuleBot +25% combo/evader trong protocol có đối chứng mới sau khi chọn biến cần kiểm chứng. Chưa ghép nhánh này với thay nhãn hoặc mask.

Không cần người dùng chạy thêm để đóng audit hiện tại. Nếu triển khai pilot tiếp, phải hoàn thành code/test/smoke và đưa bộ lệnh của protocol mới; chưa có runner hoặc yêu cầu chạy train mới trong báo cáo này.

## File tái lập và giới hạn

- [summary.json](/D:/fpt/subject/BomberMan/runs/t3-approach-audit-20261015-final/summary.json): nguồn/hash, cohorts và256 game summaries.
- [analysis_results.json](/D:/fpt/subject/BomberMan/runs/t3-approach-audit-20261015-final/analysis_results.json), [analyze_results.py](/D:/fpt/subject/BomberMan/runs/t3-approach-audit-20261015-final/analyze_results.py): recompute tables/assertions/hash kiểm tra sau run.
- `label_frames.jsonl.gz`, `teacher_rollouts.json`, `teacher_path_probabilities.json`: mọi frame, teacher tape và probabilities trên trạng thái chung.
- `counterbomb_fixtures.json`, `seen_counterbombs.json`:96 certificates và2 states thực đã xem.
- [119-test log](/D:/fpt/subject/BomberMan/runs/t3-approach-tests-20261015-final.log), [console](/D:/fpt/subject/BomberMan/runs/t3-approach-audit-20261015-final-console.log).

Chỉ một train seed, synthetic validation gần train family; roles/orientations/RNG repeats phụ thuộc nhau. Chưa train cải tiến, sửa production guard hoặc đo ActiveWR mới.32000 đã dùng diagnostic;31000/wood_branch vẫn chưa mở. Audit không làm curriculum dừngC1 thành thành công và không chứng minh các ablation được đề xuất sẽ tăng thắng.
