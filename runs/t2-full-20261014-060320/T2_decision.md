# T2 đã hoàn tất: giữ checkpoint gốc, ưu tiên L2 trước train tiếp

Ngày đánh giá: 14/10/2026, Asia/Saigon. Lượt: `t2-full-20261014-060320`.

**Quyết định:** giữ B=`runs/pressure-C-next/final_phaseC.pt`. Final two_bomb200 kém B trên audit set này. Ưu tiên **L2: chọn nhãn từ kế hoạch khả thi qua future mask**, bắt đầu bằng kiểm tra candidate offline; chưa mở train dài hơn, PPO/GAE hoặc đổi reward. L2 chưa được triển khai và chưa có bằng chứng nó cải thiện model.

Tệp này chứa kết luận sau review. `summary.json` và `report.md` là output runner tại lúc thu dữ liệu, khi quyết định còn pending.

## 1. Dữ liệu và việc kiểm chứng

- Test log: 74 tests, `OK`. Dòng NativeCommandError là cách PowerShell hiển thị stderr của unittest; không có test failure.
- 400 trận tổng: 200 B, 200 final200; 100 seed bàn28000–28099, mỗi seed A0/A1. Đây là audit chẩn đoán, khác gate T1 400 trận/model.
- ZIP gốc qua CRC, SHA từng member khớp file trong thư mục. Source snapshot và ba checkpoint khớp hashes config.
- Tính lại toàn bộ summary, từng góc, từ 158.293 trace rows, 400 game records, 10.695 plan windows và 88.872 clone branches tại14.812 sampled states: khớp.
- Kiểm tra tick trước/sau, action/label nằm trong mask, plans≤64, six branches/state, trace/checks giống nhau, window không chồng nhau, terminal và suicide khớp game result.
- Bootstrap10.000 lần theo 100 **seed bàn**, giữ A0/A1 chung: khớp CI runner. Không lấy frame, window hay sáu branch làm mẫu độc lập.
- Đã dựng lại state theo action tape của 10 replay được chọn và hai game đối chiếu:12 games/5.608 frames. Fingerprint state, vị trí/alive và mask tại điểm chặn đều khớp.

Tái lập phân tích:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
& 'D:\anaconda\envs\torch\python.exe' .\runs\t2-full-20261014-060320\analyze_t2_results.py
```

Script sử dụng source snapshot của lượt audit, có asserts kiểm chứng. Nó đọc dữ liệu đã lưu; không train hay chơi các trận đánh giá mới. Kết quả: `analysis_results.json`, `T2_replay_review.json`.

## 2. Final200 giảm chất lượng chơi

| Chỉ số | B | Final two_bomb200 |
|---|---:|---:|
| ActiveWR |33,00% (66/200)|16,00% (32/200)|
| Hòa |57,00% (114/200)|75,50% (151/200)|
| Tự nổ |1,00% (2/200)|1,00% (2/200)|
| Passive win |7,50% (15/200)|4,50% (9/200)|
| Kills/game |0,33|0,16|
| Bom đặt thành công/game |40,17|26,68|
| ActiveWR A0 |32/100|15/100|
| ActiveWR A1 |34/100|17/100|

Final−B = **−17,0 điểm phần trăm**, CI95% **[−25,5; −8,5]**. Regression xuất hiện ở cả hai góc. Trên các model cố định và board seeds này có bằng chứng giảm ActiveWR; chưa suy thành mọi cách train two_bomb đều thất bại. Đây là một train seed và audit exploratory, không phải gate promote mới.

Best two_bomb bằng B ở mọi tensor và được ghi alias. Kết quả xấu ở đây thuộc final200, không phải một model được auto-selected vượt B.

## 3. Điểm nghẽn ưu tiên: bước đầu qua mask, prefix phía sau thường không qua

Teacher hiện đánh giá physics/đường sống theo bom đã biết và chỉ kiểm tra `mask[path[0]]`. CNN luôn dùng `_safe_action_mask` ở state thực của từng bước. Mask còn có điều kiện hữu ích của bom, counter-bomb, safety margin và rời vùng nguy hiểm sớm. Hai phép kiểm tra không có cùng semantics.

Đối chiếu hai branch từ cùng state, cùng policy/RNG: chỉ lấy các state có **physical prefix complete**, rồi đếm masked branch bị chặn. Cách này tránh coi terminal-before-prefix là lỗi mask.

| Model / đối thủ | Physical prefix complete | Bị future mask chặn | Tỷ lệ có điều kiện |
|---|---:|---:|---:|
| B / static |5.002|4.424|88,44%|
| B / evader |6.846|6.166|90,07%|
| B / RuleBot |6.139|5.580|90,89%|
| Final / static |6.289|5.690|90,48%|
| Final / evader |7.787|7.066|90,74%|
| Final / RuleBot |7.086|6.454|91,08%|

Tất cả các điểm chặn ở delay>0; hành động nhãn đầu tiên thực sự qua mask. Với final/RuleBot, trong6.454 gap có4.794 lần chặn BOMB:4.080 trước bom đầu và714 trước bom hai;665 lần chặn STOP. Có5.834 gap mà learner vẫn sống hết horizon trong branch physical. Đây là quan sát theo branch/policy, không chứng minh mask thừa trong mọi tình huống.

Mask trên static/evader vẫn robust, có thể xét counter-bomb giả định. Tỷ lệ lớn ngay ở static cho thấy vấn đề không chỉ do RuleBot đặt thêm bom thật. Mask được giữ để bảo vệ learner; ưu tiên sửa teacher cho tương thích với nó.

### Case cụ thể đã dựng lại

- `two_bomb_final-28000-A0`, plan tick30: clone physical đi tới bom hai tại(3,3), tick34 và sống hết horizon. Mask chặn BOMB ở tick34; bom đầu còn hoạt động, ô hiện tại có nguy hiểm dự báo, robust escape không đạt. Đây là gap dù opponent static và không có CNN lệch prefix trong branch.
- Cùng game, plan tick60: clone đi tới bom đầu tại(3,4), tick61; còn đủ ammo, robust escape đạt, nhưng mask chặn BOMB. Mask còn đòi hỏi mục tiêu trên tia nổ hoặc bước lân cận, trong khi teacher chấm pressure trên tập vị trí đối thủ có thể tới. Physics đủ chưa đồng nghĩa action được policy cho phép.
- `original-28001-A0`, plan start177: tới tick185 CNN đặt một bom tại(5,3), còn lịch dự kiến đòi(3,3). Vì thế completion theo kế hoạch thất bại; không được diễn giải như CNN tuyệt đối không đặt bom tiếp theo.

Các record/ticks/capacity/mask chi tiết nằm trong `T2_replay_review.json`. Review dựa trên state và timeline được dựng lại, không dựa vào một trận GUI ngẫu nhiên.

## 4. Completion và tín hiệu nhãn đều yếu

| Chỉ số | B | Final200 |
|---|---:|---:|
| Coverage |69.774/75.136=92,86%|79.147/83.157=95,18%|
| Agreement mọi nhãn |61,23%|52,98%|
| Chance uniform K/M trên cùng state |61,23%|52,65%|
| Agreement khi K<M |18.025/45.073=39,99%|24.279/61.495=39,48%|
| Chance khi K<M |39,98%|39,06%|
| Đúng bom đầu / all starts |1.068/4.940=21,62%|500/5.755=8,69%|
| Đúng bom hai / two-bomb starts có bom đầu đúng |23/905=2,54%|5/427=1,17%|
| Hai placement đúng / all two-bomb starts |23/4.321=0,53%|5/5.045=0,10%|
| Strict prefix + hai placement |1/4.321|0/5.045|
| Nhãn two-bomb dự báo complete trap |1/55.027|3/63.973|
| Nhãn pressure<0,5 |89,95%|89,55%|

Agreement gần K/M cho thấy raw agreement không đủ làm bằng chứng CNN học chiến thuật. K/M là mốc chọn action đều trên **state đang quan sát**, không phải kết quả chơi của một random policy. State distribution hai model khác nhau; bảng này không cô lập tác động của gradient/loss.

Completion dùng tick/vị trí bom chính xác theo một tied path đại diện. Nó có thể bỏ các combo khác vị trí/lịch vẫn hợp lý. Window không chồng nhau; mọi start giữ trong mẫu số, kể cả early terminal. Death là failure quan sát được. Kill trong window chưa có attribution chứng minh do combo.

### Combo thật có tồn tại, nhưng rất hiếm và chưa đi qua mask

Trong corpus sampled, combo đóng toàn bộ đường sống trong khi từng bom riêng vẫn để lại refuge: **0/5.484** kế hoạch hai bom của B và **1/6.390** của final. Không suy tỷ lệ này thành mọi64 candidates: audit kiểm tra đại diện được teacher chọn ở các tick sampled.

Case hiếm `two_bomb_final-28050-A0`, tick60: BOMB tại(4,1) rồi UP và BOMB tại(3,1), tick62. So cùng horizon11: baseline28 refuge; bom đầu riêng25; bom hai riêng4; combo0. Cả ba physical branches sống hết horizon, nhưng cả ba masked branches chặn BOMB ở delay2. CNN thực chọn UP/DOWN/UP ở tick60–62; probability BOMB đầu40,32%. Một lần không sample BOMB không chứng minh không học; vấn đề mask ở bước hai vẫn hiện rõ trên clone.

Game đối chiếu `two_bomb_final-28035-A1`, window325–342: CNN đặt đúng hai bom và sống, dù lệch prefix đại diện. Teacher không dự báo trap; không có owned kill trong window và trận hòa. Đặt được hai bom chưa đủ chứng minh combo hạ đối thủ.

## 5. Vì sao chưa chọn L1, L3, L4 hay PPO

- **L1:** đa số pressure yếu/ít trap, nên có cơ sở thử sau. Lọc pressure≥0,5 không sửa mâu thuẫn future mask; case tick30 pressure0,54 và case tick60/28050 pressure1 vẫn bị chặn.
- **L3:** churn85,48%/86,26% phần lớn đi kèm CNN lệch prefix hoặc thay đổi enemy/hazard/item. Event association là proxy; unknown chưa chứng minh teacher đổi vô lý trên state tương đương. Khóa mục tiêu lúc này có thể giữ một prefix không qua mask.
- **L4:** instrumented teacher mean25,16 ms/query trên B và15,31 ms trên final; đây là khác state distribution cùng code, chưa phải speedup. Chi phí có thật, nhưng semantics là vấn đề trước.
- **T4/PPO/reward:** thay optimizer không làm một BOMB bị mask loại trở thành action thực thi được. Chưa có gradient/reward attribution để chọn một thay đổi optimizer hay reward từ dữ liệu này.

## 6. Nhánh tiếp theo: L2 theo từng bước nhỏ

### L2.0 — kiểm tra candidates offline, chưa train

Chỉ lọc best plan đang có sẽ giữ rất ít combo. Trên static masked corpus, B có578/6.939 prefix pass nhưng chỉ3 là hai bom; final có599/7.873 pass nhưng chỉ4 là hai bom. Các kế hoạch pass còn lại là một bom. Đây là projection đối với **đại diện best được audit**, không phải mọi phương án planner có thể tìm.

1. Dùng source/model snapshot và một corpus audit đã khóa trước, lấy đủ các candidate/path được đánh giá trong giới hạn64 schedules. Capture điểm số và tied paths bằng observer; không đổi policy hay scoring.
2. Trên clone đối thủ STOP, kiểm tra lại mask trước **từng prefix action tới bom cuối**, kiểm tra movement/capacity/placement thật; suffix vẫn phải có đường sống theo forecast. Mask robust giữ như production.
3. Đếm riêng: best bị loại; alternative candidate đi qua mask; valid two-bomb; valid complete-trap; pressure/first-label support; lý do reject. So lại RuleBot/evader trên cùng state như stress checks, không coi static pass là bảo đảm mọi phản ứng.
4. Đo ms và số mask/clone calls. Đăng ký corpus/budget trước; nếu không xử lý hết thì ghi coverage thực, không lấy phần chạy xong làm toàn bộ.
5. Nếu alternatives đủ hữu ích, chuyển L2.1. Nếu combo hợp lệ vẫn rất hiếm, mở **T3.1 benchmark tình huống** để kiểm tra khả năng tạo/gặp combo qua mask trước khi train thêm trên map ngẫu nhiên.

### L2.1 — sửa teacher nhỏ, chỉ sau L2.0

- Chọn kế hoạch khả thi qua future mask trước khi tạo nhãn. Ưu tiên trong những kế hoạch đã qua guard, thay vì chọn best cũ rồi mặc nhiên giữ nhãn.
- Giữ tập first actions tương đương mà có path khả thi tương ứng. Nếu hết candidate/budget hợp lệ, fallback local.
- Guard chạy trên clone, cô lập state/RNG; CNN vẫn sample action qua mask. Prefix pass chỉ áp dụng assumption đối thủ đã định; không tuyên bố chứng minh mọi bom tương lai.
- Khóa bound số plans và ngân sách guard. Dùng L2.0 đo phương án đầy đủ rồi mới chọn một bound thực tế; không thêm64 clone dài mỗi query vào train mà chưa đo tốc độ/recall candidate.
- Giữ observation12 kênh, reward, mask, coefficient0,5 và optimizer. Chưa kèm pressure filter, memory/goal lock, curriculum train hoặc GAE/PPO.
- Fixture phải gồm: physical-pass/mask-fail được reject; combo thật qua mask được giữ; ammo1/2 và delayed release; collision/range; tied paths; pure state/RNG; budget exhaustion fallback.

### L2.2 — chỉ mở pilot sau kiểm tra nhãn/hiệu năng

Warm-start B. Candidate L2 và local control cùng ngân sách200 episodes, train seed, max_steps500, lr1e-4; giữ validation và guard. Chỉ thay lựa chọn nhãn. Ghi coverage/valid combo không được biến mất âm thầm; chọn best như hiện tại, giữ final để chẩn đoán.

Nếu phần lớn nhãn còn lại là một bom hoặc corpus không có kỹ năng combo có thể học, ưu tiên T3.1/T3.2 thay vì kéo dài random-map train. Curriculum là thử nghiệm riêng với đối chứng, không gộp vào cùng pilot L2.

Đánh giá xác nhận theo gate trong PLAN:400 trận/model trên bộ seed mới đã đăng ký (confirm_id1 đề xuất31000–31199, cần kiểm tra registry trước), mỗi seed cả A0/A1; +5 điểm ActiveWR so B và control, tự nổ không cao hơn, cận dưới CI>0. Audit28000 đã được dùng chẩn đoán nên không dùng làm test cuối để promote. Một seed tốt vẫn cần replication.

## 7. Phạm vi kết luận và nguồn quy trình

T2 chẩn đoán đã đóng: dữ liệu đầy đủ, số liệu kiểm chứng được, ≥10 replay được review và đã chọn nhánh ưu tiên. Teacher L2/curriculum chưa được triển khai hay train. Kết quả này giải thích một lỗi thiết kế có quan sát trên corpus; chưa định lượng bao nhiêu regression do riêng mask, nhãn yếu hoặc loss.

Skill sử dụng: [scientific-critical-thinking](C:/Users/ADMIN/.codex/skills/scientific-critical-thinking/SKILL.md), để xác định đơn vị seed bàn, mẫu số, và tách mô tả/nhân quả. Nguồn quy trình, không phải bằng chứng Bomberman: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065). Metadata đã đối chiếu arXiv; bản hiện hành sửa02/09/2026.
