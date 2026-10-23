# Báo cáo tổng kết: agent Bomberman bằng reinforcement learning và expert iteration

Ngày 24/10/2026. Số liệu lấy từ `RESULTS.md`; mỗi con số có mục gốc ghi trong ngoặc. Phương pháp và code ở `IMPROVEMENTS.md`, insight ở `INSIGHTS.md`.

## 1. Tóm tắt

- **Agent chính: S3** = `runs/sd9-20261023/train/epoch_12.pt` (SHA `4185375f…69ed7`). Đây là một CNN thuần, mỗi bước chỉ chạy mạng một lần rồi lấy mẫu hành động.
- **Đấu RuleBot** (9×9, 1v1, 500 bước, seed test 57000 chỉ dùng một lần), S3 đạt **ActiveWR 92.25%**:
  - checkpoint xuất phát B khoảng 33%;
  - G (RL tốt nhất, không search) 49%;
  - G + search ba nước (trần viết tay) 94.0%.
  
  S3 kém trần 1.75 điểm, CI95 [−5.0, +1.5] chứa 0, trong khi nhanh hơn khoảng 70 lần.
- **Cách đạt được:** expert iteration. RL tạo G, search viết tay biến G thành thầy mạnh, chưng cất thầy vào CNN, rồi lặp lại: G → S2 → S3.
- **Đấu CNN mạnh** (S2, G, S3), S3 vẫn hòa 53–64% và ActiveWR chỉ 6.5–10.75%. Ba probe cho thấy đây là tính chất của game 1v1 9×9 với mask an toàn chung, không phải điểm yếu có thể sửa bằng search hay đổi lịch vòng (mục 5).

## 2. Bài toán và cách đo

- **Game:** `BombermanFFAEnv`, bàn 9×9, 2 agent ở hai góc, tối đa 500 bước. Observation 13 kênh. Sudden death lịch E: vòng co bắt đầu bước 475, cách 25 bước, cảnh báo 40 bước.
- **Mask an toàn:** mọi agent (kể cả RuleBot) đi qua `_safe_action_mask`, nên không tự bước vào ô sắp nổ.
- **Chỉ số chính, ActiveWR:** % trận thắng *và* có hạ gục bằng bom của chính mình. WR bị loại khỏi vai trò chỉ số chính vì nó tăng cả khi đối thủ tự nổ hoặc chết vì vòng (thắng thụ động), dù agent không giỏi hơn.
- **Giao thức:**
  - Chọn checkpoint trên seed validation 21000.
  - Test **một lần** trên seed mới, 200 bàn × 2 góc = 400 trận mỗi model.
  - So sánh ghép cặp theo bàn, bootstrap 10 000 lần.
  - Gate chốt trước khi chạy: ActiveWR ≥ +5pp, cận dưới CI95 > 0, tự nổ không tăng quá 1pp.
  - Seed test đã dùng không bao giờ dùng lại; danh sách ở `CLAUDE.md`.

## 3. Các mốc chính (đấu RuleBot, lịch E)

| Mốc | Agent | ActiveWR | Seed test | Mục RESULTS |
|---|---|---|---|---|
| Xuất phát | B (Phase C, MC actor-critic) | 32.75 | 32000 | control confirmation |
| RL nhánh bạn | F (~3000 ep từ B) | 44.25 | 43000 | SD5.3 |
| RL + advantage theo nhóm | **G** | 48.25–51.0 | 45000, 46000 | SD7.5, SD7.8 |
| Search một nước lúc chơi | G + search | 64.75 | 46000 | SD7.8 |
| Search hai nước | G + search2 | 82.0 | 47000 | SD7.9c |
| Search ba nước (trần) | G + search3 | 94.0 | 49000 (và 57000) | SD8B, SD9 |
| Chưng cất lần 1 | S (từ F + search) | 60.5 | 43000 | SD5.3 |
| Chưng cất lần 2 | S2 (từ G + search2) | 83.0 | 48000 | SD8A |
| **Chưng cất lần 3** | **S3 (từ S2 + search3)** | **92.25** | 57000 | SD9 |

Trên test 57000, S3 − S2 = +10.5 [+6.5, +14.5] pp ActiveWR. Phần tăng là thắng chủ động: thắng thụ động −1.5 (CI chứa 0), hòa −6.25, ván ngắn hơn (131 so với 202 bước). Tắt sudden death cho kết quả gần như y hệt (+10.75).

## 4. Hành trình

1. **Tinh chỉnh RL quanh B (T1–T4).** Teacher hai bom, curriculum, MOVE guidance, GAE/PPO và reward shaping đều trượt gate trên seed test riêng. Kết luận: train RL thêm với tín hiệu thưa không đổi được hành vi đặt bom của CNN.
2. **Sudden death (SD1–SD4, SD6).** Vòng co và luật chặn đường tăng WR nhưng toàn bằng thắng thụ động; ActiveWR không đổi. Từ đây, mọi quyết định đều dựa trên ActiveWR.
3. **Kill-search lúc chơi (SD4.6, SD5.0, SD7.8–SD8B).** `kill_search.py` mô phỏng 12 bước tới, giả định đối thủ là RuleBot, và chọn nước có kế hoạch giết rồi né được. Search một, hai rồi ba nước nâng G từ khoảng 48 lên 94. Nhược điểm: mỗi trận tốn 20–30 giây.
4. **RL có cấu trúc (SD7).** Advantage theo nhóm, chơi lại K lần từ các khoảnh khắc quyết định, tăng tỉ lệ giết (+7.1pp) và tạo ra G. Tuy vậy gate test vẫn FAIL, và phạt KL không tách được "đặt bom đúng lúc" khỏi "đặt bom bừa".
5. **Chưng cất dày (SD5.3, SD8A, SD9).** Ghi nhãn search ở *mọi* trạng thái mà các nước có điểm khác nhau, rồi train CNN bằng cross-entropy mục tiêu mềm, cộng KL về model gốc trên 5% trạng thái neo. Đây là bước làm CNN thuần tiến gần trần search: F → S (+16.25), G → S2, S2 → S3 (+10.5).
6. **RL tiếp từ model chưng cất (D-S2, S2-KL, nhánh bạn).** Phase D 2000 ep làm ActiveWR đấu RuleBot rơi từ 83.75 xuống 57.5–65. KL neo toàn cục chỉ giữ được 2.5pp. Kết luận: RL hiện tại xóa kỹ năng chưng cất, nên dừng.

## 5. Đấu CNN: vì sao hòa nhiều

| Probe | Câu hỏi | Kết quả |
|---|---|---|
| PDE, PDE2 | Agent mạnh nhất có còn mạnh khi đấu CNN? | G + search3 đấu CNN chỉ WR 19–44, hòa 50–70% |
| SD9.1 | S3 có bớt hòa so với S2? | Macro CNN: ActiveWR +3.33 [+1.25, +5.50], hòa −4.08; S3 vẫn hòa 53–64% |
| SD9.2 | Trong trận hòa có nước giết bị bỏ lỡ? | Đoán đối thủ bằng chính CNN của nó: chỉ 0.2% trạng thái và 15.9% [9.9, 22.7] trận hòa có kế hoạch giết. Giả định RuleBot thấy 90.3%, nghĩa là các bẫy kiểu RuleBot không ăn CNN |
| SD9.3 | Vòng co sớm hơn có tạo thắng chủ động? | Không: ActiveWR −1.42…−0.17. 80–87% trận hòa thành "cùng chết khi vòng cuối lấp kín bàn" |
| SDM | Search đoán đối thủ bằng CNN thật? | Giúp khi đấu G (+14.0), không rõ khi đấu S; luôn đoán bằng CNN thì mất mạnh khi đấu RuleBot (SDM2 −33) |

**Giải thích:** hai agent cùng dùng mask an toàn, nên mỗi bên đều không bao giờ tự bước vào vùng nổ. RuleBot né theo luật cố định nên bị bẫy được. CNN né theo nhiều cách, và hầu như luôn còn đường thoát. Kết quả là trong 1v1 trên bàn 9×9 hầu như không tồn tại cú giết chắc chắn. Điều này giống hai kỳ thủ mạnh hay hòa nhau.

Vì vậy, khi đấu CNN, báo cáo này dùng ActiveWR và tỉ lệ thua, không dùng WR.

## 6. Giả thuyết đã bị bác bỏ (rút gọn, đủ ở `INSIGHTS.md`)

- Train RL thêm với nhãn search thưa (SD5.1: −2.75).
- Hạ nhiệt độ lấy mẫu của F (argmax −9.25).
- Search chỉ ở giai đoạn cuối trận (+0).
- Luật đứng chặn ô cửa (ActiveWR −1.75, WR tăng nhờ thắng thụ động).
- Search luôn đoán đối thủ bằng CNN của mình (đấu RuleBot −33).
- KL neo toàn cục khi train RL tiếp (S2-KL).
- Vòng sudden death sớm hơn (SD9.3).

## 7. Giới hạn

- Mỗi kết quả chính dựa trên một train seed và một test seed, nên chưa chứng minh độ ổn định qua nhiều lần train.
- Thầy của S3 giả định đối thủ là RuleBot. S3 mạnh nhất khi đấu RuleBot, còn khi đấu CNN thì chỉ nhỉnh hơn S2 chút ít.
- Policy lấy mẫu: số liệu tái lập bit-to-bit trên cùng phiên bản torch, không tái lập được giữa các phiên bản.
- Chủ yếu đo 1v1 9×9. FFA 3–4 agent và bàn 11×11 mới được đo mô tả (PDE, PDE2), chưa train.

## 8. Hướng tiếp theo

**SD10 (hướng A):** chưng cất S3 + search3 thành S4.
- Khi thu dữ liệu, search đoán đối thủ bằng chính CNN của đối thủ (biết được vì đối thủ là S2/G/S3). Ván đấu RuleBot vẫn đoán bằng RuleBot.
- S4 lúc chơi là CNN thuần, không cần biết đối thủ.
- Theo SD9.2, mức lợi ích tối đa đấu CNN chỉ khoảng +10pp, và đây là ước lượng lạc quan.
- Spec: `docs/superpowers/specs/2026-10-24-sd10-cnn-aware-distillation.md`, lệnh chạy ở `SD10_GUIDE.md`.

## 9. Tái lập

- **Đánh giá S3 đấu RuleBot:**
  `python evaluate.py --mode seeded --board_size 9 --num_agents 2 --num_episodes 400 --max_steps 500 --seed <seed> --device cpu --ac_model runs/sd9-20261023/train/epoch_12.pt --output <json>`
- **Chuỗi tạo S3:**
  - `run_sd8a_distill.py` tạo S2 (bạn chạy);
  - `run_sd9_distill.py collect | train | validate | test` tạo S3;
  - lệnh đầy đủ ở `SD9_GUIDE.md`.
- **Probe đấu CNN:** `probe_s3_cnn.py` (SD9.1), `probe_draw_kills.py` (SD9.2), `probe_sd_schedule.py` (SD9.3). Mỗi runner có test `test_*.py` đi kèm, chạy bằng `unittest`.
- **Môi trường:** Python 3.11, torch 2.5.1 (`D:\anaconda\envs\torch\python.exe`), CPU, 5 tiến trình song song.
