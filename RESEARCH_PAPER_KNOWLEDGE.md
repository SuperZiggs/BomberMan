# Knowledge base cho research paper: Bomberman RL → expert iteration với search viết tay

Ngày 25/10/2026. File này gom mọi thứ cần để viết một research paper nộp hội nghị: bài toán, insight rút ra từ 35 tài liệu, research gap, câu hỏi nghiên cứu, giải pháp của repo, các phát hiện đã có bằng chứng, các phát hiện tiềm năng còn phải kiểm, và giới hạn.

**Nguồn:**
- Số liệu thí nghiệm lấy từ `RESULTS.md`, `INSIGHTS.md`, `BAO_CAO_TONG_KET.md`. Mỗi con số ghi mục gốc trong ngoặc.
- Insight tài liệu lấy từ việc đọc toàn văn 34/35 tài liệu ngày 25/10/2026. Riêng Goulart 2019 chỉ đọc tóm tắt, vì HAL chặn tải tự động.
- Báo cáo đầy đủ (có công thức và bảng thí nghiệm): https://claude.ai/artifact/AdtApiN8GuySvgs7S7RbGz

**Quy tắc khi dùng file này để viết paper:**
- Chỉ dùng mục 6 ("Phát hiện đã có bằng chứng") làm claim.
- Mục 7 ("Phát hiện tiềm năng") là giả thuyết. Có thể viết vào Discussion hoặc Future work, không được viết thành kết quả.
- Luôn báo ActiveWR cùng CI95. Không dùng WR làm chỉ số chính.
- Các mốc ở mục 6.1 được đo trên seed test khác nhau. Chỉ so trực tiếp các cặp ghép trên cùng seed; so giữa các mốc chỉ mang tính mô tả.

---

## 0. Ký hiệu và quy ước

| Ký hiệu | Nghĩa |
|---|---|
| B | `runs/pressure-C-next/final_phaseC.pt`. Checkpoint Phase C (policy gradient CNN + mask + teacher), mốc xuất phát |
| F | `runs/phaseC-sd-best-fallback/best_phaseC.pt`. Khoảng 3000 ep RL tiếp từ B (nhánh bạn), 13 kênh |
| G | `runs/sd75-20261018/full/best_phaseC.pt`. RL + advantage theo nhóm (SD7.5); CNN tốt nhất không có search |
| search1/2/3 | `kill_search.search_action`, `search_action2`, `search_action3`: kill-search 1, 2, 3 nước, mô phỏng 12 tick, đối thủ giả định là RuleBot |
| S | CNN chưng cất từ F + search (SD5.3) |
| S2 | CNN chưng cất từ G + search2 (SD8A), `runs/sd8a-20261020/train/epoch_16.pt` |
| S3 | CNN chưng cất từ S2 + search3 (SD9), `runs/sd9-20261023/train/epoch_12.pt`. **Agent chính** |
| S3-11 | S3 chưng cất thêm trên bàn 11×11, `runs/s3-map11-20261024-auto-v1` |
| ActiveWR | % trận thắng **và** có hạ gục đối thủ bằng bom của chính mình |
| Lịch E | Sudden death: vòng co bắt đầu bước 475, mỗi 25 bước, cảnh báo 40 bước |

**Điều kiện mặc định của mọi số liệu:**
- Bàn 9×9, 1v1, đối thủ RuleBot, tối đa 500 bước, lịch E.
- Policy lấy mẫu (không argmax).
- 400 trận mỗi model (200 bàn × 2 góc).
- CI95 bằng paired bootstrap theo bàn, 10 000 lần.

---

## 1. Thông điệp chính của paper

### 1.1 Một câu đóng góp

Trong Bomberman 1v1, RL model-free bão hòa ở mức khoảng 33–51% ActiveWR. Một search viết tay **không dùng mạng** (kill-search 1–3 nước, mô phỏng 12 tick) nâng agent lên 94%. Chưng cất dày nhãn search lặp ba vòng (G → S2 → S3) đưa một CNN thuần lên 92.25% ActiveWR, kém thầy 1.75 điểm (CI chứa 0) mà mỗi quyết định nhanh hơn khoảng 115 lần (đo 25/10/2026, `paper/figures/decision_timing.json`). Đồng thời, các thí nghiệm có đối chứng chỉ ra hai điều:
- Win rate thông thường đánh giá sai agent, vì thắng thụ động làm WR tăng giả.
- Hòa giữa các agent mạnh là hiện tượng cấu trúc của game, không phải lỗi của agent.

### 1.2 Tên paper ứng viên

1. *Winning Is Not Killing: Active Win Rate and Expert Iteration with Hand-Crafted Search in Bomberman*
2. *From Search to Reflex: Distilling a Non-Neural Kill-Search into a CNN Policy for Bomberman*
3. *When Does Search Help Reinforcement Learning? A Controlled Study in a Bomberman Markov Game*

### 1.3 Đóng góp (dạng bullet cho Introduction)

- **C1, thước đo.** Định nghĩa ActiveWR và ghi công kill theo chủ quả bom. Thước đo này cho thấy một agent "đạt gate" có WR 52.5% nhưng 0 lần hạ gục trong 40 trận, và loại được nhiều cải tiến giả (mục 6.2).
- **C2, so sánh có đối chứng ba cách dùng search.** Cả ba cùng dùng một search:
  - nhãn search thưa làm loss phụ trong RL: −2.75;
  - search lúc chơi: +16.5;
  - chưng cất dày nhãn search: +16.25.

  Kết quả này kiểm chứng định lượng hai hướng tương lai mà Kartal et al. (2018) tự đề xuất nhưng chưa thử.
- **C3, expert iteration với chuyên gia không dùng mạng.** ActiveWR đi B 40.0 (lịch E, block 60000; 32.75 khi chưa có sudden death) → G 48.25 → S2 83.0 → S3 92.25. Khoảng cách học trò–thầy (mỗi học trò so với thầy của chính nó, cùng seed test) là −3.75 → −3.5 → −1.75 qua ba vòng.
- **C4, bộ kết quả âm có kiểm soát.** Gồm GAE, PPO, chuẩn hóa, shaping, curriculum, sudden death, luật chặn đường, group RL, KL neo. Tất cả có luật chọn chốt trước, seed test chỉ dùng một lần và gate định lượng.
- **C5, phân tích hòa cấu trúc.** Probe SD9.2 cho thấy khi đấu CNN, chỉ 15.9% [9.9, 22.7] trận hòa có kế hoạch giết theo mô hình đối thủ đúng. Mô hình đối thủ sai loại làm mất 33pp (SDM2).
- **C6, chuyển sang bàn lớn hơn.** Chưng cất lại trên 11×11 cho +41.0 [34.5, 47.25] khi đấu RuleBot, nhưng chỉ +3.83 [1.92, 5.83] khi đấu pool CNN (chưa đạt gate).

---

## 2. Bài toán (Problem)

### 2.1 Phát biểu hình thức

Đây là Markov game hai người G = (S, A₁, A₂, P, r, γ):
- 9×9 ô, cột cứng ở các ô (chẵn, chẵn), tường gỗ phá được.
- Mỗi agent có 6 hành động: STOP, 4 hướng, BOMB. Bom nổ sau một khoảng fuse cố định, có nổ dây chuyền.
- Observation 13 kênh dạng lưới. Kênh 11 là nghịch đảo thời gian tới khi bị sát thương, tính bằng `_forecast_blasts`. Kênh 12 là cảnh báo vòng sudden death.
- Learner tối ưu policy π_θ khi đấu policy cố định π₋ᵢ (RuleBot, hoặc pool CNN đông cứng).

Mục tiêu nghiên cứu **không phải** là xác suất thắng, mà là:

```
ActiveWR(π) = P( learner thắng  ∧  ít nhất một đối thủ chết vì bom của learner )
```

Ràng buộc:
- Tự nổ không được tăng.
- Ngân sách tính toán là **một laptop CPU**, tối đa khoảng 6 tiến trình song song.
- Mỗi lượt train là 200–3000 episode.

### 2.2 Vì sao khó (bằng chứng từ repo và tài liệu)

1. **Khám phá tự sát.** Sau khi đặt một quả bom, đi ngẫu nhiên sẽ tự sát khoảng 40%. Đặt k bom thì xác suất sống còn (0.6)^k (Gao et al. 2019, Hard Exploration). Phase C ban đầu của repo tự nổ 84% và rơi vào bomb shyness (không đặt bom).
2. **Thưởng đánh lừa.** Đối thủ tự nổ vẫn cho learner +1 hoặc +10. Ít nhất bốn nhóm tác giả độc lập gọi tên vấn đề này: Skynet "deceptive reward", Kartal "false positive", Huynh "false-positive reward", Weil "win rate alone is not enough".
3. **Khoảnh khắc quyết định cực thưa.** Một trận dài khoảng 230 bước nhưng chỉ khoảng 2.2 bước có thông tin (các hành động cho kết quả khác nhau), và khoảng 1.25 bước có nước giết (INSIGHTS §4). Theo Williams (1992) §8, episodic REINFORCE chia đều công cho mọi bước nên rất chậm. Với γ = 0.99, tín hiệu từ một kill ở bước 300 chỉ còn khoảng e⁻³ ở bước 0 (GAE).
4. **Cửa sổ giết rất ngắn.** 31/37 cửa sổ giết thất bại chỉ kéo dài 1 bước. Ở 28 trong số đó, nước giết duy nhất là đặt bom ngay, nhưng F chỉ gán xác suất ≈ 0.58 cho nước này (SD7.6a).
5. **Mô hình đối thủ.** Một mô hình đối thủ chính xác nhưng sai loại làm search hỏng: đấu RuleBot mất −33 (SDM2). Weil 2023 kết luận không có mô hình đối thủ đơn nào tốt cho mọi đối thủ.
6. **Hòa cấu trúc.** Khi hai bên đều có bộ lọc an toàn thì gần như không tồn tại cú giết chắc chắn. Bằng chứng ở mục 6.6.

---

## 3. Insight rút ra từ tài liệu

Mỗi insight gồm: phát biểu, bằng chứng (paper + số liệu), và hệ quả cho paper này.

### I1. RL model-free thuần không đủ cho Bomberman; mọi agent mạnh đều có bộ lọc an toàn hoặc search

- Gao 2019: không cắt hành động thì PPO đạt 0% sau 5 ngày; có cắt thì học được.
- Resnick 2018: không có batch lớn + shaping thì DQN/PPO không thắng SimpleAgent; DAgger chỉ khoảng 20%.
- Osogami 2019: ba agent search thắng hai agent RL hàng đầu 87.1%, thua 2.7%, hòa 10.2% (22 500 trận).
- Huynh 2024: sau 200 triệu bước, RL vẫn thua agent search 70–95% (dypm 30%, eisenach 8.89%, hakozaki 5.17%).
- Peng 2018 (COMBAT): cần 220 CPU + 32 GPU, hàng chục ngày.
- **Hệ quả:** con đường "mask + search + học" của repo trùng với hướng mà chính Gao 2019 và Osogami 2019 đề xuất. Ngân sách một laptop khiến RL quy mô lớn không khả thi, nên lựa chọn này là bắt buộc chứ không chỉ là sở thích.

### I2. Thưởng giả do đối thủ tự sát được nhiều bài gọi tên, nhưng chưa ai tách nó ra bằng một thước đo

- Skynet (Gao 2019): "thắng thường do đối thủ vô tình tự sát". Họ giải bằng đối thủ không đặt bom; khi đó tỉ lệ thắng không vượt 20%, phần còn lại là hòa.
- Kartal 2018: false positive thưởng cả lối chơi cắm trại.
- Huynh 2024: đo bằng số địch bị hạ và phạt hòa −1.
- Weil 2023: ExIt làm agent trở nên thụ động (tỉ lệ IDLE tăng), thắng vì SimpleCpp tự đặt mình vào thế xấu.
- Henderson 2018: Swimmer "cuộn tròn quẫy" cho return đẹp mà sai hành vi.
- **Hệ quả:** ActiveWR là đóng góp thước đo có động cơ rõ từ tài liệu (gap G1).

### I3. Hòa áp đảo khi các agent mạnh gặp nhau là hiện tượng chung

- Osogami: trận gương hòa 776–886/1000.
- Perez-Liebana: MCTS hòa 59–84% khi các thuật toán mạnh đấu nhau.
- Skynet: đối thủ an toàn dẫn tới hòa áp đảo.
- Kartal (Action Guidance): MCTS thuần đấu Static được 88 thắng / 108 hòa / 4 thua.
- AlphaZero vs Stockfish: 72/100 hòa.
- Weil: phần lớn hòa là do hết 800 bước.
- **Hệ quả:** "S3 hòa 53–64% khi đấu CNN" không phải điểm yếu riêng của repo. Điểm mới của repo là probe tách "hòa do luật" khỏi "hòa do agent" (SD9.2, gap G5).

### I4. Có hai cách dùng search trong RL; chỉ cách "lưu nhãn và học có giám sát nhiều lần" hiệu quả khi xuất phát từ policy đã mạnh

- Kartal 2018/2019: dùng MCTS làm demonstrator trong A3C, đo trên bản mini 8×8 bằng đường cong reward vài seed. Họ tự đề xuất hai hướng tương lai: (a) gọi demonstrator ad-hoc chỉ ở tình huống liên quan, (b) lưu nhãn vào buffer. Cả hai chưa được thử.
- Kartal 2019 (Safer): 75 rollout giúp học nhanh hơn 150 rollout, tức số lượng nhãn quan trọng hơn chất lượng nhãn.
- **Hệ quả:** repo thử cả hai hướng:
  - (a) ad-hoc trong RL (SD5.1): −2.75, vì chỉ khoảng 0.9 nhãn/episode và học một lần;
  - (b) buffer + học nhiều epoch (SD5.3, SD8A, SD9): +16.25, rồi +35.0, rồi +10.5.

  Đây là bằng chứng định lượng mới cho gap G2.

### I5. Expert iteration chỉ tự nâng trần khi chuyên gia dùng học trò

- Anthony 2017 (ExIt): học trò được dùng để cải thiện chuyên gia (prior và value trong UCT). Bỏ bước này thì online ExIt = DAgger. Nhãn mềm TPT hơn nhãn cứng CAT +50 ± 13 Elo. Online (gộp dữ liệu các vòng) tốt hơn batch. Lấy một trạng thái mỗi ván để giảm tương quan.
- AlphaZero: MCTS dùng prior p và value v của mạng.
- **Hệ quả:**
  - Search của repo không dùng CNN, nên không có bước cải thiện chuyên gia. Trần dừng ở khoảng 94%.
  - SD9.0 xác nhận: nền mạnh hơn (S2 so với G) không làm search3 mạnh hơn (−5.0 [−11, +1]).
  - Đây vừa là giới hạn, vừa là hướng tiếp theo có căn cứ (P1, P2 ở mục 7).

### I6. Học có giám sát từ thầy hiệu quả khi dữ liệu gần phân phối học trò và nhãn mềm, nhọn

- DAgger: học trên phân phối của chuyên gia làm lỗi tăng T²ε. Học trên trạng thái học trò thăm thì lỗi chỉ tăng tuyến tính.
- Policy Distillation: KL với softmax(q/τ), τ = 0.01, tốt nhất. MSE trên Q kém. Học trò 25% tham số đạt 108% thầy.
- Backplay và Resnick: DAgger bắt chước một chuyên gia khác kiểu chỉ khoảng 20%.
- **Hệ quả:** thầy của repo = chính học trò + search can thiệp ở bước có thông tin, nên trạng thái gần phân phối học trò. Nhãn chia đều các nước có điểm search cao nhất (≈ τ → 0, giữ đồng hạng). Đây là lý do chưng cất thành công, trái với thất bại của bắt chước trong Pommerman.

### I7. RL sau giai đoạn có thầy dễ xóa kỹ năng; hệ số neo cố định không đủ

- Kickstarting (Schmitt 2018): λ hằng số giữ học trò quanh mức thầy (41.2 / 40.8). λ giảm tuyến tính 1 → 0 cho học trò vượt thầy (59.4 / 60.8).
- PPO: phạt KL với β cố định kém hơn clip và khó chọn một β dùng chung.
- Meisheri 2019: chuyển từ bắt chước sang PPO bị quên thảm họa. Họ sửa bằng entropy = 0, clip 0.01, value net tách riêng.
- Peng 2018: kỹ năng cơ bản không học sớm thì về sau không học lại được.
- **Hệ quả:** giải thích các kết quả âm của repo:
  - D-S2: 83.75 → 57.5–65;
  - S2-KL: β = 1 chỉ giữ +2.5pp;
  - SD7.7: không β nào tách được hai hiệu ứng.

  Cả ba đều dùng hệ số cố định và neo về chính model cũ.

### I8. RL làm vững cái đã biết, không thêm năng lực mới

- GRPO (Shao 2024): RL tăng Maj@K nhưng không tăng Pass@K.
- **Hệ quả:** khớp với SD7–SD7.5. Group RL tăng tỉ lệ giết +7.12pp ở những trạng thái mà F đã giết được "lúc được lúc không" (86%), nhưng ActiveWR test chỉ +3.75 (FAIL). Năng lực mới chỉ đến từ thầy mạnh hơn (search) và chưng cất.

### I9. Mô hình đối thủ là nút thắt của search; một mô hình đơn lẻ sẽ overfit

- Weil 2023: dùng mạng làm mô hình đối thủ có thể khiến search phòng thủ quá mức (0.78 → 0.67) và chậm 4–9×.
- PSRO (Lanctot 2017): policy học độc lập mất 34–72% phần thưởng khi gặp policy được huấn luyện riêng (JPC). Meta-strategy hỗn hợp giảm JPC 29–57%.
- DOMAC: mô hình đối thủ đặt trong policy, học chỉ từ return, gần cận trên.
- Osogami: tránh hẳn vấn đề bằng kịch bản **bi quan** thay vì đoán đúng đối thủ.
- **Hệ quả:** SDM2 (+12 khi đấu F, −33 khi đấu RuleBot) là một trường hợp JPC. Hướng thay thế có căn cứ: kịch bản bi quan, mô hình hỗn hợp, mô hình đối thủ học được (P3, P4).

### I10. Shield/mask: đúng cách cài và đúng cách học

- Huang & Ontañón 2022: che logit vẫn cho policy gradient hợp lệ. Phạt hành động sai không mở rộng được. "Naive masking" kém dần theo kích thước bài toán.
- Alshiekh 2018: shield tổng hợp từ trò chơi an toàn vừa đúng vừa can thiệp tối thiểu. Không phạt hành động bị sửa thì học tốt hơn.
- **Hệ quả:**
  - Repo che logit đúng kiểu: `masked_fill(~mask, -1e9)` dùng khi lấy mẫu, khi tính `log_softmax` cho loss và cho anchor KL (`train_actor_critic.py` dòng 439, 450, 1147).
  - Mask là shield heuristic có các lớp dự phòng (`safety_margin`, `avoid_contests`, `robust`), chưa phải vùng thắng trước mọi nước đi của đối thủ.
  - Margin k = 15 (SD2) bảo thủ quá: chỉ tăng thắng thụ động. Đây là minh họa cho nguyên tắc "can thiệp tối thiểu".

### I11. Shaping theo sự kiện có thể đổi chính sách tối ưu; thưởng dày phải giảm dần

- Ng 1999: chỉ shaping dạng thế γΦ(s′) − Φ(s) mới bảo toàn chính sách tối ưu. Ví dụ shaping sai: xe đạp chạy vòng, robot "rung" cạnh bóng.
- Bansal 2018: agent giữ thưởng dày thua agent đã giảm thưởng dày về 0.
- **Hệ quả:** thưởng của repo là theo sự kiện: thời gian −0.001, item, phá tường 0.15, kill 4.0, thắng chủ động 10, chết −1, tự nổ −1.5. Việc trừ +10 cho thắng thụ động là sửa mục tiêu cho khớp ActiveWR, không phải shaping. Các hệ số teacher cố định có thể đang giới hạn trần.

### I12. Đánh giá RL cần CI theo đơn vị độc lập, cỡ hiệu ứng tối thiểu và nhiều lần train

- Henderson 2018: 5 seed so với 5 seed, cùng siêu tham số, đã khác nhau có ý nghĩa (t = −9.09, p = 0.0016).
- Agarwal 2021: nên dùng CI bootstrap phân tầng, IQM, P(X > Y), và tránh p < 0.05 nhị phân. Đổi giao thức đánh giá có thể tạo khác biệt lớn hơn cả khác biệt thuật toán.
- Efron 1979: phải lấy mẫu lại đúng đơn vị trao đổi được. Dùng thống kê hiệu để so sánh hiệu quả hơn.
- **Hệ quả:**
  - Repo làm đúng phần lớn: bootstrap ghép cặp theo seed bàn, gate (Δ ≥ +5pp và cận dưới CI > 0), giao thức cố định, seed test chỉ dùng một lần, báo cả FAIL.
  - Chỗ thiếu: CI của repo đo biến thiên do bàn đánh giá, chưa đo biến thiên giữa các lần train (mỗi nhánh chỉ một train seed).

### I13. Không augment đối xứng khi động lực bất đối xứng

- AlphaZero bỏ augment ở cờ vua và shogi vì luật bất đối xứng.
- Weil 2023 có augment xoay/lật.
- **Hệ quả:** repo đo được 21% trạng thái đổi tập nước tốt nhất khi xoay bàn, vì search và RuleBot không đối xứng. Vì vậy repo không augment. Đây là chi tiết phương pháp đáng nêu.

### I14. Pha quyết định của Bomberman 1v1 là giao tranh giữa trận, không phải tàn cuộc

- M2CTS (Helfenstein 2024): chia chuyên gia theo "mixedness" (khoảng cách tới đối thủ) giúp 5–11% trong Pommerman.
- **Hệ quả:**
  - Cổng REACH ≤ 6 của kill-search tương đương một cổng pha.
  - Repo cho thấy teacher hoặc search chỉ ở giai đoạn cuối không giúp (SD4.2a; SD4.6 search@360 −0.25).
  - ActiveWR được quyết định trước bước 475: trong 190 trận F sống tới bước 475, 174 trận chưa có kill nào.

---

## 4. Research gap

| Gap | Tài liệu hiện có (bằng chứng toàn văn) | Repo lấp bằng gì | Kết quả |
|---|---|---|---|
| **G1.** Thước đo không tách thắng chủ động khỏi thắng thụ động | Skynet, Kartal, Huynh, Weil gọi tên thưởng giả nhưng vẫn báo WR hoặc điểm giải; Weil kết luận "WR không đủ" mà không đề xuất thước đo | ActiveWR, ghi công kill theo chủ quả bom, tách thắng thụ động / nhờ vòng / hòa / tự nổ | Agent "PASS" WR 52.5% có 0 kill / 40 trận; SD2 WR +10.75 nhưng ActiveWR không đổi; luật chặn đường WR +3.75 nhưng ActiveWR −1.75 |
| **G2.** Search + RL mới chỉ đo trên bản mini, theo tốc độ học | Kartal 2018/2019: bàn 8×8, vài seed, đường cong reward; hai hướng tương lai chưa thử | So sánh có đối chứng ba cách dùng cùng một search, có seed test riêng và CI | Nhãn thưa −2.75; search lúc chơi +16.5; chưng cất dày +16.25, rồi +35.0, rồi +10.5 |
| **G3.** Chưa có expert iteration cho Bomberman; ExIt trong Pommerman làm agent thụ động | ExIt, AlphaZero (cờ bàn, search dùng mạng); Weil 2023 (Pommerman, agent trôi về thụ động) | Search viết tay làm thầy, nhãn chỉ +1 cho kill bằng bom của learner, chưng cất trạng thái có thông tin + KL neo, lặp ba vòng | S3 = 92.25, kém thầy 1.75 (CI chứa 0), nhanh ~70×, không thụ động (thắng thụ động không tăng) |
| **G4.** Thiếu báo cáo âm có kiểm soát cho các kỹ thuật RL chuẩn trong game kiểu này | Bài báo thường chỉ báo cấu hình thành công; Henderson và Agarwal chỉ ra hệ quả | Hơn chục thí nghiệm một biến, luật chốt trước, seed test dùng một lần | Đa số FAIL; ghi đầy đủ (mục 6.5) |
| **G5.** Chưa rõ hòa giữa agent mạnh là do agent hay do luật | Osogami, Perez-Liebana, Skynet, AlphaZero đều thấy hòa cao nhưng không phân tích nguồn gốc | Probe có luật chốt trước: search với mô hình đối thủ đúng tìm kế hoạch giết trong trận hòa; probe đổi lịch vòng | 15.9% trận hòa có kế hoạch giết (0.19% trạng thái); vòng sớm hơn chỉ biến hòa thành cùng chết |
| **G6.** Mô hình đối thủ trong search: đúng loại quan trọng hơn chính xác | Weil, PSRO (JPC), DOMAC | Đo bằng probe có đối chứng (SDM, SDM2) | Oracle +14.0 [+3, +25] khi đấu G; luôn đoán bằng CNN thì −33.0 [−41, −25] khi đấu RuleBot |

**Độ mới cần nói rõ trong paper:** từng thành phần (policy gradient, mask, search, chưng cất, group advantage) đều đã có trong tài liệu. Đóng góp nằm ở:
- cách ghép các thành phần cho Bomberman với ngân sách CPU;
- thước đo ActiveWR;
- bộ bằng chứng có kiểm soát, kể cả kết quả âm và phân tích hòa.

---

## 5. Câu hỏi nghiên cứu và giải pháp

### 5.1 Câu hỏi nghiên cứu

- **RQ1.** WR đánh giá sai agent Bomberman đến mức nào, và ActiveWR sửa được gì?
- **RQ2.** Với cùng một search, cách dùng nào chuyển được năng lực của search vào CNN: loss phụ thưa trong RL, search lúc chơi, hay chưng cất dày?
- **RQ3.** Expert iteration với chuyên gia không dùng mạng đưa CNN thuần tới gần trần search tới đâu, với chi phí lúc chơi bao nhiêu?
- **RQ4.** Vì sao các tinh chỉnh RL (GAE, PPO, shaping, curriculum, group RL, KL neo) không vượt mốc khi xuất phát từ policy đã tốt?
- **RQ5.** Hòa giữa các agent mạnh do cấu trúc game hay do agent bỏ lỡ cơ hội?

### 5.2 Giải pháp của repo (pipeline)

```
Phase C: policy gradient CNN + mask an toàn + teacher phụ  →  B
   │ RL tiếp + advantage theo nhóm ở trạng thái quyết định (SD7)
   ▼
G (CNN, ~48 ActiveWR)
   │ + kill-search 1/2/3 nước lúc chơi (chuyên gia)
   ▼
G + search3 (~94, chậm)
   │ thu nhãn dày ở trạng thái có thông tin → chưng cất (CE mục tiêu mềm + KL neo)
   ▼
S2 (83.0)  →  S2 + search3 làm thầy  →  S3 (92.25, CNN thuần, nhanh ~70×)
```

### 5.3 Công thức chính (để chép vào Method)

**Policy có mask.** Xem `train_actor_critic.py`.

```latex
\pi_\theta(a\mid s) = \frac{m_a(s)\,\exp z_a(s)}{\sum_{b} m_b(s)\,\exp z_b(s)},\qquad m_a(s)\in\{0,1\}
```

Cài đặt bằng `masked_fill(~mask, -1e9)`. Theo Huang & Ontañón 2022, đây vẫn là policy gradient hợp lệ.

**Mục tiêu RL mặc định.** Đây là REINFORCE với baseline học được (Sutton & Barto, ch. 13), không bootstrap. Chỉ biến thể GAE mới là actor-critic đúng nghĩa.

```latex
G_t=\sum_{k\ge 0}\gamma^k r_{t+k},\quad
\mathcal{L}=-\frac1T\sum_t \log\pi_\theta(a_t\mid s_t)\,\big(G_t-V_\phi(s_t)\big)
+c_v\,(G_t-V_\phi(s_t))^2 - c_e\,\mathcal{H}(\pi_\theta(\cdot\mid s_t)) + \sum_k c_k\mathcal{L}_k^{\text{teacher}}
```

Hệ số mặc định:
- c_v = 0.5;
- c_e = 0.0075, giảm tuyến tính còn 20%;
- lr 2.5e-5;
- teacher: escape 0.5, combat 0.9, navigation 0.4, ring 0.6;
- nhân 1.75 từ bước 350.

**Kill-search.** Xem `kill_search.py`.
- Mỗi nước đầu a (và với search2/3 là chuỗi 2 hoặc 3 nước) được mô phỏng 12 tick.
- Đối thủ đi theo RuleBot (`OPPONENT_MODEL`); learner sau chuỗi nước chỉ né.
- Điểm +1 nếu đối thủ chết vì bom của learner và learner sống; −1 nếu learner chết; 0 nếu không có gì.
- Chỉ gọi search khi đối thủ trong tầm REACH ≤ 6.
- Chọn nước có điểm cao nhất; khi đồng hạng, dùng nước của CNN.

**Advantage theo nhóm (SD7, kiểu GRPO).** Chơi lại K lần từ trạng thái quyết định s, mỗi lần có kết quả o_k:

```latex
A_k = \frac{o_k-\bar o}{\operatorname{std}(o)+\epsilon},\qquad
\mathcal{L}_{\text{group}}=-\frac1K\sum_k A_k\log\pi_\theta(a_k\mid s)\;\;(+\;\beta\,\mathrm{KL}(\pi_{\text{ref}}\Vert\pi_\theta))
```

**Chưng cất dày.** Xem `run_sd53_pilot.py`, `run_sd9_distill.py`.
- B(s) là tập các nước hợp lệ có điểm search cao nhất.
- Mục tiêu mềm: q(a|s) = 1/|B(s)| với a ∈ B(s), bằng 0 với nước khác.

```latex
\mathcal{L}_{\text{distill}}=\mathbb{E}_{s\in\mathcal{D}_{\text{dec}}}\big[-\textstyle\sum_a q(a\mid s)\log\pi_\theta(a\mid s)\big]
+\lambda\,\mathbb{E}_{s\in\mathcal{D}_{\text{anchor}}}\,\mathrm{KL}\big(\pi_{\text{ref}}(\cdot\mid s)\,\Vert\,\pi_\theta(\cdot\mid s)\big)
```

Thiết lập:
- D_dec là các trạng thái mà các nước có điểm khác nhau;
- D_anchor là khoảng 5% trạng thái thường;
- Adam lr 1e-4, batch 256;
- chọn epoch bằng validation seed 21000.

**Đánh giá.**
- d_b là hiệu ActiveWR của hai model trên bàn b, gộp 2 góc.
- Δ̂ là trung bình của d_b.
- CI95 percentile bằng bootstrap lấy lại bàn, 10 000 lần.
- Gate: Δ̂ ≥ +5pp, cận dưới CI95 > 0, tự nổ không tăng quá 1pp.

---

## 6. Phát hiện đã có bằng chứng (dùng làm claim)

### 6.1 Đường đi tới agent tốt nhất (đấu RuleBot)

| Mốc | Agent | ActiveWR | Seed test | Nguồn |
|---|---|---|---|---|
| Xuất phát | B | 40.0 (lịch E); 32.75 khi chưa có sudden death | 60000; 32000 | crosscheck E3; control confirmation |
| RL tiếp | F | 44.25 | 43000 | SD5.3 |
| RL + advantage theo nhóm | G | 48.25–51.0 | 45000, 46000 | SD7.5, SD7.8 |
| Search 1 nước lúc chơi | G + search1 | 64.75 | 46000 | SD7.8 |
| Search 2 nước | G + search2 | 82.0 | 47000 | SD7.9c |
| Search 3 nước (trần) | G + search3 | 94.0 | 49000 | SD8B |
| Chưng cất vòng 1 | S (F + search1) | 60.5 | 43000 | SD5.3 |
| Chưng cất vòng 2 | S2 (G + search2) | 83.0 | 48000 | SD8A |
| **Chưng cất vòng 3** | **S3 (S2 + search3)** | **92.25** | 57000 | SD9 |

Các mốc đo trên seed test khác nhau, nên bảng này chỉ mang tính mô tả. Các so sánh ghép cặp cùng seed ở mục 6.3 và 6.4 mới là bằng chứng.

### 6.2 RQ1: WR đánh giá sai

- Checkpoint FFA 4 agent "PASS" theo gate cũ: WR 52.5% nhưng **0 lần hạ gục bằng bom của mình** trong 40 trận. Thắng chủ yếu vì RuleBot tự nổ.
- SD2 + biên mask k = 15: WR +10.75 (test 41000), số trận chết vì vòng 116 → 76/400, nhưng ActiveWR không đổi.
- Luật đứng giữ ô cửa: WR +3.75 [1.75, 6.0], nhưng ActiveWR −1.75 [−4.25, 0.75]. Thắng thụ động tăng 9.5 → 15.25 (SD6.1b).
- Ghi công theo chủ quả bom có ý nghĩa: luật chặn đường đổi 16 trận F vốn thắng chủ động thành thắng thụ động.
- Đánh giá chỉ với RuleBot đánh giá quá cao agent có search:
  - G + search3 đấu RuleBot đạt 94 (FFA cũng 86–94);
  - đấu CNN đông cứng chỉ đạt ActiveWR 8.5–20.5 và WR 19–44, hòa 50–70% (PDE).

### 6.3 RQ2: ba cách dùng cùng một search

| Cách dùng | Thí nghiệm | Kết quả (ActiveWR, CI95) | Seed |
|---|---|---|---|
| Nhãn search thưa làm loss phụ trong RL | SD5.1 | search − F −2.75 [−9.75, +4.0] | test 42000 |
| Search lúc chơi | SD7.8 | G 48.25 → 64.75, +16.5 [+12.5, +20.5] | test 46000 |
| Chưng cất dày | SD5.3 | S − F +16.25 [+9.75, +22.5] | test 43000 |

Giải thích có bằng chứng:
- Nhãn thưa chỉ khoảng 0.9 nhãn/episode và học một lần (INSIGHTS §4).
- Chưng cất dày dùng khoảng 6 200 trạng thái quyết định, có cả ví dụ ngược và 8 epoch:
  - P(nước tốt nhất) trên held-out tăng 0.52 → 0.91;
  - nhóm "đứng ở ô cửa" tăng 0.28 → 0.93;
  - P(BOMB) ở trạng thái thường gần như không đổi (0.088 → 0.095).

### 6.4 RQ3: expert iteration với chuyên gia không dùng mạng

- **Độ sâu search có lợi tăng dần.**
  - Search1 có trần chung khoảng 65% bất kể nền CNN (F, G, S).
  - Search2 phá trần đó: 61.0 → 82.0, +21.0 [+17.0, +25.25], hòa 18.0 → 5.75 (test 47000).
  - Search3: 85.75 → 94.0, +8.25 [+5.25, +11.5] (test 49000).
  - Đòn bẩy chính là **tạo** cơ hội giết (đi một bước rồi đặt bom), không phải kết liễu tốt hơn.
- **Chưng cất lặp.**
  - S3 − S2 = +10.5 [+6.5, +14.5] (test 57000).
  - S3 − (G + search3) = −1.75 [−5.0, +1.5]: CI chứa 0.
  - Phần tăng đến từ thắng chủ động: thắng thụ động −1.5 (CI chứa 0), hòa −6.25, trận ngắn hơn (131 so với 202 bước).
  - Tắt sudden death cho kết quả gần như y hệt (+10.75).
- **Khoảng cách học trò–thầy** qua ba vòng (S − (F+s1) seed 43000, S2 − (G+s2) seed 48000, S3 − (G+s3) seed 57000): −3.75 [−10.0, 2.25] → −3.5 [−8.25, 1.25] → −1.75 [−5.0, 1.5]. Số −45.75 ghi trước đây không có nguồn, đã sửa ngày 25/10/2026. Xu hướng này khớp với dự đoán của ExIt rằng khoảng cách thu hẹp dần.
- **Chi phí:** S3 chỉ chạy mạng một lần mỗi bước: 1.85 ms/quyết định so với 213 ms của G + search3 (khoảng 115 lần; 745 quyết định, 6 trận validation 21000–21002, 1 luồng CPU, torch 2.5.1). Số "70 lần" ghi trước đây không có phép đo gốc. Search tốn 20–30 giây mỗi trận.
- **Nền mạnh hơn không làm search mạnh hơn:** S2 + search3 so với G + search3 đấu RuleBot −5.0 [−11, +1] (SD9.0, validation). Đây là bằng chứng cho trần của search không dùng mạng.

### 6.5 RQ4: kết quả âm có kiểm soát

| Giả thuyết | Kết quả | Nguồn |
|---|---|---|
| Train RL thêm 3000 ep từ F | control −6.5 [−12.75, 0.0] trên test | SD5.1 |
| Hạ nhiệt độ lấy mẫu / argmax | argmax −9.25 [−16.0, −2.5]; τ 0.25 +1.75 (CI chứa 0) | SD6.0 |
| Teacher giai đoạn cuối | đều kém B | SD4.2a |
| Search chỉ giai đoạn cuối | search@360 −0.25 | SD4.6 |
| Luật chặn đường lúc chơi | −1.75 [−4.25, 0.75] | SD6.1b |
| Group RL tín hiệu yếu (K 8, coef 1) | +0.25 [−5.75, 6.25] | SD7 |
| Group RL tín hiệu mạnh (K 16, coef 4), 3000 ep | tỉ lệ giết +7.12 [4.86, 9.72] nhưng ActiveWR full − control +0.75 [−5.75, +7.25] | SD7.4, SD7.5 |
| Đẩy BOMB ở trạng thái hẹp | P(nước giết) 0.72 → 0.97 nhưng kills/trận không đổi, bom/trận +20%, ActiveWR −0.5 | SD7.6 |
| Phạt KL về G (mọi β) | không tách được "đặt bom đúng lúc" khỏi "đặt bom bừa" | SD7.7 |
| RL tiếp từ model chưng cất (Phase D 2000 ep) | ActiveWR 83.75 → 65.0 / 57.5; chọn đúng nước search 97.4 → 75.6 / 73.1 | D-S2 |
| KL neo toàn cục β = 1 | ActiveWR 84 → 68 (control 76); KL trên trạng thái quyết định lớn gấp ~10 lần trạng thái thường | S2-KL |
| Vòng sudden death sớm hơn | macro CNN ActiveWR −1.42…−0.17; 80–87% trận hòa thành cùng chết | SD9.3 |
| Search luôn đoán đối thủ bằng CNN | đấu RuleBot −33.0 [−41, −25] | SDM2 |

Các bài học có bằng chứng rút ra từ bảng:
- **Tăng chỉ số cơ chế cục bộ không đủ để kết luận về ActiveWR.** Mọi probe cơ chế phải báo kèm kills/trận và bom/trận trên trận của chính model đó (SD7.6).
- **CNN học được điều kiện "chỉ BOMB mới giết" khi nhãn có cả ví dụ ngược** (chưng cất), nhưng không học được từ tín hiệu RL một phía (SD7.7).
- **Train RL tiếp sau chưng cất xóa kỹ năng.** KL toàn cục bị trạng thái thường lấn át; muốn giữ kỹ năng phải neo đúng trạng thái quyết định.

### 6.6 RQ5: hòa cấu trúc

- Đấu CNN, S3 vẫn hòa 53–64%, phần lớn do hết 500 bước. S3 − S2 trên ba đối thủ CNN: ActiveWR +3.33 [+1.25, +5.50], hòa −4.08 [−8.17, −0.17] (SD9.1).
- SD9.2 xét 195 trận hòa của S3. Khi search3 mô phỏng đối thủ bằng chính CNN của nó:
  - kế hoạch giết chỉ có ở 0.19% trạng thái và 15.9% [9.9, 22.7] trận hòa;
  - với giả định RuleBot thì con số là 2.98% trạng thái và 90.3% trận hòa.

  Đối chứng: 20/20 trận thắng chủ động đều có kế hoạch giết. Vậy bẫy kiểu RuleBot là ảo khi đấu CNN.
- Thời gian trận: đấu CNN, khoảng 60% trận đi đủ 500 bước. Đấu RuleBot, hòa dưới 1% và trung vị khoảng 105 bước (SD9.1-diag).
- **Diễn giải:** hai bên dùng chung mask an toàn nên không ai bước vào vùng nổ. CNN né theo nhiều cách và hầu như luôn còn đường thoát, nên gần như không tồn tại cú giết chắc chắn.

### 6.7 Chuyển sang bàn 11×11 (S3-11)

- Baseline (validation): S3 nội suy 50%, S3 + search3 98%, thầy − S3 +48 [38, 58].
- Chưng cất trên 11×11: 200 trận thu, 746 nhãn quyết định, 1310 nhãn neo, 16 epoch.
- Test 820000–820199 (400 trận/model): ActiveWR 46.25 → 87.25, +41.0 [34.5, 47.25], tự nổ +0.5pp. Gate PASS.
- Đấu pool CNN (S3/S2/G, 2400 trận, seed 830000–830199):
  - macro ActiveWR 8.58 so với 4.75, +3.83 [1.92, 5.83]: **chưa đạt gate**;
  - hòa 64.33 so với 63.08.
- **Kết luận:** chưng cất lại trên phân phối mới hiệu quả khi đấu RuleBot, nhưng không giải được hòa cấu trúc khi đấu CNN. Kết quả này nhất quán với 6.6.

### 6.8 Phát hiện về quy trình (có thể đưa vào Experimental setup)

- **Dao động của một seed test.** Một seed test 400 trận dao động khoảng ±5pp cho cùng một cặp model. Ví dụ control SD5.1 − F: −6.5 (seed 42000), +3.5 (44000), +3.0 (45000); gộp 600 cặp bàn thì 0.0 [−3.75, +3.75] (KC1).
- **Chọn best bằng validation thiên lạc quan.** Chọn best trong nhiều điểm validation lạc quan vài pp: control SD5.1 validation 51.0, test 42.0.
- **Không so validation giữa các phiên bản torch.** Cùng F, cùng seed: 47.0 (torch 2.5.1) so với 43.75 (torch 2.14.1), vì policy lấy mẫu.
- **Policy phải lấy mẫu.** Argmax làm policy tất định bị kẹt trước RuleBot tất định (−9.25).

---

## 7. Phát hiện tiềm năng (giả thuyết, chưa kiểm)

Mỗi giả thuyết ghi: căn cứ, dự đoán, và cách kiểm (giữ giao thức của repo: seed test mới, gate chốt trước).

| # | Giả thuyết | Căn cứ tài liệu | Dự đoán | Cách kiểm | Chi phí |
|---|---|---|---|---|---|
| P1 | Gộp dữ liệu chưng cất qua các vòng (D_SD8A ∪ D_SD9) và lấy ít trạng thái tương quan hơn mỗi ván sẽ thu hẹp khoảng 1.75 còn lại | ExIt (online > batch, một trạng thái mỗi ván), DAgger | S3′ ≥ S3, tiến sát search3 | train lại từ S2 trên dữ liệu gộp, test seed mới, so ghép cặp với S3 | thấp |
| P2 | Search dùng prior/value của S3 (MCTS kiểu PUCT) phá trần ~94% | ExIt, AlphaZero: bước cải thiện chuyên gia bằng mạng là nguồn tăng chính | thầy mới > G + search3; vòng chưng cất mới vượt gate | viết search mới, so với search3 trên validation trước | cao |
| P3 | Kịch bản bi quan thay mô hình RuleBot giữ được sức mạnh khi đấu cả RuleBot lẫn CNN | Osogami 2019 (độ sâu 1 + bi quan thắng RL 87.1%; ablation mức bi quan 36.9 → 77.8% khi đấu dypm) | không mất −33 như SDM2; tăng ActiveWR khi đấu CNN | thay `OPPONENT_MODEL`, probe 2 loại đối thủ | trung bình |
| P4 | Mô hình đối thủ học được hoặc hỗn hợp {RuleBot, F, G, S2, S3} giảm chi phí SD10 (~53 giờ) và giảm overfit | DOMAC, PSRO (giảm JPC 29–57%), Weil | gần bằng oracle với chi phí thấp | train bộ dự đoán hành động đối thủ, probe như SDM | trung bình |
| P5 | RL tiếp với neo về **thầy search** (không phải về model cũ), λ giảm dần, entropy 0, clip nhỏ sẽ giữ được kỹ năng | Kickstarting (λ 1 → 0 vượt thầy), Meisheri (entropy 0, clip 0.01), PPO (β thích nghi) | D-S2 không còn rơi; có thể tăng khi đấu CNN | Phase D từ S3 với loss CE tới nhãn search trên quỹ đạo học trò | trung bình |
| P6 | Hòa khi đấu CNN chủ yếu do luật; đổi luật (vòng cuối không lấp kín, trận dài hơn) mới tạo thêm thắng chủ động | SD9.2, SD9.3; Perez-Liebana đề xuất collapsing boards | ActiveWR khi đấu CNN tăng, hòa giảm mà không thành cùng chết | probe lịch mới trên validation, luật chọn chốt trước | trung bình |
| P7 | Hệ số teacher giảm dần về 0 (thay vì cố định) nâng trần RL | Bansal (thưởng dày phải giảm), Kickstarting | G′ > G | một biến, một lượt 3000 ep | trung bình |
| P8 | Kết quả chính ổn định qua nhiều train seed | Henderson, Agarwal | S3 qua ≥ 3 seed vẫn trong khoảng ±3pp | chạy lại SD9 với 3 train seed, báo IQM và P(S3 > S2) | trung bình–cao |
| P9 | Shaping dạng thế (Φ = −khoảng cách tới đối thủ, hoặc Φ = giá trị search một nước) không đổi chính sách tối ưu và có thể tăng tốc học | Ng 1999 | học nhanh hơn, không sinh vòng lặp farm | thay thưởng phá tường/item bằng thế, một biến | thấp |
| P10 | CNN dùng shield "vùng thắng trước mọi nước đối thủ trong k bước" sẽ giảm tự nổ còn lại mà không tăng hòa | Alshiekh 2018 | tự nổ ↓, hòa ≈ | cài shield mới, đánh giá mô tả | trung bình |

---

## 8. Định vị so với công trình liên quan (bảng cho Related Work)

| Công trình | Môi trường | Học | Search | Thước đo | Khác biệt của repo |
|---|---|---|---|---|---|
| Resnick 2018 (Pommerman) | Pommerman 11×11 | DQN, PPO, DAgger | — | WR | ActiveWR; repo đạt 92% khi đấu RuleBot |
| Gao 2019 (Skynet, Hard Exploration) | Pommerman Team | PPO + ActionFilter | — | WR | mask đầy đủ hơn + search + chưng cất |
| Osogami 2019 | Pommerman | — | độ sâu 1 + kịch bản bi quan | WR / hòa | search của repo tấn công (nhãn chỉ cho kill của mình); chưng cất vào CNN |
| Perez-Liebana 2019 | Pommerman (Java) | — | MCTS, RHEA, mô hình đối thủ ngẫu nhiên | thắng/hòa/thua ± SE | mô hình đối thủ RuleBot; phân tích hòa |
| Kartal 2018/2019 | Pommerman mini 8×8 | A3C + MCTS demonstrator | MCTS nông | đường cong reward | 9×9 đầy đủ, so ba cách dùng search có seed test và CI |
| Weil 2023 | Pommerman | AlphaZero-style ExIt | MCTS SP/TP | WR | ExIt không trôi về thụ động nhờ nhãn kill-only; ActiveWR |
| Huynh 2024 | Pommerman Team | PPO + LSTM, curriculum, PBT | — | WR, số địch bị hạ | ngân sách một laptop; vẫn đạt mức cao khi đấu RuleBot |
| Meisheri 2019 | Pommerman Team | IL → PPO | — | thắng/thua/hòa | chưng cất từ search thay vì bắt chước bot |
| Anthony 2017 (ExIt) | Hex | ExIt | MCTS + mạng | Elo | search không dùng mạng (một giới hạn; P2) |
| Goulart 2019 | Bomberman (Unity) | IL → PPO | — | (chưa đọc toàn văn) | — |

---

## 9. Giới hạn và threats to validity (bắt buộc đưa vào paper)

1. **Một train seed mỗi nhánh.** CI đo biến thiên do bàn đánh giá, không đo biến thiên giữa các lần train (Henderson 2018 cho thấy biến thiên này có thể lớn).
2. **Đối thủ chuẩn là RuleBot**, và search giả định RuleBot. S3 mạnh nhất khi đấu RuleBot; khi đấu CNN chỉ nhỉnh hơn S2 chút ít.
3. **Các mốc của đường đi đo trên seed test khác nhau.** Một seed 400 trận dao động khoảng ±5pp.
4. **Tái lập:** policy lấy mẫu, nên số liệu tái lập bit-to-bit chỉ trên cùng phiên bản torch.
5. **Phạm vi:** chủ yếu 1v1 trên bàn 9×9. FFA và bàn 11×11 mới đo mô tả hoặc một run.
6. **Seed test S3-11 và pool CNN** là một lần chạy. Kết quả pool CNN chưa đạt gate.
7. **Tài liệu:** Goulart 2019 chỉ đọc tóm tắt. Sutton & Barto chỉ đọc chương 13.
8. **Một số nhánh do người khác chạy** (bạn của người dùng: F, S, S2, D-S2, S2-KL) trên máy khác. Không so validation giữa các phiên bản torch.

---

## 10. Gợi ý cấu trúc paper (8 trang kiểu hội nghị game AI)

1. **Introduction:** vấn đề thắng thụ động; RL bão hòa; search mạnh nhưng chậm; đóng góp C1–C6.
2. **Related work:** bảng mục 8; ba nhóm (Pommerman RL, search trong Pommerman, ExIt/chưng cất).
3. **Problem setting:** Markov game, luật, observation 13 kênh, ActiveWR (mục 2, 5.3).
4. **Method:** mask, policy gradient + teacher, kill-search, group advantage, chưng cất dày, vòng ExIt (mục 5.3).
5. **Evaluation protocol:** seed một lần, paired bootstrap theo bàn, gate chốt trước (mục 5.3, 6.8).
6. **Results:**
   - RQ1 (6.2);
   - RQ2 (6.3, có hình bar ba cách dùng search);
   - RQ3 (6.4, có hình đường đi ActiveWR theo vòng);
   - RQ4 (6.5, bảng kết quả âm);
   - RQ5 (6.6);
   - chuyển map (6.7).
7. **Discussion:** liên hệ insight I4–I9; phát hiện tiềm năng P1–P10.
8. **Limitations** (mục 9) và **Conclusion**.

**Hình và bảng nên có:**
- **Hình 1:** pipeline (mục 5.2).
- **Hình 2:** ActiveWR qua các mốc, có CI, ghi rõ khác seed.
- **Hình 3:** WR so với ActiveWR cho các can thiệp "tăng WR giả".
- **Bảng 1:** ba cách dùng search.
- **Bảng 2:** kết quả âm.
- **Bảng 3:** hòa khi đấu CNN và SD9.2.

**Venue phù hợp về chủ đề:** hội nghị và workshop game AI (ví dụ IEEE CoG, AIIDE, ICEC). Cần kiểm hạn nộp và trang giới hạn trên trang chính thức trước khi chọn.

---

## 11. Tài liệu tham khảo (đã đọc toàn văn trừ khi ghi khác)

**Pommerman / Bomberman**
1. Resnick et al. (2018). Pommerman: A Multi-Agent Playground. arXiv:1809.07124.
2. Resnick, Raileanu, Kapoor, Peysakhovich, Cho, Bruna (2018). Backplay: "Man muss immer umkehren". arXiv:1807.06919.
3. Peng, Pang, Yuan, Gao (2018). Continual Match Based Training in Pommerman. arXiv:1812.07297.
4. Gao, Hernandez-Leal, Kartal, Taylor (2019). Skynet: A Top Deep RL Agent in the Inaugural Pommerman Team Competition. RLDM 2019; arXiv:1905.01360.
5. Gao, Kartal, Hernandez-Leal, Taylor (2019). On Hard Exploration for Reinforcement Learning: a Case Study in Pommerman. AIIDE 2019; arXiv:1907.11788.
6. Osogami, Takahashi (2019). Real-time tree search with pessimistic scenarios: Winning the NeurIPS 2018 Pommerman Competition. ACML 2019, PMLR 101; arXiv:1902.10870.
7. Perez-Liebana et al. (2019). Analysis of Statistical Forward Planning Methods in Pommerman. AIIDE 2019, 15(1), 66–72.
8. Weil, Czech, Meuser, Kersting (2023). Know your Enemy: Investigating Monte-Carlo Tree Search with Opponent Models in Pommerman. ALA @ AAMAS 2023; arXiv:2305.13206.
9. Huynh, Cao, Wu (2024). Multi-Agent Training for Pommerman: Curriculum Learning and Population-based Self-Play Approach. arXiv:2407.00662.
10. Meisheri, Shelke, Verma, Khadilkar (2019). Accelerating Training in Pommerman with Imitation and Reinforcement Learning. arXiv:1911.04947.
11. Goulart, Paes, Clua (2019). Learning How to Play Bomberman with Deep Reinforcement and Imitation Learning. ICEC-JCSG 2019, LNCS 11863, 121–133. *(Chỉ đọc tóm tắt.)*
12. Sun, Chen, Zhang, Ma, Zhang (2022). Decision-making with Speculative Opponent Models (DOMAC). arXiv:2211.11940.
13. Helfenstein, Czech, Blüml, Eisel, Kersting (2024). Checkmating One, by Using Many (M2CTS). arXiv:2401.16852.

**Search và học**
14. Kartal, Hernandez-Leal, Taylor (2018). Using Monte Carlo Tree Search as a Demonstrator within Asynchronous Deep RL. arXiv:1812.00045.
15. Kartal, Hernandez-Leal, Gao, Taylor (2019). Safer Deep RL with Shallow MCTS: A Case Study in Pommerman. arXiv:1904.05759.
16. Kartal, Hernandez-Leal, Taylor (2019). Action Guidance with MCTS for Deep Reinforcement Learning. AIIDE 2019; arXiv:1907.11703.
17. Anthony, Tian, Barber (2017). Thinking Fast and Slow with Deep Learning and Tree Search (Expert Iteration). NIPS 2017; arXiv:1705.08439.
18. Silver et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. arXiv:1712.01815.
19. Ross, Gordon, Bagnell (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS 2011; arXiv:1011.0686.
20. Rusu et al. (2016). Policy Distillation. ICLR 2016; arXiv:1511.06295.
21. Schmitt et al. (2018). Kickstarting Deep Reinforcement Learning. arXiv:1803.03835.

**Tối ưu policy, shaping, an toàn, đa tác tử**
22. Sutton, Barto (2018). Reinforcement Learning: An Introduction, 2nd ed., ch. 13. MIT Press. *(Chỉ đọc chương 13.)*
23. Williams (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning 8, 229–256.
24. Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. ICML 2016; arXiv:1602.01783.
25. Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016; arXiv:1506.02438.
26. Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
27. Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO). arXiv:2402.03300.
28. Ng, Harada, Russell (1999). Policy Invariance Under Reward Transformations. ICML 1999, 278–287.
29. Alshiekh et al. (2018). Safe Reinforcement Learning via Shielding. AAAI 2018; arXiv:1708.08611.
30. Huang, Ontañón (2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. FLAIRS-35; arXiv:2006.14171.
31. Bansal et al. (2018). Emergent Complexity via Multi-Agent Competition. ICLR 2018; arXiv:1710.03748.
32. Lanctot et al. (2017). A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning (PSRO). NeurIPS 2017; arXiv:1711.00832.

**Đánh giá**
33. Henderson et al. (2018). Deep Reinforcement Learning that Matters. AAAI 2018; arXiv:1709.06560.
34. Agarwal, Schwarzer, Castro, Courville, Bellemare (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021; arXiv:2108.13264.
35. Efron (1979). Bootstrap methods: another look at the jackknife. The Annals of Statistics 7(1), 1–26. doi:10.1214/aos/1176344552.

Trước khi nộp, kiểm lại tác giả, năm và venue của từng tài liệu trên trang gốc. Danh sách này lấy từ trang arXiv/venue đã mở khi viết báo cáo, nhưng chưa qua bước kiểm định dạng citation.
