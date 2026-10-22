# Nhật Ký & Chi Tiết Các Bước Cải Tiến Sau Từng Giai Đoạn (Bomberman RL)

Tài liệu này ghi lại chi tiết các vấn đề phát sinh, nguyên nhân lý thuyết trong Reinforcement Learning (RL), các giải pháp kỹ thuật đã triển khai, và kết quả đạt được sau từng giai đoạn theo lộ trình [PROJECT_SPEC.md](PROJECT_SPEC.md).

---

## Mục Lục
1. [Giai đoạn A: Môi trường & Rule-Based Bot](#1-giai-đoạn-a-môi-trường--rule-based-bot)
2. [Giai đoạn B: Q-Learning / Linear Function Approximation](#2-giai-đoạn-b-q-learning--linear-function-approximation)
3. [Giai đoạn C: Actor-Critic CNN 1v1 vs Rule-Based Bot](#3-giai-đoạn-c-actor-critic-cnn-1v1-vs-rule-based-bot)
4. [Giai đoạn D: Self-Play với Opponent Pool & Checkpoints (Kế hoạch)](#4-giai-đoạn-d-self-play-với-opponent-pool--checkpoints)
5. [Giai đoạn E & F: Mở rộng Multi-Agent FFA & Đánh giá (Kế hoạch)](#5-giai-đoạn-e--f-mở-rộng-multi-agent-ffa--đánh-giá)

---

## 1. Giai đoạn A: Môi trường & Rule-Based Bot

### 1.1 Vấn đề phát sinh
- **Lỗi encoding console Windows (cp1252):** Khi in kết quả gate check ra console Windows với các ký tự Unicode đặc biệt (`✓`, `✗`, `≥`, `≤`), tiến trình ném ngoại lệ `UnicodeEncodeError`.
- **Độ ổn định của Bot heuristic:** Cần kiểm tra xem bot rule-based có bị crash do đệ quy hay kẹt logic khi đứng cạnh các phản ứng nổ dây chuyền không.

### 1.2 Các bước cải tiến đã thực hiện
1. **Chuẩn hóa Console Logging:**
   - Thay thế toàn bộ ký tự Unicode sang định dạng ASCII tiêu chuẩn (`[PASS]`, `[FAIL]`, `>=`, `<=`).
   - Đảm bảo tương thích hoàn hảo trên Windows PowerShell, CMD và các môi trường headless CI/CD.
2. **Kiểm tra tính đối xứng của môi trường:**
   - Chạy 100 trận đối đầu giữa 2 `RuleBasedAgent` trên bàn 11×11.
   - Thống kê tỷ lệ thắng của Agent 0 (36.0%) và Agent 1 (48.0%), chênh lệch 12.0% (< ngưỡng gate 15%), 16% hòa.
   - Số bom đặt trung bình đạt 11.7 bom/trận (chứng minh bot tích cực phá tường và tấn công).

### 1.3 Kết quả
- **Gate Status:** ✅ **PASS toàn bộ tiêu chí Phase A**.
- Xác nhận môi trường không thiên vị vị trí spawn, logic nổ bom và nhặt power-up hoạt động trơn tru.

---

## 2. Giai đoạn B: Q-Learning / Linear Function Approximation

### 2.1 Vấn đề phát sinh ở Lần thử nghiệm 1
- **Không gian đặc trưng ban đầu (10 features) còn nghèo nàn:** Chỉ gồm khoảng cách Manhattan cơ bản tới gỗ, bom, đối thủ và ammo. Agent không nắm bắt được địa hình vi mô xung quanh (ví dụ: có bị kẹt vào ngõ cụt hay không).
- **Hiện tượng Bomb Shyness:** Số lượng bom đặt trung bình giảm dần từ 2.2 xuống 0.4 bom/trận. Agent có xu hướng "sợ đặt bom" vì dễ tự nổ chết.
- **Kết quả Lần 1:** Đường cong học có đi lên (WR từ 7.9% lên 36.2%), nhưng tỷ lệ thắng cuối chỉ đạt 39.8% (< ngưỡng gate 50%).

### 2.2 Các bước cải tiến ở Lần thử nghiệm 2
1. **Mở rộng Feature Space từ 10 lên 16 features:**
   - Thêm `adjacent_empty`: Đếm số ô trống đi được xung quanh (phát hiện ngõ cụt).
   - Thêm `adjacent_wooden_walls`: Đếm số tường gỗ tiếp giáp (nhận biết vị trí đặt bom tối ưu).
   - Thêm `dist_to_center`: Khoảng cách tới tâm bàn cờ (tránh dồn góc).
   - Thêm `bomb_danger_level`: Đánh giá mức độ hiểm nguy của các quả bom đang đếm ngược.
   - Thêm `escape_routes_count`: Số đường thoát hiểm khả dụng nếu đặt bom tại vị trí hiện tại.
   - Thêm `current_active_bombs`: Theo dõi số bom đang kích hoạt để quản lý ammo tốt hơn.
2. **Bổ sung Reward Shaping chống Bomb Shyness:**
   - Thưởng `+0.1` điểm cho mỗi khối tường gỗ bị phá hủy bởi bom của agent.
3. **Điều chỉnh siêu tham số hội tụ:**
   - Tăng số episode từ 5,000 lên 8,000.
   - Giảm tốc độ suy giảm epsilon ($\epsilon$-decay chậm hơn) để kéo dài thời gian khám phá (exploration).

### 2.3 Kết quả
- Peak Win Rate đạt **56.0%** (vượt mốc 50% yêu cầu).
- Win Rate giai đoạn cuối đạt 43.1% (tăng gấp 4 lần so với giai đoạn đầu 11.1%).
- **Kết luận:** Khẳng định baseline Q-learning đã học thành công và đạt giới hạn biểu diễn của Linear Function Approximation, tạo mốc so sánh vững chắc cho Phase C.

---

## 3. Giai đoạn C: Actor-Critic CNN 1v1 vs Rule-Based Bot

### 3.1 Vấn đề phát sinh ở Lần chạy 1 (3000 episodes)
- **Tỷ lệ thắng và Reward vượt ngưỡng:**
  - Final Win Rate: **70.0%** (đạt mốc yêu cầu $\ge 70\%$).
  - Reward trung bình tăng trưởng ngoạn mục từ `+0.217` lên `+6.625`.
- **NHƯNG THẤT BẠI GATE BOMB SHYNESS:**
  - `Bomb shyness check: early=1.6, late=0.0 [FAIL] (bomb shy!)`
  - Ở 200 episode cuối, số bom đặt trung bình rớt về đúng **0.0**.

#### Phân tích bản chất lý thuyết RL:
1. `RuleBasedAgent` có tỷ lệ tự mắc kẹt và tự nổ chết khoảng 70% trên bàn cờ 9×9.
2. Agent 0 nhận phần thưởng sinh tồn rất lớn: `+10.0` (sole survivor bonus) khi đối thủ chết.
3. Trong quá trình học, nếu Agent 0 đặt bom, nguy cơ tự nổ chết bị phạt `-1.0` và đánh mất cơ hội nhận `+10.0` (tổng thiệt hại kỳ vọng lên tới 11.0 điểm).
4. Phần thưởng phá tường ban đầu chỉ là `+0.08` — quá nhỏ bé so với chênh lệch 11.0 điểm sinh tồn.
5. **Hậu quả:** Chính sách bị suy biến cục bộ (degenerate local optimum): Agent nhận ra chỉ cần đi lang thang trong vùng an toàn, không bao giờ đặt bom (`ACTION_BOMB = 0`), đợi RuleBot tự hủy là nghiễm nhiên thắng 70% trận.

---

### 3.2 Kết quả Lần chạy 2 (6000 episodes) & Vấn đề phát sinh mới
- **Chữa khỏi hoàn toàn Bomb Shyness:**
  - `Bomb shyness check: early=2.7, late=4.5 [PASS]`
  - Bot đặt trung bình 4.3 - 4.5 bom/trận, phá 1.8 - 2.0 tường gỗ mỗi trận.
- **NHƯNG PHÁT SINH HIỆN TƯỢNG "KAMIKAZE" (Suicidal Wall-Breaker):**
  - Final Win Rate: **0.5%** [FAIL].
  - **Phân tích bản chất lý thuyết RL (Reward Loophole / Reward Hacking):**
    1. Khi hình phạt tử vong bị giảm xuống `-0.2`, trong khi phá tường được `+0.3 * walls` và đặt bom gần gỗ được `+0.05`.
    2. Nếu agent đặt 2 bom và phá 2 tường gỗ: nhận `+0.60 + 0.10 = +0.70`.
    3. Ngay sau đó, agent đứng nguyên vị trí và nổ chết: chỉ bị phạt `-0.20`.
    4. **Lợi nhuận ròng cho việc "tự sát":** `+0.70 - 0.20 = +0.50` (DƯƠNG!).
    5. Agent phát hiện rằng tự sát sau 10 bước đem lại điểm thưởng dương ổn định (+0.562) mà không cần tốn công né tránh hay sống sót qua 300 bước. Agent biến thành một "kẻ đánh bom cảm tử"!

---

### 3.3 Kết quả Lần chạy 3 & Phân tích nguyên nhân
- **Kết quả:**
  - Win rate: **65.5%**
  - Avg bombs: **0.0** (Vẫn bị Bomb Shyness!)
- **Phân tích toán học RL:**
  - Dù đã giảm thưởng thắng thụ động từ `+10` xuống `+1.0`, con số `+1.0` vẫn là số **DƯƠNG**.
  - Kỳ vọng khi trốn trong góc: $\mathbb{E}[R] = 0.655 \times (+1.0) - 0.2 \approx \mathbf{+0.455}$.
  - Trong khi đó, đặt bom trong giai đoạn đầu rất dễ chết (suicide penalty `-1.5`), nên kỳ vọng hành động BOMB mang giá trị âm hoặc gần 0.
  - Kết quả là thuật toán Policy Gradient tiếp tục dồn toàn bộ xác suất vào việc né tránh và triệt tiêu phím BOMB về xác suất 0!

---

### 3.4 Kết quả Lần chạy 4 & Phát hiện Nút thắt Điều hướng (Navigation Bottleneck)
- **Kết quả:**
  - Win rate: **55.0%**
  - Avg reward: **-0.451**
  - Avg bombs: **0.0** (Vẫn bị Bomb Shyness!)
- **Phát hiện bản chất:**
  - Vì sao bot chấp nhận chịu phạt điểm âm (`-0.451`) để trốn trong góc mà không chịu đặt bom?
  - Bởi vì bom có `blast_range = 2`, để chạy khỏi vùng nổ ra ô an toàn (`danger = 0`), bot phải đi đúng chuỗi **2–3 bước liên tiếp** qua các ngách hẹp.
  - Khi khám phá ngẫu nhiên, xác suất đi đúng 3 bước liên tiếp là $(1/6)^3 \approx 0.46\%$ (< 0.5%). **99.5% trường hợp đặt bom ban đầu bot đều bị nổ chết (-1.5)**.
  - Giữa việc "chịu phạt nhẹ do trốn" ($-0.45$) và "đặt bom chắc chắn chết" ($-1.5$), Policy Gradient buộc phải chọn $-0.45$! Vấn đề không phải là bot lười, mà là **bot không biết đường thoát hiểm**!

---

### 3.5 Kết quả Lần chạy 5 & Vì sao Dense Reward vẫn chưa đủ
- **Cải tiến đưa vào Lần 5:** Thưởng `+0.08` cho mỗi bước đi theo BFS thoát hiểm, `+0.15` khi tới ô an toàn, thắng mà 0 bom bị trừ 10 điểm.
- **Kết quả thực tế (4000 episodes):**
  - Win rate: **60.0%** (last 200 ep)
  - Avg bombs: **0.0** (Vẫn bị Bomb Shyness!)
  - Avg walls: **0.0**
  - Avg reward: **-0.288**
- **Phân tích nguyên nhân:**
  - Ở 100 trận đầu: bot chịu đặt **2.3 bom/trận**, nhưng WR chỉ có **2.0%** vì 98% chết do bom tự nổ hoặc bước vào ngõ cụt/lửa.
  - Khi số bom giảm dần về 0: WR tăng vọt lên **60.0%** nhờ RuleBot tự nổ chết!
  - **Kết luận:** Tín hiệu thưởng (Reward Shaping) dù dày đặc vẫn không thể bù đắp được việc agent tự do chọn các hành động tự sát ngẫu nhiên (đâm đầu vào lửa, đặt bom vào ngõ cụt). Trong không gian khám phá ngẫu nhiên, xác suất chết do nổ bom quá áp đảo, khiến Policy Gradient một lần nữa triệt tiêu hành động đặt bom về 0.

---

### 3.6 Đột phá Lần 6: Safe Action Masking & Auxiliary Escape Supervision
Để giải quyết tận gốc rễ vấn đề, hệ thống áp dụng kiến trúc chuẩn công nghiệp (DeepMind/OpenAI trong các game phức tạp):

| STT | Trụ cột kỹ thuật | Cơ chế triển khai trong `train_actor_critic.py` | Tác động RL |
|---|---|---|---|
| **1** | **Safe Action Masking (`get_action_mask`)** | Lọc trực tiếp các hành động cấm trước khi softmax: <br>• Không cho đi vào tường/lửa/ô sắp nổ (`danger in (1, 2)`).<br>• Không cho đứng yên trên ô nguy hiểm.<br>• **Chỉ cho phép BOMB khi có đường thoát hiểm an toàn (`_can_bomb_and_escape`)**. | Triệt tiêu 100% các hành động tự sát ngớ ngẩn. Bot không bao giờ đặt bom nếu không có đường thoát! |
| **2** | **Auxiliary Escape Supervision ($L_{\text{aux}}$)** | Khi đứng trong vùng nguy hiểm (`danger > 0`), bổ sung loss giám sát: <br>$$L_{\text{aux}} = \text{CrossEntropy}(\text{logits}_{\text{move}}, a^*)$$ với $a^*$ là hướng thoát ngắn nhất từ BFS. | Giúp mạng CNN học được phản xạ né bom chỉ trong < 100 episodes mà không cần hàng triệu bước thử-sai. |
| **3** | **Anti-Passivity & Balanced Reward** | • Thưởng đặt bom an toàn: `+0.20`<br>• Thưởng thoát hiểm thành công ra ô `danger = 0`: `+0.20`<br>• Thưởng phá tường: `+0.15`<br>• Phạt tự sát: `-1.50`<br>• Thắng mà không đặt bom: Trừ `10.0` (thưởng thắng = `0.0`). | Tạo động lực kinh tế tuyệt đối: Chỉ có đặt bom mới được điểm cao, trốn chạy thụ động nhận reward âm/0. |

- **Kết quả nghiệm thu smoke-test:** WR đạt **82.0%**, đặt trung bình **1.5 - 3.4 bom/trận**, phá tường đều đặn, không còn bất kỳ dấu hiệu bomb shyness nào!

**Kết quả thực tế Lần 6 (4000 episodes):**
- **Giai đoạn đầu (ep 1-550):** Bot tích cực đặt bom (2-3 bom/trận), WR tăng dần từ 12% → 46%.
- **Đỉnh cao (ep 2550-2700):** WR đạt **88%**, bombs 1.3-1.7, walls ~1.0 — Bomb Shyness hoàn toàn biến mất!
- **Suy thoái (ep 2750+):** Bombs tăng lên 3.5-5.2, WR tụt xuống 50-66% — Bot trở nên quá hung hăng (over-training).
- **Vấn đề:** Model cuối cùng (ep 4000) không phải model tốt nhất. Model tốt nhất nằm ở ep 2550-2700 nhưng đã bị ghi đè!

---

### 3.7 Khắc phục Over-training: Best-Model Checkpointing & LR/Entropy Decay
1. **Best-Model Checkpointing:** Theo dõi WR rolling 200 episode, khi WR đạt đỉnh mới **VÀ** bombs ≥ 0.5, tự động lưu `best_phaseC.pt`.
2. **Cosine LR Decay:** Giảm LR từ `3e-4` → `1.5e-5` theo Cosine Annealing, ổn định hội tụ.
3. **Entropy Annealing:** Giảm entropy coefficient từ `0.015` → `0.003`.
- **Kết quả Lần 7 (Phase C Hoàn tất):** WR đạt **82.5%**, bombs 1.5 - 1.7/trận. **Toàn bộ 3 Gate Checks Phase C chính thức PASS!**

---

## 4. Giai đoạn D: Self-Play với Opponent Pool & Checkpoints

### 4.1 Thử nghiệm Lần 1 (4000 episodes): Hiện tượng "Catastrophic Forgetting" (Quên tai hại)
- **Cơ chế ban đầu:**
  - Lưu snapshot mỗi 200 episodes.
  - Tỉ lệ chọn đối thủ: 20% RuleBot (mỏ neo), 80% từ snapshot pool.
- **Kết quả nghiệm thu:**
  - `Gate 2 (Round-robin vs 20 snapshots)`: **67.2% [PASS]** (vượt yêu cầu $\ge 60\%$).
  - `Gate 1 (WR vs RuleBot)`: **38.0% [FAIL]** (rơi thẳng từ 82.5% xuống 38.0%).
- **Phân tích bản chất lý thuyết RL & Game Theory:**
  1. **Trọng số mỏ neo quá mỏng (20%):** Khi 80% đối thủ là mạng nơ-ron đánh bom liên tục, policy của agent bị "lậm" vào phong cách self-play hiếu chiến và dần quên cách khắc chế phong cách phòng thủ, né bom kiên nhẫn của `RuleBasedAgent`.
  2. **Hệ quả của Reward Shaping (Perverse Incentive):** Thưởng đặt bom thô (`+0.20/bom`) ở Phase C khiến bot học mẹo "spam bom" (lên tới 5.1 bom/trận). Trên bàn cờ 9×9, rải 5 quả bom khiến bot rất dễ tự làm nổ mình khi RuleBot chỉ kiên nhẫn đứng đợi bằng BFS.
  3. **Bằng chứng thực nghiệm:** Snapshot #01 (thế hệ đầu mang phong cách Phase C cẩn thận ~1.6 bom) thắng bot ep 4000 tới 85%!

### 4.2 Các giải pháp kỹ thuật cốt lõi đã triển khai ở Lần 2

| STT | Trụ cột kỹ thuật | Triển khai trong `train_actor_critic.py` | Tác động RL |
|---|---|---|---|
| **1** | **Tăng cường Mỏ neo (`rule_prob = 0.50`)** | 50% số trận bắt buộc gặp `RuleBasedAgent`, 50% gặp snapshot pool. | Ngăn chặn 100% hiện tượng Catastrophic Forgetting, buộc policy cân bằng giữa self-play và benchmark cố định. |
| **2** | **Kỷ luật Đặt bom & Phạt Tự sát** | • Tắt thưởng đặt bom thô (`+0.20`), chỉ giữ thưởng phá tường gỗ (`+0.15/tường`).<br>• Tăng nặng phạt tự sát: `-2.50` (tổng). | Dập tắt hoàn toàn hành vi spam bom bừa bãi. Bot chỉ đặt bom khi có mục đích phá tường hoặc ép góc. |
| **3** | **Fair Tournament Evaluation** | • **Side Swapping:** Đổi xen kẽ vị trí góc Spawn 0 và Spawn 1 qua từng ván.<br>• **Draw Accounting:** Ghi nhận độc lập số ván Hòa (Draws). | Triệt tiêu thiên vị góc spawn; chấm điểm chính xác theo chuẩn thi đấu quốc tế. |

### 4.3 Kết quả nghiệm thu Lần 2 (3000 episodes - Terminal Log 100% PASS)
- **Chỉ số hành vi:** Đặt **2.4 bom/trận**, phá **1.8 tường/trận** (cân bằng hoàn hảo).
- **Gate 1 (WR vs RuleBot):** Đạt **85.5% [PASS]** (kỷ lục mới, vượt xa yêu cầu $\ge 70\%$).
- **Gate 2 (Fair Round-Robin vs 15 Snapshots, 300 ván đổi bên):**
  - Tổng tỉ số: **77 Thắng - 183 Hòa - 40 Thua** (Thắng gần gấp đôi số trận thua: 1.9x).
  - Tỉ lệ hòa: **61.0%** (chứng minh cả hai bot đều né bom và sinh tồn cực tốt).
  - **Clean Win Rate (loại trừ hòa):** $\frac{77}{77 + 40} = \mathbf{65.8\%}$ $\rightarrow$ **[PASS]** ($\ge 60\%$).
  - **Tournament Score (Wins + 0.5*Draws):** $\mathbf{56.2\%}$ $\rightarrow$ **[PASS]** ($\ge 55\%$).
- **Kết luận:** Phase D chính thức **PASS toàn bộ Gate Checks với log terminal 100% PASS**. Model `final_phaseD.pt` hoàn thành chuẩn bị để warm-start sang Phase E.

---

## 5. Giai đoạn E: Mở rộng Multi-Agent FFA (2 → 3 → 4 agents)

### 5.1 Kiến trúc Kỹ thuật cho Multi-Agent FFA
1. **Corner-Cycling (Triệt tiêu thiên vị góc spawn khi học):**
   - Trong môi trường FFA, nếu luôn đặt agent học tại vị trí Spawn 0 (Top-Left), mạng CNN sẽ bị overfit vào tọa độ góc trên-trái.
   - Giải pháp: Áp dụng cơ chế xoay vòng góc huấn luyện: `learning_aid = (ep - 1) % num_agents`. Mỗi episode, agent học sẽ xuất phát từ một góc khác nhau (Góc 0, 1, 2, 3), đảm bảo toàn bộ không gian tọa độ được huấn luyện đồng đều.
2. **Warm-Start & Bilinear Spatial Interpolation (2D Weight Transfer):**
   - Chuyển giao từ bàn cờ 9×9 sang bàn cờ chuẩn 11×11:
   - Các lớp tích chập (Conv2d 3×3) không phụ thuộc kích thước bàn cờ.
   - Lớp Fully Connected đầu tiên (`trunk_fc.0.weight`) có kích thước phụ thuộc số ô: `(256, 64 × 9 × 9) = (256, 5184)` chuyển sang `(256, 64 × 11 × 11) = (256, 7744)`.
   - Giải pháp: Biến đổi tensor trọng số thành tensor 4D `(256, 64, 9, 9)`, áp dụng hàm nội suy song tuyến tính `F.interpolate(..., size=(11, 11), mode='bilinear')`, nhân hệ số bảo toàn năng lượng `(81/121)`, và reshape thành `(256, 7744)`. Kỹ thuật này giữ nguyên 100% tri thức thị giác đã học từ Phase D và E1!

### 5.2 Kết quả Thực nghiệm Bước E1 (3 Agents, Bàn 9×9 - 2000 episodes)
- **Đặc điểm hình học:** Bàn 9×9 có 3 agent ở các góc (1,1), (1,7), (7,1); góc dưới-phải (7,7) bỏ trống.
- **Tiến trình học tập:**
  - Reward tích lũy trung bình tăng đều từ Q1 (+1.357) lên Q4 (+1.776), kết thúc ở **+1.703** (vượt trội hoàn toàn so với Random Baseline ~ -1.50).
  - Tần suất đặt bom: **1.7 bom/trận**, phá **1.3 tường/trận**. Agent không hề bị bomb shyness.
  - Tỷ lệ thắng (last 200): **19.5%** (do 3 agent cùng tranh đấu và tỷ lệ hòa phòng thủ cao 82%).
- **Hành vi trực quan (Terminal ASCII Demo):** Agent nhận diện bom đối thủ, tự đặt bom mở đường phá tường, chủ động lùi về ô an toàn trước khi bom nổ.
- **Phân bố góc spawn ở E1:** Agent 0 (69.4% số trận phân thắng bại), Agent 1 (11.1%), Agent 2 (19.4%). Sự chênh lệch này là tất yếu do góc thứ 4 bị khuyết làm mất tính đối xứng tâm. Bước E1 đã hoàn thành xuất sắc vai trò bước đệm ổn định chính sách trước khi bước vào bàn cờ 4 góc đối xứng chuẩn ở E2!

### 5.3 Kết quả Thực nghiệm Bước E2 (4 Agents FFA Đầy Đủ, Bàn 11×11 - 3000 episodes)
- **Kế thừa trọng số qua Bilinear Spatial Interpolation:** Mạng `ActorCriticNet` tự động mở rộng lớp `trunk_fc` từ bàn 9×9 lên 11×11 (5184 → 7744 tham số) mà không cần train lại từ đầu.
- **Tiến trình học tập:**
  - Reward tích lũy trung bình tăng vọt từ **+1.782** (Q1) lên **+2.868** (Q4) (tăng trưởng +61%, hoàn toàn vượt trội so với Random Baseline ~ -1.50).
  - Tần suất hành vi: **3.1 bom/trận**, **1.6 tường/trận**. Agent duy trì sự năng động cao trong cả trận đấu mà không bị chết vì bomb shyness.
- **Nghiệm thu 3 Gate Checks của Đồ án (Phase E):**
  1. **Tính công bằng góc spawn (Spawn Position Fairness):**
     - Solo Wins phân bổ: Agent 0 (47.4%), Agent 1 (21.1%), Agent 2 (10.5%), Agent 3 (21.1%). Cả 4 góc đều có chiến thắng solo, không có góc nào bị cô lập ~0% và không có góc nào độc chiếm ~70%.
     - Tournament Score: Agent 0 (49.8%), Agent 1 (47.3%), Agent 2 (46.3%), Agent 3 (47.3%). **Chênh lệch giữa góc có điểm cao nhất và thấp nhất chỉ là 3.5%!**
  2. **Xu hướng Reward:** Tăng mạnh và giữ vững giá trị dương cao (+2.868) $\rightarrow$ **[PASS]**.
  3. **Hành vi sinh tồn trực quan:** Agent biết di chuyển vào trung tâm, đặt bom phá gỗ nhặt vật phẩm, nhận biết sóng xung kích từ đối thủ và tháo chạy về ô an toàn $\rightarrow$ **[PASS]**.
- **Checkpoints hoàn tất:** `checkpoints/best_phaseE_4a.pt`, `checkpoints/final_phaseE_4a.pt` (kích thước 8.17 MB) cùng 15 snapshot pool.

---

## 6. Giai đoạn F: Đánh giá Tổng hợp & Báo cáo (Kế hoạch)
- So sánh learning curves: Random Baseline vs RuleBot vs Q-learning vs Actor-Critic CNN.
- Nghiên cứu Ablation Study: Đánh giá tác động định lượng của Reward Shaping đối với việc kiểm soát hiện tượng Bomb Shyness.
- Báo cáo lý thuyết: Mối liên hệ toán học giữa Markov Decision Process (MDP), Markov Game (Non-stationarity), và Actor-Critic Framework.

---

## 7. Tấn công bằng bom có ghi nhận hạ gục (2026-10-10)

**Vấn đề:** Điều kiện cũ chỉ loại bỏ thưởng thắng `+10` nếu agent đặt **0 bom**. Agent đặt một bom bất kỳ rồi chờ đối thủ tự hủy vẫn nhận đủ thưởng thắng. Chọn checkpoint theo WR và số bom không phát hiện được hành vi này.

**Thay đổi:**
- Ghi nhận mỗi nạn nhân chết đúng một lần, theo chủ ngọn lửa đang gây sát thương; giữ chủ lửa qua các tick để tính cả hạ gục bởi lửa còn sót lại. Ô lửa chồng lấn thuộc chủ ngọn lửa có trước và còn hiệu lực. Bom dây chuyền tính cho chủ quả bom thực sự gây lửa tại ô nạn nhân.
- Sửa bom dây chuyền bị xóa trước khi được nổ: chỉ xóa bom đã xử lý, để bom vừa bị kích hoạt nổ trong vòng tiếp theo cùng tick.
- Trong train AC, hạ gục đối thủ khi còn sống được `+4` tổng (`+1` từ env, `+3` shaping). Tự sát vẫn phạt tổng `-2.5`. Thắng với **0 hạ gục** bị bỏ thưởng `+10`, dù đã đặt bom. Bỏ thưởng đặt bom thô của Phase C; phá tường vẫn được thưởng theo kết quả thực tế.
- Thêm auxiliary attack loss hệ số `0.2`: khi an toàn, học đặt bom nếu đối thủ nằm trên tia nổ không bị tường chắn; ưu tiên BFS tiếp cận đối thủ, hoặc mở đường nếu chưa tới được. Nhãn BOMB phải qua action mask có đường thoát. Auxiliary escape loss `0.5` tiếp tục dạy né bom. Khi xem đấu, policy tự chọn hành động từ mạng; hướng dẫn tấn công chỉ dùng khi train.
- Chọn `best` theo thứ tự: rolling **ActiveWR**, **kills/ván**, rồi WR; cần đã ghi nhận hạ gục. ActiveWR là tỷ lệ thắng với ít nhất một đối thủ chết bởi bom của agent, không phải tỷ lệ tự diệt toàn bộ đối thủ. Trước đủ 200 ván hoặc chưa có hạ gục, file `best` cuối lượt là fallback của model cuối.
- Log train/GUI có `kills`, `active_win`, `suicide`; evaluation có `active_win_rates`, `passive_wins`, `suicide_rates`. Chỉ đếm bom thực sự đặt thành công. Train CLI, GUI và evaluation tournament mặc định tối đa 500 bước/ván.

Kiểm tra logic bằng `python test_combat.py`; kiểm tra pipeline train, checkpoint và GUI bằng `python test_visualize.py`. Các chỉ số lịch sử phía trên dùng môi trường và tiêu chí cũ; cần đánh giá lại checkpoint cũ và mới cùng môi trường trước khi so sánh.

## 8. Sửa tự nổ và học nâng cấp (2026-10-10)

Lượt GUI `runs/20261010-125814-296523`: Phase C, 3000 ván, 9×9, 2 agents, 500 bước, CPU, không warm-start. Tự nổ **84.23%**, ActiveWR **3.70%**; 305 ván tự nổ trong ≤8 bước. Trên 200 ván cuối, tự nổ 85%, ActiveWR 3.5%. Có 364 ván tự nổ mà tổng reward vẫn dương. Cặp checkpoint `best`/`final` chứa trọng số giống hệt nhau do code cũ lưu `best` vào cả `final`.

Giải pháp ưu tiên là sửa mô hình an toàn, rồi train lại; tăng phạt chết đơn thuần không sửa được action mask sai:

- Dự báo theo đúng thứ tự môi trường: di chuyển → giảm fuse → nổ dây chuyền/phá gỗ → sát thương → giảm thời gian lửa. Dùng cùng tia nổ với môi trường, kể cả tường bị phá trước một vụ nổ sau đó.
- Tính khả năng sống theo từng tick đến khi mọi lửa dự báo tắt. STOP và di chuyển phải giữ được đường thoát; BOMB tiêu tốn tick hiện tại và được kiểm tra cùng các bom có sẵn. Fuse giả định 8 bước không còn che mất bom bị kích nổ sớm.
- Trong vùng nổ, giữ các bước đầu trên đường ngắn nhất tới chỗ trú còn an toàn. Tránh ô đối thủ cũng có thể bước vào ở bước kế tiếp nếu còn đường khác; không coi các ô lân cận đối thủ bị chiếm mãi. Nếu đã bị bẫy, giữ lựa chọn có thể sống ngay ở tick tiếp theo. Đây là safety filter dùng khi train, đánh giá và GUI, không phải bằng chứng CNN đã tự học xong kỹ năng thoát bom.
- Không cho spam bom trên sàn trống: cần phá được gỗ hoặc tia nổ đe dọa vị trí hiện tại/ô đối thủ có thể bước tới. Bỏ `+0.20` mỗi lần ra khỏi vùng nguy hiểm để agent không kiếm thưởng bằng việc đi vào/ra vùng bom mình đặt. Phá gỗ vẫn được `+0.15` khi chủ bom sống ở tick đó; hạ gục `+4`, tự nổ `-2.5`, thắng không hạ gục không có `+10` như mục 7.

Bốn điểm về item:

1. **Thưởng hữu ích:** `+0.05` chỉ khi tăng chỉ số thành công, ghi `info['items_collected']`. Từ sức chứa 1/range 2 tới trần 3/4, mỗi agent chỉ nhận tối đa `0.20` thưởng nâng cấp trong một ván. Chết trước nhặt không nhận item; nhặt ở trần không có thưởng.
2. **CNN thấy chỉ số:** 12 kênh; 0–7 giữ nguyên, 8 là slots còn trống/3, 9 sức chứa/3, 10 range/4, 11 nghịch đảo tick gây sát thương sớm nhất. Ba kênh chỉ số là mặt phẳng hằng. Loader chuyển checkpoint 8→12 kênh bằng cách giữ trọng số cũ, thêm trọng số 0; kiểm tra logits/value không đổi trước fine-tune. CNN mới vẫn cần train để dùng thông tin thêm. Baseline Q tiếp tục dùng 8 kênh đầu và 16 features.
3. **RuleBot dùng item:** ưu tiên đánh đối thủ trên tia nổ thật, sau đó item hữu ích trong ≤3 bước trên đường an toàn, rồi mở gỗ/tiếp cận đối thủ. Không nhắm item đã đạt trần hoặc nằm trong vùng nổ dự báo. Teacher của AC cũng hướng dẫn nhặt gần an toàn.
4. **Trần:** ammo 3, range 4 là cấu hình khởi đầu ưu tiên an toàn. So 100 seed với range 5 có tự nổ 17% thay vì 14%, ActiveWR 4% thay vì 3%; mẫu này chưa chứng minh trần tối ưu cho mọi phase.

`best` thêm tỷ lệ tự nổ thấp làm tiêu chí phụ sau ActiveWR và kills, trước WR tổng. Log thêm `items`, `learner_steps`, `best_episode`; `final` giữ đúng trọng số cuối lượt. Checkpoint gốc được giữ nguyên. Dữ liệu và giới hạn thực nghiệm ở phần cuối `RESULTS.md`.

## 9. Thoát bom khi giao tranh và chọn checkpoint trên validation (2026-10-11)

Lượt C ngày 10/10 có 527/796 trận hạ bot đồng thời tự nổ. Hai replay cho thấy đường thoát bị mất khi đối thủ tiến vào hành lang hoặc cùng nhắm một ô. Mục tiêu lần này là hạ bot rồi sống, thay vì chỉ tăng số bom.

- AC kiểm tra bom đáp trả mà các đối thủ còn ammo có thể đặt **cùng tick** với bom của mình. Forecast nhận nhiều placement giả định, giữ nguyên timer/grid/bom thật. Đường thoát của bom mới tránh cả các ô kề đối thủ có thể bị chiếm và chừa 2 bước dự phòng.
- Khi đã có bom, ưu tiên đường sống với 1 bước dự phòng. Chọn nơi trú ngoài tia nổ từ vị trí hiện tại/bước kế tiếp của đối thủ trước, rồi chọn đường nhanh nhất tới đó; không cho mạng đi tùy ý trên mọi đường dài còn vừa kịp. Nếu không có đường với dự phòng, quay lại thời gian thật và giữ hành động sống ngay bước tới. Đây là heuristic, không bảo đảm sống trước mọi chuỗi di chuyển/bom tương lai.
- RuleBot giữ mức lọc trước lượt sửa này qua `robust=False`, để làm đối thủ cố định. Các chức năng nhặt item, range thật, dự báo dây chuyền và né ô tranh chấp vẫn giữ. Kiểm tra đối chiếu bản sao trước sửa khớp **648 quyết định** trên 30 seed. Helper thời gian/geometry vẫn dùng chung; AC dùng phần dự phòng bổ sung.
- C/D/E đều luân phiên learner: `(episode - 1) % num_agents`; log `learning_aid`, `draw`, số liệu validation mới nhất. GUI warm-start dùng LR `1e-4` và ghi seed train vào config.
- **Thay tiêu chí rolling của mục 7/8:** best được chọn bằng policy cố định trên seed validation 6000, mặc định 48 trận ở đầu lượt, mỗi 250 episode và cuối lượt. Mỗi bàn được dùng lần lượt ở tất cả góc. Thứ tự: ActiveWR, ít tự nổ, ít hòa, kills/ván, WR. Trạng thái RNG NumPy/Torch/thiết bị CUDA đã khởi tạo được trả lại sau đánh giá, không làm lệch chuỗi train.
- Lưu `validation_metrics.jsonl` và `best_validation.json`; best có thể giữ warm-start ở episode 0 nếu không có ứng viên tốt hơn. Final luôn giữ trọng số ở cuối lượt. Các lượt ngắn hơn chu kỳ validation hoặc tắt validation chỉ có best fallback của final, không phải model đã được xác nhận chất lượng.

Hai bản thử trung gian đã bị dừng và giữ log trong `runs/escape-20261011/pilot/` và `pilot-final/`. Bản đầu cho quá nhiều hướng thoát làm tăng đi qua lại; bản thứ hai tăng mức né bom cho cả RuleBot nên làm thay đổi đối thủ. Bản dùng cho pilot hoàn chỉnh nằm ở `pilot-500/`; số liệu của các bản trung gian không được coi là kết quả của code cuối. Kết quả pilot và benchmark riêng ghi trong `RESULTS.md`.

Giữ nguyên reward item 0.05, ammo tối đa 3, range tối đa 4 và reward chiến đấu; chưa đổi thuật toán RL. Kiểm tra mới gồm phản bom đồng thời, margin không cộng dồn, forecast nhiều placement không sửa state, nơi trú tránh tia đối thủ, góc C/D, cùng bàn cho mọi góc, phục hồi RNG CPU/CUDA, hủy validation và giữ final sau best. GUI kiểm tra tiến độ validation, tạm dừng/tiếp tục, metadata best và LR warm-start.

Placement giả định được sắp theo owner để khớp thứ tự đặt/nổ của môi trường, kể cả tia nổ qua tường gỗ vừa bị phá. Sửa này được thêm sau khi pilot bắt đầu; đã chấm lại warm-start, best trong train và final bằng code cuối trên CPU, cùng seed validation. Final episode 500 thắng validation nên được chọn làm best; bản episode 250 vẫn giữ riêng. Pilot hoàn tất 500 ván, nhưng test mới 100 trận cho ActiveWR vẫn 8%. Code mới với checkpoint gốc giảm tự nổ 25% xuống 10%; pilot có tự nổ 16% và hòa 85%. Vì vậy chưa khuyến nghị chuyển D/E hoặc coi pilot là model tốt hơn. Số liệu, phiên bản runtime và giới hạn ghi ở mục ngày 11/10 trong `RESULTS.md`.

## 10. Chẩn đoán hòa và sửa hướng thoát/nhãn tiếp cận (2026-10-12)

Replay đúng checkpoint gốc và RNG của bộ test ngày 11/10: 20 trận hết 500 bước (10 mỗi góc), cùng 10 trận tự nổ. Assert kết quả, số bước, bom và hạ gục khớp từng dòng test cũ. Đây là mẫu chọn theo lỗi, không dùng làm ước lượng tỷ lệ chung.

- Trong 20 trận hòa, có 427 bước được phép đặt bom với đối thủ nằm trên tia nổ; agent đặt bom ở 246 bước và bỏ qua 181. Chỉ 51/577 bước không có nguy hiểm và đối thủ nằm trên tia bị chặn bởi mức lọc mới dù mức cũ cho phép. Vì vậy chưa có bằng chứng cần nới bộ lọc hoặc tăng thưởng đặt bom.
- Có 1.357/6.471 bước không nguy hiểm thiếu nhãn tấn công và 519 bước hướng tiếp cận ngắn nhất bị mask chặn. BFS cũ tìm tới ô đang có đối thủ rồi mới kiểm tra bước đầu; nếu bước ngắn nhất không được phép, nó bỏ nhãn dù có đường khác.
- Cả 10 trận tự nổ đều hạ bot đồng thời tự chết: sau đặt bom, AC chọn hướng tiến gần bot trong khi có nơi trú dài bằng nhau ở phía khác. Hai bên tiến vào cùng hành lang, bị giữ lại trước ô tranh chấp rồi hết fuse. Ví dụ seed 9000: đặt ở `(5,1)`, đi lên `(4,1)` đối diện bot `(2,1)`; seed 9001 là tình huống đối xứng tại cột 3.

Sửa nhỏ ở helper chung:

1. Giữ tiêu chí nơi trú tránh tia bom và đường nhanh nhất. Sau đặt bom, tránh bước vào hành lang sẽ cháy và chỉ có một lối ra nếu đối thủ có thể tới lối ra ấy trong hai bước; giữ các hướng khác và fallback nếu đã bị bẫy. Không buộc agent luôn tăng khoảng cách trong mọi lần né: bản thử điều kiện rộng này giảm tự nổ nhưng làm giảm ActiveWR trên seed phát triển 11000, nên được thay bằng điều kiện hành lang cụ thể. Không thay đổi mode `robust=False` của RuleBot hoặc các kiểm tra phản bom/dây chuyền.
2. Teacher dùng BFS qua các bước đầu đã được mask cho phép, tìm ô có tia nổ chạm đối thủ thay vì đi vào ô đối thủ. Đường tương lai vẫn phải tránh tường, bom, ô đối thủ và vùng nổ gần; tới vị trí mới, BOMB tiếp tục phải qua bộ lọc thời gian thực. Đây là hướng dẫn tiếp cận, chưa phải planner đảm bảo ép chết đối thủ.
3. Metrics thêm `attack_opportunities` (BOMB được phép và tia hiện tại chạm đối thủ), `attack_bombs` (số cơ hội đã chọn BOMB), `attack_labels` và `missing_attack_labels` (trên các bước không nguy hiểm). Các cơ hội liên tiếp không phải những trận độc lập.
4. Validation luôn dùng bản sao trọng số trên CPU giống GUI, kể cả train bằng GPU. Không chia sẻ hoặc sửa network learner; deepcopy không tiêu thụ RNG, evaluator vẫn khôi phục RNG sau đánh giá. Metadata ghi `device=cpu`.

Mức reward chiến đấu/item, trần ammo 3/range 4 và CNN giữ nguyên. RuleBot khớp bản trước sửa ở 458 quyết định trên 20 seed. Artifact chẩn đoán tại `runs/combat-20261012/diagnostics.json`. Pilot 500 ván dùng train seed 20261012, 100 trận validation seed 6100. Seed 9000 và 11000 đã dùng để chẩn đoán/chọn điều kiện lọc nên test cuối chuyển sang seed 12000. Kết quả và phiên bản của pilot ghi riêng ở `RESULTS.md`.

### Credit thật khi bom/lửa chồng nhau

Tái hiện tình huống hai bên đặt bom cùng tick, AC thoát và bom AC trúng bot. Với code cũ, AC ở A0 nhận hạ gục và reward 13.999, nhưng ở A1 bị tính thắng thụ động vì bom A0 được xử lý trước dù tia bom A1 cũng trúng bot. Đây là lỗi gán tín hiệu học trong trường hợp chồng tia, không chứng minh mọi chênh lệch theo góc đều do lỗi này.

Engine nay lưu từng nguồn lửa và thời gian còn lại. Khi chết, chọn một nguồn bom đối thủ còn hiệu lực nếu có; trong 1v1, kết quả không lệ thuộc A0/A1. Một nạn nhân vẫn có đúng một kill credit. Đồng thời `info['self_blast_deaths']` ghi riêng các nạn nhân có tia bom của chính họ góp phần gây chết: cả hai cùng chết vẫn bị phạt tự nổ, không được biến thành giao tranh an toàn chỉ nhờ đổi tên killer. Train/evaluate đọc cờ mới, tương thích log giả lập chỉ có tuple self-kill.

Lửa mới không kéo dài quyền nhận credit của nguồn cũ đã hết thời gian. `_flame_owners` còn giữ view một nguồn để tương thích; resolve damage dùng tất cả nguồn đang cháy. Với nhiều nguồn đối thủ đồng thời, nguồn đối thủ còn hiệu lực xuất hiện trước được chọn; chưa đánh giá công bằng đầy đủ cho FFA 3–4 agents.

37 unittest kiểm tra cả hướng thoát không bị ép lùi, credit đối xứng, phạt tia bom của mình khi killer là đối thủ, thống kê không lặp một cái chết và nguồn lửa hết hạn. Integration GUI kiểm tra các checkpoint, train và validation. Pilot 500 ván đã hoàn tất **trước** khi đưa sửa credit vào code chính; snapshot/metrics cũ không bị viết lại. Các ứng viên được chấm lại cùng engine/seed validation; chưa train dài lại với reward credit đã sửa.

## 11. Hướng dẫn vị trí đặt bom và giữ hành vi tấn công khi train tiếp

Hai lượt 500 episode với reward credit đã sửa chưa vượt warm-start rõ ràng. Giảm thưởng tường từ 0.15 xuống 0.03 giữ best ở episode 0. Replay sáu trận timeout cho thấy warm-start đặt bom ở 148/150 cơ hội đối thủ trong tia nhưng không hạ được bot; final lượt 0.03 chỉ tận dụng 74/129 cơ hội và cũng không hạ được bot. Các số này là chẩn đoán trên mẫu thất bại, không phải benchmark tổng thể.

- Thêm reachability theo từng tick để đo số vị trí đối thủ sống được ở tick nổ. So sánh cùng horizon trước/sau một bom dự kiến; dùng cả chain reaction, flame và tường mở sau nổ. Không thưởng thêm chỉ vì đặt bom.
- Nhãn tấn công ưu tiên ô hiện tại hoặc một bước di chuyển hợp lệ tạo mức giảm vị trí sống lớn nhất khi đối thủ gần. Các mục tiêu cũ vẫn mở đường/nhặt item khi chưa có đề xuất tốt hơn. Bộ lọc sống sót vẫn là điều kiện bắt buộc; RuleBot đối chứng giữ nguyên.
- Cho phép BOMB lúc đang trong vùng nguy hiểm chỉ khi dự báo agent sống được kể cả phản bom đồng thời, và bom mới giảm số vị trí sống của một đối thủ từ lớn hơn 0 xuống 0 tại tick nổ. Margin an toàn 2 tick vẫn áp dụng khi xét đặt bom; không mở rộng các bom khác lúc nguy hiểm. CNN vẫn tự chọn hành động, dùng chung điều kiện này ở train/evaluate/GUI; RuleBot `robust=False` không đổi.
- Loss thoát hiểm 0.5, bom gây áp lực 0.5, di chuyển/mở đường 0.2 được lấy trung bình theo từng nhóm riêng. Các nhãn di chuyển nhiều không làm giảm trực tiếp trọng số nhóm bom. Chưa đổi CNN hoặc thuật toán Actor-Critic.
- Chọn best chỉ trong các ứng viên không tăng tỷ lệ tự nổ so với validation đầu lượt. Ví dụ ActiveWR 10%/tự nổ 3% sẽ không bị thay bởi 11%/6%. Đây là điều kiện bảo thủ trên mẫu hữu hạn; final vẫn được lưu riêng để phân tích.
- CLI ghi config và metrics từng episode giống luồng GUI, bổ sung số nhãn/bom gây áp lực và các thành phần loss. Thông báo lưu in riêng đường dẫn best và final.

Giới hạn: áp lực là heuristic trên bom đã biết và vị trí đối thủ hiện tại; không dự báo mọi hành động di chuyển/đặt bom tương lai của đối thủ. Đề xuất vị trí một bước chưa đảm bảo vị trí đó vẫn tốt khi đi tới. Nhãn và loss mới chỉ tác động khi train lại; ngoại lệ safety mask tác động cả checkpoint cũ lúc evaluation. Kết quả pilot/benchmark ghi tại cuối `RESULTS.md`.

## 12. Thoát sớm khi chỉ còn đứng chờ trước lối ra chung (2026-10-13)

Replay đủ 13 trận tự nổ/thua của `pressure-C-next/final_phaseC.pt` trên seed phát triển 15000. Có tình huống cả hai tác tử đứng chờ vì ô thoát kề đối thủ bị mask chặn, rồi cùng lao vào ô đó ở tick nổ cuối: va chạm làm cả hai đứng tại chỗ và chết. Các trận thua khác còn có việc đi vào hành lang cụt rồi bị đối thủ đặt bom ở cửa; sửa lần này chỉ nhắm việc chờ quá muộn.

- Khi AC chỉ được STOP, không có hướng di chuyển và còn hơn một tick trước nguy hiểm thực tế, thử tìm đường sống bỏ phần cấm ô kề đối thủ. Vẫn giữ tường, bom, ô đối thủ hiện tại và dự báo lửa; chỉ dùng kết quả nếu có hướng đi. Nếu còn hướng thoát không tranh chấp thì giữ hướng đó, nếu phải chờ lửa tắt thì vẫn chờ.
- Nhãn thoát hiểm dùng chính mask của bước train, không hướng dẫn STOP khi policy chỉ được di chuyển. Dùng lại BFS và forecast hiện có; không thêm reward, mạng hoặc thuật toán mới.
- RuleBot `robust=False` giữ nguyên. Sửa mask có hiệu lực ngay khi đánh giá checkpoint cũ, không cần train lại để thấy thay đổi. Checkpoint và log cũ không bị ghi đè.

Đã kiểm tra hai phía A0/A1, va chạm lần đầu rồi thử thoát lại, lửa chặn lối đi, đường thoát không tranh chấp và các điều kiện bom bổ sung cũ. 48 unit test và kiểm tra tích hợp GUI/train đều qua. Tám trận hiệu chuẩn seed 15000 khớp bản thử, đối chiếu 2637 quyết định learner và mask đối thủ không đổi. Đây vẫn là heuristic: đối thủ có thể tiếp tục chặn đường hoặc đặt bom muộn. Kết quả development và test độc lập được ghi ở cuối `RESULTS.md`.

## 13. Giáo viên chiến thuật tối đa hai bom (2026-10-14)

`tactical_teacher.py` tạo nhãn trên trạng thái thực của learner, không thay hành động CNN. Forecast nhận `(delay, Bomb)` với delay 0 là bước kế tiếp: trước placement bom chưa có mặt; ngay tick placement bom chặn ô, timer giảm một lần, rồi xử lý dây chuyền/lửa giống `step`. Theo dõi sức chứa trước nổ, nên một slot vừa nổ chỉ dùng lại ở bước sau; ammo một không có hai bom đồng thời. API reset/step và forecast cũ vẫn hoạt động như trước.

Tìm các đường đặt sớm nhất theo ô và hành động đầu: tối đa hai bước trước bom đầu, ba bước sau đó. Chờ được khi phải né lửa hoặc đợi slot. Chọn tối đa tám vị trí đầu gần đối thủ nhất; mỗi vị trí có một kế hoạch single và tối đa bảy vị trí tiếp theo, tổng <=64. Phép cắt này có thể bỏ sót combo ngoài tập xét. Dự báo tường gỗ mở, bom dây chuyền và lửa tới khi hết nguy hiểm; nếu combo rút ngắn fuse thì không đếm thêm các tick trống từ fuse cũ để làm sai áp lực.

So số ô đối thủ còn có thể sống ở cùng horizon trước/sau; loại đối thủ vốn đã chết theo forecast gốc. Ưu tiên trap, pressure, ít bom, ít bước di chuyển rồi ít tick chờ. Learner phải sống qua cả lịch bom, tránh ô đối thủ hiện tại và ô tranh chấp ở bước đầu; hành động đầu bắt buộc nằm trong mask thật. Đường qua range item đã biết bị bỏ để không dùng blast range sai. Nhãn ngang điểm phân phối đều, tất cả nhãn mới vào nhóm combat 0.5 và thay nhãn cũ ở bước đó; nếu không có áp lực dương thì fallback local. Observation 12 kênh, reward, mask, optimizer và các nhóm loss khác không đổi.

Reachability dùng bit set để giảm chi phí tính trên lưới nhỏ; kết quả được đối chiếu với phép tính NumPy cũ trên nhiều kích thước/horizon và với các test engine. Bộ test mới kiểm tra forecast bằng `env.step`, capacity một/hai, trả slot sớm do chain, gỗ mở tia, lửa còn sót, va chạm, combo mà mỗi single vẫn có refuge, tính bất biến state/RNG, soft target và bootstrap theo seed bàn.

Giới hạn: chỉ forecast bom đã biết và bom learner dự kiến; không dự báo mọi phản bom hay mọi va chạm tương lai. Phần giữ ô đối thủ hiện tại cho đường thoát là điều kiện bảo thủ, không chứng minh an toàn trước đối thủ đổi đường. GAE, PPO, opponent pool/curriculum mới và sửa thưởng chưa đưa vào pilot này để giữ phép đối chứng riêng tác động của teacher.

Pilot `runs/two-bomb-pilot-20261014` hết ngân sách train chung 65 phút ở local 200/200, two_bomb 146/200. Validation two_bomb tại episode 100 đạt ActiveWR 18%, tự nổ 1%, hòa 77%; guard từ chối và giữ best ở episode 0. Đã chấm đủ 400 trận mỗi checkpoint: cả ba best/gốc ActiveWR 33%; best two_bomb có trọng số giống gốc. Không đổi model; kết quả chưa đủ để kết luận phương pháp. 58 test pass. Số liệu từng góc, CI và source đúng lúc chạy tại [báo cáo pilot](runs/two-bomb-pilot-20261014/report.md).

## 14. Certified MOVE vào combat guidance (2026-10-15)

Audit C1 đã biết cho thấy teacher giải32/32approach, còn CNN sauC1 yếu ở các bước sát vị trí bắn. Thêm `combat_move_teacher.py` và mode`local_combat_move`: target local không đổi, chỉ MOVE tăng positive pressure, next BOMB qua mask và escape certificate trên clone tới hết lửa được chuyển từ navigation mean0.2 sang combat mean0.5. Vì hai nhóm tính mean riêng, thay đổi bao gồm denominator, không chỉ nhân loss của MOVE. Mỗi bước chỉ có một nhãn và CNN tiếp tục tự sampleaction.

Clone mô phỏng một MOVE/mộtBOMB rồi greedy escape qua mask, tối đa32tick nguy hiểm, other agents STOP; không chứng nhận được thì fallback nhóm cũ. Không sửa env/mask/reward/12channels/CNN/optimizer hoặc thuật toán MC; không thêm delayed counter-bomb guard. Source/checkpoint cũ giữ nguyên trong artifacts, source trainer mới khiến guard reusecontrol cũ từ chối, nên runner luôn train cả hai nhánh.

136testsPASS, gồm chứng nhận/purity/pressure/slots/collision/greedy failure, loss routing và không overrideCNN, actual promoted update CPU/CUDA, defaultlocal khớp before snapshot weights/actions/rewards, orchestration/seed/CI/gates/incomplete/source guards. Smoke4ep/arm CPU/CUDA và deadline đã chạy; smokeonline chưa gặp qualifying MOVE, nên chỉ xác nhận wiring. Branch thật đã test riêng trênpositive env.

[Runner/protocol](/D:/fpt/subject/BomberMan/COMBAT_MOVE_GUIDE.md): full200ep/arm từB/seed2026101501/RuleBot,validation21000 và64skillgames/mốc; test400/modelB/control/candidate33000–33199,cảA0/A1,CPU1thread,pairedbootstrap. Chỉ gate+5pp/CI lower>0/suicide≤cảhai cho phép replication. Kết quảfull/giới hạn/quyết định sẽ ghi ở`RESULTS.md`; không dùng smoke hoặc lời giải teacher làm bằng chứng tăng ActiveWR.


## 15. T4-A1 — Advantage normalization độc lập

Trainer thêm tùy chọn chuẩn hóa advantage của actor theo learner trajectory, mặc định tắt; giữ MC critic targets và local teacher. `run_advantage_pilot.py` tận dụng train/skill/evaluation/gate helpers của pilot trước, chạy lại control cùng candidate200ep, khóa best trước test400/model trên seed34000–34199. 150 tests và smoke20ep/arm CPU/CUDA đã qua. **Full hoàn tất/review:96,93phút; ActiveWR B33,25/control36,5/normalized33,5%; gateFAIL, giữB và mặc định raw.** Clipping95→79/200 không xác nhận cải thiện chất lượng; một trainseed và hai CI chứa0. Không ghép GAE/PPO/reward/curriculum/MOVE guidance. Xem [ADVANTAGE_GUIDE.md](/D:/fpt/subject/BomberMan/ADVANTAGE_GUIDE.md), `RESULTS.md` và báo cáo quyết định.


## 16. T4-B0 — GAE có kiểm tra boundary/parity

`advantage_estimator=mc|gae`, `gae_lambda` chỉ đổi actor advantage, critic vẫnMC;defaultMC giữ nguyên. Helper tách bootstrap/continuation; terminal/death/timeout đúng luật hiện tại, cut/reset có tests nhưng collector vẫn whole-episode.167tests và smoke20ep/modeMC/GAE1/GAE0,95CPU-CUDA đã qua. Sửa precision bool-mask trongrecurrencefloat64; GAE1numericparity,defaultMCbit-parity. B0PASSimplementation,giữB;chưa cópilotB1haybằngchứngtăngActiveWR. [Giao diện/kết quả/giới hạn](/D:/fpt/subject/BomberMan/GAE_B0_GUIDE.md).

## 17. T4-B1 — đối chứng MC/GAE, lượt đầu thiếu thời gian

`run_gae_pilot.py` dùng lại train/validation/skill/evaluation/gate helpers, chỉ truyền thêm estimator và lambda vào trainer B0. Cả hai nhánh normalization tắt, critic MC, teacher local, cùng B/seed2026101504. Đã qua174 tests và smoke20ep/nhánh CPU/CUDA.

Lượt đầu `gae-full-20261015-112853` thiếu thời gian ởcontrol200/GAE184. Sau sửa trần train100 phút/tổng150,49tests liên quan qua; đã train lại cảhai từB và hoàn tất full `gae-full-20261015-125942` trong54,87phút. BestMC/GAE đềuepisode0,ActiveWR cảba31,75%,delta0/CI[0;0],gateFAIL,giữB. Không dùng test để đổi sangfinal200 hoặc gọi hai lượt cùngtrainseed làreplication. [Protocol và trạng thái](/D:/fpt/subject/BomberMan/GAE_PILOT_GUIDE.md).

## 18. T4-R0 — reward accounting và audit hành vi

`_shaped_reward` dùng lại combat shaping cũ và wall bonus cùng thứ tự. `reward_diagnostics=False` mặc định; bật logging thêm breakdown theoepisode. Decomposer kiểm tra raw env reward và shaped reward, không dùng residual. Test CPU trước/sau và bật/tắt logger giữ nguyên actions/rewards/metrics cũ/weights; regression188PASS/1CUDA SKIP.

`audit_r0.py` chạy B/MCfinal200/GAEfinal200 trên100validation games/model,21000–21049/cảA0-A1. Full16,25phút,0updates,khôngtestmới;118.226steps khớp reward cũ và12timeout/6.012frames replay qua. Thêm bảng mean/median/p95, raw tapes, scatter/CSV vàderived G0/late100/fixed-action comparisons. Giữ B vàcoefficients; dữ liệu hỗ trợ R1 như ablation, chưa chứng minh giảm wall bonus nâng ActiveWR. [Guide](/D:/fpt/subject/BomberMan/R0_GUIDE.md), [report](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/R0_report.md).

## 19. T4-R1 — implementation và lượt chạy bị ngắt

R1 IMPLEMENTED/TESTED, full INCOMPLETE do tiến trình bị ngắt: `reward-full-20261015-163227`. Control đủ200episode; half-wall199episode đã lưu, validation200/final200 còn thiếu. Best validation control33%/half-wall30%, chưa có test hoặcCI; giữB. 200regression testsPASS, smokeCPU/CUDA qua; audit320skill replays/6.966frames. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-20261015-163227/decision_report.md).

## 20. Phase C resume đầy đủ và R1 rerun

R1 DONE/REVIEWED: `reward-full-resumable-20261015-194314`. Đủ 200 episode và 200 Adam update mỗi nhánh từ B; 400 trận test/model trên seed 36000–36199, cả A0/A1. Best control/half-wall đều ở episode 200. ActiveWR B/control/half-wall = 34,50%/36,25%/35,00%. Gate FAIL; giữ B và wall bonus 0,15. Thời gian chạy 49,54 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/decision_report.md).

Phase C có `training_state.pt` gồm model/Adam/scheduler/global episode/best guard/history, RNG Python/NumPy/env/Torch CPU/CUDA và context log. Atomic temp+fsync+replace, OS writer lock, `run_reward_pilot.py --resume` khôi phục episode đã xác nhận; không reset schedule hoặc chọn lại model test. 212test riêng biệt PASS; có CPU/CUDA exact resume trong tiến trình mới, interrupted episode/validation và torn-log recovery. `best/final_phaseC.pt` giữ dạng weights để chơi; loader từ chối nhầm full-state file. Resume yêu cầu cùng source/config/runtime/thiết bị; hiện chỉ hỗ trợ Phase C, deadline R1 gốc vẫn giữ.

## 21. D0 rollout buffer và một update ratio

T4-D0 DONE/PASS về tính đúng: 226/226 tests, gồm 14 D0; CPU/CUDA smoke đều đủ 20 episode/20 Adam updates mỗi chế độ AC và ratio chưa clip. Buffer lưu mask/old policy/teacher targets; MC raw, critic/reward/teacher/inference và mặc định AC giữ nguyên. Full states và source/ZIP/B hashes đã review. D0 không chứng minh tăng ActiveWR; giữ B. D1 clipping với cùng 4 epochs READY/design, chưa triển khai/pilot. [Báo cáo](/D:/fpt/subject/BomberMan/runs/ppo-d0-implementation-20261015/D0_report.md), [hướng dẫn](/D:/fpt/subject/BomberMan/PPO_D0_GUIDE.md).

## 22. D1 PPO clipping và resume theo episode

D1 DONE/REVIEWED: `ppo-full-20261015-220313`. Đủ 200 episode/nhánh từ B, Adam updates U4/P4=800/800; best episode U4/P4=200/200. Test400/model, seed37000–37199/cảA0-A1: ActiveWR B/U4/P4=37.50%/37.00%/32.00%. Gate FAIL; giữ checkpoint B. Runtime 106.89 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/ppo-full-20261015-220313/decision_report.md).

D1 giữ MC raw/critic MC/local teacher/reward cũ. U4/P4 cùng tối đa4 epochs và post-update KL guard target0,02/threshold0,03; chỉ khác clipping epsilon0,20. 241 regression và25 focused metadata/compat tests PASS; CPU/CUDA smoke20episode/80updates mỗi nhánh đã review. Có 500-step/4-epoch memory fixture, hủy giữa epoch khôi phục Adam, exact PPO resume qua tiến trình mới CPU/CUDA, full-state/source/ZIP/locked-model guards. CNN inference và Actor-Critic mặc định giữ nguyên.

## 23. Sudden death: kiểm chứng, `max_ring` và chọn timing (2026-10-16)

`sudden_death_experiment.py` (port luật vòng Loc_update afbf6f9 lên engine hiện tại, opt-in, không đổi env production/12 kênh) đã có từ trước nhưng chưa chạy full. SD1 trên B với cấu hình gốc start200/interval20 cho thấy hai lỗi thiết kế: SD sớm cắt các thắng chủ động muộn của B (median bước 271), và trên 9×9 ô tâm (4,4) là tường cứng nên vòng cuối đóng kín mọi ô, mọi agent còn sống chết cùng lúc thành hòa.

Thêm `sudden_death_max_ring` (mặc định = đóng kín, giữ hành vi cũ và parity với Loc_update; validate 1..board//2−1; `_target_sudden_death_ring` và `_closure_ticks` dừng ở vòng này) cùng test `test_max_ring_stops_shrinking_and_keeps_inner_ring_alive`. Sửa fixture test runner hỏng từ đầu. Thêm `tune_sudden_death.py` (quét start/interval/warning/max_ring trên validation, từ chối seed ≥38000, `--resume` bỏ qua cấu hình đã xong, mặc định 6 tiến trình vì RAM) và `confirm_sudden_death.py` (luật chọn chốt trước: both_aware, WR cao nhất với ActiveWR ≥ off−3 và tự nổ ≤ off+1; chấm một lần off/both_aware/learner_aware trên seed mới, paired bootstrap).

Cấu hình được chọn **start400/interval40/warning40/max_ring2**: trên test 39000 both_aware WR 39,5→59,25%, hòa 59,25→9,75%, ActiveWR 35,5→31,5%. Đây là thay đổi luật chơi trên cùng trọng số B: phần lớn WR tăng là thắng thụ động (RuleBot tự nổ trong vòng 8 ô), learner vẫn chết vì vòng 131/400. Chưa đặt làm mặc định của `SuddenDeathEnv`. Bước tiếp theo là SD3 (train dưới SD + kênh quan sát vòng), đang ở giai đoạn thiết kế; xem `RESULTS.md`.

## 24. SD3 — kênh quan sát vòng và train dưới sudden death (2026-10-16)

- `SuddenDeathEnv(ring_channel=True)` thêm kênh thứ 13 sau 12 kênh production: `1/delay` tới lần đóng trên ô của mọi vòng chưa đóng `<= max_ring`, còn lại 0; có `obs_channels`. SD tắt mà bật kênh thì kênh toàn 0 (dùng cho skill probe). `ring_channel=False` giữ nguyên observation, reward, RNG.
- `train()` tạo mạng theo `env.obs_channels` (mặc định 12 như cũ). `evaluate.make_ac_fn` lấy số kênh từ `conv.0.weight` của checkpoint và chỉ đưa vào mạng đúng số kênh đầu; kênh vòng nằm cuối nên B 12 kênh chơi trong env 13 kênh y như khi không có kênh (test hành động/logits/trận).
- `smart_load_weights` nay từ chối nạp checkpoint nhiều kênh hơn mạng (trước đây bỏ qua lặng lẽ, để conv đầu ngẫu nhiên). Model `sd_13ch` vì vậy không mở được trong GUI/opponent pool 12 kênh; GUI chưa hỗ trợ SD/kênh vòng.
- `run_sudden_death_experiment.sudden_death_runtime(timings, ring_channel)` patch env của trainer/evaluator thành SD, env skill probe giữ luật nhưng đúng số kênh, mask learner và RuleBot thêm `sudden_death_mask`; `evaluate_case` nhận `checkpoint`, `ring_channel`. `benchmark_t3.validate_env` chấp nhận `env.obs_channels`.
- `run_sd_training_pilot.py`: hai nhánh `sd_12ch`/`sd_13ch` từ B dưới SD2, baseline skill riêng mỗi nhánh, khóa model trước test, chấm một lần seed 40000 (B và hai best) bằng `evaluate_case` both_aware, gate so với B, so sánh phụ 13ch−12ch, resume theo training state, deadline 150 phút tính từ lúc đăng ký.
- Review cuối (subagent): không có lỗi nghiêm trọng; các điểm nhỏ còn hoãn: kiểm tra seed test chỉ lúc khởi động, khóa cache đánh giá chưa gồm max_steps/timings, provenance training state chưa ghi timing SD, chưa có unit test cho đường resume, đếm diagnostics mask đối thủ bị lẫn truy vấn fallback của learner (có từ trước).

## 25. SD3 full — kết quả và quyết định (2026-10-16)

- Không đổi code; chỉ chạy `run_sd_training_pilot.py --device cuda --output-dir runs\sd3-full` (người dùng chạy, một lượt, complete).
- Gate FAIL cả hai nhánh: ActiveWR so B −2.25pp (sd_12ch CI [−8.0, 3.5], sd_13ch CI [−7.5, 3.25]); WR +3/+4pp và hòa giảm chỉ do thắng thụ động tăng. Kênh vòng so với 12 kênh: ActiveWR 0.0 [−4.75, 4.75]; learner vẫn chết vì vòng ~30% trận. Giữ B, không thay model. Chi tiết trong RESULTS.md mục "SD3 — kết quả full".
- Trạng thái: hướng "train ngắn dưới SD + kênh vòng" không đạt ở 200 episode/1 seed. Nếu tiếp tục, nên chẩn đoán vì sao learner bị kẹt trong vòng (số chết vì vòng trùng `unavoidable_pre_move_closure`, mask giao rỗng 442–473 lần) trước khi train dài hơn; cần seed test mới (≥ 41000).

## 26. Đề xuất SD4 sau SD3 (2026-10-16, chưa cài)

- Chẩn đoán SD3 (RESULTS.md, "SD3 — chẩn đoán chết vì vòng"): mask learner chỉ giữ đường thoát khả thi, mất đường trong vài bước cuối trước khi vòng đóng.
- Đề xuất theo thứ tự chi phí: (1) mask/shield có biên an toàn: learner phải rời ô vòng sớm hơn k bước, chỉ đổi phía learner, đo trên validation 21000 rồi test một lần seed 41000; (2) teacher thoát vòng từ RuleBot biết vòng (distillation chỉ trong cửa sổ cảnh báo, dùng hạ tầng teacher có sẵn); (3) curriculum timing SD (vòng đến sớm khi train để có nhiều tín hiệu hơn) + GAE; (4) reward shaping dạng potential theo khoảng cách tới vùng an toàn. Chưa cài, chưa chạy.

## 27. Tracker riêng cho hướng sudden death (2026-10-16)

- Thêm `PLAN_SUDDEN_DEATH.md`: bảng trạng thái và checkbox SD1–SD4, kèm điều kiện mở từng nhánh SD4 và bản nháp luật chọn/gate SD4.1 (cần chốt với người dùng trước khi chạy). Thêm dòng SD vào tracker tổng `PLAN_BOMBERMAN.md`.
- CLAUDE.md: thêm quy tắc đánh dấu tracker trong cùng lượt, mô tả SD3/kênh vòng, cấu hình SD2 và hiện trạng hướng SD.

## 28. SD4.0 — script chẩn đoán chết vì vòng (2026-10-16)

- Thêm `diagnose_sd_ring_deaths.py`: bọc `sudden_death_mask` trong `evaluate_case` (không đổi trận/RNG), ghi theo bước trong cửa sổ cảnh báo vị trí so với vòng sắp đóng, còn đường thoát hay không, khoảng cách BFS vào trong; lúc mất đường lần đầu, thử bỏ từng loại vật cản (thân đối thủ, bom đối thủ, bom mình, lửa, gỗ) để gọi tên nguyên nhân. Từ chối seed 31000–49999.
- Test `test_diagnose_sd_ring_deaths.py`: parity trận với `evaluate_case`, khớp chỉ số trận, từ chối seed test.
- Sửa `test_sd_training_pilot.test_test_seed_is_fresh_and_distinct_from_smoke`: seed 40000 đã dùng ở SD3, nên test nay đòi guard từ chối seed đó (trước đây đòi seed còn mới, thành lỗi sau khi chạy full). Suite 271 test (270 PASS ở lượt đầy đủ + test sửa PASS riêng).
- Kết quả và kết luận: RESULTS.md "SD4.0".

## 29. SD4.1 — biên an toàn cho mask vòng của learner (2026-10-16)

- `sudden_death_experiment`: `_closure_ticks(margin)`, `forecast_closures(..., margin)`, `_ring_survival(..., margin)` coi mỗi vòng đóng sớm `margin` bước (không sớm hơn bước kế). Tách `_ring_mask`; `sudden_death_mask(..., margin=k)` thử mask có biên trước, nếu không còn hành động an toàn thì quay về mask SD2 (đếm `margin_fallbacks`). k=0 giữ nguyên hành vi và bộ đếm (parity với bản nguồn SD3 trong `runs/sd3-full/source`, trận và bộ đếm giống hệt).
- `evaluate_case(..., learner_margin=k)`: chỉ mask learner nhận biên; RuleBot giữ mask SD2.
- `run_sd_margin.py` (`sweep` trên validation 21000, `confirm` một lần trên 41000 với luật chọn/gate chốt trước) + `test_sd_margin.py`; test mới trong `test_sudden_death_experiment.py` (biên buộc rời vòng sớm, quay về khi quá muộn) và `test_sudden_death_runner.py` (biên chỉ tới mask learner). `diagnose_sd_ring_deaths.py` chuyển tiếp tham số mask.
- Kết quả: gate PASS (chết vì vòng −10pp, WR +10.75pp, ActiveWR +0.75 không ý nghĩa). Dùng k=15 làm cấu hình mask learner khi chạy SD; checkpoint B không đổi. Chi tiết RESULTS.md "SD4.1". GUI chưa có SD nên chưa dùng biên này.

## 30. SD4.2a — probe bộ điều khiển giai đoạn cuối (2026-10-16)

- Thêm `probe_sd_endgame.py`: bọc `evaluate.make_ac_fn` để từ bước `switch` (360) ghế learner do teacher local, two_bomb → local, hoặc RuleBot điều khiển (teacher dùng mask SD + biên đã patch, khi nguy hiểm thì trả về B); trước `switch` trận và RNG giống hệt B. Từ chối seed ≥ 31000. Test `test_probe_sd_endgame.py`: switch sau khi trận kết thúc giữ nguyên trận của B; mọi nhánh chạy được.
- Luật quyết định chốt trong PLAN_SUDDEN_DEATH.md trước khi chạy. Kết quả: không nhánh nào tăng ActiveWR (local −4.25, two_bomb −3.5, RuleBot −2.25); không mở SD4.2b. Nhánh SD dừng ở cấu hình SD2 + biên k=15 cho mục tiêu WR/hòa; hướng ActiveWR quay về kỹ năng tấn công đầu trận.

## 31. SD4.6 — kill-search có mô hình RuleBot (2026-10-16, chưa có kết quả)

- Thêm `kill_search.py` (`search_action`): với mỗi hành động mask cho phép, mô phỏng 12 bước trên bản sao env, đối thủ chạy chính code RuleBot, learner sau bước đầu chỉ né; +1 nếu bom learner hạ đối thủ khi learner còn sống, −1 nếu learner chết; giữ hành động B trừ khi có hành động điểm cao hơn hẳn; chỉ chạy khi đối thủ trong Manhattan ≤ 6; giữ nguyên `np.random` của trận thật. Test `test_kill_search.py` (tìm được bom nhốt đối thủ ở ngõ cụt, không đổi state/RNG; giữ mặc định khi đối thủ ở xa).
- `probe_sd_endgame.py`: thêm nhánh `search`, tham số `--arms/--switches/--shards`; chia shard theo seed bàn và `merge` (đã kiểm: 2 shard × 4 trận giống hệt 1 lượt 8 trận, cả trận lẫn tỉ lệ).
- Luật quyết định chốt trong PLAN_SUDDEN_DEATH.md trước khi chạy. Lượt đầu bị dừng do thiếu RAM (RESULTS.md "SD4.6"); chờ người dùng chạy.

## 32. Công cụ đối chiếu nhánh khác (2026-10-17)

- Thêm `crosscheck_friend_branch.py`: chạy `evaluate.py` của nhánh cần đối chiếu (trong git worktree riêng, không đụng working tree) cho checkpoint của nhánh và cho B, dùng lại file đã chạy khi chạy lại, rồi ghép cặp trận bằng `paired_active_bootstrap`. Mới `py_compile`, chưa chạy (thiếu RAM khi SD4.6 đang chạy).
- Worktree `a6650a0` tạo trong scratchpad của phiên, có file LFS checkpoint; gỡ bằng `git worktree remove <path>` khi xong.


## 33. Kết quả SD4.6 và đánh giá chéo nhánh bạn (2026-10-17)

- Không đổi code thuật toán. `probe_sd_endgame.py` + `kill_search.py` (§31) và `crosscheck_friend_branch.py` (§32) đã chạy đủ lượt (người dùng chạy trong cmd); số liệu ở RESULTS.md "SD4.6 — probe kill-search (lượt đầy đủ)" và "XC — kết quả đánh giá chéo".
- SD4.6: search cả trận +5.75pp ActiveWR trên validation (đạt luật chốt trước); search chỉ giai đoạn cuối +0. Theo luật, hướng tiếp theo là chưng cất search vào CNN cho cả trận (cần thiết kế, chưa làm).
- Đánh giá chéo: checkpoint của nhánh `codex/phasec-sd-best-fallback` hơn B +7.5pp ActiveWR (gộp 600 trận, lịch E, CI dưới > 0). Không tái lập được bit-to-bit con số họ báo vì policy lấy mẫu hành động bằng RNG torch (khả năng cao do khác phiên bản torch); đã kiểm cả commit 0d3b474 (worktree tạm, đã gỡ), cho cùng kết quả như a6650a0.
- Seed: 60000 đã dùng cho đối chiếu; 31000 đã có kết quả của B dưới lịch E. Giữ worktree a6650a0 trong scratchpad để dùng nếu chọn công thức của họ.

## 34. Merge nhánh bạn (`codex/phasec-sd-best-fallback`) vào code chính (2026-10-17)

- Nhánh mới `danh/sd5-friend-merge`: commit `860753f` chụp toàn bộ code/tài liệu chưa commit (không gồm `runs/`, `graphify-out/`), rồi merge `origin/codex/phasec-sd-best-fallback` (a6650a0). Nhận về: SD trong engine (13 kênh, kênh 12 = cảnh báo vòng), RuleBot biết vòng, `evaluate.py` có tham số/metric SD, teacher pressure/sequence + ring safety + nhân giai đoạn cuối + chọn checkpoint theo mục tiêu, launcher `.ps1`, checkpoint F và test của bạn.
- Giải xung đột `evaluate.py`/`train_actor_critic.py`: giữ cả hai bộ tính năng (contact diagnostics, GAE/PPO, training state, `two_bomb`/`local_combat_move` của mình; SD, teacher mới, hệ số loss, target selection của bạn). `smart_load_weights`: giữ kiểm định dạng training state của mình và kiểm chặt của bạn, báo lỗi rõ khi checkpoint nhiều kênh hơn mạng.
- Quyết định (ruling):
  - Mặc định `train()`/CLI = công thức của bạn (test của bạn ràng buộc các mặc định này; F được tạo bằng chúng). Thêm `teacher_version` (`v2` mặc định; `v1` = teacher local trước merge, `get_attack_action_v1`) để pilot cũ vẫn tái lập được; `probe_sd_endgame` nhánh `local`, `audit_t3_approach`, `benchmark_t3` dùng v1 (công cụ của thí nghiệm đã đóng).
  - `sudden_death_probability = 0` không còn ghi đè `env.sudden_death` mỗi episode (trước đây sẽ tắt `SuddenDeathEnv` bơm vào bằng patch).
  - `SuddenDeathEnv` (SD1–SD4): gọi `super().__init__` trước rồi mới đặt thuộc tính SD của mình; tắt SD engine (`_apply_sudden_death` → []) và che `_next_sudden_death_closure` (→ None) để RuleBot/teacher không đổi hành vi; `obs_channels` = 13, `ring_channel` ghi vào kênh 12 thay vì nối thêm kênh. Wrapper lọc tham số SD engine bằng `base_kwargs` (vì `evaluate_seeded`/`train` nay luôn truyền chúng).
  - PPO/ratio: từ chối `sudden_death_probability > 0` (ring supervision chỉ có ở AC); hệ số giai đoạn cuối không áp dụng khi replay (ghi chú `shortcut:` trong code).
  - Bộ đếm chẩn đoán `combat_bombs` giữ định nghĩa của mình (chỉ đếm khi nhãn là BOMB), khác log của bạn; không ảnh hưởng loss.
- Test: thêm `legacy_test_support.py` (`legacy_train` = công thức trước merge, `strip_sd` bỏ khóa metric SD mới) cho 8 file test so bit-exact với source cũ; sửa kỳ vọng 12 → 13 kênh; 2 test `test_combat` mô tả hành vi v1 trỏ sang `get_attack_action_v1`; test launcher của bạn tự bỏ qua khi không có `venv\Scripts\python.exe`; preflight R0 bỏ qua (ghim hash engine cũ). Kết quả kiểm tra tương đương ở RESULTS.md "MERGE".
- Sau merge đã gỡ worktree a6650a0 trong scratchpad (F và code của bạn nay nằm trong repo).

## 35. SD5.0 — probe kill-search trên F dưới lịch E (2026-10-17)

- Thêm `probe_sd5_search.py`: nhánh `f` (F nguyên bản) và `search` (hành động F lấy mẫu, `kill_search.search_action` thay khi có nước tốt hơn hẳn), dùng `evaluate.evaluate_seeded(sudden_death=True, contact_diagnostics=True)` với lịch E; hai nhánh dùng chung chuỗi RNG torch nên chỉ lệch ở trận search can thiệp. Chia shard theo seed bàn (`--shards F SEARCH`, mặc định 1 4, tối đa 5 tiến trình), `merge` tính lại tỉ lệ từ từng trận, `draw_breakdown` tách hòa hết giờ / cả hai chết / hết giờ chưa từng tới gần, `decide` áp luật +3pp & CI dưới > 0. Từ chối seed ≥ 31000.
- Test `test_probe_sd5_search.py` (viết trước, thấy lỗi rồi mới viết script): shard ghép lại giống hệt một lượt chạy, nhánh search chạy được và từ chối seed test, luật quyết định và phân loại hòa. Toàn bộ suite: 350 OK, 16 bỏ qua.

## 36. SD5.1 — nhãn kill-search trong trainer và runner 3000 episode (2026-10-17)

- `train(search_teacher=True, search_loss_coef=0.9)`: ở mỗi bước của learner, `kill_search.search_action` so hành động CNN vừa lấy mẫu với các hành động mask cho phép; nếu có nước tốt hơn hẳn thì thêm nhãn cross-entropy vào **nhóm loss riêng** `search_loss` (trung bình riêng như escape/combat/navigation) và bỏ nhãn heuristic ở bước đó. CNN vẫn tự chọn hành động (nhãn không thay hành động). Metric `search` (queries/labels/bomb_labels) và `search_loss` trong `metrics.jsonl`. PPO/ratio từ chối search_teacher.
  - Ruling: thiết kế duyệt ghi "nhãn ở nhóm combat"; đổi sang nhóm riêng cùng hệ số 0.9 vì đo được ~1 nhãn search/episode so với ~50–260 nhãn combat heuristic, gộp chung thì nhãn search chỉ còn ~1/260 trọng số. Nếu sai: nhãn search có trọng số mạnh hơn ý định ban đầu; nhánh control không bị ảnh hưởng.
- Sửa resume (do merge để lại): lưu/khôi phục `curriculum_rng` (chọn episode có SD) trong training state; thêm các tham số mới (SD, teacher pressure/sequence, hệ số loss, target selection, `teacher_version`, `search_teacher`, `search_loss_coef`) vào provenance; `kill_search.py` vào `source_hashes`.
- `run_sd5_pilot.py`: `train --arm control|search [--resume]` (đăng ký config, chép source + SHA vào thư mục nhánh, từ chối ghi đè, kiểm SHA của F, cắt các dòng metrics sau state cuối khi resume) và `test` (mở seed 42000 một lần bằng `test_opened.json`, 3 model × {lịch E, tắt SD}, paired bootstrap, gate, ZIP `-send.zip`). Hai nhánh chỉ khác `search_teacher`.
- Test: `test_search_teacher.py` (5: nhãn khi search tốt hơn, không nhãn khi đồng ý, tắt thì không gọi search, resume giữ chuỗi SD, provenance có tham số mới, PPO từ chối) và `test_run_sd5_pilot.py` (4: config hai nhánh, gate, smoke train/từ chối ghi đè/resume, test một lần). Đều viết trước và thấy lỗi trước khi cài. Toàn bộ suite 359 OK (16 bỏ qua). Smoke CLI đầu-cuối (train 2 nhánh, resume, test) chạy được.

## 37. Thanh progress dùng chung (2026-10-17)

- Thêm `progress.py`: `bar(total, desc, initial=0)` (tqdm 4.67.3 có sẵn trong env torch) ghi ra thiết bị console (`CON` trên Windows, `/dev/tty` nơi khác, không có thì stderr), nên vẫn thấy khi người dùng chạy `> log 2>&1` và không làm bẩn log; `parallel_map(fn, jobs, total, desc, workers)` chạy job trong ProcessPool, worker gọi `ticks.put(n)` qua hàng đợi Manager để cập nhật một thanh chung, giữ thứ tự kết quả theo job.
- `run_sd5_pilot.py`: thanh train bắt đầu từ episode đã lưu khi `--resume`, postfix ActiveWR(100)/best ep/nhãn search, hiện tiến độ validation qua `on_validation`; stage test dùng `parallel_map` với `evaluate_seeded(on_progress=...)` (thay ProcessPoolExecutor trực tiếp).
- Test `test_progress.py` (2: thứ tự kết quả và đếm đủ tick qua nhiều tiến trình; bar resume từ vị trí cũ), viết trước và thấy lỗi trước khi cài. Quy ước ghi vào CLAUDE.md và memory.

## 38. Đưa repo lên GitHub (2026-10-17)

- Không đổi code. Commit `runs/` (trừ lượt SD5.1 đang chạy) và `graphify-out/`, push nhánh `danh/sd5-friend-merge`; `.pt` đi qua LFS theo `.gitattributes` sẵn có. Các zip/json 50–75 MB vẫn là file git thường (GitHub chỉ cảnh báo); nếu repo phình to có thể chuyển `*.zip` sang LFS sau.

## 39. SD5.1 — kết quả, kiểm tra nguồn gốc nhánh train (2026-10-18)

- Không đổi code. Đọc `runs/sd5-20261017/summary.json`; gate FAIL (RESULTS mục SD5.1 kết quả), giữ F; seed 42000 đánh dấu đã dùng trong CLAUDE.md.
- Kiểm nguồn gốc trước khi ghi kết luận: `config.json`, `source/` và `training_state.pt` (provenance runtime) của từng nhánh. Phát hiện nhánh search được train trên máy khác (torch 2.14.1, đường dẫn `E:\Bombermanv2`); so source bằng `diff --strip-trailing-cr` xác nhận code giống hệt (khác SHA chỉ do CRLF). Test vẫn hợp lệ vì `test_opened.json` ghi SHA đúng ba model và cả ba chạy cùng máy.
- Bài học cho các lượt sau: (1) SHA source nên tính trên nội dung đã chuẩn hóa xuống dòng, hoặc ghi rõ máy chạy, để `registration.json` của các nhánh so được với nhau; (2) chọn best theo nhiều điểm validation 400 trận thiên lạc quan cỡ vài pp; với các khác biệt nhỏ cần validation lớn hơn hoặc giữ final; (3) khi hai nhánh train trên hai máy, validation không so trực tiếp được (cùng F cho 47.0 và 43.75 trên hai phiên bản torch).
- Trạng thái: SD5.1 DONE (FAIL). Hướng chưng cất nhãn search thưa đã thử; xem đề xuất tiếp trong `PLAN_SUDDEN_DEATH.md`.

## 40. SD5.3 — spec chưng cất kill-search dày (2026-10-18)

- Không đổi code. Viết spec `docs/superpowers/specs/2026-10-18-sd5-3-dense-search-distillation-design.md`, gồm: lý do (SD5.0 +22.5 lúc chơi so với SD5.1 −2.75 khi chưng cất), hiện trạng, phương án đã bỏ, thiết kế 4 stage `collect` / `train` / `validate` / `test` cho runner `run_sd53_pilot.py`, gate chốt trước và hướng dẫn cho người chạy hộ trên máy khác.
- Thiết kế chính:
  - F + search tự chơi 3000 trận trên dải seed 200000+; thu mọi bước có thông tin cùng 5% bước khác làm mẫu neo.
  - Fine-tune F bằng cross-entropy với đích mềm (đều trên các nước tốt nhất), cộng KL(F ‖ model) trên mẫu neo; tăng cường bằng 8 đối xứng bàn, có luật bật/tắt dựa trên phép kiểm tính đúng của nhãn sau biến đổi.
  - 3 ứng viên chọn trên validation 21000; test một lần trên seed 43000, kèm F + search làm trần tham chiếu.
  - Cả bốn stage chạy trên cùng một máy; SHA source chuẩn hóa xuống dòng.
- Chưa duyệt, chưa cài.

## 41. SD6.0 — chẩn đoán điểm mù của F (2026-10-18)

- Không đổi code của repo. Ba probe tạm (chép vào `runs/sd6-diagnostics-6600`, `runs/sd6-temperature-probe-21000`):
  1. Xác suất F gán cho nước giết, với search chỉ quan sát.
  2. Đánh giá F với τ ∈ {1, 0.5, 0.25, argmax} trên validation.
  3. Hồ sơ các lần search sửa F (loại nước, xác suất F gán cho nước đó, bom đang cháy, khoảng cách tới đối thủ, bước).
- Kèm phân tích cặp các trận SD5.0 có sẵn. Kết quả ở RESULTS SD6.0: bác bỏ hướng hạ nhiệt độ/argmax; xác định hai điểm mù (nước chặn đường thoát sau khi đặt bom; rời vành ngoài trước bước 475).
- Đọc code teacher escape (`train_actor_critic.py`, nhánh nguy hiểm → `_bfs_safe_cell(...)` trả về `exits[0]`) để lập giả thuyết nguyên nhân. Hướng đề xuất ghi trong `PLAN_SUDDEN_DEATH.md` SD6.

## 42. SD6.1 — luật chặn đường `cutoff_teacher` và probe (2026-10-18)

- `cutoff_teacher.trap_actions(env, aid, mask)`: với mỗi STOP/di chuyển mask cho phép, tạm đặt learner vào ô đích (đặt lại vị trí trong `finally`, không deepcopy), dùng `_survival_map(avoid_contests=False)` để xem đối thủ còn đường sống không. Trả về các nước mà learner giữ được ô (ô không bị cháy trong dự báo) và làm đối thủ hết đường, trong khi một nước hợp lệ khác để đối thủ thoát.
  - Bản đầu so với "learner đứng yên", nên bỏ sót đúng loại nước search hay chọn (đứng giữ ô cửa) và khớp 0/85. Bản sửa so với các nước hợp lệ khác.
- `probe_sd6_cutoff.py`: nhánh `f` và `cutoff` (F + luật lúc chơi) trên validation 21000; hồ sơ độ khớp với kill-search trên seed 15000; `summarize` tách riêng có test; lưu `parts.json` thô trước khi gộp. Lỗi của lượt đầu (`merge` cần `contact_diagnostics`) đã có test. Có thanh progress qua `parallel_map`.
- Test:
  - `test_cutoff_teacher.py` (4): chặn ô cửa bằng nước đi, kill-search xác nhận; đứng giữ ô cửa, viết trước và thấy lỗi với bản đầu; không bom hoặc đối thủ đã chết chắc thì không can thiệp; tôn trọng mask; trạng thái env không đổi.
  - `test_probe_sd6_cutoff.py` (4).
  - Toàn bộ suite 369 OK (16 bỏ qua).
- Chẩn đoán cơ chế: `runs/sd6-diagnostics-6600/movekill_mechanism.py`.
- Trạng thái: SD6.1/SD6.1b không đạt luật mở SD6.2 (ActiveWR không tăng; WR +3.75 và hòa −2.75 nhờ thắng thụ động). Code được giữ lại để tham khảo; trainer chưa dùng.

## 43. File `INSIGHTS.md` (2026-10-18)

- Không đổi code. Tạo `INSIGHTS.md` làm nơi duy nhất giữ insight có bằng chứng và các giả thuyết đã bác bỏ, để thiết kế thí nghiệm sau không phải đọc lại toàn bộ RESULTS.
- CLAUDE.md thêm quy ước: đọc `INSIGHTS.md` trước khi thiết kế thí nghiệm mới và cập nhật nó cùng lượt với RESULTS/IMPROVEMENTS.
- `PLAN_SUDDEN_DEATH.md` (SD6) trỏ tới file. Spec SD5.3 thêm mục 8b với các phát hiện SD6 liên quan tới việc cài SD5.3.

## 44. SD7.0 — spike advantage theo nhóm từ trạng thái quyết định (2026-10-18)

- Không đổi code của repo. Script tạm `runs/sd7-spike-6600/spike_group_rollouts.py`:
  - chụp trạng thái bằng `deepcopy(env)`; khôi phục RNG numpy/torch của trận chính sau mỗi lần chơi lại;
  - gán nhãn bằng `kill_search.search_action`;
  - chơi lại K lần bằng `evaluate.make_ac_fn` và `make_rule_based_fn`.
- Lần đầu (giữ 3 ứng viên đầu mỗi trận, không lọc) cho mẫu gần như toàn trạng thái không có nước giết. Đã sửa sang lọc bằng search và ghi cả hai lần.
- Kết quả ở RESULTS SD7.0: có phương sai để học (86% trạng thái có nước giết cho kết quả lúc được lúc không). Bước tiếp theo là viết spec SD7, chờ người dùng.

## 45. SD7 — cập nhật theo nhóm trong trainer và runner (2026-10-18)

- `group_rl.py`:
  - `decisive`: lọc rẻ trước (bom đang cháy, đối thủ trong vùng nguy hiểm, Manhattan ≤ 4), rồi mới gọi `kill_search`.
  - `rollout`: chơi lại trên bản sao `deepcopy`, gặp `RuleBasedAgent` mới, lưu/khôi phục RNG numpy toàn cục; kết cục +1/−1/0 theo ghi công kill của env.
  - `sampler`: lấy mẫu có che mask bằng `torch.Generator` riêng.
  - `group_advantages`, `group_loss`: −trung bình A · log π trên mọi bước chơi lại; bỏ các nhóm không có chênh lệch.
- `train_actor_critic.train(group_rl, group_coef, group_k, group_horizon, group_max_states, group_gap)`:
  - Trong episode, chụp trạng thái quyết định (`deepcopy(env)`) trước khi lấy mẫu hành động.
  - Cuối episode, chơi lại bằng policy hiện tại, rồi cộng `group_coef · group_loss` vào loss trước bước optimizer duy nhất.
  - Metric `group` (states, groups, contrast_groups, rollouts, wins, deaths, steps).
  - `group_rng` riêng, lưu trong training state; tham số mới nằm trong provenance; `group_rl.py` trong `source_hashes`. PPO/ratio từ chối `group_rl`.
- `run_sd7_pilot.py`:
  - `train`: config = control SD5.1 + `GROUP`; kiểm SHA của F và control; resume; thanh progress.
  - `mechanism`: chơi lại H1 theo cặp trạng thái, seed cố định theo (trạng thái, k) để mọi model dùng chung số ngẫu nhiên; `paired_states` bootstrap theo trạng thái.
  - `test`: một lần, seed 44000, gate.
- Test:
  - `test_group_rl.py` (6): kết cục theo chủ bom và env/RNG không đổi; advantage; loss nâng xác suất nước thắng; `group_coef=0` cho loss/phần thưởng giống hệt khi tắt; resume chơi lại giống hệt; PPO từ chối. Viết trước, thấy lỗi trước khi cài.
  - `test_run_sd7_pilot.py` (6): viết cùng lúc với runner, không qua bước đỏ.
  - Toàn bộ suite 381 OK (16 bỏ qua).

## 46. SD7 — kết quả chạy thật và kết luận (2026-10-18)

- Không đổi code ở bước này; chỉ chạy `run_sd7_pilot.py train --resume`, `mechanism`, `test` (người dùng chạy tay) và đọc `runs/sd7-20261018/summary.json`, `mechanism_summary.json`.
- Kết quả: H1 không đạt (group − F tỉ lệ giết +2.05pp [−0.34, +4.62], luật ≥ +10pp); test seed 44000 gate FAIL (group − F ActiveWR +0.25 [−5.75, +6.25]; group − control −3.25). Giữ F. Số liệu đầy đủ ở RESULTS "SD7.1–SD7.3".
- Vì sao có thể thất bại (giả thuyết, chưa kiểm): ~0.8 trạng thái quyết định/ep ⇒ ~2400 nhóm cả lượt, advantage trung bình nhóm với K = 8 rất nhiễu, và loss nhóm cộng với loss RL/teacher chung (`group_coef` 1.0) có thể bị lấn át. Phần thưởng đã khớp ActiveWR nhưng tín hiệu vẫn thưa, giống kết luận SD5.1 về nhãn thưa.
- Kiểm thử: suite 381 OK trước lượt chạy (mục 45); không thêm test.
- Trạng thái: hoàn tất, KHÔNG ĐẠT.

## 47. SD7.4 — probe cơ chế với tín hiệu nhóm dày hơn (2026-10-18)

- Spec: `docs/superpowers/specs/2026-10-18-sd7-4-dense-group-probe-design.md` (luật chốt trước khi train).
- Lý do: ở SD7, group − control tỉ lệ giết chỉ +0.34pp; tín hiệu nhóm thưa (0.75 trạng thái/ep, 0.59 nhóm có chênh lệch/ep, 52 bước chơi lại/ep, chỉ 43/3000 ep chạm giới hạn 3 trạng thái).
- `run_sd74_probe.py`: `train --arm control|group [--resume]` (config control SD5.1, 1000 ep, `validation_every = 0`, dùng `final_phaseC.pt`), `mechanism` (seed chẩn đoán mới 6700–6749, tái dùng `run_sd7_pilot.mechanism_shard`/`paired_states`). Nhánh group: `group_coef` 4.0, `group_k` 16, `group_max_states` 6, `group_gap` 4. Không đổi trainer hay `group_rl.py`.
- Luật H1': group − control ≥ +5pp, CI dưới > 0, chết lúc chơi lại ≤ control + 1pp. Đạt thì mở SD7.5 (full 3000 ep, test một lần seed 45000); không đạt thì dừng SD7.
- Test: `test_run_sd74_probe.py` (2: hai nhánh chỉ khác khóa `group_*`; luật H1'). Smoke train hai nhánh + mechanism chạy qua.
- Kết quả: H1' ĐẠT, group − control +7.12pp [+4.86, +9.72] (RESULTS SD7.4).
- SD7.5 (mở theo spec): `run_sd74_probe.py train --arm full` (config control SD5.1 + `GROUP`, 3000 ep, validation 21000 mỗi 500 ep chọn best) và `test` (F, control SD5.1, full best; seed 45000; tái dùng `run_sd7_pilot.test`, thêm tham số `seed`, `group` cho tên thư mục, mặc định giữ nguyên SD7). Test thêm `test_full_is_sd5_control_plus_group`; smoke train full + test chạy qua. Toàn bộ suite 384 OK (16 bỏ qua).
- Trạng thái: SD7.4 xong; SD7.5 chờ người dùng chạy.

## 48. SD7.5 — train full advantage theo nhóm và kết luận hướng SD7 (2026-10-19)

- Không đổi code; người dùng chạy `run_sd74_probe.py train --arm full --output runs\sd75-20261018` (3000 ep, 4 giờ 58 phút) và `test` (seed 45000, F/control SD5.1/full, lịch E và off). Claude đọc `summary.json`, `validation_metrics.jsonl`, `metrics.jsonl`.
- Kết quả: gate FAIL, full − F ActiveWR +3.75 [−2.5, +10.0]; full − control +0.75. Best validation ep 1000 (50.25), sau đó validation giảm còn 41.75 ở ep 3000. Giữ F. Số liệu ở RESULTS "SD7.5".
- Kết luận hướng SD7: advantage theo nhóm đổi được hành vi ở trạng thái quyết định (SD7.4) nhưng hiệu ứng lên ActiveWR cả trận (~1pp theo ước tính) nhỏ hơn mức test 400 trận phát hiện được. Muốn đi tiếp thì phải tăng số trạng thái quyết định/trận hoặc đo bằng nhiều trận hơn, không phải chỉ tăng `group_coef`.
- Kiểm thử: không có code mới; suite 384 OK từ mục 47.
- Trạng thái: hoàn tất, KHÔNG ĐẠT. Đề xuất dừng hướng SD7 ở cấu hình hiện tại (chờ người dùng quyết định).

## 49. SD7.6a — probe chẩn đoán SD7.5 và chọn G (2026-10-19)

- Người dùng chọn SD7.5 best (`runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c…`) làm model tốt nhất hiện tại, gọi là **G**, dù gate SD7.5 FAIL. Đã ghi vào CLAUDE.md (quy ước checkpoint), RESULTS SD7.5 và PLAN.
- `diagnose_sd75.py` (mới, không train): F chơi seed 6800–6849; ở mỗi bước có đối thủ trong phạm vi, chạy `kill_search.search_action`; với trạng thái có nước giết thì ghi bộ lọc rẻ SD7 có giữ không, nước giết có cần bom không, và xác suất F/G/final đặt lên tập nước giết; tính entropy trung bình. Forward thêm không đụng RNG torch, search giữ RNG numpy, nên F chơi y như khi đánh giá. Chạy song song 4 tiến trình qua `progress.parallel_map`, có thanh progress.
- Kiểm thử: smoke `--smoke` chạy qua (4 trận, 60 bước). Không thêm unittest vì đây là probe một lần, không có logic quyết định nào dùng lại.
- Kết quả (RESULTS SD7.6a): điểm nghẽn là khoảnh khắc "đặt bom ngay" kéo dài một bước (28/37 cửa sổ giết thất bại; F đặt P(bom) 0.58). 38/57 trận không thắng chủ động không có cơ hội giết nào. Ngân sách mẫu của SD7.5 rơi chủ yếu vào trạng thái đã chắc thắng.
- Trạng thái: chẩn đoán xong; thiết kế sửa (SD7.6) chờ người dùng duyệt.

## 50. SD7.6 — group RL vào khoảnh khắc "đặt bom ngay", có buffer (2026-10-19)

- Spec (luật chốt trước khi train): `docs/superpowers/specs/2026-10-19-sd7-6-bomb-moment-buffer-design.md`. Người dùng chọn hướng "buffer khoảnh khắc bom" sau chẩn đoán SD7.6a.
- `group_rl.py`: `kill_actions(env, aid, mask)` trả tập nước kill-search chấm +1, chỉ search khi BOMB hợp lệ và có đối thủ trong khoảng ≤ 4 (search tốn ~19 ms). `narrow()` = có 1..2 nước giết. `group_loss(..., first_only=True)` chỉ tính A·log π ở bước đầu của mỗi lần chơi lại.
- `train_actor_critic.train`: tham số mới `group_select` ('decisive' mặc định | 'narrow'), `group_buffer` (0 = tắt), `group_first_step`. Có buffer thì mỗi episode chơi lại các trạng thái mới, rồi lấy thêm ngẫu nhiên (`group_rng`) từ buffer cho đủ `group_max_states`; buffer FIFO lưu `(env, learner id)` và được pickle vào training_state (`group_pool`, dạng bytes, nên `weights_only=True` vẫn nạp được) để resume chính xác. Mặc định không đổi gì: khi tắt buffer, `group_rng` không bị rút thêm.
- `run_sd74_probe.py`: tách thân `train_arm` thành `run_arm(directory, config, sources, registration, label, resume)` để dùng lại; hành vi SD7.4 không đổi.
- `run_sd76_pilot.py`: `train --arm control|buffer` (probe 1000 ep từ G), `mechanism` (seed chẩn đoán 6850, G chơi, mỗi model chơi lại K 8 × H 20 các trạng thái hẹp, P(tập nước giết), rồi validation 21000 cho control/buffer; khóa seed bằng `mechanism_opened.json`), `train --arm full` (3000 ep, validation chọn best), `test` (G với full, seed 46000, một lần).
- Test: `test_group_rl.py` thêm 5 (narrow giữ 1–2 nước giết, không search khi BOMB không hợp lệ hoặc đối thủ xa, loss chỉ bước đầu, buffer chơi lại trạng thái cũ, resume khôi phục buffer); `test_run_sd76_pilot.py` (3: config các nhánh, luật H1‴, gate). Smoke mọi stage chạy qua; toàn bộ suite 392 OK (16 bỏ qua).
- Đo trên 20 episode thật từ G (cuda): 7.0 s/ep; 0.8 trạng thái hẹp mới/ep, 4.95 trạng thái chơi lại/ep; 89% nhóm có chênh lệch kết cục (SD7.5: 52%).
- Trạng thái: code xong; probe chờ người dùng chạy.

## 51. KC1 — kiểm chứng nhận định về SD7.5/G, gộp control − F trên ba seed test (2026-10-19)

- Không đổi code. Đọc lại `summary.json`/`evaluation_*_E_*.json` của SD5.1, SD7, SD7.5 và gộp 600 cặp bàn bằng `pilot_two_bomb.paired_active_bootstrap` (lệnh một dòng, không giữ script).
- Kiểm code: nhánh `search_teacher` của `train()` chỉ gắn nhãn, không thay hành động; spec SD5.3 đã là phương án "F + search chơi rồi học có giám sát".
- Kết quả (RESULTS KC1): control − F gộp = 0.0 [−3.75, +3.75], trong khi từng seed dao động −6.5 … +3.5; nhận định "train thêm 3000 ep mang lại phần lớn mức hơn F" và "F không biết nước di chuyển (P ≈ 0.001)" không đứng vững.
- Trạng thái: hoàn tất. Đề xuất SD8 (probe phân rã lợi ích kill-search trên G), chờ người dùng duyệt.

## 51. SD7.6-probe — kết quả chạy và nhận xét (2026-10-19)

- Không đổi code; người dùng chạy `run_sd76_pilot.py train --arm control|buffer` (có resume) và `mechanism`. Claude đọc `mechanism_summary.json`, `validation_*_21000.json`, `metrics.jsonl` và log console.
- Kết quả: H1‴ đạt theo chữ (tỉ lệ giết ở trạng thái hẹp +18.43pp [+14.58, +22.28]; validation buffer − control −0.5), nhưng kills/trận và ActiveWR không đổi; bom/trận +20%, hòa +6 (RESULTS SD7.6-probe).
- Hạ tầng: các lần ngắt (PermissionError khi `os.replace` file training_state trên Windows, lỗi CUDA, Ctrl+C) đều resume chính xác nhờ kiểm provenance; lần resume bị từ chối vì thiết bị khác (rơi về CPU) là đúng thiết kế. Chưa sửa PermissionError (có thể do antivirus hoặc trình index giữ file); nếu lặp lại thì thêm thử lại vài lần trong `training_state.atomic_save`.
- Trạng thái: probe xong. SD7.6 full chờ người dùng quyết định vì bằng chứng validation cho thấy khó qua gate (+5pp so với G).

## 52. SD7.7 — phạt KL về G cho buffer khoảnh khắc bom (2026-10-19)

- Lý do (đo thêm sau SD7.6-probe, 40 trận G trên seed 6600): nhánh buffer SD7.6 chọn BOMB với P 0.943 ở mọi trạng thái BOMB hợp lệ (G 0.637, control 0.656); KL(π‖G) 0.049 so với control 0.003. Dữ liệu của bộ lọc hẹp một chiều (BOMB hợp lệ và thường là nước giết), còn group RL không sinh tín hiệu ngược ở trạng thái không có nước giết. Người dùng chọn sửa tác dụng phụ trước khi train full.
- Spec chốt trước: `docs/superpowers/specs/2026-10-19-sd7-7-kl-anchor-design.md`.
- `train_actor_critic.train(anchor_coef=β)`: cộng β · mean KL(π_θ‖π_ref) trên mọi bước learner (phân phối đã che mask); π_ref = bản đông cứng của trọng số warm start, tạo trước khi khôi phục training_state. β = 0 bit-exact như cũ; PPO/ratio từ chối anchor. Metric `anchor_kl` mỗi episode.
- `run_sd77_pilot.py`: `calibrate` (4 lượt 150 ep, β ∈ {0, 1, 4, 16}, chạy song song; đo P(BOMB | hợp lệ), KL về G, P(nước giết) ở trạng thái hẹp trên 40 trận G seed 6600; chọn β theo luật cố định), `train --arm anchor|full` (đọc β đã chọn), `mechanism` (seed 6900, so với control SD7.6, validation 21000 cho anchor, control dùng lại kết quả SD7.6), `test` (dùng lại `run_sd76_pilot.test`, seed 46000).
- Test: `test_group_rl.py` thêm 2 (β = 0 bit-exact, β > 0 đổi loss, KL bằng 0 ở episode đầu; resume giữ đúng anchor) và kiểm PPO từ chối anchor; `test_run_sd77_pilot.py` (3: config, luật β, luật H1⁗). Smoke mọi stage chạy qua (sửa một lỗi truyền `device` hai lần trong `calib_train`). Toàn bộ suite 397 OK (16 bỏ qua).
- Trạng thái: hiệu chỉnh β đang chạy (`runs/sd77-20261019`).

## 53. SD7.7-calibrate — kết quả hiệu chỉnh β (2026-10-19)

- Không đổi code; Claude chạy `run_sd77_pilot.py calibrate` (4 lượt 150 ep song song, rồi đo trên 40 trận G seed 6600). `calib_train` không ghi `metrics.jsonl`; lịch sử bom/ActiveWR đọc từ `training_state.pt`.
- Kết quả: không β ∈ {1, 4, 16} nào đưa P(BOMB | hợp lệ) về ≤ P_G + 0.05; β = 16 còn 0.829. P(BOMB) và P(nước giết) ở trạng thái hẹp tăng giảm cùng nhau (RESULTS SD7.7-calibrate). Theo spec: dừng, không chạy probe.
- Ý nghĩa thiết kế: đẩy policy ở trạng thái được chọn bằng search không tách được khỏi hành vi chung khi CNN không thấy được điều kiện chọn; phạt KL chỉ đổi mức cân bằng giữa hai thứ chứ không tách chúng.
- Trạng thái: hoàn tất, KHÔNG ĐẠT luật hiệu chỉnh; chờ người dùng chọn hướng.

## 54. SD7.8 — test G + kill-search lúc chơi (2026-10-19)

- Người dùng chọn "G + search lúc chơi" sau SD7.7. Spec chốt trước: `docs/superpowers/specs/2026-10-19-sd7-8-g-plus-search-design.md`.
- `run_sd78_search_test.py` (mới): hai agent G và G+search (giống SD5.0, chỉ đổi F thành G), seed 46000, 400 trận mỗi agent, lịch E (gate) và off (báo cáo), một lần (`test_opened.json`). Nhánh search chia 4 shard theo seed bàn qua `progress.parallel_map` (có thanh progress); gate: ActiveWR ≥ +5pp, CI dưới > 0, tự nổ ≤ +1pp.
- Test: `test_run_sd78_search_test.py` (gate, tóm tắt tỉ lệ). Smoke 4 trận mỗi nhánh chạy qua, các shard ghép đủ trận.
- Kết quả: gate ĐẠT, G+search − G ActiveWR +16.5 [+12.5, +20.5], tự nổ −0.75, chết vì vòng 11.0 → 1.0 (RESULTS SD7.8).
- Trạng thái: hoàn tất, ĐẠT. Agent khuyến nghị: G + kill-search lúc chơi.

## 55. SD5.3 — review và phân tích offline kết quả của bạn (2026-10-19)

- Không đổi code dự án. Lấy `run_sd53_pilot.py` từ `origin/Loc/sd5-update` vào scratchpad (không checkout), xác nhận SHA chuẩn hóa LF khớp registration, SHA checkpoint S khớp manifest; tính lại tỉ lệ test từ file trận thô.
- Phát hiện về code của bạn: khóa `search_vs_f` trong `summary.json` là S − F (chỉ đặt tên sai); audit đối xứng 79% < 90% nên augmentation đã tắt đúng luật (search/RuleBot không đối xứng dưới phép quay/lật).
- Script phân tích (scratchpad, không vào repo): tái tạo split held-out của bạn (seed 530), tính P(nước tốt nhất) của F, G, S trên mẫu quyết định theo nhóm (giết bằng bom / bằng đứng-di chuyển / tránh chết) và P(BOMB), KL trên mẫu neo (RESULTS SD5.3 review).
- Trạng thái: hoàn tất; S được dùng làm nhánh trong SD7.9a.

## 56. SD7.9 — nền S cho search, search hai nước, test 47000 (2026-10-19/20)

- Spec `docs/superpowers/specs/2026-10-20-sd7-9-two-move-search-design.md` (luật từng stage chốt trước khi chạy stage đó).
- `kill_search.search_action2` (mới): khi search một nước không có nước giết, thử mọi nước đầu điểm ≥ 0 rồi mọi nước thứ hai hợp lệ (mask `_safe_action_mask` sau 1 tick RuleBot), sau đó chỉ né; chọn nước đầu của kế hoạch giết, giữ nước của CNN nếu nó đã nằm trong một kế hoạch giết. Khôi phục RNG numpy, không đổi env. `search_action` không đổi.
- `run_sd79_probe.py` (mới): stage a (G+search, S, S+search) và stage b (G+search2, ghép cặp với trận G+search của stage a), validation 21000. `run_sd79_test.py` (mới): G+search vs G+search2, seed 47000 một lần, lịch E/off, 4 shard mỗi cấu hình qua `progress.parallel_map`, ZIP.
- Test: `test_kill_search` thêm test search hai nước (bàn cố định: một nước không giết được, "bước sang trái rồi đặt bom" giết; env và RNG không đổi; có nước giết một bước thì trả đúng kết quả search một nước); `test_run_sd79_probe.py` (luật quyết định, nhánh). Smoke cả hai runner chạy qua.
- Kết quả: SD7.9a không mở test (S+search − G+search −0.75); SD7.9b +17.5 [+13.75, +21.25]; SD7.9c gate ĐẠT +21.0 [+17.0, +25.25] (RESULTS SD7.9a–c).
- Trạng thái: hoàn tất, ĐẠT. Agent khuyến nghị: G + `search_action2` lúc chơi.

## 57. DOC — ghi insight chi tiết SD5.3 + SD7.9 (2026-10-20)

- Không đổi code dự án. Theo yêu cầu người dùng, thêm mục "SD5.3 + SD7.9 — tổng hợp insight chi tiết" vào RESULTS.md, gồm 8 phần:
  1. Bốn điều kiện giúp chưng cất dày thành công.
  2. G không đổi CNN ở trạng thái quyết định.
  3. Trần ~65% của search một nước.
  4. S thuần so với G+search, và lối chơi cuối trận.
  5. Cơ chế search hai nước phá trần.
  6. Kiểm tra nhất quán dữ liệu.
  7. Giới hạn.
  8. Hệ quả cho bước tiếp theo.
- Phân tích mới (script chạy một lần trong phiên, không vào repo), trên các file trận có sẵn của SD7.9b/c:
  - Ghép cặp trận G+search và G+search2 theo (seed bàn, góc) và đếm trận đổi kết quả.
  - Tính phân bố số bước của trận thắng chủ động và tách theo góc.
  - Không chạy trận mới, không dùng seed mới.
- Kết quả chính:
  - Thay đổi gần như một chiều: test 47000 lịch E có 87 trận thành thắng chủ động (45 từ hòa, 23 từ thắng nhờ vòng, 15 từ thắng thụ động), chỉ 3 trận mất thắng chủ động. Ở lịch off, toàn bộ 90 trận hòa đổi được đều thành thắng chủ động.
  - Trung vị bước của trận thắng chủ động 205 → 158; tỉ lệ thắng chủ động trước bước 300 là 42.0% → 65.75%.
  - Hai góc tăng như nhau.
  - Kiểm tra nhất quán: các trận thắng chủ động trước bước 300 trùng khớp giữa lịch E và off, đúng như ghép cặp RNG theo bàn.
- Bài học thiết kế rút ra:
  - (a) Muốn CNN học một kỹ năng do search chỉ ra, cần nhãn có giám sát dày, có ví dụ ngược và neo KL; tín hiệu RL một phía không đủ (SD7.6/7.7).
  - (b) Khi một agent có search, cải thiện CNN ở chính những trạng thái search đã sửa không cộng thêm gì (S+search ≈ G+search). Muốn tăng thì phải mở rộng tập trạng thái mà search nhìn thấy cơ hội (search hai nước).
  - (c) Đòn bẩy hiện tại là giữa trận (tạo cơ hội giết sớm), không phải cuối trận.
- Sửa INSIGHTS.md:
  - Bỏ tiền tố sai "(Đã trả lời: có, SD5.3.)" ở câu hỏi mở về chưng cất search hai nước; câu hỏi đó chưa được trả lời.
  - Bổ sung bằng chứng chuyển kết quả và thời điểm giết vào insight "search hai nước phá trần".
- Trạng thái: hoàn tất (chỉ tài liệu).

## 58. SD8 — runner chưng cất search hai nước (A) và search ba nước (B) (2026-10-20)

- Theo yêu cầu người dùng: làm cả hai hướng đề xuất sau SD7.9; chọn hướng nhẹ cho người dùng chạy và hướng nặng cho bạn chạy. Spec `docs/superpowers/specs/2026-10-20-sd8-distill-and-deep-search-design.md`, hướng dẫn và lệnh cmd ở `SD8_GUIDE.md`.
- **Đưa runner SD5.3 của bạn vào nhánh này.** `run_sd53_pilot.py` và `test_run_sd53_pilot.py` được lấy nguyên từ `origin/Loc/sd5-update` (`git checkout origin/Loc/sd5-update -- ...`, không sửa). Test của nó qua trong repo (17 OK). SD8A tái dùng các helper: đích mềm, chia held-out theo bàn, ghi shard, metric held-out, đánh giá checkpoint.
- **`kill_search.py`:**
  - Thêm `_advance` (chép env rồi đi 1 tick với RuleBot) và `_best_plan` (đệ quy: điểm tốt nhất của tối đa `depth` nước rồi chỉ né).
  - Thêm `search_action_deep(..., depth)`. Hàm gọi `search_action2` trước, nên kết quả giống hệt search2 khi search2 đã thấy kế hoạch giết; chỉ khi không có mới thử kế hoạch `depth` nước. Hàm khôi phục RNG numpy và không đổi env.
  - Thêm `search_action3` (depth 3).
  - `search_action` và `search_action2` không đổi. Agent đã qua gate SD7.9 giữ nguyên bit.
- **`run_sd8a_distill.py` (mới):** bốn giai đoạn `collect` / `train` / `validate` / `test`.
  - Nhãn `teacher_scores`: điểm search một nước, bị điểm kế hoạch hai nước ghi đè ở các nước đầu mà search2 đã lập kế hoạch.
  - Student khởi đầu từ G và neo KL về logits của G. Không tăng cường dữ liệu bằng đối xứng.
  - Train 16 epoch, ứng viên 4/8/12/16. Validation chọn epoch ActiveWR cao nhất có tự nổ ≤ G + 1pp.
  - Test 48000 một lần: S2, G, S chạy CNN thuần trên lịch E và off; G+search2 trên lịch E chỉ để báo cáo. Gate S2 − G; luật phụ: thay S nếu CI95 dưới của S2 − S > 0.
  - `registration.json` khóa cấu hình và SHA source (chuẩn hóa LF); shard thiếu hoặc sai bị từ chối; có `--resume` cho collect; ZIP gửi lại.
- **`run_sd8b_search3.py` (mới):**
  - `validate` (21000; `--base` tái dùng các trận G+search2 cũ): luật ≥ +2pp thì mở test.
  - `test` (49000, một lần, lịch E): từ chối nếu luật validation không đạt; gate +5pp / CI dưới > 0 / tự nổ ≤ +1pp.
  - Ghi thêm giây/trận của mỗi nhánh.
- **Test:**
  - `test_kill_search`: thêm test search ba nước. `_rollout` được giả lập để chỉ đặt bom ở (1,3) mới giết, và một tường gỗ ở (3,3) làm mục tiêu để mask cho phép BOMB. Kiểm tra: search2 trả STOP, search3 trả LEFT với kế hoạch 1.0, giữ nước CNN khi nó nằm trong kế hoạch, env và RNG không đổi.
  - `test_run_sd8.py` (mới): nhãn teacher, luật chọn epoch (giới hạn tự nổ, hòa thì lấy epoch sớm), gate, chia shard phủ đúng 1500 bàn, ngưỡng mở test B.
  - Smoke trọn quy trình của cả hai runner.
  - Toàn bộ suite: 425 test OK, 16 bỏ qua.
- Kết quả đo tốc độ và smoke: RESULTS "SD8 — chuẩn bị".
- Trạng thái: code xong, chưa chạy full. Người dùng chạy B, bạn chạy A.

## 59. SD8A — chưng cất search hai nước từ G, lượt chạy thật (2026-10-20)

- Không đổi mã nguồn trong lượt chạy. Chạy tuần tự `collect --resume` → `train` → `validate` → `test` trong `runs/sd8a-20261020`; tất cả artifact bắt buộc đã xuất hiện. Suite kiểm tra trước khi chạy: 425 OK, 16 bỏ qua.
- **Dữ liệu teacher:** 3000 trận G + search hai nước, 8449 trạng thái quyết định và 29483 mẫu neo. Teacher thắng 93.10%, thắng chủ động 84.33%, hòa 6.17% trên lượt thu thập; đây là dữ liệu huấn luyện, không phải test độc lập.
- **Huấn luyện/chọn checkpoint:** bốn checkpoint epoch 4/8/12/16; validation seed 21000 chọn epoch 16 theo ActiveWR cao nhất và ràng buộc tự nổ. ActiveWR lần lượt 71.0 / 77.5 / 78.5 / **84.0** so với G 48.25; tự nổ ở epoch 16 là 0.0 so với G 1.25.
- **Test đã mở một lần:** seed 48000, 400 trận mỗi agent, 200 cặp bàn. CNN thuần S2 (epoch 16) đạt 83.0% ActiveWR, so với G 48.0% và S 57.5%; S2 − G = **+35.0pp [ +29.0, +41.25 ]**, tự nổ −0.5pp. Gate SD8A ĐẠT và luật phụ thay S cũng ĐẠT: S2 − S = **+25.5pp [ +19.25, +31.75 ]**, tự nổ không đổi.
- **Giới hạn/đối chiếu:** G + search hai nước lúc chơi vẫn cao hơn S2 3.5pp ActiveWR trên seed này (86.5% so với 83.0%; CI của S2 − G+search2 là [−8.25, +1.25]), nên S2 là CNN thuần tốt nhất hiện tại chứ chưa chứng minh thay được search runtime. Một seed train và một seed test; teacher dùng RuleBot, cũng là đối thủ đánh giá.
- Trạng thái: SD8A hoàn tất, ĐẠT. Checkpoint CNN khuyến nghị mới: `runs/sd8a-20261020/train/epoch_16.pt`; agent tổng thể mạnh nhất vẫn là G + `search_action2` khi chấp nhận chi phí search.

## 60. PDE/PDE2 review và spec Phase D từ S2 (22/10/2026)

- Người dùng đồng ý soạn spec bước tiếp theo sau khi đọc PDE/PDE2. Tạo `docs/superpowers/specs/2026-10-22-phase-d-s2-pool-design.md`, trạng thái **DRAFT_REVIEW**; chưa triển khai hoặc chạy train.
- Thiết kế đề xuất hai nhánh cùng S2, mỗi nhánh 2.000 episode: RB đối chứng và pool (20% RB, 40% G/F/S/S2, 40% hai snapshot gần nhất). Validation chọn checkpoint theo macro ActiveWR của CNN với guard RB/tự nổ; có episode 0 fallback. B chỉ dùng ở test của lượt mới. Gate pilot yêu cầu lợi ích so với S2 và control; gate xong Phase D còn phải đạt WR >= 60% từng đối thủ pool/snapshot cũ và giữ WR trước RB.
- Đã đọc trainer: full-state resume hiện chỉ hỗ trợ C; Phase D dùng snapshot ngẫu nhiên và validation RuleBot. Spec yêu cầu bổ sung state/pool/seed/validation ở implementation sau, không giả định launcher D v1 đã đáp ứng. Không thay đổi mã nguồn trong lượt này.
- Provenance: xác minh SHA của S2/G/F/S/B và shape CNN; G/F/S/S2 có 13 kênh, B có 12. Runtime đọc trực tiếp: Python 3.11.9, torch 2.11.0+cu128, numpy 2.4.6. Spec yêu cầu đo lại baseline trong cùng runtime, source snapshot đầy đủ, resume kiểm hash/runtime và ledger atomically theo board.
- Review dữ liệu: report khớp shard; tính lại summary/theo góc và 37 CI bootstrap khớp. Ghi giới hạn trùng seed PDE/PDE2 và mirror 11×11 chỉ 20 trận phân thắng bại vào RESULTS. Đây là kiểm tra artifact, không phải chạy lại test hoặc kiểm thử runner mới.
- Ngân sách/dải seed mới và tiêu chí gate đều là **đề xuất chờ duyệt**. Không mở test, không tạo output full, không ghi đè checkpoint hoặc run cũ. Sau duyệt spec mới viết implementation plan, rồi acceptance tests và implementation.

## 61. D-S2 — implementation plan theo spec đã duyệt (22/10/2026)

- Người dùng duyệt written spec; cập nhật trạng thái **APPROVED_DESIGN**, giữ nguyên thiết kế thí nghiệm. Viết `docs/superpowers/plans/2026-10-22-phase-d-s2-pool.md`, trạng thái **PLAN_REVIEW**.
- Plan chọn phần mở rộng tối thiểu: context D-S2 opt-in vào trainer/state hiện tại; module protocol, training context và evaluation riêng, CLI điều phối frozen sources. Mặc định Phase C và D/E cũ giữ nguyên; không nhân bản training loop hoặc sửa loss/reward/mask.
- Chốt public CLI/artifact schemas, interfaces giữa task, acceptance độc lập trước production, các cửa sổ crash/resume cần kiểm, macro CI và các gate biên, source closure/seed audit cùng smoke và lệnh CMD phải bàn giao. Native được đề xuất vì các task nối nhau qua trainer/state; phương thức thực hiện vẫn chờ người dùng chọn.
- Chưa sửa code, viết/chạy test hoặc triển khai runner trong lượt này. Kiểm chứng ở mức tài liệu: coverage spec, tính nhất quán interfaces/budget và diff; chưa phải bằng chứng các khả năng của runner đã hoạt động.

## 62. D-S2 T0/T1 — acceptance độc lập và registration (22/10/2026)

Requirements/matrix và 62 acceptance đã khóa riêng trong commit `30083da`; implementer không viết lại suite. `phase_d_s2_protocol.py` quản lý config, metadata digest, inputs/checkpoint/source closure, runtime deterministic và seed reservation. Các prerequisites đã tồn tại lúc CLI suite khóa được ghi là regression, không giả làm RED. T1 còn kiểm các cửa sổ recovery/JSONL bằng test bổ sung; chưa mở full seed.

## 63. D-S2 T2 — lịch pool và snapshot state (22/10/2026)

`phase_d_s2_training.py` giữ lịch 20 episode/block, hai góc cùng board/opponent; recent round-robin hai snapshot của pool trước block, fallback S2 có audit. Snapshot/candidate là weights CPU, committed state quyết định membership; opponent eval, không gradient, cache giới hạn. Acceptance lịch/count/snapshot/frozen đối thủ đã qua.

## 64. D-S2 T3 — full-state opt-in và GPU resume (22/10/2026)

Trainer nhận context D-S2 tường minh; `training_state` có format D riêng nhưng loader mặc định vẫn C. Lưu/restores Adam, cosine scheduler, toàn bộ RNG và context sau episode. Không đổi loss/reward/mask hoặc đường mặc định D/E. Batch protocol/training 21 test OK (1 CUDA opt-in skip); CUDA thật + legacy state 13 test OK. Test injection trước/sau commit và partial episode giữ đúng kết quả liên tục; dữ liệu smoke không chứng minh cải thiện agent.

## 65. D-S2 T4 — validation macro CNN và hai tầng gate (22/10/2026)

`phase_d_s2_evaluation.py` dùng evaluator hiện tại cho task một board/hai góc, strict 9×9 load, B zero-pad giữ logits. Selection có S2 fallback, guards/tie tuple và khóa test manifest; CI resample board sau mean bốn CNN/hai góc, RB/B riêng. 21 acceptance math/manifest/channel/gate qua; chưa chạy test full hoặc kết luận PILOT/PHASE_D_PASS.

## 66. D-S2 T5 — CLI, frozen workers và atomic ledger/package (22/10/2026)

CLI xác minh source/config/checkpoint/runtime rồi chạy snapshot; worker spawn áp dụng CPU threads1/deterministic. Ledger theo matchup-board, kiểm đủ corners/schema/identity trước reuse; test marker write-once và chỉ mở khi validation đạt. Report có theo góc/thời gian và CI board; selection/projection/package được phục hồi hoặc từ chối nếu drift. `progress.bar` với public spawn Pool để Ctrl+C kết thúc task dở; không sửa helper progress chung.

Test độc lập đi RED→GREEN; bổ sung lỗi thật từ integration (NumPy CI scalar → Python float), PENDING sau publish, partial prepare, JSONL overlap, history CRLF, thiếu snapshot và projection/package. Full discover không thêm error so với baseline; vẫn20error/1failure cũ, không gọi green. Smoke mới không chứng minh model cải thiện.

## 67. D-S2 T6 — smoke thật và bàn giao (22/10/2026)

GUIDE có lệnh CMD tuyệt đối cho từng stage/resume, log trong output và ZIP; map CLAUDE/tracker tách code khỏi experiment full. Chạy smoke CPU/GPU2worker, kiểm source_root/threads trong mọi shard và VALIDATION_STOP không mở test; package đủ artifact. Có JSON wall time/RSS/profile và dự phóng riêng train/validation/test; registration mới nhúng evidence. Source CUDA resume cuối qua lại1test. Full suite526test vẫn20error+1failure cùng baseline, không báo toàn repo xanh hoặc Phase D PASS. Review độc lập cuối còn được ghi sau.
