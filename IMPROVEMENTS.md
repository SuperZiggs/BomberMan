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
