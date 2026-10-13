# Multi-Agent Free-For-All Bomberman — Đồ án Reinforcement Learning

## 0. Mục đích tài liệu

Tài liệu này mô tả đầy đủ ý tưởng, phạm vi, và kế hoạch phát triển theo từng giai đoạn cho
đồ án. Mỗi giai đoạn có **input**, **công việc cần làm**, **file cần tạo/sửa**, và **tiêu chí
đạt (gate)** để được coi là hoàn thành và chuyển sang giai đoạn kế tiếp. Một coding agent
(hoặc con người) có thể đọc tài liệu này và triển khai tuần tự mà không cần thêm ngữ cảnh.

**Không được nhảy cóc giai đoạn.** Nếu tiêu chí đạt của một giai đoạn chưa thỏa, phải quay
lại debug/tinh chỉnh giai đoạn đó trước khi đi tiếp — lý do được giải thích ở mục 6.

---

## 1. Ý tưởng dự án

Xây dựng một game đặt bom nhiều tác tử theo kiểu **Bomberman/Pommerman, chế độ Free-For-All
(FFA)**: 2–4 agent cùng xuất hiện trên một bàn cờ lưới, đặt bom để phá tường và tiêu diệt lẫn
nhau, người sống sót cuối cùng thắng. Đồ án huấn luyện các agent chơi game này bằng
Reinforcement Learning, đi từ các phương pháp cơ bản (Q-learning, function approximation)
đến **Actor-Critic** — áp dụng trực tiếp nội dung đã học trong môn học (danh sách bài học ở
mục 8).

**Điểm nghiên cứu chính của đồ án:** so sánh hiệu quả học giữa (a) Q-learning/linear function
approximation baseline và (b) Actor-Critic với CNN, đồng thời khảo sát các vấn đề đặc thù của
multi-agent RL trong môi trường cạnh tranh (non-stationarity, credit assignment, reward
shaping để tránh hành vi suy biến).

---

## 2. Hướng đi tổng thể

- **Kiểu multi-agent**: Free-For-All (cạnh tranh, không hợp tác).
- **Môi trường**: tự dựng từ đầu (numpy, không phụ thuộc gym/pommerman), lấy cảm hứng thiết
  kế từ Pommerman (bàn cờ checkerboard, tường phá được, power-up).
- **Framework huấn luyện**: PyTorch.
- **Thuật toán chính**: Actor-Critic (mạng CNN dùng chung — self-play với parameter sharing).
- **Baseline so sánh**: Q-learning / linear function approximation trên bàn cờ nhỏ.
- **Chiến lược huấn luyện**: curriculum tăng dần độ khó (xem mục 5) — KHÔNG train thẳng FFA
  4 agent từ đầu, vì lý do nêu ở mục 6.

---

## 3. Trạng thái hiện tại (đã hoàn thành)

| File | Mô tả | Trạng thái |
|---|---|---|
| `bomberman_env.py` | Môi trường FFA đầy đủ: grid, bomb, blast, power-up, reward; observation ban đầu 8 kênh, hiện 12 kênh | Đã viết, đã smoke-test bằng random actions (chạy được, không lỗi) |
| `train_actor_critic.py` | Actor-Critic CNN dùng chung cho 4 agent, self-play thuần (chưa có curriculum) | Đã viết, đã kiểm tra cú pháp (`py_compile`), **chưa chạy thử thật** (môi trường viết code không cài được torch) |

Hai file trên là điểm khởi đầu. Các giai đoạn dưới đây sẽ **sửa/mở rộng** chúng, không viết
lại từ đầu.

---

## 4. Đặc tả kỹ thuật môi trường (đã triển khai trong `bomberman_env.py`)

- Bàn cờ mặc định 11×11 (`board_size`, phải là số lẻ), viền và checkerboard là tường cứng
  (`RIGID_WALL`), phần còn lại rải ngẫu nhiên tường gỗ (`WOODEN_WALL`, tỉ lệ
  `wooden_wall_density`), trừ vùng an toàn 3×3 quanh 4 góc spawn.
- Action space rời rạc, 6 giá trị: `STOP, UP, DOWN, LEFT, RIGHT, BOMB`.
- Bomb: timer đếm ngược (`bomb_timer`), nổ hình dấu cộng theo `blast_range` của agent đặt
  bom, dừng lại khi gặp tường cứng, phá tường gỗ và dừng (kèm xác suất `powerup_prob` rơi
  power-up), **chain reaction** nếu bom khác dính trong vùng nổ.
- Power-up: `ammo` (+1 số bom đặt được cùng lúc, trần 3), `range` (+1 tầm nổ, trần 4). Nâng cấp thành công thưởng `+0.05`; tại trần không tăng chỉ số/thưởng.
- Reward mỗi bước (per-agent): `-1.0` khi chết, `+1.0` khi tiêu diệt agent khác (quy cho chủ
  sở hữu quả bom gây chết), `+10.0` cho người sống sót cuối cùng khi kết thúc episode,
  `-0.001` mỗi bước còn sống (khuyến khích hành động, chống thụ động).
- Observation hiện tại: numpy array `(12, H, W)` float32 — 8 kênh đầu giữ nguyên: tường cứng,
  tường gỗ, bom (timer chuẩn hóa), lửa, ammo item, range item, bản thân, đối thủ (gộp).
  Thêm slots bom còn trống, sức chứa, tầm nổ và thời gian nguy hiểm dự báo; xem mục 8 `IMPROVEMENTS.md`.
- API: `reset() -> list[obs]`, `step(actions: list[int]) -> (obs_list, rewards, dones, info)`,
  `info["episode_done"]` báo hiệu kết thúc toàn episode; `info["alive"]` là list bool.

**Không thay đổi API này** ở các giai đoạn sau, chỉ thêm tham số mới nếu cần (ví dụ
`num_agents=2` cho giai đoạn 1v1) — các giai đoạn dưới đều dùng lại đúng class
`BombermanFFAEnv`.

---

## 5. Lộ trình phát triển theo giai đoạn (curriculum)

### Giai đoạn A — Rule-based bot & sanity check môi trường

**Việc cần làm:**
1. Tạo file `rule_based_bot.py` chứa một hàm/class `RuleBasedAgent` chọn action theo heuristic
   đơn giản: nếu đang đứng trong hoặc sắp vào vùng lửa/bom sắp nổ → né; nếu không, di chuyển về
   tường gỗ gần nhất; nếu đã đứng cạnh tường gỗ và còn ammo → đặt bom rồi né ra xa.
2. Viết script nhỏ cho hai `RuleBasedAgent` tự đấu với nhau 100 episode trên `BombermanFFAEnv`
   (`num_agents=2`), in ra thống kê: tỉ lệ mỗi bên thắng, độ dài trung bình episode, số bom đặt
   trung bình/episode.

**Tiêu chí đạt (gate):**
- Không có exception/crash trong 100 episode.
- Tỉ lệ thắng giữa hai bot rule-based gần 50/50 (chênh lệch < 15 điểm %) — xác nhận môi
  trường không thiên vị vị trí spawn.
- Số bom đặt trung bình > 0 rõ rệt mỗi episode (xác nhận heuristic đặt bom hoạt động, dùng làm
  mốc so sánh "bomb shyness" ở các giai đoạn sau).

### Giai đoạn B — Baseline Q-learning / linear function approximation

**Việc cần làm:**
1. Tạo file `train_qlearning_baseline.py`.
2. Dùng `BombermanFFAEnv(board_size=7, num_agents=2)` (bàn nhỏ để state space khả thi).
3. Trích đặc trưng tuyến tính từ observation 8 kênh (ví dụ: khoảng cách Manhattan tới tường gỗ
   gần nhất, tới bom gần nhất, tới đối thủ, số ô an toàn trong bán kính 2, còn ammo hay
   không...) — feature construction thủ công, đúng tinh thần bài 3.5.
4. Cài Q-learning với linear function approximation (trọng số cập nhật theo TD error), agent
   train vs `RuleBasedAgent` từ giai đoạn A.
5. Log win rate vs rule-based bot theo từng mốc (mỗi 100 episode).

**Tiêu chí đạt (gate):**
- Win rate vs `RuleBasedAgent` tăng dần theo thời gian train (đường cong học có xu hướng đi
  lên rõ ràng, không phẳng/nhiễu hoàn toàn).
- Đạt win rate ≥ 50% vs `RuleBasedAgent` sau khi train hội tụ.
- Nếu không đạt: nghi ngờ đầu tiên là bộ feature chưa đủ thông tin hoặc reward chưa được
  chuẩn hóa — không chuyển sang giai đoạn C khi baseline này còn chưa học được gì.

### Giai đoạn C — Actor-Critic 1v1 vs rule-based bot

**Việc cần làm:**
1. Sửa `train_actor_critic.py`: thêm tham số cho phép 1 agent là `RuleBasedAgent` cố định
   (không cập nhật gradient), agent còn lại là `ActorCriticNet` đang học.
2. Dùng `board_size=9` (nhỏ hơn bản 11×11 gốc), `num_agents=2`.
3. Thêm **reward shaping chống "bomb shyness"**: cộng thêm `+0.05` đến `+0.1` vào reward của
   agent mỗi khi một quả bom do chính nó đặt phá được ít nhất một tường gỗ (tách biệt khỏi
   reward tiêu diệt đối thủ). Cần sửa `bomberman_env.py` để trả thêm thông tin "ai phá được
   tường nào" trong `info`, hoặc tính riêng ở tầng training loop bằng cách so sánh số tường gỗ
   trước/sau mỗi bước cho mỗi bomb owner.
4. Log: win rate vs `RuleBasedAgent`, reward trung bình/episode, số bom đặt trung bình/episode
   (so sánh với mốc ở Giai đoạn A để phát hiện bomb shyness).

Bản triển khai hiện tại luân phiên góc học ở cả C/D/E. Checkpoint best được chọn trên seed validation cố định, chỉ xét ứng viên có tự nổ không tăng so với validation đầu lượt, rồi ưu tiên ActiveWR, ít tự nổ và ít hòa; số liệu rolling train chỉ dùng quan sát. Validation dùng bản sao trọng số trên CPU giống GUI, độc lập thiết bị train. Báo cáo cuối dùng seed khác validation và có số liệu riêng theo góc. WR tổng và số bom không thay thế bằng chứng hạ gục bởi bom của agent. Nhãn hướng dẫn xét số vị trí sống của đối thủ theo timer bom; loss bom gây áp lực được lấy trung bình riêng với loss di chuyển. Safety mask cho phép một bom bổ sung lúc nguy hiểm chỉ nếu dự báo agent sống được và đối thủ mất toàn bộ vị trí sống ở tick nổ; RuleBot đối chứng giữ luật cũ. Điều kiện trên dự báo không đảm bảo thắng khi đối thủ có hành động tương lai chưa biết.

**Tiêu chí đạt (gate):**
- Số bom đặt trung bình/episode của agent đang học **không** giảm dần về gần 0 theo thời gian
  train (nếu giảm về 0 → bomb shyness, phải tăng hệ số thưởng phá tường hoặc giảm hình phạt
  chết trước khi đi tiếp).
- Win rate vs `RuleBasedAgent` ổn định ở mức **≥ 70%** trong ít nhất 200 episode liên tiếp gần
  cuối quá trình train.
- Đường cong reward trung bình/episode có xu hướng tăng và ổn định dần (không dao động mạnh
  không kiểm soát ở cuối training).

### Giai đoạn D — Self-play với opponent pool

**Việc cần làm:**
1. Thêm cơ chế lưu snapshot trọng số mạng (`torch.save`) mỗi N episode (N cấu hình được, ví
   dụ 200) vào một thư mục `checkpoints/`.
2. Sửa training loop: mỗi episode, đối thủ của agent đang học được lấy **ngẫu nhiên** từ tập
   snapshot đã lưu (bao gồm cả bản mới nhất), thay vì luôn là bản sao hiện tại của chính nó.
   Giữ `RuleBasedAgent` trong pool đối thủ như một "mỏ neo" để tránh agent quên hoàn toàn cách
   đối phó với chiến lược cơ bản.
3. Vẫn 2 agent (`num_agents=2`), `board_size=9` hoặc quay lại 11.

**Tiêu chí đạt (gate):**
- Agent mới nhất (bản hiện tại) đạt win rate ≥ 60% khi đấu round-robin với **toàn bộ**
  snapshot cũ trong pool (không chỉ thắng bản ngay trước nó — xác nhận đang thực sự tiến bộ,
  không chỉ overfit vào một đối thủ gần nhất).
- Không có hiện tượng "quên": win rate vs `RuleBasedAgent` (mốc cố định) không giảm so với
  Giai đoạn C.

### Giai đoạn E — Mở rộng số agent: 2 → 3 → 4 (FFA đầy đủ)

**Việc cần làm:**
1. Lặp lại quy trình self-play + opponent pool của Giai đoạn D với `num_agents=3`, dùng agent
   đã train ở Giai đoạn D làm điểm khởi tạo (warm start) thay vì train từ đầu.
2. Sau khi ổn định với 3 agent, tăng lên `num_agents=4`, `board_size=11` (cấu hình đầy đủ như
   thiết kế gốc).
3. Log thêm: phân bố agent nào sống sót cuối cùng theo vị trí spawn (kiểm tra không có agent
   nào bị lợi thế/bất lợi hệ thống do vị trí góc).

**Tiêu chí đạt (gate) — coi như hoàn thành đồ án phần huấn luyện:**
- Trong FFA 4 agent (mạng dùng chung, self-play pool), tỉ lệ sống sót cuối cùng của 4 vị trí
  spawn không lệch quá xa nhau (mỗi vị trí đạt tỉ lệ thắng trong khoảng roughly 20–30%, không
  agent nào ~0% hoặc ~70%).
- Điểm (reward tích lũy trung bình mỗi agent) tăng dần qua các mốc train so với baseline
  random action.
- Agent thể hiện hành vi có thể quan sát bằng mắt khi render một vài episode: biết né lửa,
  biết đặt bom phá tường, không đứng yên thụ động hoàn toàn.

### Giai đoạn F — Đánh giá & viết báo cáo

**Việc cần làm:**
1. So sánh học curve (reward theo episode) của: random baseline, `RuleBasedAgent`, Q-learning
   linear (Giai đoạn B), Actor-Critic 1v1 (Giai đoạn C), Actor-Critic self-play FFA (Giai đoạn
   E).
2. Ablation tối thiểu: có/không reward shaping chống bomb shyness (so Giai đoạn C có/không
   phần thưởng phá tường).
3. Viết phần lý thuyết liên hệ lại: MDP → Markov Game (non-stationarity), TD/Actor-Critic
   framework áp dụng ra sao cho từng agent, vì sao cần curriculum thay vì train FFA trực tiếp.

**Tiêu chí đạt:** có đủ số liệu/biểu đồ cho từng mục trên để đưa vào báo cáo/slide.

---

## 6. Vì sao phải theo đúng thứ tự curriculum (không train thẳng FFA 4 agent)

- **Non-stationarity**: khi nhiều agent cùng học đồng thời bằng một mạng chia sẻ, "đối thủ"
  thay đổi liên tục qua từng episode, khiến tín hiệu học rất nhiễu nếu không có mỏ neo
  (rule-based bot) hoặc opponent pool ổn định.
- **Bomb shyness**: vấn đề đã ghi nhận trong nghiên cứu Pommerman — nếu không có reward
  shaping riêng cho việc phá tường, agent học rất nhanh chiến lược "không bao giờ đặt bom" vì
  early-training chỉ thấy hậu quả tiêu cực của việc đặt bom gần bản thân.
- **Credit assignment trong FFA 4 người** khó hơn nhiều so với 1v1 — nếu agent chưa thắng nổi
  một bot đơn giản 1v1, gần như chắc chắn sẽ không học được gì trong môi trường 4 người hỗn
  loạn hơn.

Mỗi gate ở mục 5 tồn tại để phát hiện sớm các thất bại trên trước khi đầu tư thời gian
compute vào giai đoạn phức tạp hơn.

---

## 7. Cấu trúc file dự kiến

```
project/
├── bomberman_env.py            # đã có — môi trường FFA
├── rule_based_bot.py           # Giai đoạn A
├── train_qlearning_baseline.py # Giai đoạn B
├── train_actor_critic.py       # đã có (bản gốc self-play 4-agent) — sẽ sửa ở Giai đoạn C/D/E
├── checkpoints/                # snapshot mạng, tạo ở Giai đoạn D
├── evaluate.py                 # Giai đoạn F — chạy round-robin, xuất số liệu/biểu đồ
└── PROJECT_SPEC.md             # tài liệu này
```

---

## 8. Bối cảnh môn học (để liên hệ lý thuyết trong báo cáo)

Đồ án dựa trên nội dung các bài học: K-Armed Bandit, Estimating Action Values,
Exploration vs Exploitation, MDP, Goal of RL, Continuing Tasks, Policies & Value Functions,
Bellman Equations, Optimality, Policy Evaluation/Iteration, Generalized Policy Iteration,
Monte-Carlo Methods & Control, Off-policy prediction, Temporal Difference Learning (TD,
Sarsa, Q-learning), Expected Sarsa, Model/Planning trong RL, Dyna, Estimating Value Functions
as Supervised Learning, Objective cho On-policy Prediction/TD, Linear TD, Feature Construction
for Linear Methods, Episodic Sarsa with Function Approximation, Exploration under Function
Approximation, Average Reward, Learning Parameterized Policies, Policy Gradient, **Actor-Critic
for Continuing Tasks**, Policy Parameterizations.

Trọng tâm áp dụng trực tiếp vào đồ án: Feature Construction (Giai đoạn B), TD/Sarsa/Q-learning
(Giai đoạn B), Policy Gradient & Actor-Critic (Giai đoạn C–E).
