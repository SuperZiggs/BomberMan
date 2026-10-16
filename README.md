# Multi-Agent Free-For-All Bomberman (Reinforcement Learning)

Đồ án Reinforcement Learning: Xây dựng và huấn luyện Multi-Agent chơi game Bomberman chế độ Free-For-All (FFA) từ 2 đến 4 agents sử dụng **Actor-Critic (CNN)** và baseline **Q-learning (Linear Function Approximation)**.

---

## 📌 Tổng quan dự án

- **Môi trường:** Tự xây dựng từ đầu bằng Python/NumPy (`bomberman_env.py`), lấy cảm hứng từ Pommerman (bàn cờ grid checkerboard, tường gỗ phá được, vật phẩm/power-ups, cơ chế bom nổ và tính sát thương).
- **Thuật toán chính:**
  - **Actor-Critic (A2C/CNN)** với cơ chế self-play và parameter sharing (`train_actor_critic.py`).
  - **Q-learning baseline** với xấp xỉ tuyến tính (Linear Function Approximation) (`train_qlearning_baseline.py`).
- **Chiến lược huấn luyện:** Curriculum learning qua nhiều giai đoạn (Phase A -> Phase E), tinh chỉnh reward shaping và cấu trúc CNN để giải quyết các thách thức đặc thù của Multi-Agent RL (non-stationarity, credit assignment, lẩn tránh và đặt bom thông minh).

---

## 📁 Cấu trúc thư mục

```text
├── bomberman_env.py             # Môi trường FFA hoàn chỉnh (Numpy-based)
├── rule_based_bot.py            # Bot đối thủ dựa trên luật (BFS tìm đường, né bom, săn đối thủ)
├── train_actor_critic.py        # Pipeline huấn luyện Actor-Critic CNN (Curriculum self-play)
├── train_qlearning_baseline.py  # Pipeline huấn luyện Q-learning Linear FA baseline
├── evaluate.py                  # Script đánh giá và đối đầu giữa các mô hình
├── checkpoints/                 # Trọng số mô hình đã huấn luyện (best & final qua các Phase)
├── PROJECT_SPEC.md              # Đặc tả chi tiết dự án, tiêu chuẩn kỹ thuật & roadmap
├── RESULTS.md                   # Nhật ký kết quả thực nghiệm chi tiết qua từng Phase
├── IMPROVEMENTS.md              # Phân tích cải tiến kiến trúc mô hình & reward shaping
└── .gitignore                   # Cấu hình bỏ qua virtualenv, cache và snapshot tạm
```

---

## 🚀 Cài đặt & Hướng dẫn sử dụng

### 1. Cài đặt môi trường
Trên Windows, dùng Python 3.10 trở lên. Mở PowerShell tại repo rồi chạy:

```powershell
cd D:\fpt\subject\BomberMan
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch numpy
```

Nếu `.venv` hiện tại đã có `torch` và `numpy`, bỏ qua bước tạo/cài môi trường. Các lệnh dưới dùng trực tiếp Python trong `.venv`, không cần activate hay đổi execution policy.

### 2. Huấn luyện mô hình

- **Dùng GUI:** chạy `.\.venv\Scripts\python.exe visualize.py`, vào **Train trực tiếp**, chọn Phase C và khoảng 3000 episodes. Bỏ **Warm-start** nếu muốn học từ trọng số ngẫu nhiên; giữ chọn nếu muốn học tiếp checkpoint trên thanh phía trên. Bấm **Bắt đầu train**. Khi hoàn tất, model `best_*.pt` của lượt train được chọn để xem đấu; bảng trạng thái có **WR có hạ gục** và **Hạ gục/ván**.
- **Học tiếp model FFA hiện có qua CLI:**

```powershell
.\.venv\Scripts\python.exe train_actor_critic.py --phase E --episodes 2000 --board_size 11 --num_agents 4 --max_steps 500 --resume_from checkpoints/best_phaseE_4a.pt --checkpoint_dir runs/combat-training
```

- **Pilot Phase C có Sudden Death:** fine-tune từ checkpoint Phase C hiện tại, bật thu hẹp bản đồ ở 75% số ván và lưu sang thư mục mới. Script in trực tiếp ra PowerShell, không tạo transcript/log terminal.

```powershell
.\run_train_phaseC_sd.ps1
```

Pilot dùng 500 episodes, seed `2026101401`, bắt đầu thu hẹp ở bước 350, mỗi 20 bước đóng thêm một vòng và cảnh báo trước 40 bước. Checkpoint đầu vào là `runs/pressure-C-next/final_phaseC.pt`; đầu ra nằm trong `runs/phaseC-sd-pilot/`. Không thay checkpoint đang dùng chỉ vì WR tăng: `sudden_death_win` được báo riêng, không tính vào `active_win`/ActiveWR.

- **Fine-tune Phase C theo Schedule E và tự đánh giá:** một lệnh train 1500 episodes, lưu checkpoint tốt nhất theo validation có Sudden Death, rồi load `best_phaseC.pt` để đánh giá 200 trận và in các gate PASS/FAIL.

```powershell
.\run_train_phaseC_sd_1500.ps1
```

Lượt này dùng xác suất Sudden Death `0.75`, lịch `475/25/40`, validation mỗi 250 episodes và ghi toàn bộ kết quả mới vào `runs/phaseC-sd-1500-e/`. Script dừng trước khi train nếu thư mục đầu ra đã tồn tại để không ghi đè lượt cũ.

- **Fine-tune attack teacher hai bước:** warm-start từ best Phase C-SD, train 750 episodes với lookahead hai bước, ngưỡng pressure `0.15`, tăng supervision tấn công từ bước 400 và tự đánh giá 200 trận sau khi train.

```powershell
.\run_train_phaseC_sd_attack_750.ps1
```

Kết quả mới được lưu riêng trong `runs/phaseC-sd-attack-750/`. RuleBot và reward môi trường không đổi; launcher chỉ điều chỉnh attack teacher, hệ số supervision, learning rate và entropy.

- **Fine-tune Phase C-SD theo mục tiêu cuối:** warm-start từ `runs/phaseC-sd-attack-750/best_phaseC.pt`, train 750 episodes với teacher chuỗi hai bom và supervision thoát vòng đóng, validation 200 trận tại episode 0/250/500/750, rồi đánh giá checkpoint tốt nhất trên 200 trận mới.

```powershell
.\run_train_phaseC_sd_target_750.ps1
```

Kết quả được lưu trong `runs/phaseC-sd-target-750/`. Checkpoint được xếp hạng theo số mục tiêu đạt, biên mục tiêu tệ nhất, ActiveWR, Draw, SD death và WR; checkpoint có Suicide trên `1.5%` bị loại. Các gate cuối là WR `>=70%`, ActiveWR `>=55%`, Draw `<=15%`, SD death `<=10%`, Suicide `<=1.5%`. Script không ghi đè thư mục đầu ra đã tồn tại.

- **Train Actor-Critic từ đầu theo curriculum:** chạy lần lượt, đợi mỗi lệnh hoàn tất trước lệnh tiếp theo. Các checkpoint mới lưu riêng trong `runs/from-scratch/`.

```powershell
# C: học từ trọng số ngẫu nhiên, đấu RuleBot 1v1
.\.venv\Scripts\python.exe train_actor_critic.py --phase C --episodes 3000 --board_size 9 --num_agents 2 --max_steps 500 --checkpoint_dir runs/from-scratch/C

# D: học self-play, bắt đầu từ kết quả C
.\.venv\Scripts\python.exe train_actor_critic.py --phase D --episodes 3000 --board_size 9 --num_agents 2 --max_steps 500 --resume_from runs/from-scratch/C/best_phaseC.pt --checkpoint_dir runs/from-scratch/D

# E1: tăng lên 3 tác tử
.\.venv\Scripts\python.exe train_actor_critic.py --phase E --episodes 2000 --board_size 9 --num_agents 3 --max_steps 500 --resume_from runs/from-scratch/D/best_phaseD.pt --checkpoint_dir runs/from-scratch/E3

# E2: tăng lên 4 tác tử, bàn 11x11
.\.venv\Scripts\python.exe train_actor_critic.py --phase E --episodes 3000 --board_size 11 --num_agents 4 --max_steps 500 --resume_from runs/from-scratch/E3/best_phaseE_3a.pt --checkpoint_dir runs/from-scratch/E4
```

`episodes` là số ván học; `max_steps=500` là số bước tối đa **mỗi ván**. 100 episodes phù hợp thử pipeline, chưa đủ để kết luận agent đã học tấn công tốt. Lệnh CLI tự dùng CUDA khi PyTorch nhận GPU; GUI dùng CPU. CLI còn chạy báo cáo đánh giá cuối phase nên có thể tiếp tục chạy sau dòng lưu checkpoint.

- **Baseline Q-learning (để so sánh):** `.\.venv\Scripts\python.exe train_qlearning_baseline.py`.

### 3. Đánh giá mô hình

Đánh giá agent đã huấn luyện đối đầu với bot hoặc tự đối đầu:

```powershell
.\.venv\Scripts\python.exe evaluate.py --mode tournament --board_size 11 --num_agents 4 --num_episodes 100 --max_steps 500 --ac_model runs/from-scratch/E4/best_phaseE_4a.pt
```

Đổi `--ac_model` thành checkpoint muốn kiểm tra, ví dụ `checkpoints/best_phaseE_4a.pt` hoặc `runs/combat-training/best_phaseE_4a.pt`. Báo cáo FFA có agent AC ở A0 đấu 3 RuleBot, với `Kills`, `ActiveWR` (thắng và có ít nhất một hạ gục bằng bom của mình) và `Suicide`. Đừng dùng WR hoặc số bom đơn lẻ để kết luận agent tấn công tốt.

Đánh giá cố định, luân phiên AC ở tất cả góc trên **cùng bàn cho mỗi seed**, và xuất JSON:

```powershell
& "D:\anaconda\envs\torch\python.exe" evaluate.py --mode seeded --board_size 9 --num_agents 2 --num_episodes 100 --max_steps 500 --seed 9000 --ac_model runs/20261010-135449-587798/best_phaseC.pt --output runs/evaluation.json
```

Số trận phải chia hết cho số agents. Báo cáo có ActiveWR, tự nổ, hòa, kills và số liệu theo góc. Dùng cùng seed khi so các checkpoint; seed đánh giá cuối phải khác seed validation dùng để chọn model.

Đánh giá cùng cấu hình Sudden Death và tách riêng thắng do bom, thắng do vòng thu hẹp, thắng thụ động, tự nổ, chết do vòng thu hẹp và hòa:

```powershell
.\.venv\Scripts\python.exe evaluate.py --mode seeded --board_size 9 --num_agents 2 --num_episodes 100 --max_steps 500 --seed 21000 --ac_model runs/phaseC-sd-pilot/best_phaseC.pt --sudden_death --sudden_death_start 350 --sudden_death_interval 20 --sudden_death_warning_steps 40 --output runs/phaseC-sd-pilot/evaluation-21000.json
```

### 4. Giao diện trực quan: train, xem đấu và chơi

```powershell
python visualize.py
# Nếu dùng virtualenv trong repo:
.\.venv\Scripts\python.exe visualize.py
```

Giao diện desktop dùng Tkinter có sẵn trong Python, chỉ cần `torch` và `numpy` như pipeline hiện tại. Trên Linux cần có gói hệ thống `python3-tk`.

- **Train trực tiếp:** Actor-Critic Phase C/D/E, chọn số episode và warm-start từ checkpoint. Phase C/D tự đặt ô **Tác tử** thành 2 và khóa bộ chọn, dùng bàn 9×9; Phase E cho chọn 2 hoặc 3 tác tử trên 9×9, hoặc 4 trên 11×11. Bàn cờ hiển thị episode đang học; biểu đồ reward, win rate, số bom và loss dùng trung bình 50 episode gần nhất. Kéo thanh tốc độ để xem chậm, tạm dừng hoặc **Dừng & lưu**.
- **Agent vs Bots:** checkpoint đã train đấu 1–3 RuleBot trên bàn 11×11. Có tạm dừng, chạy từng bước, chỉnh tốc độ và bật vùng nguy hiểm.
- **Bạn vs Agent + Bots:** bạn điều khiển A0, agent đã train là A1, cộng 1–2 RuleBot. Bấm **Ván mới → Chạy**, click bàn cờ rồi dùng **WASD / phím mũi tên** để di chuyển, **Space / E** để đặt bom; cũng có nút điều khiển trên màn hình.

Chế độ người chơi mặc định **12 bước/giây**; có thể chỉnh **Bước / giây** từ 1 đến 60. Lệnh nhấn nhanh được giữ tới tick kế tiếp, kể cả đã thả phím; bom được ưu tiên trước di chuyển. Khoảng chờ giữa các tick được trừ thời gian suy luận/render để giữ tốc độ đặt. Đây là tốc độ mô phỏng: tăng tốc cũng làm bot đi nhanh và bom nổ sớm hơn theo thời gian thực (bom đếm 8 bước). GUI hiện dùng CPU; GPU không loại bỏ thời gian chờ do tick hay phần vẽ Tkinter.

Cả ba tab dùng đồ họa pixel từ `assets/`: gạch cho tường cứng, thùng cho tường phá được, robot/bom cùng màu theo A0–A3, rương `+A` tăng sức chứa và vàng `+R` tăng tầm nổ. Hiệu ứng nổ và khói dùng các frame PNG có sẵn. Ảnh được tải một lần, scale bằng Tkinter và cache theo kích thước ô; không tải lại mỗi tick. Font Emulogic dùng cho tiêu đề trên Windows, được nạp riêng trong tiến trình. Giao diện không cần thêm thư viện đồ họa.

Checkpoint mặc định là `checkpoints/best_phaseE_4a.pt`. Sau khi train, bấm **Làm mới**, chọn checkpoint mới rồi vào **Agent vs Bots** và bấm **Ván mới → Chạy**. GUI tái dùng action masking và cách lấy mẫu policy của pipeline train; checkpoint bàn khác kích thước được nội suy bằng helper có sẵn.

Sau bản sửa ngày 13/10, đóng và mở lại GUI để nạp code mới. Checkpoint `runs/pressure-C-next/final_phaseC.pt` kết hợp logic thoát sớm được đánh giá trên 200 trận độc lập seed 16000: ActiveWR **37%**, tự nổ **1.5%**, hòa **54%**. Cùng trọng số trước sửa là 30.5% / 6.5% / 61%; thay đổi này đến từ mask, chưa train thêm. Dùng checkpoint này để xem đấu tiếp; giữ `runs/combat-20261012/combat_bc_phaseC.pt` làm mốc đối chiếu. `pressure-C-next/best_phaseC.pt` vẫn là trọng số warm-start được chọn ở ep 0 với code trước sửa, không bị ghi đè hoặc đổi tên thành final. Benchmark dùng **9×9, hai tác tử, 500 bước**; tab xem đấu/chơi dùng 11×11 nên chưa có cùng mức đảm bảo. Chưa chuyển D/E vì hòa còn cao; chi tiết phiên bản/seed ở `RESULTS.md`. GUI dùng LR `1e-4` khi warm-start, `3e-4` khi train từ đầu. Checkpoint 8 kênh cũ vẫn nạp được nhưng cần học các kênh chỉ số thêm. Trên máy hiện tại đã kiểm tra được môi trường Conda:

```powershell
& "D:\anaconda\envs\torch\python.exe" visualize.py
```

Agent AC và RuleBot dùng chung mô hình thời gian nổ, tính cả dây chuyền, lửa còn sót và ô tranh chấp. AC kiểm tra thêm bom đáp trả cùng lúc, giữ đường thoát tránh cửa ngõ đối thủ có thể chiếm và chừa 2 bước khi đặt bom; khi đang thoát, ưu tiên nơi trú ít bị tia bom ở vị trí hiện tại/bước kế tiếp của đối thủ đe dọa. Nếu không còn đường có dự phòng, vẫn giữ các hành động sống được theo thời gian thật. RuleBot giữ hành vi trước lượt sửa này để làm đối thủ cố định, vẫn nhặt item và dùng range thực tế. Đây là dự báo và heuristic, vẫn có thể chết khi đối thủ đặt bom muộn hơn hoặc đổi đường ngoài dự báo.

Sau đặt bom, AC tránh đi vào hành lang sẽ cháy và chỉ có một lối ra nếu bot có thể tới lối ra đó trong hai bước; những đường có lựa chọn thoát khác vẫn được giữ. Teacher tìm vị trí có tia nổ chạm đối thủ qua bước đầu được mask cho phép; không cố đi vào ô đối thủ đang đứng. Metrics thêm `attack_opportunities`, `attack_bombs`, `attack_labels`, `missing_attack_labels` để theo dõi cơ hội tấn công và nhãn trên các bước không nguy hiểm. Validation dùng bản sao model trên CPU giống GUI, độc lập thiết bị train; metadata ghi `device=cpu`.

Khi lửa nhiều bom chồng nhau, kill credit ưu tiên một nguồn bom đối thủ thực sự trúng nạn nhân, thay vì cho bom A0 luôn nhận trước. Engine lưu thời gian riêng từng nguồn lửa. `self_blast_deaths` vẫn đánh dấu nạn nhân đứng trong tia bom của chính mình, kể cả có thêm bom đối thủ: phạt tự nổ và thống kê `suicide` không bị mất khi đổi kill credit. Những số liệu cũ dùng quy ước credit trước sửa phải được chấm lại trước khi so sánh.

Hai item giữ xác suất rơi tổng `0.3` và chia đều loại. Ammo tăng sức chứa tối đa **3**, range tối đa **4**; mỗi nâng cấp thực sự thưởng `+0.05`, item ở trần vẫn bị nhặt nhưng không tăng chỉ số hay thưởng. Observation hiện có **13 kênh**: giữ nguyên 12 kênh trước đó và thêm cảnh báo vòng Sudden Death kế tiếp. Loader hỗ trợ checkpoint 8 hoặc 12 kênh, chép trọng số cũ và khởi tạo các kênh mới bằng 0 để giữ nguyên logits trước fine-tune khi cảnh báo bằng 0.

Theo dõi `active_win`, `kills`, `suicide`, `draw`, `items`, `learner_steps`, `learning_aid` và `best_episode` trong metrics. **C/D/E đều luân phiên góc học.** `best` được chọn bằng policy cố định đấu RuleBot trên seed validation. Ứng viên phải có tỷ lệ tự nổ không cao hơn lần validation đầu lượt; trong các ứng viên đạt điều kiện này, ưu tiên ActiveWR, ít tự nổ, ít hòa, kills rồi WR. `suicide_limit` ghi mức giới hạn này. Đây là điều kiện trên mẫu validation, chưa đảm bảo tỷ lệ tự nổ trên mọi seed. Mặc định đánh giá 48 trận tại đầu lượt, mỗi 250 episode và cuối lượt; trả lại trạng thái RNG sau đánh giá để không đổi chuỗi ngẫu nhiên của train. Khi dùng test để quyết định chỉnh code/train tiếp, lần đánh giá cuối sau đó cần bộ seed mới. `validation_metrics.jsonl` giữ các lần đánh giá, `best_validation.json` ghi số liệu của model được chọn; `best_episode=0` có thể là warm-start nếu học tiếp chưa tốt hơn hoặc tự nổ nhiều hơn. Các lượt ngắn hơn `validation_every` hoặc đặt `--validation_every 0` không được đánh giá: `best` chỉ là fallback của final. `final` luôn giữ trọng số cuối lượt. GUI hiển thị riêng số liệu train và số liệu best đã đánh giá. Không chuyển phase chỉ vì WR tổng cao.

Nhãn hướng dẫn tấn công so sánh số vị trí đối thủ có thể sống tại tick nổ với/không có bom dự kiến, có tính timer, chain reaction, tường và bom chặn đường. Khi đối thủ ở gần, agent được hướng dẫn tới một ô lân cận tạo áp lực lớn hơn hoặc đặt bom tại ô hiện tại nếu nó tốt nhất; vẫn tìm vị trí khi chỗ đặt bom đang bận. Mọi hành động phải qua safety mask. Phép dự báo này dùng vị trí đối thủ hiện tại và bom đã biết, không đảm bảo một hạ gục.

Safety mask dùng trong train/evaluate/GUI nay có một ngoại lệ cho BOMB khi đang trong vùng nguy hiểm: dự báo đặt bom vẫn giữ đường sống cho agent (kể cả phản bom đồng thời), còn đối thủ có vị trí sống trước khi đặt nhưng không còn vị trí sống ở tick nổ sau khi đặt. Các bom thông thường lúc nguy hiểm vẫn bị cấm. CNN tự chọn trong các hành động hợp lệ; không có thao tác ép BOMB hoặc cộng thưởng đặt bom. Ngoại lệ này có thể cải thiện hành vi với checkpoint cũ mà không đổi trọng số. RuleBot đối chứng dùng `robust=False`, giữ nguyên luật trước sửa.

AC thử thoát sớm qua ô kề đối thủ nếu chỉ còn STOP và còn hơn một tick trước nguy hiểm thực tế. Đây là cách tránh cả hai cùng chờ đến tick cuối rồi va chạm ở lối ra chung. Tường, bom, ô đối thủ đang đứng và lửa vẫn được xét; nếu có đường không tranh chấp thì giữ đường đó, nếu cần chờ lửa tắt thì vẫn chờ. Nhãn escape dùng cùng mask với policy. Heuristic này chưa đảm bảo thoát khi đối thủ tiếp tục chặn đường hoặc đặt bom mới.

Loss hướng dẫn lấy trung bình riêng cho thoát hiểm, bom gây áp lực và di chuyển/mở đường, với trọng số lần lượt **0.5 / 0.5 / 0.2**. Log `combat_labels`, `combat_bombs` cho biết agent chọn bao nhiêu nhãn bom gây áp lực, gồm `followup_labels`/`followup_bombs` cho ngoại lệ lúc nguy hiểm; đây không phải số hạ gục hay xác suất thắng. `policy_loss`, `value_loss`, `escape_loss`, `combat_loss`, `navigation_loss` giúp phân biệt nguồn dao động của loss tổng.

Lượt fine-tune 500 episode dưới đây đã hoàn tất ngày 13/10. Validation 6600 tăng ActiveWR 32% lên 36%, tự nổ giữ 1%, chọn best ở ep 500. Test độc lập 17000 lại có ActiveWR đối chứng **38.5%** so với candidate **36%**, tự nổ **0%** so với **0.5%**. WR tổng candidate tăng do thắng thụ động tăng 8 lên 15 trận. Tiếp tục dùng `runs/pressure-C-next/final_phaseC.pt`; giữ candidate để đối chiếu, ưu tiên chẩn đoán các timeout trước khi train thêm. Lệnh dưới đây ghi lại lượt đã chạy; thư mục output đã có checkpoint/log:

```powershell
& "D:\anaconda\envs\torch\python.exe" train_actor_critic.py --phase C --episodes 500 --board_size 9 --num_agents 2 --max_steps 500 --lr 0.0001 --wall_destroy_bonus 0.15 --seed 2026101302 --resume_from runs/pressure-C-next/final_phaseC.pt --validation_every 250 --validation_seed 6600 --validation_episodes 100 --checkpoint_dir runs/escape-C-6600
```

Hai lệnh chấm dưới đây cũng đã chạy xong, JSON được giữ lại. So ActiveWR, tự nổ và hòa; `best_episode=0` vẫn có thể là kết quả hợp lệ ở các lượt khác nếu chưa có ứng viên tốt hơn. Mỗi lượt train/test mới dùng thư mục và tên output mới để giữ log cũ. Seed 17000 hiện dùng cho chẩn đoán; nếu chỉnh tiếp, dành seed mới như 18000 cho test cuối:

```powershell
& "D:\anaconda\envs\torch\python.exe" evaluate.py --mode seeded --ac_model runs/pressure-C-next/final_phaseC.pt --board_size 9 --num_agents 2 --max_steps 500 --num_episodes 200 --seed 17000 --device cpu --output runs/escape-early/test_control_17000.json
& "D:\anaconda\envs\torch\python.exe" evaluate.py --mode seeded --ac_model runs/escape-C-6600/best_phaseC.pt --board_size 9 --num_agents 2 --max_steps 500 --num_episodes 200 --seed 17000 --device cpu --output runs/escape-early/test_candidate_17000.json
```

Kết quả trên seed 16000 của warm-start với cùng code là ActiveWR 24%, tự nổ 0.5%, hòa 64.5%; `pressure-C-next/final` đạt 37% / 1.5% / 54%. Trên seed 17000, checkpoint đang dùng đạt 38.5% / 0% / 56%. Các số liệu thuộc từng mẫu/seed, chưa đảm bảo mức tự nổ hoặc tỷ lệ thắng trên mọi bàn. Chi tiết lượt fine-tune và replay timeout ở cuối `RESULTS.md`.

Mỗi lần train từ GUI tạo `runs/<thời-gian>/` chứa `config.json`, `metrics.jsonl` và checkpoint riêng. CLI cũng ghi config và metrics từng episode; dùng thư mục mới cho mỗi lượt để không trộn log hoặc ghi đè checkpoint của lượt trước. GUI cập nhật danh sách model sau khi train; có thể chọn model mới để xem đấu hoặc chơi. Dừng giữa episode bỏ cập nhật của episode chưa hoàn tất và lưu các cập nhật trước đó. Đóng cửa sổ khi đang train cũng chờ lưu xong. Các checkpoint gốc được giữ nguyên. Learning curves chỉ hiển thị dữ liệu của lần train hiện tại; repo chưa có log từng episode của các lần train cũ.

Kiểm tra nhanh (dùng checkpoint thật, train ngắn và kiểm tra ba tab):

```powershell
.\.venv\Scripts\python.exe test_combat.py
.\.venv\Scripts\python.exe test_visualize.py
```

---

## 📊 Kết quả thực nghiệm

Chi tiết các chỉ số đo lường (Win Rate, Episode Length, Bomb Usage, Survival Rate) được ghi nhận đầy đủ tại [RESULTS.md](RESULTS.md) và phân tích cải tiến tại [IMPROVEMENTS.md](IMPROVEMENTS.md).

## Tải đầy đủ code, kết quả và checkpoint

Nhánh `danh/bomberman-rl-results-checkpoints` lưu cả `runs/`, báo cáo Markdown, metrics, scripts và các checkpoint của project chính. Các file `.pt` dùng Git LFS; sau khi clone, chạy `git lfs pull` trong thư mục repository để lấy đủ trọng số. Python cần file trọng số thật, không phải file pointer LFS.

Checkout đối chứng `_comparison/BomberMan-Loc_update` là submodule, giữ đúng commit của nhánh `Loc_update` dùng khi so sánh. Clone với `--recurse-submodules`, hoặc chạy `git submodule update --init --recursive` sau khi clone để lấy code và checkpoint đối chứng. Thư mục test tạm, cache Python và môi trường ảo được giữ ở máy cục bộ.

## Pilot giáo viên hai bom

Trainer/CLI nhận `--attack_teacher local|two_bomb|local_combat_move`, mặc định `local`. `two_bomb` tìm vị trí bom đầu trong hai bước di chuyển và bom sau trong ba bước; được chờ sức chứa trả lại. Mỗi query xét tối đa 64 kế hoạch, kiểm tra đường sống của learner và số vị trí đối thủ sống được qua toàn bộ lửa dự báo. Nhãn các hành động đầu tốt ngang nhau có trọng số đồng đều; di chuyển phục vụ combo dùng loss combat 0.5. CNN vẫn tự chọn hành động qua mask; evaluation và GUI không gọi planner.

```powershell
python -m unittest test_combat test_visualize test_tactical_teacher -q
python pilot_two_bomb.py --output runs/two-bomb-pilot-new
```

Dùng thư mục output mới: runner không ghi đè lượt cũ. Pilot warm-start từ `runs/pressure-C-next/final_phaseC.pt`, chạy 200 episode mỗi teacher với seed 2026101303, validation 100 trận seed 21000 ở episode 0/100/200. Test cuối gồm 400 trận mỗi model (gốc, best local, best two_bomb), seed bàn 25000–25199, cả A0/A1, bootstrap 10.000 lần theo **cặp góc của cùng seed bàn**.

Runner giới hạn 15 phút preflight, 65 phút train và 40 phút evaluation, lưu config, bản chụp source, SHA256, metrics và trạng thái trong output. Metrics `teacher` chứa số query/kế hoạch/nhãn, nhãn hai bom, tổng pressure, lý do loại và tổng milliseconds; chia pressure cho accepted để lấy mức giảm trung bình. Thiếu lượt train hoặc test thì trạng thái `inconclusive` và giữ checkpoint gốc. Chỉ đủ điều kiện đề xuất model khi ActiveWR tăng ít nhất 5 điểm phần trăm so với cả hai đối chứng, tự nổ không tăng và hai CI 95% đều có cận dưới >0; vẫn cần lặp lại với nhiều seed train.

Pilot đầu tiên hoàn tất local 200 episode, two_bomb 146 episode trước giới hạn train; test cuối vẫn đủ 1.200 trận. Chưa đủ điều kiện thay model. Xem [báo cáo và giới hạn phép so sánh](runs/two-bomb-pilot-20261014/report.md); lần chạy tiếp dùng output và seed test mới.

## Pilot nhãn MOVE chiến đấu

`local_combat_move` giữ target của local teacher, chuyển riêng MOVE tăng pressure và có chứng nhận MOVE→BOMB→escape qua mask trên clone vào combat mean0.5; các MOVE không được chứng nhận giữ navigation0.2. Certificate dự báo bom đã biết, other agents STOP, dùng greedy escape có giới hạn; không bảo đảm trước future counter-bombs. Không đổi reward/CNN/optimizer/mask và không thay action CNN.

[run_combat_move_pilot.py](/D:/fpt/subject/BomberMan/run_combat_move_pilot.py) train lại cảlocal/control vàcandidate200ep từB, validation/guard chung, lưu skill tapes và test400/model trên seed mới đã đăng ký. [COMBAT_MOVE_GUIDE.md](/D:/fpt/subject/BomberMan/COMBAT_MOVE_GUIDE.md) ghi protocol, lệnh, trạng thái và files đánh giá.136 tests và smokeCPU/CUDA đã qua; xem `RESULTS.md` trước khi chạy lại hoặc chọn checkpoint. Chưa chọn mode mới làm mặc định.

T4-A1 thêm `--normalize_advantage` (mặc định tắt), chỉ chuẩn hóa advantage của actor theo learner trajectory; critic vẫn MC. [run_advantage_pilot.py](/D:/fpt/subject/BomberMan/run_advantage_pilot.py) chạy đối chứng raw/normalized với teacher local; xem [ADVANTAGE_GUIDE.md](/D:/fpt/subject/BomberMan/ADVANTAGE_GUIDE.md) cho protocol, trạng thái và kết quả.

T4-B0 thêm `--advantage_estimator gae --gae_lambda 1`; mặc định MC, critic target không đổi. [GAE_B0_GUIDE.md](/D:/fpt/subject/BomberMan/GAE_B0_GUIDE.md) ghi boundary và kiểm tra số học. Runner B1 qua174tests/smokeCPU-CUDA; sau sửa deadline qua49tests liên quan. **Full B1 hoàn tất:**200ep/nhánh,400test/model,54,87phút;bestMC/GAE đềuepisode0(weightsB),ActiveWR cảba31,75%,gateFAIL,giữB. Lượt đầu thiếu thời gian được lưu riêng. [Protocol/kết quả](/D:/fpt/subject/BomberMan/GAE_PILOT_GUIDE.md).

T4-R0 đã audit đủ300 trận validation với B/MC final200/GAE final200, review118.226 bước reward và12replay timeout. Thêm `--reward_diagnostics` (mặc định tắt), giữ reward vàweights cũ. Mọi timeout nhận reward dương chủ yếu từwall bonus, nhưng 100bước cuối không cònwall reward vàagent vẫn đặt bom gây sức ép. R0 không chứng minh farm gỗ là nguyên nhân chính.188tests PASS/1CUDA SKIP;giữB,đề xuất R1 ở mức thiết kế đối chứng. [Hướng dẫn](/D:/fpt/subject/BomberMan/R0_GUIDE.md), [báo cáo](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/R0_report.md).

## R1: pilot giảm thưởng phá tường

R1 IMPLEMENTED/TESTED, full INCOMPLETE do tiến trình bị ngắt: `reward-full-20261015-163227`. Control đủ200episode; half-wall199episode đã lưu, validation200/final200 còn thiếu. Best validation control33%/half-wall30%, chưa có test hoặcCI; giữB. 200regression testsPASS, smokeCPU/CUDA qua; audit320skill replays/6.966frames. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-20261015-163227/decision_report.md).

## Phase C full-state resume và R1 hoàn tất

R1 DONE/REVIEWED: `reward-full-resumable-20261015-194314`. Đủ 200 episode và 200 Adam update mỗi nhánh từ B; 400 trận test/model trên seed 36000–36199, cả A0/A1. Best control/half-wall đều ở episode 200. ActiveWR B/control/half-wall = 34,50%/36,25%/35,00%. Gate FAIL; giữ B và wall bonus 0,15. Thời gian chạy 49,54 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/decision_report.md).

Phase C có `training_state.pt` gồm model/Adam/scheduler/global episode/best guard/history, RNG Python/NumPy/env/Torch CPU/CUDA và context log. Atomic temp+fsync+replace, OS writer lock, `run_reward_pilot.py --resume` khôi phục episode đã xác nhận; không reset schedule hoặc chọn lại model test. 212test riêng biệt PASS; có CPU/CUDA exact resume trong tiến trình mới, interrupted episode/validation và torn-log recovery. `best/final_phaseC.pt` giữ dạng weights để chơi; loader từ chối nhầm full-state file. Resume yêu cầu cùng source/config/runtime/thiết bị; hiện chỉ hỗ trợ Phase C, deadline R1 gốc vẫn giữ.

## T4-D0 — rollout và gradient parity

T4-D0 DONE/PASS về tính đúng: 226/226 tests, gồm 14 D0; CPU/CUDA smoke đều đủ 20 episode/20 Adam updates mỗi chế độ AC và ratio chưa clip. Buffer lưu mask/old policy/teacher targets; MC raw, critic/reward/teacher/inference và mặc định AC giữ nguyên. Full states và source/ZIP/B hashes đã review. D0 không chứng minh tăng ActiveWR; giữ B. D1 clipping với cùng 4 epochs READY/design, chưa triển khai/pilot. [Báo cáo](/D:/fpt/subject/BomberMan/runs/ppo-d0-implementation-20261015/D0_report.md), [hướng dẫn](/D:/fpt/subject/BomberMan/PPO_D0_GUIDE.md).

## T4-D1 — U4/P4 hoàn tất

D1 DONE/REVIEWED: `ppo-full-20261015-220313`. Đủ 200 episode/nhánh từ B, Adam updates U4/P4=800/800; best episode U4/P4=200/200. Test400/model, seed37000–37199/cảA0-A1: ActiveWR B/U4/P4=37.50%/37.00%/32.00%. Gate FAIL; giữ checkpoint B. Runtime 106.89 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/ppo-full-20261015-220313/decision_report.md).

D1 giữ MC raw/critic MC/local teacher/reward cũ. U4/P4 cùng tối đa4 epochs và post-update KL guard target0,02/threshold0,03; chỉ khác clipping epsilon0,20. 241 regression và25 focused metadata/compat tests PASS; CPU/CUDA smoke20episode/80updates mỗi nhánh đã review. Có 500-step/4-epoch memory fixture, hủy giữa epoch khôi phục Adam, exact PPO resume qua tiến trình mới CPU/CUDA, full-state/source/ZIP/locked-model guards. CNN inference và Actor-Critic mặc định giữ nguyên.
