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
Yêu cầu Python 3.8+ và PyTorch:

```bash
# Cài đặt PyTorch và các thư viện cần thiết
pip install torch numpy
```

### 2. Huấn luyện mô hình

- **Chạy baseline Q-learning:**
  ```bash
  python train_qlearning_baseline.py
  ```

- **Huấn luyện Actor-Critic:**
  ```bash
  python train_actor_critic.py
  ```

### 3. Đánh giá mô hình

Đánh giá agent đã huấn luyện đối đầu với bot hoặc tự đối đầu:

```bash
python evaluate.py --model checkpoints/best_phaseE_4a.pt --episodes 100
```

### 4. Giao diện trực quan: train, xem đấu và chơi

```powershell
python visualize.py
# Nếu dùng virtualenv trong repo:
.\.venv\Scripts\python.exe visualize.py
```

Giao diện desktop dùng Tkinter có sẵn trong Python, chỉ cần `torch` và `numpy` như pipeline hiện tại. Trên Linux cần có gói hệ thống `python3-tk`.

- **Train trực tiếp:** Actor-Critic Phase C/D/E, chọn số episode và warm-start từ checkpoint. Phase C/D dùng 2 tác tử trên bàn 9×9; E dùng 3 tác tử trên 9×9 hoặc 4 trên 11×11. Bàn cờ hiển thị episode đang học; biểu đồ reward, win rate, số bom và loss dùng trung bình 50 episode gần nhất. Kéo thanh tốc độ để xem chậm, tạm dừng hoặc **Dừng & lưu**.
- **Agent vs Bots:** checkpoint đã train đấu 1–3 RuleBot trên bàn 11×11. Có tạm dừng, chạy từng bước, chỉnh tốc độ và bật vùng nguy hiểm.
- **Bạn vs Agent + Bots:** bạn điều khiển A0, agent đã train là A1, cộng 1–2 RuleBot. Bấm **Ván mới → Chạy**, click bàn cờ rồi dùng **WASD / phím mũi tên** để di chuyển, **Space / E** để đặt bom; cũng có nút điều khiển trên màn hình.

Chế độ người chơi mặc định **12 bước/giây**; có thể chỉnh **Bước / giây** từ 1 đến 60. Lệnh nhấn nhanh được giữ tới tick kế tiếp, kể cả đã thả phím; bom được ưu tiên trước di chuyển. Khoảng chờ giữa các tick được trừ thời gian suy luận/render để giữ tốc độ đặt. Đây là tốc độ mô phỏng: tăng tốc cũng làm bot đi nhanh và bom nổ sớm hơn theo thời gian thực (bom đếm 8 bước). GUI hiện dùng CPU; GPU không loại bỏ thời gian chờ do tick hay phần vẽ Tkinter.

Cả ba tab dùng đồ họa pixel từ `assets/`: gạch cho tường cứng, thùng cho tường phá được, robot/bom cùng màu theo A0–A3, rương `+A` tăng sức chứa và vàng `+R` tăng tầm nổ. Hiệu ứng nổ và khói dùng các frame PNG có sẵn. Ảnh được tải một lần, scale bằng Tkinter và cache theo kích thước ô; không tải lại mỗi tick. Font Emulogic dùng cho tiêu đề trên Windows, được nạp riêng trong tiến trình. Giao diện không cần thêm thư viện đồ họa.

Checkpoint mặc định là `checkpoints/best_phaseE_4a.pt`. GUI tái dùng action masking và cách lấy mẫu policy của pipeline train; checkpoint bàn khác kích thước được nội suy bằng helper có sẵn.

Mỗi lần train từ GUI tạo `runs/<thời-gian>/` chứa `config.json`, `metrics.jsonl` và checkpoint riêng. GUI cập nhật danh sách model sau khi train; có thể chọn model mới để xem đấu hoặc chơi. Dừng giữa episode bỏ cập nhật của episode chưa hoàn tất và lưu các cập nhật trước đó. Đóng cửa sổ khi đang train cũng chờ lưu xong. Các checkpoint gốc được giữ nguyên. Learning curves chỉ hiển thị dữ liệu của lần train hiện tại; repo chưa có log từng episode của các lần train cũ.

Kiểm tra nhanh (dùng checkpoint thật, train ngắn và kiểm tra ba tab):

```powershell
python test_visualize.py
```

---

## 📊 Kết quả thực nghiệm

Chi tiết các chỉ số đo lường (Win Rate, Episode Length, Bomb Usage, Survival Rate) được ghi nhận đầy đủ tại [RESULTS.md](RESULTS.md) và phân tích cải tiến tại [IMPROVEMENTS.md](IMPROVEMENTS.md).
