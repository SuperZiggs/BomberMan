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

---

## 📊 Kết quả thực nghiệm

Chi tiết các chỉ số đo lường (Win Rate, Episode Length, Bomb Usage, Survival Rate) được ghi nhận đầy đủ tại [RESULTS.md](RESULTS.md) và phân tích cải tiến tại [IMPROVEMENTS.md](IMPROVEMENTS.md).
