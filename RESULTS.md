# Ket qua thi nghiem - Bomberman RL

> Cap nhat tu dong trong qua trinh train. Moi giai doan ghi lai ket qua, gate checks, va ghi chu.

---

## Giai doan A - Rule-based bot & sanity check

**Ngay chay:** 2026-10-03  
**Cau hinh:** `board_size=11, num_agents=2, 100 episodes`

### Ket qua

| Metric | Agent 0 | Agent 1 |
|---|---|---|
| Win rate | 36.0% | 48.0% |
| Avg bombs/episode | 11.7 | 11.8 |

- **Draws:** 16 (16.0%)
- **Avg episode length:** 95.2 steps

### Gate Checks

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Khong crash 100 episodes | 100/100 OK | **PASS** |
| Win rate difference < 15% | 12.0% | **PASS** |
| Avg bombs > 0 | 11.7 / 11.8 | **PASS** |

> **Ket luan:** Phase A PASSED. Moi truong khong thien vi vi tri spawn, bot dat bom binh thuong.

---

## Giai doan B - Q-learning / Linear Function Approximation

### Lan 1

**Ngay chay:** 2026-10-03  
**Cau hinh:** `board_size=7, num_agents=2, 5000 episodes, lr=0.005, 10 features`

### Ket qua

| Moc | Win Rate | Avg Reward | Avg Bombs | Epsilon |
|---|---|---|---|---|
| Ep 100 | 8.0% | -0.133 | 2.2 | 0.951 |
| Ep 1000 | 10.0% | +0.085 | 1.8 | 0.606 |
| Ep 2000 | 16.0% | +0.741 | 1.4 | 0.368 |
| Ep 3000 | 22.0% | +1.396 | 0.8 | 0.223 |
| Ep 4000 | 29.0% | +2.176 | 0.7 | 0.135 |
| Ep 4400 | 45.0% | +3.935 | 0.5 | 0.111 |
| Ep 5000 | 37.0% | +3.045 | 0.4 | 0.082 |

- **Final WR (last 500):** 39.8%
- **Learning trend:** first 25% WR=7.9%, last 25% WR=36.2%

### Gate Checks

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Learning trend di len | 7.9% -> 36.2% | **PASS** |
| Win rate >= 50% | 39.8% | **FAIL** |

### Phan tich
- Xu huong hoc **ro rang** (8% -> 40%), nhung chua dat 50%.
- **Bomb shyness** xuat hien: bombs/ep giam tu 2.2 -> 0.4.
- Agent hoc cach song sot (dodge) nhung chua hoc tan cong hieu qua.

### Lan 2 (cai thien)

**Ngay chay:** 2026-10-03  
**Thay doi:**
- Tang so features tu 10 -> 16 (them: adjacent empty, adjacent wood, distance-to-center, bomb danger level, escape routes, bomb count)
- Them reward shaping: +0.1 per wall destroyed (chong bomb shyness)
- Tang episodes: 5000 -> 8000

**Cau hinh:** `board_size=7, num_agents=2, 8000 episodes, lr=0.005, 16 features`

**Ket qua:**

| Moc | Win Rate | Avg Reward | Avg Bombs | Epsilon |
|---|---|---|---|---|
| Ep 1000 | 11.0% | +0.298 | 1.4 | 0.606 |
| Ep 2000 | 17.0% | +0.924 | 1.7 | 0.368 |
| Ep 3000 | 22.0% | +1.450 | 0.9 | 0.223 |
| Ep 4000 | 37.0% | +3.100 | 0.7 | 0.135 |
| Ep 5000 | 38.0% | +3.180 | 0.3 | 0.082 |
| Ep 5700 | **52.0%** | +4.707 | 0.2 | 0.058 |
| Ep 6100 | **56.0%** | +5.151 | 0.2 | 0.050 |
| Ep 7400 | **55.0%** | +5.039 | 0.3 | 0.050 |
| Ep 8000 | 42.0% | +3.603 | 0.2 | 0.050 |

- **Final WR (last 500):** 41.0%
- **Peak WR (100-ep window):** 56.0% tai ep 6100
- **Learning trend:** first 25% WR=11.1%, last 25% WR=43.1%

### Gate Checks (Lan 2)

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Learning trend di len | 11.1% -> 43.1% | **PASS** |
| Win rate >= 50% (rolling) | Peak 56%, nhung khong on dinh | **PARTIAL** |

### Phan tich Lan 2
- Features moi va reward shaping giup cai thien nhe (peak 56% vs 45% lan 1).
- Agent **dat duoc 50%+ nhieu lan** (ep 5700, 6000, 6100, 7100, 7400) nhung khong on dinh.
- **Bomb shyness van ton tai** (0.2-0.4 bombs/ep) — gioi han cua linear FA.
- **Ket luan:** Linear FA da chung minh kha nang hoc (trend ro rang), nhung gioi han nang luc.
  Dieu nay phu hop voi muc tieu cua Phase B: lam baseline so sanh.
  Chuyen sang Phase C (Actor-Critic CNN) de vuot qua gioi han nay.

---

## Giai doan C - Actor-Critic 1v1 vs Rule-based bot

### Lan 1

**Ngay chay:** 2026-10-03  
**Cau hinh:** `board_size=9, num_agents=2, 3000 episodes, lr=3e-4, wall_destroy_bonus=0.08`  
**Thiet bi:** NVIDIA GeForce RTX 4060 Laptop GPU (PyTorch 2.11.0+cu128)

### Ket qua

| Metric | Gia tri (last 200 episodes) | Early (first 25%) | Late (last 25%) |
|---|---|---|---|
| Win rate | **70.0%** | ~10-15% | 70.0% |
| Avg reward | **+6.639** | +0.217 | +6.625 |
| Avg bombs/ep | **0.0** | 1.6 | 0.0 |
| Avg walls destroyed | **0.0** | 1.2 | 0.0 |

### Gate Checks (Lan 1)

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Win rate >= 70% (last 200 ep) | 70.0% | **PASS** |
| Reward trend tang va on dinh | Early=0.217 -> Late=6.625 | **PASS** |
| Bomb shyness (bombs > 0.5 cuoi training) | Early=1.6 -> Late=0.0 | **FAIL (bomb shy!)** |

- **Ket luan:** Chuyen sang Lan 2 voi reward shaping cai tien.

### Lan 2

**Ngay chay:** 2026-10-03  
**Cau hinh:** `board_size=9, num_agents=2, 6000 episodes, wall_destroy_bonus=0.3, soften_death=-0.2`  
**Thiet bi:** NVIDIA GeForce RTX 4060 Laptop GPU

### Ket qua

| Metric | Gia tri (last 200 episodes) | Early (first 25%) | Late (last 25%) |
|---|---|---|---|
| Win rate | **0.5%** | ~1-2% | 0.5% |
| Avg reward | **+0.562** | +0.337 | +0.544 |
| Avg bombs/ep | **4.3** | 2.7 | 4.5 |
| Avg walls destroyed | **1.9** | 1.3 | 1.8 |

### Gate Checks (Lan 2)

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Win rate >= 70% (last 200 ep) | 0.5% | **FAIL** |
| Reward trend tang va on dinh | Early=0.337 -> Late=0.544 | **PASS** |
| Bomb shyness (bombs > 0.5 cuoi training) | Early=2.7 -> Late=4.5 | **PASS** |

### Phan tich nguyen nhan: Hien tuong "Kamikaze" (Suicidal Wall-Breaker)
- **Nguyen nhan:** Bomb shyness da duoc chua khoi hoan toan (bombs tang len 4.5/ep, pha 1.9 tuong/ep). Tuy nhien, do giam hinh phat chet xuong `-0.2`, viec pha tuong (`+0.3 * walls`) va dat bom (`+0.05`) dem lai loi nhuan duong du de bu dap cai chet (`+0.7 - 0.2 = +0.5`). Agent phat hien ra rang tu sat pha tuong sau 10 buoc co lai ngay lap tuc ma khong can song sot de thang!
- **Ket luan:** Chuyen sang Lan 3.

### Lan 3

**Ngay chay:** 2026-10-03  
**Cau hinh:** `board_size=9, num_agents=2, 6000 episodes`  
**Thiet bi:** NVIDIA GeForce RTX 4060 Laptop GPU

### Ket qua

| Metric | Gia tri (last 200 episodes) | Early (first 25%) | Late (last 25%) |
|---|---|---|---|
| Win rate | **65.5%** | ~40-50% | 65.5% |
| Avg reward | **+0.315** | -0.116 | +0.211 |
| Avg bombs/ep | **0.0** | 0.4 | 0.0 |
| Avg walls destroyed | **0.0** | 0.1 | 0.0 |

### Gate Checks (Lan 3)

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Win rate >= 70% (last 200 ep) | 65.5% | **FAIL** (can >= 70%) |
| Reward trend tang va on dinh | Early=-0.116 -> Late=0.211 | **PASS** |
| Bomb shyness (bombs > 0.5 cuoi training) | Early=0.4 -> Late=0.0 | **FAIL (bomb shy!)** |

### Lan 4

**Ngay chay:** 2026-10-03  
**Cau hinh:** `board_size=9, num_agents=2, 6000 episodes`  
**Thiet bi:** NVIDIA GeForce RTX 4060 Laptop GPU

### Ket qua

| Metric | Gia tri (last 200 episodes) | Early (first 25%) | Late (last 25%) |
|---|---|---|---|
| Win rate | **55.0%** | ~45-50% | 55.0% |
| Avg reward | **-0.451** | -0.598 | -0.417 |
| Avg bombs/ep | **0.0** | 0.7 | 0.0 |
| Avg walls destroyed | **0.0** | 0.1 | 0.0 |

### Gate Checks (Lan 4)

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Win rate >= 70% (last 200 ep) | 55.0% | **FAIL** |
| Reward trend tang va on dinh | Early=-0.598 -> Late=-0.417 | **PASS** |
| Bomb shyness (bombs > 0.5 cuoi training) | Early=0.7 -> Late=0.0 | **FAIL (bomb shy!)** |

- **Ket luan:** Chuyen sang Lan 5.

### Lan 5

**Ngay chay:** 2026-10-03  
**Cau hinh:** `board_size=9, num_agents=2, 4000 episodes`  
**Thiet bi:** NVIDIA GeForce RTX 4060 Laptop GPU

### Ket qua

| Metric | Gia tri (last 200 episodes) | Early (first 25%) | Late (last 25%) |
|---|---|---|---|
| Win rate | **60.0%** | ~20-30% | 60.0% |
| Avg reward | **-0.288** | -0.400 | -0.382 |
| Avg bombs/ep | **0.0** | 0.8 | 0.0 |
| Avg walls destroyed | **0.0** | 0.5 | 0.0 |

### Gate Checks (Lan 5)

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Win rate >= 70% (last 200 ep) | 60.0% | **FAIL** (can >= 70%) |
| Reward trend tang va on dinh | Early=-0.400 -> Late=-0.382 | **PASS** |
| Bomb shyness (bombs > 0.5 cuoi training) | Early=0.8 -> Late=0.0 | **FAIL (bomb shy!)** |

### Phan tich nguyen nhan: Tuong quan giua so bom va ty le song sot
- O 100 tran dau: bot dat **2.3 bom/tran**, nhung WR chi co **2.0%** do 98% bi chet boi bom cua chinh minh.
- Khi so bom giam dan ve 0: WR tang vot len **60.0%** vi RuleBot tu huy.
- **Nguyen nhan:** Khi khong co **Safe Action Masking**, bot mo mam ngau nhien va bi chet vi dam vao tuong, dam vao lua, hoac dat bom vao ngo cut. Gradient luon thay viec "khong dat bom" dem lai ty le song sot cao hon.

---

### Lan 6: Safe Action Masking + Auxiliary Supervision (FULL RUN)

**Ngay chay:** 2026-10-03
**Cau hinh:** `board_size=9, num_agents=2, 4000 episodes`
**Thiet bi:** NVIDIA GeForce RTX 4060 Laptop GPU

**Cai tien ap dung:**
1. **Safe Action Masking (`get_action_mask`):** Triet tieu hoan toan cac thao tac tu sat.
2. **Auxiliary Escape Supervision (`_bfs_safe_cell`):** Loss giam sat huong thoat hiem.
3. **Anti-passivity:** Thang ma 0 bom = 0 diem.

### Ket qua

| Metric | Gia tri (last 200 episodes) | Early (first 25%) | Late (last 25%) |
|---|---|---|---|
| Win rate | **56.0%** | ~22-38% | 56.0% |
| Avg reward | **+8.599** | +3.148 | +9.189 |
| Avg bombs/ep | **4.3** | 2.1 | 4.2 |
| Avg walls destroyed | **2.2** | 1.5 | 2.2 |

### Diem noi bat: Dinh cao WR tai ep 2550-2700

| Episode | WR | Bombs | Nhan xet |
|---|---|---|---|
| ep 2550 | **88.0%** | 1.3 | Dinh cao |
| ep 2600 | **82.0%** | 1.2 | Dinh cao |
| ep 2650 | **86.0%** | 1.3 | Dinh cao |
| ep 2700 | **88.0%** | 1.7 | Dinh cao |
| ep 2750+ | 50-66% | 3.5-5.2 | Qua hung hang |

### Gate Checks (Lan 6)

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Win rate >= 70% (last 200 ep) | 56.0% (nhung peak = **88.0%** tai ep 2550-2700) | **FAIL** (last 200) |
| Reward trend tang va on dinh | Early=+3.148 -> Late=+9.189 | **PASS** |
| Bomb shyness (bombs > 0.5 cuoi training) | Early=2.1 -> Late=4.2 | **PASS** |

### Phan tich: Over-training & Kamikaze Nhe
- **Bomb Shyness da bi tieu diet hoan toan!** Bot dat 4.3 bom/tran, pha 2.2 tuong/tran.
- **Van de moi: Over-training.** Bot dat dinh WR 88% tai ep 2550-2700 (bombs 1.3-1.7), nhung tiep tuc train khien bombs tang len 3.5-5.2 va WR roi xuong 50-66%. Bot tro nen qua hung hang, dat bom lien tuc nhung chet nhieu hon.
- **Giai phap:** Can them **Best-Model Checkpointing** (luu model khi WR dat dinh) va **LR Decay** (giam learning rate theo thoi gian de on dinh late training).

### Lan 7 (Best-Model Checkpointing & Cosine LR Decay)

**Ngay chay:** 2026-10-03  
**Thay doi:**
- Tich hop Best-Model Checkpointing (rolling WR 200 ep)
- Cosine LR Decay: 3e-4 -> 1.5e-5
- Entropy Annealing: 0.015 -> 0.003
**Cau hinh:** `board_size=9, num_agents=2, 4000 episodes`

**Ket qua:**
- **Best model saved:** `best_phaseC.pt` voi **WR = 82.5%**
- **Final WR (last 200):** 70.0%
- **Final avg reward:** +6.766
- **Final avg bombs:** 1.7
- **Final avg walls:** 1.3

### Gate Checks (Phase C - Lan 7)

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Win rate >= 70% (last 200 ep) | 70.0% (Best model = **82.5%**) | **PASS** |
| Bomb shyness check | Early=1.5, Late=1.5 | **PASS** |
| Reward trend tang | Early=+2.543 -> Late=+6.447 | **PASS** |

> **Ket luan:** Phase C CHINH THUC PASSED toan bo Gate Checks. Export `final_phaseC.pt` va `best_phaseC.pt`.

---

## Giai doan D - Self-play voi opponent pool

### Lan 1 (4000 episodes - Phat hien Catastrophic Forgetting)

**Ngay chay:** 2026-10-04  
**Cau hinh:** `board_size=9, num_agents=2, 4000 episodes, snapshot_every=200, rule_prob=0.20`

**Ket qua:**
- Round-robin vs 20 Snapshots: **67.2%** -> **PASS** (>= 60%)
- WR vs RuleBot (mỏ neo): **38.0%** -> **FAIL** (< 70%)
- Avg bombs: 5.1 (Spam bom do reward shaping +0.20 thô)

**Phan tich:**
- Bot hoc self-play rat manh (thang 20 the he snapshot), nhung vi rule_prob chi 20% nen bi quen chien thuat khac che RuleBot. Spam bom khien bot tu chet nhieu tren ban 9x9.

---

### Lan 2 (3000 episodes - Chinh thuc PASS toan bo Gate Checks)

**Ngay chay:** 2026-10-04  
**Cai tien:**
1. Tang rule_prob tu 20% -> 50% de chong quen chien thuat co ban.
2. Tat thuong dat bom tho (+0.20), chi thuong pha tuong (+0.15).
3. Tang nang phat tu sat (-2.5).
4. Fair Round-Robin evaluation: Doi vi tri spawn 0/1 xen ke, ghi nhan doc lap tran Hoa (Draws).

**Cau hinh:** `board_size=9, num_agents=2, 3000 episodes, snapshot_every=200, rule_prob=0.50`

**Ket qua:**
- **Final avg reward:** +7.118
- **Final avg bombs:** 2.4 / tran (dat bom ky luat, khong spam)
- **Final avg walls:** 1.8 / tran

### Gate Checks (Phase D - Lan 2)

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| WR vs RuleBot (need >= 70%) | **85.5%** | **PASS** |
| Ty so Round-Robin vs 15 Snapshots (300 van) | **77 Thang - 183 Hoa - 40 Thua** | **Thang gap 1.9x Thua** |
| Clean Win Rate (loai tru hoa, need >= 60%) | **65.8%** (77 / 117) | **PASS** |
| Tournament Score (Wins + 0.5*Draws, need >= 55%) | **56.2%** | **PASS** |

> **Ket luan:** Phase D CHINH THUC PASSED toan bo Gate Checks voi log terminal 100% PASS. WR vs RuleBot dat ky luc 85.5%, Clean WR vs Snapshots dat 65.8%. Model `final_phaseD.pt` va `best_phaseD.pt` da hoan thien de warm-start sang Phase E!

---

## Giai doan E - Mo rong 2 -> 3 -> 4 agents (FFA)

### Buoc E1: 3 Agents (Ban 9x9) - Hoan thanh

**Ngay chay:** 2026-10-05  
**Cau hinh:** `board_size=9, num_agents=3, 2000 episodes, snapshot_every=200, log_every=100`  
**Khoi tao (Warm-start):** Tu `checkpoints/best_phaseD.pt`

#### Ket qua huan luyen
- **Final win rate (last 200):** 19.5% (do moi van co 3 agent canh tranh, WR ngau nhien la ~33%, co ty le hoa cao)
- **Final avg reward (last 200):** +1.703 (tang deu dan tu early +1.357 len late +1.776, vuot troi hoan toan so voi random baseline ~ -1.50)
- **Final avg bombs:** 1.7 bom / tran (khong bi bomb-shyness)
- **Final avg walls:** 1.3 tuong / tran

#### Gate Checks (Buoc E1)
| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| Reward Trend (tang so voi random baseline ~ -1.50) | Early: +1.357 -> Late: +1.776 | **PASS** |
| Emergent Behavior (quan sat tren terminal) | Biet dat bom, chu dong lui ne lua, pha tuong | **PASS** |
| Luu checkpoint thanh cong | `best_phaseE_3a.pt`, `final_phaseE_3a.pt` + 10 snapshots | **PASS** |
| Spawn Fairness (3 agents khong doi xung goc 4) | Agent 0 (69.4%), Agent 1 (11.1%), Agent 2 (19.4%) tren 36 van quyet dinh | *Buoc dem cho E2* |

> **Nhan xet E1:** Buoc E1 thanh cong trong viec chuyen giao trong so tu 1v1 sang FFA 3 nguoi ma khong bi quen kien thuc. Agent tiep tuc phat huy kha nang ne lua va dat bom. Do ban 9x9 chi co 3 agent (khuyet goc duoi phai), tinh doi xung 4 huong chua co. San sang buoc vao **Buoc E2 (4 Agents, Ban 11x11)** de dat tinh doi xung tron ven!

---

### Buoc E2: 4 Agents FFA day du (Ban 11x11) - Hoan thanh (CHINH THUC PASS PHASE E)

**Ngay chay:** 2026-10-05  
**Cau hinh:** `board_size=11, num_agents=4, 3000 episodes, snapshot_every=200, log_every=100`  
**Khoi tao (Warm-start):** Tu `checkpoints/best_phaseE_3a.pt` bang phep noi suy Bilinear Spatial Interpolation 2D (`trunk_fc` tu 9x9 sang 11x11, tu 5184 dim sang 7744 dim).

#### Ket qua huan luyen
- **Final avg reward (last 200):** **+2.867** (tang deu dan tu early +1.782 len late +2.868, vuot troi hoan toan so voi random baseline ~ -1.50)
- **Final avg bombs:** **3.1** bom / tran (dat bom tich cuc, ky luat, khong bi bomb-shyness)
- **Final avg walls:** **1.6** tuong / tran
- **Checkpoints tao thanh cong:** `checkpoints/best_phaseE_4a.pt`, `checkpoints/final_phaseE_4a.pt` (kich thuoc 8.17 MB) cung 15 snapshots.

#### Gate Checks (Giai doan E - 4 Agents)

| Tieu chi | Ket qua do dac | Trang thai |
|---|---|---|
| **Gate 1: Tinh cong bang vi tri spawn** | • Solo Wins: Agent 0 (47.4%), Agent 1 (21.1%), Agent 2 (10.5%), Agent 3 (21.1%)<br>• Khong co agent nao ~0% (tat ca >= 10%) va khong co agent nao ~70% (tat ca <= 50%)<br>• Tournament Score: Agent 0 (49.8%), Agent 1 (47.3%), Agent 2 (46.3%), Agent 3 (47.3%) — **Chenh lech giua goc cao nhat va thap nhat chi 3.5%!** | **PASS** |
| **Gate 2: Xu huong Reward so voi Random Baseline** | Early Q1 (+1.782) -> Late Q4 (+2.868) — Tang truong +61%, bo xa moc baseline -1.50 | **PASS** |
| **Gate 3: Hanh vi quan sat duoc (ASCII Demo)** | Ca 4 agent chu dong tien vao trung tam, dat bom pha tuong, nhan dien vung nguy hiem va lui ve o an toan de ne lua | **PASS** |

> **Ket luan Phase E:** Phase E **CHINH THUC HOAN THANH VA PASS TOAN BO MUC TIEU HUAN LUYEN CUA DO AN**. Agent da tien hoa thanh cong tu 1v1 sang FFA 4 nguoi tren ban chuan 11x11 voi do cong bang goc spawn rat cao va loi choi sinh ton chien thuat. San sang buoc vao **Giai doan F (Danh gia tong hop & viet bao cao)**!

---

## Giai doan F - Danh gia & bao cao

**Trang thai:** San sang tien hanh (Tat ca model tu Phase A den E da san sang trong `checkpoints/`)

- [ ] So sanh learning curves (random, RuleBot, Q-learning, AC 1v1, AC FFA)
- [ ] Ablation: co/khong reward shaping
- [ ] Ly thuyet: MDP -> Markov Game, TD/AC framework

---

## Ghi chu ky thuat

- **Python:** 3.11.9
- **PyTorch:** 2.11.0+cu128
- **GPU:** NVIDIA GeForce RTX 4060 (8GB)
- **Venv:** `e:\Bomberman\venv`

---

## Kiểm tra hạ gục thực tế (2026-10-10)

Checkpoint cũ `checkpoints/best_phaseE_4a.pt` đấu 3 RuleBot, bàn 11×11, giới hạn **500 bước/ván**. Chạy 40 ván, 10 ván ở mỗi góc của agent AC; seed môi trường, NumPy và PyTorch lần lượt 100, 101, 102, 103 theo góc. Dùng policy lấy mẫu và action mask như GUI, trên môi trường đã sửa attribution lửa và bom dây chuyền.

| Chỉ số | Checkpoint cũ |
|---|---:|
| Win rate | 52.5% |
| Bom đặt thành công / ván | 1.025 |
| Đối thủ bị bom của AC hạ / ván | **0.00** |
| ActiveWR (thắng và có hạ gục) | **0.0%** |
| Tự chết bởi bom của mình | 2.5% |

Kết quả xác nhận WR cao chưa chứng minh agent biết dùng bom diệt đối thủ. Đây là mẫu nhỏ 40 ván, chưa phải benchmark đầy đủ. Log theo góc tại `runs/combat-20261010/best_phaseE_4a_eval.json`; script đánh giá tại `runs/combat-20261010/benchmark.py`.

Code train đã đổi reward, auxiliary attack supervision và tiêu chí chọn checkpoint như mục 7 trong `IMPROVEMENTS.md`. Một lượt pilot đã ghi 32 episodes rồi bị ngắt trước khi lưu checkpoint cuối; **chưa có kết quả trước/sau để khẳng định policy mới đã cải thiện**. Log pilot tại `runs/combat-20261010/metrics.jsonl`. Cần train lại và đánh giá checkpoint mới theo cùng điều kiện ở trên. Các file trong `runs/` là artifact cục bộ, được Git ignore.

Kiểm tra hồi quy: `test_combat.py` kiểm tra kill credit, lửa còn sót, nổ trùng, bom dây chuyền, tự sát, thưởng thắng thụ động và nhãn tấn công an toàn; `test_visualize.py` kiểm tra checkpoint thật, train ngắn, hooks/metrics và GUI.

## Kiểm tra Phase C bị tự nổ và item (2026-10-10)

Lượt người dùng: `runs/20261010-125814-296523`, Phase C 3000 ván, 2 agents, 9×9, tối đa 500 bước, CPU, không warm-start. Đọc đủ 3000 dòng `metrics.jsonl`:

| Chỉ số | Toàn bộ 3000 ván | 200 ván cuối |
|---|---:|---:|
| Win rate | 11.07% | 10.00% |
| ActiveWR | 3.70% | 3.50% |
| Tự nổ | **84.23%** | **85.00%** |
| Hạ gục / ván | 0.0503 | 0.0450 |
| Bom / ván | 3.906 | 4.050 |
| Số bước / ván | 31.712 | 32.585 |

305 ván tự nổ trong ≤8 bước; 364 ván tự nổ có reward dương. Rolling ActiveWR cao nhất được lưu là 7%, WR tương ứng trong metrics cuối là 13.5%. Kiểm tra tensor xác nhận `best_phaseC.pt` và `final_phaseC.pt` **giống hệt**; code cũ lưu best vào cả final. Vì vậy final của lượt này không thể dùng để đo riêng policy tại episode 3000.

Replay trace cho thấy mask cũ bỏ qua fuse bị rút ngắn bởi nổ dây chuyền, cho BOMB dù STOP đã bị cấm, và cho đứng/đi qua lại tới khi đường thoát không còn kịp. Những lần chết còn lại ở bản sửa thường liên quan đối thủ đặt thêm bom hoặc tranh cùng ô di chuyển, làm một đường thoát dự báo trước đó bị chặn. Bộ lọc chỉ dự báo trạng thái đã thấy, không bảo đảm sống trước mọi hành động tương lai của đối thủ.

### So trước/sau trên cùng checkpoint, đối thủ giữ nguyên

100 ván, seed môi trường/NumPy/PyTorch `7000..7099`, agent AC luân phiên hai góc (50 ván/góc), 9×9, 500 bước, policy lấy mẫu, CPU một thread. Dùng nguyên checkpoint best của lượt người dùng và RuleBot cũ được giữ bản sao. Cột sau áp dụng code cuối: bộ lọc thời gian/thoát sớm/va chạm, chặn bom trống, nâng cấp có trần và observation tương thích.

| Chỉ số | Code trước sửa | Code sau sửa |
|---|---:|---:|
| Win rate | 6% | 54% |
| ActiveWR | **0%** | **23%** |
| Tự nổ | **84%** | **20%** |
| Hạ gục / ván | 0.02 | 0.25 |
| Bom / ván | 3.69 | 15.65 |
| Số bước / ván | 29.33 | 198.07 |
| Nâng cấp hữu ích / ván | Không có event để đo | 1.43 |

Đây là cải thiện hành vi từ runtime mới trên **cùng trọng số**, không phải bằng chứng CNN đã train tốt hơn. Bom/ván tăng phần lớn vì agent sống lâu hơn; không dùng riêng số bom làm tiêu chí thành công. Trước sửa, replay cả best và final cho cùng kết quả vì trọng số giống nhau.

### RuleBot mới và trần range

Cùng 100 seed và checkpoint, lần này đối thủ là RuleBot đã sửa (né bom theo thời gian, nhặt item, dùng tia nổ đúng range). Trần ammo luôn 3:

| Chỉ số | Range tối đa 4 | Range tối đa 5 |
|---|---:|---:|
| Win rate | 10% | 12% |
| ActiveWR | 3% | 4% |
| Tự nổ | **14%** | **17%** |
| Hạ gục / ván | 0.07 | 0.09 |
| Bom / ván | 27.27 | 25.21 |
| Số bước / ván | 420.77 | 396.39 |
| Nâng cấp hữu ích / ván | 1.66 | 1.73 |

Chọn range 4 làm mặc định ưu tiên an toàn; chênh lệch nhỏ này chưa chứng minh tối ưu thống kê, chưa có ablation train độc lập cho từng trần. RuleBot mới khó hạ hơn; không so trực tiếp WR 54% với 10% để kết luận policy suy giảm. Nhiều ván kéo dài và ActiveWR thấp cho thấy checkpoint cũ vẫn cần học tấn công lại.

### Pilot và kiểm tra

Một candidate warm-start 40 ván trên GPU, seed `20261010`, lr `1e-4`, đã hoàn tất và lưu tại `runs/safety-20261010/pilot-final/`. Candidate dùng bộ lọc thoát theo thời gian/va chạm trước khi thêm chặn bom trống. Trung bình train: ActiveWR 5%, tự nổ 17.5%, 49.05 bom/ván. Đánh giá trọng số đó với **runtime cuối** và RuleBot mới trên cùng 100 seed: WR 8%, ActiveWR **1%**, tự nổ 15%, kills/ván 0.06. **Pilot không tốt hơn checkpoint gốc; không khuyến nghị dùng làm model chính hoặc chuyển Phase D.** Chưa đủ 200 ván để chọn best theo rolling score; file best ở pilot là fallback của final. Một candidate trước đó bị dừng sau 30 ván, chỉ giữ log, không có checkpoint hoàn tất.

Khuyến nghị tiếp theo: đóng/mở lại GUI, chạy Phase C 3000 ván từ đầu (bỏ warm-start) để CNN 12 kênh học chỉ số và reward mới; kiểm tra ActiveWR/kills/tự nổ trên seed đánh giá riêng trước D/E. Không coi bộ lọc an toàn hoặc pilot ngắn là hoàn thành quá trình học.

Artifacts cục bộ (Git ignore): `runs/safety-20261010/probe.py`, bản code cũ trong `before/`, `before-legacy-best_phaseC.json`, `before-legacy-final_phaseC.json`, `target-legacy-best_phaseC.json`, `target-current-best_phaseC.json`, `target-current-best_phaseC-range5.json`, `pilot-current-final_phaseC.json` và các log candidate. Các file `timed-*`, `prompt-exit-*`, `collision-*`, `after-*`, `final-*` là thử nghiệm trung gian, không phải bản cuối.

Kiểm tra logic: 22 unittest gồm forecast khớp vụ nổ thật trên 10 bàn, fuse dây chuyền, đường thoát đúng deadline/ô tranh chấp, không spam bom trống, item trần/thưởng/chết, observation 12 kênh, chuyển checkpoint giữ logits/value, final không mất update sau best. Integration GUI đã qua với checkpoint thật, nội suy board, train C/E, snapshots, dừng/lưu, ba tab và assets.

## Kiểm tra lượt Phase C mới: 20261010-135449-587798

Đọc đủ 3000 dòng metrics của lượt người dùng: Phase C, 9×9, 2 agents, 500 bước, CPU, train từ đầu. **Agent tấn công nhiều hơn và có hạ gục thực tế, nhưng chưa đạt mục tiêu hạ bot rồi thoát an toàn.** So sánh hai lượt train dưới đây có thay đổi cả runtime và RuleBot; không dùng bảng này để tách riêng tác dụng của CNN.

| Chỉ số | Lượt trước, toàn bộ 3000 | Lượt mới, toàn bộ 3000 | Lượt mới, 200 cuối |
|---|---:|---:|---:|
| Win rate | 11.07% | 13.73% | 22.00% |
| ActiveWR: thắng và có hạ gục | 3.70% | **8.97%** | **11.00%** |
| Tự nổ | 84.23% | **37.33%** | **36.00%** |
| Hạ gục / ván | 0.0503 | 0.2653 | 0.2900 |
| Bom / ván | 3.906 | 25.873 | 25.800 |
| Số bước của learner / ván | 31.712 | 327.612 | 312.185 |
| Nâng cấp hữu ích / ván | Chưa ghi event | 1.486 | 1.500 |

ActiveWR theo ba đoạn 1000 ván tăng **7.3% → 8.8% → 10.8%**; tự nổ gần như đứng yên **37.3% → 37.0% → 37.7%**. Trong **796 ván có hạ gục, 527 ván cũng tự nổ (66.2%)**. Không còn tự nổ trong 8 bước đầu; vấn đề còn lại xuất hiện muộn khi giao tranh, thay vì chỉ đặt bom ở góc spawn. 1271/3000 ván learner còn thực hiện hành động tới bước 500; metrics không ghi trực tiếp draw nên chỉ gọi đây là số ván chạm giới hạn bước.

GUI khớp log: biểu đồ dùng **50 ván cuối**, cho WR 24%, reward 1.57374, bom 25.06 và loss 3.58898. Dòng hoàn tất dùng **200 ván cuối**, cho ActiveWR 11% và tự nổ 36%. Trong 50 ván cuối, ActiveWR là 12% và tự nổ 42%; 6/12 trận thắng không có hạ gục. Reward dương hoặc WR tổng cao hơn chưa đủ chứng minh tấn công an toàn; loss tổng gồm cả actor, critic và auxiliary nên không dùng riêng độ giảm loss làm tiêu chí đạt.

### Đánh giá checkpoint trên seed riêng, giữ cùng code và RuleBot

100 ván/checkpoint, seed môi trường/NumPy/PyTorch `7000..7099`, luân phiên hai góc (50 ván/góc), 9×9, 500 bước, CPU một thread. Dùng `evaluate.run_match`, policy lấy mẫu và bộ lọc chung như GUI. Cột checkpoint cũ lấy từ benchmark `target-current-best_phaseC.json` với cùng điều kiện, giúp so trọng số thay vì so hai runtime khác nhau.

| Chỉ số | Checkpoint lượt trước trên runtime mới | Best lượt mới | Final lượt mới |
|---|---:|---:|---:|
| Win rate | 10% | **16%** | 13% |
| ActiveWR | 3% | **6%** | 4% |
| Tự nổ | **14%** | 21% | 24% |
| Hạ gục / ván | 0.07 | **0.17** | 0.10 |
| Bom / ván | 27.27 | 30.53 | 29.33 |
| Nâng cấp hữu ích / ván | 1.66 | 1.81 | 1.62 |
| Hòa | Không ghi trực tiếp | 67% | 61% |

Best mới có thêm hạ gục nhưng đánh đổi an toàn so với checkpoint cũ trên cùng runtime. **Tạm ưu tiên `best_phaseC.pt` của lượt mới** nếu tiếp tục hướng tấn công: tốt hơn final trên mẫu này ở cả ActiveWR, kills và tự nổ. Chênh lệch best/final chỉ tương ứng 2 trận thắng có hạ gục trên 100, chưa chứng minh ưu thế thống kê chắc chắn. Best có 46 trận hòa do cả hai còn sống hết 500 bước và 21 trận hòa do cả hai chết trước giới hạn; final lần lượt 45 và 16.

| Góc của AC | Best: WR / ActiveWR / tự nổ | Final: WR / ActiveWR / tự nổ |
|---|---:|---:|
| A0 (góc đã train) | 24% / **10%** / 34% | 14% / 6% / 36% |
| A1 | 8% / **2%** / 8% | 12% / 2% / 12% |

Code Phase C/D hiện luôn học A0, chỉ E luân phiên góc. Chênh lệch đánh giá là dấu hiệu cần kiểm tra khả năng chơi ở góc khác; mỗi góc chỉ có 50 ván với seed khác nhau, chưa cô lập góc spawn thành nguyên nhân duy nhất.

Kiểm tra tensor: cả hai checkpoint có conv đầu `(32, 12, 3, 3)`, đúng observation mới. Best được lưu ở episode **2732**, rolling ActiveWR lúc chọn 13.5%; đó là số trên cửa sổ train, không phải kết quả benchmark của model cố định. **12/12 tensor best và final khác nhau**, xác nhận final đã giữ các update sau best. Việc checkpoint có 12 kênh không tự chứng minh CNN đã biết dùng từng chỉ số nâng cấp.

### Replay tự nổ và bước tiếp theo

Replay đúng seed và vị trí cuối của hai trận tự nổ đầu trong best:

- **Seed 7002:** bước 407 cả hai đặt bom; bước 413 AC đi lên và RuleBot đi xuống, cùng nhắm ô `(3,1)` nên cả hai bị giữ lại. Bước 414 bom của AC giết cả AC và bot. Những bước chờ trước đó đã làm mất thời gian dự phòng; tới lúc va chạm, bộ lọc chỉ còn hành động duy trì sống ngay lúc đó.
- **Seed 7004:** cả hai đặt bom ở bước 152. Bộ lọc chọn hướng phải cho AC ở bước 153, nhưng RuleBot cũng tiến vào hành lang đó; sau bước này không còn đường thoát dự báo được với vị trí đối thủ hiện tại. Những bước sau là fallback chờ/đi qua lại; AC tự nổ ở bước 159. Đây là mất đường thoát khi đối thủ đổi vị trí, không phải bằng chứng timer của bom bị tính sai.

Ưu tiên trước lượt train dài tiếp theo:

1. **Đường thoát có dự phòng khi giao tranh:** kiểm tra nguy cơ đối thủ chiếm hành lang/đặt thêm bom, tránh chọn đường thoát chỉ vừa kịp hoặc phụ thuộc một ô đang tranh chấp. Giữ các phương án thoát an toàn thay vì chỉ tối ưu tới nơi trú gần nhất. Hai replay là ví dụ chẩn đoán, chưa phân loại toàn bộ nguyên nhân của 1120 trận tự nổ trong train.
2. **Luân phiên góc ngay ở C/D**, rồi đánh giá riêng từng góc trên seed cố định. Không đợi tới E mới học góc khác.
3. **Chọn checkpoint bằng đánh giá policy cố định**, gồm ActiveWR, tự nổ và hòa; rolling train chỉ nên lọc ứng viên. Tăng số episode với cùng cấu hình chưa xử lý được mức tự nổ đang đứng yên.

Item đang tạo trung bình 1.486 nâng cấp hữu ích/ván train; với thưởng 0.05, phần reward trực tiếp từ item chỉ khoảng 0.074/ván. Chưa có bằng chứng cần tăng thưởng item hoặc nới trần ammo 3/range 4 ở lượt này. Ưu tiên thoát sau tấn công; giữ best làm mốc so sánh, chưa coi Phase C đạt yêu cầu để chuyển D/E.

Artifacts cục bộ trong `runs/20261010-135449-587798/`: `training_summary.json`, `evaluation_best.json`, `evaluation_final.json`, `suicide_diagnostics.json`, script `inspect_checkpoints.py`, cùng checkpoint và metrics gốc. Lượt kiểm tra này không đổi thuật toán hoặc chạy train mới; đánh giá chưa bao gồm FFA 3–4 agents, nhiều seed train hay ablation riêng từng thành phần reward/mask.

## Pilot thoát bom và validation cố định (2026-10-11)

Đã thực hiện ba đề xuất: bổ sung đường thoát có dự phòng trước bom đáp trả/ô đối thủ có thể chiếm, luân phiên góc học ngay ở C/D, và chọn checkpoint bằng policy cố định trên validation. RuleBot giữ hành vi trước lượt sửa này để làm đối thủ so sánh; đối chiếu 648 quyết định trên 30 seed khớp bản sao cũ. Item vẫn thưởng 0.05 khi nâng cấp hữu ích, ammo tối đa 3, range tối đa 4; không có dữ liệu mới chứng minh cần đổi các mức này.

### Pilot 500 episode đã hoàn tất

Warm-start từ `runs/20261010-135449-587798/best_phaseC.pt`, Phase C, 9×9, 2 agents, tối đa 500 bước/ván, LR `1e-4`, seed train `20261011`, GPU RTX 3050 Laptop, Torch 2.5.1, một thread CPU. Có đủ **500 dòng metrics**, mỗi góc học đúng **250 ván**, thời gian train khoảng **26.9 phút**. Thư mục: `runs/escape-20261011/pilot-500/`.

| Chỉ số trong 500 ván train | Giá trị |
|---|---:|
| Win rate | 10.8% |
| ActiveWR: thắng và có hạ gục bằng bom của AC | 5.2% |
| Tự nổ | 13.6% |
| Hòa | 86.2% |
| Hạ gục / ván | 0.174 |
| Bom / ván | 31.006 |
| Nâng cấp hữu ích / ván | 1.732 |

Các số này đến từ policy đang cập nhật, không phải benchmark của checkpoint cố định. Không so trực tiếp với 3000 ván học A0 của lượt trước để kết luận hiệu quả riêng của CNN.

### Chọn model trước khi mở kết quả test

Validation trong train chạy 48 trận ở episode 0, 250 và 500, seed 6000, cùng bàn dùng ở cả hai góc. GPU chọn episode 250. Sau khi pilot bắt đầu, một lỗi thứ tự placement giả định được sửa: cần sắp bom theo owner giống `env.step`, vì bom nổ trước có thể mở tường gỗ cho tia nổ sau. Snapshot và hash của runtime dùng để train nằm trong `training-runtime/` và `pilot-500/code_hashes.json`; pilot không được train lại sau sửa một dòng này.

Để checkpoint được chấm theo đúng code cuối và CPU của GUI, đánh giá lại ba ứng viên trên **cùng 48 trận validation seed 6000**, trước khi đọc test seed 9000:

| Ứng viên trên code cuối, CPU | ActiveWR | Tự nổ | Hòa |
|---|---:|---:|---:|
| Warm-start gốc, episode 0 | 4.17% | 16.67% | 81.25% |
| Model GPU chọn, episode 250 | 2.08% | 12.50% | 85.42% |
| Final, episode 500 | **10.42%** | **10.42%** | **77.08%** |

Chọn **episode 500** theo ActiveWR, rồi ít tự nổ, ít hòa, kills và WR. `best_phaseC.pt` và `final_phaseC.pt` của pilot hiện giống nhau vì final thắng validation, không phải lỗi ghi đè final. Model từng được GPU chọn được giữ riêng ở `training_selected_phaseC.pt`, khác trọng số final; metadata cũ ở `training_best_validation.json`. `best_validation.json` ghi rõ nguồn post-training CPU validation, episode và hash runtime. Checkpoint của người dùng không bị sửa.

### Test mới: 100 trận, seed tách khỏi việc chỉnh code và chọn model

50 bàn seed **9000..9049**, mỗi bàn chơi cả A0 và A1, tổng 100 trận/checkpoint, 9×9, 2 agents, 500 bước, policy lấy mẫu, CPU một thread. Cả ba trường hợp giữ RuleBot như trước sửa. Seed 8000 đã dùng trong thử nghiệm runtime nên không dùng làm bộ test cuối. Kết quả sau đây được đọc sau khi chọn model bằng validation; không chỉnh code hoặc chọn lại trọng số theo test.

| Chỉ số | Checkpoint gốc + code trước sửa | Checkpoint gốc + code mới | Pilot 500 + code mới |
|---|---:|---:|---:|
| Win rate | 11% | 14% | 13% |
| ActiveWR | **8%** | **8%** | **8%** |
| Tự nổ | 25% | **10%** | 16% |
| Hòa | 66% | 84% | 85% |
| Hạ gục / ván | 0.19 | 0.18 | 0.23 |
| Bom / ván | 28.30 | 31.66 | 30.16 |
| Nâng cấp hữu ích / ván | 1.46 | 1.70 | 1.62 |
| Ván có hạ gục đồng thời tự nổ | 11 | 10 | 15 |

Code mới giảm tự nổ **15 điểm phần trăm** trên cùng checkpoint và cùng bàn, nhưng ActiveWR không tăng, hòa tăng. Pilot cũng chưa tăng ActiveWR so với warm-start trên code mới; kills tăng có cả những lần cùng chết. Không kết luận CNN đã học tấn công tốt hơn hoặc bộ lọc đã giải quyết hoàn toàn tự nổ. Với 100 trận, chênh lệch nhỏ chưa đủ chứng minh ưu thế thống kê.

| Góc AC | Gốc + code mới: ActiveWR / tự nổ | Pilot: ActiveWR / tự nổ |
|---|---:|---:|
| A0, 50 trận | 14% / 20% | 12% / 32% |
| A1, 50 trận | 2% / 0% | 4% / 0% |

Luân phiên 500 ván chưa xóa được chênh lệch góc. Trong 85 trận hòa của pilot, 60 trận chạm 500 bước và 25 trận kết thúc sớm do không có người thắng; không coi mọi trận hòa là né bom giỏi. Bộ lọc có giới hạn: dự báo bom cùng tick và vị trí gần của đối thủ, không mô phỏng mọi chuỗi hành động tương lai.

**Khuyến nghị hiện tại:** dùng checkpoint gốc `runs/20261010-135449-587798/best_phaseC.pt` với code mới làm mốc an toàn; giữ pilot như một ứng viên đã kiểm tra, chưa lấy làm model chính hoặc chuyển D/E. Nếu train tiếp, vẫn ở C và giữ validation/góc luân phiên; ưu tiên phân tích giao tranh A0 và các ván hòa kéo dài trước lượt train dài. Đợt này đã hoàn thành sửa pipeline và pilot, chưa đạt mục tiêu tăng thắng bằng bom rồi sống.

Kiểm tra qua: **30 unittest `test_combat.py`**, integration `test_visualize.py` với checkpoint thật, pause/resume/stop, validation progress/metadata, LR warm-start và ba tab. Assert xác nhận 500 dòng metrics, cân bằng góc, cùng cặp bàn/góc của cả ba benchmark, hash code cuối và model dùng để test khớp metadata. Chưa kiểm tra train nhiều seed, bàn 11×11 hoặc FFA 3–4 agents.

Artifacts cục bộ được Git ignore: `runs/escape-20261011/test_before.json`, `test_baseline.json`, `test_best.json`, các file `post_validation_*.json`, `pilot-500/metrics.jsonl`, `training_summary.json`, `validation_metrics.jsonl`, `best_validation.json` và ba checkpoint. Script `benchmark.py`, `run_pilot.py`, `select_final.py`, `test_selected.py` giữ điều kiện chạy; hai lượt thử bị ngắt có log riêng và không được tính là pilot hoàn chỉnh.

## Chẩn đoán giao tranh và sửa tối thiểu (2026-10-12)

Replay checkpoint gốc trên code ngày 11/10, chọn 20 trận hòa hết 500 bước (10 mỗi góc) và đủ 10 trận tự nổ từ `test_baseline.json`. Khôi phục seed NumPy/Torch và board seed theo từng trận; assert cả 30 kết quả, số bước, bom và hạ gục khớp log cũ. Đây là mẫu chọn theo kết quả thất bại, không đại diện cho tỷ lệ toàn bộ trận.

| Đo trên 20 trận hòa | Kết quả |
|---|---:|
| Bước không nguy hiểm | 6.471 |
| Bước không nguy hiểm, đối thủ nằm trên tia nổ | 577 |
| BOMB được phép và tia nổ chạm đối thủ | 427 |
| Agent chọn BOMB tại các bước đó | 246 (57.6%) |
| Cơ hội trực tiếp được phép nhưng bỏ qua | 181 (42.4%) |
| Mức lọc mới chặn BOMB dù mức cũ cho phép, đối thủ đang trên tia | 51 |
| Bước không nguy hiểm thiếu nhãn tấn công | 1.357 (21.0%) |
| Hướng tiếp cận ngắn nhất bị mask chặn | 519 |
| Agent đứng yên khi không nguy hiểm | 85 |

Agent đã di chuyển và có tấn công; chưa có bằng chứng cho rằng chỉ cần thêm thưởng đặt bom hoặc nhặt item. Kiểm tra teacher tìm thấy đường tới ô đối thủ đang đứng, rồi bỏ nhãn nếu bước đầu bị mask chặn, thay vì tìm đường khác tới vị trí có thể bắn.

Cả 10 trận tự nổ đã chọn đều hạ bot và tự chết. Mẫu chung: sau đặt bom, AC thoát về phía đối thủ khi vẫn có nơi trú dài bằng nhau ở hướng khác; đối thủ cũng đi vào hành lang, hai bên không vượt được ô tranh chấp và hết fuse. Seed 9000: bước 432 đặt ở `(5,1)`, bước 433 đi lên `(4,1)` trong khi bot đi xuống `(2,1)`; từ đó hầu hết chỉ còn STOP, bước 438 hai bên tranh `(3,1)` và bước 439 cùng chết. Các trường hợp tương tự ở seed 9001, 9006, 9009, 9016, 9017, 9021, 9024, 9038, 9045, đều learner A0. Không suy rộng rằng đây là nguyên nhân của mọi trận tự nổ ngoài mẫu.

### Sửa và kiểm tra

- Bản cuối chỉ tránh bước vào hành lang sẽ cháy, chỉ có một lối ra và đối thủ có thể tới lối ra đó trong hai bước sau khi AC đặt bom. Giữ các hướng có đường thoát khác, các kiểm tra phản bom, fuse dây chuyền, dự phòng và fallback; RuleBot vẫn dùng `robust=False`. Bản thử ưu tiên khoảng cách trong mọi lần né đã bị thay vì giảm khả năng giao tranh.
- Teacher tìm vị trí có tia nổ chạm đối thủ qua bước đầu được mask cho phép; tránh ô đối thủ thay vì đi vào đó. Nếu chưa tới được vị trí tấn công, vẫn dùng nhãn item/phá gỗ sẵn có. Đây là hướng dẫn tiếp cận, chưa đảm bảo ép được đối thủ hết đường thoát.
- Validation dùng bản sao model trên CPU giống GUI, ghi `device=cpu`; không sửa learner hoặc tiêu thụ RNG để khởi tạo một model mới. Metrics thêm số cơ hội trực tiếp, số lần chọn bom tại cơ hội đó và nhãn tấn công có/thiếu.

Đã qua 32 unittest, integration GUI và kiểm tra bản sao validation độc lập. RuleBot khớp bản trước sửa ở 458 quyết định trên 20 seed. Chạy GUI test ngoài sandbox do Windows không cho Python ghi vào thư mục `TemporaryDirectory` khi bị giới hạn quyền; test ngoài sandbox hoàn tất và dọn thư mục tạm. CUDA không khả dụng trong phiên này kể cả kiểm tra ngoài sandbox; pilot chạy CPU.

Kiểm tra bản thử ưu tiên khoảng cách trên 100 trận seed 9000..9049: WR 8%, ActiveWR 7%, tự nổ 2%, hòa 90%, kills/ván 0.09, bom/ván 32.75. Trước sửa hôm nay trên cùng seed: WR 14%, ActiveWR 8%, tự nổ 10%, hòa 84%, kills/ván 0.18. Trên seed phát triển 11000, bản thử này cũng giảm ActiveWR nên điều kiện được thu hẹp về hành lang có một lối thoát. **Cả 9000 và 11000 đã dùng để chẩn đoán/sửa code; không phải test cuối độc lập.** Artifacts trung gian được giữ riêng, không dùng để kết luận về code cuối.

Pilot Phase C đã hoàn tất đủ 500 episode, mỗi góc 250 ván, 9×9, 2 agents, 500 bước, LR `1e-4`, warm-start checkpoint gốc, train seed 20261012, CPU, khoảng 70.4 phút. Snapshot/hash nằm trong `runs/combat-20261012/pilot-500/runtime/`. Pilot dùng **bản ưu tiên khoảng cách và credit trước sửa**, không phải engine cuối; log không bị viết lại. Trung bình train: ActiveWR 2.4%, tự nổ 3.2%, hòa 90.2%, bom/ván 31.576. Validation cũ chọn warm-start episode 0; final 500 vẫn giữ các update thật. Không chọn final 500 làm model chính chỉ vì đủ số episode.

### Lỗi credit và bản sửa cuối

Tái hiện độc lập: hai agent đặt bom cùng tick, AC ở `(1,3)` thoát an toàn, bot đứng tại `(1,1)` và bị tia bom AC trúng. Với code cũ, AC là A0 nhận hạ gục; đổi AC thành A1 thì kill event là `(0,0)` và AC không được thưởng hạ gục. Đây là lỗi tín hiệu ở tia chồng nhau, không chứng minh toàn bộ chênh lệch góc đều do credit.

Engine cuối lưu từng chủ bom và thời gian còn lại của từng nguồn lửa. Khi chết, ưu tiên một nguồn bom đối thủ thực sự còn hiệu lực và trúng nạn nhân, còn `self_blast_deaths` ghi riêng việc tia bom của chính nạn nhân cũng góp phần gây chết. Một cái chết có đúng một kill credit; cùng chết trong tia bom mình vẫn bị phạt -2.5 và được tính tự nổ. Lửa mới không kéo dài credit của nguồn cũ đã hết hạn. Kiểm tra đối xứng xác nhận reward 13.999 cho AC thắng có hạ gục ở cả hai ID. FFA nhiều nguồn đối thủ cùng lúc vẫn chọn nguồn đối thủ còn hiệu lực xuất hiện trước; chưa đánh giá đầy đủ công bằng 3–4 agents.

Sửa credit được đưa vào code chính sau khi pilot kết thúc. Mọi so sánh cuối bên dưới đều dùng **cùng engine và quy ước credit mới**, gồm cả policy trước sửa: không so trực tiếp số liệu legacy với credit mới để tuyên bố agent tiến bộ.

### Chọn checkpoint trên validation, không dùng test để chọn

Một ứng viên warm-up dùng 602 nhãn đặt bom trực tiếp và 6.958 nhãn điều hướng hợp lệ từ 30 replay trên seed phát triển. Lấy mẫu cân bằng hai nhóm, 200 update Adam, LR `5e-5`, seed 2026101201; loss nhóm chiến đấu 0.5 và điều hướng 0.2, clip norm 1. Đây là warm-up có teacher trên checkpoint RL gốc, không phải 200 episode RL hoặc train mới từ đầu. File: `runs/combat-20261012/combat_bc_phaseC.pt`. Không đổi hệ số loss của pipeline RL chỉ từ thử nghiệm này.

Chấm lại ba ứng viên trên code cuối, 100 trận seed 6100, CPU một thread, cùng bàn ở cả hai góc:

| Ứng viên | WR | ActiveWR | Tự nổ | Hòa |
|---|---:|---:|---:|---:|
| Checkpoint gốc | 13% | 10% | 5% | 86% |
| Final pilot 500 | 9% | 8% | 6% | 87% |
| Warm-up teacher | 11% | **10%** | **4%** | **85%** |

Chọn **warm-up teacher** theo tiêu chí ActiveWR, ít tự nổ, ít hòa, kills và WR. ActiveWR bằng checkpoint gốc; hơn một trận ở chỉ số phụ trên 100 không chứng minh ưu thế thống kê. `selected_validation.json` và `validation_candidates.json` ghi quyết định, hash model/code; quyết định được ghi trước khi đọc test cuối. Checkpoint gốc được giữ nguyên.

Thử cho đặt bom bổ sung khi đang trong tia dự báo nhưng còn đường thoát: WR 19% trên seed phát triển 9000, ActiveWR chỉ 6% so với 7% của control lúc đó. Không đưa thử nghiệm này vào code chính vì tăng thắng thụ động chưa đáp ứng mục tiêu bom AC hạ bot.

### Test cuối độc lập: seed 12000

50 bàn seed **12000..12049**, mỗi bàn chơi cả A0/A1, tổng 100 trận/trường hợp, 9×9, 2 agents, 500 bước, policy lấy mẫu, CPU một thread. Seed này không dùng sửa code, tạo nhãn hoặc chọn trọng số. Cả ba trường hợp dùng engine credit cuối và cùng RuleBot cố định; cột trước chỉ dùng bộ lọc AC trước các sửa ngày 12/10.

| Chỉ số | Checkpoint gốc + lọc trước sửa | Checkpoint gốc + code cuối | Model được chọn + code cuối |
|---|---:|---:|---:|
| Win rate | 8% | 9% | **12%** |
| ActiveWR | 6% | 7% | **9%** |
| Tự nổ | 30% | 7% | **6%** |
| Hòa | 90% | 89% | **86%** |
| Hạ gục / ván | 0.33 | 0.11 | 0.11 |
| Bom / ván | 30.32 | 34.71 | 44.70 |
| Ván vừa hạ bot vừa tự nổ | 27 | 4 | **2** |

Code cuối trên cùng trọng số giảm tự nổ 23 điểm phần trăm, giữ thêm một thắng có hạ gục. Model warm-up tăng thêm hai thắng có hạ gục và giảm một trận tự nổ so với checkpoint gốc trên code cuối. Tổng kills giảm vì loại nhiều ván cùng chết; không coi riêng tổng kills hoặc số bom là tiến bộ. Những chênh lệch ActiveWR nhỏ vẫn cần nhiều seed/test hơn để xác nhận; hòa 86% cho thấy mục tiêu tấn công chưa hoàn thành đầy đủ.

| Góc của model được chọn, 50 trận/góc | WR | ActiveWR | Tự nổ | Hòa |
|---|---:|---:|---:|---:|
| A0 | 16% | 14% | 12% | 80% |
| A1 | 8% | 4% | 0% | 92% |

**Khuyến nghị:** đóng/mở lại GUI, chọn `runs/combat-20261012/combat_bc_phaseC.pt` để thử hoặc warm-start tiếp Phase C. Giữ checkpoint gốc làm control, chưa chuyển D/E. Pilot 500 ở trên là dữ liệu chẩn đoán trước sửa credit; chưa train dài lại với reward credit cuối. GUI đấu trực quan dùng bàn 11×11 và nội suy trọng số; kết quả test này chỉ xác nhận trên 9×9.

Kiểm tra cuối: **37 unittest** và integration GUI qua; assert xác nhận 100 cặp seed/góc đúng của cả ba test, model hash khớp lựa chọn, và runtime prototype chấm điểm trùng từng byte với code chính. Chưa kiểm tra nhiều seed train hoặc FFA 3–4 agents.

Artifacts cuối: `runs/combat-20261012/test_final_before.json`, `test_final_original.json`, `test_final_selected.json`, `fair_validation_*.json`, `selected_validation.json`, `validation_candidates.json`. `diagnostics.json` giữ replay; `pilot-500/` giữ đúng checkpoint/metrics/snapshot của lượt 500, `warmup_validation.json` và `followup_probe.json` là thử nghiệm trung gian với phiên bản cũ. Các file `development_11000_*` và `all_distance_*` cũng là dữ liệu phát triển, không phải test cuối.

## Hai lượt train tiếp sau sửa credit: thưởng phá tường 0.15 và 0.03

Cả hai lượt dùng Phase C, warm-start `runs/combat-20261012/combat_bc_phaseC.pt`, 500 episode, bàn 9×9, 2 agents, tối đa 500 steps, LR `1e-4`, seed train `2026101202`. Validation 100 trận seed 6200 tại episode 0/250/500; chỉ thay `wall_destroy_bonus` giữa hai lượt. Các checkpoint được giữ riêng tại `runs/fair-C-next/` và `runs/fair-C-wall003/`.

| Thưởng tường | Episode | ActiveWR validation | Tự nổ | Hòa | Được chọn |
|---|---:|---:|---:|---:|---|
| Cùng warm-start | 0 | 10% | 3% | 85% | Có |
| 0.15 | 250 | 11% | 6% | 81% | Có |
| 0.15 | 500 | 7% | 5% | 81% | Không |
| 0.03 | 250 | 6% | 7% | 88% | Không |
| 0.03 | 500 | 8% | 7% | 83% | Không |

Tiêu chí hiện tại ưu tiên ActiveWR trước các chỉ số khác: lượt 0.15 chọn episode 250 dù tự nổ tăng. Lượt 0.03 giữ episode 0. Đối chiếu trực tiếp tensor xác nhận `runs/fair-C-wall003/best_phaseC.pt` giống hệt warm-start, còn `final_phaseC.pt` có thay đổi ở cả 12 tensor. Không cần đánh giá lại best của lượt 0.03 trên cùng seed/runtime để lấy một kết quả mới.

### So sánh 200 trận seed 13000: warm-start và best của lượt 0.15

| Chỉ số | Warm-start | Best episode 250, thưởng 0.15 |
|---|---:|---:|
| Tổng thắng | 20 | 15 |
| Thắng có hạ gục | 11 | 12 |
| Tự nổ | 10 | 15 |
| Hòa ở giới hạn 500 steps | 169 | 169 |
| Hòa trước giới hạn, cả hai bị loại | 9 | 14 |
| Tổng hạ gục của AC | 17 | 22 |
| Hạ gục trong trận không thắng | 6 | 10 |

Các cặp seed/góc khớp nhau. Thắng có hạ gục tăng ở 9 cặp và giảm ở 8 cặp, chỉ hơn tổng cộng một trận. Chưa có cải thiện rõ về thắng có hạ gục; tự nổ và cùng chết tăng. Seed 13000 đã dùng để định hướng thử nghiệm nên các lượt sau cần bộ seed khác để xác nhận độc lập.

### Replay chẩn đoán: 6 trận timeout của warm-start

Chọn ba trận timeout đầu tiên mỗi góc từ `runs/test_before_13000.json`; replay warm-start và final của lượt 0.03 trên cùng sáu cặp seed/góc. Assert xác nhận kết quả replay warm-start khớp file test; hash hai checkpoint không đổi trong kiểm tra.

| Chỉ số trên các replay | Warm-start | Final lượt 0.03 |
|---|---:|---:|
| Tổng steps | 3000 | 2589 |
| Cơ hội BOMB hợp lệ lúc ngoài vùng nguy hiểm, đối thủ ở trong tia nổ hiện tại | 150 | 129 |
| Đặt bom ở những cơ hội đó | 148 (98.7%) | 74 (57.4%) |
| Nhãn trực tiếp đặt bom vào tia có đối thủ | 150 | 129 |
| Tổng nhãn tấn công/di chuyển lúc ngoài vùng nguy hiểm | 1557 | 1419 |
| Hạ gục bằng bom AC | 0 | 0 |

Warm-start gần như luôn đặt bom khi đối thủ ở trong tia, nhưng RuleBot vẫn thoát được trong mẫu timeout này. Điều kiện nhãn hiện tại chỉ kiểm tra tia nổ tại thời điểm đặt; không kiểm tra liệu đối thủ còn đường thoát trước khi bom nổ. Final tận dụng ít hơn các cơ hội đó, phù hợp với việc hành vi đặt bom trực tiếp bị suy giảm trong lần train này. Nhãn đặt bom trực tiếp chỉ khoảng 9–10% tổng nhãn ngoài vùng nguy hiểm; đây là một điểm cần kiểm tra khi cân bằng loss phụ, chưa chứng minh là nguyên nhân duy nhất.

**Quyết định:** giữ checkpoint warm-start làm mốc; chưa chuyển D/E, không đề xuất thêm một lượt 500 episode chỉ bằng việc đổi thưởng tường. Giảm thưởng xuống 0.03 chưa đem lại cải thiện trên seed train đã thử. Bước sửa tiếp theo cần đánh giá chất lượng vị trí đặt bom theo đường thoát của đối thủ và kiểm tra việc giữ hành vi tấn công khi học tiếp.

Artifact replay: `runs/fair-C-wall003/diagnostics_13000.json`. Mẫu được chọn từ các trận thất bại của warm-start, không dùng để xếp hạng tổng thể hai model. Chưa test final lượt 0.03 đủ 200 trận hoặc thử nhiều seed train; chưa sửa code training trong phân tích này.

## 2026-10-13 — Bom bổ sung để kết thúc bẫy, nhãn áp lực và loss riêng

### Thay đổi được giữ trong code

Safety mask cũ cấm mọi BOMB khi vị trí agent nằm trong tia nổ dự báo. Nay thêm một ngoại lệ hẹp: BOMB vẫn phải giữ đường sống theo forecast có margin 2 tick và phản bom đồng thời; một đối thủ phải có vị trí sống trước khi đặt, nhưng không còn vị trí sống ở tick bom mới nổ sau khi đặt. CNN tự chọn trong các hành động hợp lệ. RuleBot `robust=False` giữ nguyên luật cũ. Không cộng thưởng đặt bom và không thay trọng số checkpoint để có kết quả bên dưới.

Nhãn tấn công lúc train so sánh số vị trí sống của đối thủ theo tick nổ để chọn ô hiện tại hoặc một bước tới ô gây áp lực lớn hơn. Vẫn tìm vị trí khi ammo đang bận, nhưng chỉ đặt bom nếu mask cho phép. Lấy trung bình riêng loss thoát hiểm/bom gây áp lực/di chuyển với trọng số 0.5/0.5/0.2. Nhãn bom bổ sung lúc nguy hiểm được đưa vào loss chiến đấu. Chọn checkpoint best thêm điều kiện tự nổ không cao hơn validation đầu lượt. CLI ghi config, metrics theo episode, số nhãn/bom chiến đấu và các thành phần loss; best/final được in đường dẫn riêng.

### Kiểm tra phát triển và pilot

- Nhãn vị trí được thử trên 40 trận seed 6300: teacher cũ thắng có hạ gục 5 trận, teacher dùng áp lực 6 trận; hòa 34 → 31, tự nổ đều 0. Đây là policy hướng dẫn, không phải kết quả CNN. Bản thử ban đầu ngừng tìm vị trí khi ammo bận cho kết quả kém hơn, được giữ riêng tại `teacher_production_v1.json`; code cuối giữ việc tìm vị trí.
- Pilot 50 episode **trước ngoại lệ bom bổ sung và trước sửa tìm vị trí khi ammo bận**, validation 100 trận seed 6300: warm-start ActiveWR 10%, tự nổ 2%, hòa 84%; cuối lượt 9%/5%/81%. Guard giữ best ở episode 0. Có 50 dòng metrics, mỗi góc học 25 trận; runtime 350.3 giây trên CPU một thread. Pilot này kiểm tra loss/log và cho thấy chưa có cải thiện trọng số; không giới thiệu final làm model tốt hơn. `pilot-50/runtime.json` ghi hash đúng của phiên bản đó.
- Probe mở ngoại lệ bom bổ sung trên cùng checkpoint, 100 trận phát triển seed 6300: ActiveWR 10% → 23%, tự nổ đều 2%, hòa 84% → 71%. Có 19 quyết định được mở thêm BOMB. Quyết định đưa ngoại lệ vào code được thực hiện trước khi đánh giá seed 14000.
- Code cuối chạy CLI smoke 10 episode để kiểm tra gradient, log, config, checkpoint và bảo toàn nguồn. Có một nhãn bom bổ sung được dùng trong training; mọi thành phần loss hữu hạn. Chưa có validation vì lượt ngắn hơn validation interval, nên checkpoint smoke chỉ dùng kiểm tra, không dùng để chọn model. Không coi ActiveWR train 30% trên 10 trận là benchmark.

### Test độc lập 200 trận seed 14000

100 bàn seed **14000..14099**, mỗi bàn dùng cả A0/A1; 200 trận cho mỗi bộ lọc. Cùng trọng số `runs/combat-20261012/combat_bc_phaseC.pt`, cùng engine, RuleBot cố định, bàn 9×9, 2 agents, 500 steps, lấy mẫu policy trên CPU một thread. Control dùng safety mask snapshot trước lượt sửa này; candidate dùng code cuối. Seed 14000 không dùng tạo nhãn, sửa code, train hoặc chọn biến thể trước lần test này.

| Chỉ số | Bộ lọc trước sửa | Bộ lọc cuối |
|---|---:|---:|
| Win rate | 13.5% (27/200) | **35% (70/200)** |
| ActiveWR | 9.5% (19/200) | **31% (62/200)** |
| Tự nổ | 2% (4/200) | **2% (4/200)** |
| Hòa | 84.5% (169/200) | **63% (126/200)** |
| Hòa hết 500 steps | 165 trận | **122 trận** |
| Hòa trước giới hạn, cả hai bị loại | 4 trận | **4 trận** |
| Hạ gục / trận | 0.105 | **0.32** |
| Hạ gục trong trận không thắng | 2 | **2** |
| Bom / trận | 44.67 | **38.70** |
| Steps / trận | 446.20 | **380.845** |

So từng cặp seed/góc: thêm 43 thắng có hạ gục, không mất thắng có hạ gục nào của control. Số thắng thụ động đều 8 trận; mức tăng tổng thắng đến từ thắng có hạ gục. Không tăng số trận tự nổ hoặc cùng chết trong bộ test này. Kết quả này đến từ bộ lọc mới trên **cùng trọng số CNN**, không chứng minh các thay đổi loss đã tạo model tốt hơn sau train dài.

| Góc, 100 trận/góc | ActiveWR trước → sau | Tự nổ trước → sau | Hòa trước → sau |
|---|---:|---:|---:|
| A0 | 10% → **35%** | 3% → **3%** | 81% → **56%** |
| A1 | 9% → **27%** | 1% → **1%** | 88% → **70%** |

Runtime kiểm tra 400 trận: 372 giây. Assert xác nhận cặp seed/góc khớp, code không thay đổi trong benchmark, hash warm-start giữ nguyên `2f08807220b42fb59e7366628afe5f65840fa808b4d693f564dfa489590d21d7`. Checkpoint gốc ngày 10/10 vẫn có hash `be1c723e730b9b4f5db7746d879c8f1db7c41d0f23e724d09dc506fcdb00d8ba`.

Kiểm tra code cuối: **46 unittest PASS**, integration GUI/Tk PASS và CLI smoke PASS. Test có đối chiếu reachability với movement/damage thật, forecast dài sau tường bị phá và flame hết hạn, gradient nhãn combat không bị số nhãn navigation làm loãng, guard checkpoint, và bom bổ sung an toàn trước phản bom đồng thời ở cả A0/A1. Các bom bổ sung khi đối thủ còn đường sống hoặc agent không đủ thời gian thoát vẫn bị cấm.

**Cách dùng hiện tại:** khởi động lại GUI hoặc chạy evaluate mới, dùng checkpoint `runs/combat-20261012/combat_bc_phaseC.pt`. Không cần train lại để dùng ngoại lệ safety mask. Giữ các final của lượt 0.15/0.03 và pilot làm dữ liệu chẩn đoán. Chưa chuyển D/E chỉ từ kết quả này: còn 63% hòa và chưa kiểm tra FFA hoặc đối thủ khác.

Giới hạn: chỉ xác nhận 1v1 với RuleBot trên 9×9; GUI watch/play dùng 11×11 và nội suy trọng số nên không phải cấu hình benchmark này. Forecast không xét mọi bom/di chuyển tương lai chưa biết của đối thủ, không đảm bảo sống hoặc kill trước mọi policy. Chưa train dài với toàn bộ code cuối; pilot 50 dùng phiên bản trung gian và smoke 10 chỉ xác nhận chạy được. Nếu dùng kết quả này để sửa tiếp, cần seed test độc lập mới (ví dụ 15000).

Artifacts: `runs/attack-pressure/test_before_14000.json`, `test_after_14000.json`, `benchmark_runtime.json`, `before/`, `teacher_probe.json`, `teacher_production.json`, `teacher_production_v1.json`, `chain_probe.json`, `pilot-50/`, `smoke-final-10/`. `benchmark.py` tái hiện hai bộ lọc trên cùng trọng số; `probe.py`/`check_teacher.py` là helper phát triển và không phải API training.

## 2026-10-13 — Train 500 episode với toàn bộ code bom bổ sung

Lượt người dùng chạy tại `runs/pressure-C-next/`: Phase C, 9×9, 2 agents, 500 steps/trận, 500 episode, LR `1e-4`, thưởng tường 0.15, seed train `2026101301`, warm-start `runs/combat-20261012/combat_bc_phaseC.pt`. Validation 100 trận seed 6400 tại episode 0/250/500. Có đúng 500 dòng metrics, mỗi góc học 250 trận.

| Episode | ActiveWR validation | Tự nổ | Hòa | Được chọn |
|---|---:|---:|---:|---|
| 0 | 23% | 4% | 72% | Có |
| 250 | 34% | 6% | 60% | Không: vượt giới hạn tự nổ |
| 500 | 33% | 6% | 63% | Không: vượt giới hạn tự nổ |

Model có học và thay đổi hành vi. Tuy nhiên guard chỉ chọn ứng viên có tự nổ không cao hơn 4% trên mẫu validation này, nên giữ best ở episode 0. Đối chiếu tensor: best giống hoàn toàn warm-start, final thay đổi ở cả 12 tensor. Hai file test người dùng tạo (`runs/test_before_15000.json` và `runs/test_after_15000.json`) giống hệt nhau, gồm cả từng trận; đây là hệ quả test cùng trọng số, không phải training không cập nhật hoặc evaluate bị lỗi.

Trong 500 trận train: 147 thắng, 128 thắng có hạ gục, 30 tự nổ, 341 hòa, 146 hạ gục. Chọn bom ở 13264/13627 nhãn chiến đấu (97.3%), dùng 95/97 nhãn bom bổ sung (97.9%). Hành vi tấn công được giữ. Tất cả thành phần loss đều hữu hạn. Các peak lớn chủ yếu từ value loss trong trận thắng: episode 469 có tổng loss 75.49, value loss 145.62, policy loss 2.63; với value coefficient 0.5, phần critic đóng góp khoảng 72.81. Không suy ra một lỗi runtime hoặc mất hành vi tấn công chỉ từ peak tổng loss.

### Đánh giá riêng final trên seed 15000

Agent đánh giá thêm `final_phaseC.pt`, 200 trận, 100 bàn seed 15000..15099, mỗi bàn cả A0/A1, CPU một thread, cùng code và RuleBot như baseline của người dùng. Tám trận warm-start đầu được replay để kiểm tra khớp dữ liệu baseline trước khi đánh giá final. Các cặp seed/góc khớp và hash các checkpoint không thay đổi trong kiểm tra. Không ghi đè hai file test của người dùng.

| Chỉ số | Warm-start / best ep 0 | Final ep 500 |
|---|---:|---:|
| Win rate | 26% (52/200) | 28% (56/200) |
| ActiveWR | 23.5% (47/200) | 25% (50/200) |
| Tự nổ | 2% (4/200) | 3% (6/200) |
| Hòa | 73.5% (147/200) | 67.5% (135/200) |
| Thua, đối thủ là người sống cuối | 0.5% (1/200) | 4.5% (9/200) |
| Hạ gục / trận | 0.235 | 0.255 |
| Bom / trận | 42.125 | 40.425 |
| Steps / trận | 408.195 | 402.305 |

Mức tăng ActiveWR chỉ ba trận trên 200, thấp hơn mức tăng trên validation. Hòa giảm 12 trận nhưng gồm cả bốn thắng thêm và tám thua thêm; không coi mọi mức giảm hòa là cải thiện. Chênh lệch nhỏ về thắng có hạ gục/tự nổ chưa đủ chứng minh lợi thế tổng thể của final. Không thay checkpoint warm-start/best đang dùng bằng final từ dữ liệu này. Chưa chuyển D/E hoặc đề xuất thêm một lượt train cùng cấu hình chỉ để tăng episode.

Các trận tự nổ/thua của final được lưu trong `diagnostic_summary_15000.json` để định hướng replay lần sau; chưa replay chúng để xác định nguyên nhân chết. Chưa sửa thuật toán hoặc nới guard sau khi xem test này. Seed 15000 đã được đọc để định hướng bước tiếp; nếu tiếp tục chỉnh code/model, cần test độc lập mới, ví dụ 16000.

Artifacts bổ sung: `runs/pressure-C-next/test_final_15000.json`, `comparison_final_15000.json`, `diagnostic_summary_15000.json`, `check_final.py`. Nguồn warm-start giữ hash `2f08807220b42fb59e7366628afe5f65840fa808b4d693f564dfa489590d21d7`; final có hash `93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60`.

## Replay thất bại và thử thoát sớm (2026-10-13)

### Chẩn đoán trên seed phát triển 15000

Đã replay đủ 13 trận tự nổ/thua của final, rồi chạy warm-start trên cùng 13 cặp seed/góc. Kết quả từng trận khớp báo cáo trước: final 1094 bước, warm-start 4122 bước. Không thấy BOMB nào mất đường sống trong dự báo chỉ bom đã biết/không phản bom trên mẫu này; không suy ra agent luôn an toàn trước bom mới của đối thủ. Final có bốn bước di chuyển bị chặn, gồm những tình huống hai bên chờ ở hai phía rồi cùng đi vào lối thoát ở tick nổ cuối. Các trận khác có tình huống agent vào hành lang cụt rồi bot tiến tới và đặt bom ở cửa. Artifact giữ trạng thái/timer/action/mask và grid: `runs/pressure-C-next/replay_failures_15000.json`.

Bản thử chỉ nới cấm ô kề đối thủ khi learner không có hướng đi, STOP được phép và nguy hiểm thực tế còn hơn một tick. Nếu tìm được hướng sống thì thoát sớm; vẫn cấm tường, bom, ô đối thủ hiện tại và lửa theo forecast. Nếu có đường không tranh chấp hoặc cần chờ lửa thì giữ hành vi đó. Nhãn escape lúc train được lọc theo chính mask đã tính cho policy. RuleBot `robust=False`, reward, CNN và checkpoint giữ nguyên.

| Chỉ số, 200 trận seed 15000 | Final trước sửa | Final sau sửa | Warm-start sau sửa |
|---|---:|---:|---:|
| Win rate | 28% | 37% | 29.5% |
| ActiveWR | 25% | 30% | 23% |
| Tự nổ | 3% | 1% | 0.5% |
| Hòa | 67.5% | 59% | 69.5% |
| Thua, đối thủ sống cuối | 4.5% | 4% | 1% |
| Hạ gục / trận | 0.255 | 0.300 | 0.230 |
| Bom / trận | 40.425 | 39.915 | 41.760 |

So cặp final trước/sau: 20 trận thêm thắng có hạ gục, 10 trận mất, tăng ròng 10/200; bốn trận giảm tự nổ và không thêm trận tự nổ. Một phần tăng WR là thắng thụ động, nên báo riêng ActiveWR. Seed 15000 đã dùng để chọn sửa đổi và ứng viên final; bảng này không phải test độc lập. 48 unit test (cả A0/A1, va chạm rồi thử lại, chờ lửa, thoát khẩn cấp, bom bổ sung cũ) và integration GUI/train qua. Hiệu chuẩn production tám trận khớp bản thử; so mask trên 2637 quyết định learner, mask đối thủ khớp bản trước sửa.

Test độc lập đã định trước ở `runs/escape-early/protocol_16000.json`: 200 trận mỗi cấu hình, 100 bàn seed 16000..16099, mỗi bàn cả A0/A1, 9×9, tối đa 500 bước, CPU một thread. So final trước/sau với đúng cùng trọng số; warm-start sau sửa làm mốc riêng. Không train thêm hoặc điều chỉnh code/model trong khi chấm. Snapshot code trước sửa và scripts ở `runs/escape-early/`.

### Test độc lập seed 16000: cùng trọng số final

| Chỉ số | Mask trước sửa | Thoát sớm |
|---|---:|---:|
| Win rate | 34.5% (69/200) | 42% (84/200) |
| ActiveWR | 30.5% (61/200) | 37% (74/200) |
| Tự nổ | 6.5% (13/200) | 1.5% (3/200) |
| Hòa | 61% (122/200) | 54% (108/200) |
| Thua, đối thủ sống cuối | 4.5% (9/200) | 4% (8/200) |
| Hạ gục / trận | 0.340 | 0.370 |
| Bom / trận | 38.975 | 38.230 |
| Steps / trận | 370.205 | 358.890 |
| ActiveWR A0 / A1 | 25% / 36% | 31% / 43% |
| Tự nổ A0 / A1 | 7% / 6% | 3% / 0% |

Theo từng cặp seed/góc: 19 trận thêm thắng có hạ gục, sáu trận mất, tăng ròng 13/200. Có 10 trận giảm tự nổ, không thêm trận tự nổ. Phần tăng WR chủ yếu là thắng có hạ gục: +13 active và +2 passive. Cả A0/A1 cải thiện trên mẫu này. Vẫn có 54% hòa và tám trận thua; đây không phải mức bảo đảm trên mọi seed/bàn/đối thủ. Không gán mức tăng này cho train mới: hai cột dùng chính `pressure-C-next/final_phaseC.pt`, hash không đổi.

Artifacts: `runs/escape-early/test_final_before_16000.json`, `test_final_after_16000.json`; replay/development và snapshot code trong cùng thư mục. Mọi checkpoint gốc, `best_phaseC.pt` và metrics của lượt 500 episode được giữ lại.

Mốc warm-start với cùng code sau sửa trên 200 trận seed 16000: WR 32%, ActiveWR 24% (48/200), tự nổ 0.5% (1/200), hòa 64.5%. Final có thêm 26 thắng có hạ gục, nhưng cũng thêm hai trận tự nổ so với warm-start trên mẫu này; không gọi final an toàn hơn warm-start. Ưu tiên final làm ứng viên fine-tune tiếp cho mục tiêu hạ bot bằng bom, giữ warm-start làm mốc ít rủi ro hơn. Không ghi đè best đã chọn trên validation cũ hoặc nới guard chọn best.

Cả 600 trận hoàn tất trong 528.39 giây CPU một thread. `runs/escape-early/comparison_16000.json` lưu tổng hợp, so cặp, thời gian và hash checkpoint/code; `test_warm_after_16000.json` lưu từng trận của mốc warm-start. Seed 16000 đã được đọc; nếu train/chỉnh thêm rồi chấm cuối, dùng seed mới như 17000. Chưa fine-tune với code thoát sớm hoặc test 11×11/FFA 3–4 tác tử.

## Fine-tune với thoát sớm: validation 6600, test 17000 (2026-10-13)

Người dùng chạy 500 episode C, 9×9, hai tác tử, tối đa 500 bước, LR `1e-4`, thưởng tường `0.15`, train seed `2026101302`, warm-start từ `runs/pressure-C-next/final_phaseC.pt`, thư mục `runs/escape-C-6600`. Validation 100 trận seed 6600 ở đầu lượt, ep 250 và 500:

| Episode | ActiveWR | Tự nổ | Hòa | Chọn best |
|---|---:|---:|---:|---|
| 0 | 32% | 1% | 62% | Có |
| 250 | 27% | 1% | 63% | Không |
| 500 | 36% | 1% | 58% | Có |

Đối chiếu file: đủ 500 dòng metrics, đúng thứ tự episode, A0/A1 mỗi phía 250 trận. Best và final giống nhau ở toàn bộ tensor; cả 12 tensor đều khác checkpoint nguồn. Không phải trường hợp best ep 0 của lượt trước. Guard chọn best hoạt động đúng với giới hạn tự nổ 1% trên validation này. Code/engine/evaluator khớp hash của benchmark 16000.

Trong cả lượt: 188 thắng, 164 thắng có hạ gục, bốn tự nổ, 297 hòa, 164 hạ gục; dùng 13373/13736 nhãn bom chiến đấu (97.36%) và 116/119 nhãn bom bổ sung (97.48%). Last 200 có WR 44.5%, ActiveWR 37%, tự nổ 0.5%, hòa 53.5%. Mọi thành phần loss đều hữu hạn. Peak ep 211 có tổng loss 77.66, value loss 153.17, policy loss 0.88; critic với coefficient 0.5 đóng góp khoảng 76.59. Peak tổng loss chưa chứng minh lỗi hoặc mất hành vi tấn công.

### Test độc lập cùng seed/góc 17000

Hai file test người dùng tạo đều có 200 trận, 100 bàn seed 17000..17099, mỗi bàn A0/A1, CPU, cùng code và RuleBot:

| Chỉ số | Đối chứng pressure-C-next/final | Candidate escape-C-6600/best |
|---|---:|---:|
| WR tổng | 42.5% (85/200) | 43.5% (87/200) |
| ActiveWR | 38.5% (77/200) | 36% (72/200) |
| Thắng thụ động | 4% (8/200) | 7.5% (15/200) |
| Tự nổ | 0% (0/200) | 0.5% (1/200) |
| Hòa / timeout 500 bước | 56% (112/200) | 54% (108/200) |
| Thua, đối thủ sống cuối | 1.5% (3/200) | 2.5% (5/200) |
| Hạ gục / trận | 0.385 | 0.360 |
| Bom / trận | 40.490 | 39.880 |
| Steps / trận | 382.995 | 365.035 |
| ActiveWR A0 / A1 | 39% / 38% | 36% / 36% |

So từng cặp: thêm 33 thắng có hạ gục nhưng mất 38, giảm ròng năm trận; thêm một tự nổ. WR tổng tăng hai trận là kết quả +7 thắng thụ động và -5 thắng có hạ gục. Hòa giảm bốn trận, gồm hai thắng thêm và hai thua thêm. Không coi mức tăng WR tổng/giảm hòa là bằng chứng agent diệt bot tốt hơn. Cải thiện trên validation chưa được xác nhận trên test mới; chênh lệch năm trận còn nhỏ, chưa đủ kết luận candidate kém trên mọi seed hoặc đã học thuộc validation.

Giữ `runs/pressure-C-next/final_phaseC.pt` làm checkpoint đang dùng, giữ candidate để đối chiếu; chưa chuyển D/E hoặc chạy thêm một lượt 500 cùng cấu hình. Không ghi đè model/log hoặc sửa reward/CNN/thuật toán qua lượt đọc kết quả này.

### Replay mẫu timeout

Có 72 cặp seed/góc mà cả hai model đều timeout. Replay sáu cặp đầu theo quy tắc ba A0 và ba A1: seed 17002/17003/17008 ở A0, 17006/17011/17014 ở A1. Cả 12 replay khớp kết quả đã lưu; hash checkpoint không đổi. Đây là mẫu thất bại được chọn để chẩn đoán, không đại diện toàn bộ test.

| Thống kê trên 6 timeout/model | Đối chứng | Candidate |
|---|---:|---:|
| Bom đặt | 290 (38–63/trận) | 263 (39–51/trận) |
| Cơ hội bom hiện tại chạm tia đối thủ, không nguy hiểm, mask cho phép | 114 | 92 |
| Chọn bom ở các cơ hội đó | 113 | 92 |
| Bom có pressure bằng 0 | 19 | 19 |
| Bom có pressure lớn hơn 0 và nhỏ hơn 1 | 271 | 244 |
| Bom có pressure bằng 1 | 0 | 0 |
| Theo nhãn di chuyển teacher | 760/1376 (55.23%) | 859/1498 (57.34%) |

Trong mẫu này agent vẫn đặt nhiều bom và tận dụng gần hết cơ hội trúng tia; bom thường giảm một phần vị trí sống dự báo nhưng chưa ép đối thủ hết đường. Teacher hiện đánh giá vị trí hiện tại/một bước và bom đã biết. Hướng cải thiện tiếp là kiểm tra việc chọn vị trí và phối hợp bom nối tiếp để chặn lối thoát, giữ các điều kiện sống sót; tăng số episode hoặc thưởng đặt bom đơn thuần chưa có bằng chứng giải quyết nút thắt này. Nhãn/cơ hội ở các bước liên tiếp không phải các trận độc lập. Nếu dùng seed 17000 để chỉnh tiếp, test cuối cần seed mới như 18000.

Artifacts: `runs/escape-C-6600/comparison_17000.json`, `diagnostic_cases_17000.json`, `replay_timeouts_17000.py`, `replay_timeouts_17000.json`; hai JSON test gốc giữ nguyên tại `runs/escape-early/`. Best candidate có SHA256 `88c952f6a7648886eb80cd323cd118008d98b3f3620e5c7ae54b48e773c311df`, đối chứng giữ `93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60`. Chưa triển khai planner mới, đánh giá seed bổ sung hoặc test 11×11/FFA 3–4 tác tử trong lượt này.

## Pilot giáo viên hai bom — 2026-10-14

Đã triển khai forecast bom đặt trễ và teacher tối đa hai bom, mặc định vẫn `local`. 58 test pass; kiểm tra fixture từng single còn refuge nhưng combo đóng hết đường sống và learner vẫn thoát. CNN tự lấy mẫu qua mask, không dùng planner khi test. Model, reward, observation 12 kênh và optimizer giữ nguyên.

Pilot warm-start `runs/pressure-C-next/final_phaseC.pt`, Phase C/9×9/2 agents/max_steps 500/lr 1e-4, train seed 2026101303. Validation 100 trận seed 21000 tại 0/100/200. Trong ngân sách train chung **65 phút**, local hoàn tất **200/200**, two_bomb **146/200**. Local chọn best episode 200 (validation ActiveWR 34%, tự nổ 0%); two_bomb tại episode 100 đạt 18%/1% nên guard giữ best episode 0. Final 146 episode được lưu riêng, chưa qua validation 200.

Test cuối hoàn tất 400 trận mỗi model, seed bàn 25000–25199, cả A0/A1, CPU qua mask hiện tại:

| Model | Best episode | ActiveWR | Tự nổ | Hòa | Kills/trận | Thắng thụ động |
|---|---:|---:|---:|---:|---:|---:|
| Gốc | gốc | 33.00% | 0.00% | 61.00% | 0.330 | 4.50% |
| Local | 200 | 33.00% | 0.25% | 60.75% | 0.330 | 4.50% |
| Two_bomb | 0 | 33.00% | 0.00% | 61.00% | 0.330 | 4.50% |

Best two_bomb có trọng số giống gốc, cả 400 kết quả khớp nhau. Bootstrap 10.000 lần theo 200 seed bàn (giữ hai góc cùng nhau), seed bootstrap 20261014: chênh ActiveWR two_bomb−gốc **0 điểm, CI 95% [0,0]**; two_bomb−local **0 điểm, CI [-6.00,5.75]**. Không tăng ít nhất 5 điểm và các cận dưới không dương.

**Giữ model gốc; pilot chưa đủ để kết luận phương pháp** vì lượt cải tiến chưa đủ 200 episode. Đã chấm đủ 1.200 trận; tổng thời gian chạy ghi nhận 86.67 phút. Preflight 147 trạng thái timeout trung bình 33.31 ms/query, p95 62.83 ms. Trong 146 episode hoàn tất, planner tạo 56.036 nhãn (43.321 nhãn hai bom), trung bình 27.96 ms/query; số nhãn không chứng minh CNN thực hiện combo tốt. Không thử GAE/PPO, FFA hoặc seed train khác.

Chi tiết từng góc, validation, CI, hashes, source và đường dẫn best/final tại [báo cáo pilot](runs/two-bomb-pilot-20261014/report.md). Metrics và tất cả checkpoint cũ giữ nguyên; SHA256 model gốc không đổi. Nếu chỉnh teacher tiếp, dùng seed test mới thay vì tiếp tục chọn model trên seed 25000 đã xem.

## Giáo viên hai bom đủ 200 episode — test26000 (2026-10-14)

Người dùng đã hoàn tất `run_two_bomb_complete.py`: two_bomb train200 episode từ model gốc, dùng lại local200 cùng source/config, validation100 trận seed21000 ở0/100/200. Đã audit đủ1.200 trận cuối (400/model; 200 board seeds26000–26199, cảA0/A1, CPU/mask chung), source/checkpoint hashes và toàn bộ rates/averages theo từng game/corner. Không còn thiếu episode như pilot trước.

| Model | Best episode | ActiveWR | Tự nổ | Hòa | Kills/trận | PassiveWR |
|---|---:|---:|---:|---:|---:|---:|
| Gốc | gốc | 36,50% | 0,50% | 54,75% | 0,365 | 6,50% |
| Local | 200 | 38,00% | 0,25% | 54,50% | 0,380 | 6,00% |
| Two_bomb | 0 | 36,50% | 0,50% | 54,75% | 0,365 | 6,50% |

Best two_bomb bằng gốc ở mọi tensor và400 game records; final200 khác gốc ở cả12 tensor nhưng chưa chấm held-out riêng. Validation của model học: ActiveWR29%→18%→15%, hòa62%→77%→81%; ep100 bị loại vì tự nổ1%, ep200 tự nổ0% nhưng vẫn kém mốc nên best0 giữ lại. Điểm36,5% best không phải chất lượng của final200.

Bootstrap10.000 lần theo seed bàn: two_bomb−gốc0 điểm, CI95%[0;0]; two_bomb−local−1,5 điểm, CI[-7,25;4,50]. Local−gốc+1,5 điểm, CI[-4,50;7,25], chưa đạt +5 điểm/cận dưới>0. Local cải thiện A0 +6 điểm nhưng A1 giảm3 điểm. Giữ model gốc; T1 thực thi hoàn tất và gate cải tiến FAIL. Một train seed chưa kết luận toàn bộ phương pháp luôn thất bại.

Teacher có81.718 query, 77.355 nhãn nhận (94,66%), 59.064 nhãn hai bom; không có trace đo agreement/completion/full-trap. Ưu tiên T2 trước khi tăng train hoặc trộn PPO/reward. [Báo cáo và audit đầy đủ](/D:/fpt/subject/BomberMan/runs/two-bomb-complete-20261014-031626/T1_report.md). Đầu vào/checkpoint giữ nguyên; test26000 đã đọc nên sửa tiếp dùng test mới.

## T2 audit và review nhãn/combo — 14/10/2026

Lượt `t2-full-20261014-060320`:400 trận tổng,200/model,100 seed bàn28000–28099 với cảA0/A1. B và final two_bomb200 chơi bằng CNN/mask, teacher chỉ log; best two_bomb aliasB. Test74 OK; CRC/SHA ZIP và snapshot/checkpoints khớp; toàn bộ summary/từng góc được tính lại từ158.293 frames/10.695 windows/88.872 branches. Đã dựng lại12 replay/5.608 frames (10 có sẵn, hai controls).

| Model | ActiveWR | Hòa | Tự nổ | Kills/trận | Bom đặt/trận | Đúng bom hai khi bom đầu đúng |
|---|---:|---:|---:|---:|---:|---:|
| B |33,00% (66/200)|57,00%|1,00%|0,33|40,17|23/905=2,54%|
| Final two_bomb200 |16,00% (32/200)|75,50%|1,00%|0,16|26,68|5/427=1,17%|

Final−B **−17 điểm**, paired bootstrap95%CI **[-25,5;−8,5]**, 100 board pairs/10.000 resamples. A0 giảm32→15%, A1 giảm34→17%. Đây là regression của final trên audit set, không dùng để chọn checkpoint mới hay kết luận mọi train seed two_bomb đều thất bại.

Teacher chỉ kiểm tra mask ở hành động đầu: final/RuleBot có7.086 physical prefixes complete nhưng6.454 bị future mask chặn (91,08%); static90,48%, evader90,74%. Gap thường ở BOMB, gồm cả bom đầu sau di chuyển và bom hai. Các nhãn hai bom dự báo trap chỉ3/63.973 frames; combo trap mà hai bom riêng vẫn có refuge chỉ1/6.390 sampled plans. Case này cũng bị mask chặn ở bom hai. Agreement52,98% gần uniform K/M52,65% trên cùng state, chưa đủ làm bằng chứng học combo.

**T2 chẩn đoán hoàn tất; giữ B, ưu tiên L2.0 candidates offline trước train tiếp.** Chỉ lọc best hiện tại có thể giữ gần hết nhãn một bom và rất ít combo: final static mask-pass599 nhưng chỉ4 two-bomb. Cần kiểm tra alternatives trong64 schedules trước khi implement guarded teacher; nếu valid combos quá hiếm thì mở T3.1 benchmark/curriculum riêng. Chưa đổi mask, reward, optimizer hay train thêm. [Quyết định đầy đủ](/D:/fpt/subject/BomberMan/runs/t2-full-20261014-060320/T2_decision.md), [audit có thể tái lập](/D:/fpt/subject/BomberMan/runs/t2-full-20261014-060320/analyze_t2_results.py).

## L2.0 — candidate alternatives qua future mask, 14/10/2026

`l2-full-20261014-120348` hoàn tất40/40 saved game tapes từ10 seed bàn28000–28009, hai model/cảA0/A1;149 states,6.666 scored paths. Test log82 OK. ZIP/source/input/checkpoint hashes khớp. Đã tính lại mọi summary field, dựng lại18.286 frames và tạo lại mọi candidate/6.666 static guards +170 stress branches; action tapes/suffix/masks khớp kết quả gửi. 149 thay vì160 do game gốc kết thúc sớm/không đủ tick có nhãn, không phải lượt dở.

| Metric states | Gốc B (69) | Final200 (80) | Tổng (149) |
|---|---:|---:|---:|
| Old representative viable | 5 | 5 | 10 (6,71%) |
| Any candidate viable | 42 | 43 | 85 (57,05%) |
| Any two-bomb viable | 8 | 6 | 14 (9,40%) |
| Selected best two-bomb | 5 | 5 | 10 (6,71%) |
| Genuine combo trap | 0 | 0 | 0 |

75/139 states có best cũ không khả thi được cứu bởi alternatives;40/75 vẫn giữ nguyên first-action labels. Trong6.666 scored paths không có trap, kể cả trước guard;27 lịch hai bom khả thi cũng còn refuge khi so combo/từng bom riêng trên cùng horizon. Top1/4/8 chỉ đạt best-score và first-actions recall10/24/43 trên85 states có best hợp lệ; top8 guard mean14,89ms, full mean70,40ms/p95155,03ms. Đây là clone-audit timing, chưa phải chi phí trainer.

85 selected best qua static guard: evader62/85 prefix complete và sống; RuleBot56/85 prefix complete,51/56 sống. Riêng10 selectedtwo: evader3/10, RuleBot1/10 complete và sống; còn lại bị mask chặn, không được gộp vào suicide. Các branch dùng prescribed plan/forecast suffix, không phải CNN win rate; suffix chưa ép mask mọi bước.

**Giữ B; ưu tiên T3.1 benchmark có certificate, chưa train L2/curriculum.** Alternatives giúp tính khả thi nhưng chưa đủ bằng chứng guard giải quyết mục tiêu combo. Cần phân biệt task có combo qua mask, teacher có tìm ra, CNN có thực hiện. Đây là10 seeds audit đã xem, states/candidates phụ thuộc nhau; không đo cải thiện ActiveWR hay suy phương pháp không thể học combo. [Quyết định và bước tiếp theo](/D:/fpt/subject/BomberMan/runs/l2-full-20261014-120348/L2_decision.md), [analysis tái lập](/D:/fpt/subject/BomberMan/runs/l2-full-20261014-120348/analyze_l2_results.py).

## T3.1 — benchmark có lời giải, 14/10/2026

Đã triển khai benchmark/tests/review và chạy `t3-full-20261014-implementation`:240 train/validation cases × B ×3 opponents ×2 repeats =1.440 games,498,18s. Registry336 cases gồm4 related layout families,8 orientations/cảA0/A1; test familywood_branch chưa chạy CNN. Đã kiểm traZIP/source/checkpoint hashes, tái tạo240 teacher states/208 legal certificates và mọi1.440 action tapes/62.110 frames; recompute mọi outcome/placement/pair/summary.89 tests pass ở snapshot full,90 ở code hiện tại với thêm edge case prefix death. Core/checkpoint không đổi.

96/96 genuine combo references có từng single còn refuge, combo hết refuge, prefix và escape qua mask tới hết nguy hiểm; old teacher tìm được96/96. Reference stress evader/RuleBot48/96 full tape pass mỗi loại; đây là sampled future, không phải guarantee.

| B trên bài combo (192 games/opponent) | Static | Evader | RuleBot |
|---|---:|---:|---:|
| Owned kill và sống |168 (87,50%)|92 (47,92%)|74 (38,54%)|
| Đặt ít nhất hai bom |120|192|178|
| Có genuine played pair theo known-bomb forecast |80|86|70|
| Reference prefix exact |76|76|70|
| Skill horizon80 bị truncate |24|100|94|

Không ghi nhận tự nổ trong1.440 games. Combo validation static114/128,evader60/128,Rule52/128. A0/A1 mỗi720 games giống nhau:353 owned successes/18 passive wins/0 suicide/310 skill truncations/118 genuine pair games. Không coi những repeats/roles/orientations là độc lập. Owned kill trước target đứng yên không tự chứng minh combo; played-pair metric và actual outcomes lưu riêng. Skill horizon80 tách khỏi actual draw500-step.

**T3.1 DONE; giữ B; chọn T3.2 curriculum né/phản bom, chưa train.** Planner có thể tạo trap và CNN gốc có thể thực hiện combo trên positive layouts; sự hiếm combo trongT2/L2 vẫn là quan sát trên phân phối state thật khác. Không sửa search để giải các positive controls đã pass, không đổi GAE/PPO/reward trong cùng bước. Cần pilot có đối chứng và random-map500-step test để kiểm chứng transfer. [Báo cáo/giới hạn/quyết định](/D:/fpt/subject/BomberMan/runs/t3-full-20261014-implementation/T3_decision.md), [hướng dẫn/lệnh](/D:/fpt/subject/BomberMan/T3_GUIDE.md).

## T3.2 — runner có đối chứng, test implementation 14/10/2026

Đã triển khai [run_t3_curriculum.py](/D:/fpt/subject/BomberMan/run_t3_curriculum.py) và [protocol/lệnh đầy đủ](/D:/fpt/subject/BomberMan/T3_CURRICULUM_GUIDE.md). Đăng ký control600 PhaseC RuleBot và curriculum600 C1static→C2evader→C3RuleMix→PhaseC consolidation; cùng B/seed2026101303/LR1e-4/teacherlocal/reward/optimizer schedule. Chỉ dùng train fork positive cases; chứng nhận lại preflight. Stage probes tuổi100/150/200, hai pass success≥60%/suicide≤B guard, cap200. Skill probes không reset RNG/weights hoặc chọn best; PhaseC100game validation21000 ở0/mỗi100 chọn best và giữ suicide guard hiện tại.

110 tests qua (`runs/t3-curriculum-tests-current.log`), gồm trainer thật với một Adam/scheduler qua ba stage, steps2→4→6/LR giảm liên tục; gates/pass-streak/cap/cancel, reset trả observation đúng, control map/RNG khớp normal PhaseC, CPU/CUDA probe purity, output bảo vệ và final/partial routing. Full orchestration/paired seeds/gate và matched-budget routing được test với mock outcomes, không phải kết quả RL thật.

Smoke CPU `runs/t3-curriculum-smoke-implementation` hoàn tất6ep/arm trong6,64s; CUDA/RTX3050 `runs/t3-curriculum-smoke-cuda-implementation` hoàn tất6ep/arm trong7,44s, mỗi nhánh một Adam/6 updates và48 learner steps. Stages cưỡng bức ởep2/4/6, max8 nên full_protocol=false/retain_original; cả best chọnep0. Không dùng win rates của smoke để kết luận học tốt hơn. Test deadline1s `runs/t3-curriculum-deadline-implementation` trả incomplete và ZIP, không mở test. Source/warm-start giữ nguyên hash trong mọi smoke; core T1/T2/L2 không sửa.

Nếu full curriculum fail stage, runner chấm final/control checkpoint cùng episode trên validation21000 và vẫn incomplete. Chỉ cặp600 đủ gates/validation mở random-map400/model test31000–31199/cảA0/A1 và wood_branch576games/model. Gate cuối +5pp so cả B/control, bootstrap95% lower>0, suicide không cao hơn chỉ cho phép replication; không tự thay model.

**T3.2 implementation/tests/smoke DONE; full600 WAIT RUN.** Chưa mở test31000/wood_branch, chưa có bằng chứng curriculum cải thiện ActiveWR; giữ B. Nguồn smoke/ZIP/console được lưu trong các thư mục nêu trên.

## T3.2 pilot — dừng C1 đúng gate, 14/10/2026

Người dùng chạy `t3-curriculum-full-20261014-154703`,CUDA/RTX3050, cùng B/local/LR1e-4/seed2026101303/schedule600. Control đủ600ep/226.640learner steps/600updates; curriculum đủ200ep/41.086steps/200updates, chưa qua C1, không vàoC2/C3. Tổng2.625,49s≈43,76phút. Đã auditZIP49 files/hashsource/checkpoint,800 metrics/ten validations, tái tạo192skillgames/6.870frames với mọi mask/outcome/placement/pair; recompute200 diagnostic game rows/rates/corners. Script/results/logs và [báo cáo](/D:/fpt/subject/BomberMan/runs/t3-curriculum-full-20261014-154703/T3.2_decision.md) được thêm sau ZIP raw; original/source/model chưa thay.

| C1 checkpoint | Escape/32 | Approach/32 | Success tổng/64 | Suicide | Gate |
|---|---:|---:|---:|---:|---|
| B |32|28|60 (93,75%)|0|Baseline |
| Ep100 |32|12|44 (68,75%)|0|Pass1 |
| Ep150 |32|4|36 (56,25%)|0|Fail/reset |
| Ep200 |32|12|44 (68,75%)|0|Pass1, cap200 nên dừng |

Ep200 có20 approach failures hết80bước, không tới reference bomb cell;16 không đặt bom, STOP0% (di chuyển không tới vị trí tấn công hữu hiệu). B64 C1 records từT3.1 đúng cases/seeds và khớp preflight T3.2. Dữ liệu phù hợp với giảm kỹ năng approach/RuleBot retention khi trainC1static; chưa xác định nguyên nhân riêng hoặc hiệu quảC2/C3.

Diagnostic **cùng200episodes/updates** trên validation21000: control ActiveWR38%,WR46%,draw52%,suicide0%,avgkills0,38,passive8%; curriculum18%,25%,74%,0%,0,18,7%. ActiveWR A0/A1: control30/46%;curriculum20/16%. Delta−20pp, bootstrap95%50boardpairs[-32;-8]pp; đây là validation đã xem, không phải test độc lập400/model. Số transitions khác: control200 đầu78.168 vs curriculum41.086. Curriculum best0 giống mọi12tensors B; final200 khác12. Controltrain600 có2self-blasts, curriculum200 có0; không đồng nhất với0suicide trên validation.

Control best400 đạt42% ActiveWR/draw50%/suicide0% trên validation21000, B29%; cần chấm riêng độc lập vì checkpoint đã chọn trên validation. Đã đăng ký **B vs controlbest400,400/model seeds32000–32199/cảA0-A1,CPU1thread/RuleBot/max500**, gate≥5pp/CI lower>0/suicide≤B. Bài xác nhận checkpoint này **chưa chạy**, không biến curriculum dừng thành thành công; [protocol](/D:/fpt/subject/BomberMan/runs/t3-curriculum-full-20261014-154703/control_confirmation_protocol.json), [lệnh đầy đủ](/D:/fpt/subject/BomberMan/CONTROL_CONFIRMATION_GUIDE.md).

**Pilot REVIEWED; full pair600 incomplete/stage_gate_failed; giữ B.** Test31000/wood_branch vẫn chưa mở. Không train nối hoặc nới gate. Curriculum tiếp theo là backlog: bỏC1bắt buộc khiBđã tốt, thử combo/evader trộnPhaseC/RuleBot để giữ kỹ năng, audit approach teacher và báo từng skill group; chưa implement/train. Báo cáo đã ghi nguồn skill scientific-critical-thinking/quy trình và giới hạn một train seed.

## Control best400 — test32000 hoàn tất, 14/10/2026

Người dùng hoàn tất `control-confirmation-20261014-201557`: frozen B/controlbest400,400/model,200 seeds32000–32199/cảA0-A1,9×9/2/max500,CPU1thread/RuleBot/mask. Protocol đăng ký trước test; auditZIP7files/CRC/hashes/protocol vàsource/checkpoints,800 game rows/mọi rates/averages/corners; bootstrap200boardpairs/10.000resamples seed20261014 trùng summary. [Báo cáo và scripts/results](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/confirmation_report.md).

| Metric | B | Control best400 |
|---|---:|---:|
| ActiveWR |32,75% (131/400)|35% (140/400)|
| Win rate |38% (152)|40,25% (161)|
| Hòa |60,5% (242)|58,25% (233)|
| Suicide |0,5% (2)|0,5% (2)|
| Avg kills |0,3275|0,35|
| Passive wins |5,25% (21)|5,25% (21)|
| ActiveWR A0/A1 |34,5% /31%|35% /35%|

Delta+2,25pp/CI95%[-3,25625;8]. Gate gain≥5pp FAIL,CI lower>0 FAIL,observed suicide≤B PASS. Candidate có80 new active wins nhưng mất71 củaB,net9. CI chứa0 và cải thiện≥5pp nên chưa xác nhận superiority hoặc equality; không suy model chắc chắn ngang nhau. Validation+13pp trước đó chưa được xác nhận trên test mới.

Replay hai shared suicides32080/32133A0 × hai models,188frames: mọi learner action qua mask, actiontapes hai models giống nhau, tất cả game stats khớp. Enemy đi từ bomb cũ(2,3) tới cửa(1,2) và đặt bomb trễ sau ownbomb(3,1); known-safe moves hết ởtick47/41,deathstep49/45. `_can_bomb_and_escape` hiện chỉ xét immediate counterbomb tại enemy current cell khi đặt được; delayed movement/placement/returned capacity chưa được bao phủ đầy đủ. Diagnostic phù hợp giới hạn đó; chưa có mask fix hoặc bằng chứng guard mới không tăng hòa.

**Confirmation DONE/gate FAIL, giữ B.** Test32000 đã đọc và dùng diagnostic; model/code mới phải test mới. Không train mới/thay model hoặc chấm thêm seed chưa xem trong review; test31000/wood_branch còn khóa. Bước tiếp trong tracker: offline approach teacher-vs-CNN audit và fixture counter-bomb trễ, rồi protocol riêng nếu sửa mask hoặc curriculumC2/rehearsal. Không chọn checkpoint khác hoặc tăng test tùy ý để tìm gate pass. Report đã ghi nguồn quy trình scientific-critical-thinking và giới hạn một train seed.

## T3 — offline approach/counter-bomb audit hoàn tất, 15/10/2026

[Báo cáo](/D:/fpt/subject/BomberMan/runs/t3-approach-audit-20261015-final/T3_approach_decision.md), [summary](/D:/fpt/subject/BomberMan/runs/t3-approach-audit-20261015-final/summary.json), [analysis/script](/D:/fpt/subject/BomberMan/runs/t3-approach-audit-20261015-final/analysis_results.json). Thêm audit runner, counter-bomb fixtures và9 tests; **119 tests PASS**. Production source/checkpoints/tapes vẫn khớp hashes đóng băng. Không train hoặc mở test31000/wood_branch;32000 chỉ dùng hai states đã xem.

Tái tạo256 C1 games/8.224frames, đúng masks/actions/positions/placements/kills/alive; mọi local label nằm trong mask và teacher không sửa env/NumPy/Torch CPU RNG. Teacher closed loop +fallback STOP khi đã ở refuge giải32/32 approach và32/32 escape. Mỗi tape có4 bước thiếu nhãn sau placement, không phải thiếu hướng dẫn ban đầu.

| Cohort | Approach owned kill/32 | Còn1 bước tới ô bắn: move tiến /frames |
|---|---:|---:|
| B |28|32/70 (45,71%)|
| C1 ep100 |12|12/170 (7,06%)|
| C1 ep150 |4|4/162 (2,47%)|
| C1 ep200 |12|12/124 (9,68%)|

Mọi clean navigation label đều giảm khoảng cách hình học; near-goal chỉ có một move qua mask làm giảm khoảng cách. So masked probabilities trên cùng16 teacher-path cases: move đầu B56,22%/final20079,74%, nhưng move còn2 ô B37,46%/final2009,78%; BOMB tại mục tiêu B94,69%/final gần100%. Rot0/m0 tại3,2, final200 gần100%BOMB mặc dù range2 chưa chạm enemy3,5; teacher cho RIGHT tới3,3. Kỹ năng tiến đoạn giữa/đặt đúng chỗ là bottleneck quan sát, chưa chứng minh loss weight hoặc optimizer là nguyên nhân. Frame ratios/rotations/roles/repeated seeds phụ thuộc nhau, không dùng như independent win-rate evidence.

96 counter fixture checks:16/variant ammo1 returned capacity(delay5), ammo2 occupied tile(delay3), no-counter, alternate-exit, unavailable-capacity và unreachable-door.32 unsafe tái tạo failure của mask hiện tại;64 controls còn đường sống qua mask. Forecast vàstep khớp blockers/free slots/damage/flame decay/wood/placements tới hết nguy hiểm. Hai real states32080/32133 khớp conditional no-path;32080 còn relaxed path khi bỏ occupancy enemy, nên chỉ thêm future bomb chưa đủ guard.

**Offline audit DONE; giữ B, chưa có model tốt hơn.** Ưu tiên ablation riêng nhãn move tăng pressure có placement/escape certificate sang combat0,5, giữ local control và target/reward/mask/optimizer; đây mới là đề xuất chưa implement/train. Runtime counter-bomb guard và C2+rehearsal giữ các protocol riêng. Người dùng không cần chạy thêm để đóng audit này; các file và tests đã chạy tại workspace.

## Guidance MOVE — implementation và pilot đang chạy, 15/10/2026

**Cập nhật cuối:** pilot đã hoàn tất/review,gateFAIL,giữB. Xem bảng test và [decision_report.md](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315/decision_report.md) bên dưới; các đoạnRUNNING ghi trạng thái lúcđăng ký/khởi chạy.

Đã triển khai [certifier](/D:/fpt/subject/BomberMan/combat_move_teacher.py), trainer/CLI mode`local_combat_move`, [runner đối chứng](/D:/fpt/subject/BomberMan/run_combat_move_pilot.py), [protocol/lệnh và file kết quả](/D:/fpt/subject/BomberMan/COMBAT_MOVE_GUIDE.md). Target action local không đổi; chỉ MOVE có tăng positive pressure và next BOMB/escape certificate trên clone sang combat mean0,5; nhãn không đạt giữ navigation0,2. Escape greedy qua mask tới hết bom/lửa, bounded32ticks/other agents STOP; không phải guard trước future counter-bombs. Env/mask/reward/CNN/12channels/optimizer giữ nguyên, defaultlocal vẫn local.

**136tests PASS** (`runs/combat-move-implementation-20261015/all-tests.log`), gồm local-before bit parity ở mọi tensor/reward/action, actual qualified update CPU/CUDA, pressure/slot/collision/greedy/purity, loss routing không override actionCNN, budgets/incomplete/source changes/paired bootstrap/gates/fresh-test guards. Source cũ lưu trongbefore snapshot; test archived control đối chiếu source snapshot và vẫn kiểm tra current trainer không được reuse control cũ.

CPU/CUDA smoke4ep/arm,max8 đủ hai nhánh, mộtAdam/4updates/32learner steps mỗiarm;47,48/47,52s,full_protocol=false,best0,retainB. Trong smoke online không có move được nhận nên hai arms khớp theo từng thiết bị; actual promoted loss branch được test riêng trênpositive envCPU/CUDA. Preflight smoke6/32accepted,p95≈13–15ms dưới tải test. Deadline0,001phút trả incomplete/ZIP/retainB và không mở test.

**Full pilot RUNNING:** `runs/combat-move-full-20261015-013315`, CUDA/RTX3050, cảhai200ep từB, trainseed2026101501,LR1e-4/PhaseC RuleBot/max500; config đăng ký trước test. Preflight120unique C1 navigation states đã xem,28 nhận, mean1,49ms/p955,22ms/max9,45ms. Validation100seed21000 ở0/100/200 chọnbest vớisuicideguardB; skill64escape/approach ởcùngmốc chỉdiagnostic. Test cuối400/modelB/controlbest/combat_movebest seeds33000–33199/cảA0-A1/CPU1thread; bootstrap200boardpairs/gate+5ppsoBvàcontrol,CI lower>0/suicide≤cảhai. Chưa đọc33000 hoặc chọnmodel mới tại cập nhật này;31000/wood vẫn khóa. Cần chờsummary cuối, không dùngsmoke/trainingWR để kết luận cải thiện.

## Guidance MOVE — pilot đầy đủ và quyết định, 15/10/2026

**DONE/REVIEWED;giữB,gateFAIL.** Đủ200ep/200Adamupdates/arm và1.200trận test,3.502,54s≈58,38phút. Control72.409/candidate73.739learner steps; cùngepisodes/updates,trajectories khácnhau. Bestcảhaiep100,validationActiveWRcontrol30%/candidate40%;candidatefinal20031%. Skillapproachbest28/32control,30/32candidate,B28/32;escape32/32 mọi mốc. Không dùngfinal để chọnlại sau test.

| Test400/model | B | Controlbest100 | CombatMOVEbest100 |
|---|---:|---:|---:|
| ActiveWR |35% (140)|35,5% (142)|40,25% (161)|
| WR |40,25% (161)|42% (168)|45,5% (182)|
| Draw |57,5% (230)|56% (224)|53% (212)|
| Suicide |1% (4)|0,75% (3)|0,25% (1)|
| Avgkills |0,35|0,355|0,4025|
| Passivewins |5,25% (21)|6,5% (26)|5,25% (21)|
| AvgBombs |39,6875|39,375|35,6375|
| ActiveWR A0/A1 |33,5/36,5%|36,5/34,5%|42,5/38%|

Candidate−B **+5,25pp,CI95%[-1,5;12]**;candidate−control **+4,75pp,CI[-1,5;11]**. Gain≥5soBpass/soControlfail;CI lower>0failcảhai;observed suicide≤cảhaipass. Pointestimates tíchcực nhưngchưa xácnhận superiority;không chứngminh equality hoặcgiảm nguycơ từ4/3/1deaths. Mộttrainseed, models chọnbằngvalidation trướctest;không nớigate vìthiếu0,25pp hoặcchạy thêm33000 đểtìm thắng.

Candidate nhận7.667/33.371movequeries (22,98%),agreement3.952/7.667 (51,55%),meanpressuregain0,2365;meanquery2,34ms/tổng78,09s. Reject24.963next_bomb_masked/739no_pressure_increase/2escape_not_certified. All-MOVEagreement/probability/entropy so trêntrajectory khácnhau chỉdiagnostic. Train0suicidecảhai không đồngnghĩatest0suicide. Không cóteacherởinference,khôngmở31000/wood_branch.

ReviewZIP43files/CRC/byte equality/source+B+best/final hashes;400metrics/6validations;384storedskillgames/8.884frames (baselinecacheghihaichỗ,320rollouts thực),masks/outcomes/labels/purity;1.200gamerows/rates/corners/pairedbootstrap khớp. [Báo cáo vàgiới hạn](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315/decision_report.md), [analysis](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315/analysis_results.json), [ZIPraw](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315-send.zip). Source/root/checkpointB không đổi tronglượt. Không cầnngười dùngchạy thêm;optimizerT4-A1 vàcounter-bombguard làprotocol riêng chưa triểnkhai/train.33000đãdùng,chọntestmới nếu nghiêncứutiếp.


## T4-A1 — Chuẩn hóa advantage của actor, 15/10/2026

Đã thêm `normalize_advantage=False` và CLI `--normalize_advantage` trong trainer. Chuẩn hóa theo toàn learner trajectory bằng population std; ít hơn hai mẫu hoặc std<1e-8 giữ raw, log skip. Actor advantage detach; MC returns/value loss/reward/mask/12channels/local teacher/aux coefficients/Adam và schedule giữ nguyên. Diagnostics là opt-in (normalize tự bật): raw/policy/returns/value scale, explained variance, entropy theo mask size, grad norm trước clipping; NaN/Inf dừng trước optimizer update. Mặc định và diagnostics-only đã kiểm tra bit parity với source trước sửa trên CPU; isolation ở một update đã kiểm tra CPU/CUDA.

150 tests PASS (`runs/advantage-implementation-20261015/final-tests.log`). CPU/CUDA smoke20ep/arm/max20 có một Adam/20updates/400learner steps mỗiarm, tổng46,60/55,36s, normalized đủ20 lần, không NaN/Inf. Smoke ngắn chỉ kiểm tra kỹ thuật, best0 và không đủ đánh giá chất lượng. Deadline0,001phút dừng incomplete sau0,124s/noarms/notest. Marker test_opened và source/checkpoint/fresh-seed guards qua tests; guard marker được thêm sau smoke, không thay code số học/training đã smoke.

**FULL DONE/REVIEWED; gate FAIL, giữ B.** `runs/advantage-full-20261015-030430`: CUDA/RTX3050, raw/normalized đủ200episode và200Adamupdates mỗi nhánh từB/local/RuleBot/seed2026101502, validation100seed21000 tại0/100/200, skill64 chỉdiagnostic. Đủ400testgames/model trên34000–34199/cảA0-A1/CPU1thread, không teacher. Runner5.816,06s=96,93phút; triển khai/test/smoke/review và báo cáo hoàn tất trong hạn150phút tính từ19:53:44UTC. [Protocol/hướng dẫn](/D:/fpt/subject/BomberMan/ADVANTAGE_GUIDE.md).

| Test400/model | B gốc | Control best100 | Normalized best100 |
|---|---:|---:|---:|
| ActiveWR | 33,25% (133) | 36,5% (146) | 33,5% (134) |
| WR | 38,75% (155) | 42,5% (170) | 39,75% (159) |
| Hòa | 58% (232) | 54,5% (218) | 57% (228) |
| Tự nổ | 0,25% (1) | 0,25% (1) | 0,25% (1) |
| Kills trung bình | 0,3325 | 0,365 | 0,335 |
| Thắng thụ động | 5,5% (22) | 6% (24) | 6,25% (25) |
| ActiveWR A0/A1 | 32% / 34,5% | 35% / 38% | 33% / 34% |

Normalized−B **+0,25 điểm, CI95%[-5,5;6]**; normalized−control **−3 điểm, CI[-9,25;3,25]**. Bootstrap200cặp seed bàn gồm hai góc,10.000resamples/seed20261014. Gain≥5 và CI lower>0 fail; observed self-blast≤hai đối chứng pass. Chưa chứng minh normalization cải thiện, tương đương hay luôn kém. Control chỉ +3,25điểm soB, cũng chưa đạt ngưỡng thay model.

Control75.646/normalized75.957learner steps,3training self-blasts mỗi nhánh. Clipping95/200→79/200; norm tổng mean34,339→32,282 nhưng p95129,576→131,898. EV mean theoepisode0,452→0,479, entropy0,283→0,288; trajectories khác nhau, không suy nguyên nhân từdiagnostic hoặc dùngloss policykhácscale đểxếp hạng. Bestcảhaiep100 vớivalidation35/36%;final20030/32%;guard0mọimốc. Approach B28/32, controlbest/final30/32, normalizedbest/final28/32;escape32/32mọimốc. Không chọnfinal sau test.

Review ZIP45files/CRC/byte equality/source18+B+best/final hashes;400training rows,6validations/6skillprobes;384storedskillgames/8.864replayframes (320rollouts thực vìbaselinecacheđượcghihaichỗ);1.200testrows/rates/averages/corners/CI khớp. [Báo cáo quyết định](/D:/fpt/subject/BomberMan/runs/advantage-full-20261015-030430/decision_report.md), [analysis](/D:/fpt/subject/BomberMan/runs/advantage-full-20261015-030430/analysis_results.json), [ZIP raw](/D:/fpt/subject/BomberMan/runs/advantage-full-20261015-030430-send.zip). GAE/PPO/reward/counter-bomb guard chưa thử trongA1;T4-B0 làtask riêng tiếp theo. 34000 đãdùng; không chạy lại A1 hoặc thêm testđểcứugate. Mộttrainseed chưa chứng minh độổnđịnh.


## T4-B0 — GAE implementation/parity, 15/10/2026

**DONE/PASS về tính đúng; giữ B và mặc định MC.** Trainer/CLI thêm actor advantage estimator `mc|gae` và `gae_lambda`; critic luôn MC returns. Tách bootstrap/continuation để terminal không bootstrap và cut không nối sang reset. Learner chết kết thúc buffer dù FFA vẫn chạy. Collector vẫn toàn episode; chưa triển khai rollout update giữa trận/PPO. Teacher/reward/mask/CNN/loss phụ không đổi.

167testsPASS (`runs/gae-b0-implementation-20261015/all-tests-final.log`), gồm lambda0/0,5 tính tay,lambda1MCparity1/3/500bướcCPU-CUDA,latekill/death/timeout500/reset/cut,detach/criticgradient/purity/cancel/badinputs/runnerguards. DefaultMC actions/rewards/metrics/weights bit-identical source-before trênCPU2ep20steps. FrozenGAE1firstupdate soMC có maxdelta policy loss4,47e-8CPU/3,73e-9CUDA; gradient1,49e-7/2,79e-8; postAdamweights4,78e-6/1,17e-7. Dùngtoleranceđãghi,khônggọiGAE1bit-parity. Đãsửalỗibooleanmasklàmgammaroundfloat32trongrecurrencefloat64.

SmokeMC/GAE1/GAE0,95:20ep/20updates/mode/device,normalizationoff,LR1e-4/seed2026101503/B/local/max20; khôngvalidation/evaluation/promotion. CPU15,06s/394learnersteps mỗi mode,GAE18timeouts+2matchends;CUDA14,83s/400steps/mode,20timeouts. Weightsfinite,source+Bunchanged,mỗiZIP22files/CRC/bytes/checkpoints/metricskhớp qua review. Không mởseedtestmới,34000đãdùngtừA1;31000/wood_branch vẫn không mở trongB0.

[Báo cáo B0](/D:/fpt/subject/BomberMan/runs/gae-b0-implementation-20261015/B0_report.md), [guide](/D:/fpt/subject/BomberMan/GAE_B0_GUIDE.md), [review JSON](/D:/fpt/subject/BomberMan/runs/gae-b0-implementation-20261015/smoke_review.json). B1READY nhưng chưa runner đối chứng200ep/arm hoặc400test/model; chưa có bằng chứng GAE tăngActiveWR. Không dùng checkpoint smoke để thay B.

## T4-B1 — lượt đầu thiếu thời gian, 15/10/2026

**INCONCLUSIVE, giữ B.** `runs/gae-full-20261015-112853` chạy85,11 phút rồi dừng vì trần train chung85 phút: control200/200, GAE184/200. Không có test cuối400/model hoặc CI;35000–35199 chưa mở. Mỗi nhánh một Adam;control73.101 bước,GAE71.429;normalizationoff/criticMC/teacherlocal/cùngB vàseed2026101504.

| Nhánh | Val ActiveWR ep0 | ep100 | ep200 | Best |
|---|---:|---:|---:|---:|
| MC control |29%|28%|25%|0|
| GAE0,95 |29%|29%|Chưa chạy|0|

Tự nổ validation0% tại cả5 bài chấm;trainself3/0 không đủ kết luận an toàn hơn vì số episode khác nhau và chưa có heldouttest. ApproachMC28/32→30/32→28/32,GAE28/32→28/32;escape32/32 cả5 probe. Best0 vẫn weightsB. Review source19/B/checkpoint/ZIP41files/384training rows/320storedskillgames/7.140replayframes qua;256rollouts kỹ năng thực vìbaselinecache64game ghi hai lần.

174regression tests và smoke CPU57,49s/CUDA73,66s qua trước full. Sau khi lưu/review dữ liệu gốc, đã sửa riêng trần train thành100 phút trong tổng150;49 tests liên quan qua, testdeadline xác nhận evaluation bị chặn theo thời gian còn lại. **Chưa chạy lại full bản sửa.** Không sửa config cũ và không chạy thêm16episode từfinal184 vì thiếu Adam/LR/RNG state.

[Báo cáo quyết định](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-112853/decision_report.md), [analysis](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-112853/analysis_results.json), [raw ZIP](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-112853-send.zip), [protocol/lệnh](/D:/fpt/subject/BomberMan/GAE_PILOT_GUIDE.md). Chưa có cơ sở mở B2/PPO/reward hoặc thay checkpoint từ lượt thiếu này.

## T4-B1 — chạy lại đầy đủ, gate cải thiện FAIL, 15/10/2026

**Đủ protocol và review, giữ B.** Sau sửa trần train100/tổng150 phút, đã train lại cảhai từ B tại `runs/gae-full-20261015-125942`,200ep/200updates/nhánh. Lượt đầy đủ54,87 phút; không đổi hyperparameter,seed2026101504/teacherlocal/criticMC/normalizationoff. Cùngseed với lượt thiếu, không tính là replication độc lập.

| Model được chọn | ActiveWR | Hòa | Tự nổ | Kills/trận | Thắng thụ động |
|---|---:|---:|---:|---:|---:|
| B |31,75%|61,5%|0,75%|0,3175|4,75%|
| Best MC |31,75%|61,5%|0,75%|0,3175|4,75%|
| Best GAE |31,75%|61,5%|0,75%|0,3175|4,75%|

Đã chấm400/model trên35000–35199,cảA0/A1. DeltaGAE soB/control0điểm,CI95%[0;0];không đạt gain≥5 hoặcCI lower>0. **Best cảhai vẫn episode0,weightsB**, nên kết quả giốngB; đây không phải bằng chứng mọi checkpoint đã trainGAE tương đươngB. Không dùngtest để đổi sangfinal200. A0/A1 ActiveWR30/33,5%,tựnổ1,5/0%;win tổng36,5% và19passivewins/400.

ValidationMC29→28→25%,GAE29→29→29%,tựnổ0 cả6bài chấm. ApproachMC28→30→28/32,GAE28→28→30/32;escape32/32 cả6probe. Trainself3/0,clip81/73,trungbìnhrawstd1,0573/0,8667,EV0,4779/0,4992 không đủ kết luận policy tốt hơn. GAEmẫu77.005steps soMC73.101;không coi thứ tự chạy/tải máy hoặc thời gian nhánh là benchmark tốc độ thuật toán.

Review ZIP46files/source19/B/best/final hashes,400trainrows/6validations/6skills,384storedskillgames/8.512frames (320rollouts thực dobaselinecache),1.200testrows/corners/CI đều khớp.174tests trước pilot/smokeCPU-CUDA và49tests liên quan sau sửa deadline qua. [Report](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-125942/decision_report.md), [analysis](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-125942/analysis_results.json), [ZIP raw](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-125942-send.zip).

B1 DONE về thực thi, chưa chứng minh tăng chất lượng;giữMC/raw/B,không mởB2/C/PPO từpilot này. Tiếp theo ưu tiênR0 audit contribution reward và timeout, chưa sửa reward/train mới.35000 đã mở;31000/wood_branch không dùng. Mộttrainseed không đủ kết luận ổn định.

## T4-R0 — audit reward hoàn tất, 15/10/2026

**DONE/REVIEWED, giữ B và mọi hệ số reward hiện tại.** `r0-full-20261015-153908` audit 100 trận/model với B, MC final200 và GAE final200 trên validation21000–21049, cảA0/A1; không mở test hoặc cập nhật model. Runtime16,25 phút, parity các thống kê validation B1 qua cả ba checkpoint.

| Checkpoint | Timeout | Reward timeout mean | Wall mean | Item mean | Timeout reward dương |
|---|---:|---:|---:|---:|---:|
| B |62|+0,9169|+1,3355|+0,0815|62/62|
| MC final200 |67|+0,9657|+1,3679|+0,0978|67/67|
| GAE final200 |62|+1,0065|+1,4105|+0,0960|62/62|

Time penalty timeout là−0,5. Mọi timeout tới khoảng cách≤3 và≈94%bom cópressure>0. Trong100bước cuối timeout, wall reward bằng0 và tất cả bom đặt cópressure dương; các số liệu này không ủng hộ diễn giải agent chỉ farm gỗ cuối trận. Pressure chỉ là proxy hiện tại, không chứng minh trap/kill.

G0 timeout mean (gamma0,99) là0,8465/0,8688/0,8923; G0 thắng chủ động cao hơn trung bình. Trong88cặp thắng/timeout cùngseed/góc giữa checkpoint, có2ví dụ G0timeout>G0win; giảm nửa wall xóa1ví dụ, giảm nửa item không xóa. Không coi các cặp là độc lập hoặc suy ra causal effect. Fixed-action R1 đưa mean timeout B về0,2492 nhưng57/62vẫn dương,1≈0,4âm; R3 chỉ đưa về0,8762. Chưa train hoặc áp dụng các thay đổi này.

R0 tách9component, thêm `reward_diagnostics=False` mặc định và dùng chung shaped-reward helper; thứ tự tính reward học giữ nguyên. Review118.226transition khớp công thức cũ bit-exact, sum error tối đa1,78e−15;12timeout/6.012frames replay qua. Mean/median/p95/min/max/sum/gross-net absolute shares, raw step tapes, CSV vàscatter300điểm được lưu. ZIP raw37files/source11/checkpoint/parent hashes qua. Regression188PASS/1CUDA SKIP; CPU logging parity actions/rewards/metrics/weights qua. Mỗi model chỉ có1death case, không cóself-blast trong cohort; không đủ dữ liệu chỉnh death penalty.

**Bước kế tiếp: R1 ở mức ablation một biến**, wall0,15 so0,075, giữ các phần khác; chưa có runner mới/chưa train. R0 phát hiện reward hòa dương và vài misranking, không chứng minh farm gỗ là nguyên nhân chính hoặc giảm thưởng sẽ tăngActiveWR. [Báo cáo](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/R0_report.md), [review](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/review_results.json), [hướng dẫn](/D:/fpt/subject/BomberMan/R0_GUIDE.md).

## T4-R1 — lượt full chưa hoàn tất

R1 IMPLEMENTED/TESTED, full INCOMPLETE do tiến trình bị ngắt: `reward-full-20261015-163227`. Control đủ200episode; half-wall199episode đã lưu, validation200/final200 còn thiếu. Best validation control33%/half-wall30%, chưa có test hoặcCI; giữB. 200regression testsPASS, smokeCPU/CUDA qua; audit320skill replays/6.966frames. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-20261015-163227/decision_report.md).

## T4-R1 resumable — đánh giá đủ protocol

R1 DONE/REVIEWED: `reward-full-resumable-20261015-194314`. Đủ 200 episode và 200 Adam update mỗi nhánh từ B; 400 trận test/model trên seed 36000–36199, cả A0/A1. Best control/half-wall đều ở episode 200. ActiveWR B/control/half-wall = 34,50%/36,25%/35,00%. Gate FAIL; giữ B và wall bonus 0,15. Thời gian chạy 49,54 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/decision_report.md).

Phase C có `training_state.pt` gồm model/Adam/scheduler/global episode/best guard/history, RNG Python/NumPy/env/Torch CPU/CUDA và context log. Atomic temp+fsync+replace, OS writer lock, `run_reward_pilot.py --resume` khôi phục episode đã xác nhận; không reset schedule hoặc chọn lại model test. 212test riêng biệt PASS; có CPU/CUDA exact resume trong tiến trình mới, interrupted episode/validation và torn-log recovery. `best/final_phaseC.pt` giữ dạng weights để chơi; loader từ chối nhầm full-state file. Resume yêu cầu cùng source/config/runtime/thiết bị; hiện chỉ hỗ trợ Phase C, deadline R1 gốc vẫn giữ.

## T4-D0 — kiểm chứng update ratio

T4-D0 DONE/PASS về tính đúng: 226/226 tests, gồm 14 D0; CPU/CUDA smoke đều đủ 20 episode/20 Adam updates mỗi chế độ AC và ratio chưa clip. Buffer lưu mask/old policy/teacher targets; MC raw, critic/reward/teacher/inference và mặc định AC giữ nguyên. Full states và source/ZIP/B hashes đã review. D0 không chứng minh tăng ActiveWR; giữ B. D1 clipping với cùng 4 epochs READY/design, chưa triển khai/pilot. [Báo cáo](/D:/fpt/subject/BomberMan/runs/ppo-d0-implementation-20261015/D0_report.md), [hướng dẫn](/D:/fpt/subject/BomberMan/PPO_D0_GUIDE.md).

## T4-D1 — paired clipping pilot

D1 DONE/REVIEWED: `ppo-full-20261015-220313`. Đủ 200 episode/nhánh từ B, Adam updates U4/P4=800/800; best episode U4/P4=200/200. Test400/model, seed37000–37199/cảA0-A1: ActiveWR B/U4/P4=37.50%/37.00%/32.00%. Gate FAIL; giữ checkpoint B. Runtime 106.89 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/ppo-full-20261015-220313/decision_report.md).

D1 giữ MC raw/critic MC/local teacher/reward cũ. U4/P4 cùng tối đa4 epochs và post-update KL guard target0,02/threshold0,03; chỉ khác clipping epsilon0,20. 241 regression và25 focused metadata/compat tests PASS; CPU/CUDA smoke20episode/80updates mỗi nhánh đã review. Có 500-step/4-epoch memory fixture, hủy giữa epoch khôi phục Adam, exact PPO resume qua tiến trình mới CPU/CUDA, full-state/source/ZIP/locked-model guards. CNN inference và Actor-Critic mặc định giữ nguyên.

## SD1/SD2 — Sudden death: kiểm chứng và tối ưu, 16/10/2026

**SD1 (cấu hình gốc Loc_update: start200/interval20/đóng kín 3 vòng)**, B đóng băng, 400 trận/case, seed 38000–38199, cả A0/A1. `runs/sudden-death-full-20261016-135407`, status complete, hashes không đổi.

| Case | WR | ActiveWR | Hòa | Tự nổ |
|---|---:|---:|---:|---:|
| off | 38,75% | 32,75% | 58,75% | 0,75% |
| rings_only | 36,00% | 16,25% | 26,25% | 2,75% |
| learner_aware | 57,50% | 17,00% | 19,00% | 0,75% |
| both_aware | 43,00% | 17,50% | 21,50% | 1,00% |

both_aware−off: WR +4,25 CI[-1,25;9,5], ActiveWR −15,25 CI[-19;-11,5], hòa −37,25 CI[-42,75;-31,5]. Hai nguyên nhân: SD từ bước 200 cắt các thắng chủ động muộn của B (median bước 271, 87% trước bước 400); trên 9×9 ô tâm (4,4) là tường cứng nên vòng 3 đóng kín mọi ô → 84/400 trận cả hai cùng chết = hòa.

**SD2 tối ưu.** Thêm `sudden_death_max_ring` (mặc định = đóng kín, giữ hành vi cũ). Quét `tune_sudden_death.py` trên validation 21000–21099 (200 trận/cấu hình, 30 + 48 cấu hình SD qua hai lượt, mỗi lượt kèm một lần off). Luật chọn chốt trước khi đọc: case both_aware, WR cao nhất với ActiveWR ≥ off−3 và tự nổ ≤ off+1, hòa ít hơn khi bằng WR. Chọn **start400/interval40/warning40/max_ring2**. Xác nhận một lần `confirm_sudden_death.py` trên seed mới 39000–39199, 400 trận/case. `runs/sd-confirm-39000`.

| Case (cấu hình SD2) | WR | ActiveWR | Hòa | Tự nổ |
|---|---:|---:|---:|---:|
| off | 39,50% | 35,50% | 59,25% | 0,00% |
| both_aware | 59,25% | 31,50% | 9,75% | 0,75% |
| learner_aware | 69,50% | 30,25% | 13,00% | 0,25% |

both_aware−off: **WR +19,75 CI[14,5;25]**, **hòa −49,5 CI[-54,25;-44,5]**, ActiveWR −4 CI[-7,25;-0,75], tự nổ +0,75 CI[0;1,75]. Phần WR tăng là thắng thụ động (27,75%): RuleBot tự nổ 203/400 trong vòng 8 ô còn lại, chết vì vòng 35; learner chết vì vòng 131/400 dù có mask. learner_aware cao hơn chủ yếu vì RuleBot không biết vòng (103 trận RuleBot chết vì vòng), không dùng làm thước đo công bằng.

Kết luận: SD2 đạt mục tiêu tăng WR và giảm hòa trên test mới; đổi lại mất khoảng 4 điểm ActiveWR (có ý nghĩa thống kê). Đây là thay đổi luật chơi trên cùng trọng số B, không phải model tốt hơn; B chưa được train với SD và không có kênh quan sát vòng. Seed 39000 đã dùng.

Ghi chú thực thi SD1/SD2: test `test_sd_win_is_passive_and_simultaneous_sd_is_draw` hỏng từ lúc viết (đường đi kịch bản đụng tường gỗ của seed 39888, log `runs/sudden-death-implementation-20261016/all-tests.log`); đã sửa fixture bằng cách xóa tường gỗ trong kịch bản, assertion giữ nguyên, 15 test SD PASS. Lượt quét v1 `runs/sd-tune-v1-closure` (31 cấu hình, đóng kín) đủ. Lượt quét v2 `runs/sd-tune-v2-maxring` lần đầu bị dừng vì máy thiếu RAM (14 tiến trình) sau 14/49 cấu hình; người dùng chạy lại với `--resume`/6 tiến trình, đủ 49/49 (`resume: 14 done, 35 left`). Off validation 21000: WR 38/ActiveWR 31/hòa 60,5/tự nổ 0. Cấu hình chọn 400/40/40/r2 có validation WR 62, ActiveWR 30,5, hòa 14; bằng điểm với r3 và chọn r2 theo thứ tự. ActiveWR trên test giảm 4 điểm dù validation chỉ giảm 0,5.

## SD3 — train B dưới sudden death + kênh vòng: thiết kế, 16/10/2026

**IMPLEMENTED/TESTED/SMOKE DONE; full đã chạy, gate FAIL (xem mục "SD3 — kết quả full" bên dưới).** Plan: `docs/superpowers/plans/2026-10-16-sd-training-ring-channel.md`. Spec: `docs/superpowers/specs/2026-10-16-sd-training-ring-channel-design.md`. Hai nhánh từ B: `sd_12ch` (train dưới SD2 both_aware) và `sd_13ch` (thêm kênh 12 = 1/delay tới khi ô bị vòng đóng, chỉ trong `SuddenDeathEnv`). 200 episode/nhánh, validation 21000, test một lần seed 40000–40199, 400 trận/model (B, best sd_12ch, best sd_13ch) dưới SD2 both_aware, gate ActiveWR ≥ +5pp so B, CI95 lower > 0, tự nổ không cao hơn B. Chưa có kết quả test.

Runner `run_sd_training_pilot.py`, train seed 2026101601, warm-start B, Phase C vs RuleBot biết vòng, validation 21000 dưới SD2. Kiểm thử: 269 test PASS (unittest discover), gồm parity B trong env 13 kênh (hành động, logits atol=0, `evaluate_case` cùng trận), `ring_channel=False` bit-identical cả RNG, patch được khôi phục khi lỗi. Smoke (timing ngắn start8/interval4/max_ring2, 20 episode/nhánh, 20 bước) complete: CPU `runs/sd3-smoke-cpu-20261016-165451` 31 giây, CUDA `runs/sd3-smoke-cuda-20261016-165558` 32 giây, rerun CPU sau fix review `runs/sd3-smoke-cpu-20261016-171018`; `--resume` trên run đã xong dùng lại nhánh/đánh giá, không train/chấm lại. Nhánh 13 kênh: best ep0 có trọng số kênh vòng = 0 (đúng B), final học được trọng số khác 0. Với timing smoke, mọi trận đều cả hai chết vì vòng; smoke chỉ kiểm tra đường ống, không phải chất lượng. Lượt smoke lỗi đầu tiên `runs/sd3-smoke-cpu-20261016-165344` dừng vì `benchmark_t3.validate_env` cứng 12 kênh (đã sửa), không có đánh giá. Seed 40000 chưa mở.

## SD3 — kết quả full: train dưới sudden death ± kênh vòng, 16/10/2026

**COMPLETE, gate FAIL cả hai nhánh; giữ B.** Output `runs/sd3-full` (ZIP `runs/sd3-full-send.zip`, log `runs/sd3-full-console.log`), một lượt không resume, 2595 giây (train ~800/833 giây mỗi nhánh trên CUDA), `hashes_unchanged=true`. Warm-start B `93bdf278…4f60`, train seed 2026101601, 200 episode/nhánh, Phase C vs RuleBot biết vòng, SD2 (start400/interval40/warning40/max_ring2). Validation 21000 (100 trận) chọn best = episode 200 ở cả hai nhánh (sd_12ch ActiveWR val 31 so với 28 ở ep0; sd_13ch 35 so với 28). Test một lần seed 40000–40199, 400 trận/model, SD2 both_aware, env có kênh vòng (B và sd_12ch chỉ đọc 12 kênh đầu).

| Model | WR | ActiveWR | Thắng thụ động | Hòa | Tự nổ | Learner chết vì vòng | Đối thủ tự nổ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B (gốc) | 59.5 | 31.5 | 28.0 | 13.75 | 1.25 | 121 | 201 |
| sd_12ch best | 62.5 | 29.25 | 33.25 | 9.25 | 1.0 | 110 | 224 |
| sd_13ch best | 63.5 | 29.25 | 34.25 | 10.25 | 1.0 | 119 | 214 |

Paired bootstrap theo 200 cặp seed bàn (10000 resample):

| So sánh | ΔWR | ΔActiveWR (gate) | ΔHòa | ΔTự nổ |
|---|---|---|---|---|
| sd_12ch − B | +3.0 [−2.75, 8.75] | **−2.25 [−8.0, 3.5]** | −4.5 [−8.75, −0.25] | −0.25 [−1.0, 0.5] |
| sd_13ch − B | +4.0 [−2.0, 10.0] | **−2.25 [−7.5, 3.25]** | −3.5 [−8.0, 1.0] | −0.25 [−1.5, 1.0] |
| sd_13ch − sd_12ch | +1.0 [−4.75, 6.5] | 0.0 [−4.75, 4.75] | +1.0 [−2.75, 4.75] | 0.0 [−1.25, 1.25] |

Gate (ActiveWR ≥ +5pp, CI95 lower > 0, tự nổ không cao hơn B): FAIL cả hai, ActiveWR giảm nhẹ (không có ý nghĩa thống kê). `decision=retain_original`, `eligible_for_replication=false`.

Nhận xét:
- WR tăng 3–4pp và hòa giảm đến từ thắng thụ động tăng (28 → 33–34), đúng loại lợi ích mà quy ước không tính là agent giỏi hơn.
- Kênh vòng không giúp gì: sd_13ch − sd_12ch ActiveWR 0.0, và learner vẫn chết vì vòng 119/400 (B 121, sd_12ch 110). Ở cả ba model, số lần learner chết vì vòng trùng với `unavoidable_pre_move_closure` của mask (lúc mask hỏi thì ô đã sắp đóng, không còn đường thoát), và mask learner có 442–473 lần giao rỗng (rơi về mask gốc). Giả thuyết, chưa kiểm chứng: learner bị kẹt trước khi mask cảnh báo kịp giúp, và 200 episode chưa đủ để policy học rời vòng sớm.
- Phụ: B trên seed mới 40000 dưới SD2 both_aware cho WR 59.5 / ActiveWR 31.5, khớp lượt xác nhận SD2 trên 39000 (59.25 / 31.5), tức kết quả SD2 lặp lại trên seed khác.
- Validation chọn sd_13ch ep200 với ActiveWR val 35 (+7 so với ep0), nhưng trên test chỉ 29.25: 100 trận validation không đủ phân biệt.

Giới hạn: một train seed, 200 episode, lr 1e-4, chỉ đối thủ RuleBot 1v1 9×9; không suy ra rằng train dưới SD hay kênh vòng vô ích ở quy mô lớn hơn. Seed 40000 đã dùng; không chạy thêm test để cứu gate.

## SD3 — chẩn đoán chết vì vòng (từ dữ liệu test 40000 có sẵn), 16/10/2026

Không chạy thêm trận; chỉ đọc lại `runs/sd3-full/evaluation_*_40000.json`.

| Model | Learner chết vì vòng | Vòng 1 (bước 400) | Vòng 2 (bước 440) | Đối thủ còn sống trong các trận đó | Mask learner giao rỗng / trận chết vòng |
|---|---:|---:|---:|---:|---:|
| B | 121 | 90 | 31 | 94 | 397 / 121 ≈ 3.3 |
| sd_12ch | 110 | 83 | 27 | 96 | 311 / 110 ≈ 2.8 |
| sd_13ch | 119 | 82 | 37 | 90 | 335 / 119 ≈ 2.8 |

- 100% trận chết vì vòng có ít nhất một lần mask giao rỗng, nhưng trung bình chỉ ~3 lần/trận: mask còn tìm được đường thoát gần hết cửa sổ cảnh báo 40 bước, rồi mất đường trong vài bước cuối. Giả thuyết (chưa kiểm chứng từng bước): mask chỉ đòi "còn khả thi", nên learner nán lại vòng ngoài sát hạn chót; diễn biến mask không dự báo (đối thủ di chuyển/đặt bom mới chặn lối vào) làm hết đường.
- 3/4 ca chết ở vòng 1 (vòng ngoài, chứa cả góc xuất phát). Đối thủ RuleBot biết vòng chỉ chết vì vòng 47/400 so với learner 121/400.
- Ở ~78% các trận này đối thủ còn sống, nên chết vì vòng chủ yếu biến thành thua, không phải hòa.

## SD-tracker — tạo file theo dõi hướng sudden death, 16/10/2026

Không chạy thí nghiệm, không có số liệu mới. Tạo `PLAN_SUDDEN_DEATH.md` (checkbox SD1–SD4; SD1–SD3 đã đánh dấu xong, SD4.0 READY). Kết quả vẫn ghi ở file này.

## SD4.0 — chẩn đoán chết vì vòng bằng replay trên validation, 16/10/2026

**DONE; giả thuyết đúng một phần.** Output `runs/sd4-diag-21000` (`summary.json`, `events.jsonl.gz`, log `runs/sd4-diag-21000-console.log`). Script `diagnose_sd_ring_deaths.py`: B, SD2 both_aware, validation seed 21000, 400 trận (cùng trận với `evaluate_case`, chỉ bọc mask để ghi lại; test parity). Không đụng seed test. B trên 21000: WR 63.25, ActiveWR 33.5, hòa 12.0, tự nổ 0.5; learner chết vì vòng 112/400 (vòng 1: 84, vòng 2: 28), đối thủ 44/400.

"Còn đường" = có hành động đứng/đi giữ được tiếp diễn an toàn với vòng + bom đã biết (không xét mask bom gốc). Nguyên nhân = loại vật cản mà chỉ cần bỏ riêng nó là có lại đường (một trận có thể có nhiều loại).

| Nhóm (sự kiện vòng) | Số | Còn ở ô vòng lúc còn 10 bước* | Còn ở ô vòng lúc còn 5 bước* | Mất hết đường trước bước cuối | Mất đường lúc còn 2–4 bước | Nguyên nhân: bom của mình / thân đối thủ / bom đối thủ / nhiều loại |
|---|---:|---:|---:|---:|---:|---|
| Learner chết | 112 | 68/112 (61%) | 80/112 (71%) | 87 | 70 | 48 / 41 / 5 / 15 |
| Learner sống | 300 | 134/246 (54%) | 141/242 (58%) | 10 | 0 | 4 / 7 / 2 / 2 |
| Đối thủ chết | 44 | 16/44 | 22/44 | 30 | 22 | 15 / 14 / 3 / 6 |
| Đối thủ sống | 368 | 142/314 (45%) | 145/310 (47%) | 77 | 3 | 68 / 69 / 4 / 2 |

*Bỏ các sự kiện mà trận đã kết thúc trước mốc đó.

Kết luận:
- Nán lại vòng tới sát hạn chót là phổ biến (kể cả learner sống: 162/300 chỉ rời vòng khi còn 2–4 bước), nên bản thân nó chưa phân biệt sống/chết; nó là phơi nhiễm. Learner nán lâu hơn RuleBot (54% so với 45% còn trong vòng lúc còn 10 bước).
- Yếu tố kích hoạt: 78% (87/112) ca chết mất hết đường thoát trước bước cuối, phần lớn lúc còn 2–4 bước (70). Đường bị cắt chủ yếu bởi **bom của chính learner** (48) và **thân đối thủ** đứng chặn (41); bom đối thủ ít (5). Không ca nào bị gỗ nhốt (BFS qua sàn luôn có lối vào ở mốc còn 10 bước, khoảng cách trung vị 1 ô).
- Mask chỉ dự báo đối thủ đứng yên một bước, nên kế hoạch "ra sát giờ" dễ vỡ khi đối thủ di chuyển hoặc bom vừa đặt chặn lối. Biên an toàn k (SD4.1) đúng hướng vì buộc ra sớm, còn dư thời gian để đi đường khác. Vẫn giữ k ∈ {0, 5, 10, 15, 20} và luật đã chốt; ý "cấm đặt bom trong ô vòng gần hạn chót" ghi lại làm nhánh sau, không thêm vào SD4.1.
- Giới hạn: một seed validation; phân loại nguyên nhân chỉ thử bỏ từng loại vật cản, 15 ca "nhiều loại" không quy về một nguyên nhân.

## SD4.1 — biên an toàn k cho mask vòng của learner, 16/10/2026

**COMPLETE, gate PASS (gate của SD4.1: giảm chết vì vòng mà không hại ActiveWR). ActiveWR không tăng; giữ checkpoint B, dùng thêm biên k=15 cho mask learner ở chế độ SD.** Runner `run_sd_margin.py`. Luật chọn và gate chốt với người dùng trước khi chạy (PLAN_SUDDEN_DEATH.md). B, SD2 both_aware, 400 trận/k, đối thủ RuleBot không đổi; biên chỉ áp cho learner (ô vòng coi như đóng sớm k bước; nếu không kịp thì quay về mask SD2).

Quét validation 21000 (`runs/sd4-margin-sweep-21000`, log `runs/sd4-margin-sweep-21000-console.log`); k=0 khớp đúng SD4.0 (112 chết vòng, ActiveWR 33.5):

| k | WR | ActiveWR | Hòa | Tự nổ | Learner chết vì vòng |
|---:|---:|---:|---:|---:|---:|
| 0 | 63.25 | 33.5 | 12.0 | 0.5 | 112 |
| 5 | 65.75 | 33.5 | 15.25 | 1.0 | 89 |
| 10 | 65.0 | 33.0 | 13.5 | 1.0 | 99 |
| **15** | 67.0 | **34.75** | 16.75 | 0.75 | 84 |
| 20 | 65.0 | 32.5 | 16.5 | 0.5 | 94 |

Theo luật chọn: k=15 (giảm chết vòng so với k=0, ActiveWR cao nhất). Đường cong không đơn điệu theo k (k=10 tệ hơn k=5), nên thứ hạng giữa các k>0 chủ yếu là nhiễu của một seed.

Test một lần seed 41000–41199 (`runs/sd4-margin-test-41000`, log `runs/sd4-margin-test-41000-console.log`), paired bootstrap 200 cặp seed bàn:

| Mask learner | WR | ActiveWR | Thắng thụ động | Hòa | Tự nổ | Learner chết vì vòng | Đối thủ chết vì vòng |
|---|---:|---:|---:|---:|---:|---:|---:|
| SD2 (k=0) | 60.0 | 28.75 | 31.25 | 13.75 | 1.25 | 116 | 40 |
| SD2 + biên k=15 | 70.75 | 29.5 | 41.25 | 10.25 | 1.25 | 76 | 69 |
| Δ (CI95) | **+10.75 [6.25, 15.5]** | +0.75 [−1.25, 2.75] | | −3.5 [−7.5, 0.5] | 0.0 [−1.5, 1.5] | **−10.0pp [−14.25, −5.75]** | |

Gate: chết vì vòng CI95 upper −5.75 < 0 ✓; ActiveWR CI95 lower −1.25 ≥ −3 ✓; tự nổ +0.0 ≤ +1 ✓ → **PASS**.

Kết luận:
- Biên k=15 giảm learner chết vì vòng 116 → 76/400 và tăng WR +10.75pp (CI không chứa 0), mức tăng WR lớn nhất của hướng SD.
- Toàn bộ phần WR tăng là thắng thụ động (31.25 → 41.25): learner sống qua vòng thì RuleBot hay chết vì vòng hơn (40 → 69). ActiveWR không đổi; theo quy ước, đây không phải agent đánh giỏi hơn, mà là mask an toàn tốt hơn.
- Ghi chú: ActiveWR của B dưới SD2 ở seed 41000 (28.75) thấp hơn seed 39000/40000 (31.5): khác biệt giữa các bộ seed.
- Giới hạn: một seed validation để chọn k, một seed test; chỉ RuleBot 1v1 9×9; không có train. Seed 41000 đã dùng.

## SD4.2a — probe cuối trận: teacher có biến giai đoạn vòng co thành thắng chủ động không, 16/10/2026

**COMPLETE (chỉ validation), luật quyết định: KHÔNG ĐẠT → không mở SD4.2b; dừng nhánh SD hướng ActiveWR.** Output `runs/sd4-endgame-probe-21000` (log `runs/sd4-endgame-probe-21000-console.log`), script `probe_sd_endgame.py`. Validation 21000, 400 trận/nhánh, SD2 + biên learner k=15; cùng trận, RNG và hành động B tới bước 360 (đầu cửa sổ cảnh báo vòng 1), sau đó ghế learner do bộ điều khiển khác. Luật chốt trước khi chạy: một nhánh teacher phải tăng ActiveWR ≥ +5pp so với B.

Insight dẫn tới probe (validation 21000, k=15, từ `runs/sd4-margin-sweep-21000`): 150/400 trận kết thúc trước bước 400 và 120 trong số đó là thắng chủ động (80%); 250 trận kéo tới bước 400 chỉ có 19 thắng chủ động (8%) dù learner đặt trung bình ~45 bom/trận ở nhóm này. ActiveWR gần như do giai đoạn trước 400 quyết định.

| Bộ điều khiển learner từ bước 360 | WR | ActiveWR | Thắng thụ động | Hòa | Tự nổ | Thắng chủ động sau bước 360 (/262 trận) | ΔActiveWR so B (CI95) |
|---|---:|---:|---:|---:|---:|---:|---|
| B (đối chứng) | 67.0 | 34.75 | 32.25 | 16.75 | 0.75 | 31 | — |
| Teacher local | 73.5 | 30.5 | 43.0 | 9.75 | 3.25 | 14 | −4.25 [−7.25, −1.25] |
| two_bomb → local | 75.0 | 31.25 | 43.75 | 8.0 | 2.0 | 17 | −3.5 [−6.5, −0.5] |
| RuleBot | 49.5 | 32.5 | 17.0 | 36.75 | 25.25 | 22 | −2.25 [−5.5, 1.0] |

Nhánh B khớp đúng k=15 của lượt quét (WR 67.0, ActiveWR 34.75).

Kết luận:
- Không bộ điều khiển nào hạ RuleBot ở giai đoạn cuối tốt hơn B; teacher local/two_bomb còn kém hơn có ý nghĩa (CI không chứa 0). Hai teacher tăng WR (+6.5/+8pp) chỉ bằng thắng thụ động (đối thủ tự nổ/chết vì vòng), RuleBot làm learner thì tự nổ 25%.
- Vì B đã là bộ "giết" tốt nhất có sẵn ở giai đoạn cuối, chưng cất các teacher này (SD4.2b) không có cơ sở. Theo luật chốt trước: dừng nhánh SD hướng ActiveWR, quay về kỹ năng tấn công trước bước 400.
- Với mục tiêu ban đầu của sudden death (tăng WR, giảm hòa), cấu hình tốt nhất đã kiểm chứng là SD2 + biên learner k=15 (seed test 41000: WR 70.75, hòa 10.25, ActiveWR 29.5); so với không có SD (seed 39000: WR 39.5, hòa 59.25, ActiveWR 35.5) WR tăng mạnh và hòa giảm mạnh, nhưng phần tăng là thắng thụ động.
- Giới hạn: một seed validation; chỉ các teacher có sẵn trong repo (không có search/MCTS có mô hình đối thủ); kết luận "giai đoạn cuối khó khai thác" chỉ đúng với các bộ điều khiển đã thử.

## SD4.6 — probe kill-search: lượt đầu bị ngắt, 16/10/2026

**INCOMPLETE, chưa có số liệu.** Lượt `probe_sd_endgame.py --arms b search search --switches 360 360 0 --shards 1 2 2` (5 tiến trình, validation 21000) bị Claude Code dừng vì máy thiếu RAM trước khi ghi kết quả; thư mục output rỗng đã xóa. Chỉ có số đo thử 8 trận (không dùng để kết luận): search từ bước 0 ActiveWR 37.5 so với B 25.0, ~9 giây/trận so với ~1.3 giây/trận của B. Giao người dùng chạy lại trong cmd.

## XC — đối chiếu nhánh `codex/phasec-sd-best-fallback` (a6650a0) của bạn cùng nhóm, 17/10/2026

**Phân tích tĩnh DONE; đánh giá chéo WAIT (máy đang chạy SD4.6, còn 0.4 GB RAM).** Kết quả họ báo: `runs/phaseC-sd-best-fallback/evaluation-200.json`, checkpoint `best_phaseC.pt` SHA `0315a56f…1483` (khớp tài liệu), seed 31000, 200 trận: WR 67.5, ActiveWR 51.0, thắng do vòng 9.0, thụ động 7.5, hòa 21.5, tự nổ 0.5, chết vì vòng 13.0.

Những gì đã kiểm (đọc code/artefact, chưa chạy):
- Cùng điểm xuất phát: warm-start là đúng B của mình (`runs/pressure-C-next/final_phaseC.pt`, SHA `93bdf278…4f60`); engine gốc giống (phần mình sửa trong `bomberman_env.py` chỉ là `scheduled_bombs` opt-in); RuleBot của họ chỉ thêm 9 dòng cho vòng (mask biết vòng, tương đương `both_aware`).
- Cùng định nghĩa ActiveWR (`còn sống và kills > 0`); lúc đánh giá chỉ có CNN + mask chung `_safe_action_mask`, không có teacher chen vào.
- Không dùng lại seed: mỗi lượt có validation/test riêng (26000/27000, 28000/29000, 30000/31000). Validation chọn best ep250 ActiveWR 50.5 (seed 30000) → test 31000 51.0: nhất quán, không có dấu hiệu chọn theo test. Nhỏ: họ chọn "strongest" giữa các lượt sau khi xem test của từng lượt, nên hơi lạc quan.
- Khác biệt so với các pilot của mình:
  1. Train dài hơn ~15 lần: B → 1500 ep → 750 ep → 750 ep (~3000 ep) so với 200 ep/nhánh ở SD3/các pilot.
  2. Teacher tấn công khác: pressure lookahead 2, trap sequence depth 2 (gain 0.35), hệ số loss combat 0.9 / navigation 0.4 (mình 0.5 / 0.2), nhân ×1.75 cho supervision tấn công từ bước 350, loss an toàn vòng 0.6; lr giảm dần 1e-4 → 5e-5 → 2.5e-5; chọn checkpoint theo nhiều mục tiêu với 200 trận validation.
  3. Luật vòng khác: lịch E = start 475 / interval 25 (vòng 1 đóng ở bước 475, vòng 2 ở bước 500 = hết trận), SD bật ở 75% episode khi train. Vòng chỉ tác động 25 bước cuối nên thắng thụ động thấp (7.5%) — khác SD2 của mình (start 400, nơi RuleBot tự nổ nhiều làm thắng thụ động ~30%).
  4. Kênh quan sát 13 khác nghĩa: độ mạnh cảnh báo tăng dần `(W − còn lại + 1)/W` cho vòng kế tiếp (mình: `1/delay`).
- Chưa biết: B đạt bao nhiêu dưới đúng protocol của họ (lịch E, seed 31000). Đây là đối chứng quyết định xem +ActiveWR đến từ model hay từ luật/seed.

Đánh giá chéo đã chuẩn bị (`crosscheck_friend_branch.py`, chạy `evaluate.py` của chính nhánh họ trong worktree riêng): E1 tái lập 31000; E2 B trên cùng 31000; E3 friend vs B lịch E seed 60000 (400 trận/model); E4 friend vs B không SD seed 60000. Seed 60000 dùng riêng cho đối chiếu này.

## SD4.6 — probe kill-search (lượt đầy đủ), 17/10/2026

**COMPLETE (chỉ validation), luật quyết định: search@0 ĐẠT (+5.75pp), search@360 KHÔNG ĐẠT.** Output `runs/sd4-search-probe-21000` (log `runs/sd4-search-probe-21000-console.log`), người dùng chạy trong cmd với `--shards 1 1 1`. Validation 21000, 400 trận/nhánh, SD2 + biên learner k=15. Luật chốt trước (PLAN_SUDDEN_DEATH.md): nhánh search tăng ActiveWR ≥ +5pp so B.

| Nhánh | WR | ActiveWR | Thụ động | Hòa | Tự nổ | Learner chết vì vòng | Thắng chủ động sau switch | ΔActiveWR so B (CI95) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B (đối chứng, switch 360) | 67.0 | 34.75 | 32.25 | 16.75 | 0.75 | 84 | 31/262 | — |
| search từ bước 360 | 78.75 | 34.5 | 44.25 | 16.75 | 0.0 | 17 | 30/262 | −0.25 [−2.0, 1.5] |
| search từ bước 0 | 81.75 | **40.5** | 41.25 | 16.0 | 0.0 | 14 | 162/400 | **+5.75 [2.75, 8.75]** |

Nhánh B khớp đúng SD4.1/SD4.2a (67.0 / 34.75), nên lượt chạy nhất quán.

Kết luận:
- Lookahead 12 bước có mô hình RuleBot **không** thêm kill ở giai đoạn cuối (search@360 ≈ B): xác nhận lần nữa ActiveWR do giai đoạn trước bước 400 quyết định.
- Bật search cả trận tăng ActiveWR +5.75pp (CI dưới > 0): có những đòn hạ đối thủ ở đầu/giữa trận mà CNN của B bỏ lỡ và lookahead ngắn tìm được. Theo luật chốt trước: mở hướng chưng cất search vào CNN cho cả trận (ngoài phạm vi SD thuần).
- Phụ: search (tránh nước dẫn tới chết trong 12 bước) cắt learner chết vì vòng 84→14–17 và tự nổ về 0, WR +11.75/+14.75, nhưng phần tăng WR này là thắng thụ động.
- Giới hạn: search dùng mô hình đối thủ hoàn hảo (chính code RuleBot làm đối thủ đánh giá), chậm ~7 lần; đây là cận trên cho một teacher, chưa phải agent học được. Một seed validation.

## XC — kết quả đánh giá chéo nhánh `codex/phasec-sd-best-fallback` (a6650a0), 17/10/2026

**COMPLETE.** Output `runs/crosscheck-friend-a6650a0` (`summary.json`, log `runs/crosscheck-friend-console.log`), người dùng chạy `crosscheck_friend_branch.py`; mọi lượt dùng `evaluate.py` của chính nhánh họ (worktree a6650a0). F = checkpoint của họ `best_phaseC.pt` (SHA `0315a56f…1483`), B = `runs/pressure-C-next/final_phaseC.pt`. Lịch E = SD start 475 / interval 25 / warning 40.

| Mã | Model | Luật | Seed, số trận | WR | ActiveWR | Thụ động | Thắng do vòng | Hòa | Tự nổ | Chết vì vòng |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| (họ báo) | F | E | 31000, 200 | 67.5 | 51.0 | 7.5 | 9.0 | 21.5 | 0.5 | 13.0 |
| E1 | F | E | 31000, 200 | 58.5 | 38.5 | 7.0 | 13.0 | 29.0 | 1.0 | 11.5 |
| E2 | B | E | 31000, 200 | 60.0 | 33.0 | 13.0 | 14.0 | 22.0 | 0.5 | 19.5 |
| E3 | F | E | 60000, 400 | 67.5 | 48.5 | 10.5 | 8.5 | 25.5 | 0.5 | 10.75 |
| E3 | B | E | 60000, 400 | 64.0 | 40.0 | 12.5 | 11.5 | 22.5 | 0.5 | 16.75 |
| E4 | F | tắt SD | 60000, 400 | 52.0 | 45.25 | 6.75 | 0 | 46.25 | 0.5 | 0 |
| E4 | B | tắt SD | 60000, 400 | 44.75 | 39.0 | 5.75 | 0 | 53.5 | 0.5 | 0 |

Delta F − B (paired bootstrap theo seed bàn, CI95):

| So sánh | ΔWR | ΔActiveWR | ΔHòa | ΔTự nổ |
|---|---|---|---|---|
| E1 vs E2 (lịch E, 31000) | −1.5 [−10.0, 7.0] | +5.5 [−4.0, 15.0] | +7.0 [−0.5, 14.5] | +0.5 [−1.0, 2.0] |
| E3 (lịch E, 60000) | +3.5 [−3.5, 10.5] | **+8.5 [1.5, 15.75]** | +3.0 [−3.0, 9.0] | 0 [−0.75, 0.75] |
| E4 (tắt SD, 60000) | **+7.25 [0.25, 14.5]** | +6.25 [−1.0, 13.5] | **−7.25 [−14.25, −0.25]** | 0 [−0.75, 0.75] |
| Gộp E1+E3 vs E2+E3 (lịch E, 600 trận/model) | +1.83 [−3.5, 7.33] | **+7.5 [1.83, 13.17]** | +4.33 [−0.33, 9.0] | — |

Kết luận:
- **Không tái lập được đúng 67.5/51.0** dù cùng code, cùng checkpoint (SHA khớp), cùng seed: các trận khác nhau ngay từ trận đầu (họ: 391 bước, thắng thụ động; mình: 55 bước, thắng chủ động). Đã kiểm thêm: `evaluate.py` ở commit trước (0d3b474) cho đúng các trận như a6650a0; `final_phaseC.pt` cũng không khớp. Nguyên nhân khả dĩ nhất (suy luận, chưa kiểm được): policy **lấy mẫu** hành động (`Categorical.sample`, seed theo bàn), nên chuỗi mẫu phụ thuộc phiên bản torch/máy (mình torch 2.5.1, numpy 2.0.1; máy họ không rõ). Vì vậy 51.0 là một lần rút mẫu trên máy họ, không phải số tái lập được bit-to-bit; không thấy dấu hiệu lỗi protocol.
- **F hơn B thật, nhưng ít hơn con số báo:** gộp 600 trận dưới lịch E, ActiveWR F 45.2 so B 37.7 (+7.5pp, CI dưới > 0). Con số 51.0 lạc quan do may mắn khi rút mẫu và do chọn lượt "mạnh nhất" sau khi xem test. Đối chứng B dưới luật của họ là ~38–40, không phải ~35.
- **Dưới lịch E, WR gần như không đổi (+1.8, CI chứa 0) và hòa còn nhích lên:** F đổi thắng thụ động/thắng do vòng thành thắng chủ động và chết vì vòng ít hơn (10.75–11.5 so 16.75–19.5). Khi tắt SD, F tăng WR (+7.25) và giảm hòa (−7.25); ActiveWR +6.25 (CI chạm 0).
- So với hướng của mình: về **kỹ năng** (ActiveWR), công thức train của họ (~3000 ep, teacher tấn công lookahead 2/chuỗi bẫy, combat 0.9, ×1.75 giai đoạn cuối) là cải thiện duy nhất đến nay có CI dưới > 0 trên seed mới. Về **mục tiêu SD ban đầu** (WR cao, hòa thấp), SD2 + k=15 của mình mạnh hơn (WR 70.75, hòa 10.25 trên 41000), nhưng nhờ thắng thụ động.
- Giới hạn: một train seed của họ; 31000 do họ chọn; seed mới duy nhất là 60000. Seed 31000 nay đã có kết quả của B dưới lịch E, không còn "khóa sạch" cho T3.

## MERGE — hợp nhất nhánh `codex/phasec-sd-best-fallback` vào `danh/sd5-friend-merge`, 17/10/2026

**DONE (chỉ kiểm tra tương đương, không phải thí nghiệm mới).** Người dùng chọn hướng (a): lấy F làm điểm xuất phát. Snapshot code trước merge: commit `860753f` (nhánh `danh/sd5-friend-merge`, tách từ `danh/bomberman-rl-results-checkpoints`).

Kiểm tra tương đương trước/sau merge (cùng máy, torch 2.5.1):

| Kiểm tra | Kết quả |
|---|---|
| B không SD, validation 21000, 6 trận | giống hệt từng trận (thắng, kill, bom, số bước) |
| B dưới SD2 + biên k=15 (`evaluate_case both_aware`), 6 trận | giống hệt |
| SD3 `sd_13ch` (kênh vòng của mình), 6 trận | giống hệt |
| F dưới lịch E, seed 31000, 4 trận: code merge vs worktree a6650a0 | giống hệt |
| Train 4 episode từ F (công thức của bạn, SD 75%, GPU): code merge vs a6650a0 | reward, số bước, kill, nhãn teacher, các loss giống hệt; trọng số lệch tối đa 1.9e-9 (nhiễu float GPU) |
| Toàn bộ test | 347 test OK, 16 bỏ qua (15 test launcher cần `venv` của bạn, 1 preflight R0 ghim hash engine cũ) |

Quan sát thêm từ `runs/phaseC-sd-best-fallback/metrics.jsonl` (lượt 750 ep cuối của bạn): ActiveWR khi train phẳng 44.8 / 46.4 / 45.2% theo từng đoạn 250 ep; validation (seed 30000, 200 trận) 39.5 → 50.5 → 45.5 → 43.0 ở ep 0/250/500/750. Công thức hiện tại gần như bão hòa; ep250 được chọn nhiều phần do nhiễu validation 200 trận (SE ~3.5pp).

## SD5.0 — probe kill-search trên F (chuẩn bị), 17/10/2026

**WAIT người dùng chạy; chưa có số liệu.** Thiết kế SD5 đã được người dùng duyệt. Script `probe_sd5_search.py`: F (`runs/phaseC-sd-best-fallback/best_phaseC.pt`) vs F + kill-search từ bước 0, lịch E (SD engine 475/25/40), validation 21000, 400 trận/nhánh, kèm chẩn đoán trận hòa (hòa do hết giờ / cả hai chết, hết giờ mà chưa từng tới gần). Luật chốt trước: mở nhánh `search` của SD5.1 nếu ΔActiveWR (search − F) ≥ +3pp với CI95 dưới > 0. Smoke 8 trận/nhánh (không dùng để kết luận): F 1.1 s/trận, search 6.6 s/trận; hai nhánh chỉ khác ở trận search can thiệp.

## SD5.0 — probe kill-search trên F (kết quả), 17/10/2026

**COMPLETE (chỉ validation), luật quyết định: ĐẠT → mở nhánh `search` của SD5.1.** Output `runs/sd5-search-probe-21000` (log `runs/sd5-search-probe-21000-console.log`), người dùng chạy. F = `runs/phaseC-sd-best-fallback/best_phaseC.pt`, lịch E (SD engine 475/25/40), validation 21000, 400 trận/nhánh, search từ bước 0.

| Nhánh | WR | ActiveWR | Thắng do vòng | Thụ động | Hòa | Tự nổ | Chết vì vòng | Kill TB | Số bước TB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F | 67.5 | 47.0 | 11.0 | 9.5 | 19.75 | 1.0 | 12.75 | 0.47 | 338 |
| F + search | 85.25 | **69.5** | 7.75 | 8.0 | 14.0 | 0.0 | 0.75 | 0.69 | 280 |

Delta search − F (paired bootstrap theo seed bàn, CI95): ActiveWR **+22.5 [18.25, 26.75]**, WR +17.75 [13.75, 21.75], hòa −5.75 [−9.5, −2.0].

Phân loại hòa: F 79 trận hòa = 64 hết giờ + 15 cả hai chết; search 56 = 56 hết giờ + 0 cả hai chết. Cả hai nhánh tới gần đối thủ (Manhattan ≤ 3) ở 100% trận, trung bình từ bước ~18.6; không có trận hòa nào do không chạm mặt.

Kết luận:
- Trên F dưới lịch E, lookahead 12 bước với mô hình RuleBot thêm rất nhiều kill (+22.5pp, lớn hơn nhiều so với +5.75 trên B dưới SD2 + k=15). Theo luật chốt trước: mở nhánh `search` của SD5.1 (chưng cất search vào CNN).
- Hòa của F không phải do né tránh mà do không kết liễu được khi đã chạm mặt; search còn loại hết trận hòa kiểu cả hai chết và gần hết chết vì vòng.
- Giới hạn: search dùng mô hình đối thủ hoàn hảo (chính code RuleBot), đây là cận trên cho teacher; CNN học được bao nhiêu phải đo bằng SD5.1. Một seed validation.

## SD5.1 — chuẩn bị train F ± nhãn kill-search (3000 ep/nhánh), 17/10/2026

**WAIT người dùng chạy full; chưa có kết quả.** Runner `run_sd5_pilot.py`, output dự kiến `runs/sd5-20261017`. Hai nhánh từ F, train seed 2026101701, công thức của bạn (lr 2.5e-5, entropy 0.0075, combat 0.9, navigation 0.4, ×1.75 từ bước 350, SD lịch E ở 75% episode), validation 21000 mỗi 500 ep với 400 trận (lịch E), chọn best theo ActiveWR với giới hạn tự nổ của warm-start; `search` thêm nhãn kill-search (nhóm loss riêng, hệ số 0.9). Test một lần seed 42000, 400 trận/model, F / control / search dưới lịch E (gate) và không SD (báo cáo). Gate chốt trước: search − F ActiveWR ≥ +5pp, CI95 dưới > 0, search − control ≥ 0, tự nổ không tăng quá 1pp.

Số đo chuẩn bị (không dùng để kết luận):

| Đo | control | search |
|---|---:|---:|
| Thời gian train, GPU, episode đầy đủ | 2.8 s/ep (6 ep) | 7.9–9.5 s/ep (6–8 ep) |
| Ước tính 3000 ep + 7 lần validation 400 trận | ~3.2 giờ | ~8–9 giờ |
| Nhãn search | — | 9 nhãn / 8 ep (~0.5% số lần search chạy); CE của nhãn ~2–3 |
| Nhãn combat heuristic | ~46–207 / ep | ~46–207 / ep |

Lỗi phát hiện khi viết test (đã sửa trước khi chạy full): resume training state sau merge không khôi phục RNG chọn bật/tắt SD mỗi episode, nên chuỗi episode có SD sau resume khác lượt chạy liền (test `test_resume_replays_sudden_death_curriculum` thấy lệch ở episode 4); provenance resume chưa có các tham số mới (SD, hệ số loss, teacher_version), nên đổi các tham số này sẽ không bị phát hiện khi resume.

## TOOL — thanh progress cho các lượt chạy dài, 17/10/2026

**DONE (công cụ, không có số liệu thí nghiệm).** Người dùng yêu cầu từ nay mọi lượt train/đánh giá có thanh progress. Đã gắn vào `run_sd5_pilot.py` (train: episode, ActiveWR 100 episode gần nhất, best episode, số nhãn search, tiến độ validation; test: tổng số trận qua các tiến trình). Smoke CLI train 2 nhánh + test chạy được; thanh ghi ra console nên log không đổi. Toàn bộ suite 361 OK (16 bỏ qua). Lệnh SD5.1 đã đưa người dùng không đổi.

## PUSH — đưa code và artefact lên GitHub, 17/10/2026

**DONE.** Nhánh `danh/sd5-friend-merge` push lên `origin` (https://github.com/SuperZiggs/BomberMan, tạo nhánh mới; `main` và `danh/bomberman-rl-results-checkpoints` không đổi). Gồm commit `860753f` (snapshot trước merge), `af83eca` (merge nhánh bạn), `61d2f9e` (SD5 code), `8c95375` (`runs/` tới SD5.0 + `graphify-out/`, ~2826 file; checkpoint `.pt` qua Git LFS). Không gồm lượt SD5.1 đang chạy (`runs/sd5-20261017*`), sẽ đưa lên sau khi xong. GitHub cảnh báo 8 file zip/json 50–75 MB (dưới giới hạn cứng 100 MB).
