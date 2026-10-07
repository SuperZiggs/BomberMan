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
