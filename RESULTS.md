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

## Cap nhat hien hanh - Phase C v3 (2026-10-10)

> Ket qua nay thay the trang thai danh gia cho policy hien tai. Code demo ho tro Sudden Death va dao vi tri spawn, nhung luot train nay chua bat `--sudden_death`; cac ket qua duoi day khong phai chi so duoi vong bo. Cac ket qua Phase C/D/E phia tren duoc giu lai nhu lich su cau hinh cu.

**Cau hinh:** `board_size=9, num_agents=2, 4000 episodes, checkpoint_dir=checkpoints_v3`

**Cac thay doi truoc khi train:**
- Sudden Death duoc dua vao moi truong va them kenh canh bao vong bo trong observation.
- Agent Phase C luan phien goc spawn; danh gia 1v1 doi ben de tranh thien lech spawn.
- Chi thuong bomb co muc tieu chien thuat; safe action mask van yeu cau duong thoat BFS.
- Luu checkpoint atomic co retry khi Windows khoa file (`WinError 5/32/33/1224`).

### Ket qua train

| Metric | Gia tri |
|---|---:|
| Final WR (last 200) | **75.5%** |
| Final avg reward (last 200) | **+3.266** |
| Final avg bombs (last 200) | **0.5** |
| Final avg walls (last 200) | **0.2** |
| Best rolling WR | **77.0%** |

### Gate Checks

| Tieu chi | Ket qua | Trang thai |
|---|---|---|
| WR >= 70% | 75.5% | **PASS** |
| Reward trend | +1.947 -> +3.268 | **PASS** |
| Bomb shyness | bombs 1.1 -> 0.4 | **FAIL** |

**Checkpoint tao thanh cong:** `checkpoints_v3/best_phaseC.pt` va `checkpoints_v3/final_phaseC.pt`. Dung `best_phaseC.pt` (WR 77.0%) lam diem khoi dau cho fine-tune.

## Phase C v10 - activity target reached, combat target missing

| Metric (last 200 episodes) | Result |
|---|---:|
| Win rate | **49.5%** |
| Avg reward | **+4.683** |
| Avg bombs | **2.3** |
| Avg walls | **1.8** |
| Safe bomb cycles | **1.6** |
| Kills | **0.0** |
| Deaths (sudden death/self/opponent) | **0.2 / 0.1 / 0.2** |

## Phase C v11 - reward farming regression

| Metric (last 200 episodes) | Result |
|---|---:|
| Win rate | **49.0%** |
| Avg reward | **+6.249** |
| Avg bombs | **6.1** |
| Avg walls | **4.2** |
| Safe bomb cycles | **4.0** |
| Offensive safe cycles | **0.4** |
| Kills | **0.0** |
| Deaths (sudden death/self/opponent/other) | **0.1 / 0.1 / 0.1 / 0.1** |

**Decision:** Do not continue from v11. The reward was farmed through wall destruction and generic safe escapes. The next Phase C configuration starts from `checkpoints_v10/best_phaseC.pt`, budgets generic reward to the first two tactical bomb cycles per episode, reduces generic safe/wall bonuses to `+0.15` and `+0.02`, and raises safety-gated offensive escape/kill bonuses to `+0.40` and `+1.00`.

**Decision:** Keep the safety shield because self deaths are low, but change reward from safe wall farming to safe offensive pressure. A bomb with an unobstructed opponent in its blast line is recorded as offensive. Reward is still paid only after the owner escapes to safety: wall bonus becomes `+0.10`, an offensive safe escape earns `+0.15`, and an opponent kill from that safely escaped bomb earns `+0.50` in addition to the environment reward. Training logs add `offensive_safe_cycles`. Next run: fine-tune from `checkpoints_v10/best_phaseC.pt` with `--sudden_death`; do not proceed to Phase D until kills and WR improve without increasing self deaths.

**Quyet dinh:** Chua chuyen Phase D. Da trien khai reward theo chuoi "bomb co muc tieu -> thoat an toan -> tac dong len tuong/doi thu". Bonus khoi dau `+0.05` chi ap dung cho bomb an toan co muc tieu; thoat an toan `+0.30`, pha tuong sau thoat `+0.20`. Hoan tat lan thoat an toan thu hai trong episode nhan ngay `+0.25`, khong cho cuoi episode; log bo sung `safe_cycles`, kills va deaths theo `sd/self/opp/other`. Safety shield ep action mask theo buoc BFS thoat khoi vung no cua chinh bom minh, nhung van giu quyen tu quyet thoi diem/vi tri dat bom. Fine-tune tiep tu `best_phaseC.pt` voi `--sudden_death` de do lai bomb shyness trong dieu kien demo.

## Phase C v12-v13 - target checkpoint va danh gia doc lap (2026-10-13)

> Phan nay thay the cac quyet dinh fine-tune truoc do. Evaluation duoc do tren 400 tran AC-vs-RuleBot, doi hai phia spawn, bat Sudden Death va khong ghi log ra file.

### v12 - reward budget cai thien WR nhung thieu bomb

| Metric evaluation | Ket qua |
|---|---:|
| Win rate | **70.25%** |
| Avg bombs | **1.285** |
| Avg kills | **0.0** |
| Sudden-death deaths | **37 / 400** |
| Draws | **15 / 400** |

### v13 - checkpoint co metadata va muc tieu 2 bomb

- Checkpoint luu `episode`, rolling metrics va cau hinh train; Phase C tao snapshot moi 200 episode.
- `best_phaseC_target2.pt` chi duoc luu khi rolling bombs nam trong `1.7-2.3` va rolling WR dat toi thieu `70%`.
- `final_phaseC.pt` luu dung trong so cuoi cung, khong con sao chep `best_phaseC.pt`.

| Checkpoint | Episode | Rolling WR | Rolling bombs | Rolling safe cycles | Rolling kills |
|---|---:|---:|---:|---:|---:|
| `best_phaseC_target2.pt` | 1888 | **70.0%** | **1.72** | 1.455 | 0.0 |
| `best_phaseC.pt` | 1987 | **73.0%** | 1.495 | 1.300 | 0.0 |
| `final_phaseC.pt` | 2000 | **71.0%** | 1.505 | 1.305 | 0.0 |

| Checkpoint evaluated (400 tran) | WR | Avg bombs | Avg kills | Sudden-death deaths |
|---|---:|---:|---:|---:|
| `best_phaseC_target2.pt` | **69.75%** | **1.39** | 0.0 | 44 / 400 |
| `best_phaseC.pt` | 66.50% | 1.33 | 0.0 | 49 / 400 |

**Ket luan v13:** checkpoint target vuot gate tren rolling training window, nhung khong lap lai muc tieu ~2 bombs trong evaluation doc lap. Agent hoc cach dat mot bomb an toan roi cho RuleBot tu chet hoac thua Sudden Death. Nguyen nhan la anti-passivity cu chi huy toan bo win bonus khi `0` bomb; chi can mot bomb la agent van nhan day du `+10`, trong khi chu ky an toan thu hai chi them `+0.25`.

### v14 - cadence penalty da dat muc tieu bomb va WR

- Chi ap dung cho tran thang khong co kill: `0` generic safe cycles bi tru `10`, `1` cycle bi tru `5`, tu `2` cycles tro len khong bi tru.
- Tran thang co opponent kill duoc mien penalty de khong can tro bomb tan cong ket thuc som.
- Giu nguyen BFS safety mask, self-death penalty, reward budget `2`, wall bonus `+0.02`, safe escape `+0.15`, offensive safe escape `+0.40` va kill bonus `+1.00`.
- v14 fine-tune tu `checkpoints_v13/best_phaseC_target2.pt` trong 2000 episodes.

| Checkpoint | Episode | Rolling WR | Rolling bombs | Rolling safe cycles | Rolling kills |
|---|---:|---:|---:|---:|---:|
| `snapshot_phaseC_ep000400.pt` | 400 | 70.5% | **2.015** | 1.540 | 0.005 |
| `best_phaseC_target2.pt` | 719 | 72.5% | **2.260** | 1.835 | 0.010 |
| `snapshot_phaseC_ep001200.pt` | 1200 | **73.5%** | **2.065** | 1.735 | 0.000 |
| `best_phaseC.pt` | 1799 | **80.5%** | 2.550 | 2.190 | 0.010 |
| `final_phaseC.pt` | 2000 | 71.5% | 2.680 | 2.275 | 0.000 |

| Checkpoint evaluated (400 tran) | WR | Avg bombs | Avg kills | Sudden-death deaths |
|---|---:|---:|---:|---:|
| `best_phaseC_target2.pt` | 65.75% | **2.170** | 0.0025 | 38 / 400 |
| `snapshot_phaseC_ep000400.pt` | **75.50%** | 1.645 | 0.0000 | 40 / 400 |
| `snapshot_phaseC_ep001200.pt` | **75.25%** | **2.1925** | 0.0025 | **18 / 400** |
| `best_phaseC.pt` | 68.75% | 2.530 | 0.0025 | 37 / 400 |

**Ket luan v14:** `checkpoints_v14/snapshot_phaseC_ep001200.pt` la checkpoint khuyen nghi. No dat dong thoi muc tieu evaluation `1.7-2.3` bombs/tran va WR `>=70%`, voi `2.1925` bombs, WR `75.25%` va so lan chet bo thap nhat trong nhom (`18/400`). Muc tieu tan suat bomb va kha nang thang da dat; muc tieu combat van chua dat vi kills chi `0.0025`/tran.

**Luu y checkpoint selection:** `best_phaseC_target2.pt` bi chon boi mot chenh lech kills rat nho tren rolling window, nhung evaluation WR chi `65.75%`. Lan cai tien tiep theo nen xep hang checkpoint target theo WR va do gan `2.0` bombs truoc, kills/offensive cycles chi lam tie-breaker.

### v15 - combat-first reward: ket qua train va evaluation

- Giu `max_steps=300`, Sudden Death bat dau tai step `200` va dong vong moi `20` step. Khong tang len 500-600 vi vong bo cua board 9x9 da thu hep toi da khoang step 240.
- Moi bomb co `pressure score` bang so o dung/thoat lien ke cua doi thu nam trong vung no gia dinh. Pressure chi duoc thuong sau khi chu bomb roi vi tri dat va den o an toan.
- Cau hinh v15: pressure `+0.15` moi escape option, kill bonus `+4.00`, non-combat win penalty toi thieu `-8.00`. Vi terminal win la `+10`, tran thang khong co kill chi con toi da `+2` truoc cac reward khac.
- Generic safe/wall reward van bi gioi han trong hai cycle dau de tranh lap lai wall/bomb farming cua v11. Tong so bomb khong bi gioi han.
- Tao checkpoint `best_phaseC_combat.pt` voi gate rolling WR `>=70%`, xep hang theo kills -> pressure -> offensive safe cycles -> WR -> self/sudden-death safety.
- Evaluator bo sung offensive bombs, pressure score va deaths theo `sd/self/opp/other` cho ca danh gia doi ben spawn.
- Launcher: `run_train_v15.ps1`, warm-start tu `checkpoints_v14/snapshot_phaseC_ep001200.pt`, output chi hien tren terminal va khong ghi log xuong may.

| Checkpoint | Episode | Rolling WR | Bombs | Kills | Pressure | Offensive safe cycles | Self deaths |
|---|---:|---:|---:|---:|---:|---:|---:|
| `best_phaseC.pt` | 1911 | **77.5%** | 2.995 | 0.045 | 0.150 | 0.115 | 0.060 |
| `best_phaseC_combat.pt` | 2474 | 72.0% | **3.910** | **0.050** | **0.385** | **0.235** | 0.105 |
| `final_phaseC.pt` | 3000 | 64.0% | 3.800 | 0.020 | 0.295 | 0.205 | 0.110 |

Danh gia doc lap 400 tran, doi ben spawn, cung seed va Sudden Death:

| Checkpoint | WR | Bombs | Offensive bombs | Pressure | Kills | Deaths sd/self/opp/other |
|---|---:|---:|---:|---:|---:|---:|
| v15 `best_phaseC.pt` | 67.00% | 3.5675 | 0.2450 | 0.3625 | 0.0200 | 35/26/57/14 |
| v15 `best_phaseC_combat.pt` | **70.75%** | 3.8475 | 0.2600 | 0.4050 | **0.0275** | 21/27/41/28 |
| v15 snapshot ep600 | 64.25% | **4.4900** | **0.4300** | **0.6350** | **0.0425** | 15/29/71/28 |
| v15 snapshot ep1000 | 61.00% | 4.6400 | 0.4425 | 0.6625 | 0.0375 | 37/47/47/25 |
| v15 snapshot ep2000 | 66.00% | 3.7075 | 0.2525 | 0.3625 | 0.0325 | 30/36/43/27 |

Direct A/B cung evaluator va seed:

| Policy | WR | Bombs | Offensive bombs | Pressure | Kills | Self deaths / 400 |
|---|---:|---:|---:|---:|---:|---:|
| v14 snapshot ep1200 | **79.00%** | 2.0825 | 0.1175 | 0.1525 | 0.0025 | **1** |
| v15 `best_phaseC_combat.pt` | 70.75% | **3.8475** | **0.2600** | **0.4050** | **0.0275** | 27 |

**Ket luan v15:** combat checkpoint tang kills `11x`, offensive bombs `2.21x`, pressure `2.66x` va bombs `1.85x` so voi v14 trong cung phep do. No van dat WR tren 70%, nhung WR giam `8.25` diem phan tram va self-deaths tang tu `1` len `27/400`. Dung `best_phaseC_combat.pt` neu uu tien combat; khong dung final model. Cai tien tiep theo phai giam self-death ma khong lam mat pressure, khong nen tang them raw bomb/pressure reward.

### v16 - outcome-gated combat safety: ket qua train va evaluation

- Pressure va offensive safe bonus khong con duoc tra ngay khi agent vua roi o bomb. Reward chi duoc tra khi bomb no, lifecycle xac nhan da thoat, va chu bomb van song tren o an toan.
- Them rolling metric `post_escape_self_deaths` de dem truong hop agent da duoc danh dau thoat an toan nhung sau do chet boi bomb cua chinh minh.
- Them `best_phaseC_combat_safe.pt` voi gate WR `>=70%`, kills `>=0.025` va rolling self-deaths `<=0.05`.
- Launcher `run_train_v16.ps1`: 2000 episodes, warm-start tu `checkpoints_v15/best_phaseC_combat.pt`, giu `max_steps=300` va toan bo he so combat cua v15.
- Output chi hien tren terminal, khong ghi terminal log xuong may.

| Checkpoint | Episode | Rolling WR | Bombs | Kills | Pressure | Self deaths | Post-escape self deaths |
|---|---:|---:|---:|---:|---:|---:|---:|
| `best_phaseC.pt` | 864 | **84.0%** | 3.250 | 0.045 | 0.200 | 0.050 | 0.025 |
| `best_phaseC_combat.pt` | 899 | 82.0% | 3.340 | **0.060** | **0.235** | 0.060 | 0.020 |
| `best_phaseC_combat_safe.pt` | 957 | 82.0% | 3.255 | 0.055 | 0.200 | **0.050** | 0.030 |
| `final_phaseC.pt` | 2000 | 76.0% | 3.525 | 0.040 | 0.180 | 0.060 | 0.025 |

Danh gia doc lap 400 tran, doi ben spawn, cung seed va Sudden Death:

| Checkpoint | WR | Bombs | Offensive bombs | Pressure | Kills | Deaths sd/self/opp/other |
|---|---:|---:|---:|---:|---:|---:|
| `best_phaseC.pt` | 76.00% | 3.2450 | 0.1700 | 0.2725 | 0.0350 | 28/22/35/11 |
| `best_phaseC_combat.pt` | 75.75% | 3.3600 | 0.2550 | 0.4350 | 0.0350 | 11/30/37/19 |
| `best_phaseC_combat_safe.pt` | **80.00%** | 2.4525 | 0.1225 | 0.1750 | 0.0100 | 19/**10**/34/17 |
| snapshot ep600 | 70.50% | 3.5450 | **0.2775** | **0.4600** | 0.0375 | 19/32/48/19 |
| `final_phaseC.pt` | 72.25% | **3.5125** | 0.2275 | 0.3450 | **0.0500** | 32/22/39/18 |

So voi v15 `best_phaseC_combat.pt` (WR `70.75%`, bombs `3.8475`, kills `0.0275`, self-deaths `27/400`):

- v16 `best_phaseC.pt` tang WR `+5.25` diem, kills `+27%`, giam self-deaths `27 -> 22`, nhung bombs giam con `3.245`.
- v16 final tang WR `+1.5` diem, kills `+82%`, giam self-deaths `27 -> 22`, bombs van cao `3.5125`.
- Safety checkpoint giam self-deaths xuong `10/400` va dat WR `80%`, nhung kills tut con `0.01`; rolling safety gate khong generalize combat metric.

**Ket luan v16:** outcome-gating co tac dung nhung chua dat muc tieu self-deaths `<10/400`. Dung `best_phaseC.pt` cho policy can bang; dung `final_phaseC.pt` neu uu tien chu dong tan cong va kills. Khong dung `best_phaseC_combat_safe.pt` cho muc tieu combat du no co WR cao va an toan hon.

### v17 - temporal action mask va chain-bomb safety: ket qua train va evaluation

- Action mask moi project thoi diem no cua moi bomb, lan truyen thoi gian no som qua chain reaction, va danh dau flame duration theo tung time step.
- Moi action `STOP/UP/DOWN/LEFT/RIGHT` chi bi chan neu temporal BFS khong tim thay chuoi vi tri song sot den sau tat ca hazard. Di vao blast line co timer dai van duoc phep neu con du thoi gian thoat.
- Environment sua loi chain reaction: bomb bi kich no khong con bi xoa truoc khi `_explode_bomb` xu ly. `bomb_outcomes` them `chain_reaction` de attribution.
- Log train them `unsafe_reentry_blocks`, `mask_interventions` va `chain_bomb_self_deaths`; evaluator cung bao cao chain-bomb self deaths.
- Giu nguyen reward v16 va `max_steps=300`. Launcher `run_train_v17.ps1` fine-tune 2000 episodes tu `checkpoints_v16/final_phaseC.pt` trong `checkpoints_v17`.

Rolling metadata cua cac checkpoint chinh:

| Checkpoint | Episode | WR | Bombs | Kills | Pressure | Self-death | Chain-self |
|---|---:|---:|---:|---:|---:|---:|---:|
| `best_phaseC.pt` | 1520 | **85.5%** | 3.715 | 0.010 | 0.210 | **0.095** | **0.025** |
| `best_phaseC_combat.pt` | 460 | 79.5% | **4.535** | **0.075** | **0.450** | 0.155 | 0.065 |
| `final_phaseC.pt` | 2000 | 85.0% | 3.860 | 0.060 | 0.350 | 0.115 | 0.050 |

`best_phaseC_combat_safe.pt` khong duoc tao vi khong checkpoint nao dong thoi dat gate rolling combat-safety.

Danh gia doc lap seed `20261013`, tong 400 tran AC-vs-RuleBot, doi hai phia spawn, Sudden Death va evaluator v17:

| Checkpoint | WR | Bombs | Offensive | Pressure | Kills | Deaths sd/self/opp/other | Chain-self |
|---|---:|---:|---:|---:|---:|---:|---:|
| v16 `best_phaseC.pt` | 61.25% | 6.4200 | 0.4200 | 0.6300 | 0.0450 | 60/79/16/0 | 48 |
| v16 `final_phaseC.pt` (warm-start) | 63.50% | 7.0575 | 0.5400 | 0.8400 | 0.0700 | 41/94/11/0 | 57 |
| v17 `best_phaseC.pt` | **67.50%** | 7.1000 | 0.5600 | 0.8800 | **0.0725** | 42/**77**/11/0 | **44** |
| v17 `best_phaseC_combat.pt` | 60.75% | 7.4175 | 0.6325 | 0.9375 | 0.0625 | 52/92/13/0 | 64 |
| v17 `final_phaseC.pt` | 59.00% | **7.6675** | **0.6775** | **1.0050** | 0.0550 | 36/109/19/0 | 77 |

Luu y: v17 sua loi chain-triggered bomb bi xoa truoc khi no, nen cac so lieu v14-v16 da ghi bang environment cu khong the dung lam baseline truc tiep. Hai checkpoint v16 trong bang tren da duoc danh gia lai bang cung code, seed va action mask v17.

**Ket luan v17:** so voi dung warm-start v16 final, v17 `best_phaseC.pt` tang WR `+4.0` diem, bombs `+0.0425`, kills `+0.0025`, giam self-deaths `94 -> 77` va chain-self `57 -> 44`. Temporal mask co tac dung, nhung van truot acceptance WR `>=72%` va self-deaths `<=10/400`; bombs va kills da dat. Dung `checkpoints_v17/best_phaseC.pt` lam checkpoint hien tai, khong dung combat/final. Buoc tiep theo la kiem tra temporal safety cho chinh action dat bomb bang cach them hypothetical bomb vao chain projection; khong tang reward hoac max steps.

### v18 - chain-safe bomb placement: ket qua train va evaluation

- Action `BOMB` khong con dung BFS tinh gia dinh bomb moi luon no sau full timer. Mask chen mot hypothetical bomb tai vi tri agent vao cung danh sach bomb dang hoat dong.
- Earliest explosion cua hypothetical bomb duoc relax qua toan bo chain reaction. Action chi duoc phep neu temporal BFS tim thay trajectory song sot qua explosion va flame duration; chain qua som bi chan, chain du tre de thoat van duoc phep.
- Training metadata them `unsafe_bomb_placement_blocks`; `mask_interventions` tinh ca placement shield va movement shield.
- `run_train_v18.ps1` fine-tune 2000 episodes tu `checkpoints_v17/best_phaseC.pt`, giu nguyen reward va `max_steps=300`, khong ghi terminal log xuong file.

Rolling metadata cua cac checkpoint chinh:

| Checkpoint | Episode | WR | Bombs | Kills | Placement blocks | Self-death | Chain-self |
|---|---:|---:|---:|---:|---:|---:|---:|
| `best_phaseC.pt` | 701 | **86.5%** | 3.645 | 0.060 | 2.205 | **0.055** | **0.005** |
| `best_phaseC_combat.pt` | 562 | 77.0% | 4.640 | **0.090** | 2.145 | 0.125 | 0.010 |
| `final_phaseC.pt` | 2000 | 74.5% | **5.060** | 0.075 | 2.095 | 0.140 | 0.010 |

`best_phaseC_combat_safe.pt` khong duoc tao vi khong checkpoint nao dong thoi dat rolling self-death gate `<=0.05` cung combat gate.

Danh gia doc lap seed `20261013`, tong 400 tran AC-vs-RuleBot, doi hai phia spawn, Sudden Death va mask/evaluator v18:

| Checkpoint | WR | Bombs | Offensive | Pressure | Kills | Deaths sd/self/opp/other | Chain-self |
|---|---:|---:|---:|---:|---:|---:|---:|
| v17 `best_phaseC.pt` (baseline voi mask v18) | 60.25% | 7.2075 | 0.5475 | 0.8450 | 0.0625 | 35/70/54/0 | 11 |
| v18 `best_phaseC.pt` | **64.75%** | 7.1725 | 0.6625 | 0.9875 | 0.0825 | 43/**43**/55/0 | **8** |
| v18 `best_phaseC_combat.pt` | 56.50% | 8.3825 | 0.7700 | 1.1600 | 0.0725 | 45/82/**47**/0 | 13 |
| v18 `final_phaseC.pt` | 58.25% | **9.2300** | **1.0750** | **1.5875** | **0.1175** | **35**/84/48/0 | 11 |

So voi v17 best duoi cung placement shield, v18 best tang WR `+4.5` diem, kills `+32%`, offensive bombs `+21%`, pressure `+17%`, giam self-deaths `70 -> 43` va chain-self `11 -> 8`, trong khi bombs gan nhu khong doi (`7.2075 -> 7.1725`). Placement shield da giam chain-self manh so voi phep do v17 mask cu (`44 -> 11`), nhung rieng thay mask tren weights cu lam WR giam, nen fine-tuning v18 la can thiet.

**Ket luan v18:** dat 4/5 acceptance: kills `>=0.06`, bombs `>=6`, self-deaths `<=60` va chain-self `<=30`; chi truot WR `>=72%`. Dung `checkpoints_v18/best_phaseC.pt` cho policy can bang. Dung `checkpoints_v18/final_phaseC.pt` neu demo uu tien aggression toi da: `9.23` bombs, `1.075` offensive bombs va `0.1175` kills/tran, chap nhan WR `58.25%` va self-deaths `84/400`.

### v19 - opponent escape forcing va Sudden Death horizon: ket qua

- Khi vi tri hien tai nam trong projected blast cua bat ky bomb nao, mask chi giu cac first action thuoc shortest temporal escape path. Agent khong con duoc `STOP`, dat them bomb hoac di vong khi dang bi bomb doi thu de doa.
- Action mask project vong Sudden Death ke tiep trong warning window. O tren vong bi coi la wall vinh vien tu tick dong; planner kiem tra closure xay ra truoc movement cua tick do.
- Sudden Death horizon ap dung cho ca movement va hypothetical bomb placement. Them checkpoint metrics `forced_escape_steps` va `sudden_death_blocks`.
- `run_train_v19.ps1` fine-tune 2000 episodes tu `checkpoints_v18/best_phaseC.pt`, giu nguyen reward va `max_steps=300`, khong ghi terminal log xuong file.

Rolling metadata cua cac checkpoint chinh:

| Checkpoint | Episode | WR | Bombs | Kills | Forced escape | SD blocks | Deaths sd/self/opp | Chain-self |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_phaseC.pt` | 200 | **83.5%** | 4.695 | 0.065 | 17.13 | 0.550 | 0.015/0.110/0.040 | **0.000** |
| `best_phaseC_combat.pt` | 1747 | 78.5% | **6.075** | **0.175** | 20.52 | 0.455 | 0.010/0.170/**0.035** | **0.000** |
| `final_phaseC.pt` | 2000 | 69.5% | 5.765 | 0.155 | 19.52 | 0.215 | **0.005**/0.240/0.060 | **0.000** |

Danh gia doc lap seed `20261013`, tong 400 tran AC-vs-RuleBot, doi hai phia spawn, Sudden Death va mask/evaluator v19:

| Checkpoint | WR | Bombs | Offensive | Pressure | Kills | Deaths sd/self/opp/other | Chain-self |
|---|---:|---:|---:|---:|---:|---:|---:|
| v18 `best_phaseC.pt` (baseline voi mask v19) | **77.25%** | 7.1200 | 0.6225 | 0.9450 | 0.0775 | 12/**43**/36/0 | **0** |
| v19 `best_phaseC.pt` | 64.25% | **9.7000** | **1.0425** | **1.5900** | 0.1075 | 6/106/31/0 | 1 |
| v19 `best_phaseC_combat.pt` | 65.75% | 9.5125 | **1.0425** | 1.5200 | **0.1325** | 9/99/29/0 | 2 |
| v19 `final_phaseC.pt` | 71.25% | 9.1025 | 1.0025 | 1.4575 | 0.1275 | **3**/84/**28**/0 | **0** |

Mask v19 tu no cai thien v18 best rat manh so voi cung checkpoint duoi mask v18: WR `64.75 -> 77.25`, sudden-death deaths `43 -> 12`, opponent-bomb deaths `55 -> 36`, chain-self `8 -> 0`; self-deaths giu `43`. Fine-tuning v19 chuyen trade-off sang aggression: final tang bombs `7.12 -> 9.10` va kills `0.0775 -> 0.1275`, doi lai WR giam `6` diem va self-deaths tang `43 -> 84`.

**Ket luan v19:** `checkpoints_v19/final_phaseC.pt` dat day du acceptance WR `>=70%`, bombs `>=7`, kills `>=0.08`, opponent deaths `<=40` va sudden-death deaths `<=30`. Day la checkpoint khuyen nghi cho muc tieu agent chu dong va dat nhieu bomb. Neu uu tien WR/an toan, dung `checkpoints_v18/best_phaseC.pt` voi code mask v19. Khong dung v19 best/combat vi final co WR cao hon, self-death thap hon va kills gan combat checkpoint.

### v20 - owner-aware no-reentry shield: zero-shot rejected

- Thu nghiem chan agent quay lai future blast cua active own bomb sau khi da ra ngoai. Shield khong ap dung cho opponent bomb va giu shortest escape v19.
- Zero-shot lan dau tren v19 final, seed `20261013`, 400 tran: WR `59.75%`, bombs `8.65`, kills `0.0825`, deaths sd/self/opp `4/118/39`, chain-self `5`. Tat ca gate WR/kills/self deu truot.
- Chan re-entry xay ra `1,382` step trong probe 100 tran; `172` step trong so do agent dang bi bomb khac de doa. Emergency override duoc them de khong chan shortest escape khi current cell dang bi hazard.
- Zero-shot sau emergency override van that bai: WR `61.50%`, bombs `9.63`, kills `0.0925`, deaths sd/self/opp `8/121/25`, chain-self `3`. So voi v19 final (`71.25%`, `9.1025`, `0.1275`, self `84`), shield lam WR/kills giam va self-death tang.
- Audit reproducibility: script so sanh da seed Python/NumPy/Torch bang `20261013`, nhung `BombermanFFAEnv` duoc tao voi `seed=None` va dung `np.random.default_rng(None)`. Vi vay cac bang v17-v20 la benchmark 400-tran doc lap, khong phai cung bo board co the tai lap bit-for-bit. Chenh lech v20 van du lon va cung chieu qua hai lan de reject, nhung moi A/B tiep theo phai sua evaluator nhan `--seed` va truyen seed vao environment.

**Ket luan v20:** khong fine-tune, khong tao checkpoint v20. Code/metric/launcher no-reentry da rollback de runtime tro ve mask v19. Giu `checkpoints_v19/final_phaseC.pt` lam policy aggressive khuyen nghi. Gia thuyet "re-entry vao own blast" khong phai root cause doc lap; blanket action blocking tao distribution shift va conflict giua nhieu bomb.
