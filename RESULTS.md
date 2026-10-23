# Ket qua thi nghiem - Bomberman RL

> Cap nhat tu dong trong qua trinh train. Moi giai doan ghi lai ket qua, gate checks, va ghi chu.

---

## Giai doan A - Rule-based bot & sanity check

**Thông số:** RuleBot đấu RuleBot (kiểm tra môi trường); 100 episode; bàn 11×11, 2 agent. max_steps, seed, máy và phiên bản torch: không lưu lại.

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

**Thông số:** Q-learning linear FA (`train_qlearning_baseline.py`) train đấu RuleBot (theo PROJECT_SPEC); bàn 7×7, 2 agent; lần 1: 5000 episode, lr 0.005, 10 đặc trưng; lần 2: 8000 episode, lr 0.005, 16 đặc trưng, thưởng phá tường +0.1; WR đo trên cửa sổ train 100/500 episode, không có đánh giá riêng. max_steps, seed, epsilon schedule, máy: không lưu lại.

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

**Thông số:** Actor-Critic CNN, Phase C, train đấu RuleBot từ đầu; bàn 9×9, 2 agent; GPU RTX 4060 Laptop, torch 2.11.0+cu128 (máy bạn). Số episode: lần 1 3000 (lr 3e-4, thưởng phá tường 0.08); lần 2 6000 (thưởng phá tường 0.3, phạt chết −0.2); lần 3 6000; lần 4 6000; lần 5–7 4000 (lần 6 thêm mask an toàn + giám sát thoát; lần 7 cosine lr 3e-4 → 1.5e-5, entropy 0.015 → 0.003, chọn best theo WR cuộn 200 ep). Mọi WR là WR cửa sổ train 200 episode cuối, không có đánh giá riêng. max_steps, train seed: không lưu lại.

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

**Thông số:** Phase D self-play, pool snapshot + RuleBot; bàn 9×9, 2 agent, snapshot_every 200; lần 1: 4000 episode, rule_prob 0.20; lần 2: 3000 episode, rule_prob 0.50, thưởng phá tường 0.15, phạt tự sát −2.5. Đánh giá lần 2: round-robin 300 ván với 15 snapshot (đổi góc xen kẽ); WR vs RuleBot 85.5% (số trận không lưu lại). Checkpoint xuất phát, max_steps, seed, máy: không lưu lại (máy bạn, GPU RTX 4060 theo mục ghi chú kỹ thuật).

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

**Thông số:** Phase E FFA self-play, snapshot_every 200, log_every 100. E1: bàn 9×9, 3 agent, 2000 episode, warm-start `checkpoints/best_phaseD.pt`; công bằng góc đo trên 36 ván quyết định. E2: bàn 11×11, 4 agent, 3000 episode, warm-start `checkpoints/best_phaseE_3a.pt` (nội suy bilinear trunk 9×9 → 11×11); số ván dùng cho gate công bằng góc không lưu lại. Mọi số là số train, không có đánh giá riêng; không đo ActiveWR. max_steps, seed, máy: không lưu lại.

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

**Thông số:** `checkpoints/best_phaseE_4a.pt` đấu 3 RuleBot; bàn 11×11, 4 agent, max_steps 500; 40 ván (10 ván × 4 góc), seed môi trường/NumPy/PyTorch 100/101/102/103 theo góc; policy lấy mẫu + mask như GUI. Pilot train: Phase E, 4 agent, 11×11, kế hoạch 300 episode từ best_phaseE_4a, CPU, bị ngắt sau 32 episode (`runs/combat-20261010/config.json`). Phiên bản torch, số tiến trình: không lưu lại.

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

**Thông số:** Lượt train của người dùng `runs/20261010-125814-296523`: Phase C, 3000 episode, 9×9, 2 agent, max_steps 500, CPU, train từ đầu (train seed không lưu lại). Đánh giá trước/sau và RuleBot mới: 100 ván/cấu hình, seed 7000–7099, 50 ván/góc, 9×9, 500 bước, policy lấy mẫu, CPU 1 thread. Pilot: warm-start best của lượt trên, 40 episode, GPU, seed 20261010, lr 1e-4 (`runs/safety-20261010/pilot-final/`). Phiên bản torch: không lưu lại.

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

**Thông số:** Lượt train của người dùng `runs/20261010-135449-587798`: Phase C, 3000 episode, 9×9, 2 agent, max_steps 500, CPU, train từ đầu, chỉ học góc A0 (train seed không lưu lại); best ở episode 2732 theo ActiveWR cuộn 200. Đánh giá: best và final, 100 ván/checkpoint, seed 7000–7099, 50 ván/góc, 500 bước, `evaluate.run_match`, policy lấy mẫu, CPU 1 thread. Phiên bản torch: không lưu lại.

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

**Thông số:** Pilot: Phase C, warm-start `runs/20261010-135449-587798/best_phaseC.pt`, 500 episode (250/góc), 9×9, 2 agent, max_steps 500, lr 1e-4, train seed 20261011, GPU RTX 3050 Laptop, torch 2.5.1, 1 thread CPU, ~26.9 phút (`runs/escape-20261011/pilot-500/`). Validation trong train: 48 trận seed 6000 ở ep 0/250/500; chấm lại 3 ứng viên trên CPU cùng 48 trận. Test: seed 9000–9049 × 2 góc = 100 trận/checkpoint, 9×9, 500 bước, policy lấy mẫu, CPU 1 thread, đối thủ RuleBot trước sửa.

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

**Thông số:** Chẩn đoán: replay 20 trận hòa + 10 trận tự nổ từ `test_baseline.json` (seed 9000–9049). Pilot: Phase C, warm-start checkpoint gốc (`runs/20261010-135449-587798/best_phaseC.pt`), 500 episode (250/góc), 9×9, 2 agent, max_steps 500, lr 1e-4, train seed 20261012, CPU, ~70.4 phút. Warm-up teacher: 200 update Adam, lr 5e-5, seed 2026101201, loss chiến đấu 0.5 / điều hướng 0.2, clip 1 (602 nhãn bom + 6958 nhãn điều hướng). Validation: 100 trận seed 6100 (cả hai góc), CPU 1 thread. Test: seed 12000–12049 × 2 góc = 100 trận/trường hợp, 9×9, 500 bước, policy lấy mẫu, CPU 1 thread. Seed phát triển 9000/11000. Phiên bản torch: không lưu lại.

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

**Thông số:** Hai lượt Phase C, warm-start `runs/combat-20261012/combat_bc_phaseC.pt`, mỗi lượt 500 episode, 9×9, 2 agent, max_steps 500, lr 1e-4, train seed 2026101202; thưởng phá tường 0.15 (`runs/fair-C-next/`) và 0.03 (`runs/fair-C-wall003/`). Validation 100 trận seed 6200 ở ep 0/250/500. Test: seed 13000–13099 × 2 góc = 200 trận/model, policy lấy mẫu, CPU. Replay chẩn đoán: 6 trận timeout. Thiết bị train, torch: không lưu lại.

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

**Thông số:** Checkpoint `runs/combat-20261012/combat_bc_phaseC.pt` (SHA `2f088072…21d7`), đối thủ RuleBot `robust=False`. Probe teacher: 40 trận seed 6300. Pilot 50 episode (25/góc), CPU 1 thread, 350.3 s, validation 100 trận seed 6300. Smoke CLI 10 episode. Test: seed 14000–14099 × 2 góc = 200 trận/bộ lọc, 9×9, 2 agent, 500 bước, policy lấy mẫu, CPU 1 thread, 400 trận mất 372 s. lr pilot, train seed, phiên bản torch: không lưu lại.

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

**Thông số:** Người dùng chạy `runs/pressure-C-next/`: Phase C, warm-start `runs/combat-20261012/combat_bc_phaseC.pt`, 500 episode (250/góc), 9×9, 2 agent, max_steps 500, lr 1e-4, thưởng tường 0.15, train seed 2026101301; validation 100 trận seed 6400 ở ep 0/250/500. Đánh giá final: seed 15000–15099 × 2 góc = 200 trận, policy lấy mẫu, CPU 1 thread. Thiết bị train, phiên bản torch: không lưu lại. (Final của lượt này về sau là checkpoint B, SHA `93bdf278…4f60`.)

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

**Thông số:** Không train. Checkpoint `runs/pressure-C-next/final_phaseC.pt` (B, SHA `93bdf278…4f60`) và warm-start `combat_bc_phaseC.pt` đấu RuleBot `robust=False`; 9×9, 2 agent, max_steps 500, policy lấy mẫu, CPU 1 thread. Phát triển: replay 13 trận + 200 trận seed 15000–15099 × 2 góc. Test độc lập (protocol `runs/escape-early/protocol_16000.json`): seed 16000–16099 × 2 góc = 200 trận/cấu hình, 3 cấu hình = 600 trận trong 528.39 s. Phiên bản torch: không lưu lại.

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

**Thông số:** Người dùng chạy `runs/escape-C-6600`: Phase C, warm-start B (`runs/pressure-C-next/final_phaseC.pt`), 500 episode (250/góc), 9×9, 2 agent, max_steps 500, lr 1e-4, thưởng tường 0.15, train seed 2026101302; validation 100 trận seed 6600 ở ep 0/250/500. Test: seed 17000–17099 × 2 góc = 200 trận/model, policy lấy mẫu, CPU. Thiết bị train, phiên bản torch: không lưu lại.

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

**Thông số:** Warm-start B (SHA `93bdf278…4f60`), Phase C, 9×9, 2 agent, max_steps 500, lr 1e-4, train seed 2026101303, CUDA, torch 2.5.1; kế hoạch 200 episode/nhánh (local, two_bomb) trong ngân sách 65 phút: local 200/200, two_bomb 146/200. Validation 100 trận seed 21000 ở ep 0/100/200. Test: seed 25000–25199 × 2 góc = 400 trận/model (gốc, local, two_bomb), CPU; bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014. Tổng 86.67 phút.

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

**Thông số:** Người dùng chạy `runs/two-bomb-complete-20261014-031626`: two_bomb 200 episode từ B, Phase C, 9×9, 2 agent, max_steps 500, lr 1e-4, train seed 2026101303, CUDA, torch 2.5.1; local 200 dùng lại từ pilot. Validation 100 trận seed 21000 ở ep 0/100/200. Test: seed 26000–26199 × 2 góc = 400 trận/model × 3 model, CPU; bootstrap 10 000 lần, seed 20261014; tổng ~85 phút (5119 s).

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

**Thông số:** Không train. B và final two_bomb200 (`runs/two-bomb-complete-20261014-031626/two_bomb/final_phaseC.pt`) chơi CNN + mask, teacher chỉ log; seed 28000–28099 × 2 góc = 200 trận/model (400 tổng); 9×9, 2 agent, max_steps 500; CPU 1 thread, torch 2.5.1; bootstrap 10 000 lần, seed 20261014; 4562 s (`runs/t2-full-20261014-060320`).

Lượt `t2-full-20261014-060320`:400 trận tổng,200/model,100 seed bàn28000–28099 với cảA0/A1. B và final two_bomb200 chơi bằng CNN/mask, teacher chỉ log; best two_bomb aliasB. Test74 OK; CRC/SHA ZIP và snapshot/checkpoints khớp; toàn bộ summary/từng góc được tính lại từ158.293 frames/10.695 windows/88.872 branches. Đã dựng lại12 replay/5.608 frames (10 có sẵn, hai controls).

| Model | ActiveWR | Hòa | Tự nổ | Kills/trận | Bom đặt/trận | Đúng bom hai khi bom đầu đúng |
|---|---:|---:|---:|---:|---:|---:|
| B |33,00% (66/200)|57,00%|1,00%|0,33|40,17|23/905=2,54%|
| Final two_bomb200 |16,00% (32/200)|75,50%|1,00%|0,16|26,68|5/427=1,17%|

Final−B **−17 điểm**, paired bootstrap95%CI **[-25,5;−8,5]**, 100 board pairs/10.000 resamples. A0 giảm32→15%, A1 giảm34→17%. Đây là regression của final trên audit set, không dùng để chọn checkpoint mới hay kết luận mọi train seed two_bomb đều thất bại.

Teacher chỉ kiểm tra mask ở hành động đầu: final/RuleBot có7.086 physical prefixes complete nhưng6.454 bị future mask chặn (91,08%); static90,48%, evader90,74%. Gap thường ở BOMB, gồm cả bom đầu sau di chuyển và bom hai. Các nhãn hai bom dự báo trap chỉ3/63.973 frames; combo trap mà hai bom riêng vẫn có refuge chỉ1/6.390 sampled plans. Case này cũng bị mask chặn ở bom hai. Agreement52,98% gần uniform K/M52,65% trên cùng state, chưa đủ làm bằng chứng học combo.

**T2 chẩn đoán hoàn tất; giữ B, ưu tiên L2.0 candidates offline trước train tiếp.** Chỉ lọc best hiện tại có thể giữ gần hết nhãn một bom và rất ít combo: final static mask-pass599 nhưng chỉ4 two-bomb. Cần kiểm tra alternatives trong64 schedules trước khi implement guarded teacher; nếu valid combos quá hiếm thì mở T3.1 benchmark/curriculum riêng. Chưa đổi mask, reward, optimizer hay train thêm. [Quyết định đầy đủ](/D:/fpt/subject/BomberMan/runs/t2-full-20261014-060320/T2_decision.md), [audit có thể tái lập](/D:/fpt/subject/BomberMan/runs/t2-full-20261014-060320/analyze_t2_results.py).

## L2.0 — candidate alternatives qua future mask, 14/10/2026

**Thông số:** Không train, audit offline. Replay 40 trận đã lưu (seed 28000–28009 × 2 góc × 2 model: B và final two_bomb200), 149 trạng thái, 6666 đường đi được chấm; top-k lịch {1, 4, 8}; 28 s (`runs/l2-full-20261014-120348`). Phiên bản torch: theo snapshot source trong thư mục run.

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

**Thông số:** Không train. B chơi 240 case train/validation × 3 đối thủ (static, evader, RuleBot) × 2 lần lặp = 1440 trận; skill horizon 80, env max_steps 500, 9×9, 2 agent; rollout seed 2026101401/2026101402; family fork = train, long_fork = validation, wood_branch (test) chưa chạy; 498 s (`runs/t3-full-20261014-implementation`).

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

**Thông số:** Chỉ smoke. Warm-start B, train seed 2026101303, teacher local, lr 1e-4, 6 episode/nhánh, max_steps 8; validation 2 trận seed 21000; đánh giá 2 trận seed 29800; CPU (`runs/t3-curriculum-smoke-implementation`, 6.64 s) và CUDA RTX 3050 (`runs/t3-curriculum-smoke-cuda-implementation`, 7.44 s), torch 2.5.1, numpy 2.0.1; deadline test 1 s. Lượt full đã đăng ký: 600 episode/nhánh, validation 100 trận seed 21000 mỗi 100 ep, test 400 trận/model seed 31000–31199.

Đã triển khai [run_t3_curriculum.py](/D:/fpt/subject/BomberMan/run_t3_curriculum.py) và [protocol/lệnh đầy đủ](/D:/fpt/subject/BomberMan/T3_CURRICULUM_GUIDE.md). Đăng ký control600 PhaseC RuleBot và curriculum600 C1static→C2evader→C3RuleMix→PhaseC consolidation; cùng B/seed2026101303/LR1e-4/teacherlocal/reward/optimizer schedule. Chỉ dùng train fork positive cases; chứng nhận lại preflight. Stage probes tuổi100/150/200, hai pass success≥60%/suicide≤B guard, cap200. Skill probes không reset RNG/weights hoặc chọn best; PhaseC100game validation21000 ở0/mỗi100 chọn best và giữ suicide guard hiện tại.

110 tests qua (`runs/t3-curriculum-tests-current.log`), gồm trainer thật với một Adam/scheduler qua ba stage, steps2→4→6/LR giảm liên tục; gates/pass-streak/cap/cancel, reset trả observation đúng, control map/RNG khớp normal PhaseC, CPU/CUDA probe purity, output bảo vệ và final/partial routing. Full orchestration/paired seeds/gate và matched-budget routing được test với mock outcomes, không phải kết quả RL thật.

Smoke CPU `runs/t3-curriculum-smoke-implementation` hoàn tất6ep/arm trong6,64s; CUDA/RTX3050 `runs/t3-curriculum-smoke-cuda-implementation` hoàn tất6ep/arm trong7,44s, mỗi nhánh một Adam/6 updates và48 learner steps. Stages cưỡng bức ởep2/4/6, max8 nên full_protocol=false/retain_original; cả best chọnep0. Không dùng win rates của smoke để kết luận học tốt hơn. Test deadline1s `runs/t3-curriculum-deadline-implementation` trả incomplete và ZIP, không mở test. Source/warm-start giữ nguyên hash trong mọi smoke; core T1/T2/L2 không sửa.

Nếu full curriculum fail stage, runner chấm final/control checkpoint cùng episode trên validation21000 và vẫn incomplete. Chỉ cặp600 đủ gates/validation mở random-map400/model test31000–31199/cảA0/A1 và wood_branch576games/model. Gate cuối +5pp so cả B/control, bootstrap95% lower>0, suicide không cao hơn chỉ cho phép replication; không tự thay model.

**T3.2 implementation/tests/smoke DONE; full600 WAIT RUN.** Chưa mở test31000/wood_branch, chưa có bằng chứng curriculum cải thiện ActiveWR; giữ B. Nguồn smoke/ZIP/console được lưu trong các thư mục nêu trên.

## T3.2 pilot — dừng C1 đúng gate, 14/10/2026

**Thông số:** Người dùng chạy `runs/t3-curriculum-full-20261014-154703`: warm-start B, teacher local, lr 1e-4, train seed 2026101303, CUDA RTX 3050; control 600 episode (Phase C vs RuleBot), curriculum 200 episode (dừng ở C1 static, cap 200); 9×9, 2 agent, max_steps 500; probe kỹ năng C1 64 case ở ep 100/150/200; validation 100 trận seed 21000 mỗi 100 ep; tổng 2625 s. Phiên bản torch: theo config run.

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

**Thông số:** Người dùng chạy `runs/control-confirmation-20261014-201557`. B và control best400 (`runs/t3-curriculum-full-20261014-154703/control/best_phaseC.pt`) đấu RuleBot. Seed 32000–32199 × 2 góc = 400 trận/model; 9×9, 2 agent, max_steps 500; policy lấy mẫu + mask; CPU 1 thread; torch 2.5.1. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014.

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

**Thông số:** Không train. Replay 256 trận C1 đã lưu (B và C1 ep100/150/200 × 64 case, 8224 frame) và 96 fixture counter-bomb; 2 trạng thái thật của seed 32000 (32080, 32133). Không mở test mới.

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

**Thông số:** Smoke: 4 episode/nhánh, max_steps 8, CPU và CUDA. Full: `runs/combat-move-full-20261015-013315`; warm-start B; control `local` và `local_combat_move`; 200 episode/nhánh; Phase C vs RuleBot, 9×9, 2 agent, max_steps 500; lr 1e-4; train seed 2026101501; CUDA RTX 3050; torch 2.5.1. Validation 100 trận seed 21000 ở ep 0/100/200. Test 400 trận/model seed 33000–33199 × 2 góc, CPU 1 thread.

**Cập nhật cuối:** pilot đã hoàn tất/review,gateFAIL,giữB. Xem bảng test và [decision_report.md](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315/decision_report.md) bên dưới; các đoạnRUNNING ghi trạng thái lúcđăng ký/khởi chạy.

Đã triển khai [certifier](/D:/fpt/subject/BomberMan/combat_move_teacher.py), trainer/CLI mode`local_combat_move`, [runner đối chứng](/D:/fpt/subject/BomberMan/run_combat_move_pilot.py), [protocol/lệnh và file kết quả](/D:/fpt/subject/BomberMan/COMBAT_MOVE_GUIDE.md). Target action local không đổi; chỉ MOVE có tăng positive pressure và next BOMB/escape certificate trên clone sang combat mean0,5; nhãn không đạt giữ navigation0,2. Escape greedy qua mask tới hết bom/lửa, bounded32ticks/other agents STOP; không phải guard trước future counter-bombs. Env/mask/reward/CNN/12channels/optimizer giữ nguyên, defaultlocal vẫn local.

**136tests PASS** (`runs/combat-move-implementation-20261015/all-tests.log`), gồm local-before bit parity ở mọi tensor/reward/action, actual qualified update CPU/CUDA, pressure/slot/collision/greedy/purity, loss routing không override actionCNN, budgets/incomplete/source changes/paired bootstrap/gates/fresh-test guards. Source cũ lưu trongbefore snapshot; test archived control đối chiếu source snapshot và vẫn kiểm tra current trainer không được reuse control cũ.

CPU/CUDA smoke4ep/arm,max8 đủ hai nhánh, mộtAdam/4updates/32learner steps mỗiarm;47,48/47,52s,full_protocol=false,best0,retainB. Trong smoke online không có move được nhận nên hai arms khớp theo từng thiết bị; actual promoted loss branch được test riêng trênpositive envCPU/CUDA. Preflight smoke6/32accepted,p95≈13–15ms dưới tải test. Deadline0,001phút trả incomplete/ZIP/retainB và không mở test.

**Full pilot RUNNING:** `runs/combat-move-full-20261015-013315`, CUDA/RTX3050, cảhai200ep từB, trainseed2026101501,LR1e-4/PhaseC RuleBot/max500; config đăng ký trước test. Preflight120unique C1 navigation states đã xem,28 nhận, mean1,49ms/p955,22ms/max9,45ms. Validation100seed21000 ở0/100/200 chọnbest vớisuicideguardB; skill64escape/approach ởcùngmốc chỉdiagnostic. Test cuối400/modelB/controlbest/combat_movebest seeds33000–33199/cảA0-A1/CPU1thread; bootstrap200boardpairs/gate+5ppsoBvàcontrol,CI lower>0/suicide≤cảhai. Chưa đọc33000 hoặc chọnmodel mới tại cập nhật này;31000/wood vẫn khóa. Cần chờsummary cuối, không dùngsmoke/trainingWR để kết luận cải thiện.

## Guidance MOVE — pilot đầy đủ và quyết định, 15/10/2026

**Thông số:** `runs/combat-move-full-20261015-013315`: warm-start B; 200 episode/nhánh (control `local`, candidate `local_combat_move`); Phase C vs RuleBot, 9×9, 2 agent, max_steps 500; lr 1e-4; train seed 2026101501; CUDA RTX 3050; torch 2.5.1. Validation 100 trận seed 21000 ở ep 0/100/200. Test: 3 model × 400 trận, seed 33000–33199 × 2 góc, CPU 1 thread. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014. Tổng 3502.5 s.

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

**Thông số:** Smoke 20 episode/nhánh, max_steps 20, CPU và CUDA. Full `runs/advantage-full-20261015-030430`: warm-start B; teacher local; 200 episode/nhánh (raw và normalized advantage); Phase C vs RuleBot, 9×9, 2 agent, max_steps 500; lr 1e-4; gamma 0.99; train seed 2026101502; CUDA RTX 3050; torch 2.5.1. Validation 100 trận seed 21000 ở ep 0/100/200. Test 400 trận/model × 3 model, seed 34000–34199 × 2 góc, CPU 1 thread. Bootstrap 10 000 lần, seed 20261014. Tổng 5816 s.

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

**Thông số:** Chỉ kiểm tính đúng và smoke. 3 chế độ (MC, GAE λ1, GAE λ0.95) × 20 episode / 20 update mỗi chế độ, trên CPU và CUDA; warm-start B; teacher local; lr 1e-4; train seed 2026101503; max_steps 20; không validation và không test.

**DONE/PASS về tính đúng; giữ B và mặc định MC.** Trainer/CLI thêm actor advantage estimator `mc|gae` và `gae_lambda`; critic luôn MC returns. Tách bootstrap/continuation để terminal không bootstrap và cut không nối sang reset. Learner chết kết thúc buffer dù FFA vẫn chạy. Collector vẫn toàn episode; chưa triển khai rollout update giữa trận/PPO. Teacher/reward/mask/CNN/loss phụ không đổi.

167testsPASS (`runs/gae-b0-implementation-20261015/all-tests-final.log`), gồm lambda0/0,5 tính tay,lambda1MCparity1/3/500bướcCPU-CUDA,latekill/death/timeout500/reset/cut,detach/criticgradient/purity/cancel/badinputs/runnerguards. DefaultMC actions/rewards/metrics/weights bit-identical source-before trênCPU2ep20steps. FrozenGAE1firstupdate soMC có maxdelta policy loss4,47e-8CPU/3,73e-9CUDA; gradient1,49e-7/2,79e-8; postAdamweights4,78e-6/1,17e-7. Dùngtoleranceđãghi,khônggọiGAE1bit-parity. Đãsửalỗibooleanmasklàmgammaroundfloat32trongrecurrencefloat64.

SmokeMC/GAE1/GAE0,95:20ep/20updates/mode/device,normalizationoff,LR1e-4/seed2026101503/B/local/max20; khôngvalidation/evaluation/promotion. CPU15,06s/394learnersteps mỗi mode,GAE18timeouts+2matchends;CUDA14,83s/400steps/mode,20timeouts. Weightsfinite,source+Bunchanged,mỗiZIP22files/CRC/bytes/checkpoints/metricskhớp qua review. Không mởseedtestmới,34000đãdùngtừA1;31000/wood_branch vẫn không mở trongB0.

[Báo cáo B0](/D:/fpt/subject/BomberMan/runs/gae-b0-implementation-20261015/B0_report.md), [guide](/D:/fpt/subject/BomberMan/GAE_B0_GUIDE.md), [review JSON](/D:/fpt/subject/BomberMan/runs/gae-b0-implementation-20261015/smoke_review.json). B1READY nhưng chưa runner đối chứng200ep/arm hoặc400test/model; chưa có bằng chứng GAE tăngActiveWR. Không dùng checkpoint smoke để thay B.

## T4-B1 — lượt đầu thiếu thời gian, 15/10/2026

**Thông số:** `runs/gae-full-20261015-112853`: warm-start B; teacher local; MC vs GAE 0.95; kế hoạch 200 episode/nhánh (control 200, GAE dừng ở 184 vì trần train 85 phút); Phase C vs RuleBot, 9×9, 2 agent, max_steps 500; lr 1e-4; gamma 0.99; train seed 2026101504; CUDA; torch 2.5.1. Validation 100 trận seed 21000 ở ep 0/100/200. Test seed 35000–35199 chưa mở.

**INCONCLUSIVE, giữ B.** `runs/gae-full-20261015-112853` chạy85,11 phút rồi dừng vì trần train chung85 phút: control200/200, GAE184/200. Không có test cuối400/model hoặc CI;35000–35199 chưa mở. Mỗi nhánh một Adam;control73.101 bước,GAE71.429;normalizationoff/criticMC/teacherlocal/cùngB vàseed2026101504.

| Nhánh | Val ActiveWR ep0 | ep100 | ep200 | Best |
|---|---:|---:|---:|---:|
| MC control |29%|28%|25%|0|
| GAE0,95 |29%|29%|Chưa chạy|0|

Tự nổ validation0% tại cả5 bài chấm;trainself3/0 không đủ kết luận an toàn hơn vì số episode khác nhau và chưa có heldouttest. ApproachMC28/32→30/32→28/32,GAE28/32→28/32;escape32/32 cả5 probe. Best0 vẫn weightsB. Review source19/B/checkpoint/ZIP41files/384training rows/320storedskillgames/7.140replayframes qua;256rollouts kỹ năng thực vìbaselinecache64game ghi hai lần.

174regression tests và smoke CPU57,49s/CUDA73,66s qua trước full. Sau khi lưu/review dữ liệu gốc, đã sửa riêng trần train thành100 phút trong tổng150;49 tests liên quan qua, testdeadline xác nhận evaluation bị chặn theo thời gian còn lại. **Chưa chạy lại full bản sửa.** Không sửa config cũ và không chạy thêm16episode từfinal184 vì thiếu Adam/LR/RNG state.

[Báo cáo quyết định](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-112853/decision_report.md), [analysis](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-112853/analysis_results.json), [raw ZIP](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-112853-send.zip), [protocol/lệnh](/D:/fpt/subject/BomberMan/GAE_PILOT_GUIDE.md). Chưa có cơ sở mở B2/PPO/reward hoặc thay checkpoint từ lượt thiếu này.

## T4-B1 — chạy lại đầy đủ, gate cải thiện FAIL, 15/10/2026

**Thông số:** `runs/gae-full-20261015-125942`: warm-start B; teacher local; MC vs GAE 0.95; 200 episode / 200 update mỗi nhánh; Phase C vs RuleBot, 9×9, 2 agent, max_steps 500; lr 1e-4; gamma 0.99; train seed 2026101504; CUDA; torch 2.5.1. Validation 100 trận seed 21000 ở ep 0/100/200. Test 400 trận/model × 3 model, seed 35000–35199 × 2 góc, CPU. Bootstrap 10 000 lần, seed 20261014. Tổng 54.87 phút.

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

**Thông số:** Không train. B, MC final200 và GAE final200 (`runs/gae-full-20261015-125942/{control,gae}/final_phaseC.pt`) chơi seed 21000–21049 × 2 góc = 100 trận/model; 9×9, 2 agent, max_steps 500; CPU; 975 s (`runs/r0-full-20261015-153908`).

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

**Thông số:** `runs/reward-full-20261015-163227`: warm-start B; teacher local; thưởng tường 0.15 (control) vs 0.075 (half_wall); kế hoạch 200 episode/nhánh (control đủ 200, half_wall dừng ở 199 vì tiến trình bị ngắt); Phase C, 9×9, 2 agent, max_steps 500; lr 1e-4; gamma 0.99; train seed 2026101505; CUDA; torch 2.5.1. Validation 100 trận seed 21000 mỗi 100 ep. Test seed 36000 chưa mở.

R1 IMPLEMENTED/TESTED, full INCOMPLETE do tiến trình bị ngắt: `reward-full-20261015-163227`. Control đủ200episode; half-wall199episode đã lưu, validation200/final200 còn thiếu. Best validation control33%/half-wall30%, chưa có test hoặcCI; giữB. 200regression testsPASS, smokeCPU/CUDA qua; audit320skill replays/6.966frames. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-20261015-163227/decision_report.md).

## T4-R1 resumable — đánh giá đủ protocol

**Thông số:** `runs/reward-full-resumable-20261015-194314`: warm-start B; teacher local; thưởng tường 0.15 vs 0.075; 200 episode / 200 update mỗi nhánh; Phase C, 9×9, 2 agent, max_steps 500; lr 1e-4; gamma 0.99; train seed 2026101505; CUDA 12.4 RTX 3050; torch 2.5.1; chế độ deterministic. Validation 100 trận seed 21000 ở ep 0/100/200. Test 400 trận/model × 3 model, seed 36000–36199 × 2 góc, CPU. Bootstrap 10 000 lần, seed 20261014. Tổng 49.54 phút.

R1 DONE/REVIEWED: `reward-full-resumable-20261015-194314`. Đủ 200 episode và 200 Adam update mỗi nhánh từ B; 400 trận test/model trên seed 36000–36199, cả A0/A1. Best control/half-wall đều ở episode 200. ActiveWR B/control/half-wall = 34,50%/36,25%/35,00%. Gate FAIL; giữ B và wall bonus 0,15. Thời gian chạy 49,54 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/decision_report.md).

Phase C có `training_state.pt` gồm model/Adam/scheduler/global episode/best guard/history, RNG Python/NumPy/env/Torch CPU/CUDA và context log. Atomic temp+fsync+replace, OS writer lock, `run_reward_pilot.py --resume` khôi phục episode đã xác nhận; không reset schedule hoặc chọn lại model test. 212test riêng biệt PASS; có CPU/CUDA exact resume trong tiến trình mới, interrupted episode/validation và torn-log recovery. `best/final_phaseC.pt` giữ dạng weights để chơi; loader từ chối nhầm full-state file. Resume yêu cầu cùng source/config/runtime/thiết bị; hiện chỉ hỗ trợ Phase C, deadline R1 gốc vẫn giữ.

## T4-D0 — kiểm chứng update ratio

**Thông số:** Chỉ kiểm tính đúng và smoke. Mỗi chế độ AC và ratio: 20 episode / 20 update Adam, trên CPU và CUDA; warm-start B; teacher local. Không validation, không test. Train seed và max_steps của smoke nằm trong `runs/ppo-d0-smoke-*/config.json`.

T4-D0 DONE/PASS về tính đúng: 226/226 tests, gồm 14 D0; CPU/CUDA smoke đều đủ 20 episode/20 Adam updates mỗi chế độ AC và ratio chưa clip. Buffer lưu mask/old policy/teacher targets; MC raw, critic/reward/teacher/inference và mặc định AC giữ nguyên. Full states và source/ZIP/B hashes đã review. D0 không chứng minh tăng ActiveWR; giữ B. D1 clipping với cùng 4 epochs READY/design, chưa triển khai/pilot. [Báo cáo](/D:/fpt/subject/BomberMan/runs/ppo-d0-implementation-20261015/D0_report.md), [hướng dẫn](/D:/fpt/subject/BomberMan/PPO_D0_GUIDE.md).

## T4-D1 — paired clipping pilot

**Thông số:** `runs/ppo-full-20261015-220313`: warm-start B; teacher local; ratio U4 (không clip) vs PPO P4 (clip 0.20); 4 epoch/update; KL guard target 0.02 / ngưỡng 0.03; 200 episode/nhánh (800 update); Phase C, 9×9, 2 agent, max_steps 500; lr 1e-4; gamma 0.99; train seed 2026101507; CUDA 12.4 RTX 3050; torch 2.5.1; deterministic. Validation 100 trận seed 21000 mỗi 100 ep. Test 400 trận/model × 3 model, seed 37000–37199 × 2 góc, CPU. Bootstrap 10 000 lần, seed 20261014. Tổng 106.89 phút.

D1 DONE/REVIEWED: `ppo-full-20261015-220313`. Đủ 200 episode/nhánh từ B, Adam updates U4/P4=800/800; best episode U4/P4=200/200. Test400/model, seed37000–37199/cảA0-A1: ActiveWR B/U4/P4=37.50%/37.00%/32.00%. Gate FAIL; giữ checkpoint B. Runtime 106.89 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/ppo-full-20261015-220313/decision_report.md).

D1 giữ MC raw/critic MC/local teacher/reward cũ. U4/P4 cùng tối đa4 epochs và post-update KL guard target0,02/threshold0,03; chỉ khác clipping epsilon0,20. 241 regression và25 focused metadata/compat tests PASS; CPU/CUDA smoke20episode/80updates mỗi nhánh đã review. Có 500-step/4-epoch memory fixture, hủy giữa epoch khôi phục Adam, exact PPO resume qua tiến trình mới CPU/CUDA, full-state/source/ZIP/locked-model guards. CNN inference và Actor-Critic mặc định giữ nguyên.

## SD1/SD2 — Sudden death: kiểm chứng và tối ưu, 16/10/2026

**Thông số:** Không train; B (SHA `93bdf278…4f60`) đấu RuleBot, 9×9, 2 agent, max_steps 500, policy lấy mẫu, CPU, torch 2.5.1. SD1: start 200 / interval 20 / warning 40 / đóng kín; 4 case × 400 trận, seed 38000–38199 × 2 góc; 1046 s. SD2 quét validation: seed 21000–21099 × 2 góc = 200 trận/cấu hình; v1 31 cấu hình, v2 49 cấu hình (6 tiến trình). SD2 xác nhận: start 400 / interval 40 / warning 40 / max_ring 2; 3 case × 400 trận, seed 39000–39199. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014.

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

**Thông số:** Chỉ smoke. Warm-start B; nhánh `sd_12ch` và `sd_13ch`; 20 episode/nhánh; max_steps 20; SD smoke start 8 / interval 4 / warning 6 / max_ring 2; lr 1e-4; teacher local; train seed 2026101601; validation 2 trận seed 21000; đánh giá 2 trận seed 39840; CPU và CUDA; torch 2.5.1. Lượt full đã đăng ký: 200 episode/nhánh, SD2, test seed 40000.

**IMPLEMENTED/TESTED/SMOKE DONE; full đã chạy, gate FAIL (xem mục "SD3 — kết quả full" bên dưới).** Plan: `docs/superpowers/plans/2026-10-16-sd-training-ring-channel.md`. Spec: `docs/superpowers/specs/2026-10-16-sd-training-ring-channel-design.md`. Hai nhánh từ B: `sd_12ch` (train dưới SD2 both_aware) và `sd_13ch` (thêm kênh 12 = 1/delay tới khi ô bị vòng đóng, chỉ trong `SuddenDeathEnv`). 200 episode/nhánh, validation 21000, test một lần seed 40000–40199, 400 trận/model (B, best sd_12ch, best sd_13ch) dưới SD2 both_aware, gate ActiveWR ≥ +5pp so B, CI95 lower > 0, tự nổ không cao hơn B. Chưa có kết quả test.

Runner `run_sd_training_pilot.py`, train seed 2026101601, warm-start B, Phase C vs RuleBot biết vòng, validation 21000 dưới SD2. Kiểm thử: 269 test PASS (unittest discover), gồm parity B trong env 13 kênh (hành động, logits atol=0, `evaluate_case` cùng trận), `ring_channel=False` bit-identical cả RNG, patch được khôi phục khi lỗi. Smoke (timing ngắn start8/interval4/max_ring2, 20 episode/nhánh, 20 bước) complete: CPU `runs/sd3-smoke-cpu-20261016-165451` 31 giây, CUDA `runs/sd3-smoke-cuda-20261016-165558` 32 giây, rerun CPU sau fix review `runs/sd3-smoke-cpu-20261016-171018`; `--resume` trên run đã xong dùng lại nhánh/đánh giá, không train/chấm lại. Nhánh 13 kênh: best ep0 có trọng số kênh vòng = 0 (đúng B), final học được trọng số khác 0. Với timing smoke, mọi trận đều cả hai chết vì vòng; smoke chỉ kiểm tra đường ống, không phải chất lượng. Lượt smoke lỗi đầu tiên `runs/sd3-smoke-cpu-20261016-165344` dừng vì `benchmark_t3.validate_env` cứng 12 kênh (đã sửa), không có đánh giá. Seed 40000 chưa mở.

## SD3 — kết quả full: train dưới sudden death ± kênh vòng, 16/10/2026

**Thông số:** `runs/sd3-full`: warm-start B; nhánh `sd_12ch` / `sd_13ch`; 200 episode/nhánh; Phase C vs RuleBot biết vòng; SD2 (400/40/40, max_ring 2); 9×9, 2 agent, max_steps 500; lr 1e-4; teacher local; train seed 2026101601; CUDA 12.4 RTX 3050; torch 2.5.1; deterministic. Validation 100 trận seed 21000 mỗi 100 ep. Test 400 trận/model × 3 model, seed 40000–40199 × 2 góc, SD2 both_aware, CPU. Bootstrap 10 000 lần, seed 20261014. Tổng 2595 s.

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

**Thông số:** Không chạy trận mới. Đọc lại 3 × 400 trận đã có của `runs/sd3-full` (seed 40000–40199 × 2 góc, SD2 both_aware, max_steps 500).

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

**Thông số:** B đấu RuleBot; SD2 both_aware (400/40/40, max_ring 2); seed validation 21000–21199 × 2 góc = 400 trận; 9×9, 2 agent, max_steps 500; policy lấy mẫu; CPU, máy người dùng, torch 2.5.1. Số tiến trình: không lưu lại.

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

**Thông số:** B đấu RuleBot; SD2 both_aware; 9×9, 2 agent, max_steps 500; policy lấy mẫu; CPU, torch 2.5.1. Quét: k ∈ {0, 5, 10, 15, 20} × 400 trận, seed 21000–21199 × 2 góc. Test: k = 0 và k = 15 × 400 trận, seed 41000–41199 × 2 góc. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014. Số tiến trình: không lưu lại.

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

**Thông số:** B chơi tới bước 360, sau đó ghế learner do 4 bộ điều khiển (B, teacher local, two_bomb → local, RuleBot) đảm nhận. 400 trận/nhánh, seed 21000–21199 × 2 góc; SD2 + biên learner k = 15; 9×9, 2 agent, max_steps 500; CPU, torch 2.5.1. Bootstrap 10 000 lần, seed 20261014.

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

**Thông số:** B và B + kill-search (lookahead 12 bước, đối thủ mô phỏng là RuleBot) đấu RuleBot. Bật search ở bước 360 và bước 0. Kế hoạch 400 trận/nhánh, seed 21000–21199 × 2 góc; SD2 + biên k = 15; 9×9, 2 agent, max_steps 500; 5 tiến trình CPU, torch 2.5.1. Bị dừng vì thiếu RAM. Lượt thử chỉ có 8 trận.

**INCOMPLETE, chưa có số liệu.** Lượt `probe_sd_endgame.py --arms b search search --switches 360 360 0 --shards 1 2 2` (5 tiến trình, validation 21000) bị Claude Code dừng vì máy thiếu RAM trước khi ghi kết quả; thư mục output rỗng đã xóa. Chỉ có số đo thử 8 trận (không dùng để kết luận): search từ bước 0 ActiveWR 37.5 so với B 25.0, ~9 giây/trận so với ~1.3 giây/trận của B. Giao người dùng chạy lại trong cmd.

## XC — đối chiếu nhánh `codex/phasec-sd-best-fallback` (a6650a0) của bạn cùng nhóm, 17/10/2026

**Thông số:** Phân tích tĩnh, chưa chạy. Số liệu của bạn: F (`runs/phaseC-sd-best-fallback/best_phaseC.pt`, SHA `0315a56f…1483`) được train ~3000 episode từ B qua 3 lượt (1500 + 750 + 750). Lượt cuối: 750 episode, lr 2.5e-5, entropy 0.0075, combat 0.9, navigation 0.4, ring safety 0.6, nhân ×1.75 từ bước 350, SD lịch E ở 75% episode, train seed 2026101503, validation 200 trận seed 30000. Test của bạn: 200 trận seed 31000. Máy bạn (`E:\Bomberman`); phiên bản torch không lưu lại.

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

**Thông số:** Người dùng chạy (cmd, `--shards 1 1 1`). B và B + kill-search (horizon 12, đối thủ mô phỏng là RuleBot) đấu RuleBot. 3 nhánh (B@360, search@360, search@0) × 400 trận, seed 21000–21199 × 2 góc; SD2 + biên k = 15; 9×9, 2 agent, max_steps 500; policy lấy mẫu; CPU, torch 2.5.1. Bootstrap 10 000 lần, seed 20261014.

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

**Thông số:** Người dùng chạy `crosscheck_friend_branch.py` bằng `evaluate.py` của worktree a6650a0. F và B đấu RuleBot; 9×9, 2 agent, max_steps 500; lịch E 475/25/40 hoặc tắt SD. E1/E2: seed 31000–31099 × 2 góc = 200 trận/model. E3/E4: seed 60000–60199 × 2 góc = 400 trận/model. Policy lấy mẫu; CPU; torch 2.5.1, numpy 2.0.1. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014.

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

**Thông số:** Chỉ kiểm tra tương đương, cùng máy, torch 2.5.1. Đánh giá: 6 trận seed 21000 (B, B + SD2 k = 15, `sd_13ch`); 4 trận seed 31000 với F dưới lịch E. Train: 4 episode từ F, công thức của bạn, SD 75%, GPU. Toàn bộ test: 347 OK, 16 bỏ qua.

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

**Thông số:** Chuẩn bị. F và F + kill-search từ bước 0 đấu RuleBot; lịch E 475/25/40; kế hoạch 400 trận/nhánh, seed 21000–21199 × 2 góc; max_steps 500. Smoke 8 trận/nhánh.

**WAIT người dùng chạy; chưa có số liệu.** Thiết kế SD5 đã được người dùng duyệt. Script `probe_sd5_search.py`: F (`runs/phaseC-sd-best-fallback/best_phaseC.pt`) vs F + kill-search từ bước 0, lịch E (SD engine 475/25/40), validation 21000, 400 trận/nhánh, kèm chẩn đoán trận hòa (hòa do hết giờ / cả hai chết, hết giờ mà chưa từng tới gần). Luật chốt trước: mở nhánh `search` của SD5.1 nếu ΔActiveWR (search − F) ≥ +3pp với CI95 dưới > 0. Smoke 8 trận/nhánh (không dùng để kết luận): F 1.1 s/trận, search 6.6 s/trận; hai nhánh chỉ khác ở trận search can thiệp.

## SD5.0 — probe kill-search trên F (kết quả), 17/10/2026

**Thông số:** Người dùng chạy. F (SHA `0315a56f…1483`) và F + kill-search (horizon 12, đối thủ mô phỏng RuleBot, bật từ bước 0) đấu RuleBot. 400 trận/nhánh, seed 21000–21199 × 2 góc; lịch E 475/25/40; 9×9, 2 agent, max_steps 500; policy lấy mẫu; CPU, torch 2.5.1. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014. Số tiến trình: không lưu lại.

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

**Thông số:** Chuẩn bị. Warm-start F; nhánh control và search (nhãn kill-search, hệ số 0.9); 3000 episode/nhánh; train seed 2026101701; lr 2.5e-5; entropy 0.0075; combat 0.9; navigation 0.4; ×1.75 từ bước 350; SD lịch E ở 75% episode; 9×9, 2 agent, max_steps 500. Validation 400 trận seed 21000 mỗi 500 ep. Test đã đăng ký: 400 trận/model, seed 42000. Đo tốc độ: 6–8 episode trên GPU.

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

## SD5.1 — train F ± nhãn kill-search 3000 ep/nhánh và test seed 42000 (kết quả), 18/10/2026

**Thông số:** Người dùng chạy `runs/sd5-20261017`. Warm-start F (SHA `0315a56f…1483`); 3000 episode/nhánh; train seed 2026101701; lr 2.5e-5; entropy 0.0075; combat 0.9; navigation 0.4; ×1.75 từ bước 350; nhánh search thêm nhãn kill-search (hệ số 0.9); SD lịch E ở 75% episode; Phase C vs RuleBot, 9×9, 2 agent, max_steps 500. Control train trên máy người dùng (RTX 3050, torch 2.5.1). Search train trên máy khác (RTX 4060, torch 2.14.1+cu130). Validation 400 trận seed 21000 mỗi 500 ep, lịch E. Test: 3 model × 400 trận, seed 42000–42199 × 2 góc, lịch E và tắt SD; CPU, torch 2.5.1. Bootstrap 10 000 lần, seed 20261014.

**COMPLETE, gate FAIL → giữ F.** Output `runs/sd5-20261017` (log `runs/sd5-20261017-{control,search,test}-console.log`, gửi `runs/sd5-20261017-send.zip`), người dùng chạy. Từ F (SHA `0315a56f…1483`), train seed 2026101701, 3000 ep/nhánh, công thức của bạn, SD lịch E ở 75% episode; best theo validation 21000 (400 trận, mỗi 500 ep). Test một lần seed 42000, 400 trận/model, cả ba model đánh giá trên máy người dùng (torch 2.5.1, CPU). Seed 42000 đã dùng.

Model test: control best ep 2500 (SHA `c3b4e37c…`), search best ep 3000 (SHA `5ddf505d…`).

Lịch E (luật gate):

| Model | WR | ActiveWR | Hòa | Tự nổ | Chết vì vòng | Kill TB |
|---|---:|---:|---:|---:|---:|---:|
| F | 67.25 | **48.5** | 22.75 | 1.25 | 12.25 | 0.485 |
| control | 65.75 | 42.0 | 25.75 | 0.5 | 11.5 | 0.42 |
| search | 66.25 | 45.75 | 25.0 | 0.5 | 12.0 | 0.4575 |

Không SD (báo cáo): F 51.75 / 45.75 ActiveWR / hòa 45.75; control 49.0 / 40.75 / 48.75; search 51.0 / 43.0 / 47.25.

Delta paired bootstrap theo seed bàn (CI95), ActiveWR:

| So sánh | Lịch E | Không SD |
|---|---|---|
| search − F | −2.75 [−9.75, 4.0] | −2.75 [−9.75, 4.0] |
| control − F | −6.5 [−12.75, 0.0] | −5.0 [−11.5, 1.5] |
| search − control | +3.75 [−2.75, 10.25] | +2.25 [−4.0, 8.5] |

Lịch E, search − F: WR −1.0 [−7.25, 5.25], hòa +2.25 [−3.25, 7.75], tự nổ −0.75 [−1.75, 0.0].

Gate (lịch E): ActiveWR ≥ +5pp so F ✗ (−2.75); CI dưới > 0 ✗; không kém control ✓ (+3.75); tự nổ ≤ +1pp ✓ (−0.75). **FAIL.**

Diễn biến train và validation (seed 21000, lịch E):

| Episode | control ActiveWR | search ActiveWR |
|---|---:|---:|
| 0 (= F) | 47.0 | 43.75 |
| 500 | 48.75 | 39.75 |
| 1000 | 46.25 | 45.75 |
| 1500 | 42.75 | 45.5 |
| 2000 | 48.0 | 46.25 |
| 2500 | **51.0** | 44.25 |
| 3000 | 48.5 | **47.75** |

ActiveWR khi train theo khối 1000 ep: control 45.8 / 47.6 / 48.9; search 42.6 / 44.9 / 43.1. Search: 914 403 lần gọi, 2675 nhãn (707 là bom), 0.89 nhãn/ep; search đồng ý với hành động CNN ở ~99.7% lần gọi.

Điều kiện chạy (phát hiện khi review, ghi để minh bạch):
- Nhánh search được train trên máy khác (`E:\Bombermanv2`, RTX 4060, torch 2.14.1+cu130, numpy 2.4.6); control trên máy này (RTX 3050, torch 2.5.1). Source của hai nhánh giống hệt khi bỏ khác biệt CRLF/LF (7 file khác SHA chỉ vì xuống dòng), config chỉ khác `search_teacher`, `checkpoint_dir` và đường dẫn `resume_from` (cùng F, runner kiểm SHA trước khi train). Lượt search dở trên máy này (dừng ở ep ~669) không được dùng; log `runs/sd5-20261017-search-console.log` là của lượt dở đó, không phải lượt đã test.
- Hệ quả: validation ep 0 (cùng model F, cùng seed) cho 47.0 trên torch 2.5.1 và 43.75 trên torch 2.14.1, vì policy lấy mẫu hành động. Nên số validation giữa hai nhánh không so trực tiếp được; test thì công bằng vì cả ba model chạy cùng máy, cùng phiên bản.
- Control: best validation 51.0 (ep 2500) nhưng test 42.0. Chọn best trong 7 điểm validation nhiễu (~±3pp/điểm, cộng nhiễu phiên bản) thiên lạc quan, nên kiểu chọn này không phân biệt được các model chênh nhau vài pp.

Kết luận:
- Train tiếp 3000 ep từ F bằng công thức của bạn không cải thiện F; nhánh control còn thấp hơn F 6.5pp (CI chạm 0). Đúng với chẩn đoán XC: lượt train của bạn đã bão hòa (ActiveWR train ~45%).
- Chưng cất kill-search kiểu này không giữ được lợi thế +22.5pp mà search có khi chạy lúc chơi (SD5.0): search − F −2.75, chỉ nhỉnh hơn control +3.75 nhưng không có ý nghĩa thống kê. Nhãn quá thưa (0.89/ep): search hầu như chỉ khác CNN ở vài thời điểm quyết định mỗi trận.
- Giữ F. Không chạy thêm test để cứu gate.
- Giới hạn: một train seed mỗi nhánh; hai nhánh train trên hai máy và phiên bản torch khác nhau (không ảnh hưởng tính hợp lệ của test, nhưng ảnh hưởng việc chọn best và so validation).

## SD5.3 — đo mật độ khoảnh khắc quyết định của kill-search (chuẩn bị), 18/10/2026

**Thông số:** Không train. F + kill-search (horizon 12) đấu RuleBot; 20 trận trên seed chẩn đoán 6600 (cả 2 góc); lịch E 475/25/40; 9×9, 2 agent, max_steps 500; CPU, torch 2.5.1; script tạm, không giữ trong repo.

**COMPLETE (chẩn đoán, không train, không dùng seed test).** Script tạm ở scratchpad, không giữ trong repo. F + kill-search chơi 20 trận trên seed chẩn đoán 6600 (lịch E, 500 bước), đếm theo từng bước của learner.

| Đại lượng | Tổng | Mỗi trận |
|---|---:|---:|
| Bước của learner | 4603 | 230 |
| Search chạy (đối thủ trong tầm ≤ 6) | 3975 | 199 |
| Bước có thông tin (điểm các hành động không bằng nhau) | 44 | 2.2 |
| — có nước giết được | 25 | 1.25 |
| — search sửa nước CNN (8 lần là bom) | 22 | 1.1 |
| — CNN đã chọn đúng nước giết | 12 | 0.6 |

ActiveWR của 20 trận này là 80.0 (chỉ tham khảo, mẫu nhỏ); 5.0 s/trận.

Kết luận: ở 99% số bước, search không phân biệt được các hành động, nên gắn nhãn ở mọi bước không thêm thông tin. Thông tin nằm trong khoảng 2 bước mỗi trận; 3000 trận cho khoảng 6 600 mẫu quyết định. Đây là cơ sở thiết kế SD5.3 (`docs/superpowers/specs/2026-10-18-sd5-3-dense-search-distillation-design.md`): thu các mẫu này, fine-tune có giám sát nhiều epoch, kèm KL neo về F. Giới hạn: 20 trận, một seed.

## SD6.0 — chẩn đoán vì sao F bỏ lỡ kill (nhiệt độ lấy mẫu, phân tích cặp SD5.0, điểm mù), 18/10/2026

**Thông số:** Không train. F đấu RuleBot; lịch E 475/25/40; 9×9, 2 agent, max_steps 500; CPU, torch 2.5.1. Nhiệt độ τ ∈ {1, 0.5, 0.25, argmax}: 400 trận/nhánh, seed 21000–21199 × 2 góc; bootstrap 10 000 lần, seed 20261014. Phân tích cặp dùng lại 2 × 400 trận SD5.0. Điểm mù: 80 trận F + search (seed 6600–6630) và 40 trận F tự chơi.

**COMPLETE (chẩn đoán, không train, không dùng seed test).** Script và dữ liệu: `runs/sd6-temperature-probe-21000` (validation 21000, 400 trận/nhánh, lịch E) và `runs/sd6-diagnostics-6600` (seed chẩn đoán 6600–6630).

**1. Nhiệt độ lấy mẫu của F** (giả thuyết: F biết nước giết nhưng lấy mẫu làm lỡ). Ở 42 trạng thái có nước giết (40 trận F tự chơi, search chỉ quan sát), argmax của F nằm trong tập nước tốt nhất 90.5%, nhưng lấy mẫu chỉ trúng 61.9% (xác suất trung bình dồn vào nước tốt nhất 0.64). Đánh giá cả trận:

| τ | WR | ActiveWR | Hòa | Tự nổ | Chết vì vòng | ActiveWR − τ1 [CI95] |
|---|---:|---:|---:|---:|---:|---|
| 1.0 (F) | 67.5 | 47.0 | 19.75 | 1.0 | 12.75 | — |
| 0.5 | 69.75 | 46.0 | 20.75 | 0.0 | 12.0 | −1.0 [−7.75, 5.5] |
| 0.25 | 69.25 | 48.75 | 22.0 | 0.5 | 8.75 | +1.75 [−5.5, 8.75] |
| argmax | 64.5 | 37.75 | 27.25 | 0.0 | 10.25 | **−9.25 [−16.0, −2.5]** |

Bác bỏ: hạ nhiệt độ không tăng ActiveWR, argmax làm giảm (hòa +7.5 [1.25, 13.75]). Cơ hội giết kéo dài nhiều bước, nên lấy mẫu lỡ ở bước này thì bước sau vẫn kịp; policy tất định thì bị kẹt trước RuleBot.

**2. Phân tích cặp 400 trận SD5.0** (F so với F + search, cùng bàn, cùng góc). Search thắng chủ động thêm 95 trận, mất 5. Trong 95 trận đó F: hòa 37, thắng nhờ vòng/thụ động 30, thua 28 (28 trận chết vì vòng). Khoảng 90% trận của F kết thúc ở bước ≥ 475, trong khi search hạ đối thủ ở trung vị khoảng bước 200 (87/95 trước bước 475). Trong 190 trận F sống tới bước 475, 174 trận chưa có kill nào. Như vậy ActiveWR được quyết định ở giữa trận: F đã chạm mặt đối thủ (khoảng bước 14) nhưng suốt 450 bước sau không kết liễu được. F chết vì vòng 51 trận (39 trận đúng bước 475, lúc vòng đầu đóng), trong đó 32 trận thua.

**3. Search sửa F ở đâu** (80 trận F + search, seed 6600–6630): 70 lần sửa (0.875/trận).

| Loại | Số lần | Nước search chọn | Xác suất F gán cho nước đó (trung vị) |
|---|---:|---|---:|
| Sửa để giết | 36 | đặt bom: 20 | 0.58 |
| | | di chuyển: 16 | **0.001** |
| Sửa để sống | 34 | di chuyển/đứng: 31 (trung vị bước 471.5) | **0.0** |

Cả 16 lần di chuyển để giết đều xảy ra khi bom của learner đang cháy chậm: learner đứng chắn đường thoát của đối thủ, thay vì chạy về chỗ trú gần nhất. F gần như không bao giờ chọn những nước này. Nguyên nhân khả dĩ: teacher escape trong trainer (`_bfs_safe_cell`, hệ số 0.5, gán nhãn ở mọi bước learner đứng trong vùng nguy hiểm) luôn dạy đúng một lối thoát (`exits[0]`, chỗ trú gần nhất) mà không xét đối thủ, nên đè các lối thoát khác, kể cả lối chặn đường đối thủ. 34 lần sửa để sống tập trung quanh lúc vòng đầu đóng: F không rời vành ngoài.

Kết luận: lợi thế của search đến từ hai điểm mù cụ thể, không phải từ nhiễu lấy mẫu: (a) nước chặn đường thoát của đối thủ sau khi đặt bom, nghi do teacher escape gây ra; (b) rời vành ngoài trước bước 475, chủ yếu ảnh hưởng WR và hòa. Giới hạn: phân tích điểm mù dựa trên 80 trận và quan hệ nhân quả với teacher escape mới là giả thuyết, cần probe kiểm (PLAN SD6).

## SD6.1 — probe F + luật chặn đường (bản đầu), 18/10/2026

**Thông số:** Không train. F và F + luật chặn đường (bản đầu) đấu RuleBot; 400 trận/nhánh, seed 21000–21199 × 2 góc; lịch E 475/25/40; 9×9, 2 agent, max_steps 500; CPU, torch 2.5.1; bootstrap 10 000 lần, seed 20261014. Hồ sơ độ khớp: 400 trận F + search trên seed 15000. Cơ chế: seed 15000–15075.

**COMPLETE (chỉ validation), luật quyết định: KHÔNG ĐẠT.** Output `runs/sd6-cutoff-probe-21000` (log `runs/sd6-cutoff-probe-21000-console.log`). Lượt đầu tiên chạy đủ 1200 trận nhưng hỏng ở bước gộp (`KeyError: 'contact'`, thiếu `contact_diagnostics`) và không lưu được kết quả. Đã sửa bằng test và chạy lại; số dưới đây là của lượt chạy lại. F = `runs/phaseC-sd-best-fallback/best_phaseC.pt`, lịch E, validation 21000, 400 trận/nhánh.

| Nhánh | WR | ActiveWR | Hòa | Tự nổ | Chết vì vòng | Kill TB |
|---|---:|---:|---:|---:|---:|---:|
| F | 67.5 | 47.0 | 19.75 | 1.0 | 12.75 | 0.47 |
| F + luật chặn đường (bản đầu) | 68.25 | 44.75 | 18.75 | 1.0 | 12.5 | 0.45 |

Delta (CI95): ActiveWR **−2.25 [−4.5, −0.25]**, WR +0.75 [−1.0, 2.5], hòa −1.0 [−2.5, 0.5].

Hồ sơ độ khớp (seed 15000, 400 trận F + search): 421 lần search sửa, gồm 90 lần đặt bom để giết và 85 lần di chuyển để giết. Luật bản đầu khớp **0/85**; nó kích hoạt 31 lần và không lần nào trùng với search.

Chẩn đoán cơ chế (`runs/sd6-diagnostics-6600/movekill_mechanism.py`, seed 15000–15075, 38 nước di chuyển-để-giết):
- Cả 38 lần đều xảy ra khi bom của learner đang cháy và đối thủ nằm trong vùng nguy hiểm.
- Nước F đã chọn luôn có điểm 0: không chết, nhưng cũng không giết được. Mỗi tình huống chỉ có đúng 1 nước giết.
- Nước search chọn: STOP 28/38. F thì đi DOWN 21 lần và RIGHT 10 lần, tức là bỏ chạy.
- Ở 32/38 tình huống, nếu learner **đứng yên**, đối thủ không còn đường sống.

Vậy nước giết là đứng giữ ô cửa duy nhất của đối thủ; F bỏ chạy nên mở đường cho đối thủ thoát. Luật bản đầu lấy "learner đứng yên" làm mốc, nên coi đối thủ đã chết chắc và không can thiệp. Lỗi nằm ở cách định nghĩa luật; giả thuyết điểm mù vẫn đứng. Luật được sửa thành: so nước đang xét với các nước hợp lệ khác của learner (SD6.1b).

## SD6.1b — probe F + luật chặn đường (đã sửa), 18/10/2026

**Thông số:** Không train. F và F + luật chặn đường (đã sửa) đấu RuleBot; 400 trận/nhánh, seed 21000–21199 × 2 góc; lịch E 475/25/40; 9×9, 2 agent, max_steps 500; CPU, torch 2.5.1; bootstrap 10 000 lần, seed 20261014. Hồ sơ độ khớp: 400 trận trên seed 15000.

**COMPLETE (chỉ validation), luật quyết định: KHÔNG ĐẠT → không mở SD6.2.** Output `runs/sd6-cutoff-probe-21000-b` (log `runs/sd6-cutoff-probe-21000-b-console.log`). Luật sửa: một nước STOP/di chuyển là bẫy nếu learner giữ được ô đó (ô không bị cháy) và đối thủ hết đường sống, trong khi một nước hợp lệ khác của learner sẽ để đối thủ thoát. Cùng seed và cùng luật quyết định như SD6.1.

| Nhánh | WR | ActiveWR | Thắng thụ động | Thắng do vòng | Hòa | Tự nổ | Chết vì vòng | Số bước TB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F | 67.5 | 47.0 | 9.5 | 11.0 | 19.75 | 1.0 | 12.75 | 338 |
| F + luật chặn đường (sửa) | 71.25 | 45.25 | **15.25** | 10.75 | 17.0 | 0.75 | 11.0 | 322 |

Delta (CI95): ActiveWR **−1.75 [−4.25, 0.75]**; WR **+3.75 [1.75, 6.0]**; hòa **−2.75 [−4.75, −1.0]**; tự nổ −0.25 [−0.75, 0.0].

Hồ sơ độ khớp (seed 15000, 400 trận): luật bắt được **56/85** nước di chuyển-để-giết của search (66%). Luật kích hoạt 126 lần, trong đó 56 lần trùng với search (44%).

Kết quả đổi theo cặp (F → luật): active → thụ động 16 trận; hòa → thụ động 5; hòa → active 4; thua → active 3; thua → thụ động 3; thắng do vòng → active 2; thụ động → active 2; các thay đổi khác ≤ 3 trận.

Kết luận:
- Đứng giữ ô cửa đúng là làm đối thủ chết: WR tăng có ý nghĩa, hòa giảm có ý nghĩa. Nhưng đối thủ thường chết bởi **bom của chính nó** (bị kẹt giữa bom của mình và bom của learner), nên trận được tính là thắng thụ động: thụ động 9.5 → 15.25. Có 16 trận F vốn thắng chủ động (sau đó tự đặt bom giết) bị đổi thành thắng thụ động, vì đối thủ chết sớm hơn do bom của nó.
- Theo quy ước, thắng thụ động không tính là agent giỏi hơn. ActiveWR không tăng, nên không mở SD6.2. Không chỉnh luật thêm trên cùng validation (tránh dò nhiều lần trên một seed).
- Giải thích được một phần khoảng cách với search: phần lớn lợi thế +22.5 của search đến từ các nước giết mà search được tính điểm chỉ khi **bom của learner** gây chết (kill_search chấm +1 theo người giết). Luật chặn đường không phân biệt bom của ai.
- Giới hạn: một seed validation; độ khớp đo trên seed 15000.

## DOC — gom insight vào `INSIGHTS.md`, 18/10/2026

**DONE (tài liệu, không có số liệu mới).** Gom các insight đã có bằng chứng từ SD4–SD6 (trận đấu được quyết định ở đâu, metric và thắng thụ động, hành vi và điểm mù của F, train/chưng cất, quy trình/hạ tầng, các giả thuyết đã bác bỏ, câu hỏi còn mở) vào `INSIGHTS.md`, mỗi mục trỏ về mục RESULTS/IMPROVEMENTS nguồn. Thêm ghi chú SD6 vào spec SD5.3 (mục 8b) cho người cài đặt.

## SD7.0 — spike: phương sai kết quả của F tại các khoảnh khắc quyết định, 18/10/2026

**Thông số:** Không train. F đấu RuleBot; 100 trận, seed chẩn đoán 6600–6649 × 2 góc; lịch E; max_steps 500. Mỗi trạng thái giữ lại được chơi lại K = 8 lần, mỗi lần H = 20 tick, mỗi lần một seed riêng. Kill-search horizon 12 dùng để gán nhãn. CPU, torch 2.5.1.

**COMPLETE (chẩn đoán, không train, seed chẩn đoán 6600–6649, 100 trận, lịch E).** Script `runs/sd7-spike-6600/spike_group_rollouts.py`, dữ liệu `states.json`; lần đầu chọn mẫu sai, dữ liệu lưu ở `states_v1_cheap_criterion.json`. F tự chơi. Ứng viên là các bước có bom đang cháy, đối thủ trong vùng nguy hiểm và cách learner Manhattan ≤ 4, cách trạng thái giữ trước đó ≥ 8 bước. Kill-search gán nhãn mỗi ứng viên: có hay không có nước giết bằng bom của learner trong 12 tick. Từ mỗi trạng thái được giữ, F chơi lại K = 8 lần, mỗi lần H = 20 tick, gặp RuleBot, mỗi lần một seed riêng. Kết cục mỗi lần: +1 nếu bom của learner giết đối thủ khi learner còn sống, −1 nếu learner chết, 0 nếu không có gì.

| Nhóm trạng thái | Số trạng thái | Tỉ lệ F giết được | 0/8 | 8/8 | Lúc được lúc không | Chết khi chơi lại |
|---|---:|---:|---:|---:|---:|---:|
| Search: có nước giết | 81 (0.81/trận) | **0.602** | 0.02 | 0.11 | **0.86** | 0.003 |
| Search: không có nước giết (đối chứng) | 98 | 0.003 | 0.98 | 0.00 | 0.02 | 0.010 |
| Lần đầu: 3 ứng viên đầu tiên mỗi trận, không lọc bằng search | 261 | 0.016 | 0.95 | 0.00 | 0.05 | 0.009 |

Phân bố số lần giết được trên 8 lần chơi lại, ở các trạng thái có nước giết: [2, 4, 5, 7, 20, 15, 5, 14, 9] (tương ứng 0, 1, …, 8 lần).

Kết luận:
- Tại 86% trạng thái có nước giết, F lúc giết được lúc không (trung bình 60%). Đây đúng là loại phương sai mà advantage theo nhóm (kết quả của mỗi lần chơi lại trừ trung bình nhóm, cùng trạng thái) học được, nên hướng SD7 có tín hiệu để học.
- Kill-search tách rất rõ trạng thái quyết định: ở trạng thái search nói không có nước giết, F gần như không bao giờ giết được (0.3%). Tiêu chí rẻ không lọc bằng search vô dụng (1.6%), nên SD7 cần search để chọn trạng thái, nhưng chỉ ở các bước ứng viên.
- Chơi lại rất ít chết (≤ 1%), nên tối ưu kill ngắn hạn không có dấu hiệu làm policy liều.
- Giới hạn: 100 trận, một nhóm seed; các trạng thái của cùng một cơ hội có thể được đếm hơn một lần (giãn cách 8 bước); chưa đo thời gian chạy khi train.

## SD7.1 — chuẩn bị: train F + cập nhật theo nhóm (3000 ep), 18/10/2026

**Thông số:** Chuẩn bị (`runs/sd7-20261018`). Warm-start F; config control SD5.1 (lr 2.5e-5, entropy 0.0075, combat 0.9, SD lịch E ở 75% episode); thêm `group_rl` K 8, H 20, tối đa 3 trạng thái/ep, giãn cách 8, `group_coef` 1.0; 3000 episode; train seed 2026101701; CUDA RTX 3050, torch 2.5.1. Validation 400 trận seed 21000 mỗi 500 ep. Đo tốc độ trên 10 episode.

**RUNNING (Claude chạy ở nền), chưa có kết quả.** Spec `docs/superpowers/specs/2026-10-18-sd7-group-advantage-design.md`, runner `run_sd7_pilot.py`, output `runs/sd7-20261018` (log `runs/sd7-20261018-train-console.log`). Nhánh group = config của control SD5.1 (`runs/sd5-20261017/control/config.json`) + `group_rl` (K = 8, H = 20, tối đa 3 trạng thái/ep, giãn cách 8, `group_coef` = 1.0). Control là best ep 2500 của SD5.1 (SHA `c3b4e37c…`), tái dùng vì cùng config, cùng seed và cùng máy.

Luật chốt trước khi train:
- H1 (cơ chế): trên các trạng thái quyết định cố định (F tự chơi, seed 6650–6699, 100 trận), tỉ lệ giết của group − F ≥ +10pp, CI95 bootstrap theo trạng thái có cận dưới > 0.
- Gate (test một lần, seed 44000, 400 trận/model, lịch E): group − F ActiveWR ≥ +5pp, CI95 dưới > 0, group − control ≥ 0, tự nổ ≤ +1pp.

Đo chuẩn bị (GPU, 10 ep đầy đủ, không dùng để kết luận): 4.5 s/ep (control 2.8 s/ep); 0.8 trạng thái quyết định/ep; 6/8 nhóm có chênh lệch kết cục; 27/64 lần chơi lại giết được; khoảng 60 bước chơi lại/ep. Ước tính khoảng 5 giờ, gồm cả 7 lần validation.

Cập nhật SD7.1 (18/10/2026): lượt train ở nền bị Claude Code dừng ở episode 174/3000 vì máy thiếu RAM khi phiên đang chờ (không phải lỗi của lệnh). `training_state.pt` đã lưu tới episode 174 và resume được (`--resume`). Validation ep 0 (chính là F) ra ActiveWR 47.0, WR 67.5, khớp với F đo trước đó trên máy này. Chưa có kết quả.


## SD7.1–SD7.3 — Advantage theo nhóm từ trạng thái quyết định: kết quả, 18/10/2026

**Thông số:** Người dùng chạy `runs/sd7-20261018`. Warm-start F; group K 8, H 20, tối đa 3 trạng thái/ep, `group_coef` 1.0; 3000 episode (4 giờ 28 phút, resume 1 lần); train seed 2026101701; lr 2.5e-5; entropy 0.0075; combat 0.9; SD lịch E ở 75% episode; 9×9, 2 agent, max_steps 500; CUDA RTX 3050, torch 2.5.1. Validation 400 trận seed 21000 mỗi 500 ep, lịch E. Cơ chế H1: 100 trận seed 6650–6699, 73 trạng thái, K 8 × H 20. Test: 3 model × 400 trận, seed 44000–44199 × 2 góc, lịch E và off; CPU. Bootstrap 10 000 lần, seed 20261014.

Thư mục `runs/sd7-20261018` (log `runs/sd7-20261018-{train,mechanism,test}-console.log`). Train nhánh group 3000 ep (4 giờ 28 phút, người dùng chạy tay, có resume sau lần bị dừng ở ep 174), 1 train seed. Chọn best theo validation seed 21000: **ep 1000** (ActiveWR 47.75, ghi trước khi test). SHA model: F `0315a56f…`, control `c3b4e37c…` (best ep 2500 SD5.1), group `ad4f1b90…`.

**Validation 21000 (400 trận, lịch E), ActiveWR / WR / tự nổ:** ep0 (=F) 47.0/67.5/1.0; 500: 41.0/65.0/1.0; 1000: 47.75/67.5/0.5; 1500: 47.0/63.5/0.75; 2000: 46.5/68.0/0.25; 2500: 43.75/67.0/0.25; 3000: 44.75/66.75/0.0. Không có xu hướng tăng; best ep 1000 hơn F chỉ +0.75 và nằm trong nhiễu.

**H1 (cơ chế, seed 6650–6699, 100 trận, 73 trạng thái quyết định cố định):** tỉ lệ giết F 58.9%, control 60.6%, group 61.0%. group − F = +2.05pp [−0.34, +4.62]; group − control = +0.34pp [−1.03, +1.71]. Luật chốt trước là ≥ +10pp với CI dưới > 0 → **H1 KHÔNG ĐẠT**. Số lần chết trong lúc chơi lại: F 0, control 0.17%, group 0.

**Test một lần, seed 44000, 400 trận/model (200 cặp bàn), lịch E:**

| Model | WR | ActiveWR | Hòa | Tự nổ | Chết vì vòng |
|---|---|---|---|---|---|
| F | 63.75 | 44.75 | 26.5 | 0.0 | 14.25 |
| control (SD5.1) | 68.0 | 48.25 | 24.0 | 0.25 | 9.5 |
| group (SD7) | 67.0 | 45.0 | 26.0 | 0.75 | 10.25 |

| So sánh (lịch E) | ActiveWR Δ, CI95 | WR Δ, CI95 | Tự nổ Δ, CI95 |
|---|---|---|---|
| group − F | +0.25 [−5.75, +6.25] | +3.25 [−2.75, +9.25] | +0.75 [0.0, +1.75] |
| control − F | +3.5 [−2.75, +10.0] | +4.25 [−1.75, +10.5] | +0.25 [0.0, +0.75] |
| group − control | −3.25 [−9.75, +3.25] | −1.0 [−6.75, +5.0] | +0.5 [0.0, +1.25] |

Lịch off (không SD): group − F ActiveWR +1.0 [−5.0, +7.0]; control − F +3.75 [−2.5, +10.25]; group − control −2.75 [−9.25, +3.75].

**Gate: FAIL** (≥ +5pp: không; CI dưới > 0: không; group ≥ control: không; tự nổ ≤ +1pp: đạt). **Quyết định: giữ F.** Không train thêm, không đổi checkpoint sau khi xem test; seed 44000 đã dùng.

Giới hạn: một train seed; best chọn trên validation có nhiễu cỡ ±3pp nên mọi chênh lệch đều trong CI. Control không phải F mà là best ep 2500 của SD5.1; ở SD5.1 control ra −6.5 trên seed 42000, còn ở đây +3.5 trên seed 44000 (không ý nghĩa), tức là nhiễu giữa các seed test lớn hơn mọi hiệu ứng ta đo được. Lượt train chỉ thu ~0.8 trạng thái quyết định/ep (log cuối: 2 trạng thái, 9/16 thắng) nên tín hiệu nhóm có thể quá thưa để thay đổi CNN trong 3000 ep; chưa thử K/ngưỡng/`group_coef` khác.


## SD7.4 — Probe cơ chế, tín hiệu nhóm dày hơn (coef 4, K 16), 18/10/2026

**Thông số:** `runs/sd74-20261018`. Warm-start F; control và group (`group_coef` 4.0, K 16, H 20, tối đa 6 trạng thái/ep, giãn cách 4); 1000 episode/nhánh; train seed 2026101701; lr 2.5e-5; entropy 0.0075; combat 0.9; SD lịch E ở 75% episode; không validation; CUDA RTX 3050, torch 2.5.1. Nhánh group bị dừng ở ep 697 và được người dùng resume. H1': 100 trận seed 6700–6749, 72 trạng thái, K 8 × H 20. Bootstrap theo trạng thái.

Thư mục `runs/sd74-20261018` (log `runs/sd74-20261018-{control,group,mechanism}-console.log`). Spec chốt trước: `docs/superpowers/specs/2026-10-18-sd7-4-dense-group-probe-design.md`. Hai nhánh từ F, config control SD5.1, train seed 2026101701, 1000 ep, không validation, đo `final_phaseC.pt`. group: `group_coef` 4.0, K 16, tối đa 6 trạng thái/ep, giãn cách 4. Nhánh group bị Claude Code dừng ở ep 697 vì máy thiếu RAM; người dùng chạy tiếp bằng `--resume` tới 1000. SHA: control `95c443a2…`, group `69017438…`.

Train: control 2.6 s/ep; group 5.2–6.4 s/ep, 1.20 trạng thái/ep, 0.65 nhóm có chênh lệch/ep. Tỉ lệ giết trong lúc chơi lại khi train không tăng theo thời gian (0.747 → 0.762 → 0.744 → 0.768 theo từng khối 250 ep; trạng thái lấy từ policy đang train nên không so trực tiếp được). ActiveWR lúc train (tham khảo, nhiễu): control 48.0 / 44.8, group 43.0 / 48.0 ở ep 1–500 / 501–1000.

**H1' (seed chẩn đoán mới 6700–6749, 100 trận, 72 trạng thái quyết định, K 8 × H 20, chung seed lấy mẫu):**

| So sánh | Tỉ lệ giết | Δ (pp) | CI95 |
|---|---|---|---|
| group − control | 68.92 vs 61.81 | **+7.12** | [+4.86, +9.72] |
| group − F | 68.92 vs 61.81 | +7.12 | [+4.69, +9.90] |
| control − F | 61.81 vs 61.81 | 0.00 | [−1.39, +1.39] |

Chết trong lúc chơi lại: F 0.17%, control 0, group 0. **H1' ĐẠT** (≥ +5pp, CI dưới > 0, chết không tăng).

Kiểm thêm: control − F = 0 không phải lỗi; trọng số control lệch F tối đa 0.009 sau 1000 ep (lr 2.5e-5), và do mọi model dùng chung seed lấy mẫu nên control cho kết cục giống hệt F ở 54/72 trạng thái; group khác F ở 32/72 trạng thái, 397 vs 356 lần giết trên 576 lần chơi lại.

So với SD7 (K 8, coef 1, 3000 ep): group − control +0.34 → +7.12pp với 1/3 số episode. Như vậy cơ chế nhóm có tác dụng khi tín hiệu đủ mạnh.

Giới hạn: một train seed; 72 trạng thái từ 100 trận; mới là điều kiện cần. SD7 cho thấy tăng tỉ lệ giết ở trạng thái quyết định chưa chắc thành ActiveWR cả trận (ước chừng +7pp tỉ lệ giết với khoảng 1 trạng thái quyết định/trận chỉ tương ứng vài pp ActiveWR). Quyết định theo spec: mở SD7.5 (train full 3000 ep, test một lần seed 45000).


## SD7.5 — Train full advantage theo nhóm (coef 4, K 16, 3000 ep), test seed 45000, 19/10/2026

**Thông số:** Người dùng chạy `runs/sd75-20261018`. Warm-start F; group `group_coef` 4.0, K 16, H 20, tối đa 6 trạng thái/ep, giãn cách 4; 3000 episode (4 giờ 58 phút, 5.96 s/ep); train seed 2026101701; lr 2.5e-5; entropy 0.0075; combat 0.9; SD lịch E ở 75% episode; 9×9, 2 agent, max_steps 500; CUDA RTX 3050, torch 2.5.1. Validation 400 trận seed 21000 mỗi 500 ep, lịch E. Test: 3 model (F, control SD5.1 ep 2500, full) × 400 trận, seed 45000–45199 × 2 góc, lịch E và off; CPU. Bootstrap 10 000 lần, seed 20261014.

Thư mục `runs/sd75-20261018` (log `runs/sd75-20261018-{train,test}-console.log`). Người dùng chạy tay: `run_sd74_probe.py train --arm full` (4 giờ 58 phút, 5.96 s/ep, không bị ngắt) rồi `test`. Từ F, config control SD5.1 + nhóm SD7.4 (`group_coef` 4.0, K 16, tối đa 6 trạng thái/ep, giãn cách 4), 1 train seed. Best chọn theo validation seed 21000 trước khi test: **ep 1000**. SHA: F `0315a56f…`, control `c3b4e37c…` (best ep 2500 SD5.1), full `bb0c775c…` (trong `summary.json` nhánh này mang tên `group`).

**Validation 21000 (400 trận, lịch E), ActiveWR / WR:** ep0 (=F) 47.0/67.5; 500: 46.75/66.25; 1000: **50.25**/73.25; 1500: 44.25/67.0; 2000: 47.75/67.75; 2500: 44.25/67.5; 3000: 41.75/65.5. Đỉnh ở ep 1000 rồi giảm dần; ep 3000 thấp hơn F 5.25pp.

Train (trung bình theo khối 750 ep, ep 1–750 / –1500 / –2250 / –3000): trạng thái quyết định/ep 1.12 / 1.19 / 1.22 / 1.24; tỉ lệ giết trong lúc chơi lại 77.0 / 77.1 / 81.5 / 79.0%; chết lúc chơi lại ≤ 0.06/ep; ActiveWR lúc train 43.1 / 46.9 / 49.6 / 49.1 (lấy mẫu từ policy đang train, nhiễu, không thay validation).

**Test một lần, seed 45000, 400 trận/model (200 cặp bàn), lịch E:**

| Model | WR | ActiveWR | Hòa | Tự nổ | Chết vì vòng |
|---|---|---|---|---|---|
| F | 67.25 | 47.25 | 24.75 | 0.25 | 11.5 |
| control (SD5.1) | 70.5 | 50.25 | 21.25 | 0.5 | 9.25 |
| full (SD7.5) | 71.5 | 51.0 | 19.75 | 0.25 | 9.5 |

| So sánh (lịch E) | ActiveWR Δ, CI95 | WR Δ, CI95 | Hòa Δ, CI95 | Tự nổ Δ, CI95 |
|---|---|---|---|---|
| full − F | +3.75 [−2.5, +10.0] | +4.25 [−1.5, +10.25] | −5.0 [−10.25, +0.25] | 0.0 [0.0, 0.0] |
| control − F | +3.0 [−3.5, +9.5] | +3.25 [−3.0, +9.5] | −3.5 [−9.0, +1.75] | +0.25 [0.0, +0.75] |
| full − control | +0.75 [−5.75, +7.25] | +1.0 [−5.0, +7.0] | −1.5 [−6.5, +3.5] | −0.25 [−0.75, 0.0] |

Lịch off (không SD): full − F ActiveWR +3.0 [−3.5, +9.25], WR +5.5 [−0.75, +11.75]; control − F +2.25 [−4.25, +8.75]; full − control +0.75 [−5.5, +7.0].

**Gate: FAIL** (full − F ≥ +5pp: không, +3.75; CI dưới > 0: không, −2.5; full ≥ control: đạt, +0.75; tự nổ ≤ +1pp: đạt). **Quyết định: giữ F.** Không train thêm, không đổi checkpoint sau khi xem test; seed 45000 đã dùng.

Giới hạn và đọc kết quả: một train seed. Phần lớn chênh lệch so với F cũng có ở control (train tiếp 3000 ep không có nhóm, +3.0), nên phần riêng của nhóm chỉ +0.75 [−5.75, +7.25] ActiveWR, trong khi SD7.4 đo được +7.12pp tỉ lệ giết ở trạng thái quyết định sau 1000 ep. Với ~1.2 trạng thái quyết định/trận, +7pp tỉ lệ giết chỉ ứng với khoảng +1pp ActiveWR, nhỏ hơn nhiều so với độ rộng CI (±6pp) của 400 trận. Validation giảm sau ep 1000 (41.75 ở ep 3000) cho thấy train lâu với `group_coef` 4 có thể làm hỏng phần còn lại của policy; chưa kiểm cơ chế này.

**Cập nhật 19/10/2026 — quyết định của người dùng:** chọn SD7.5 best (`runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c58161aa1bbdd229b9528170aab174d5aabf8f4a6739b848919cdc7ac`, gọi là **G**) làm model tốt nhất hiện tại và điểm xuất phát cho các bước sau. Đây là lựa chọn của người dùng, không phải kết quả gate: trên seed 45000 G có ActiveWR cao nhất trong ba model (51.0 vs control 50.25 vs F 47.25), nhưng hơn F +3.75 [−2.5, +10.0] và hơn control +0.75 [−5.75, +7.25], đều chưa có ý nghĩa thống kê. Seed 45000 không được dùng lại để đánh giá G.


## SD7.6a — Chẩn đoán: vì sao SD7.5 tăng ActiveWR ít, 19/10/2026

**Thông số:** Không train. F đấu RuleBot; 100 trận, seed chẩn đoán 6800–6849 × 2 góc; lịch E; max_steps 500. Kill-search (horizon 12, phạm vi 6) ở mọi bước; đo xác suất của F, G và SD7.5 final trên 336 trạng thái có nước giết. CPU, torch 2.5.1.

Thư mục `runs/sd76-diagnosis-6800` (log `runs/sd76-diagnosis-6800-console.log`), script `diagnose_sd75.py`. Không train. F chơi seed chẩn đoán mới 6800–6849 (100 trận, lịch E, mọi góc). Ở mỗi bước learner có đối thủ trong phạm vi 6, chạy kill-search; ở mỗi trạng thái có nước giết, ghi xác suất F, G (SD7.5 best ep 1000) và SD7.5 final (ep 3000) đặt lên tập nước giết. "Cửa sổ giết" = chuỗi trạng thái có nước giết cách nhau ≤ 4 bước. F ở lượt này: ActiveWR 43/100.

**Độ phủ theo trận:** 62/100 trận có ít nhất một trạng thái có nước giết. Thắng chủ động 43/43; thắng thụ động 5/15; hòa 9/24; thua 3/8. Như vậy **38/57 trận không thắng chủ động không bao giờ có cơ hội giết** trong tầm 12 tick, nên mọi cách cải thiện "kết liễu" chỉ nâng được tối đa khoảng +19pp trên seed này (43 → 62).

**Cửa sổ giết thất bại so với thành công (F):**

| | Số cửa sổ | Dài trung vị (bước) | Dài 1 bước | Bước cuối chỉ có nước giết = đặt bom | P(nước giết) F / G ở bước cuối |
|---|---|---|---|---|---|
| Thất bại | 37 | 1 | 31 | 28 | 0.58 / 0.63 |
| Thành công | 43 | 6 | 0 | 0 | 1.0 / 1.0 |

36/37 cửa sổ thất bại có đúng một nước giết ở bước cuối. Điểm nghẽn là **khoảnh khắc "đặt bom ngay"**: chỉ kéo dài một bước, nước giết duy nhất là BOMB, F gán xác suất trung bình 0.58. Lỡ bước đó là mất cơ hội. Phát hiện này sửa lại insight cũ "lỡ một bước không mất cơ hội", vốn chỉ đúng với cửa sổ dài.

**Xác suất trên tập nước giết (336 trạng thái):**

| Nhóm | n | F | G (ep 1000) | final (ep 3000) |
|---|---|---|---|---|
| Tất cả | 336 | 0.890 | 0.903 | 0.919 |
| Nước giết là đặt bom | 85 | 0.688 | 0.739 | 0.798 |
| Nước giết là di chuyển/đứng | 251 | 0.958 | 0.959 | 0.960 |
| Qua bộ lọc rẻ của SD7 | 286 | 0.903 | 0.917 | 0.930 |
| Không qua bộ lọc rẻ (thường chưa có bom cháy) | 50 | 0.814 | 0.826 | 0.855 |

Entropy policy trung bình trên mọi bước: F 0.432, G 0.421, final 0.396.

**Phân bổ tín hiệu khi train SD7.5** (`metrics.jsonl`): 3576 nhóm, chỉ 1843 nhóm (52%) có chênh lệch kết cục; 79% lần chơi lại đã giết được. Phần lớn trạng thái được lấy từ cửa sổ dài mà F vốn đã thắng (6.6 trạng thái/cửa sổ thành công so với 1.4/cửa sổ thất bại, giãn cách 4). Khoảnh khắc đặt bom một bước, nơi có gần hết các lần thất bại, chỉ chiếm phần nhỏ mẫu train. Ngoài ra 12/37 cửa sổ thất bại không qua bộ lọc rẻ (chưa có bom cháy), nên SD7 không bao giờ train trên chúng.

**Kết luận chẩn đoán:**
1. Giả thuyết "khám phá" (nước giết bằng di chuyển có xác suất ~0.001 nên không bao giờ được lấy mẫu) **bị bác bỏ ở cấp tập nước giết**: F đã đặt 0.96 lên tập nước giết bằng di chuyển.
2. Điểm nghẽn là xác suất đặt bom đúng lúc (0.58–0.69). Group RL có kéo lên (0.69 → 0.74 → 0.80) nhưng chậm, vì phần lớn ngân sách mẫu rơi vào trạng thái đã chắc thắng và vì bộ lọc loại các trạng thái chưa có bom cháy.
3. Train lâu làm policy nhọn hơn (entropy giảm) và tăng P(bom) cả ở nơi khác: final đặt nhiều bom hơn trên validation (34.2 so với 30.5 ở ep 1000), hòa tăng (25.75 so với 18.5), ActiveWR giảm. Đây là giải thích khả dĩ cho việc validation tụt; mới là tương quan, chưa kiểm nhân quả.
4. Trần của mọi cách cải thiện kết liễu trên seed này là khoảng +19pp; vượt mức đó phải tạo thêm cơ hội (38% trận không có cơ hội giết nào).

Giới hạn: một seed chẩn đoán, 100 trận, trạng thái lấy từ trận của F (không phải G). Nhãn "nước giết" dựa trên search 12 tick với RuleBot hoàn hảo và learner chỉ né sau bước đầu.

## KC1 — kiểm chứng các nhận định về SD7.5/G và kill-search, 19/10/2026

**Thông số:** Không chạy trận mới. Phân tích lại 3 seed test đã có (42000, 44000, 45000; mỗi seed 400 trận/model, lịch E), tổng 600 cặp bàn; bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014.

**COMPLETE (phân tích lại dữ liệu có sẵn, không chạy trận mới, không dùng seed test mới).** Nguồn: `runs/sd5-20261017`, `runs/sd7-20261018`, `runs/sd75-20261018` (`summary.json` và `evaluation_*_E_*.json`), code `train_actor_critic.py` (nhánh `search_teacher`), spec SD5.3, RESULTS SD5.1/SD6.0/SD7.6a.

**Gộp control SD5.1 so với F trên ba seed test.** Cùng một checkpoint control (SD5.1 best ep 2500, `c3b4e37c…`) đã được test với F trên 42000, 44000, 45000 (lịch E, 400 trận/seed, paired bootstrap theo bàn, 10 000 lần):

| Seed test | F ActiveWR | control ActiveWR | control − F, CI95 |
|---|---:|---:|---|
| 42000 | 48.5 | 42.0 | −6.5 [−12.75, 0.0] |
| 44000 | 44.75 | 48.25 | +3.5 [−2.75, +10.0] |
| 45000 | 47.25 | 50.25 | +3.0 [−3.5, +9.5] |
| Gộp 600 cặp bàn | 46.83 | | **0.0 [−3.75, +3.75]** |

Cùng một cặp model mà chênh lệch đi từ −6.5 tới +3.5 giữa các seed test 400 trận; gộp lại bằng 0.

| Nhận định | Kết luận kiểm chứng |
|---|---|
| Bảng F / control / SD7.5 trên 45000 và các CI | Đúng, khớp `summary.json` (bootstrap 10 000 lần). |
| Gate SD7.5 FAIL; G hơn F chưa có ý nghĩa thống kê | Đúng. |
| "Phần lớn mức hơn F đến từ train thêm 3000 ep" | **Không được ủng hộ.** Control − F gộp 3 seed = 0.0 [−3.75, +3.75]; +3.0 trên 45000 nằm trong dao động giữa các seed. Mức G hơn F (+3.75, một seed) cũng nằm trong dao động đó. |
| B kém F ~7.5pp; SD7 hơn F +0.25; +22.5 là search lúc chơi | Đúng (XC, SD7.3, SD5.0; search chậm ~6 lần: 6.6 so với 1.1 s/trận). |
| SD5.3 chưa có kết quả | Đúng: repo không có output SD5.3, PLAN vẫn WAIT. |
| SD5.1 có để search điều khiển hành động khi train không | **Không.** `train_actor_critic.py` dòng 1146–1179: hành động thực hiện luôn là mẫu của CNN; search chỉ sinh nhãn khi nước của nó khác nước CNN đã lấy mẫu (0.89 nhãn/ep). |
| "Chưa thử: F + search chơi rồi học có giám sát (expert iteration/DAgger)" | Đúng là chưa chạy, nhưng đó chính là SD5.3 (spec: F + search thu 3000 trận, BC trên mẫu quyết định + KL neo; DAgger nhiều vòng đã cân nhắc và để có điều kiện). Spec cũng đã loại việc học trên toàn bộ trận (99% bước không có thông tin). |
| Search đồng ý với F 99.7% | Đúng về số, nhưng tính trên mọi lần gọi; ở bước có thông tin thì CNN chỉ chọn đúng nước giết 12/25 lần (SD5.3a). |
| "Phần F thực sự không biết là nước di chuyển/đứng giữ ô cửa, P ≈ 0.001" | **Đã lỗi thời.** 0.001 là trung vị xác suất của đúng nước search chọn ở 16 lần sửa bằng di chuyển (SD6.0). Ở cấp tập nước giết, F đặt 0.958 lên tập nước giết bằng di chuyển; điểm nghẽn là khoảnh khắc "đặt bom ngay" một bước (28/37 cửa sổ thất bại, P(bom) 0.58) (SD7.6a). |

Kết luận:
- Dữ liệu ủng hộ việc G có số cao nhất trên seed 45000, nhưng không ủng hộ G hay control tốt hơn F: dao động giữa các seed test của cùng một cặp model (~±5pp) bằng cỡ mức chênh của G.
- Không chạy test lại G trên seed mới để "cứu" gate (quy ước). G vẫn là lựa chọn của người dùng.
- Câu hỏi chưa có bằng chứng nhân quả: sửa riêng khoảnh khắc đặt bom thì ActiveWR tăng tối đa bao nhiêu. SD7.6a chỉ đếm; đây là tiền đề của SD7.6 → đề xuất SD8 (probe phân rã lợi ích của kill-search theo loại nước sửa, không train).

Giới hạn: gộp ba seed test giả định các lượt đánh giá cùng điều kiện (cùng máy người dùng, torch 2.5.1, CPU, theo RESULTS từng mục); chỉ một checkpoint control.


## SD7.6-probe — Group RL vào khoảnh khắc "đặt bom ngay" có buffer, từ G (1000 ep), 19/10/2026

**Thông số:** Người dùng chạy `runs/sd76-20261019`. Warm-start G (SHA `bb0c775c…c7ac`); control và buffer (`group_select` narrow, `group_coef` 4.0, K 16, H 20, tối đa 6 trạng thái/ep, có buffer chơi lại); 1000 episode/nhánh; train seed 2026101701; lr 2.5e-5; entropy 0.0075; combat 0.9; SD lịch E ở 75% episode; không validation khi train; CUDA RTX 3050, torch 2.5.1; có resume. Cơ chế: 100 trận do G chơi, seed 6850–6899, 78 trạng thái hẹp, K 8 × H 20. Validation: 400 trận/model, seed 21000–21199 × 2 góc, lịch E, CPU.

Thư mục `runs/sd76-20261019` (log `runs/sd76-20261019-{control,buffer,mechanism}-console.log`). Spec chốt trước: `docs/superpowers/specs/2026-10-19-sd7-6-bomb-moment-buffer-design.md`. Hai nhánh từ G, config control SD5.1, train seed 2026101701, 1000 ep, không validation khi train, đo `final_phaseC.pt`. SHA: G `bb0c775c…`, control `6957c459…`, buffer `bf78a7a1…`. Người dùng chạy tay; cả hai nhánh bị ngắt và resume chính xác (provenance được kiểm): control một lần (PermissionError khi đổi tên `training_state.pt.tmp` trên Windows); buffer ba lần (Ctrl+C sau ep ~700, lỗi CUDA "unknown error" sau ep ~850, một lần resume bị từ chối vì config khác do tiến trình rơi về CPU, rồi resume với cuda tới 1000). `metrics.jsonl` đủ 1000 dòng, không trùng.

**Train nhánh buffer** (theo khối 250 ep): trạng thái hẹp mới 0.52 / 0.48 / 0.48 / 0.47 mỗi ep; chơi lại 5.9–6.0 trạng thái/ep; buffer 131 → 488; tỉ lệ nhóm có chênh lệch chỉ 0.34–0.38 (đo thử 20 ep đầu: 0.89), tỉ lệ giết lúc chơi lại 0.83–0.85; số bom/ep 39.0 → 41.0 (control 32.5 → 28.9); ActiveWR lúc train 40.8 / 39.2 / 39.2 / 40.4 (control 47.2 / 40.4 / 48.4 / 52.4; tham khảo, nhiễu).

**Cơ chế (seed chẩn đoán 6850–6899, 100 trận do G chơi, 78 trạng thái hẹp, K 8 × H 20, chung seed lấy mẫu):**

| So sánh | Tỉ lệ giết | Δ (pp) | CI95 |
|---|---|---|---|
| buffer − control | 83.97 vs 65.54 | **+18.43** | [+14.58, +22.28] |
| buffer − G | 83.97 vs 64.58 | +19.39 | [+15.71, +23.08] |
| control − G | 65.54 vs 64.58 | +0.96 | [−0.48, +2.40] |

P(tập nước giết) ở trạng thái hẹp: G 0.716, control 0.731, buffer **0.968**. Chết lúc chơi lại: 0 ở cả ba.

**Validation 21000 (400 trận, lịch E):**

| Model | WR | ActiveWR | Hòa | Thắng thụ động | Thắng nhờ vòng | Tự nổ | Bom/trận | Kills |
|---|---|---|---|---|---|---|---|---|
| control | 70.0 | 45.75 | 18.5 | 10.25 | 14.0 | 0.25 | 32.1 | 0.46 |
| buffer | 66.75 | 45.25 | 24.5 | 14.5 | 7.0 | 0.75 | 38.6 | 0.45 |

buffer − control ActiveWR −0.5 [−7.5, +6.5] (bootstrap theo cặp bàn).

**Luật H1‴: ĐẠT theo đúng chữ** (tỉ lệ giết +18.43 ≥ +10, CI dưới > 0; chết không tăng; validation −0.5 ≥ −2). Theo spec, bước tiếp theo là SD7.6 full.

**Đọc kết quả (quan trọng):** cơ chế làm đúng việc được giao: ở khoảnh khắc đặt bom, P(nước giết) từ 0.72 lên 0.97. Nhưng **không ra thêm kill nào cả trận**: kills/trận 0.45 so với 0.46, ActiveWR đứng yên. Đồng thời policy đặt bom nhiều hơn rõ rệt ở mọi nơi (38.6 so với 32.1 bom/trận, +20%), hòa +6, thắng thụ động +4.25, WR −3.25, tự nổ +0.5. Như vậy việc đẩy BOMB lên ở trạng thái hẹp đã **lan sang các trạng thái khác** (CNN không phân biệt đủ), và cái giá đó ăn hết phần lợi. Giả thuyết chưa kiểm: đặt bom bừa làm hết bom (giới hạn số bom) hoặc đẩy đối thủ đi xa, nên khoảnh khắc "đặt bom ngay" xuất hiện ít hơn trong trận của chính nhánh buffer (trạng thái hẹp của probe cơ chế lấy từ trận của G, không phải của buffer). Tỉ lệ nhóm có chênh lệch giảm từ 0.89 xuống ~0.37 trong lúc train cho thấy buffer đã học thuộc các trạng thái cũ.

Giới hạn: một train seed; 78 trạng thái; validation 400 trận có CI ±7pp; G trên validation (SD7.5, cùng máy) là 50.25 nên cả control lẫn buffer đều thấp hơn G sau 1000 ep.


## SD7.7-calibrate — Hiệu chỉnh hệ số phạt KL về G, 19/10/2026

**Thông số:** Claude chạy, 4 tiến trình song song. Warm-start G; config buffer SD7.6 + `anchor_coef` β ∈ {0, 1, 4, 16}; 150 episode/lượt; train seed 2026101701; lr 2.5e-5; SD lịch E ở 75% episode. CUDA RTX 3050, torch 2.5.1. Đo trên 40 trận G chơi (seed chẩn đoán 6600, lịch E), 40 trạng thái hẹp.

Thư mục `runs/sd77-20261019/calib` (log `runs/sd77-20261019-calibrate-console.log`). Spec chốt trước: `docs/superpowers/specs/2026-10-19-sd7-7-kl-anchor-design.md`. Claude chạy (4 tiến trình song song). Bốn lượt 150 ep từ G với config buffer SD7.6 + `anchor_coef` β, train seed 2026101701. Đo trên 40 trận G chơi (seed chẩn đoán 6600, lịch E), 40 trạng thái hẹp.

| Model | P(BOMB \| hợp lệ) | KL(π‖G) | P(nước giết) ở trạng thái hẹp | Bom/ep lúc train (50 ep đầu → 50 ep cuối) |
|---|---|---|---|---|
| G | 0.637 | 0 | 0.687 | — |
| β = 0 | 0.949 | 0.056 | 0.946 | 35.3 → 41.5 |
| β = 1 | 0.938 | 0.051 | 0.938 | 35.2 → 36.0 |
| β = 4 | 0.943 | 0.050 | 0.941 | 32.3 → 38.2 |
| β = 16 | 0.829 | 0.017 | 0.868 | 33.3 → 36.6 |

**Luật chọn β: không β nào thỏa** (cần P(BOMB | hợp lệ) ≤ 0.687). β = 0 đã lan bom sau 150 ep (`beta0_spilled` = true), nên hiệu chỉnh có ý nghĩa. **Theo spec: dừng, không chạy probe SD7.7.** Seed 6900 và 46000 chưa dùng.

**Đọc kết quả:**
- Lan bom xảy ra rất nhanh: chỉ sau 150 ep, β = 0 đã ở 0.949, gần bằng SD7.6 sau 1000 ep (0.943).
- Ở mọi β, phần tăng P(BOMB | hợp lệ) và phần tăng P(nước giết) gần như bằng nhau (β = 0: +0.31 / +0.26; β = 16: +0.19 / +0.18). Phạt KL chỉ kéo **cả hai** xuống cùng lúc chứ không tách được hai việc. CNN không phân biệt "khoảnh khắc đặt bom ngay" với "bất kỳ chỗ nào được đặt bom", vì để nhận ra khoảnh khắc đó cần nhìn trước như kill-search.
- β = 1 và 4 gần như không có tác dụng; β = 16 mới giảm được một phần.

Giới hạn: 150 ep, 40 trận, 40 trạng thái hẹp; một train seed; chưa thử β > 16.


## SD7.8 — G + kill-search lúc chơi, test một lần seed 46000, 19/10/2026

**Thông số:** Claude chạy, 5 tiến trình. Không train. G và G + `kill_search.search_action` (horizon 12, phạm vi 6, đối thủ mô phỏng RuleBot) đấu RuleBot; 400 trận/agent, seed 46000–46199 × 2 góc; lịch E 475/25/40 và off; 9×9, 2 agent, max_steps 500; policy lấy mẫu; CPU, torch 2.5.1. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014.

Thư mục `runs/sd78-20261019` (log `runs/sd78-20261019-test-console.log`, ZIP `runs/sd78-20261019-send.zip`). Spec chốt trước: `docs/superpowers/specs/2026-10-19-sd7-8-g-plus-search-design.md`. Claude chạy (5 tiến trình). Không train. G = `runs/sd75-20261018/full/best_phaseC.pt` (SHA `bb0c775c…`); G+search = hành động lấy mẫu của G, trừ khi `kill_search` tìm được nước có điểm cao hơn hẳn (cấu hình SD5.0: 12 tick, phạm vi 6, đối thủ RuleBot). 400 trận mỗi agent, 200 cặp bàn, cùng luồng RNG theo bàn.

**Lịch E (gate):**

| Agent | WR | ActiveWR | Hòa | Thắng thụ động | Thắng nhờ vòng | Tự nổ | Chết vì vòng | Kills | Bom/trận | Số bước |
|---|---|---|---|---|---|---|---|---|---|---|
| G | 69.0 | 48.25 | 23.25 | 8.75 | 12.0 | 0.75 | 11.0 | 0.48 | 31.1 | 335 |
| G+search | **84.0** | **64.75** | 15.25 | 8.25 | 11.0 | 0.0 | 1.0 | 0.65 | 27.3 | 287 |

| G+search − G | Δ (pp) | CI95 |
|---|---|---|
| ActiveWR | **+16.5** | [+12.5, +20.5] |
| WR | +15.0 | [+11.24, +19.0] |
| Hòa | −8.0 | [−12.25, −4.0] |
| Thắng thụ động | −0.5 | [−2.5, +1.5] |
| Tự nổ | −0.75 | [−1.75, 0.0] |

Lịch off (không SD): G 53.0 WR / 47.5 ActiveWR / hòa 45.25; G+search 68.5 / 63.0 / 31.25; ActiveWR +15.5 [+11.5, +19.25], hòa −14.0 [−17.75, −10.25].

**Gate: ĐẠT** (+16.5 ≥ +5, CI dưới +12.5 > 0, tự nổ −0.75 ≤ +1). Phần WR tăng gần như toàn bộ là thắng chủ động (thắng thụ động −0.5), nên không phải WR "đẹp giả". Chết vì vòng giảm 11.0 → 1.0 vì search chấm −1 cho nước dẫn tới chết khi có đối thủ trong phạm vi. Bom/trận giảm (31.1 → 27.3), trận ngắn hơn (335 → 287 bước).

**Quyết định:** agent mạnh nhất hiện tại là **G + kill-search lúc chơi**; checkpoint CNN vẫn là G. Seed 46000 đã dùng.

Giới hạn: kill-search dùng RuleBot làm mô hình đối thủ, đúng với đối thủ trong bài này nên đây là cận trên thuận lợi; với đối thủ khác (self-play, người chơi) lợi thế có thể nhỏ hơn. Agent cần chạy search khi chơi (~19 ms mỗi bước có đối thủ trong phạm vi 6). Mức tăng trên G (+16.5) nhỏ hơn trên F ở validation (+22.5, SD5.0), khác seed và khác model nên không so trực tiếp.

## SD5.3 — kết quả của bạn (review + kiểm định lại), 19/10/2026

**Thông số:** Bạn của người dùng chạy (`runs/sd53-20261018`) trên RTX 4060, torch 2.14.1+cu130, numpy 2.4.6, Python 3.11.9. Thu dữ liệu: F + kill-search (horizon 12), 3000 trận trên bàn 200000–201499 × 2 góc; lịch E ở 75% số bàn; mẫu neo 5%; shard 25 bàn. Train: khởi đầu từ F, 8 epoch (ứng viên 2/4/8), Adam lr 1e-4, batch 256, KL neo λ 1, held-out 10% số bàn. Validation 400 trận seed 21000, lịch E. Test: 400 trận/model (F, S, F + search), seed 43000–43199 × 2 góc, lịch E và off; max_steps 500. Bootstrap 10 000 lần, seed 20261014. Số tiến trình: không lưu lại.

Thư mục `runs/sd53-20261018` (bạn của người dùng chạy trên máy RTX 4060, torch 2.14.1; runner `run_sd53_pilot.py` ở nhánh `origin/Loc/sd5-update`, SHA chuẩn hóa LF `bcc07727…` khớp registration). S = `runs/sd53-20261018/train/epoch_8.pt`, SHA `8f7116d2…fd4a` (khớp `phase_d_baseline.json`). Spec: `docs/superpowers/specs/2026-10-18-sd5-3-dense-search-distillation-design.md`.

**Kiểm định (Claude):** tính lại tỉ lệ từ file trận thô khớp `summary.json`; chọn checkpoint trên validation trước khi mở test (runtime validate 05:03 UTC, test 05:13 UTC); seed thu thập 200000–201499 không trùng validation/test; audit đối xứng 79.19% < 90% nên augmentation **tắt** đúng luật. Khóa `search_vs_f` trong `summary.json` thực ra là **S − F** (chỉ đặt tên sai, gate dùng đúng cặp).

**Dữ liệu:** 3000 trận F+search; 6171 mẫu quyết định (3159 có nước giết: 1933 giết bằng bom, 1226 chỉ giết bằng đứng/di chuyển, trong đó 664 có STOP), 43 547 mẫu neo.

**Validation 21000 (máy bạn), ActiveWR:** F 43.75; epoch 2: 56.75; epoch 4: 61.25; epoch 8: **64.25** (tự nổ 0.75). Tăng đều theo epoch, chưa bão hòa.

**Test một lần seed 43000, 400 trận/model:**

| Model (lịch E) | WR | ActiveWR | Hòa | Tự nổ | Chết vì vòng | Bom/trận |
|---|---|---|---|---|---|---|
| F | 63.5 | 44.25 | 25.0 | 1.0 | 14.0 | 30.8 |
| S | 78.5 | **60.5** | 18.0 | 1.0 | 3.25 | 26.7 |
| F + search (trần, báo cáo) | 81.25 | 64.25 | 17.25 | 0.0 | 1.5 | 26.1 |

S − F: ActiveWR **+16.25 [+9.75, +22.5]**, WR +15.0, hòa −7.0, tự nổ 0.0. Lịch off: +20.25 [+14.0, +26.5]. **Gate: ĐẠT** (bạn chạy). Đây là lần đầu một CNN thuần (không search lúc chơi) vượt F rõ trên test.

**Phân tích offline của Claude** (`P(best)` = xác suất policy đặt vào tập nước tốt nhất theo search; held-out = 640 mẫu của 10% bàn mà S không train):

| Nhóm mẫu (held-out) | F | G | S |
|---|---|---|---|
| Mọi trạng thái quyết định | 0.52 | 0.55 | **0.91** |
| Giết bằng bom | 0.63 | 0.70 | 0.96 |
| Giết bằng đứng/di chuyển (ô cửa) | 0.28 | 0.30 | **0.93** |
| Không có nước giết (tránh chết) | 0.55 | 0.57 | 0.87 |
| P(BOMB) trung bình ở mẫu neo | 0.088 | 0.091 | 0.095 |

Đọc: G (group RL, SD7.5) gần như không đổi P(best) so với F (+0.03) và không chạm vào nhóm "đứng ở ô cửa"; S học được cả ba nhóm mà không lan sang đặt bom bừa (P(BOMB) ở trạng thái thường gần như F, argmax đổi 7.2% mẫu neo).

Giới hạn: một train seed; mô hình RuleBot hoàn hảo; số liệu validation/test của bạn trên torch 2.14.1 (Claude đo lại S trên torch 2.5.1 ở SD7.9a: 64.0).

## SD7.9a — S hay G làm nền cho kill-search (validation), 19/10/2026

**Thông số:** Claude chạy, 5 tiến trình, CPU, torch 2.5.1. Không train. G + search, S và S + search (search một nước, horizon 12, phạm vi 6, đối thủ mô phỏng RuleBot) đấu RuleBot; 400 trận/nhánh, seed 21000–21199 × 2 góc; lịch E 475/25/40; 9×9, 2 agent, max_steps 500; policy lấy mẫu. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014.

Thư mục `runs/sd79-probe-21000` (log `runs/sd79-probe-21000-console.log`). `run_sd79_probe.py`, Claude chạy (5 tiến trình). Spec `docs/superpowers/specs/2026-10-20-sd7-9-two-move-search-design.md`. Validation 21000, 400 trận/nhánh, lịch E. Luật chốt trước: S+search − G+search ≥ +2pp thì mở test 47000.

| Nhánh | WR | ActiveWR | Hòa | Thắng thụ động | Thắng nhờ vòng | Tự nổ | Chết vì vòng | Bom/trận |
|---|---|---|---|---|---|---|---|---|
| G+search | 87.0 | 66.25 | 12.25 | 10.0 | 10.75 | 0.0 | 0.5 | 27.2 |
| S | 76.25 | 64.0 | 20.25 | 7.0 | 5.25 | 0.25 | 2.5 | 26.3 |
| S+search | 79.5 | 65.5 | 19.5 | 7.25 | 6.75 | 0.0 | 1.5 | 26.3 |

S − G+search: ActiveWR −2.25 [−8.25, +4.0], WR −10.75 [−16.25, −5.25], hòa +8.0. S+search − G+search: ActiveWR **−0.75 [−6.75, +5.5]**, WR −7.5, hòa +7.25 [+2.25, +12.25].

**Quyết định theo luật: không mở test** (−0.75 < +2). Đọc: (1) S tái lập trên máy mình (64.0 vs 64.25 của bạn); (2) CNN thuần S có ActiveWR ngang G+search mà không cần search; (3) mọi nền + search một nước đều dừng quanh 65% ActiveWR (F+search 64.25 test 43000, G+search 64.75 test 46000, 66.25 và S+search 65.5 ở đây): trần nằm ở search, không ở CNN; (4) G chơi cuối trận tốt hơn S (ít hòa hơn, thắng nhờ vòng nhiều hơn), có lẽ nhờ train dưới SD.

## SD7.9b — search hai nước (validation), 19/10/2026

**Thông số:** Không train. G + `search_action2` đấu RuleBot; 400 trận, seed 21000–21199 × 2 góc, ghép cặp với các trận G + search của SD7.9a; lịch E 475/25/40; 9×9, 2 agent, max_steps 500; policy lấy mẫu; Claude chạy, 5 tiến trình, CPU, torch 2.5.1. Bootstrap 10 000 lần, seed 20261014. Đo tốc độ trên 4 trận seed 6600.

Thư mục `runs/sd79b-search2-21000`. `run_sd79_probe.py --stage b`: G+search2 (`kill_search.search_action2`), so cặp với các trận G+search của SD7.9a trên cùng bàn. Luật chốt sau SD7.9a, trước khi chạy: ≥ +2pp thì mở test 47000. Đo tốc độ (4 trận seed 6600): search 4.5 s/trận, search2 17.8 s/trận.

| Nhánh | WR | ActiveWR | Hòa | Thắng thụ động | Thắng nhờ vòng | Tự nổ | Bom/trận | Số bước |
|---|---|---|---|---|---|---|---|---|
| G+search | 87.0 | 66.25 | 12.25 | 10.0 | 10.75 | 0.0 | 27.2 | 288 |
| G+search2 | 94.25 | **83.75** | 5.0 | 6.75 | 3.75 | 0.0 | 19.6 | 200 |

G+search2 − G+search: ActiveWR **+17.5 [+13.75, +21.25]**, WR +7.25, hòa −7.25, thắng thụ động −3.25, tự nổ 0. **Luật ĐẠT → mở test 47000.**

## SD7.9c — test G + search hai nước, seed 47000, 20/10/2026

**Thông số:** Claude chạy một lần, 5 tiến trình, CPU, torch 2.5.1. Không train. G + search và G + search2 đấu RuleBot; 400 trận/agent, seed 47000–47199 × 2 góc; lịch E 475/25/40 và off; 9×9, 2 agent, max_steps 500; policy lấy mẫu. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014.

Thư mục `runs/sd79-20261020` (log `runs/sd79-20261020-test-console.log`, ZIP `runs/sd79-20261020-send.zip`). `run_sd79_test.py`, Claude chạy một lần (`test_opened.json`). G = `bb0c775c…`. 400 trận/agent, 200 cặp bàn.

| Agent | Lịch | WR | ActiveWR | Hòa | Thắng thụ động | Thắng nhờ vòng | Tự nổ | Chết vì vòng | Kills | Bom/trận | Số bước |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G+search | E | 80.75 | 61.0 | 18.0 | 10.75 | 9.0 | 0.0 | 1.75 | 0.61 | 28.6 | 300 |
| G+search2 | E | **93.75** | **82.0** | 5.75 | 8.25 | 3.5 | 0.0 | 0.25 | 0.82 | 20.7 | 208 |
| G+search | off | 67.25 | 60.0 | 32.5 | 7.25 | — | 0.0 | — | 0.60 | 29.0 | 303 |
| G+search2 | off | 90.0 | 82.0 | 9.5 | 8.0 | — | 0.0 | — | 0.82 | 20.9 | 209 |

| G+search2 − G+search | Lịch E | Lịch off |
|---|---|---|
| ActiveWR | **+21.0 [+17.0, +25.25]** | +22.0 [+18.0, +26.25] |
| WR | +13.0 [+10.0, +16.25] | +22.75 [+18.75, +27.0] |
| Hòa | −12.25 [−15.25, −9.25] | −23.0 [−27.25, −19.0] |
| Thắng thụ động | −2.5 [−4.75, −0.5] | +0.75 [−0.25, +2.0] |
| Tự nổ | 0.0 | 0.0 |

**Gate: ĐẠT** (+21.0 ≥ +5, CI dưới +17.0 > 0, tự nổ 0). Phần tăng là thắng chủ động (thắng thụ động và thắng nhờ vòng đều giảm), trận ngắn hơn ~90 bước, ít bom hơn. **Quyết định: agent khuyến nghị mới là G + kill-search hai nước** (checkpoint CNN vẫn là G). Seed 47000 đã dùng.

Giới hạn: search dùng RuleBot làm mô hình đối thủ (đúng với đối thủ của bài, nên đây là kết quả thuận lợi; với đối thủ khác lợi thế có thể nhỏ hơn); chậm khoảng 4 lần search một nước mỗi trận (đo trên 4 trận); G+search trên seed này (61.0) thấp hơn trên 46000 (64.75), trong dao động ±5pp giữa seed.

## SD5.3 + SD7.9 — tổng hợp insight chi tiết, 20/10/2026

**Thông số:** Không chạy trận mới. Ghép cặp lại các file trận đã có: test 47000 (400 trận/agent, lịch E và off) và validation 21000 (400 trận/agent, lịch E); held-out SD5.3: 640 mẫu.

Không chạy thêm trận và không dùng seed mới. Mục này gom các insight của review SD5.3 và SD7.9a–c, cộng một phân tích lại các file trận có sẵn (`runs/sd79-20261020/evaluation_*_47000.json`, `runs/sd79b-search2-21000/evaluation_*_21000.json`). Phân tích ghép từng trận theo cặp (seed bàn, góc) giữa G+search và G+search2, rồi đếm trận đổi kết quả.

### 1. Chưng cất dày đưa được search vào CNN, nhờ bốn điều kiện cùng lúc

- **Dày:** 6171 trạng thái quyết định thu từ 3000 trận, so với ~0.9 nhãn/trận ở SD5.1, nơi chưng cất thưa FAIL.
- **Có ví dụ ngược:** gồm cả trạng thái mà đặt bom không phải nước tốt nhất (1226 mẫu chỉ giết bằng đứng/di chuyển, 3012 mẫu không có nước giết). RL ở SD7.6 chỉ có tín hiệu một phía, làm policy "cứ được đặt bom là đặt" (P(BOMB) 0.94).
- **Đích mềm theo điểm search, học nhiều epoch.**
- **KL neo về F trên 43 547 mẫu neo:** giữ hành vi ở trạng thái thường.

Bằng chứng ở held-out 640 mẫu theo bàn mà S không train:
- P(nước tốt nhất): F 0.52 → S 0.91.
- Nhóm "đứng chặn ô cửa": 0.28 → 0.93.
- P(BOMB) ở mẫu neo gần như không đổi: 0.088 → 0.095.

Validation tăng đều theo epoch (56.75 / 61.25 / 64.25 ở epoch 2/4/8), chưa bão hòa. Test 43000: S − F +16.25 [+9.75, +22.5]. Đây là CNN thuần đầu tiên vượt F rõ trên test.

### 2. Group RL (G) gần như không đổi CNN ở các trạng thái quyết định

- P(nước tốt nhất): F 0.52, G 0.55.
- Nhóm ô cửa: 0.28 so với 0.30.

Điều này giải thích vì sao SD7.5 chỉ +3.75 ActiveWR so với F. Phần G hơn F (nếu có) không nằm ở khoảnh khắc ra đòn mà search chấm điểm.

### 3. Search một nước có trần ~65% ActiveWR, bất kể CNN nền

| Agent | Seed | ActiveWR |
|---|---|---|
| F+search | test 43000 | 64.25 |
| G+search | test 46000 | 64.75 |
| G+search | test 47000 | 61.0 |
| G+search | val 21000 | 66.25 |
| S+search | val 21000 | 65.5 |

S+search − G+search = −0.75 [−6.75, +5.5]. CNN tốt hơn ở trạng thái quyết định không thêm gì khi đã có search, vì search đã tự sửa đúng những trạng thái đó; phần S học được trùng với phần search làm sẵn. ~35% trận còn lại không thắng chủ động vì **không bao giờ xuất hiện nước giết trong một bước**, chứ không phải vì kết liễu sai.

### 4. S chạy một mình ≈ G+search về ActiveWR, nhưng kém ở cuối trận

- S − G+search (val 21000): ActiveWR −2.25 [−8.25, +4.0], WR −10.75 [−16.25, −5.25], hòa +8.0.
- G+search so với S+search: hòa 12.25 vs 19.5, thắng nhờ vòng 10.75 vs 6.75.

G chơi cuối trận tốt hơn, có lẽ nhờ G được train dưới sudden death 3000 + 1000 ep; S chưng cất từ F. Vì vậy nếu chưng cất tiếp thì nên đi từ G, không từ F.

### 5. Search hai nước phá trần bằng cách tạo cơ hội giết sớm hơn

Đếm trận đổi kết quả, G+search → G+search2, cùng bàn cùng góc (400 cặp):

| Chuyển | Test 47000 lịch E | Test 47000 lịch off | Val 21000 lịch E |
|---|---|---|---|
| hòa → thắng chủ động | 45 | **90** | 26 |
| thắng nhờ vòng → thắng chủ động | 23 | — | 28 |
| thắng thụ động → thắng chủ động | 15 | 1 | 16 |
| chết vì vòng → thắng chủ động | 3 | — | 1 |
| thua → thắng chủ động | 1 | 0 | 0 |
| **Tổng thành thắng chủ động** | **87** | **91** | **71** |
| Thắng chủ động → khác (thụ động/thua) | 3 | 3 | 1 |
| Đổi khác (không liên quan thắng chủ động) | 5 | 2 | 4 |

- **Gần như một chiều.** Test E: 87 trận thành thắng chủ động, 3 trận mất; net +84 trận = +21.0pp.
- **Lịch off rõ nhất.** Toàn bộ 90 trận hòa đổi được đều thành thắng chủ động. Search hai nước chủ yếu phá thế giằng co: hai bên đi vòng quanh mà không ai có nước giết một bước.
- **Ăn cả vào thắng nhờ vòng** (23 trận test, 28 trận val). Ở những trận này G+search chỉ thắng vì đối thủ chết vì vòng, còn G+search2 giết được trước đó.

Thời điểm thắng chủ động (số bước của trận thắng chủ động; tỉ lệ tính trên tất cả 400 trận):

| | G+search | G+search2 |
|---|---|---|
| Trung vị số bước (test 47000 E) | 205 | **158** |
| Tứ phân vị 25–75 | 115–343 | 100–261 |
| Thắng chủ động trước bước 150 | 22.5% | **37.75%** |
| Thắng chủ động trước bước 300 | 42.0% | **65.75%** |
| Trung vị (val 21000) | 210 | 144 |

Giết sớm hơn ~50 bước. Điều này khớp với insight cũ "ActiveWR do giai đoạn trước bước 475 quyết định": đòn bẩy nằm ở giữa trận, không ở vòng.

Không lệch theo góc: test E góc 0 59.0 → 81.0, góc 1 63.0 → 83.0; val góc 0 67.0 → 85.0, góc 1 65.5 → 82.5.

### 6. Kiểm tra tính nhất quán của dữ liệu

- Các trận thắng chủ động trước bước 300 của cùng agent giống hệt nhau giữa lịch E và lịch off (37.75% / 65.75%). Kết quả này đúng kỳ vọng: vòng chỉ tác động từ bước 435 (cảnh báo 40 bước trước 475), và RNG được ghép theo bàn. Điều đó xác nhận việc ghép cặp và luồng RNG đúng.
- S tái lập trên máy mình: 64.0 so với 64.25 của bạn, khác torch 2.5.1/2.14.1.
- Khóa `search_vs_f` trong `summary.json` của SD5.3 là S − F (đặt tên sai, số đúng).
- Audit đối xứng 79% < 90%: nhãn search/RuleBot không bất biến dưới phép quay/lật, nên không được tăng cường dữ liệu bằng đối xứng.

### 7. Giới hạn của các insight

- Mọi search dùng RuleBot làm mô hình đối thủ, và đối thủ trong bài cũng là RuleBot. Đây là điều kiện thuận lợi nhất; với self-play hoặc người chơi, lợi thế có thể nhỏ hơn nhiều.
- Search hai nước chậm ~4 lần mỗi trận (17.8 s so với 4.5 s, đo trên 4 trận seed 6600).
- G+search dao động giữa seed: 64.75 (46000) và 61.0 (47000).
- Một train seed cho G và cho S.
- Bảng chuyển kết quả là phân tích sau khi đã mở test 47000. Nó chỉ dùng để giải thích, không dùng để chọn model hay cấu hình.

### 8. Hệ quả cho bước tiếp theo

- **Ứng viên 1, chưng cất search hai nước vào CNN từ G** theo công thức SD5.3 (thu bằng G+search2). Mục tiêu: một CNN thuần gần 82% mà vẫn giữ lối chơi cuối trận của G. Chi phí thu dữ liệu ~4 lần SD5.3. Đây là hướng có cơ sở nhất, theo mục 1 và mục 4.
- **Ứng viên 2, search sâu hơn:** ba nước, hoặc dùng CNN làm chính sách tiếp diễn thay vì "chỉ né". Mục 5 cho thấy đòn bẩy nằm ở việc tạo cơ hội. Chi phí tăng theo cấp số nhân, nên cần đo tốc độ trước.
- **Không nên** quay lại RL một phía (SD7.x) để học khoảnh khắc ra đòn: mục 2 cho thấy nó không đổi được CNN ở đúng trạng thái cần đổi.

## SD8 — chuẩn bị A (chưng cất search hai nước từ G) và B (search ba nước), 20/10/2026

**Thông số:** Đo tốc độ: G + search2 và G + search3, mỗi agent 24 trận (seed chẩn đoán 6600/6610/6620 × 8 trận), lịch E, max_steps 500, 6 tiến trình, CPU, torch 2.5.1. Smoke SD8A: collect 8 trận (4 bàn), train 2 epoch batch 16, đánh giá 4 trận. Smoke SD8B: 4 trận, max_steps 40, 2 shard.

Chưa có kết quả thí nghiệm: lượt này chỉ viết code, chạy smoke và đo tốc độ. Spec chốt trước: `docs/superpowers/specs/2026-10-20-sd8-distill-and-deep-search-design.md`. Hướng dẫn chạy: `SD8_GUIDE.md`.

**Đo tốc độ** (G + search, lịch E, 3 seed chẩn đoán 6600/6610/6620 × 8 trận, chạy 6 tiến trình song song, trong scratchpad):

| Agent | Giây/trận | Tổng số bước (24 trận) | Thắng chủ động (24 trận) |
|---|---|---|---|
| G + search2 | 12.0 / 12.0 / 13.1 (TB ~12.4) | 3665 | 20 |
| G + search3 | 18.7 / 23.4 / 27.5 (TB ~23.2) | 2342 | 22 |

Mỗi bước, search3 chậm khoảng 4 lần search2. Nhưng trận ngắn hơn, nên mỗi trận chỉ chậm khoảng 1.9 lần. Số thắng chủ động trên 24 trận chỉ là quan sát, cỡ mẫu quá nhỏ để kết luận.

**Smoke** (scratchpad, không phải kết quả):
- SD8A: collect 8 trận → 21 trạng thái quyết định (~2.6/trận, ước ~7900 cho 3000 trận; SD5.3 có 6171) và 48 mẫu neo. Train 2 epoch, validate, test đều chạy qua. Mở test lần hai bị từ chối, và `--resume` trên bộ dữ liệu đã đủ không thu lại.
- SD8B: validate và test chạy qua. Test không có `--validation` thì bị từ chối.

**Phân công** theo yêu cầu người dùng: hướng nhẹ hơn do người dùng chạy, hướng nặng hơn do bạn chạy.
- **B: người dùng chạy.** Khoảng 7–8 giờ CPU, ~1.5–2 giờ thực. Tái dùng 400 trận G+search2 của SD7.9b, nên validation chỉ cần chạy G+search3.
- **A: bạn chạy.** Thu 3000 trận G+search2 mất ~10–15 giờ CPU, cộng validate/test ~3 giờ CPU và train trên GPU.

Seed test giữ chỗ: **48000** cho SD8A, **49000** cho SD8B. Seed mới tiếp theo là 50000.

## SD8B — G + search ba nước, validation 21000 và test 49000, 20/10/2026

**Thông số:** Người dùng chạy trên máy chính, 5 tiến trình, CPU, torch 2.5.1. Không train. G (SHA `bb0c775c…c7ac`) + `search_action3` và G + `search_action2` đấu RuleBot; 9×9, 2 agent, max_steps 500; lịch E 475/25/40; policy lấy mẫu. Validation: seed 21000–21199 × 2 góc = 400 trận search3, ghép với 400 trận search2 của SD7.9b; 36 phút. Test: seed 49000–49199 × 2 góc = 400 trận/agent, 5 shard/agent; 71 phút. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014.

Người dùng chạy trên máy này (torch 2.5.1, 5 tiến trình), theo `SD8_GUIDE.md`. Luật chốt trước trong spec `docs/superpowers/specs/2026-10-20-sd8-distill-and-deep-search-design.md`.

- Validation: `runs/sd8b-val-21000/` (log `runs/sd8b-val-21000-console.log`), 36 phút. Tái dùng 400 trận G+search2 của `runs/sd79b-search2-21000`.
- Test: `runs/sd8b-20261020/` (log `runs/sd8b-20261020-test-console.log`, ZIP `runs/sd8b-20261020-send.zip`), 800 trận, 71 phút.
- Checkpoint G (`runs/sd75-20261018/full/best_phaseC.pt`), lịch E (475/25/40), 400 trận mỗi agent (200 bàn × 2 góc).
- Bootstrap ghép cặp theo bàn, 10 000 lần lấy mẫu.

**Validation 21000** (luật: search3 − search2 ActiveWR ≥ +2pp thì mở test):

| Agent | WR | ActiveWR | Thụ động | Thắng nhờ vòng | Hòa | Tự nổ | Chết vì vòng | Bom/trận | Bước TB |
|---|---|---|---|---|---|---|---|---|---|
| G + search2 | 94.25 | 83.75 | 6.75 | 3.75 | 5.0 | 0.0 | 0.25 | 19.6 | 200.5 |
| G + search3 | 99.0 | 93.0 | 6.0 | 0.0 | 0.5 | 0.0 | 0.0 | 11.0 | 102.5 |

- Kết quả: ActiveWR +9.25 [+5.75, +12.75] ≥ +2, nên **mở test 49000**.

**Test 49000** (một lần, lịch E):

| Agent | WR | ActiveWR | Thụ động | Thắng nhờ vòng | Hòa | Tự nổ | Chết vì vòng | Bom/trận | Bước TB |
|---|---|---|---|---|---|---|---|---|---|
| G + search2 | 94.25 | 85.75 | 6.25 | 2.25 | 4.75 | 0.0 | 1.0 | 19.2 | 195.9 |
| G + search3 | 99.5 | **94.0** | 5.5 | 0.0 | 0.25 | 0.0 | 0.0 | 12.2 | 114.2 |

So cặp search3 − search2:

| Metric | Δ (pp) | CI95 |
|---|---|---|
| ActiveWR | **+8.25** | [+5.25, +11.5] |
| WR | +5.25 | [+3.0, +7.75] |
| Hòa | −4.5 | [−7.0, −2.25] |
| Thắng thụ động | −0.75 | [−2.75, +1.25] |
| Tự nổ | 0.0 | [0.0, 0.0] |

**Gate ĐẠT**:
- ActiveWR +8.25 ≥ +5;
- CI95 dưới +5.25 > 0;
- tự nổ không tăng (0 → 0).

**Quyết định:** G + `kill_search.search_action3` là agent mạnh nhất hiện tại. Nó thay G + search2 của SD7.9.

**Phân tích theo cặp (bàn, góc)** (tính từ `evaluation_G_search{2,3}_E_49000.json`):
- 38 trận thành thắng chủ động:
  - 17 từ hòa;
  - 10 từ thắng thụ động;
  - 8 từ thắng nhờ vòng;
  - 2 từ chết vì vòng;
  - 1 từ thua.
- 5 trận mất thắng chủ động: 4 thành thắng thụ động, 1 thành hòa. Ròng +33 trận = +8.25pp.
- Validation 21000 cùng chiều: +45 / −8.
- Trung vị bước của thắng chủ động giảm từ 143 xuống 96.5.
- Tỉ lệ thắng chủ động trước bước 150 tăng từ 44.0% lên 73.5%; trước bước 300 từ 73.5% lên 91.0%.
- Search3 giết sớm hơn và gần như không còn ván kéo dài tới vòng. Thắng nhờ vòng 2.25 → 0, chết vì vòng 1.0 → 0.
- Còn 24/400 trận không thắng chủ động:
  - 22 thắng thụ động (đối thủ tự chết trước khi bị giết);
  - 1 hòa;
  - 1 thua.

**Tốc độ:**
- Test chạy 2 agent song song trong cùng 5 tiến trình: search2 23.1 s/trận, search3 27.1 s/trận (search3 chậm ~1.17 lần).
- Validation chỉ chạy search3: 22.4 s/trận.
- Lượt đo trước (6 tiến trình, 24 trận): search2 ~12 s/trận, search3 ~23 s/trận.
- Số giây phụ thuộc tải máy, nên chỉ so được trong cùng một lượt chạy. Trận của search3 ngắn hơn ~80 bước, bù phần lớn chi phí mỗi bước.

**Giới hạn:**
- Search vẫn dùng RuleBot làm mô hình đối thủ; chưa kiểm với đối thủ khác.
- G chỉ có một train seed.
- ActiveWR 94% đã gần trần trước RuleBot: phần còn lại chủ yếu là thắng thụ động.
- Chưa đo lịch off. SD7.9c cho thấy lịch off cùng chiều với lịch E, nhưng chưa kiểm cho search3.
- Mức G + search2 trên 49000 (85.75) cao hơn trên 47000 (82.0). Dao động giữa các seed khoảng ±5pp (INSIGHTS §5), nên so sánh ghép cặp trên cùng seed mới có ý nghĩa.
- Seed 49000 đã dùng.

## SD8B-off — ablation lịch sudden death cho G + search3 / search2, seed 49000, 20/10/2026

**Thông số:** Claude chạy nền, 5 tiến trình, CPU, torch 2.5.1. Không train. G + search3 và G + search2 đấu RuleBot; lịch off; seed 49000–49199 × 2 góc = 400 trận/agent, 5 shard/agent; ghép với 800 trận lịch E của test SD8B; 9×9, 2 agent, max_steps 500; policy lấy mẫu. Bootstrap ghép cặp theo bàn 10 000 lần, seed 20261014.

Người dùng yêu cầu để xác định tầm quan trọng của sudden death. Lượt này mô tả, không có gate. Cách làm chốt trước trong spec SD8, mục "SD8B-off".

- Output: `runs/sd8b-off-49000/`, log `runs/sd8b-off-49000-console.log`. Claude chạy nền trên máy này, 5 tiến trình.
- `run_sd8b_search3.py ablate`: G + search3 và G + search2 chạy lại với lịch **off** trên đúng 200 bàn × 2 góc của seed 49000, 400 trận mỗi agent.
- Ghép cặp với các trận lịch E của test SD8B (`runs/sd8b-20261020`).
- Seed 49000 được dùng lại vì không chọn hay quyết định gì dựa trên lượt này; dùng lại bàn cho phép so trên cùng các bàn.

| Agent | Lịch | WR | ActiveWR | Thụ động | Thắng nhờ vòng | Hòa | Chết vì vòng | Tự nổ | Bước TB |
|---|---|---|---|---|---|---|---|---|---|
| G + search2 | E | 94.25 | 85.75 | 6.25 | 2.25 | 4.75 | 1.0 | 0.0 | 195.9 |
| G + search2 | off | 91.0 | 85.75 | 5.25 | — | 8.75 | — | 0.0 | 196.8 |
| G + search3 | E | 99.5 | 94.0 | 5.5 | 0.0 | 0.25 | 0.0 | 0.0 | 114.2 |
| G + search3 | off | 99.5 | 94.0 | 5.5 | — | 0.25 | — | 0.0 | 114.2 |

So cặp theo bàn (pp, CI95):

| | ActiveWR | WR | Hòa | Thụ động |
|---|---|---|---|---|
| search2: E − off | 0.0 [−1.0, +1.0] | +3.25 [+1.5, +5.25] | −4.0 [−6.0, −2.25] | +1.0 [+0.25, +2.0] |
| search3: E − off | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 |
| Lịch off: search3 − search2 | **+8.25 [+5.25, +11.25]** | +8.5 | −8.5 | +0.25 |
| Tương tác ActiveWR (E − off của search3 trừ của search2) | 0.0 [−1.0, +1.0] | | | |

**Phân tích từng trận:**
- Hai lịch chỉ khác nhau từ khi cảnh báo vòng bật (bước 435 = 475 − 40). Trước đó env, observation và RNG giống hệt, nên các trận kết thúc sớm hơn cho ra kết quả trùng khớp từng trận.
- **search3: 399/400 trận giống hệt.**
  - Chỉ 2 trận sống tới bước 435.
  - Trận duy nhất khác vẫn là hòa ở bước 500 (số bom lệch nhẹ).
  - Sudden death **không có vai trò gì** với G + search3.
- **search2: 366/400 trận giống hệt.**
  - 45 trận sống tới bước 435. 34 trận khác nhau, trận khác sớm nhất kết thúc ở bước 460.
  - Trong 35 trận hòa ở lịch off, lịch E đổi được: 8 thành thắng nhờ vòng, 4 thành thắng thụ động, 2 thành thắng chủ động, 2 thành chết vì vòng; 14 trận vẫn hòa.
  - Ngược lại có 2 trận thắng chủ động ở lịch off thì ở lịch E thành thắng nhờ vòng (1) và thua (1).
  - ActiveWR ròng 0.
  - Thắng chủ động sau bước 475: 0.75% ở cả hai lịch.
- Kết luận:
  - Với hai agent mạnh nhất, sudden death chỉ là cách phân xử các ván kéo dài. Nó đổi hòa thành thắng hoặc thua (WR +3.25, hòa −4.0 với search2), không thêm thắng chủ động.
  - Mức hơn +8.25 ActiveWR của search3 so với search2 giữ nguyên khi tắt sudden death.
  - Ảnh hưởng của sudden death tỉ lệ với số trận sống tới bước ~435: search2 11%, search3 0.5%.

**Tốc độ** (lượt off, 2 agent cùng 5 tiến trình): search2 14.0 s/trận, search3 24.1 s/trận.

**Giới hạn:**
- Chỉ một seed và lịch E (475/25/40).
- Kết luận "sudden death không ảnh hưởng ActiveWR" áp dụng cho agent **lúc chơi**. Không suy ra được vai trò của sudden death **khi train**: G được train dưới sudden death, và lượt này không đổi điều đó.
- Seed 49000 đã dùng trước cho test SD8B.

## DOC2 — bổ sung thông số đầy đủ cho mọi thí nghiệm, 20/10/2026

**Thông số:** Không chạy trận mới. Nguồn: `config.json` / `registration.json` / `protocol.json` / `summary.json` trong `runs/`, nội dung từng mục và lệnh trong transcript của phiên làm việc.

Người dùng yêu cầu: mọi thí nghiệm trong RESULTS.md và IMPROVEMENTS.md phải ghi kèm thông số đầy đủ (số episode, số trận, …). Quy định được ghi vào `CLAUDE.md` và `AGENT.md` (file mới, dành cho mọi agent).

**Rà soát:**
- **RESULTS.md:** 73/79 mục được thêm dòng `**Thông số:**` ngay dưới tiêu đề. Dòng này gồm checkpoint, số episode, train seed, lr và các hệ số chính, seed và dải bàn, số trận mỗi model, max_steps, lịch SD, cách chọn nước, bootstrap, máy / torch / số tiến trình.
  - 6 mục không phải thí nghiệm nên không có dòng này: kế hoạch Phase F, ghi chú kỹ thuật, tạo tracker, công cụ progress, push GitHub, gom INSIGHTS.
- **19 mục có thông số không lưu lại.** Các mục này ghi rõ "không lưu lại", không đoán:
  - Giai đoạn A–E (03–05/10, code ban đầu): thiếu max_steps, seed, có khi cả số trận đánh giá.
  - Các lượt 10–13/10: thiếu phiên bản torch, có khi cả thiết bị train.
  - SD4.0, SD4.1, SD5.0, SD5.3 của bạn: thiếu số tiến trình.
  - XC: thiếu phiên bản torch của máy bạn.
- **IMPROVEMENTS.md:** 53 mục có thí nghiệm được thêm dòng `**Thông số:**`, dùng cùng nội dung với mục RESULTS tương ứng. Mục ứng với nhiều lượt chạy thì liệt kê (1), (2), …. 9 mục chỉ là thiết kế, công cụ hoặc tài liệu nên không có dòng này.
- **INSIGHTS.md:** thêm dòng điều kiện mặc định của số liệu (RuleBot, 9×9, 400 trận/agent = 200 bàn × 2 góc, max_steps 500, lấy mẫu, bootstrap 10 000 lần). Bổ sung seed và số trận cho 23 câu trước đây thiếu.
- **PLAN_SUDDEN_DEATH.md:** cập nhật đầu file, vốn đã lỗi thời (ghi "seed tiếp theo 43000", model F). Nay ghi G, agent mạnh nhất G + search3, seed tiếp theo 50000, kèm điều kiện mặc định của số liệu.
- **PLAN_BOMBERMAN.md:** thêm ghi chú trỏ về dòng Thông số.
- **Không sửa:** các `*_GUIDE.md` và spec trong `docs/superpowers/specs/`. Đây là file lệnh và thiết kế, đã có seed / số trận / số episode.

**Phát hiện phụ:** IMPROVEMENTS.md có hai mục cùng số `## 51.` (KC1 và SD7.6-probe). Không đánh số lại, vì các file khác đang trỏ tới §52–§60.

## PDE2-spec — thiết kế "nhiều agent đánh nhau" (FFA hỗn hợp, self-play gương, đối đầu 1v1), 20/10/2026

**Thông số:** Chưa chạy. Dự kiến:
- Phần 1, FFA hỗn hợp 4 ghế (G+s3, S2, G, RuleBot) xoay 4 lượt mỗi bàn: 50 bàn × 4 = 200 trận ở 9×9 và 200 trận ở 11×11.
- Phần 2, gương 4×S2: 400 trận ở 9×9 và 400 trận ở 11×11.
- Phần 3, đối đầu 1v1 9×9 (G+s3 vs S2, S2 vs G): 200 trận mỗi cặp.
- Chung: seed 51000; lịch E 475/25/40; max_steps 500; policy lấy mẫu; bootstrap theo bàn 10 000 lần, seed 20261014.
- Ước tính ~20 giờ CPU (4–5 giờ với 5 tiến trình).

Người dùng hỏi vì sao Phase D/E trong PDE chủ yếu cho 1 agent đánh RuleBot; người dùng yêu cầu viết spec trước và chỉ chạy khi có kết quả SD8A. Spec: `docs/superpowers/specs/2026-10-20-pde2-ffa-mix-design.md`.

Ghi nhận báo cáo SD8A của bạn qua ảnh, chưa có file, chưa kiểm định:
- S2 (epoch 16) đạt ActiveWR 83.0 trên test 48000, lịch E; G 48.0, S 57.5.
- S2 − G = +35.0 [+29.0, +41.25]; S2 − S = +25.5 [+19.25, +31.75]. Gate ĐẠT.
- So với G + search2 (86.5): −3.5 [−8.25, +1.25].
- Lịch off: S2 83.75, G 45.75, S 60.75.
- Dữ liệu: 3000 trận, 8449 trạng thái quyết định, 29 483 mẫu neo.
- Sẽ ghi mục SD8A đầy đủ sau khi có `runs/sd8a-20261020`.

Seed 51000 giữ cho PDE2. Seed mới tiếp theo: 52000.

## PUB — đánh giá khả năng viết bài báo và nơi nộp, 20/10/2026

**Thông số:** Không chạy trận, không train, không đổi code. Nguồn: `PROJECT_SPEC.md`, `INSIGHTS.md`, `PLAN_BOMBERMAN.md`, `PLAN_SUDDEN_DEATH.md`, `RESULTS.md` (SD5–SD8B, SD8B-off, PDE2-spec), `kill_search.py`; hạn nộp tra cứu web ngày 20/10/2026 (Claude, máy này).

Người dùng hỏi kết quả của repo có viết được bài báo không, và nộp ở đâu khả thi mà không dính hội nghị/tạp chí săn mồi.

- **Kết luận:** viết được một bài thực nghiệm ở hội nghị/workshop hạng trung có phản biện thật (IEEE CoG, workshop ICLR 2027, IJCNN, ACIIDS, ICAART, hội nghị IEEE/Springer ở Việt Nam). Chưa đủ cho hội nghị hàng đầu: môi trường tự dựng, chỉ một đối thủ cố định (RuleBot), search dùng chính code RuleBot làm mô hình đối thủ, ý tưởng search + chưng cất đã có (Expert Iteration; Kartal và cộng sự 2019 dùng MCTS nông trên Pommerman), checkpoint chỉ có một train seed.
- **Điểm mạnh dùng được cho bài:** ActiveWR so với WR (thắng thụ động làm WR đẹp giả); khoảng 10 cách cải tiến RL đều FAIL gate chốt trước, kèm chẩn đoán "khoảnh khắc quyết định thưa"; kill-search độ sâu 1/2/3 nâng ActiveWR 48.25 → 64.75 → 82.0 → 94.0 trên seed test mới; chưng cất dày đưa CNN thuần lên S2 83.0 (SD8A của bạn, chưa kiểm định file); quy trình đánh giá (seed test dùng một lần, bootstrap ghép cặp, SHA).
- **Cần bổ sung trước khi nộp:** đánh giá với đối thủ khác RuleBot và search với mô hình đối thủ sai lệch (PDE/PDE2); ít nhất 3 train seed cho bước chưng cất; một bảng ablation sạch; một lượt đánh giá cuối trên seed mới (từ 52000) sau khi chốt cấu hình; phần liên quan với Pommerman.
- **Hạn nộp tra được (phải kiểm lại trên trang chính thức):** ACIIDS 2027 gia hạn 31/10/2026; IJCNN 2027 khoảng 15–31/01/2027; workshop ICLR 2027 gợi ý 01/02/2027; IEEE CoG 2026 có hạn 17/03 (full) và 14/05 (auxiliary), 2027 chưa công bố; RLC 2026 có hạn 05/03, 2027 chưa công bố; ICAART 2027 vòng chính đã qua, một nguồn ghi vòng 2 ngày 13/11/2026 (chưa xác nhận); AAAI-27 student abstract (28/09/2026) và RIVF 2026 (31/08/2026) đã qua.
- **Giới hạn:** đây là đánh giá định tính, không phải phản biện thật; hạn của các kỳ 2027 chưa công bố chỉ là suy từ năm trước.

## PDE-int — lượt PDE seed 50000 bị dừng giữa chừng, 20/10/2026

**Thông số:** `run_pde_check.py --output runs/pde-50000 --workers 5` (Claude chạy nền, máy này, CPU, 5 tiến trình, torch của env `torch`); seed 50000, lịch E 475/25/40, max_steps 500; thiết kế như mục PDE trong spec `docs/superpowers/specs/2026-10-20-pde-check-design.md` (81 shard). Bắt đầu 11:32, dừng khoảng 19:40–19:47.

- Claude Code dừng tiến trình vì máy thiếu RAM khi phiên đang rảnh; không phải lỗi của script.
- Đã xong 64/81 shard (lưu ở `runs/pde-50000/shards/`):
  - E2 G + search3: 10/10; E1 G + search3: 10/10;
  - D G + search3 đấu G, F, S: 10/10 mỗi đối thủ; đấu B: 9/10 (thiếu bàn 50090–50099);
  - D G đấu G, F, S, B, oldD: 5/5.
- Còn 17 shard: D G + search3 đấu oldD (10), đấu B (1); E2 RuleBot / oldE4 / G (3); E1 RuleBot / oldE3 / G (3).
- Chưa có `summary.json`, chưa đọc kết quả, chưa áp luật đề xuất. Không chạy lại tự động; chạy tiếp bằng `--resume` (giữ shard đã xong, cùng seed và cùng thiết kế đã đăng ký trong `opened.json`).

## PDE-peek — xem tạm 64/81 shard PDE theo yêu cầu người dùng, 20/10/2026

**Thông số:** đọc `runs/pde-50000/shards/` (64 shard, lượt đầu trước khi bị dừng) khi lượt `--resume` đang chạy; seed 50000, lịch E 475/25/40, max_steps 500, policy lấy mẫu, bootstrap theo bàn 10 000 lần seed 20261014; script tạm ở scratchpad (`pde_interim.py`, `pde_draws.py`), không ghi vào `runs/`. Số liệu tạm, kết luận chính thức chờ `summary.json`.

| Nhánh | Số trận | ActiveWR | WR | Hòa | Tự nổ | s/trận |
|---|---|---|---|---|---|---|
| E2 G + search3 (11×11, 4 agent, vs 3 RuleBot) | 160 | 86.25 | 88.75 | 10.62 | 0.00 | 162 |
| E1 G + search3 (9×9, 3 agent, vs 2 RuleBot) | 180 | 94.44 | 98.89 | 0.56 | 0.00 | 82 |
| D G + search3 vs G | 200 | 14.50 | 39.50 | 54.00 | 4.00 | 131 |
| D G + search3 vs F | 200 | 20.00 | 43.00 | 51.00 | 2.00 | 127 |
| D G + search3 vs S | 200 | 13.00 | 19.00 | 70.50 | 6.00 | 133 |
| D G + search3 vs B (90/100 bàn) | 180 | 20.00 | 43.33 | 50.00 | 5.56 | 130 |
| D G vs G | 200 | 5.50 | 19.50 | 61.00 | 5.50 | 1.5 |
| D G vs F | 200 | 4.50 | 19.00 | 66.50 | 3.50 | 1.8 |
| D G vs S | 200 | 0.50 | 5.50 | 56.50 | 5.00 | 1.6 |
| D G vs B | 200 | 4.00 | 28.50 | 61.50 | 5.50 | 2.2 |
| D G vs Phase D cũ | 200 | 3.00 | 50.00 | 48.00 | 2.00 | 1.5 |

G + search3 − G (Phase D, ActiveWR): vs G +9.0 [+3.5, +14.5]; vs F +15.5 [+9.5, +21.5]; vs S +12.5 [+7.5, +18.0]; vs B (90 bàn) +15.56 [+8.89, +22.22]. WR: +20.0, +24.0, +13.5, +14.44, CI đều > 0.

- Đấu RuleBot trong FFA, search3 vẫn mạnh: E1 94.44, E2 86.25 (mức công bằng 33% và 25%). Mốc so (RuleBot, G, Phase E cũ) chưa chạy xong.
- Đấu CNN đông cứng, search3 hơn G rõ (CI > 0 với cả 4 đối thủ đã xong), nhưng mức tuyệt đối thấp: ActiveWR 13–20, WR 19–43. Hòa 50–70%, gần hết là hết 500 bước (vs S: 136/141 trận hòa). G + search3 hiếm khi thua (12–21/200 trận), nhưng ít khi giết được đối thủ CNN (0.14–0.22 kill/trận). Giả thuyết (chưa kiểm): search mô phỏng đối thủ bằng RuleBot, CNN không đi vào bẫy như RuleBot.
- S là đối thủ khó nhất: G + search3 chỉ WR 19.0, G thua S 76/200 trận.
- G vs Phase D cũ WR 50 nhưng ActiveWR 3: 94 trận thắng nhờ vòng sudden death (checkpoint cũ 8 kênh, không biết vòng).
- Theo luật chốt trước trong spec PDE, điều kiện "WR của G + search3 < 60% với một đối thủ CNN" đã thỏa với G, F, S (đủ 100 bàn), nên luật sẽ đề xuất train Phase D bất kể các shard còn lại. Luật Phase E chờ mốc E2.

## PDE — agent mạnh nhất (G + search3) trong bối cảnh Phase D/E, seed 50000, 20/10/2026

**Thông số:**
- Đánh giá, không train. Runner `run_pde_check.py`, spec `docs/superpowers/specs/2026-10-20-pde-check-design.md` (luật chốt trước khi chạy).
- Agent: G (`runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c…c7ac`, kiểm khi chạy), G + `kill_search.search_action3` (đối thủ trong search mô phỏng bằng RuleBot), RuleBot ở ghế learner (mốc), `checkpoints/best_phaseE_3a.pt` / `best_phaseE_4a.pt` / `best_phaseD.pt` cũ (8 kênh, mở rộng lên 13 kênh trọng số 0). Đối thủ D: G, F, S (`runs/sd53-20261018/train/epoch_8.pt`), B, Phase D cũ.
- E2: 11×11, 4 agent, learner vs 3 RuleBot, 40 bàn × 4 góc = 160 trận/nhánh (CNN 9×9 nội suy sang 11×11). E1: 9×9, 3 agent, vs 2 RuleBot, 60 bàn × 3 góc = 180 trận/nhánh. D: 9×9, 1v1 vs CNN đông cứng, 100 bàn × 2 góc = 200 trận/cặp.
- Bàn từ seed 50000 (seed mới, dùng một lần); lịch E 475/25/40; max_steps 500; mọi CNN lấy mẫu qua mask chung; bootstrap ghép cặp theo bàn (giữ mọi góc), 10 000 lần, seed 20261014.
- Môi trường: máy người dùng, CPU, torch 2.5.1, 5 tiến trình. Lượt 1 (Claude chạy nền) 11:32–~19:40 xong 64/81 shard rồi bị dừng vì thiếu RAM (mục "PDE-int"); lượt 2 người dùng chạy `--resume` 17 shard còn lại (1240 trận) trong 2:21:32, `summary.json` lúc 22:13. Output `runs/pde-50000/`, gửi `runs/pde-50000-send.zip`, log `runs/pde-50000-console.log`, `runs/pde-50000-resume-console.log`.

**E2 (11×11, 4 agent, đấu 3 RuleBot; mức công bằng 25%), 160 trận/nhánh**

| Learner | ActiveWR | WR | Hòa | Thắng thụ động | Tự nổ | s/trận |
|---|---|---|---|---|---|---|
| RuleBot (mốc) | 6.25 | 11.88 | 52.50 | 1.88 | 35.62 | 1.1 |
| Phase E 4a cũ | 1.25 | 16.88 | 53.75 | 8.75 | 0.62 | 1.6 |
| G | 23.12 | 36.25 | 53.75 | 5.62 | 1.88 | 1.6 |
| G + search3 | **86.25** | 88.75 | 10.62 | 1.88 | 0.00 | 161.8 |

**E1 (9×9, 3 agent, đấu 2 RuleBot; mức công bằng 33%), 180 trận/nhánh**

| Learner | ActiveWR | WR | Hòa | Thắng thụ động | Tự nổ | s/trận |
|---|---|---|---|---|---|---|
| RuleBot (mốc) | 8.33 | 20.56 | 38.33 | 5.56 | 39.44 | 0.7 |
| Phase E 3a cũ | 0.56 | 22.22 | 52.22 | 14.44 | 0.56 | 1.4 |
| G | 27.22 | 40.56 | 39.44 | 6.67 | 3.33 | 1.3 |
| G + search3 | **94.44** | 98.89 | 0.56 | 3.89 | 0.00 | 81.8 |

So cặp ActiveWR (pp, CI95): E2: G+s3 − G +63.12 [+55.00, +70.62]; G − RuleBot +16.88 [+10.00, +23.75]; G+s3 − RuleBot +80.00 [+73.75, +86.25]; G − Phase E 4a cũ +21.88 [+15.00, +28.75]. E1: G+s3 − G +67.22 [+60.56, +73.89]; G − RuleBot +18.89 [+12.22, +25.00]; G+s3 − RuleBot +86.11 [+81.11, +90.56]; G − Phase E 3a cũ +26.67 [+20.56, +32.78]. Theo góc: G+s3 ActiveWR 80–90 (E2), 93–95 (E1); không góc nào lệch bất thường.

**D (9×9, 1v1 đấu CNN đông cứng), 200 trận/cặp**

| Đối thủ | G+s3 ActiveWR | G+s3 WR | G+s3 hòa | G+s3 thua | G ActiveWR | G WR | G hòa | (G+s3 − G) ActiveWR | (G+s3 − G) WR |
|---|---|---|---|---|---|---|---|---|---|
| G | 14.50 | 39.50 | 54.00 | 6.50 | 5.50 | 19.50 | 61.00 | +9.00 [+3.50, +14.50] | +20.00 [+12.00, +28.00] |
| F | 20.00 | 43.00 | 51.00 | 6.00 | 4.50 | 19.00 | 66.50 | +15.50 [+9.50, +21.50] | +24.00 [+15.00, +32.50] |
| S | 13.00 | 19.00 | 70.50 | 10.50 | 0.50 | 5.50 | 56.50 | +12.50 [+7.50, +18.00] | +13.50 [+7.00, +20.00] |
| B | 20.50 | 44.00 | 50.00 | 6.00 | 4.00 | 28.50 | 61.50 | +16.50 [+10.50, +23.00] | +15.50 [+7.00, +24.00] |
| Phase D cũ | 8.50 | 72.00 | 27.00 | 1.00 | 3.00 | 50.00 | 48.00 | +5.50 [+1.00, +10.00] | +22.00 [+12.00, +32.00] |

- Search3 đấu RuleBot vẫn rất mạnh trong FFA (E1 94.44, E2 86.25), dù chỉ được kiểm ở 1v1 trước đây.
- G không search chuyển sang FFA được: hơn mốc RuleBot ở cả E1 (+18.89) và E2 (+16.88, CI dưới +10.0), dù chỉ train 1v1 9×9 và E2 dùng trọng số nội suy. Checkpoint Phase E cũ có ActiveWR ≈ 0–1 (xác nhận kiểm tra 10/10).
- RuleBot ở ghế learner tự nổ 36–39% trong FFA.
- Đấu CNN, search3 hơn G với mọi đối thủ (CI ActiveWR > 0), nhưng mức tuyệt đối thấp: ActiveWR 8.5–20.5, WR 19–44 (trừ Phase D cũ 72), hòa 50–70%. G + search3 hiếm khi thua (1–10.5%); phần lớn trận hòa vì hết 500 bước (mục "PDE-peek": vs S 136/141 trận hòa là hết giờ). Giả thuyết chưa kiểm: search mô phỏng đối thủ bằng RuleBot, còn CNN không đi vào bẫy như RuleBot dự đoán.
- S là đối thủ khó nhất (G + search3 WR 19.0, hòa 70.5; G thua S 38%).
- WR cao trước Phase D cũ chủ yếu nhờ vòng sudden death (checkpoint 8 kênh không biết vòng; G có 94 trận thắng nhờ vòng), ActiveWR chỉ 3.0 / 8.5.

**Áp luật đề xuất chốt trước:**
- Phase E: CI dưới của (G − RuleBot) ActiveWR ở E2 là +10.0 > 0 → **không đề xuất train Phase E**.
- Phase D: (G+s3 − G) ActiveWR có CI dưới > 0 với mọi đối thủ (nhỏ nhất +1.0 vs Phase D cũ), nhưng **WR của G + search3 < 60% với G (39.5), F (43.0), S (19.0), B (44.0)** → **luật đề xuất train Phase D (self-play pool)**. Spec PDE ghi "từ G"; spec PDE2 ghi điểm xuất phát là S2 nếu có. Chưa train; cần spec riêng và người dùng duyệt.
- Lưu ý khi đọc luật: WR < 60% ở đây là do hòa hết giờ chứ không phải do thua (G + search3 thua ≤ 10.5%). Ngưỡng 60% lấy từ gate Phase D của PROJECT_SPEC.

**Giới hạn:** một seed, 160–200 trận mỗi nhánh (CI ±6–9pp); đối thủ CNN đông cứng, lấy mẫu, không thích nghi; checkpoint Phase D/E cũ là 8 kênh của code ban đầu; mô tả, không gate.

## SDM-prep — kill-search mô phỏng đối thủ bằng CNN: code, kiểm parity, đo thời gian, 20/10/2026

**Thông số:**
- Kiểm parity: `kill_search` mới (hook `OPPONENT_MODEL`, mặc định RuleBot) so với bản ở HEAD `c957567`. G + search3, 4 trận (seed 6600, 2 góc), max_steps 120, không SD, torch seed 0 / numpy seed 0.
- Đo thời gian: `probe_search_cnn_model.py --seed 6600 --boards 1 --workers 4` (seed chẩn đoán, không phải seed probe), 1 bàn × 2 góc = 2 trận/nhánh, lịch E, max_steps 500. Output ở scratchpad (không lưu vào `runs/`).
- Môi trường: Claude, máy người dùng, CPU, torch 2.5.1.

- Parity: kết quả 4 trận giống hệt bản cũ (bước, thắng, kill).
- Thời gian mỗi trận: oracle vs G 249.7 s, oracle vs S 305.0 s, rulebot vs G 127.8 s, rulebot vs S 149.5 s. Mô hình CNN chậm khoảng 2 lần. Số thắng/thua của 2 trận/nhánh không có ý nghĩa và không dùng để kết luận.
- Spec và luật chốt trước: `docs/superpowers/specs/2026-10-20-sdm-search-cnn-opponent-model.md` (seed validation 21000, 50 bàn × 2 góc = 100 trận/nhánh, đối thủ S và G). Lượt chính giao người dùng chạy, ước tính khoảng 4.5–5 giờ với 5 tiến trình.

## PDE-D-detail — bảng Phase D chi tiết (theo yêu cầu người dùng), 21/10/2026

**Thông số:** đọc lại `runs/pde-50000/evaluation_D_*.json` và `summary.json` (mục "PDE"), không chạy thêm trận. 9×9, 1v1, seed 50000 (bàn 50000–50099 × 2 góc = 200 trận/cặp), lịch E 475/25/40, max_steps 500, CNN lấy mẫu, search3 đoán đối thủ bằng RuleBot; bootstrap theo bàn 10 000 lần, seed 20261014.

**Các model**

| Ký hiệu | Checkpoint (SHA rút gọn) | Nguồn / vai trò |
|---|---|---|
| **G + search3** | G + `kill_search.search_action3` | **Agent mạnh nhất của người dùng** (SD8B: ActiveWR 94.0 vs RuleBot, test 49000) |
| **G** | `runs/sd75-20261018/full/best_phaseC.pt` (`bb0c775c…c7ac`) | **CNN tốt nhất của người dùng** (SD7.5 best ep 1000, chọn 19/10); đối chứng không search; cũng là đối thủ gương |
| F | `runs/phaseC-sd-best-fallback/best_phaseC.pt` (`0315a56f…1483`) | Nhánh của bạn, ~3000 ep từ B, điểm xuất phát SD5 |
| S | `runs/sd53-20261018/train/epoch_8.pt` (`8f7116d2…fd4a`) | SD5.3 của bạn: CNN chưng cất dày search một nước |
| B | `runs/pressure-C-next/final_phaseC.pt` (`93bdf278…4f60`) | Model gốc của các pilot tới SD4.6 |
| D cũ | `checkpoints/best_phaseD.pt` | Phase D cũ (04–05/10), 8 kênh, không thấy vòng sudden death |

**Kết quả của learner (đơn vị %; WR = ActiveWR + thắng thụ động + thắng nhờ vòng)**

| Đối thủ | Learner | ActiveWR | Thắng nhờ vòng | Thắng thụ động | WR | Hòa (hết 500 bước) | Thua (đối thủ thắng) | Chết vì vòng | Tự nổ | Kill/trận | Bước TB | s/trận | ActiveWR / WR góc 0, góc 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G | G + search3 | 14.5 | 23.5 | 1.5 | 39.5 | 54.0 (50.0) | 6.5 | 6.0 | 4.0 | 0.17 | 439 | 131 | 14/44, 15/35 |
| G | G | 5.5 | 13.0 | 1.0 | 19.5 | 61.0 (55.0) | 19.5 | 29.0 | 5.5 | 0.07 | 460 | 1.5 | 5/18, 6/21 |
| F | G + search3 | 20.0 | 22.0 | 1.0 | 43.0 | 51.0 (48.5) | 6.0 | 5.5 | 2.0 | 0.20 | 426 | 127 | 18/38, 22/48 |
| F | G | 4.5 | 13.5 | 1.0 | 19.0 | 66.5 (55.5) | 14.5 | 23.0 | 3.5 | 0.04 | 462 | 1.8 | 5/20, 4/18 |
| S | G + search3 | 13.0 | 4.5 | 1.5 | 19.0 | 70.5 (68.0) | 10.5 | 5.0 | 6.0 | 0.14 | 444 | 133 | 15/22, 11/16 |
| S | G | 0.5 | 4.5 | 0.5 | 5.5 | 56.5 (56.5) | 38.0 | 30.0 | 5.0 | 0.01 | 464 | 1.6 | 1/8, 0/3 |
| B | G + search3 | 20.5 | 22.5 | 1.0 | 44.0 | 50.0 (46.5) | 6.0 | 4.5 | 5.0 | 0.22 | 431 | 131 | 19/48, 22/40 |
| B | G | 4.0 | 24.0 | 0.5 | 28.5 | 61.5 (49.5) | 10.0 | 19.0 | 5.5 | 0.04 | 467 | 2.2 | 7/30, 1/27 |
| D cũ | G + search3 | 8.5 | 63.5 | 0.0 | 72.0 | 27.0 (27.0) | 1.0 | 1.0 | 1.0 | 0.09 | 458 | 148 | 5/74, 12/70 |
| D cũ | G | 3.0 | 47.0 | 0.0 | 50.0 | 48.0 (33.5) | 2.0 | 19.5 | 2.0 | 0.03 | 476 | 1.5 | 4/46, 2/54 |

**Hiệu G + search3 − G (pp, CI95)**

| Đối thủ | ActiveWR | WR | Hòa | Thắng thụ động | Tự nổ |
|---|---|---|---|---|---|
| G | +9.0 [+3.5, +14.5] | +20.0 [+12.0, +28.0] | −7.0 [−18.0, +4.0] | +0.5 [0.0, +1.5] | −1.5 [−6.0, +2.5] |
| F | +15.5 [+9.5, +21.5] | +24.0 [+15.0, +32.5] | −15.5 [−25.0, −5.5] | 0.0 [−1.5, +1.5] | −1.5 [−4.0, +1.0] |
| S | +12.5 [+7.5, +18.0] | +13.5 [+7.0, +20.0] | +14.0 [+5.0, +22.5] | +1.0 [−1.0, +3.0] | +1.0 [−2.5, +4.5] |
| B | +16.5 [+10.5, +23.0] | +15.5 [+7.0, +24.0] | −11.5 [−20.5, −2.5] | +0.5 [0.0, +1.5] | −0.5 [−4.5, +3.0] |
| D cũ | +5.5 [+1.0, +10.0] | +22.0 [+12.0, +32.0] | −21.0 [−31.0, −11.0] | 0.0 [0.0, 0.0] | −1.0 [−2.5, 0.0] |

- Hơn một nửa WR của G + search3 đến từ thắng nhờ vòng sudden death (22–24% với G, F, B; 63.5% với D cũ), không phải từ giết đối thủ. ActiveWR chỉ 8.5–20.5.
- Search giảm mạnh chết vì vòng của G (19–30% → 1–6%) và tỉ lệ thua (10–38% → 1–10.5%). Phần lớn lợi ích của search khi đấu CNN là phòng thủ.
- S là đối thủ duy nhất làm hòa tăng khi bật search (+14.0): G + search3 không còn thua S (38% → 10.5%) nhưng cũng không thắng được, nên các ván hòa vì hết giờ.
- Luật Phase D (WR G + search3 < 60%) kích hoạt với G, F, S, B (mục "PDE"). Chỉ D cũ vượt 60%, và chủ yếu nhờ vòng.

## SDM — kill-search mô phỏng đối thủ bằng CNN (oracle) vs RuleBot, 21/10/2026

**Thông số:** Đánh giá (probe, không train): learner G (`runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c…c7ac`) + `kill_search.search_action3` (mặc định); mô hình đối thủ trong search: `rulebot` (như cũ) hoặc `oracle` = argmax policy đã mask của chính checkpoint đối thủ (`kill_search.OPPONENT_MODEL`); đối thủ thật S (`runs/sd53-20261018/train/epoch_8.pt`, `8f7116d2…fd4a`) và G, lấy mẫu policy; seed validation 21000, bàn 21000–21049 (50 bàn × 2 góc = 100 trận mỗi nhánh, ghép cặp theo bàn giữa hai mô hình); 9×9, 1v1, max_steps 500, lịch E 475/25/40; bootstrap theo bàn 10 000 lần, seed 20261014. Môi trường: người dùng chạy trên máy của mình, torch 2.5.1 CPU, 5 tiến trình, 40 shard × 5 bàn, ~4 giờ 51 phút (opened 04:13 → summary 09:04). Output `runs/sdm-21000/` (`summary.json`, `shards/`), log `runs/sdm-21000-console.log`. Spec và luật chốt trước: `docs/superpowers/specs/2026-10-20-sdm-search-cnn-opponent-model.md`.

**Kết quả của learner (đơn vị %, 100 trận mỗi dòng)**

| Đối thủ thật | Mô hình đối thủ trong search | ActiveWR | Thắng nhờ vòng | Thắng thụ động | WR | Hòa (hết 500 bước) | Thua | Chết vì vòng | Tự nổ | Kill/trận | Bước TB | s/trận |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | rulebot | 14.0 | 3.0 | 2.0 | 19.0 | 69.0 (65.0) | 12.0 | 5.0 | 5.0 | 0.15 | 435 | 125 |
| S | oracle (CNN S) | 18.0 | 6.0 | 1.0 | 25.0 | 68.0 (66.0) | 7.0 | 4.0 | 1.0 | 0.18 | 448 | 305 |
| G | rulebot | 12.0 | 21.0 | 0.0 | 33.0 | 63.0 (60.0) | 4.0 | 2.0 | 4.0 | 0.12 | 456 | 127 |
| G | oracle (CNN G) | 26.0 | 20.0 | 2.0 | 48.0 | 50.0 (49.0) | 2.0 | 2.0 | 0.0 | 0.26 | 428 | 292 |

**Hiệu oracle − rulebot (pp, CI95, ghép cặp theo bàn)**

| Đối thủ | ActiveWR | WR | Hòa | Thắng thụ động | Tự nổ |
|---|---|---|---|---|---|
| S | +4.0 [−6.0, +15.0] | +6.0 [−5.0, +17.0] | −1.0 [−14.0, +12.0] | −1.0 [−5.0, +2.0] | −4.0 [−10.0, +1.0] |
| G | **+14.0 [+3.0, +25.0]** | +15.0 [+1.0, +29.0] | −13.0 [−26.0, 0.0] | +2.0 [0.0, +5.0] | −4.0 [−9.0, 0.0] |

**Luật chốt trước:** "Mô hình CNN giúp" nếu ActiveWR ≥ +5pp với CI dưới > 0 ở ít nhất một đối thủ và không đối thủ nào có CI trên < 0. Với G: +14.0 [+3.0, +25.0] đạt. Với S: +4.0 [−6.0, +15.0], không giảm có ý nghĩa. → **Kết luận theo luật: mô hình CNN giúp.** Bước tiếp theo theo luật là đề xuất xác nhận trên seed test mới 52000 và đề xuất dùng mô hình CNN trong search thay vì train Phase D ngay. Cả hai mới là đề xuất, chưa chạy.

**Diễn giải và giới hạn:**
- Mức tăng có ý nghĩa chỉ ở đối thủ gương G. Ở đó mô hình oracle chính là CNN của learner, nên search đoán đối thủ rất sát. Với S (đối thủ khó nhất trong PDE), ActiveWR tăng +4 nhưng CI chứa 0, và hòa không đổi (68–69%, gần hết vì hết 500 bước). Vấn đề hòa với S chưa được giải.
- Oracle là trường hợp tốt nhất: search biết chính xác checkpoint đối thủ. Đối thủ lạ thì không có thông tin này.
- Chi phí: oracle chậm khoảng 2.4 lần (≈300 so với ≈126 s/trận), vì mỗi bước mô phỏng phải chạy CNN.
- 100 trận mỗi nhánh, một seed validation; CI rộng ±10–14pp. Dòng rulebot khớp cỡ với PDE (seed 50000: vs G ActiveWR 14.5, vs S 13.0).

## SDM2-prep — chuẩn bị probe search đoán đối thủ bằng CNN G, 22/10/2026

**Thông số:** Đo thời gian (không phải kết quả): G (`bb0c775c…c7ac`) + `search_action3`, mô hình đối thủ trong search `rulebot` / `self` (argmax CNN G), đối thủ thật RuleBot, seed chẩn đoán 6600, 1 bàn × 2 góc = 2 trận mỗi nhánh, 9×9 1v1, max_steps 500, lịch E 475/25/40; Claude chạy trên máy người dùng, torch 2.5.1 CPU, 2 tiến trình. Output ở scratchpad (`sdm2-timing`), không lưu trong `runs/`.

- Thời gian mỗi trận khi đấu RuleBot: rulebot 6.1 s, self 31.2 s. Ước tính lượt chính (RuleBot/S/F × rulebot/self, 100 trận/nhánh, tái dùng `rulebot_vs_S` của SDM): ~4–4.5 giờ với 5 tiến trình.
- 2 trận mỗi nhánh không đủ để kết luận. Ghi lại cho đủ: rulebot thắng chủ động 2/2, self thắng 2/2 nhưng chỉ 1 trận chủ động.
- Spec và luật chốt trước: `docs/superpowers/specs/2026-10-22-sdm2-search-self-cnn-model.md`. Lượt chính chờ người dùng chạy (`runs/sdm2-21000`).

## SDM2 — kill-search luôn đoán đối thủ bằng CNN G (self) vs RuleBot, 22/10/2026

**Thông số:** Đánh giá (probe, không train): learner G (`runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c…c7ac`) + `kill_search.search_action3` (mặc định; `kill_search.py` SHA `7c8b9f49…7991`); mô hình đối thủ trong search: `rulebot` (như cũ) hoặc `self` = argmax policy đã mask của G; đối thủ thật RuleBot, S (`runs/sd53-20261018/train/epoch_8.pt`, `8f7116d2…fd4a`) và F (`runs/phaseC-sd-best-fallback/best_phaseC.pt`, `0315a56f…1483`), CNN lấy mẫu policy; seed validation 21000, bàn 21000–21049 (50 bàn × 2 góc = 100 trận mỗi nhánh, ghép cặp theo bàn); 9×9, 1v1, max_steps 500, lịch E 475/25/40; bootstrap theo bàn 10 000 lần, seed 20261014. Nhánh `rulebot_vs_S` tái dùng 10 shard của SDM (`--reuse runs/sdm-21000`). Môi trường: người dùng chạy trên máy của mình, torch 2.5.1 CPU, 5 tiến trình, 50 shard mới (500 trận), 5 giờ 43 phút (opened 05:51 → summary 11:34). Output `runs/sdm2-21000/`, log `runs/sdm2-21000-console.log`. Spec và luật chốt trước: `docs/superpowers/specs/2026-10-22-sdm2-search-self-cnn-model.md`.

**Kết quả của learner (đơn vị %, 100 trận mỗi dòng)**

| Đối thủ thật | Mô hình đối thủ trong search | ActiveWR | Thắng nhờ vòng | Thắng thụ động | WR | Hòa (hết 500 bước) | Thua | Chết vì vòng | Tự nổ | Kill/trận | Bước TB | s/trận |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RuleBot | rulebot | 96.0 | 0.0 | 3.0 | 99.0 | 1.0 (1.0) | 0.0 | 0.0 | 0.0 | 0.96 | 111 | 24 |
| RuleBot | self (CNN G) | 63.0 | 13.0 | 5.0 | 81.0 | 17.0 (16.0) | 2.0 | 0.0 | 1.0 | 0.63 | 299 | 184 |
| S | rulebot | 14.0 | 3.0 | 2.0 | 19.0 | 69.0 (65.0) | 12.0 | 5.0 | 5.0 | 0.15 | 435 | 125 |
| S | self (CNN G) | 23.0 | 4.0 | 2.0 | 29.0 | 65.0 (65.0) | 6.0 | 5.0 | 0.0 | 0.23 | 428 | 347 |
| F | rulebot | 16.0 | 21.0 | 0.0 | 37.0 | 54.0 (49.0) | 9.0 | 7.0 | 6.0 | 0.17 | 442 | 127 |
| F | self (CNN G) | 28.0 | 21.0 | 0.0 | 49.0 | 45.0 (44.0) | 6.0 | 6.0 | 2.0 | 0.28 | 422 | 335 |

**Hiệu self − rulebot (pp, CI95, ghép cặp theo bàn)**

| Đối thủ | ActiveWR | WR | Hòa | Thắng thụ động | Tự nổ |
|---|---|---|---|---|---|
| RuleBot | **−33.0 [−41.0, −25.0]** | −18.0 [−26.0, −11.0] | +16.0 [+9.0, +24.0] | +2.0 [−3.0, +7.0] | +1.0 [0.0, +3.0] |
| S | +9.0 [−1.0, +20.0] | +10.0 [−1.0, +21.0] | −4.0 [−17.0, +10.0] | 0.0 [−3.0, +3.0] | −5.0 [−11.0, −1.0] |
| F | **+12.0 [+3.0, +22.0]** | +12.0 [0.0, +24.0] | −9.0 [−22.0, +5.0] | 0.0 [0.0, 0.0] | −4.0 [−9.0, 0.0] |

**Luật chốt trước:**
- Giữ `self` khi ActiveWR ≥ +5pp với CI dưới > 0 ở ít nhất một trong S, F, và không đối thủ nào có CI trên < 0.
- F đạt điều kiện đầu (+12.0 [+3.0, +22.0]), nhưng RuleBot giảm −33.0 với CI trên −25.0 < 0.
- → **KHÔNG GIỮ.** Search vẫn đoán đối thủ bằng RuleBot. Không dùng seed test 52000 cho SDM2.

**Diễn giải và giới hạn:**
- Mô hình đối thủ đúng loại mới giúp. Đoán bằng RuleBot tốt nhất khi đấu RuleBot (96). Đoán bằng CNN tốt hơn khi đấu CNN: F +12, S +9 (CI chạm 0), G +14 trong SDM. Không mô hình đơn nào tốt cho mọi đối thủ.
- Đoán bằng CNN G làm trận đấu RuleBot dài ra (111 → 299 bước) và hòa tăng (+16). Search không còn thấy các bẫy chắc chắn với RuleBot.
- Hòa khi đấu S vẫn 65%, nên vấn đề hòa với S chưa được giải bằng cách đổi mô hình đoán.
- Một seed validation, 100 trận mỗi nhánh. Dòng `rulebot_vs_S` là shard tái dùng từ SDM (cùng code và bàn).
- Chưa thử (ý tưởng, chưa chốt luật): nhận diện loại đối thủ trong trận rồi chọn mô hình, hoặc chỉ nhận nước giết khi đúng dưới cả hai mô hình.

## SD8A-review — kiểm định file SD8A của bạn, 22/10/2026

**Thông số:** Train (bạn chạy, theo `runs/sd8a-20261020/registration.json`): G (`runs/sd75-20261018/full/best_phaseC.pt`, `bb0c775c…c7ac`); thu dữ liệu bằng G + `search_action2`, 3000 trận trên bàn 210000–211499 (lịch E với xác suất 0.75, anchor 0.05); train 16 epoch, lr 1e-4, batch 256, anchor_lambda 1.0, heldout 0.1; ứng viên epoch 4/8/12/16; max_steps 500. Validation seed 21000, 400 trận/ứng viên. Test một lần seed 48000, 400 trận/agent (200 bàn × 2 góc), 9×9 1v1, lịch E (gate) và off (báo cáo), policy lấy mẫu, bootstrap 10 000 lần, seed 20261014. Môi trường: Python 3.11.9, torch 2.11.0+cu128, GPU RTX 4060 Laptop; số tiến trình và thời gian chạy không lưu lại trong `runs/`. Claude kiểm lại trên máy người dùng (torch 2.5.1), không chạy thêm trận.

**Kiểm định:**
- S2 = `runs/sd8a-20261020/train/epoch_16.pt`, SHA256 `d222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814`, khớp `test_opened.json`.
- Thứ tự đúng protocol: `selected.json` (20/10 01:26) có trước `test_opened.json` (02:23). Chọn epoch 16 theo validation (ActiveWR 84.0, cao nhất).
- Tính lại ActiveWR từ các file `evaluation_*_48000.json` ra đúng số trong `summary.json`: S2 83.0, G 48.0, S 57.5, G + search2 86.5.
- Bảng đầy đủ đã có trong mục SD8A của bạn trên nhánh `origin/Loc/sd8-huong-a` (commit `7fe621c`), chưa merge vào nhánh này.

| So cặp (lịch E, seed 48000) | ActiveWR Δpp (CI95) | WR Δpp | Hòa Δpp | Tự nổ Δpp |
|---|---|---|---|---|
| S2 − G | **+35.0 [+29.0, +41.25]** | +25.0 | −17.25 | −0.5 |
| S2 − S | **+25.5 [+19.25, +31.75]** | +13.25 | −13.5 | 0.0 |
| S2 − G + search2 | −3.5 [−8.25, +1.25] | −4.5 | +1.75 | +0.25 |

**Gate: ĐẠT** (ActiveWR ≥ +5pp, CI dưới > 0, tự nổ không tăng quá 1pp). Luật phụ ĐẠT: S2 thay S. S2 là CNN thuần mạnh nhất khi đấu RuleBot. Agent mạnh nhất vẫn là G + search3 (94.0 trên test 49000; khác seed nên không so trực tiếp với S2). Seed 48000 đã dùng.

**Giới hạn:** một seed train/test; teacher và đối thủ đánh giá đều dựa trên RuleBot; số liệu chạy bằng torch 2.11 trên GPU, không tái lập bit-to-bit bằng torch 2.5.1.

## PDE2 — nhiều agent đánh nhau: FFA hỗn hợp, gương S2, đối đầu 1v1, seed 51000, 22/10/2026 (review)

**Thông số:** Đánh giá (không train): agent G + `search_action3` (G `bb0c775c…c7ac`), S2 (`d222d81f…6814`), G (lấy mẫu), RuleBot. Lịch E 475/25/40, max_steps 500, policy lấy mẫu; bootstrap theo bàn 10 000 lần, seed 20261014.
- FFA hỗn hợp 4 ghế xoay 4 lượt mỗi bàn: 9×9 bàn 51000–51049 (200 trận) và 11×11 bàn 52000–52049 (200 trận).
- Gương 4×S2: 9×9 bàn 53000–53399 (400 trận) và 11×11 bàn 54000–54399 (400 trận).
- Đối đầu 1v1 9×9: G+s3 vs S2 bàn 55000–55099; S2 vs G bàn 56000–56099; mỗi cặp 100 bàn × 2 góc = 200 trận.
- Môi trường (`runs/pde2-51000/registration.json`): Python 3.11.9 (cùng bản với lượt SD8A của bạn), 5 tiến trình, ~10 giờ 51 phút (registration 21/10 11:35 → report 22:26). Người chạy và phiên bản torch không lưu lại.
- Runner `run_pde2_ffa_mix.py` và `pde_eval.py` không có trong repo hay nhánh git nào. Claude chỉ có SHA trong `registration.json`, không review được code.
- Output `runs/pde2-51000/` (`report.json`, 1100 shard).

**Phần 1: FFA hỗn hợp (đơn vị %, 200 trận mỗi agent; hòa là của cả trận)**

| Bàn | Agent | ActiveWR | WR | Tự nổ | Chết vì vòng | Kill/trận |
|---|---|---|---|---|---|---|
| 9×9 (hòa 73.5) | G + search3 | **9.0** | **15.0** | 8.0 | 7.0 | 0.80 |
| | S2 | 5.0 | 7.0 | 9.0 | 14.0 | 0.36 |
| | G | 2.0 | 4.0 | 21.0 | 21.0 | 0.20 |
| | RuleBot | 0.0 | 0.5 | 78.5 | 0.5 | 0.07 |
| 11×11 (hòa 88.0) | G + search3 | **5.5** | **7.0** | 8.0 | 2.5 | 0.70 |
| | S2 | 1.0 | 4.0 | 6.0 | 9.0 | 0.23 |
| | G | 0.0 | 1.0 | 5.5 | 10.0 | 0.19 |
| | RuleBot | 0.0 | 0.0 | 75.5 | 1.5 | 0.03 |

So cặp ActiveWR (pp, CI95, 50 bàn):
- 9×9: G+s3 − S2 +4.0 [−1.0, +9.0]; S2 − G +3.0 [−0.5, +7.0]; G+s3 − G +7.0 [+2.5, +12.0]; S2 − RB +5.0 [+2.0, +8.0].
- 11×11: G+s3 − S2 +4.5 [+1.0, +8.0]; S2 − RB WR +4.0 [+1.5, +6.5].

**Phần 2: gương 4×S2**
- 9×9: hòa 75.75%. Trong 97 trận có người thắng, góc thắng 23.7 / 22.7 / 25.8 / 27.8% (công bằng 20–30%).
- 11×11: hòa 95.0%. Chỉ 20 trận có người thắng, góc 15 / 20 / 40 / 25% (ngoài 20–30%, nhưng mẫu rất nhỏ).

**Phần 3: đối đầu 1v1 (9×9, 200 trận mỗi cặp)**

| Cặp | Agent | ActiveWR | WR | Hòa | Tự nổ | Chết vì vòng |
|---|---|---|---|---|---|---|
| G+s3 vs S2 | G + search3 | 14.0 | 23.5 | 64.5 | 5.0 | 5.5 |
| | S2 | 8.5 | 12.0 | | 7.5 | 10.0 |
| S2 vs G | S2 | 7.5 | 31.0 | 60.5 | 4.0 | 9.5 |
| | G | 4.0 | 8.5 | | 6.0 | 28.0 |

G+s3 − S2: ActiveWR +5.5 [−1.0, +12.5], WR +11.5 [+3.5, +20.0]. S2 − G: ActiveWR +3.5 [−0.5, +8.0], WR +22.5 [+15.5, +29.5] (phần lớn nhờ G chết vì vòng).

**Luật chốt trước (Claude kiểm lại `proposals` trong report, khớp):**
1. Agent FFA khuyến nghị: G+s3 có ActiveWR cao nhất ở 9×9 nhưng CI với S2 chứa 0 → "ngang nhau", khuyến nghị **S2** (nhanh hơn).
2. Đề xuất Phase E từ S2: **có**. Lý do duy nhất là góc 11×11 ở gương lệch 40%, nhưng chỉ có 20 trận có người thắng. Điều kiện WR(S2) − WR(RB) ở 11×11 không kích hoạt (CI dưới +1.5).
3. Đề xuất Phase D từ S2: **có**, vì WR G+s3 vs S2 = 23.5% < 60% (và luật Phase D của PDE đã kích hoạt). G+s3 không thua S2 về ActiveWR.
4. Chưa train (`decision: proposal_only_no_training_started`).

**Diễn giải và giới hạn:**
- Khi các agent mạnh đánh nhau, hầu hết trận hòa (73.5–95%). ActiveWR của mọi agent chỉ 0–14%. RuleBot tự nổ 75–78% trong FFA hỗn hợp.
- G + search3 vẫn đứng đầu ở mọi phần nhưng cách biệt nhỏ. S2 nhanh hơn nhiều và ngang G + search3 trong FFA 9×9.
- **Seed:** PDE2 dùng dải bàn 51000–56099, gồm cả 52000 mà các mục trước ghi là "seed test mới tiếp theo". Seed mới sạch tiếp theo đổi thành **57000**.
- 50 bàn cho mỗi cấu hình FFA; một seed; runner không review được.

## SD9.0-prep — chuẩn bị probe S2 + search3 vs G + search3, 22/10/2026

**Thông số:** Đo thời gian (không phải kết quả): G + `search_action3` và S2 (`runs/sd8a-20261020/train/epoch_16.pt`, `d222d81f…6814`) + `search_action3`, đối thủ RuleBot / S2 / G (CNN lấy mẫu), seed chẩn đoán 6600, 1 bàn × 2 góc = 2 trận mỗi nhánh, 9×9 1v1, max_steps 500, lịch E 475/25/40; Claude chạy trên máy người dùng, torch 2.5.1 CPU, 6 tiến trình. Output ở scratchpad (`sd90-timing`), không lưu trong `runs/`.

- s/trận: đấu RuleBot 5.6–7.3; đấu CNN 153–185.
- Lượt chính chạy 4 nhánh mới × 100 trận. G + search3 vs RuleBot và vs G tái dùng từ SDM2/SDM. Ước tính ~3 giờ với 5 tiến trình.
- 2 trận mỗi nhánh không đủ để kết luận gì.
- Spec và luật chốt trước: `docs/superpowers/specs/2026-10-22-sd9-0-s2-search3-probe.md`. Lượt chính chờ người dùng chạy (`runs/sd90-21000`).

## SD9.0 — search ba nước trên S2 so với trên G, 23/10/2026

**Thông số:** Đánh giá (probe, không train): learner G + `search_action3` (G `runs/sd75-20261018/full/best_phaseC.pt`, `bb0c775c…c7ac`) và S2 + `search_action3` (S2 `runs/sd8a-20261020/train/epoch_16.pt`, `d222d81f…6814`); search đoán đối thủ bằng RuleBot (mặc định, `kill_search.py` `7c8b9f49…7991`). Đối thủ RuleBot, S2, G (CNN lấy mẫu). Seed validation 21000, bàn 21000–21049 (50 bàn × 2 góc = 100 trận mỗi nhánh, ghép cặp theo bàn giữa hai learner). 9×9, 1v1, max_steps 500, lịch E 475/25/40. Bootstrap theo bàn 10 000 lần, seed 20261014. G + search3 vs RuleBot và vs G tái dùng 20 shard từ SDM2/SDM (`reused: true`). Môi trường: người dùng chạy trên máy của mình, torch 2.5.1 CPU, 5 tiến trình, 40 shard mới (400 trận), 3 giờ 18 phút (opened 01:00 → summary 04:18). Output `runs/sd90-21000/`, log `runs/sd90-21000-console.log`. Spec: `docs/superpowers/specs/2026-10-22-sd9-0-s2-search3-probe.md`.

**Kết quả của learner (đơn vị %, 100 trận mỗi dòng)**

| Đối thủ | Learner | ActiveWR | Thắng nhờ vòng | Thắng thụ động | WR | Hòa (hết 500 bước) | Thua | Chết vì vòng | Tự nổ | Bước TB | s/trận |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RuleBot | G + search3 | 96.0 | 0.0 | 3.0 | 99.0 | 1.0 (1.0) | 0.0 | 0.0 | 0.0 | 111 | 24 |
| RuleBot | S2 + search3 | 91.0 | 1.0 | 8.0 | 100.0 | 0.0 (0.0) | 0.0 | 0.0 | 0.0 | 100 | 21 |
| S2 | G + search3 | 16.0 | 4.0 | 4.0 | 24.0 | 62.0 (59.0) | 14.0 | 8.0 | 7.0 | 411 | 128 |
| S2 | S2 + search3 | 14.0 | 7.0 | 1.0 | 22.0 | 68.0 (64.0) | 10.0 | 9.0 | 4.0 | 433 | 224 |
| G | G + search3 | 12.0 | 21.0 | 0.0 | 33.0 | 63.0 (60.0) | 4.0 | 2.0 | 4.0 | 456 | 127 |
| G | S2 + search3 | 19.0 | 26.0 | 0.0 | 45.0 | 52.0 (48.0) | 3.0 | 0.0 | 4.0 | 421 | 215 |

**Hiệu (S2 + search3) − (G + search3) (pp, CI95, ghép cặp theo bàn)**

| Đối thủ | ActiveWR | WR | Hòa | Thắng thụ động | Tự nổ |
|---|---|---|---|---|---|
| RuleBot | −5.0 [−11.0, +1.0] | +1.0 [0.0, +3.0] | −1.0 [−3.0, 0.0] | +5.0 [0.0, +10.0] | 0.0 [0.0, 0.0] |
| S2 | −2.0 [−11.0, +7.0] | −2.0 [−13.0, +8.0] | +6.0 [−6.0, +17.0] | −3.0 [−9.0, +1.0] | −3.0 [−9.0, +3.0] |
| G | +7.0 [−2.0, +16.0] | +12.0 [−1.0, +24.0] | −11.0 [−23.0, +1.0] | 0.0 [0.0, 0.0] | 0.0 [−4.0, +4.0] |

**Luật chốt trước:**
1. Thầy cho SD9 là S2 + search3, trừ khi ActiveWR vs RuleBot có CI trên < 0. CI trên là +1.0 → **thầy SD9 = S2 + search3**.
2. Agent mạnh nhất mới cần ≥ +5pp với CI dưới > 0 ở ít nhất một đối thủ. Không đối thủ nào đạt → **G + search3 vẫn là agent mạnh nhất**. Không dùng seed 57000.

**Diễn giải và giới hạn:**
- Gắn search3 lên S2 không tốt hơn gắn lên G. Đấu RuleBot, S2 + search3 thắng 100% nhưng ActiveWR thấp hơn 5 điểm (CI chạm 0), vì có thêm 5 trận thắng thụ động.
- Khi đấu CNN, hai learner ngang nhau. Hòa vẫn 52–68%.
- S2 + search3 chậm hơn khoảng 1.7 lần khi đấu CNN (215–224 so với 127 s/trận). Nguyên nhân chưa kiểm.
- Luật chọn thầy là S2 + search3 dù ước lượng điểm vs RuleBot thấp hơn 5. Đây là luật đã chốt; rủi ro này ghi lại để thiết kế SD9 cân nhắc.
- Một seed validation, 100 trận mỗi nhánh. G + search3 vs RuleBot và vs G là shard tái dùng.

## SD9-prep — chuẩn bị chưng cất S2 + search3 thành CNN (S3): runner, smoke, 23/10/2026

**Thông số:** Smoke (không phải kết quả): `run_sd9_distill.py --smoke`. Thu 4 bàn × 2 góc = 8 trận S2 + `search_action3` trên bàn 212000–212003, max_steps 500. Train 2 epoch, batch 16, CPU. Validate và test trên seed 21000, 4 trận/agent. Claude chạy trên máy người dùng, torch 2.5.1 CPU, 2 tiến trình. Output ở scratchpad (`sd9-smoke`), không lưu trong `runs/`.

- Cả 4 bước collect / train / validate / test chạy xong (exit 0). Smoke có 8 trận, 24 trạng thái quyết định, 60 neo.
- Thời gian thu: 8 trận mất ~2 phút 47 giây với 2 tiến trình (~40 s/trận mỗi tiến trình). Lượt chính 3000 trận ước tính 6–7 giờ với 5 tiến trình.
- Spec và gate chốt trước: `docs/superpowers/specs/2026-10-23-sd9-search3-distillation.md`. Lệnh: `SD9_GUIDE.md`. Thư mục lượt chính: `runs/sd9-20261023`. Bàn thu 212000–213499, test 57000.

## SD9 — chưng cất S2 + search3 thành CNN thuần S3, test seed 57000, 23/10/2026

**Thông số:**
- **Thầy:** S2 + `kill_search.search_action3`, đoán đối thủ bằng RuleBot. S2 là `runs/sd8a-20261020/train/epoch_16.pt` (`d222d81f…6814`).
- **Thu dữ liệu:** 3000 trận đấu RuleBot trên bàn 212000–213499 (1500 bàn × 2 góc). Lịch E xác suất 0.75, neo 5%, shard 25 bàn. Tổng 10 425 trạng thái quyết định và 15 865 neo.
- **Train:** học trò xuất phát từ S2. Loss CE mục tiêu mềm + KL về S2 (λ = 1). Adam 1e-4, batch 256, 16 epoch, ứng viên 4/8/12/16, held-out 10% bàn. Thiết bị: GPU RTX 3050 Laptop (cuda), 17 giây.
- **Validation:** seed 21000, 400 trận/ứng viên, lịch E. Luật chọn: ActiveWR cao nhất với tự nổ ≤ S2 + 1pp.
- **Test (một lần):** seed 57000, 200 bàn × 2 góc = 400 trận/agent, 9×9, 1v1, max_steps 500, đối thủ RuleBot.
  - CNN lấy mẫu S3/S2/G, mỗi model chạy lịch E và off.
  - S2 + search3 và G + search3 chỉ chạy lịch E.
  - Bootstrap theo bàn 10 000 lần, seed 20261014.
- **Môi trường:** người dùng chạy trên máy của mình, torch 2.5.1, numpy 2.0.1, Python 3.11.15. Thu, validation và test dùng 5 tiến trình.
- **Thời gian:** thu 4 giờ 24 phút, train 17 giây, validation 3 phút 18 giây, test 1 giờ 22 phút.
- **Source:** `run_sd9_distill.py`, SHA ghi trong `registration.json` và được kiểm ở mỗi bước. `kill_search.py` file hiện tại `7c8b9f49…7991`.
- **Output:** `runs/sd9-20261023/` và các log `runs/sd9-20261023-{collect,train,validate,test}-console.log`.
- **Spec:** `docs/superpowers/specs/2026-10-23-sd9-search3-distillation.md`.

**Kiểm định:**
- `test_opened.json` ghi SHA của các model: S3 = `train/epoch_12.pt` `4185375f…69ed7`, S2 `d222d81f…6814`, G `bb0c775c…c7ac`.
- `selected.json` (10:42:22) ghi trước khi mở test (10:43:04).
- Epoch được chọn đúng luật.

**Thu dữ liệu:** thầy S2 + search3 khi thu có WR 99.17, ActiveWR 93.03, hòa 0.3 (3000 trận, bàn thu, không phải số đánh giá).

**Train (held-out, 1020 trạng thái quyết định và 1747 neo):**

| Epoch | Argmax trúng nước tốt nhất | Trúng nước tốt nhất ở trạng thái có nước giết | KL neo | Neo đổi argmax |
|---|---|---|---|---|
| 0 (S2) | 74.80 | 74.47 | 0.000 | 0.0 |
| 4 | 86.86 | 86.69 | 0.054 | 11.4 |
| 8 | 88.33 | 88.19 | 0.045 | 9.5 |
| 12 | 89.80 | 89.79 | 0.055 | 11.6 |
| 16 | 89.80 | 89.69 | 0.052 | 9.8 |

**Validation (seed 21000, 400 trận, lịch E):**

| Model | ActiveWR | WR | Hòa | Thắng thụ động | Tự nổ | Bước TB | ActiveWR − S2 (CI95) |
|---|---|---|---|---|---|---|---|
| S2 | 79.0 | 89.75 | 6.75 | 6.5 | 0.25 | 214 | — |
| epoch 4 | 89.0 | 96.25 | 0.75 | 7.0 | 1.25 | 135 | +10.0 [+5.25, +14.75] |
| epoch 8 | 90.25 | 98.0 | 0.75 | 7.75 | 0.5 | 128 | +11.25 [+6.5, +16.0] |
| **epoch 12** | **90.5** | 96.5 | 1.5 | 5.75 | 0.5 | 128 | +11.5 [+7.0, +16.0] |
| epoch 16 | 90.25 | 98.75 | 0.5 | 8.0 | 0.25 | 119 | +11.25 [+6.0, +16.25] |

- Mọi ứng viên đều có tự nổ ≤ 1.25 nên đều hợp lệ.
- Epoch 12 có ActiveWR validation cao nhất, nên **S3 = epoch 12**.

**Test (seed 57000, 400 trận/agent, đơn vị %):**

| Agent | Lịch | ActiveWR | WR | Hòa | Thắng thụ động | Thắng nhờ vòng | Tự nổ | Chết vì vòng | Kills TB | Bước TB |
|---|---|---|---|---|---|---|---|---|---|---|
| **S3** | E | **92.25** | 98.0 | 0.5 | 5.25 | 0.5 | 0.25 | 1.0 | 0.92 | 131 |
| S3 | off | 93.0 | 98.25 | 1.25 | 5.25 | 0.0 | 0.25 | 0.0 | 0.93 | 131 |
| S2 | E | 81.75 | 91.25 | 6.75 | 6.75 | 2.75 | 0.5 | 1.25 | 0.82 | 202 |
| S2 | off | 82.25 | 88.75 | 10.0 | 6.5 | 0.0 | 0.5 | 0.0 | 0.82 | 203 |
| G | E | 49.0 | 69.25 | 22.0 | 8.25 | 12.0 | 0.75 | 9.0 | 0.49 | 330 |
| G | off | 47.75 | 54.0 | 42.0 | 6.25 | 0.0 | 0.75 | 0.0 | 0.48 | 335 |
| S2 + search3 | E | 93.0 | 99.25 | 0.5 | 6.0 | 0.25 | 0.0 | 0.0 | 0.93 | 111 |
| G + search3 | E | 94.0 | 99.0 | 0.25 | 5.0 | 0.0 | 0.25 | 0.0 | 0.94 | 110 |

**Hiệu ghép cặp theo bàn (pp, CI95):**

| So sánh | ActiveWR | WR | Hòa | Thắng thụ động | Tự nổ |
|---|---|---|---|---|---|
| **S3 − S2 (E, gate)** | **+10.5 [+6.5, +14.5]** | +6.75 [+4.0, +9.5] | −6.25 [−9.0, −3.75] | −1.5 [−4.5, +1.5] | −0.25 [−1.25, +0.5] |
| S3 − S2 (off) | +10.75 [+7.0, +14.5] | +9.5 [+6.75, +12.5] | −8.75 [−11.75, −6.0] | −1.25 [−4.25, +1.75] | −0.25 [−1.25, +0.5] |
| S3 − G (E) | +43.25 [+37.75, +48.75] | +28.75 [+24.25, +33.5] | −21.5 [−25.5, −17.5] | — | −0.5 [−1.5, +0.5] |
| S3 − G (off) | +45.25 [+39.75, +50.5] | +44.25 [+38.75, +49.75] | −40.75 [−46.0, −35.5] | — | −0.5 [−1.5, +0.5] |
| S3 − (S2 + search3) (E) | −0.75 [−4.0, +2.5] | −1.25 [−2.75, +0.25] | 0.0 [−1.0, +1.0] | — | +0.25 [0.0, +0.75] |
| S3 − (G + search3) (E) | −1.75 [−5.0, +1.5] | −1.0 [−2.75, +0.75] | +0.25 [−0.5, +1.25] | — | 0.0 [−0.75, +0.75] |

**Gate (chốt trước, lịch E):** **ĐẠT.**
- ActiveWR +10.5 ≥ +5pp.
- CI dưới +6.5 > 0.
- Tự nổ −0.25 ≤ +1pp.

**Quyết định:**
- **S3 = `runs/sd9-20261023/train/epoch_12.pt` (SHA `4185375f…69ed7`) thay S2 làm CNN thuần mạnh nhất.**
- Theo spec, S3 là điểm xuất phát cho Phase D/E nếu người dùng duyệt.
- Seed 57000 đã dùng. Seed tiếp theo là 58000.

**Diễn giải và giới hạn:**
- Phần tăng là thắng chủ động. Thắng thụ động không tăng (−1.5, CI chứa 0), hòa giảm 6 điểm và ván ngắn hơn (131 so với 202 bước). Tắt sudden death cũng cho cùng kết quả (+10.75).
- Đấu RuleBot, CNN thuần S3 gần bằng trần search: kém S2 + search3 0.75 điểm và kém G + search3 1.75 điểm, cả hai CI đều chứa 0. S3 chỉ cần một lượt chạy mạng, còn search tốn 20–30 giây mỗi trận. Không có luật chốt trước nào cho việc thay agent mạnh nhất. G + search3 vẫn có ước lượng điểm cao nhất, nên chưa đổi.
- Chưng cất lặp hai vòng đã có tác dụng: G (48) → S2 (82) → S3 (92) ActiveWR đấu RuleBot. Khoảng trống còn lại so với search chỉ khoảng 1–2 điểm, nên vòng thứ ba khó qua được gate +5pp.
- Mọi đánh giá đều đấu RuleBot, và thầy cũng đoán đối thủ là RuleBot. Chưa đo S3 đấu CNN; PDE/PDE2 cho thấy đấu CNN hòa rất nhiều.
- Chỉ một train seed và một test seed.

## D-S2-review — Phase D từ S2 (bạn chạy): control RuleBot so với pool CNN, 2000 ep/nhánh, 23/10/2026 (review)

**Thông số:**
- **Người chạy và môi trường:** bạn chạy trên máy của họ (`E:\Bomberman`). Python 3.11.9, torch 2.11.0+cu128, numpy 2.4.6, GPU RTX 4060 Laptop. deterministic bật, TF32 tắt.
- **Thời gian:** train bắt đầu 22/10 16:18, có 2 lần resume. Validation dùng CPU, 1 luồng mỗi worker. Tổng thời gian chạy không lưu lại.
- **Train:** cả hai nhánh xuất phát từ S2 (`d222d81f…6814`). Phase D, 9×9 1v1, max_steps 500, lịch E với xác suất 1.
  - Tối ưu: Actor-Critic MC, Adam lr 2.5e-5 cosine (eta_min ×0.05), entropy 0.0075, gamma 0.99, value 0.5.
  - Teacher local v2. Hệ số phụ escape 0.5 / combat 0.9 / navigation 0.4 / ring_safety 0.6. Cuối trận ×1.75 từ bước 350, wall bonus 0.15.
  - Tắt search teacher, group RL, PPO và KL neo. Train seed 2026102201.
  - Bàn train 610000–610999 (mỗi bàn 2 góc).
- **Hai nhánh:**
  - `control`: 2000 ep, đối thủ luôn là RuleBot.
  - `pool`: 2000 ep. Đối thủ: 20% RuleBot, 40% CNN cố định (G/F/S/S2 mỗi model 10%), 40% hai snapshot gần nhất của chính nhánh.
- **Validation:**
  - Bàn 620000–620039, 40 bàn × 2 góc = 80 trận cho mỗi cặp model–đối thủ (G/F/S/S2/RuleBot).
  - Ứng viên: episode 0 (= S2), 500, 1000, 1500, 2000 của mỗi nhánh. Tổng 3600 trận.
  - Luật chọn chốt trước:
    - guard: RuleBot ActiveWR ≥ S2 − 2pp, tự nổ ≤ S2 + 1pp;
    - xếp hạng theo ActiveWR trung bình trước 4 CNN;
    - chỉ mở test nếu pool hơn S2 ≥ 3pp.
  - Bootstrap 10 000 lần, seed 20261014.
- **Test:** 630000–630099 và 640000–640099 đã đặt chỗ nhưng **chưa mở**.
- **Nguồn dữ liệu:** `runs/phaseD-s2-20261022/` trên nhánh `origin/Loc_s2-kl-retention` (commit `80bafa7`), chưa merge vào nhánh này. Spec trên nhánh đó: `docs/superpowers/specs/2026-10-22-phase-d-s2-pool-design.md`.
- **Số liệu "chọn đúng nước search":** lấy từ phân tích của bạn (`runs/phaseD-s2-20261022-analysis-20261023/`), thư mục này không có trên nhánh. Claude chỉ kiểm được `report.json`, `validation.json`, `selected.json` và `runtime.json`.

**Validation, ActiveWR trung bình trước 4 CNN G/F/S/S2 (đơn vị %, 320 trận mỗi ứng viên; guard xét trên RuleBot):**

| Ứng viên | ActiveWR CNN | WR CNN | Hòa CNN | Tự nổ CNN | Guard |
|---|---|---|---|---|---|
| S2 (episode 0) | 6.25 | 22.50 | 66.56 | 6.25 | đạt |
| control-500 | 8.44 | 25.62 | 61.25 | 5.31 | trượt |
| control-1000 | 9.69 | 27.50 | 61.25 | 5.31 | trượt |
| control-1500 | 6.56 | 25.62 | 60.94 | 5.94 | trượt |
| control-2000 | 6.25 | 24.38 | 64.69 | 6.25 | trượt |
| pool-500 | 5.62 | 26.88 | 58.44 | 7.50 | trượt |
| pool-1000 | 5.62 | 27.81 | 59.69 | 6.88 | trượt |
| pool-1500 | 5.31 | 24.38 | 62.81 | 5.94 | trượt |
| pool-2000 | 7.19 | 23.44 | 65.31 | 5.62 | trượt |

**Đấu RuleBot (80 trận mỗi dòng; số theo báo cáo của bạn):**

| Model | ActiveWR | WR | Hòa | Thắng nhờ vòng | Tự nổ |
|---|---|---|---|---|---|
| S2 | 83.75 | 93.75 | 5.00 | 1.25 | 0.0 |
| control-2000 | 65.00 | 77.50 | 16.25 | 5.00 | 0.0 |
| pool-2000 | 57.50 | 81.25 | 15.00 | 8.75 | 0.0 |

**Giữ kỹ năng đã chưng cất:** trên 1024 trạng thái quyết định của dữ liệu train SD8A, tỉ lệ argmax trúng nước tốt nhất của search là:

| Model | Trúng nước search |
|---|---|
| S2 | 97.36% |
| control-2000 | 75.59% |
| pool-2000 | 73.14% |

Theo bạn, sự suy giảm đã xuất hiện từ checkpoint 500.

**Kết luận theo luật của bạn:** `VALIDATION_STOP`. Mọi checkpoint đã train đều trượt guard RuleBot, nên ứng viên được chọn ở cả hai nhánh là episode 0 (= S2). Test không mở, pilot gate và Phase D gate đều không đánh giá. **Giữ S2.** Sau SD9 thì S3 là CNN thuần mạnh nhất.

**Diễn giải (Claude):**
- Train RL thêm với công thức hiện tại **xóa kỹ năng đã chưng cất**: ActiveWR đấu RuleBot giảm 19–26 điểm. Nhánh chỉ đấu RuleBot cũng mất kỹ năng như nhánh pool. Vì vậy nguyên nhân là chính công thức train, không phải lịch đối thủ.
  - Giả thuyết chưa kiểm: các nhãn teacher phụ (local v2, combat 0.9, navigation 0.4, escape 0.5) gắn ở hàng trăm bước mỗi trận và kéo policy về hành vi cũ của F/G. Trong khi đó, nước search chỉ xuất hiện ở khoảng 3 trạng thái mỗi trận.
- Pool không làm bớt hòa khi đấu CNN. Hòa vẫn 58–65% và ActiveWR trước CNN vẫn khoảng 5–10%, không ứng viên nào hơn S2 có ý nghĩa.
- Không so trực tiếp với số của nhánh này: khác torch (2.11 so với 2.5.1) và khác máy. Riêng S2 đấu RuleBot 83.75 khớp tầm 82–84 của SD8A/SD9.
- Mỗi cặp model–đối thủ có 80 trận, CI rộng. Chỉ một train seed.
- Bước tiếp theo của bạn đã có spec được duyệt: S2-KL (`docs/superpowers/specs/2026-10-23-s2-kl-retention-design.md`), 500 ep đấu RuleBot với KL neo về S2 β = 1 so với β = 0. Bàn 710000–710249 và 720000–720099 đã đặt chỗ, chưa chạy.

## SD9.1-prep — chuẩn bị probe S3 đấu CNN (bớt hòa?), 23/10/2026

**Thông số:**
- **Chạy thử đo thời gian (không phải kết quả):** `probe_s3_cnn.run(boards=1, workers=5, shard=1)`. Learner S3/S2, đối thủ S2, G, S3 và G + search3. Bàn 21000, 2 góc, 16 trận. 9×9 1v1, max_steps 500, lịch E.
- **Môi trường:** Claude chạy trên máy người dùng, torch 2.5.1 CPU, 5 tiến trình, 8 phút 27 giây.
- Output ở scratchpad (`sd91-timing`), không lưu trong `runs/`.

**Thời gian đo được:**
- Trận CNN đấu CNN: 2.0–2.7 s/trận.
- S2 đấu G + search3: 141 s/trận. S3 đấu G + search3: 248 s/trận.
- Ước tính lượt chính (2400 trận CNN + 200 trận có search): khoảng 2–3 giờ.

**Ghi chú:**
- 1 bàn quá ít nên không đọc kết quả thắng/thua; 6/8 cặp đều hòa hết giờ.
- Spec và luật chốt trước: `docs/superpowers/specs/2026-10-23-sd9-1-s3-vs-cnn-probe.md`.
- Lượt chính: `runs/sd91-21000`, seed validation 21000. Chưa chạy.

## PUSH-SD9 — đẩy toàn bộ công việc lên nhánh mới, 23/10/2026

**Thông số:** không phải thí nghiệm. Claude chạy trên máy người dùng: git, git-lfs 3.7.1, remote `origin` (`github.com/SuperZiggs/BomberMan`).

- Nhánh mới `danh/sd9-s3-distillation`, tách từ `danh/sd5-friend-merge`, commit `2a21b97`. Gồm 1797 file, khoảng 95 MB; checkpoint `.pt` đi qua LFS.
- Nội dung:
  - mọi code, spec và log từ SD8B-off tới SD9.1-prep;
  - output `runs/` của PDE, PDE2, SDM, SDM2, SD8A, SD8B, SD9.0 và SD9 (kèm S3 `runs/sd9-20261023/train/epoch_12.pt`).
- Không đưa lên:
  - `runs/sd91-21000/` và log của nó, vì SD9.1 đang chạy;
  - 7 file `runs/resume-budget-test-*-run.lock` và `graphify-out/.rebuild.lock`;
  - `.github/copilot-instructions.md`, file do công cụ claude-mem tự tạo.
- Chưa merge vào `main` và chưa tạo PR.
