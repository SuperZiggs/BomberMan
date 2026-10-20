# Ban giao session: Phase C-SD fallback -> Phase D

Cap nhat: 2026-10-16

## 1. Muc tieu cua session tiep theo

Tiep tuc phat trien tu dung moc Phase C-SD da fallback va da day len GitHub, sau do chuyen sang Phase D 1v1 self-play. Khong quay lai cac thu nghiem temporal planner va khong thay doi checkpoint Phase C goc.

Session moi can doc tai lieu nay va `docs/phase-c-sd-best-fallback.md` truoc khi sua code hoac train.

## 2. Moc source phai su dung

- Repository: `E:\Bomberman`
- Branch: `codex/phasec-sd-best-fallback`
- Commit moc: `a6650a02dc215a5b90b48146fedd09420fe9ff3b`
- Remote branch: `origin/codex/phasec-sd-best-fallback`
- GitHub: `https://github.com/SuperZiggs/BomberMan/tree/codex/phasec-sd-best-fallback`

Truoc khi lam viec, xac minh:

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
```

Ket qua mong doi la branch va commit dung nhu tren. Cac thu muc `runs/...` cu va `tmpqxveif3_` dang untracked la artifact lich su; khong xoa, khong reset va khong dua chung vao commit.

## 3. Checkpoint Phase C-SD duoc khoa

- Thu muc dong bang: `runs/phaseC-sd-best-fallback`
- Checkpoint dung de warm-start Phase D: `runs/phaseC-sd-best-fallback/best_phaseC.pt`
- SHA-256: `0315A56F09079333D936DB5F55587093F1B6415ED6355D2EF6AE2AB63F8B1483`
- Ket qua danh gia goc: `runs/phaseC-sd-best-fallback/evaluation-200.json`
- Cau hinh goc: `runs/phaseC-sd-best-fallback/config.json`

Khong train tiep vao thu muc nay va khong ghi de `best_phaseC.pt`. Moi output Phase D phai nam trong mot thu muc moi, vi du `runs/phaseD-sd-selfplay-v1`.

Co the xac minh checkpoint truoc khi train:

```powershell
Get-FileHash .\runs\phaseC-sd-best-fallback\best_phaseC.pt -Algorithm SHA256
```

## 4. Ket qua Phase C-SD hien tai

Danh gia 200 tran, seed `31000`, board 9x9, 2 agents, toi da 500 buoc:

| Chi so | Ket qua | Muc tieu cuoi da dat ra |
|---|---:|---:|
| Win rate | 67.5% | >= 70% |
| Active win rate | 51.0% | >= 55% |
| Draw | 21.5% | <= 15% |
| Suicide | 0.5% | <= 1.5% |
| Sudden Death death | 13.0% | <= 10% |

Day la checkpoint tot nhat da duoc chon lam moc fallback, nhung chua dat toan bo muc tieu tham vong 70/55/15/10. Quyet dinh hien tai la ngung tune Phase C va dung checkpoint nay de bat dau Phase D, khong sua lai ket qua thanh PASS.

## 5. Sudden Death dang duoc giu nguyen

Source fallback van co:

- Sudden Death trong environment.
- Observation 13 kenh, gom thong tin canh bao vong thu hep.
- Lich Schedule E: `start=475`, `interval=25`, `warning=40`.
- Xac suat Sudden Death trong train Phase C la `0.75`.
- Static pressure/two-position attack supervision va target-aware checkpoint selection.

Source fallback da loai bo cac thu nghiem muon hon:

- Temporal planner/follower.
- Mo phong delayed bomb placement.
- Resumable evaluation lifecycle.
- Launcher/test/tai lieu temporal lien quan.

Khong phuc hoi cac phan temporal neu chua co yeu cau va thiet ke moi duoc chap thuan.

## 6. Phase D trong code hien tai

`train_actor_critic.py` dinh nghia Phase D nhu sau:

- Van la board 9x9, 2 agents.
- Learner luan phien hai goc spawn.
- Doi thu duoc chon giua RuleBot va snapshot trong pool.
- Khi pool co snapshot, Phase D dung RuleBot voi xac suat 50%, nguoc lai chon snapshot.
- Snapshot moi duoc luu theo `snapshot_every` va them vao live pool.
- File dau ra chinh la `best_phaseD.pt` va `final_phaseD.pt`.

Luu y quan trong:

- Khi dung mot `checkpoint_dir` moi, pool ban dau rong; training se bat dau voi RuleBot va chi co snapshot sau lan luu dau tien.
- Phai truyen ro `--resume_from runs/phaseC-sd-best-fallback/best_phaseC.pt`; khong dua vao auto warm-start tu output Phase D moi.
- Bao cao cuoi Phase D hien tai dung gate legacy `WR vs RuleBot >= 70%` va round-robin voi snapshot. No chua thay the danh gia Sudden Death 200 tran theo cac chi so WR, ActiveWR, Draw, Suicide va SD death.
- `train_actor_critic.py` dang bao cao gate cuoi tren model final; viec chon model de danh gia chinh thuc phai load `best_phaseD.pt` rieng.

## 7. Huong trien khai Phase D de xuat

Khong bat dau train dai ngay trong session moi. Thuc hien theo thu tu:

1. Xac minh branch, commit, hash checkpoint va trang thai working tree.
2. Chot cau hinh Phase D, seed, so episode va gate danh gia voi nguoi dung.
3. Tao launcher PowerShell Phase D rieng, khong sua launcher Phase C da khoa.
4. Launcher phai train vao thu muc output moi, luu best checkpoint, sau do load `best_phaseD.pt` danh gia dung 200 tran.
5. Giu Schedule E cho ca train va evaluation tru khi co bang chung moi.
6. Luu `config.json`, `metrics.jsonl`, `validation_metrics.jsonl`, `best_validation.json` va file evaluation 200 tran trong cung thu muc Phase D.
7. Chi ket luan tot/xau sau khi so sanh voi moc Phase C tren cung runtime, board, seed policy, hai goc spawn va Sudden Death config.

Cau hinh khoi diem de thao luan, chua phai lenh da duoc phe duyet de train:

```text
phase=D
board_size=9
num_agents=2
max_steps=500
resume_from=runs/phaseC-sd-best-fallback/best_phaseC.pt
checkpoint_dir=runs/phaseD-sd-selfplay-v1
lr=0.000025
entropy_coef=0.0075
snapshot_every=200
validation_every=250
validation_episodes=200
sudden_death_probability=0.75
sudden_death_start=475
sudden_death_interval=25
sudden_death_warning_steps=40
```

So episode, training seed, validation seed, evaluation seed va gate Phase D can duoc chot o session moi truoc khi tao launcher.

## 8. Tieu chi danh gia bat buoc

Danh gia ung vien Phase D tren 200 tran seeded, chia deu hai goc spawn. Bao cao it nhat:

- Win rate.
- Active win rate.
- Draw rate.
- Suicide rate.
- Sudden Death death rate.
- Kills/game, bombs/game, average steps.
- Ket qua tach theo tung goc spawn.
- Ket qua dau RuleBot va ket qua self-play/round-robin voi snapshot pool.

Muc tieu Phase C tham chieu van la:

- WR >= 70%.
- ActiveWR >= 55%.
- Draw <= 15%.
- Suicide <= 1.5%.
- Sudden Death death <= 10%.

Phase D khong duoc xem la cai tien neu chi tang WR nhung lam ActiveWR giam, draw tang, suicide tang hoac SD death tang. Can giu mot evaluation baseline cua checkpoint Phase C tren cung seed danh gia Phase D de so sanh cong bang.

## 9. Trang thai test tai moc fallback

- Fallback acceptance: 9/9 passed.
- Target/attack/Sudden Death acceptance suites da passed.
- `py_compile` passed cho `bomberman_env.py`, `rule_based_bot.py`, `train_actor_critic.py`, `evaluate.py`.
- Full historical suite: 114 tests, 3 failures trong `test_combat.py`.

Ba failure lich su da duoc giu co y de khong thay doi source tao checkpoint:

- `CombatChecks.test_danger_targets_escape_instead_of_attack`
- `CombatChecks.test_safe_followup_traps_opponent_and_survives_simultaneous_counter_bomb` voi learner corner 0
- Cung test tren voi learner corner 1

Session moi khong nen sua ba test/failure nay nhu mot tac vu phu khi dang chuan bi Phase D. Neu Phase D can thay doi hanh vi lien quan, phai tach thanh quyet dinh co test va bang chung rieng.

## 10. Nguyen tac Git va artifact

- Mốc fallback da commit va push rieng; khong force-push hoac rewrite commit moc.
- Neu sua pipeline Phase D, tao commit moi tren branch hien tai hoac tao branch moi bat dau tu commit moc.
- Khong commit cac thu muc run cu khong lien quan.
- Khong xoa artifact train truoc khi da doc log va xac nhan file can giu.
- Khong push checkpoint moi cho den khi nguoi dung yeu cau.
- Truoc moi commit: kiem tra `git diff`, test muc tieu, `git diff --check` va danh sach file staged.

## 11. Prompt ngan de mo session moi

```text
Hay doc E:\Bomberman\docs\SESSION_HANDOFF_PHASE_D.md va E:\Bomberman\docs\phase-c-sd-best-fallback.md. Tiep tuc tu branch codex/phasec-sd-best-fallback, commit a6650a02dc215a5b90b48146fedd09420fe9ff3b, dung checkpoint E:\Bomberman\runs\phaseC-sd-best-fallback\best_phaseC.pt lam warm-start. Khong sua/xoa moc fallback va cac run cu. Truoc tien hay kiem tra hien trang, sau do de xuat va chot cau hinh Phase D 1v1 self-play co Sudden Death, best-checkpoint va evaluation 200 tran trong mot launcher rieng; chua train cho den khi toi dong y.
```
