# T4-D1 — U4/P4 với cùng ngân sách rollout và KL guard

D1 DONE/REVIEWED: `ppo-full-20261015-220313`. Đủ 200 episode/nhánh từ B, Adam updates U4/P4=800/800; best episode U4/P4=200/200. Test400/model, seed37000–37199/cảA0-A1: ActiveWR B/U4/P4=37.50%/37.00%/32.00%. Gate FAIL; giữ checkpoint B. Runtime 106.89 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/ppo-full-20261015-220313/decision_report.md).

D1 giữ MC raw/critic MC/local teacher/reward cũ. U4/P4 cùng tối đa4 epochs và post-update KL guard target0,02/threshold0,03; chỉ khác clipping epsilon0,20. 241 regression và25 focused metadata/compat tests PASS; CPU/CUDA smoke20episode/80updates mỗi nhánh đã review. Có 500-step/4-epoch memory fixture, hủy giữa epoch khôi phục Adam, exact PPO resume qua tiến trình mới CPU/CUDA, full-state/source/ZIP/locked-model guards. CNN inference và Actor-Critic mặc định giữ nguyên.

## Protocol khóa trước khi train

| Mục | Giá trị |
|---|---|
| Nhánh | `control` = U4 (`ratio` chưa clip), `ppo` = P4 (`ppo`, epsilon 0,20) |
| Warm-start | Cả hai từ `runs/pressure-C-next/final_phaseC.pt`, SHA256 `93bdf278…`; Adam mới riêng mỗi nhánh |
| Bàn | Phase C / 9×9 / 2 agents / RuleBot / max500 |
| Train | 200 episode/nhánh, seed 2026101507, LR 1e-4, gamma 0,99 |
| Estimator | MC advantage raw, critic MC; không normalize/GAE |
| Reward/teacher | Local, wall 0,15, các reward và hệ số loss phụ giữ nguyên |
| Update | Một learner episode đầy đủ, full batch, tối đa 4 epochs |
| KL guard | Exact categorical KL sau mỗi Adam step; target 0,02, dừng epochs còn lại nếu KL > 0,03 |
| Scheduler | Cosine và entropy schedule theo episode đã đăng ký; không step scheduler theo số Adam updates |
| Validation | 100 trận seed21000 tại episode0/100/200; best và guard tự nổ hiện tại |
| Skill probe | 32 escape + 32 approach mỗi mốc validation |
| Test | B + best U4 + best P4; 400 trận/model, seed37000–37199, mỗi seed A0/A1 |
| Runtime | CUDA train, CPU một thread khi đánh giá; deterministic, TF32 tắt ở cả hai nhánh |
| Ngân sách | Tối đa150 phút: preflight15, train100, evaluation50, mọi stage cùng bị giới hạn bởi deadline tổng |

Khác biệt treatment duy nhất là clipping objective. KL guard có thể khiến số epoch/Adam updates thực tế khác nhau giữa hai nhánh; phải báo chúng cùng runtime/learner interactions. B là baseline policy giữ nguyên, không phải một đối chứng AC một-update được train lại trong D1 này.

## Cách cập nhật

Thu thập dưới một policy cố định qua safety mask. Dùng buffer D0 đã clone/detach observation, action thật CNN đã sample, mask, old log-prob/value/probabilities, reward/boundary, teacher target/group. Tính MC targets và advantages một lần cho toàn learner trajectory.

```text
ratio = exp(log_pi_current(action, stored_mask) - old_log_prob)
U4 policy_loss = -mean(ratio * fixed_advantage)
P4 policy_loss = -mean(min(ratio * fixed_advantage,
                          clamp(ratio, 0.8, 1.2) * fixed_advantage))
```

Mỗi epoch forward lại cùng observation/mask và dùng cùng targets. Mean escape/combat/navigation theo số nhãn của từng nhóm, hệ số 0,5/0,5/0,2; không chia cho mọi transition. Value coefficient0,5, entropy coefficient/schedule và grad clip5 giữ nguyên. Không gọi lại teacher/planner hoặc thay action bằng nhãn teacher trong epochs.

KL được tính **sau update**, so với old policy ở đầu rollout trên tất cả action hợp lệ của mask đã lưu. Khi vượt0,03, step vừa thực hiện vẫn được giữ và các epoch còn lại dừng. Đây không phải ràng buộc bảo đảm KL luôn≤0,03 hoặc bảo đảm agent thắng. Cùng quy tắc áp cho U4/P4.

Metrics `ratio_update` có số epochs thực/requested, `early_stop`/`guard_exceeded`, ratio mean/min/p95/max, clip_fraction, entropy/value/policy/total loss, gradient norm và KL trước/sau update từng epoch. Clip fraction là phần ratio nằm ngoài [0,8;1,2] ở cả U4/P4; U4 vẫn log chỉ số này dù không clip. Advantage/EV là chẩn đoán policy cũ trước rollout update; value loss theo epoch là prediction hiện tại.

## Checkpoint và resume

`best/final_phaseC.pt` là weights để inference. `training_state.pt` lưu model/Adam/scheduler/RNG/history và context log ở cuối episode đã hoàn tất, sau validation nếu đến mốc. Optimizer steps có thể từ200 đến800 cho200episodes; số thực phải bằng tổng epochs đã ghi, không giả định800.

Hủy giữa các epochs khôi phục model/Adam trước update của episode. Khi dùng full state, trainer còn khôi phục checkpoint đã xác nhận trước episode, gồm RNG và lịch sử; episode chưa hoàn tất được thu lại khi resume. Log ghi vượt checkpoint/torn gzip được giữ trong recovery rồi dựng lại từ context. NaN/Inf dừng lượt, không dùng đoạn lỗi để chọn candidate.

Resume yêu cầu cùng source/config/runtime/thiết bị. Giữ deadline tổng và từng stage ban đầu, tính cả thời gian gián đoạn. Khóa writer ngăn hai tiến trình cùng ghi. Best được khóa hash trước test; không được chọn lại model sau khi mở test. Evaluation hoàn tất có thể dùng cache với model/seed/count khớp; evaluation dở được chạy lại với cùng model đã khóa.

## Lệnh chạy

Runner từ chối output đã tồn tại; không ghi đè kết quả cũ.

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
& $py -m unittest test_ppo_d0 test_ppo_d1 test_ppo_pilot test_training_state -q
if ($LASTEXITCODE -ne 0) { throw 'D1 tests failed.' }
$ppoStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py -u .\run_ppo_pilot.py --smoke --device cpu --output-dir "runs\ppo-smoke-cpu-$ppoStamp"
if ($LASTEXITCODE -ne 0) { throw 'CPU smoke failed.' }
& $py -u .\run_ppo_pilot.py --smoke --device cuda --output-dir "runs\ppo-smoke-cuda-$ppoStamp"
if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke failed.' }
# Test37000–37199 đã mở trong lượt DONE ở đầu trang; không chạy lại để chọn model khác.
& $py -u .\run_ppo_pilot.py --device cuda --budget-minutes 150 --output-dir "runs\ppo-full-$ppoStamp"
if ($LASTEXITCODE -ne 0) { throw 'Pilot chưa đủ protocol; kiểm tra summary, giữ B.' }
```

Chỉ resume lượt đang dở và còn deadline; lượt DONE không cần resume. Nếu tiến trình bị ngắt, sau khi mở lại PowerShell:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
$ppoRun = (Get-Content -LiteralPath 'runs\ppo-last-path.txt' -Raw).Trim()
& $py -u .\run_ppo_pilot.py --resume --device cuda --budget-minutes 150 --output-dir $ppoRun
```

Không tạo timestamp mới khi resume. Lượt quá deadline hoặc DONE sẽ bị từ chối. Muốn tiếp tục nghiên cứu với ngân sách mới cần đăng ký experiment mới; không đổi horizon để tự kéo dài schedule.

## Gate và đầu ra

P4 chỉ đủ tín hiệu cho replication khi ActiveWR tăng≥5 điểm phần trăm so với cả B và U4, hai cận dưới CI95% bootstrap theo cặp seed bàn >0, tự nổ quan sát không cao hơn cả hai. Giữ cả A0/A1 trong mỗi cụm bootstrap; báo thêm hòa, kills, passive wins và từng góc. Một train seed chưa đủ chứng minh ổn định; không tự thay B ngay cả khi PASS.

Thiếu train/validation/skill/test, hash đổi hoặc stage hết thời gian: INCOMPLETE, giữ B. Smoke luôn `full_protocol=false`, không chọn winner. Chưa thêm PPO minibatch, rollout nhiều episode, value clipping, GAE, normalization, reward mới hoặc opponent pool vào D1.
