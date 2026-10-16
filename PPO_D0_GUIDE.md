# T4-D0 — rollout và gradient parity trước PPO

**D0 DONE / PASS về tính đúng, 15/10/2026.** 226/226 tests PASS, gồm 14 test D0. Smoke CPU/CUDA đều hoàn tất `ac` và `ratio`, mỗi chế độ 20 episode/20 Adam updates; source, ZIP và full-state checkpoint đã review. Giữ checkpoint B và Actor-Critic mặc định. [Báo cáo D0](/D:/fpt/subject/BomberMan/runs/ppo-d0-implementation-20261015/D0_report.md).

## Vì sao mở D0 sau R1

R1 đã chạy đủ protocol nhưng không đạt gate. A1 và B1 cũng chưa tạo candidate đạt gate, nên D0 khóa estimator **MC raw**, critic MC, teacher local và wall bonus 0,15. Đây là kiểm tra hạ tầng update; chưa có bằng chứng optimizer là nguyên nhân gây hòa. R2, GAE+normalization và các nhánh reward khác vẫn cần điều kiện mở riêng trong tracker.

PPO dùng surrogate objective và tái sử dụng dữ liệu cho nhiều update ([paper gốc](https://arxiv.org/abs/1707.06347)). D0 của dự án chỉ làm bước trước đó: một learner episode đầy đủ, full batch, một Adam update, ratio chưa clip. D1 mới so clipping với đối chứng có cùng số epoch.

## Phạm vi triển khai

Trainer thêm `policy_objective='ac'|'ratio'`; CLI là `--policy_objective`. Mặc định `ac` giữ loss, hành động, reward, metric và weights như trước. Inference vẫn dùng CNN qua mask, không gọi teacher hoặc planner.

Với `ratio`, lúc thu thập dữ liệu trainer lưu bản sao detached của observation, action CNN đã sample, mask tại lúc sample, old log-prob/value/probabilities, reward, terminal/boundary và teacher target/group. Các hành động teacher tương đương có target chia đều; teacher không thay action CNN. Mỗi transition có nhiều nhất một nhóm escape/combat/navigation.

Policy không cập nhật trong episode. Sau khi kết thúc learner trajectory, trainer tính MC returns và advantage một lần, rồi forward lại toàn bộ observation với **mask đã lưu**:

```text
ratio = exp(log_prob_current(action, stored_mask) - old_log_prob)
policy_loss = -mean(ratio * fixed_advantage)
value_loss = MSE(value_current, fixed_MC_return)
loss = policy_loss + 0.5*value_loss - entropy_coefficient*mean_entropy
       + 0.5*mean_escape + 0.5*mean_combat + 0.2*mean_navigation
```

Mỗi loss phụ lấy mean trên số nhãn của chính nhóm đó; nhóm không có nhãn đóng góp 0. Không lấy mean trên mọi transition rồi làm loãng nhãn combat hiếm. Giữ entropy schedule và grad clip 5 hiện tại.

Rollout kết thúc khi learner chết, kể cả khi FFA còn chạy bookkeeping; thắng/thua/hòa và timeout thật dùng terminal value 0. Hủy giữa episode bỏ rollout chưa hoàn tất. D0 chưa có artificial rollout cut hoặc gom nhiều episode.

`ratio_update` trong metrics ghi số mẫu/nhãn từng nhóm, ratio mean/min/max, sai lệch lớn nhất khỏi 1, exact categorical KL trước update, số terminal và boundary cuối. Chưa có clip fraction hoặc KL early-stop vì D0 không clip và không update nhiều epoch.

## Kiểm chứng và sai số

- Float64 trên cùng frozen data: gradient policy/critic/entropy/nhãn và một Adam step tương đương, tolerance 1e-12. Hai scalar policy loss không cần bằng nhau: `-ratio*A` khác `-log_prob*A`.
- CNN thật: action tape và reward của trajectory đầu phải bằng nhau. Gradient trước clip dùng atol 3e-6/rtol 2e-4; weights sau Adam dùng atol 2e-5/rtol 1e-4; loss critic/phụ dùng abs 2e-5. Batch forward float32 có thứ tự reduction khác forward từng bước; Adam khuếch đại rounding ở gradient gần 0. Sai số đo thật lưu ở `first_update_parity.json`.
- Actor-Critic mặc định vẫn phải khớp bit với source trước D0 trên fixture CPU; không nới điều kiện này.
- Có test mask chỉ một action, nhãn soft/tied, nhóm hiếm, action không hợp lệ, old probability/log-prob sai, snapshot không alias/không giữ graph, không gọi lại mask/teacher trong update, terminal kill trễ, learner death, timeout 500 và hủy partial rollout.
- Full-state checkpoint tiếp tục giữ Adam/scheduler/RNG. Test D0 so uninterrupted với pause/resume trên CPU/CUDA, cùng source/config/runtime/device. Không dùng `best/final_phaseC.pt` để resume chính xác.

Sai số đo ở frozen first update của smoke:

| Thiết bị | Gradient max abs | Weights max abs sau Adam | Ratio max abs khỏi 1 | Thời gian hai chế độ |
|---|---:|---:|---:|---:|
| CPU | 7,15e-7 | 2,14e-8 | 2,38e-7 | 15,16 giây |
| CUDA | 2,38e-7 | 7,06e-6 | 1,19e-7 | 24,80 giây |

KL trước update xấp xỉ 0; giá trị âm cỡ 1e-8 trong float32 là rounding, không phải KL âm về toán học. Smoke ngắn không đánh giá chất lượng policy hoặc tốc độ PPO trên full-game. Ratio resume được kiểm tra uninterrupted/pause/resume trong test CPU/CUDA; các bài subprocess mới của hạ tầng resume vẫn chạy qua ở chế độ AC.

## Runner kỹ thuật

`run_ppo_d0.py` chạy `ac` và `ratio`, mỗi chế độ 20 episode/max20, C/9×9/2 agents, LR 1e-4, seed 2026101506, cùng B, MC raw/local/reward cũ. Hai chế độ đều dùng deterministic algorithms, TF32 tắt và một CPU thread; config ghi đầy đủ runtime. So frozen first update, kiểm tra 20 Adam updates và lưu full state cho mỗi chế độ. Không có validation, test held-out hoặc chọn model cải tiến; luôn `full_protocol=false`, giữ B.

Runner từ chối output đã tồn tại, sai hash B hoặc device không có sẵn. Source/config/metrics/checkpoint/parity/summary và ZIP được giữ trong output mới. Smoke checkpoint phục vụ kiểm chứng kỹ thuật; chưa được chọn bằng validation.

Các lệnh chạy lại, nếu cần kiểm chứng sau này:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
& $py -m unittest test_ppo_d0 test_gae test_training_state -q
if ($LASTEXITCODE -ne 0) { throw 'D0 tests failed.' }
$d0Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py -u .\run_ppo_d0.py --device cpu --output-dir "runs\ppo-d0-smoke-cpu-$d0Stamp"
if ($LASTEXITCODE -ne 0) { throw 'D0 CPU smoke failed.' }
& $py -u .\run_ppo_d0.py --device cuda --output-dir "runs\ppo-d0-smoke-cuda-$d0Stamp"
if ($LASTEXITCODE -ne 0) { throw 'D0 CUDA smoke failed.' }
```

Sau D0 PASS, task riêng D1 sẽ triển khai U4/P4: cùng rollout/MC raw/teacher/reward, cùng 4 epochs và cùng KL guard, khác duy nhất ratio chưa clip hoặc clip epsilon 0,20. D1 hiện đã có implementation/tests; trạng thái smoke/pilot và lệnh ở [PPO_PILOT_GUIDE.md](/D:/fpt/subject/BomberMan/PPO_PILOT_GUIDE.md).
