# T4-B0 — Kiểm tra tính đúng của GAE

**DONE / PASS về tính đúng, 15/10/2026.** 167 tests PASS; CPU/CUDA smoke hoàn tất cho MC, GAE1 và GAE0,95, mỗi chế độ20episode/20updates. [Báo cáo B0](/D:/fpt/subject/BomberMan/runs/gae-b0-implementation-20261015/B0_report.md). Checkpoint B giữ nguyên; chưa mở pilot B1 hay dùng GAE làm mặc định.

## Phạm vi và giao diện

Trainer thêm `advantage_estimator='mc'|'gae'` (mặc định `mc`) và `gae_lambda=0.95`. CLI tương ứng `--advantage_estimator gae --gae_lambda 1`. Chỉ thay advantage trong loss actor. Critic luôn dùng MC returns cũ, teacher/reward/mask/CNN/loss phụ và optimizer không đổi. Diagnostics tự bật cho GAE; normalization vẫn tắt trừ khi người gọi bật riêng. B0 không thử phối hợp normalization.

Helper `gae_advantages` nhận các vector detached, next values và **hai mask riêng**:

```text
delta[t] = r[t] + gamma * bootstrap_allowed[t] * V_old(next_state[t]) - V_old(state[t])
A[t] = delta[t] + gamma * lambda * continue_segment[t] * A[t+1]
```

Đối chiếu công thức16–18 trong [paper GAE của Schulman và cộng sự](https://arxiv.org/html/1506.02438): lambda0 là TD residual, lambda1 với terminal value0 bằng MC return trừ V. Đây là kiểm tra estimator, chưa phải bằng chứng GAE cải thiện Bomberman.

Trainer hiện cập nhật một lần sau toàn learner trajectory. `V(next_state)` lấy từ forward bước kế tiếp dưới cùng policy chưa cập nhật; không cần nhân đôi buffer obs/next_obs. Lưu thêm boundary của từng transition. Khi learner chết, buffer kết thúc dù env FFA còn chạy; không thêm reward/value của các bước đã chết. Reset tạo buffer mới. Thắng/thua/hòa hoặc hết max_steps của trận dùng terminal value0 như MC hiện tại. Giới hạn20 bước của smoke cũng là max_steps của env, không phải artificial rollout cut.

Helper hỗ trợ **artificial cut**: bootstrap từ next value thật nhưng continuation=false để không nối sang đoạn/reset sau. Unit tests kiểm tra trường hợp này; collector hiện chưa cập nhật giữa trận hoặc thu nhiều episode trong một rollout. Collector đó và raw obs/mask buffer dành cho các task rollout/PPO sau. Không áp quy tắc terminal của game500 bước cho mọi truncation trong môi trường khác.

## Test và sai số số học

- Tính tay lambda0/0,5; lambda1 so MC trên1/3/500 bước; CPU/CUDA, float64.
- Next value tại terminal bị bỏ qua; artificial cut giữ next value đúng, không rò advantage/reward từ reset.
- Terminal owned kill đến muộn, learner chết khi còn hai agent sống, timeout500 còn hai agent, hủy ở giữa episode.
- Advantage/delta detach, critic vẫn nhận đúng gradient, targets và inputs/RNG không bị sửa.
- Mặc định MC có actions/rewards/metrics và mọi tensor weights **bit-identical** với source trước B0 trên test CPU hai episode20 bước.
- Một frozen first trajectory20 bước: MC so GAE1 có cùng actions/rewards/critic/auxiliary losses; kiểm tra loss, gradient trước clipping và weights sau một Adam step CPU/CUDA. GAE0,95 chỉ đổi policy loss trên cùng dữ liệu; explained variance vẫn tính từ MC residual, không dùng GAE variance thay cho critic error.

GAE1 dùng tổng TD residual float32, còn MC hiện tại tích lũy Python float rồi cast float32; không yêu cầu bit-identical cho hai đường này. Test dùng loss abs1e-5, gradient atol2e-6/rtol1e-4, weights sau Adam atol5e-6/rtol1e-4. Adam có thể khuếch đại rounding ở gradient gần0. Diagnostic frozen run đo max loss/gradient/weight delta:

| Thiết bị | Policy loss abs delta | Gradient max abs | Weights max abs sau Adam |
|---|---:|---:|---:|
| CPU | 4,47e-8 | 1,49e-7 | 4,78e-6 |
| CUDA | 3,73e-9 | 2,79e-8 | 1,17e-7 |

Test đã bắt lỗi `gamma * bool_mask` làm hệ số bị round sang default float32 trong recurrence float64. Đã sửa bằng nhân với accumulator floating trước mask. Không sửa baseline MC để ép parity.

## Smoke B0

`run_gae_b0.py`: ba chế độ `mc`, `gae1`, `gae095`, mỗi chế độ20episode/max20/9×9/2agents/RuleBot/local/LR1e-4/seed2026101503/B. Mỗi chế độ một Adam mới, diagnostics bật, normalization tắt, validation và final tournament đều tắt. CPU/CUDA chạy riêng, lưu config/metrics/source/checkpoints/summary và ZIP riêng. Từ chối output đã tồn tại, sai hash B, thiết bị không khả dụng; assert20updates, weights hữu hạn và boundary counters.

Không có test400 trận, không mở seed đánh giá mới, không chọn best bằng validation và không thay B. Các file `best_phaseC.pt` trong smoke là output kỹ thuật khi validation tắt; không được dùng làm candidate đã chứng minh cải thiện. `full_protocol=false`, quyết định luôn giữ B. Không hỗ trợ resume chính xác Adam/RNG.

Lệnh tham khảo; thay output nếu đã tồn tại:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
& $py -m unittest test_gae -q
if ($LASTEXITCODE -ne 0) { throw 'B0 tests failed.' }
& $py -u .\run_gae_b0.py --device cpu --output-dir runs\gae-b0-smoke-cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU smoke failed.' }
& $py -u .\run_gae_b0.py --device cuda --output-dir runs\gae-b0-smoke-cuda
```

B0 qua test/smoke mới mở task **B1**: runner đối chứng MC raw với actor GAE0,95/critic MC, normalization tắt, cùng B/local/reward và seed mới đăng ký trước test. B1 chưa có runner/pilot ở bước này; không tái dùng control cũ sau khi source đã thay. Seed34000–34199 đã dùng ở A1;31000/wood_branch không được mở ở B0.

Kết quả smoke đã review: CPU15,06s/394learner steps mỗi chế độ (18timeouts,2match ends trong các chế độ GAE); CUDA14,83s/400steps mỗi chế độ (20timeouts). Mỗi ZIP22files, CRC/byte equality/source+B/checkpoint hashes và metrics khớp. Các ca learner-death, kill trễ, timeout500 và cut/reset nằm trong unit/integration tests, không suy từ smoke ngắn. Logs/source-before/diff/numerical diagnostic ở `runs/gae-b0-implementation-20261015`; hai thư mục smoke và ZIP giữ riêng.
