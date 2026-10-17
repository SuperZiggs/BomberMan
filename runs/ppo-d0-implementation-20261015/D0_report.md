# T4-D0 — DONE/PASS về tính đúng

Đã triển khai buffer rollout và một update ratio chưa clip, kiểm tra với Actor-Critic trên cùng dữ liệu. **226/226 regression tests PASS**, gồm 14 test D0; smoke CPU/CUDA đều đủ hai chế độ, mỗi chế độ 20 episode/20 Adam updates. Chưa có bằng chứng PPO cải thiện ActiveWR; giữ checkpoint B và Actor-Critic mặc định.

## Quyết định sau R1

R1 đủ protocol nhưng gate FAIL. Không có cơ sở từ kết quả đó để mở R2 bỏ toàn bộ wall bonus. D0 là bước kiểm chứng độc lập cho trục optimizer: khóa MC raw/critic MC/local teacher/reward cũ, không dùng cấu hình GAE hoặc normalization chưa đạt gate. D0 không xác nhận optimizer là bottleneck chiến thuật.

## Code và kiểm chứng

`train_actor_critic.py` thêm `policy_objective=ac|ratio`, mặc định `ac`; flag được ghi trong provenance full-state và CLI. Không đổi network, observation, reward, mask, inference hoặc teacher. Caller cũ nhận mặc định; test so source trước D0 cho actions/rewards/metrics/weights khớp bit trên fixture CPU.

Nhánh ratio thu thập policy cố định, clone/detach obs, sampled action, mask, old log-prob/value/probabilities, reward, boundary và teacher target/group. Sau learner episode, tính MC targets/advantage một lần rồi forward full batch. Group means và coefficients giữ 0,5 escape/0,5 combat/0,2 navigation; tied teacher actions nhận phân phối đồng đều. Update không đọc lại env, mask hoặc teacher. Một Adam update/episode, grad clip 5 và entropy schedule cũ.

Test gồm float64 math/gradient/Adam parity; CNN first trajectory trên CPU/CUDA; một action hợp lệ; nhóm nhãn hiếm và soft targets; probability/log-prob/mask/group/boundary không hợp lệ; input không alias, graph detached và RNG không bị helper sửa. Integration test bao gồm kill trễ, learner chết trước FFA kết thúc, timeout 500, hủy partial rollout và full-state uninterrupted/pause/resume trên CPU/CUDA. Regression còn kiểm tra subprocess resume AC qua tiến trình Python mới.

Hai scalar policy loss khác nhau theo công thức; không lấy việc chúng khác số để kết luận gradient sai. Float64 gradient và một Adam step khớp trong tolerance 1e-12. CNN float32 dùng gradient atol 3e-6/rtol 2e-4; weights atol 2e-5/rtol 1e-4; value/aux losses abs 2e-5. Batch reductions có rounding khác collection từng bước; Adam khuếch đại rounding khi gradient gần 0. Mặc định AC vẫn yêu cầu bit parity.

## Smoke và review

Mỗi thiết bị: `ac` và `ratio` từ B/Adam mới, 20 episode, max20, Phase C/9×9/2 agents/RuleBot/local, seed 2026101506, LR 1e-4, MC raw, wall 0,15. Một CPU thread; deterministic algorithms và TF32 tắt cùng nhau ở hai nhánh. Validation/report tournament tắt; `full_protocol=false`.

| Thiết bị | Episode / Adam mỗi chế độ | Learner steps mỗi chế độ | Runtime | Peak CUDA AC / ratio |
|---|---:|---:|---:|---:|
| CPU | 20 / 20 | 400 | 15,16 s | N/A |
| CUDA | 20 / 20 | 400 | 24,80 s | 96,52 / 102,82 MiB |

| Frozen first update | CPU | CUDA |
|---|---:|---:|
| Actions và reward | Giống nhau | Giống nhau |
| Value loss abs delta | 5,96e-8 | 2,98e-8 |
| Escape/combat loss abs delta | 0 / 0 | 0 / 0 |
| Navigation loss abs delta | 7,45e-9 | 0 |
| Gradient max abs delta | 7,15e-7 | 2,38e-7 |
| Weights max abs delta sau Adam | 2,14e-8 | 7,06e-6 |
| Ratio max abs khỏi 1 | 2,38e-7 | 1,19e-7 |

KL trước update xấp xỉ 0 (âm cỡ 1e-8 do rounding float32). Frozen first trajectory có escape/navigation, chưa có combat; combat/soft targets/group weighting được kiểm tra riêng trong unit tests và các episode smoke có nhãn combat. Không suy ra chất lượng full-game từ smoke.

Mỗi ZIP raw có 23 file; CRC và nội dung từng file khớp bản gốc. Source/current/snapshot và B SHA256 khớp. Cả bốn full states ở episode 20 có Adam step 20, scheduler last_epoch/T_max 20, reward history khớp metrics, đủ Python/NumPy/env/Torch CPU/CUDA RNG và model khớp final weights. Inference weights và full training state tiếp tục là hai định dạng riêng.

## Đầu ra và bước kế tiếp

- [CPU summary](/D:/fpt/subject/BomberMan/runs/ppo-d0-smoke-cpu-20261015-212624/summary.json), [CPU parity](/D:/fpt/subject/BomberMan/runs/ppo-d0-smoke-cpu-20261015-212624/first_update_parity.json).
- [CUDA summary](/D:/fpt/subject/BomberMan/runs/ppo-d0-smoke-cuda-20261015-212624/summary.json), [CUDA parity](/D:/fpt/subject/BomberMan/runs/ppo-d0-smoke-cuda-20261015-212624/first_update_parity.json).
- [Review độc lập](/D:/fpt/subject/BomberMan/runs/ppo-d0-implementation-20261015/analysis_results.json), [226 tests](/D:/fpt/subject/BomberMan/runs/ppo-d0-implementation-20261015/all-tests.log), [hướng dẫn](/D:/fpt/subject/BomberMan/PPO_D0_GUIDE.md).

**D1 READY/design:** triển khai đối chứng U4/P4 cùng 4 epochs, MC raw/local/reward cũ, khác clipping epsilon 0,20; cùng KL target 0,02 và early-stop threshold 0,03. Cần test mục tiêu ratio, reuse fixed targets, KL guard, cancellation/resume và smoke trước khi đăng ký pilot. D0 chưa triển khai hoặc chạy D1, chưa mở test held-out mới, chưa thay B.

Giới hạn: D0 là kiểm tra một update và smoke ngắn; chưa có clipping, nhiều epoch, rollout nhiều episode, minibatch hay đánh giá ActiveWR. Resume ratio mới được kiểm tra trong test CPU/CUDA; chưa tuyên bố đã kiểm tra ratio qua subprocess mới hoặc chuyển thiết bị/runtime.
