# T4-B0 hoàn tất: GAE đúng theo các kiểm tra, giữ B

15/10/2026. Đã triển khai helper GAE, nối vào trainer/CLI như tùy chọn, test và smoke CPU/CUDA. B0 PASS về implementation/parity; chưa có kết quả chứng minh tăng ActiveWR. Không thay checkpoint gốc hoặc đổi mặc định MC. Task tiếp theo là chuẩn bị runner đối chứng B1.

## Thay đổi

`train_actor_critic.py`: `gae_advantages(rewards,values,next_values,bootstrap_allowed,continue_segment,gamma,gae_lambda)` trả advantage/delta detached; validate shape/dtype/device/finite và masks. Hai mask tách bootstrap V(next) khỏi nối advantage; terminal không bootstrap, cut có thể bootstrap nhưng không nối sang đoạn mới. Công thức đối chiếu [GAE, equations16–18](https://arxiv.org/html/1506.02438).

`train(...,advantage_estimator='mc',gae_lambda=0.95)` và CLI `--advantage_estimator mc|gae --gae_lambda ...`; MC mặc định. Khi bật GAE chỉ advantage actor đổi; critic giữ MC returns, explained variance dùng MC residual. Reward/teacher local/mask/CNN12channels/optimizer/schedule/auxiliary coefficients không đổi. Normalization không bật trong B0.

Trong rollout toàn episode, V(next) lấy từ value của forward kế tiếp trước optimizer update. Buffer ghi terminal của learner, không thêm bước đã chết; env FFA vẫn có thể chạy tiếp để giữ bookkeeping cũ. Hết max_steps của trận dùng value0 như MC cũ. Không lưu hai bản obs/next_obs chưa dùng. Helper đã test artificial cut; collector giữa trận/multi-episode/raw obs buffer chưa triển khai và chưa được gọi trong train hiện tại.

## Test

**167 tests PASS**, log `all-tests-final.log` (41,745s);17test mới ở `test_gae.py`, các test cũ vẫn qua.

- Tính tay lambda0/0,5; lambda1 khớp MC trên1/3/500 bước CPU/CUDA/float64, atol/rtol1e-10.
- Terminal next-value lớn bị bỏ qua; cut bootstrap next-value đúng nhưng không nối advantage; reset/segment sau không làm đổi prefix trước cut.
- Owned kill/win trễ; learner chết ở tick1 trong trận ba agents vẫn còn2sống tới tick4; timeout500 cả hai sống; hủy partial không update.
- Advantage/delta detach; critic gradient/targets không đổi trên cùng dữ liệu; inputs và RNG không bị sửa.
- Mặc định MC actions/rewards/metrics và mọi weights bit-identical với source trước B0 trên CPU2episode20bước.
- Frozen first update20bước từB: MC/GAE1 cùng actions/rewards/critic/auxiliary losses; loss/gradient/Adamweights khớp trong tolerance CPU/CUDA. GAE0,95 đổi policy loss, critic/aux vẫn giữ trên cùng trajectory. Không yêu cầu các lượt train dài có action giống hệt sau nhiều update.
- Sai estimator/lambda/gamma/shape/dtype/device/mask và NaN/Inf bị từ chối; smoke từ chối overwrite, sai B và GPU không có trước khi ghi output.

## Lỗi số học đã phát hiện và sửa

`gamma * bool_mask` tạo floating scalar theo default dtype trước khi nhân accumulator, làm recurrence float64 bị round qua float32. Test lambda1 nghiêm ngặt bắt lỗi; sửa thứ tự để nhân với accumulator floating trước mask.

GAE1 cộng TD residual float32 còn MC cũ tích lũy Pythonfloat rồi cast float32. Chấp nhận tương đương số học, không gọi bit-parity cho GAE1. Adam có thể khuếch đại rounding ở gradient gần0. Test atol/rtol: loss1e-5; gradient2e-6/1e-4; post-Adam weights5e-6/1e-4. Mặc định MC vẫn kiểm tra bit-parity riêng, không nới điều kiện đó.

| Frozen update | CPU | CUDA |
|---|---:|---:|
| Policy loss max abs delta | 4,47e-8 | 3,73e-9 |
| Gradient max abs delta | 1,49e-7 | 2,79e-8 |
| Post-Adam weight max abs delta | 4,78e-6 | 1,17e-7 |

Chi tiết: `float-parity-diagnostic.log`, source-before trong `before/`, diff trong `trainer.diff`. Source-before khớp trainer snapshot của pilot A1 đã đóng.

## Smoke thực tế

`run_gae_b0.py`: mỗi thiết bị chạy MC, GAE1, GAE0,95, mỗi chế độ20episode/max20/9×9/2agents/RuleBot/local/LR1e-4/seed2026101503, warm-start B riêng, một Adam/chế độ. Normalization/validation/tournament tắt.

| Kết quả | CPU | CUDA / RTX3050 |
|---|---:|---:|
| Episodes / updates mỗi chế độ | 20 / 20 | 20 / 20 |
| Learner steps mỗi chế độ | 394 | 400 |
| GAE terminal reasons mỗi chế độ |18time_limit +2match_end|20time_limit|
| Thời gian cả ba chế độ |15,06s|14,83s|
| Weights hữu hạn, source/B giữ nguyên | PASS | PASS |
| ZIP/CRC/bytes/metrics/checkpoint hashes |22files PASS|22files PASS|

Artifact directories: `runs/gae-b0-smoke-cpu-20261015-implementation`, `runs/gae-b0-smoke-cuda-20261015-implementation`, kèm console logs và `*-send.zip`. `review_smokes.py` kiểm tra lại và lưu `smoke_review.json`. Không có evaluation JSON hay test-opened marker. Không dùng win rate của smoke20bước để kết luận chất lượng; `best` khi validation tắt chỉ là output kỹ thuật. `full_protocol=false`, `decision=retain_original`.

## Quyết định

B0 DONE/PASS, B1 READY nhưng chưa có runner đối chứng200episode/nhánh và test400/model. B1 phải chạy control mới trên source đã cập nhật, giữ MC critic/local teacher/normalization-off và khóa seed mới trước evaluation. Không tái dùng control A1;34000đãdùng. B0 không mở31000/wood_branch hoặc bộ seed test mới.

Giữ `runs/pressure-C-next/final_phaseC.pt` SHA256 `93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60`. Chưa thử PPO, thay reward/critic target, resume Adam/RNG hay guard phản bom. B0 chỉ chứng minh implementation qua các kiểm tra đã nêu; GAE có thể không tăng ActiveWR.
