# SD3 — Train B dưới sudden death + kênh quan sát vòng

Ngày: 2026-10-16. Trạng thái: thiết kế đã duyệt trong chat, chờ duyệt spec.

## Mục tiêu

SD2 (start400/interval40/warning40/max_ring2) tăng WR +19,75pp và giảm hòa −49,5pp trên seed 39000, nhưng chủ yếu nhờ RuleBot tự nổ trong đấu trường hẹp. Learner vẫn chết vì vòng 131/400 trận và ActiveWR giảm 4pp. SD3 train tiếp B dưới luật SD2 để agent tự tránh vòng và tận dụng đấu trường co lại.

**Thành công** = trên seed test mới, model train dưới SD hơn B khi cả hai cùng chơi SD2 both_aware. Tiêu chí chính là ActiveWR; báo cáo thêm WR, hòa, số learner chết vì vòng và tự nổ.

## Phạm vi

- Thay đổi: `sudden_death_experiment.py` (kênh vòng, runtime context), chỗ tạo mạng trong `train_actor_critic.py` và `evaluate.make_ac_fn` (đọc số kênh từ checkpoint), runner mới `run_sd_training_pilot.py` và test.
- Không đổi: `bomberman_env.py` (12 kênh, OBS_CHANNELS), reward hiện có, mask production, RuleBot, GUI, checkpoint B. GUI chưa hỗ trợ model 13 kênh.

## Thiết kế

### 1. Kênh quan sát vòng

`SuddenDeathEnv(ring_channel=False)`. Khi `ring_channel=True`, `_get_obs` trả về 13 kênh: 12 kênh cũ không đổi, kênh 12 bằng `1 / delay` cho mọi ô của vòng chưa đóng `ring <= sudden_death_max_ring`, với `delay` là số bước tới lần đóng của vòng đó (lấy từ `_closure_ticks`); còn lại 0. Ô đã đóng là RIGID_WALL ở kênh 0. Khi `ring_channel=False`, observation giống hệt bản hiện tại. Khi `ring_channel=True` nhưng `sudden_death=False` (env của skill probe), vẫn trả 13 kênh và kênh 12 toàn 0. Thuộc tính `obs_channels` = 12 hoặc 13.

### 2. Runtime train/eval dưới sudden death

Context manager `sudden_death_runtime(timings, ring_channel)` trong `run_sudden_death_experiment.py` (cạnh `evaluate_case`, tránh import vòng) dùng `unittest.mock.patch` như `evaluate_case` đang làm:
- `train_actor_critic.BombermanFFAEnv` và `evaluate.BombermanFFAEnv` → subclass `SuddenDeathEnv` với `sudden_death=True`, `**timings`, `ring_channel`.
- `train_actor_critic.get_action_mask` → `sudden_death_mask(env, aid, base)` cho learner.
- `rule_based_bot._safe_action_mask` → `sudden_death_mask` cho RuleBot (both_aware).
- `benchmark_t3.BombermanFFAEnv` (skill probe của `train_arm`) → `SuddenDeathEnv(sudden_death=False, ring_channel=...)`: fixture không đổi, chỉ thêm kênh 0 cho mạng 13 kênh. Baseline skill tính riêng mỗi nhánh bằng mạng đúng số kênh.

`train()` giữ nguyên logic. Chỗ tạo `ActorCriticNet` trong `train()` lấy `in_channels=getattr(env, 'obs_channels', OBS_CHANNELS)`; `make_ac_fn` lấy `max(conv.0.weight.shape[1], OBS_CHANNELS)` từ checkpoint và chỉ đưa vào mạng `in_channels` kênh đầu của observation (kênh vòng ở cuối, nên model 12 kênh chơi được trong env 13 kênh như khi không thấy kênh). Với B và env 12 kênh, kết quả như cũ. Warm-start B vào mạng 13 kênh qua `smart_load_weights` (trọng số kênh mới = 0), nên episode 0 tương đương B.

Reward: giữ nguyên, cộng −1 khi chết vì vòng (đã có trong `SuddenDeathEnv.step`).

### 3. Runner `run_sd_training_pilot.py`

Theo mẫu `run_reward_pilot.py`, tái dùng `train_arm`, `pair_ready`, `probe_summary`, `fresh_test_check`, `stage_timer`, `paired_active_bootstrap`, resume training state. Đánh giá dùng `evaluate_case(..., checkpoint=, ring_channel=True)` (thêm tham số) để có nguyên nhân chết; kết quả mỗi model lưu riêng và dùng lại khi resume nếu hash khớp.
- Nhánh: `sd_12ch` (SD2, không kênh vòng) và `sd_13ch` (SD2 + kênh vòng). Cùng warm-start B, cùng train seed mới, Phase C vs RuleBot, 200 episode, LR 1e-4, max 500 bước, teacher local, MC.
- Validation seed 21000, 100 trận ở 0/100/200, chạy dưới SD2 both_aware; best theo validation như các pilot trước.
- Test một lần seed 40000–40199, 400 trận/model (B, best sd_12ch, best sd_13ch), CPU 1 thread, SD2 both_aware.
- Gate (giống pilot trước): ActiveWR ≥ +5pp so với B, CI95 lower > 0, tự nổ không cao hơn B. Paired bootstrap 200 cặp seed bàn.
- Ghi config, source hashes, bản sao source, ZIP; từ chối ghi đè; kiểm hash B.

## Kiểm thử

- `ring_channel=False`: obs, reward, info, RNG khớp bit với SuddenDeathEnv hiện tại.
- Kênh 12 khớp lịch đóng vòng (giá trị trước/sau khi đóng, max_ring).
- B nạp vào mạng 13 kênh cho logits/value giống mạng 12 kênh trên cùng observation.
- `sudden_death_runtime` bật/tắt khôi phục đúng các hàm/lớp đã patch.
- `make_ac_fn` nạp được checkpoint 12 và 13 kênh.
- Smoke CPU và CUDA cho runner; người dùng chạy full.

## Rủi ro

- SD bắt đầu ở bước 400 nên chỉ khoảng 60% episode có vòng; 200 episode có thể chưa đủ để học kênh mới. Không đổi config để cứu gate.
- Một train seed, không chứng minh ổn định.
- Patch module-level là toàn cục trong tiến trình; runner chạy tuần tự nên chấp nhận được.
