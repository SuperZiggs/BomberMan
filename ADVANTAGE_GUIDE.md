# T4-A1 — Chỉ chuẩn hóa advantage của actor

**Hoàn tất và review 15/10/2026; gate FAIL, giữ B.** Đủ200ep/nhánh và400trận/model trong96,93phút chạy full, toàn công việc trong hạn150phút. ActiveWR B33,25% / control36,5% / normalized33,5%; delta +0,25 / −3 điểm, hai CI chứa0; tự nổ0,25% cả ba. [Báo cáo quyết định](/D:/fpt/subject/BomberMan/runs/advantage-full-20261015-030430/decision_report.md). 150 tests, smoke CPU/CUDA20ep/nhánh, deadline check và review dữ liệu đã qua. Không cần chạy thêm; checkpoint đang dùng vẫn là B.

`train_actor_critic.py --normalize_advantage` bật chuẩn hóa trên toàn trajectory của learner trong một update. Mặc định tắt. Actor dùng `(A_raw - mean) / std(correction=0)`; nếu ít hơn hai mẫu hoặc std < 1e-8 thì giữ raw và ghi `normalization_skipped`. Advantage detach; critic vẫn dùng MC returns gốc. Không chuẩn hóa reward, value target, teacher hay auxiliary losses.

`--advantage_diagnostics` ghi raw/policy advantage, returns/value scale, explained variance (N/A nếu variance target < 1e-8), entropy theo số action trong mask, gradient norm trước clip và trạng thái clipping. Cờ normalize tự bật diagnostics. Cả hai nhánh pilot bật cùng diagnostics. NaN/Inf loss/gradient dừng trước optimizer step; partial episode do hủy không được update như trainer cũ.

## Protocol khóa trước test

| Mục | Giá trị |
|---|---|
| Hai nhánh | `control`: raw; `normalized`: chuẩn hóa actor advantage |
| Warm-start | B cho cả hai; Adam mới riêng, không tái dùng control cũ |
| Train | Phase C, RuleBot, 9×9, 2 agents, 200 episode/nhánh, max 500 bước |
| Seed/LR/gamma | 2026101502 / 1e-4 / 0.99 |
| Giữ nguyên | MC estimator/critic target, teacher local, reward, CNN 12 kênh, mask, lịch LR/entropy, grad clip 5, value 0.5, auxiliary 0.5/0.5/0.2 |
| Validation | 100 trận seed 21000 tại 0/100/200, chọn best theo thứ tự hiện có và guard tự nổ từ B |
| Skill diagnostic | 64 tình huống escape/approach validation tại mỗi mốc; không dùng chọn model; baseline cache chung |
| Test cuối | B + best control + best normalized, 400 trận/model, seed 34000–34199, mỗi seed cả A0/A1 |
| Inference | CNN categorical qua mask, CPU 1 thread, RuleBot; không gọi teacher/planner |
| Gate | ActiveWR hơn cả B và control ≥5 điểm phần trăm; cận dưới CI95% >0; self-blast quan sát không cao hơn hai đối chứng |
| Bootstrap | 10.000 lần, seed 20261014, lấy mẫu theo cặp seed bàn gồm cả hai góc |
| Giới hạn runner | Tối đa 150 phút; preflight ≤15, hai nhánh train ≤85, test ≤50; các pha dùng thời gian tổng còn lại |

Yêu cầu người dùng là 150 phút cho toàn bộ công việc; khi chạy thực tế, giảm `--budget-minutes` bằng thời gian còn lại sau triển khai/test/smoke. Dừng khi không đủ protocol: ghi incomplete, giữ B. Best được khóa bằng SHA trước test; không chọn final lại sau khi xem test. Gate pass chỉ cho phép lặp lại nghiên cứu trước khi thay B. Một train seed chưa đo được độ ổn định.

## Lệnh tái hiện

**Các lệnh dưới chỉ để tham khảo cách đã chạy. Lượt full đã xong; không chạy lại.** Seed34000–34199 đã dùng; runner sẽ từ chối. Sửa đổi tiếp theo cần protocol và seed mới, không chạy lại để chọn checkpoint trên test đã xem.

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
& $py -m unittest test_audit_t2 test_tactical_teacher test_combat test_visualize test_two_bomb_complete test_l2_candidates test_benchmark_t3 test_t3_curriculum test_t3_approach test_combat_move_teacher test_combat_move_pilot test_advantage_pilot -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $py -u .\run_advantage_pilot.py --smoke --device cpu --output-dir runs\advantage-smoke-cpu
& $py -u .\run_advantage_pilot.py --smoke --device cuda --output-dir runs\advantage-smoke-cuda
& $py -u .\run_advantage_pilot.py --device cuda --budget-minutes 120 --output-dir runs\advantage-full
```

Smoke: 20 episode/nhánh, max 20 bước, validation 2 trận tại 0/10/20, test kỹ thuật 2 trận/model seed 39840. Không dùng smoke để kết luận chất lượng. Runner từ chối ghi đè thư mục. Sau reboot đọc `runs/advantage-last-path.txt` để tìm kết quả; không tạo timestamp mới rồi tìm summary ở thư mục chưa chạy.

Kết quả lưu trong thư mục riêng: `config.json`, `source/`, `preflight.json`, `baseline_skill.json`, hai thư mục nhánh chứa config/metrics/validation/skill tapes/best/final, `selected_models.json`, `test_opened.json`, ba evaluation JSON và `summary.json`. Marker đánh dấu seed đã mở trước evaluation để lần bị gián đoạn cũng không được tái dùng như test mới. ZIP `*-send.zip` và console log là gói dữ liệu đánh giá. Checkpoint chỉ chứa weights; chưa hỗ trợ resume Adam/scheduler/RNG chính xác.

GAE, PPO, thay reward, curriculum, MOVE guidance mới và guard phản bom nằm ngoài T4-A1. Chuẩn hóa thay scale policy so với loss phụ cố định; pilot đo tác động tổng này, không chứng minh nguyên nhân của validation giảm ở lượt trước.
