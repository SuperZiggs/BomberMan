# T4-R1 — giảm riêng thưởng phá tường

R1 DONE/REVIEWED: `reward-full-resumable-20261015-194314`. Đủ 200 episode và 200 Adam update mỗi nhánh từ B; 400 trận test/model trên seed 36000–36199, cả A0/A1. Best control/half-wall đều ở episode 200. ActiveWR B/control/half-wall = 34,50%/36,25%/35,00%. Gate FAIL; giữ B và wall bonus 0,15. Thời gian chạy 49,54 phút. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-resumable-20261015-194314/decision_report.md).

**Lượt trước:** R1 IMPLEMENTED/TESTED, full INCOMPLETE do tiến trình bị ngắt: `reward-full-20261015-163227`. Control đủ200episode; half-wall199episode đã lưu, validation200/final200 còn thiếu. Best validation control33%/half-wall30%, chưa có test hoặcCI; giữB. 200regression testsPASS, smokeCPU/CUDA qua; audit320skill replays/6.966frames. [Báo cáo](/D:/fpt/subject/BomberMan/runs/reward-full-20261015-163227/decision_report.md).

**36000–36199 đã mở ở rerun này. Không chạy lại full trên dải này để chọn model khác.**

Phiên bản mới có `training_state.pt` cho từng nhánh và `--resume` để tiếp tục đúng state đã xác nhận. Lượt cũ ở trên không có Adam/RNG nên không được dùng cho chức năng này. Hai nhánh của lượt mới đều bật checkpoint và CUDA deterministic; không tái sử dụng control của lượt cũ.

Runner `run_reward_pilot.py` so hai nhánh từ checkpoint B. Chỉ hệ số thưởng phá tường khác nhau; item, kill, win, death, thời gian, teacher local, mask, 12 kênh observation và các loss khác giữ nguyên.

| Mục | Protocol đăng ký trước |
|---|---|
| Nhánh | `control`: wall bonus 0,15; `half_wall`: 0,075 |
| Khởi tạo | `runs/pressure-C-next/final_phaseC.pt`, SHA256 bắt đầu `93bdf278`; Adam mới mỗi nhánh |
| Train | Seed 2026101505; 200 episode/nhánh; Phase C; 9×9; 2 agents; max500; LR 1e-4; gamma0,99 |
| Estimator | Actor MC, advantage không chuẩn hóa; critic MC; không thêm GAE/PPO |
| Validation | 100 trận seed21000 tại episode0/100/200; chọn best và guard tự nổ hiện có |
| Skill probe | 32 escape + 32 approach tại mỗi validation, cùng bộ tình huống đã dùng |
| Test cuối | B và best của hai nhánh; 400 trận/model; 200 seed bàn36000–36199, mỗi bàn chạy A0/A1 |
| Inference | CNN lấy mẫu qua mask; không gọi teacher/planner |
| Ngân sách | Tổng tối đa150 phút; preflight15, train100, evaluation50 phút, mỗi giai đoạn bị giới hạn bởi thời gian tổng còn lại |

Preflight định giá lại 39.338 transition của B trong R0. Với cùng actions/info, chênh lệch reward phải bằng `-0.075 × số tường đủ điều kiện`, các component khác giữ nguyên, sai số tối đa1e-10. Tường phá cùng tick learner chết không nhận bonus theo luật cũ. Đây là kiểm tra accounting, chưa phải bằng chứng thay đổi policy có ích.

Hai nhánh cùng bật logging reward và tiếp cận đối thủ. Hook chỉ đọc state; test đối chiếu với evaluator trước R1 cả sampled actions và kết quả. Mặc định các caller cũ không bật hook. Công thức reward và API `reset/step` giữ nguyên; trainer thêm tùy chọn lưu/khôi phục state của Phase C.

Trong `metrics.jsonl`, `reward_components` cộng lại phải bằng reward episode. `walls` giữ số tường được attribution, bao gồm tick chết; reward wall chỉ gồm tường đủ điều kiện. `contact` ghi vị trí đầu, đã rời ô spawn hay chưa, khoảng cách nhỏ nhất và bước đầu có Manhattan≤3 tới một đối thủ còn sống. Đây là proxy khoảng cách, không chứng minh đã mở đường đi an toàn.

Evaluation có cùng các trường contact và `walls`/`walls_rewarded` cho từng game, trung bình tường và tỷ lệ tiếp cận/rời spawn tổng hợp và từng góc. `avg_first_close_step_conditional` chỉ tính những trận đã tới gần; phải đọc cùng `reached_close_rate`. Không thay thiếu mẫu bằng0. Không coi reward thấp hơn là policy tốt hơn vì coefficient khác nhau.

Mặc định runner từ chối output đã tồn tại, B sai hash và dải test đã mở, kể cả chỉ có `test_opened.json` do lượt trước bị ngắt. `--resume` chỉ cho phép tiếp tục chính thư mục đã đăng ký với source/config/runtime và model test đã khóa; không chọn checkpoint khác sau khi xem test. Nó khóa source và ba input R0, chọn/hash best trước khi mở test. Thiếu train/validation/skill/test hoặc hash đổi: `incomplete`, giữ B. Smoke luôn `full_protocol=false`, không được dùng chọn model.

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
& $py -m unittest test_reward_pilot test_reward_audit test_gae_pilot test_combat_move_pilot -q
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py -u .\run_reward_pilot.py --smoke --device cpu --output-dir "runs\reward-smoke-cpu-$stamp"
# Chỉ chạy CUDA nếu torch.cuda.is_available() trả True.
& $py -u .\run_reward_pilot.py --smoke --device cuda --output-dir "runs\reward-smoke-cuda-$stamp"
& $py -u .\run_reward_pilot.py --device cuda --budget-minutes 150 --output-dir "runs\reward-full-$stamp"
```

Chỉ chạy full khi36000–36199 chưa mở. Sau khi test đã mở, muốn lặp phải đăng ký train seed/test seed khác; không chạy lại để tìm kết quả tốt hơn. Sau khi mở lại PowerShell, lấy đường dẫn từ `runs/reward-last-path.txt`, không dùng timestamp mới để tìm output cũ.

Gate yêu cầu half_wall tăng ActiveWR ít nhất5 điểm phần trăm so với cả B lẫn control; cận dưới CI95% bootstrap theo cặp seed bàn lớn hơn0; tự nổ quan sát không cao hơn hai đối chứng. Báo thêm draw, kills, passive wins, từng góc, contact và tường. Nếu mất khả năng mở đường thì cần xem regression trước khi đề xuất replication. PASS chỉ cho phép lặp thêm train seed, không tự thay B.

Output gồm config/source snapshot, preflight, baseline skill, metrics/validation/skill replays/best/final và `training_state.pt` mỗi nhánh, selected models, test marker, ba evaluation JSON, summary và ZIP `*-send.zip`. Báo cáo review thêm sau runner lưu riêng. `best/final_phaseC.pt` vẫn là weights để chơi; `--resume_from` vẫn là warm-start với Adam mới. Không dùng chúng để tiếp tục chính xác một lượt bị ngắt.

## Checkpoint đầy đủ và resume

`training_state.pt` giữ model hiện tại, mode train/eval, toàn bộ Adam moments/step, cosine scheduler/T_max, episode hoàn tất, lịch sử train, best/guard/validation, RNG Python/NumPy/PyTorch CPU/CUDA và RNG riêng của môi trường. R1 còn lưu cả metrics, skill probes/replay, contact và validation rows đã xác nhận. Ghi file tạm, flush/fsync rồi `os.replace`; file `.tmp` không phải checkpoint để resume.

Khóa writer của hệ điều hành ngăn hai lệnh cùng train/resume một thư mục; khóa tự được nhả nếu tiến trình kết thúc. File `*-run.lock` còn tồn tại không có nghĩa tiến trình còn chạy. GUI/inference dùng `best/final_phaseC.pt`; loader từ chối full training state để tránh vô tình chơi bằng trọng số chưa được nạp.

Checkpoint được xác nhận sau toàn bộ episode, validation/skill nếu đến mốc và scheduler step. Nếu dừng giữa episode hoặc validation, resume quay về checkpoint gần nhất và tính lại phần chưa xác nhận; không cộng Adam update hai lần. Log đã ghi vượt checkpoint hoặc gzip bị dở được giữ trong `recovery/`, rồi phục hồi từ context trong state. Summary/ZIP cũ được giữ riêng khi tiếp tục.

Chỉ dùng resume khi lượt còn dang dở, đã dừng tiến trình cũ và chưa hết deadline. Lượt DONE ở đầu trang không cần resume. Sau khi mở lại PowerShell:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
$r1Run = (Get-Content -LiteralPath 'runs\reward-last-path.txt' -Raw).Trim()
& $py -u .\run_reward_pilot.py --resume --device cuda --output-dir $r1Run --budget-minutes 150
```

Không tạo timestamp mới cho lệnh resume. Root config lưu thiết bị; dùng đúng `--device` ban đầu. Không sửa source, seed, số episode, reward, teacher, LR hoặc lịch validation. Nguồn, thư viện/runtime và thiết bị khác sẽ bị từ chối. Hiện hỗ trợ full-state resume cho Phase C; không mặc định hỗ trợ opponent pool D/E hoặc chuyển CPU↔CUDA. CUDA dùng deterministic algorithms, cuDNN benchmark tắt và `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Resume không đặt lại trần150 phút hoặc trần từng giai đoạn: deadline gốc vẫn tính cả thời gian gián đoạn. Nếu deadline đã hết, snapshot vẫn được giữ; cần một lượt/ngân sách mới được đăng ký trước khi tiếp tục nghiên cứu, không tự biến lượt quá hạn thành full protocol. Checkpoint của trainer có thể dùng riêng qua `--resume_training_state` với đúng cấu hình; đó không tự hoàn tất protocol của runner R1.

Evaluation đã hoàn tất được tái sử dụng khi model/seed/count khớp. Model bị ngắt giữa evaluation có thể được chấm lại trên cùng seed với model đã khóa; không mở model mới hoặc chọn lại best. Không chạy thêm train sau khi test đã mở.

R0 cho thấy reward phụ phá tường lớn hơn item, nhưng không chứng minh đó là nguyên nhân hòa. Pilot một train seed và một mức coefficient không đủ kết luận phương pháp ổn định hoặc hệ số tối ưu.
