# T3.2 — curriculum giao tranh có đối chứng

**Cập nhật15/10:** [audit approach/counter-bomb offline đã hoàn tất](/D:/fpt/subject/BomberMan/runs/t3-approach-audit-20261015-final/T3_approach_decision.md): teacher giải32/32 approach, CNN yếu đoạn gần mục tiêu;96 fixture checks/119 tests PASS. Giữ B; chưa train hoặc sửa production mask. Không cần chạy thêm để đóng audit. Bước đề xuất tiếp là ablation guidance move có certificate; curriculum C2/rehearsal và runtime guard là các nhánh riêng chưa triển khai.

Runner: [run_t3_curriculum.py](/D:/fpt/subject/BomberMan/run_t3_curriculum.py). Đây là thí nghiệm về phân phối tình huống/đối thủ. Cả hai nhánh dùng teacher `local`, observation12 kênh, reward, loss, safety mask và trainer hiện tại. CNN tự chọn hành động; curriculum chỉ thay trạng thái khởi đầu và đối thủ trong train.

Đã test implementation bằng smoke CPU/CUDA/RTX3050 và110 tests. **Lượt full mode đã được review:** control600, curriculum dừngC1 ở200 theo gate; giữ B, test31000/wood chưa mở. Xem [T3.2_decision.md](/D:/fpt/subject/BomberMan/runs/t3-curriculum-full-20261014-154703/T3.2_decision.md). [Control confirmation32000 cũng hoàn tất](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/confirmation_report.md): gain2,25pp/CI chứa0,gate FAIL,giữ B. Không cần chạy lại hai bài này; audit offline cũng đã đóng như cập nhật15/10 phía trên. Các lệnh bên dưới giữ lại để tái lập protocol cũ, không phải đề nghị train nối hoặc nới gate.

## Protocol đã cố định trước lượt full

| Tham số | Cả hai nhánh |
|---|---|
| Warm-start B | `runs/pressure-C-next/final_phaseC.pt`, SHA256 `93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60` |
| Phase / bàn / agents / max_steps | C / 9×9 / 2 / 500 |
| Episodes / train seed / LR | 600 / 2026101303 / 1e-4 |
| Teacher / Adam / gamma / value coefficient | local / Adam / 0.99 / 0.5 |
| Entropy ban đầu / thưởng tường | 0.015 / 0.15 |
| Lịch LR / entropy | Lịch600 của trainer; không bắt đầu lại khi chuyển stage |
| Validation chọn best | 100 trận seed21000, ep0 và mỗi100ep, CPU; giữ guard tự nổ của B và cách xếp hạng hiện tại |
| Train device | CUDA mặc định; có thể chọn CPU cho **cả hai nhánh** |
| Thứ tự | Control trước, curriculum sau; mỗi nhánh khởi tạo lại cùng B và train seed |

Control dùng map ngẫu nhiên PhaseC và RuleBot suốt600ep. Curriculum:

| Stage | Phân phối train | Gate kỹ năng trên validation family |
|---|---|---|
| C1_static | 50% positive escape/approach thuộc `fork`; 50% map ngẫu nhiên density0.25. Đối thủ đứng yên | Escape/approach vs static, 64 games/probe |
| C2_evader | Positive combo `fork`, learner ammo2/range2–3; đối thủ né, không đặt bom | Combo vs evader, 128 games/probe |
| C3_rule_mix | 50% combo `fork`; 50% map PhaseC density0.7. RuleBot có né/đặt bom | Combo vs RuleBot, 128 games/probe |
| C4_transfer | Map PhaseC/RuleBot cho phần episode còn lại đến600 sau khi C3 đạt | Không có gate mới; vẫn validation chọn best mỗi100ep |

50% là xác suất lấy mẫu mỗi episode, không bảo đảm số lượng đúng bằng một nửa. C2 khởi đầu ở trạng thái có capacity đủ cho combo; map ngẫu nhiên khởi đầu ammo1 theo luật hiện tại. Đây là treatment gồm **layout/capacity và đối thủ**, nên kết quả không tách riêng tác dụng của từng thành phần.

Chỉ train trên positive train family `fork`; không train trên controls ammo1 hoặc validation/test family. Preflight đối chiếu registry và hash với T3.1, chứng nhận lại64 positive train cases qua reference/mask; combo phải cần cả hai bom. Registry cũ không bị sửa.

Skill validation dùng `long_retreat`/`long_fork`, hai RNG seeds2026101401/2026101402, horizon80 bước. Success là learner còn sống và thắng bằng owned kill. Probe ghi cả số combo thật, số bom, tự nổ và truncation; owned-kill success có thể đạt mà không cần combo, nên phải đọc thêm số combo. A0/A1 và symmetries/repeats không được coi là các mẫu độc lập.

Mỗi stage có probes ở tuổi100/150/200ep. Chuyển khi **hai probes liên tiếp** có success≥60% và suicide không vượt B trên đúng bộ skill validation của stage. Chuyển sớm nhất ở150ep; cap200ep. Guard được đo từ B trước train và lưu `baseline_skill.json`. Probe chấm trọng số hiện tại trên CPU, giữ nguyên trọng số/mode và RNG NumPy/Torch/CUDA của train. Skill validation không chọn best.

Nếu stage không qua cap, runner dừng curriculum và ghi `status=incomplete`, `reason=stage_gate_failed`. Control giữ checkpoint tại các mốc50ep từ200 để so đúng ngân sách: ví dụ curriculum dừng350 thì chấm control ep350 và curriculum final ep350 trên **100 validation games seed21000**. File `matched_budget_diagnostic_21000.json` chỉ là diagnostic; không mở test31000 hoặc `wood_branch`, không đủ kết luận pilot thành công. Nếu hết thời gian/người dùng dừng thì có thể chưa đủ diagnostic.

Chỉ khi hai nhánh đủ600ep, mọi PhaseC validation đủ và C3 đã qua gate mới mở đánh giá cuối:

- B, control best, curriculum best: **400 trận/model**, board seeds31000–31199, mỗi seed cảA0/A1; CPU, RuleBot, map ngẫu nhiên, max500. Không gọi planner để chọn hành động.
- `wood_branch`: 96 test cases × static/evader/RuleBot ×2 RNG repeats = **576 games/model**, ba model; horizon80. Đây là kiểm tra transfer kỹ năng, tách khỏi tỷ lệ thắng map ngẫu nhiên.
- ActiveWR curriculum phải hơn **cả B và control ≥5pp**, suicide không cao hơn cả hai, bootstrap95% theo cặp seed bàn có cận dưới>0 cho cả hai chênh lệch. Ghi hòa, kills, passive wins và từng góc trong JSON.
- Qua gate chỉ được `eligible_for_replication=true`, `decision=replicate_before_replacing`; không tự thay model. Một train seed chưa chứng minh ổn định. Test31000/wood_branch sau khi đọc không dùng để tune rồi báo lại như test độc lập.

## Các lệnh PowerShell đầy đủ

Chạy trong PowerShell. Phần test/smoke có thể chạy lại trước full. Mỗi lần chạy tạo thư mục mới; không ghi đè các run cũ.

### 1. Kiểm tra Python/GPU và tạo tên thư mục

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$t3Smoke = Join-Path (Get-Location) "runs\t3-curriculum-smoke-$stamp"
$t3Full = Join-Path (Get-Location) "runs\t3-curriculum-full-$stamp"
$t3Tests = Join-Path (Get-Location) "runs\t3-curriculum-tests-$stamp.log"
& $py -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) { throw 'Không chạy được Python/torch.' }
```

### 2. Bộ test

```powershell
& $py -m unittest test_audit_t2 test_tactical_teacher test_combat test_visualize test_two_bomb_complete test_l2_candidates test_benchmark_t3 test_t3_curriculum -q 2>&1 |
    ForEach-Object { $_.ToString() } | Tee-Object -FilePath $t3Tests
if ($LASTEXITCODE -ne 0) { throw 'Test thất bại; chưa chạy pilot.' }
```

Phải thấy `OK`. `Expanded observation channels 8 -> 12` là output của test tương thích checkpoint, không phải lỗi. Pipeline đã đổi stderr thành chuỗi để tránh NativeCommandError giả của PowerShell.

### 3. Smoke trên GPU sẽ dùng cho full

```powershell
& $py -u .\run_t3_curriculum.py --smoke --device cuda --seconds 300 --output $t3Smoke 2>&1 |
    ForEach-Object { $_.ToString() } | Tee-Object -FilePath "$t3Smoke-console.log"
if ($LASTEXITCODE -ne 0) { throw 'Smoke lỗi; xem console.log.' }
$smokeResult = Get-Content -LiteralPath (Join-Path $t3Smoke 'summary.json') -Raw | ConvertFrom-Json
if ($smokeResult.status -ne 'complete' -or $smokeResult.full_protocol -or -not $smokeResult.hashes_unchanged) {
    throw 'Smoke chưa hoàn tất đúng protocol.'
}
if ($smokeResult.arms.control.completed_episodes -ne 6 -or $smokeResult.arms.curriculum.completed_episodes -ne 6 -or
    $smokeResult.arms.curriculum.optimizer_instances -ne 1 -or $smokeResult.arms.curriculum.optimizer_steps -ne 6) {
    throw 'Smoke chưa chạy đủ/optimizer không liên tục.'
}
```

Nếu CUDA không khả dụng, đổi `--device cuda` thành `--device cpu` **ở cả smoke và full**. Không cần đổi trainer. Smoke không dùng seeds31000 hoặc test family.

### 4. Chạy cặp full600

```powershell
Set-Content -LiteralPath 'runs\t3-curriculum-last-path.txt' -Value $t3Full
& $py -u .\run_t3_curriculum.py --device cuda --seconds 0 --output $t3Full 2>&1 |
    ForEach-Object { $_.ToString() } | Tee-Object -FilePath "$t3Full-console.log"
if ($LASTEXITCODE -ne 0) { throw 'Runner lỗi; gửi console.log và status.json.' }
```

`--seconds 0` không giới hạn thời gian. Có thể đặt ngân sách cho **toàn lượt**, gồm preflight, hai train và đánh giá, ví dụ `--seconds 14400` cho4giờ; hết thời gian thì ghi incomplete. Không thể suy ra thời lượng600ep từ smoke8bước/trận. Giữ máy hoạt động và cắm nguồn trong lúc chạy. Không cần chạy thêm evaluator riêng: runner làm toàn bộ phần đánh giá đủ điều kiện.

### 5. Đọc trạng thái sau chạy hoặc sau khi mở PowerShell lại

Nếu đã tắt/mở PowerShell, đọc lại **đường dẫn đã lưu**, không tạo `$stamp` mới để tìm kết quả cũ:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$t3Full = (Get-Content -LiteralPath 'runs\t3-curriculum-last-path.txt' -Raw).Trim()
Get-Content -LiteralPath (Join-Path $t3Full 'status.json')
if (Test-Path -LiteralPath (Join-Path $t3Full 'summary.json')) {
    $fullResult = Get-Content -LiteralPath (Join-Path $t3Full 'summary.json') -Raw | ConvertFrom-Json
    $fullResult | Select-Object status, reason, full_protocol, hashes_unchanged, eligible_for_replication, decision, elapsed_seconds
    $fullResult.arms.control | Select-Object completed_episodes, validation_complete, learner_steps, optimizer_steps
    $fullResult.arms.curriculum | Select-Object completed_episodes, validation_complete, curriculum_stages_complete, stage_failure, stage_episode_counts
}
Get-Item -LiteralPath "$t3Full-send.zip", "$t3Full-console.log" -ErrorAction SilentlyContinue
```

`COMPLETE` trong smoke không phải full. Full hợp lệ phải có `status=complete`, `full_protocol=true`, mỗi nhánh600, đủ validation, các stage hoàn tất, random-map/held-out evaluation complete và hashes unchanged. Full hoàn tất vẫn có thể **không qua gate cải thiện**. `stage_gate_failed` là kết quả thí nghiệm, không phải crash.

Đang chạy có root `status.json`; xem chi tiết episode/stage ở `control/status.json` hoặc `curriculum/status.json`. Metrics được flush mỗi episode. Ctrl+C yêu cầu dừng hợp tác; episode train dở bị loại và lưu các updates đã hoàn tất. Tắt máy cưỡng bức có thể chỉ còn logs/checkpoint lần lưu gần nhất. **Runner chưa hỗ trợ resume**, vì checkpoint hiện tại không lưu Adam/scheduler/RNG. Sau gián đoạn không ghép metrics hai lượt hoặc bắt đầu stage giữa chừng rồi gọi là cặp600; giữ output cũ và dùng thư mục mới cho lượt mới.

## File cần gửi lại

Ưu tiên gửi **`$t3Full-send.zip` và `$t3Full-console.log`**, kèm `$t3Tests`. ZIP gồm:

| File | Dùng để kiểm tra |
|---|---|
| `config.json`, `summary.json`, `status.json`, `source/`, `scenario_registry.json` | Protocol, hash/checkpoint, runtime, completeness và source dùng thật |
| `baseline_skill.json` | Guard và skill performance của B trước train |
| `control/` và `curriculum/`: config, metrics, validation_metrics, best_validation, summary, stage_events, best/final | Hai train có cùng điều kiện, optimizer/LR/steps, validation chọn best, kết quả chuyển stage |
| `curriculum/skill_validation_metrics.jsonl`, `skill_validation_games.jsonl.gz` | Gate từng stage, combo/kill/suicide, replay đầy đủ của skill probes |
| `evaluation_original_31000.json`, `evaluation_control_31000.json`, `evaluation_curriculum_31000.json` | 400/model, cả hai góc, game-level paired comparison |
| `heldout_skill_summary.json`, `heldout_skill_games.jsonl.gz` | Transfer trên test family chưa train |
| `matched_budget_diagnostic_21000.json`, `control/budget_ep*.pt` | Nếu stage dừng: final weights ở đúng số episode; diagnostic không dùng test mới |

Các file evaluation/test chỉ có khi đủ điều kiện mở tương ứng. Run dừng theo gate/thời gian vẫn tạo ZIP nếu quá trình kết thúc bình thường. Nếu exception/mất điện trước tạo ZIP, gửi console và các JSON/logs hiện có; có thể nén thư mục thủ công:

```powershell
$t3Full = (Get-Content -LiteralPath 'runs\t3-curriculum-last-path.txt' -Raw).Trim()
$partialZip = "$t3Full-partial-send.zip"
if (Test-Path -LiteralPath $partialZip) { throw 'ZIP đã có; đổi tên để giữ bản cũ.' }
Compress-Archive -LiteralPath $t3Full -DestinationPath $partialZip
```

Không tự chọn final thay best hoặc đổi ngưỡng sau khi xem kết quả. Mình sẽ kiểm tra completeness trước, rồi đánh giá skill progression, transfer, paired ActiveWR và suicide; nếu chưa qua gate thì tiếp tục giữ B.
