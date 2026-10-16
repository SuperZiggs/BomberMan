# Pilot nhãn MOVE chiến đấu có đối chứng

**Đã hoàn tất và review15/10/2026:** [decision_report.md](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315/decision_report.md). Đủ200ep/arm và400testgames/model trong58,38phút. ActiveWRB35%/control35,5%/candidate40,25%;delta+5,25/+4,75pp,haiCI chứa0;gateFAIL,giữB. Không cần người dùng chạy thêm.136testsPASS,smokeCPU/CUDA/deadline vàrawdata đã kiểm tra.

## Thay đổi được thử

`--attack_teacher local_combat_move` giữ target action của teacher local. Chỉ một nhãn MOVE tăng positive pressure, qua mask, có next BOMB hợp lệ và escape certificate mới chuyển từ **navigation mean0,2 sang combat mean0,5**. Các nhãn còn lại giữ nhóm cũ; không thêm loss đồng thời mâu thuẫn trên cùng bước. Việc đổi nhóm còn đổi denominator của combat/navigation mean, nên đây không chỉ là nhân một CE với2,5.

Chứng nhận trên clone: enemy STOP; thực hiện MOVE/BOMB, kiểm tra placement thật, đi theo mask tới hết mọi bom/lửa đã biết. Dùng greedy escape hiện có, tối đa32 tick nguy hiểm; nếu không chứng nhận được thì giữ navigation, không tuyên bố không có lời giải. CNN vẫn sample từ categorical qua mask, planner không chọn action thay CNN. Forecast/physics/mask/reward/12 channels/CNN/optimizer/entropy schedule giữ nguyên. Mặc định trainer vẫn `local`.

Đây không phải delayed counter-bomb guard. Certificate không bảo đảm sống trước mọi move/bom tương lai của đối thủ. Các fixtures phản bom96 cases của audit trước vẫn dùng để tái tạo giới hạn mask, không bị sửa thành test mask mới đã an toàn.

## Protocol full đã đăng ký trước test

| Tham số | Cố định |
|---|---|
| Warm-start cả hai nhánh | `runs/pressure-C-next/final_phaseC.pt`, SHA256 bắt đầu `93bdf278` |
| Hai nhánh | `control`:local; `combat_move`:local_combat_move |
| Train | PhaseC/RuleBot,9×9/2agents,200ep/arm,max500,LR1e-4,train seed2026101501 |
| Optimizer/schedule | Một Adam/arm, một update/episode; MC returns và schedule200 hiện tại |
| Validation chọn best | CPU100games seed21000 tại0/100/200; giữ suicide guard từB |
| Skill diagnostic |64 escape/approach validation games mỗi mốc;64 baseline được dùng chung. Không chọn best/gate bằng skill |
| Preflight | Tối đa256 unique navigation states từ C1 tapes đã xem; phải nhận ít nhất1 move, p95≤50ms |
| Test cuối | B/controlbest/combat_movebest,400games/model;200 seeds33000–33199 ×A0/A1;CPU1thread/RuleBot/max500;CNN+mask, không gọi teacher |
| Gate | Candidate hơn **cảB vàcontrol ≥5pp** ActiveWR; paired bootstrap95% lower>0; observed suicide≤cảhai |
| Bootstrap |10.000 resamples, seed20261014, đơn vị seed bàn gồm cảA0/A1 |
| Thời gian | Tổng≤120phút; preflight≤15, train cảhai≤65, evaluation≤40 |

Chỉ đủ hai lượt train/validation/skill probes mới mở test. Hết thời gian/gián đoạn/source hoặc checkpoint bị sửa thì ghi incomplete/retain_original. Lưu best/final riêng; không dùng final để chọn lại sau khi xem test. Gate pass chỉ cho phép replication, không tự thay B.33000 là bộ mới so với các JSON evaluation có sẵn trước khi đăng ký.31000/wood_branch của curriculum vẫn chưa mở.

## Lệnh tham khảo

Các lệnh dưới là cách runner đã được dùng. **Lượt full đã hoàn tất; không cần chạy lại.**33000 đã mở, runner sẽ từ chối train lại một candidate rồi gọi cùng33000 là test mới. Muốn thử sửa khác cần đăng ký protocol và seed mới trước lượt đó; pilot này chưa được gate xếpeligible cho replication.

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
$device = (& $py -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')").Trim()
$moveStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Path 'runs' -Force | Out-Null

& $py -m unittest test_audit_t2 test_tactical_teacher test_combat test_visualize test_two_bomb_complete test_l2_candidates test_benchmark_t3 test_t3_curriculum test_t3_approach test_combat_move_teacher test_combat_move_pilot -q 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath "runs\combat-move-tests-$moveStamp.log"
if ($LASTEXITCODE -ne 0) { throw 'Tests failed; stop before train.' }

$moveSmoke = Join-Path (Get-Location) "runs\combat-move-smoke-$moveStamp"
& $py -u .\run_combat_move_pilot.py --smoke --device $device --output-dir $moveSmoke 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath "$moveSmoke-console.log"
if ($LASTEXITCODE -ne 0) { throw 'Smoke failed; send the log.' }
$smoke = Get-Content -LiteralPath (Join-Path $moveSmoke 'summary.json') -Raw | ConvertFrom-Json
if ($smoke.status -ne 'complete' -or -not $smoke.hashes_unchanged) { throw 'Smoke incomplete; stop.' }
```

Lệnh full đã dùng, giữ để kiểm tra protocol:

```powershell
$moveFull = Join-Path (Get-Location) "runs\combat-move-full-$moveStamp"
& $py -u .\run_combat_move_pilot.py --device $device --output-dir $moveFull --budget-minutes 120 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath "$moveFull-console.log"
if ($LASTEXITCODE -ne 0) { throw 'Runner failed; send available ZIP/logs.' }
```

Runner từ chối ghi đè output. Không có resume Adam/RNG giữa train; nếu máy tắt khi chưa đủ train, checkpoint weight không phải trạng thái để tiếp tục chính xác lượt đó. Giữ kết quả incomplete để review trước khi bắt đầu lượt mới. Không tạo timestamp mới rồi tìm `summary.json` trong thư mục chưa từng chạy.

Đọc lại đúng đường dẫn khi mở PowerShell mới:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$moveFull = (Get-Content -LiteralPath 'runs\combat-move-last-path.txt' -Raw).Trim()
Get-Content -LiteralPath (Join-Path $moveFull 'summary.json') -Raw
Get-Item -LiteralPath "$moveFull-send.zip", "$moveFull-console.log"
```

`summary.json` chỉ xuất khi runner kết thúc; lúc đang chạy xem console và `control/metrics.jsonl` hoặc `combat_move/metrics.jsonl`. Smoke lưu path riêng trong `combat-move-smoke-last-path.txt`.

## Dữ liệu dùng đánh giá

Runner tự tạo **`...-send.zip`** gồm:

- `config.json`, `source/`, `preflight.json`, `baseline_skill.json`, `selected_models.json`, `summary.json`.
- Mỗi nhánh: config/summary, metrics/validation_metrics/skill_metrics JSONL, skill tapes GZIP, best/final checkpoints vàbest_validation.
- Ba `evaluation_*_33000.json`: mọi game row, ActiveWR/WR/draw/suicide/kills/bombs/items/steps, passive wins vàA0/A1.

Nếu bạn chạy ở một lượt được đăng ký sau này, gửi ZIP và console log; không phải tự chọn vài metrics đẹp. `move_guidance` log queries/accepted/rejections, agreement, bomb-instead-of-MOVE, tổng probability/entropy, pressure gain, certificate ticks vàmilliseconds. Chia tổng cho denominator tương ứng khi phân tích; control không chạy certificate, nên `accepted=0` của control không phải coverage kém.

Trongconsole,`combat bombs x/y` cóy bao gồm cảMOVE được chuyển vào combat; không lấyx/y làm tỷ lệchọnBOMP trênriêngBOMPtargets. Nhãn/inference vàsafety phảiđọc từCSV/game rows/mask,không từ dòngtổnghợpđó.

Skill groups cần xem riêng. Bom chọn thay MOVE là diagnostic, không tự chứng minh mọi bom đó vô ích. Action khác nhãn có thể là một đường tiến khác hợp lệ; replay phải kiểm tra hình học và outcome. Các seed/orientations/roles của synthetic cases phụ thuộc nhau, không coi320 skill games là320 board độc lập. Chỉ một train seed thì vẫn là pilot.

## Kiểm tra implementation đã chạy

136 tests PASS: certificate/purity/slots/collision/pressure/greedy failure, chuyển đúng loss group và giữ actionCNN, actual qualified update trênCPU/CUDA, default local khớp before snapshot ở mọi tensor/reward/action; orchestration/gates/paired seeds/incomplete/changed-source/fresh-test guards. Mock full outcomes nằm trong memory, không phải kết quảRL.

CPU/CUDA smoke4ep/arm,max8 đã hoàn tất, mỗiarm mộtAdam/4updates/32learner steps; full_protocol=false,bestep0,không thayB. Smoke chưa gặp move đủ điều kiện trong32steps/arm, nên không dùng WR/approach của smoke để nói treatment hiệu quả; actual promoted branch đã được test riêng trênpositive envCPU/CUDA. Preflight smoke nhận6/32move states, p95≈13–15ms dưới tải test. Deadline0,001phút trả incomplete/ZIP/retain_original và không mở test.

Source trainer có option/metrics mới, nên guard của các runner reuse control cũ sẽ từ chối root source hiện tại. Đây là lý do phải train lại cả hai arms. Raw T1/T2/T3/source snapshots/checkpoints cũ được giữ; tests của archived control đối chiếu source snapshot, không bỏ hash guard production.
