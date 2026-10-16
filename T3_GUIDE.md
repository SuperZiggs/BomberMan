# T3.1 — benchmark tình huống có lời giải trước khi train curriculum

Runner: [benchmark_t3.py](/D:/fpt/subject/BomberMan/benchmark_t3.py). Bài này chứng nhận tình huống, kiểm tra teacher rồi đo **checkpoint gốc B** bằng evaluator hiện tại. CNN sample qua production mask; teacher không chọn hành động cho CNN. Chưa có cập nhật weights hoặc lượt train curriculum trong runner.

**Đã chạy full T3.1 ngày14/10/2026:** [báo cáo](/D:/fpt/subject/BomberMan/runs/t3-full-20261014-implementation/T3_decision.md),1.440 games/240 cases,96/96 reference/teacher combo pass. B trên combo owned-kill success static87,5%,evader47,92%,Rule38,54%;0 suicide. Đã review62.110 frames. T3.1 đóng; chưa cần chạy lại các lệnh bên dưới. T3.2 runner/tests/smoke hoàn tất (110 tests); pilot control600/curriculum200 dừngC1, [review](/D:/fpt/subject/BomberMan/runs/t3-curriculum-full-20261014-154703/T3.2_decision.md). [Control confirmation32000 đã review](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/confirmation_report.md): delta2,25pp/CI chứa0,gate FAIL,giữ B. Tiếp theo là offline approach/counter-bomb audit, chưa train mới.

## Tình huống và đầu ra

Universe336 cases, bàn9×9, hai agents, observation12×9×9. Mỗi base case có8 phép xoay/phản chiếu và cả vaiA0/A1. Các vai được hoán đổi tại vị trí tình huống; đây không phải các trận xuất phát spawn corner ngẫu nhiên.

| Family | Split | Nội dung |
|---|---|---|
| fork | train | Reuse geometry combo_arena, một bom/tiếp cận/evader/combo và ammo controls |
| long_retreat | validation | Mở rộng vùng rút lui; kiểm tra khả năng chuyển kỹ năng |
| long_fork | validation | Đối thủ xa hơn, nhiều ô thoát, range3/4 |
| wood_branch | test | Gỗ bị phá và lửa còn sót; đã unit-test physics/certificate, chưa chạy CNN trong lượt mặc định |

Các family có quan hệ hình học gần nhau; không coi đây là kiểm tra tổng quát trên map random. Tất cả biến thể của một family ở cùng split. Không dùng `--splits test` để tune phương pháp; khi mở so sánh sau train cần cố định config trước và giữ bài test Phase C mới theo registry.

Full mặc định chạy train+validation =240 cases, B×3 opponents (static/evader/RuleBot)×2 RNG seeds =1.440 games. Lượt smoke dùng một hướng vàA0, không đủ chọn phương pháp. Có tests cho certificate combo qua mask tới hết lửa, ammo1/timer trả lại, guards, state/RNG, prefix death và CNN-only evaluation.

Reference combo hợp lệ đòi hỏi baseline và từng bom riêng còn đường sống, combo hết đường sống trên cùng horizon, đặt bom đúng lịch/capacity và learner có prefix + đường thoát qua mask. Reference được định nghĩa riêng, không lấy từ teacher. Reference tiếp cận chỉ chứng minh owned kill trước target đứng yên; không gọi nó là trap trước mọi cách né.

Mask được giữ nguyên. Certificate escape search có bound2.048 states; vượt bound ghi `search_limit`, không coi là chứng minh không có lời giải. Certificate có thể tiếp tục physics sau terminal để xác minh learner sống hết nguy hiểm; kết quả đó tách khỏi actual CNN game win. Stress reference với evader/RuleBot ghi prefix, survival và mask của toàn action tape riêng; không chứng minh mọi phản ứng tương lai.

Env vẫn max_steps500/reward hiện tại. Benchmark dừng quan sát sau80 bước; `skill_horizon_truncated` là chưa giải xong bài kỹ năng, tách khỏi `actual_draw` khi env thật sự kết thúc. Không so tỷ lệ này trực tiếp với ActiveWR500-step Phase C.

Mỗi game lưu hai loại số liệu: `two_or_more_placements` và `has_genuine_combo_pair`. Pair metric dùng hai placement liên tiếp thật sự của CNN, dự báo từ state trước bom đầu với bom đã biết; không giả định đặt được kế hoạch không xảy ra. Nó vẫn là dự báo khi đối thủ có thể đặt bom mới. Owned kill/suicide là outcome thật của game và được lưu riêng.

## Chạy lại nếu cần tái lập

Đầu tiên đọc báo cáo của lượt đã có trong tracker. Các lệnh này tạo **lượt mới**, không resume hay ghi đè lượt cũ. CPU chạy được, không cần CUDA.

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
$t3Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py -m unittest test_benchmark_t3 test_l2_candidates test_audit_t2 test_tactical_teacher test_combat test_visualize test_two_bomb_complete -q 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "runs\t3-tests-$t3Stamp.log"
if ($LASTEXITCODE -ne 0) { throw 'Tests chưa qua; chưa chạy benchmark.' }

$t3Smoke = "runs\t3-smoke-$t3Stamp"
& $py -u .\benchmark_t3.py --smoke --splits train --repeats 1 --horizon 80 --seconds 120 --output $t3Smoke 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "$t3Smoke-console.log"
if ($LASTEXITCODE -ne 0) { throw 'Smoke lỗi; gửi status.json và console log.' }
$smoke = Get-Content -LiteralPath (Join-Path $t3Smoke 'summary.json') -Raw | ConvertFrom-Json
if ($smoke.status -ne 'complete' -or -not $smoke.hashes_unchanged) { throw 'Smoke chưa hoàn tất.' }

$t3Full = "runs\t3-full-$t3Stamp"
& $py -u .\benchmark_t3.py --splits train validation --repeats 2 --horizon 80 --seconds 1800 --output $t3Full 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "$t3Full-console.log"
if ($LASTEXITCODE -ne 0) { throw 'Benchmark lỗi; giữ dữ liệu và gửi status/console log.' }
$full = Get-Content -LiteralPath (Join-Path $t3Full 'summary.json') -Raw | ConvertFrom-Json
if ($full.status -ne 'complete' -or -not $full.hashes_unchanged -or $full.cases_audited -ne 240 -or $full.games_completed -ne 1440 -or $full.gate.reference_combo_full_mask_pass -ne 96) { throw 'Benchmark chưa đủ lượt mặc định; gửi dữ liệu đã chạy để chẩn đoán.' }
Write-Host "Gửi: $t3Full-send.zip và hai console logs/test log"
```

Tái lập kiểm chứng ZIP/source/checkpoints, reference/teacher và mọi action tape trên source snapshot đã lưu:

```powershell
& $py -u .\review_t3_benchmark.py $t3Full
if ($LASTEXITCODE -ne 0) { throw 'Review không khớp dữ liệu; giữ output và gửi log.' }
```

Review tạo `analysis_results.json` riêng, không thay summary/ZIP gốc. Nó kiểm tra các first-actions teacher tương đương, kết quả từng trận và chất lượng các pair thật sự được CNN đặt.

Ngân sách được kiểm tra giữa các case; case đang chạy hoàn tất rồi mới dừng. Nếu hết giờ, output ghi incomplete và vẫn xuất ZIP/summary phần đã hoàn tất. Ctrl+C hoặc lỗi giữ status và certificates đã tính, chưa bảo đảm có ZIP. Không gọi kết quả dở là lượt đầy đủ. Giữ source/checkpoint ổn định trong lúc chạy vì runner kiểm tra hashes cuối lượt.

Sau reboot, dùng đường dẫn thư mục đã chạy thay vì tạo stamp mới để đọc summary:

```powershell
$t3Full = 'D:\fpt\subject\BomberMan\runs\t3-full-20261014-implementation'
Get-Content -LiteralPath (Join-Path $t3Full 'summary.json') -Raw
```

## File cần giữ/gửi

- `*-send.zip`: config/hashes, source, scenarios, certificates_teacher, summary/status và games.jsonl.gz.
- `*-console.log` và `t3-tests-*.log`.
- `games.jsonl.gz` có đủ action tapes để dựng lại trận; giữ file này để phân tích combo/va chạm, không chỉ gửi tỷ lệ tổng.

Quyết định tiếp theo: positive reference fail thì sửa tình huống/certificate; reference pass nhưng teacher không tìm được candidate đúng thì sửa search/chọn nhãn; reference và teacher pass nhưng CNN thiếu kỹ năng thì đăng ký một pilot curriculum có đối chứng. Nếu CNN đã giải tốt bài combo, tìm hạn chế ở phân phối/layout hoặc phản ứng đối thủ trước khi train thêm. Gate này chọn hướng nghiên cứu, không thay checkpoint.
