# L2.0: kiểm tra candidate qua future mask trước khi sửa teacher

**Đã hoàn tất lượt full `l2-full-20261014-120348` và review ngày14/10/2026.** Không cần chạy lại các lệnh audit dưới đây. 40 tapes/149 states/6.666 paths: old representative10/149 viable, alternatives85/149, selected besttwo10, genuine trap0. Top8 khôi phục43/85 best; RuleBot chỉ1/10 besttwo complete và sống. Quyết định: giữ B, triển khai T3.1 benchmark có certificate trước train; L2.1/L2.2 deferred. [Báo cáo review](/D:/fpt/subject/BomberMan/runs/l2-full-20261014-120348/L2_decision.md), [tracker](/D:/fpt/subject/BomberMan/PLAN_BOMBERMAN.md). Các lệnh phía dưới là protocol lưu lại, không phải yêu cầu mở lượt train mới.

Ưu tiên hiện tại: teacher chọn nhãn từ prefix có thể thực thi qua mask. T2 đã tìm thấy best plan thường chỉ qua mask ở bước đầu. Trước khi sửa online teacher, L2.0 cần biết có candidate thay thế trong search hiện tại và chi phí guard là bao nhiêu.

Đã có `audit_l2_candidates.py`, reuse observer và clone/suffix helper của T2. Core teacher/train/mask/reward/checkpoint giữ nguyên. 82 tests qua (7 candidate tests +75 tests khác). Các unit test runner train đã mock GPU để chạy trên CPU; có test riêng bảo đảm train thiếu CUDA vẫn bị chặn. Smoke4 state từ seed28000, cả hai model/corner: old representative0/4 pass; alternative4/4 pass, chưa có viable two-bomb. Bốn state không đủ chọn cách sửa cuối.

## Corpus và phép đo

- Đọc lượt T2 `runs/t2-full-20261014-060320`, kiểm tra config/trace khớp ZIP gốc và source/checkpoint hashes.
- Full L2.0:10 board seeds28000–28009, cảA0/A1, B và final200 =40 game tapes; tối đa160 state. Rebuild env theo action tape có sẵn, không inference CNN hay chơi trận benchmark mới.
- Chọn tối đa4 state/game, từ tick50/150/250/350/450 theo thứ tự, chỉ lấy tick có nhãn T2. Không chọn theo win/loss. Game kết thúc sớm hoặc tick không có nhãn sẽ có ít state; `games_sampling` ghi coverage.
- Capture **mọi scored path** trong tối đa64 schedules, gồm candidate đứng sau best. Chỉ gồm những candidate teacher gốc đã cho qua physics/survival, positive pressure và first mask; chưa là exhaustive search mọi vị trí.
- Với mỗi candidate: static opponent STOP, recompute production robust mask tại từng prefix action tới bom cuối; kiểm tra placement/collision/capacity thật, rồi chạy physical timed-BFS suffix tới hết horizon. Được ghi viable khi prefix hoàn tất, learner sống hết horizon và blast range thực đúng như plan. Pickup làm range đổi được ghi riêng và loại khỏi lựa chọn để giữ giả định forecast/score. Suffix chưa bắt buộc đi qua mask và chưa là hành động CNN.
- Chọn best trong các candidate viable bằng score cũ, giữ tập first actions tương đương. Báo riêng any viable two-bomb và best được chọn two-bomb: tồn tại một combo không có nghĩa scoring sẽ chọn nó.
- Đối chiếu top1/4/8 **schedules**, mỗi schedule giữ mọi scored paths. Báo best-score recall và first-actions exact; hai metric khác nhau. Timing gồm clone/suffix, chưa phải timing của online guard tối ưu.
- Chạy stress evader/RuleBot cho selected best, seed RNG=`board_seed*1000+aid*500+tick`; không giả định khớp RNG của branch T2 cũ. Branch/state/RNG được cô lập.
- Nominal corpus giới hạn900 giây, cooperative deadline. Check đang chạy, tổng hợp và đóng ZIP có thể làm tổng wall time vượt nhẹ. Hết budget đánh dấu incomplete và giữ coverage, không tính partial states vào denominator đầy đủ.

## Các lệnh chạy

Mở **PowerShell**, chạy theo thứ tự. Block1 khởi tạo một lượt mới và chạy test/smoke:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
$l2Input = 'D:\fpt\subject\BomberMan\runs\t2-full-20261014-060320'
$l2Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$l2Smoke = "D:\fpt\subject\BomberMan\runs\l2-smoke-$l2Stamp"
$l2Full = "D:\fpt\subject\BomberMan\runs\l2-full-$l2Stamp"

& $py -m unittest test_l2_candidates test_audit_t2 test_tactical_teacher test_combat test_visualize test_two_bomb_complete -q 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "runs\l2-tests-$l2Stamp.log"
if ($LASTEXITCODE -ne 0) { throw 'Tests failed: gửi test log trước khi chạy tiếp.' }

& $py -u .\audit_l2_candidates.py --input $l2Input --boards 1 --states-per-game 1 --seconds 120 --output $l2Smoke 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "runs\l2-smoke-$l2Stamp-console.log"
if ($LASTEXITCODE -ne 0) { throw 'Smoke failed: gửi status.json và console log.' }
$smoke = Get-Content -LiteralPath (Join-Path $l2Smoke 'summary.json') -Raw -ErrorAction Stop | ConvertFrom-Json
if ($smoke.status -ne 'complete' -or -not $smoke.hashes_unchanged -or $smoke.games_replayed -ne 4 -or $smoke.overall.fully_checked_states -ne $smoke.states_checked) { throw 'Smoke chưa đủ điều kiện chạy full.' }
```

Smoke tối đa4 states; `full_protocol=false` là đúng. Runner có thể tạm ngừng in khi đang đọc qua những game ngoài corpus trong trace lớn.

Sau block1 thành công, chạy block2 trong **cùng cửa sổ PowerShell**:

```powershell
& $py -u .\audit_l2_candidates.py --input $l2Input --boards 10 --states-per-game 4 --seconds 900 --output $l2Full 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "runs\l2-full-$l2Stamp-console.log"
if ($LASTEXITCODE -ne 0) { throw 'L2 audit lỗi: gửi status.json và console log.' }
$full = Get-Content -LiteralPath (Join-Path $l2Full 'summary.json') -Raw -ErrorAction Stop | ConvertFrom-Json
if ($full.status -ne 'complete' -or -not $full.full_protocol -or -not $full.hashes_unchanged -or $full.games_replayed -ne 40 -or $full.overall.fully_checked_states -ne $full.states_checked) { throw 'L2 chưa hoàn tất corpus; giữ ZIP/log để đánh giá phần đã chạy.' }
Write-Host "Gửi: ${l2Full}-send.zip"
Write-Host "Gửi thêm: runs\l2-tests-$l2Stamp.log và runs\l2-full-$l2Stamp-console.log"
```

Mỗi output phải là thư mục mới; runner dừng nếu thư mục hoặc ZIP đã có. Giữ source/checkpoint/input T2 ổn định trong lúc chạy. Lượt này chưa có resume.

## Khi mở lại PowerShell

`Get-Date` chỉ dùng tạo lượt mới. Khi đọc lượt đã chạy, gán đường dẫn thư mục cũ:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
Get-ChildItem -LiteralPath .\runs -Directory -Filter 'l2-full-*' | Sort-Object Name -Descending | Select-Object Name
```

Chọn tên thư mục đã có, rồi đọc `summary.json`/`status.json` trong đó. Không khởi chạy lại vào output cũ.

## File cần gửi và quyết định sau khi đọc

Gửi ZIP `runs/l2-full-<timestamp>-send.zip`, test log và full console log. ZIP chứa:

| File | Ý nghĩa |
|---|---|
| `config.json`, `status.json` | Sampling/budget/hash, coverage và tình trạng hoàn tất |
| `summary.json`, `report.md` | Old best vs alternatives, viable combo/trap, top-K recall, timing, stress checks |
| `candidates.jsonl.gz` | Mỗi state và mọi scored path đã kiểm tra, lịch bom/score/check/mask/suffix và selection |
| `source/` | Snapshot code/helper/test |

Nếu ZIP lớn, gửi trước config/status/summary/report và console/test log; giữ candidates để phân tích case. Nếu cùng workspace, chỉ cần báo đường dẫn output. Nếu Ctrl+C/lỗi trước khi tạo ZIP, gửi status.json và console log, giữ thư mục dở.

Từ dữ liệu sẽ chọn một bước:

1. Alternatives hữu ích và chi phí phù hợp: L2.1 sửa việc chọn nhãn, giữ paths đi qua future mask và fallback local; test rồi mới pilot L2/local từB.
2. Alternatives có nhưng gần hết là một bom/ít trap: T3.1 benchmark combo có thể qua mask; curriculum là thí nghiệm riêng.
3. Corpus incomplete hoặc top-K bỏ nhiều alternatives: hoàn thiện phép đo trước, chưa train dài để bù.

L2.0 không chứng minh checkpoint mới tốt hơn. Corpus là audit đã xem, không dùng promote. Guard static pass chưa bảo đảm mọi phản ứng/bom tương lai của đối thủ.
