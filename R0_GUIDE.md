# T4-R0 — audit reward trên validation đã dùng

**Đã hoàn tất/review:** `runs/r0-full-20261015-153908`,300trận/16,25phút;118.226bước khớp reward cũ,12timeout/6.012frames replay qua,188tests PASS/1CUDA SKIP. Giữ B và coefficients; R1 chỉ được đề xuất như ablation tiếp theo. [Báo cáo](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/R0_report.md).

R0 chỉ chạy suy luận với B, MC final200 và GAE final200 từ `runs/gae-full-20261015-125942`. Mỗi model100 trận, seed21000–21049, cảA0/A1, CPU một thread, bàn9×9/2agents/max500. Đối thủ RuleBot và cách CNN lấy mẫu qua mask giống validation B1. Không cập nhật model, chọn checkpoint hoặc mở seed test mới.

`audit_r0.py` đối chiếu tỷ lệ thắng/hạ đối thủ/hòa/tự nổ và trung bình kills/bombs/items/steps với validation B1 ở đúng checkpoint. Hash checkpoint, source và dữ liệu parent được khóa trước audit, kiểm tra lại khi kết thúc. Thư mục output tồn tại sẽ bị từ chối ghi đè.

## Reward và kiểm chứng

`_shaped_reward` dùng đúng thứ tự cộng cũ: env reward → combat shaping → wall bonus. `_combat_reward` không đổi. Trainer có `reward_diagnostics=False` mặc định; bật bằng `--reward_diagnostics` để thêm `reward_components` và sai số tối đa mỗi bước vào metrics từng episode. Chỉ cộng các transition learner sống trước step và thực sự được lưu học; bước chết vẫn được lưu, bước sau khi đã chết bị bỏ.

Các thành phần: time, item, wall, owned_kill, active_win, passive_win_env, passive_win_cancellation, death, self_blast. Item hữu ích +0,05; phá gỗ +0,15 khi sống sau step; kill tổng+4 khi sống; sole-survivor+10 được giữ nếu có owned kill trong episode; passive+10 bị hủy−10; chết−1, thêm−1,5 nếu có lửa của bản thân. Tick chết không có time penalty. Không có terminal penalty riêng cho timeout.

Decomposer kiểm tra cả raw env reward và tổng shaped reward, sai số tuyệt đối≤1e−10, không thêm residual để ép tổng khớp. Số thực có thể lệch vài ulp khi nhóm lại thành phần; đường reward học được giữ bit-exact trên test parity CPU trước/sau refactor. Loss imitation không được tính là reward.

Thống kê mean/median/p95/min/max có cho từng thành phần, reward và hành vi; tách active win, passive win, timeout, self-blast và enemy-death. Nhóm không có trận ghi `games:0`, không thay dữ liệu thiếu bằng0. Timeout là hết500step khi còn cảhai sống; hai agent chết cùng lúc là draw nhưng không phải timeout.

Không dùng reward phụ chia tổng reward. `gross_abs_shares` dùng tổng trị tuyệt đối của từng component theo episode; `net_abs_shares` loại cặp passive+10/−10 vốn triệt tiêu để tránh làm phình mẫu số. Cả hai định nghĩa được giữ trong JSON.

## Chẩn đoán giao tranh

- Khoảng cách Manhattan, lần đầu quan sát khoảng cách≤3, số bước gần đối thủ và số bước thật sự di chuyển gần hơn.
- Hướng tiếp cận/firing-position từ BFS hiện có qua mask; số lần có hướng và CNN chọn hướng đó. Đây là gợi ý đầu tiên, không chứng minh toàn bộ đường đi an toàn.
- Bom có pressure>0, bom pressure0 nhưng tia hiện tại chạm gỗ, và bom không thuộc hai nhóm trên; thêm cờ enemy nằm trên tia. Chỉ tính các lần env thực sự đặt bom.
- Pressure đo giảm refuges với bom đã biết và vị trí đối thủ hiện tại; không bảo đảm kill hoặc dự báo mọi phản bom tương lai. Nó chỉ được log, không thay action.

`steps_<model>.jsonl.gz` lưu mọi transition learner, actions/mask/events/reward/thành phần/diagnostics. Bốn timeout đầu theo thứ tự seed/góc mỗi model được lưu đầy đủ frames và trace để xem lại; lựa chọn này dành cho kiểm tra, không dùng12case để ước lượng tỷ lệ toàn bộ timeout. Mọi timeout trong100trận/model đều có trong statistics và game rows.

`reward_scatter.png/.svg` biểu diễn reward theo owned kills và chỉ báo active-win0/1, tách model/outcome. CSV giữ giá trị nguyên gốc; jitter x xác định≤0,035 chỉ tránh chồng điểm. Model ActiveWR là thống kê tổng hợp trong JSON, không gọi biến0/1 là tỷ lệ. Không có smoothing, loại outlier hoặc CI; hai góc cùng seed không được coi là hai board độc lập. Figure manifest ghi dữ liệu, biến đổi và alt text.

## Lệnh

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
& $py -m unittest test_reward_audit -q
$r0Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py -u .\audit_r0.py --smoke --output-dir "runs\r0-smoke-$r0Stamp"
& $py -u .\audit_r0.py --budget-minutes 60 --output-dir "runs\r0-full-$r0Stamp"
```

Smoke2trận/model/max20 chỉ kiểm tra luồng, `full_protocol=false`. Full hoàn tất cần đủ300trận, parity validation, ít nhất10timeout replays và hash không đổi. Budget/cancel lưu phần đã hoàn thành và ZIP, không suy ra kết luận từ lượt thiếu. `runs/r0-last-path.txt` giữ đường dẫn output mới nhất qua lần mở PowerShell sau.

Kết quả gồm config/source, games từng model, step tapes, timeout replays, statistics, figure data/PNG/SVG/manifest, summary và ZIP `*-send.zip`; báo cáo review thêm sau giữ riêng khỏi ZIP raw. R0 không chứng minh sửa reward sẽ tăng ActiveWR: chỉ chọn thử nghiệm một biến tiếp theo khi dữ liệu reward và hành vi cùng hỗ trợ.

Review bổ sung `discounted_returns.json` (G0 với gamma0,99), `timeout_last100.json`, `paired_outcome_rankings.json`, `component_statistics.csv` và `review_results.json`. Counterfactual giảm wall/item giữ nguyên action tape, không dự báo tỷ lệ thắng. Mọi timeout có reward dương nhưng 100bước cuối không còn wall bonus và tất cả bom đặt có pressure>0; không kết luận agent chỉ farm gỗ. Không có self-blast trong cohort; mỗi model chỉ có1death case. R0 đã mở R1 ở mức thiết kế đối chứng, chưa train hoặc đổi mặc định. Chạy lại cần workspace/checkpoints parent; ZIP raw không chứa model weights.
