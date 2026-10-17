# Quyết định sau L2.0 — 14/10/2026

**L2.0 hoàn tất và số liệu khớp dữ liệu thô. Giữ checkpoint B; ưu tiên T3.1 benchmark tình huống trước curriculum. L2.1/L2.2 chưa mở train.** Xét candidate thay thế có ích cho tính khả thi của kế hoạch, nhưng corpus này chưa cung cấp đủ kế hoạch hai bom chặn hết đường thoát để coi guard là cách giải quyết mục tiêu combo.

## 1. Kiểm chứng lượt chạy

- Test log ghi `Ran 82 tests ... OK`. Dòng `System.Management.Automation.RemoteException` là cách PowerShell hiển thị stderr của unittest; không có test FAILED.
- `status=complete`, `full_protocol=true`, `hashes_unchanged=true`; 40/40 game tapes, 10 seed bàn 28000–28009, hai model và cả A0/A1. Lượt audit ghi 28,27 giây.
- Đã kiểm tra CRC và SHA của mọi member trong ZIP so với file đã giải nén; source/checkpoint hiện tại khớp snapshot và config T2. Config/trace T2 khớp ZIP gốc.
- Đã tính lại mọi trường summary từ 149 states/6.666 scored paths. Dựng lại đủ 40 trận/18.286 frames, kiểm tra fingerprints, sampling/mask/nhãn, tạo lại mọi candidate và replay 6.666 static guards +170 stress branches. Kết quả, action tapes, suffix và mask ở bước bị chặn khớp bản gửi, trừ thời gian tính.
- 149/160 là đúng sampling, không phải thiếu protocol. Có 5 game gốc chỉ có 3/3/2/1/0 state đủ điều kiện; trận `original-28009-A0` kết thúc ở bước49, trước tick lấy mẫu đầu tiên50. Final có đủ80 states; gốc có69.
- State và NumPy/Torch RNG được kiểm tra không đổi sau các phép tìm kiếm/clone. Checkpoints và file audit gốc giữ nguyên; báo cáo sau review nằm riêng.

Tái lập bằng [analyze_l2_results.py](/D:/fpt/subject/BomberMan/runs/l2-full-20261014-120348/analyze_l2_results.py); kết quả tại [analysis_results.json](/D:/fpt/subject/BomberMan/runs/l2-full-20261014-120348/analysis_results.json) và [analysis_console.log](/D:/fpt/subject/BomberMan/runs/l2-full-20261014-120348/analysis_console.log).

## 2. Kết quả chính và ý nghĩa

| Chỉ số trên states hoàn tất | Gốc B | Final two_bomb200 | Tổng |
|---|---:|---:|---:|
| States | 69 | 80 | 149 |
| Kế hoạch đại diện cũ khả thi | 5/69 | 5/80 | 10/149 =6,71% |
| Có ít nhất một candidate khả thi | 42/69 | 43/80 | 85/149 =57,05% |
| Có candidate hai bom khả thi | 8/69 | 6/80 | 14/149 =9,40% |
| Best khả thi được chọn là hai bom | 5/69 | 5/80 | 10/149 =6,71% |
| Có combo thật chặn hết đường thoát | 0/69 | 0/80 | 0/149 |

Trong 139 states mà đại diện cũ không khả thi, 75 có phương án thay thế qua guard (53,96%). Đây là bằng chứng nên chọn từ candidates hợp lệ thay vì chỉ lọc best cũ. Tuy nhiên, **75/85 best khả thi là một bom**. Trong 6.666 scored paths, 5.701 là hai bom; chỉ27 paths hai bom qua guard, so với288 paths một bom. 6.260 paths bị mask chặn và91 kết thúc trận trước khi hoàn thành prefix.

Không có path nào trong 6.666 scored paths có `trapped>0`, kể cả trước guard. Đã tính riêng baseline/combo/first-only/second-only trên cùng horizon cho cả27 lịch hai bom khả thi: tất cả combo vẫn để lại vị trí sống. Vì vậy số0 ở đây không chỉ do bộ lọc guard làm mất các trap đã có. Search hiện tại vẫn giới hạn ≤64 lịch/trạng thái; chưa thể suy ra không có chiến thuật thắng ngoài vùng tìm kiếm hoặc trong mọi layout.

Trong 75 states được cứu, **40 giữ nguyên tập nhãn hành động đầu**, 22 đổi sang tập không giao nhau và13 đổi tập nhưng còn giao nhau. Thay một đường đi dự báo bằng đường đi hợp lệ chưa chắc thay tín hiệu học ở bước hiện tại. Trainer chỉ nhận phân phối hành động đầu và CNN tự chơi; không được coi replay kế hoạch là CNN đã học/thực hiện combo.

## 3. Chi phí và giới hạn top-K

K là số lịch đặt bom được xét theo điểm teacher cũ; mỗi lịch giữ mọi scored paths. Mẫu số recall là85 states có best khả thi khi xét đầy đủ.

| Giới hạn | States tìm được candidate | Khớp best score và tập first actions đầy đủ | Best được chọn là hai bom | Guard trung bình / p95 |
|---|---:|---:|---:|---:|
| Top1 | 10/149 | 10/85 =11,76% | 0 | 2,12 /4,02 ms |
| Top4 | 24/149 | 24/85 =28,24% | 0 | 7,89 /14,42 ms |
| Top8 | 43/149 | 43/85 =50,59% | 2 | 14,89 /27,49 ms |
| Toàn bộ scored candidates | 85/149 | 85/85 theo định nghĩa | 10 | 70,40 /155,03 ms |

Teacher tạo candidates trung bình15,48 ms; toàn phép audit một state trung bình93,31 ms, p95 188,07 ms, gồm stress cho best. Đây là CPU clone audit, chưa phải benchmark guard trong training loop. Không chọn K=8 làm mặc định chỉ vì chạy nhanh: nó bỏ42/85 states có best hợp lệ trong corpus này. Chưa đưa toàn bộ guards vào mọi query train khi chưa có fixture hữu ích và phương án bound giữ đúng semantics/tied labels.

## 4. Khi đối thủ phản ứng

85 best từ static guard được kiểm tra với evader rồi RuleBot, dùng seed đã ghi trong từng state và giữ đúng thứ tự RNG của runner.

| Đối thủ | Prefix qua mask | Sống hết horizon trong nhóm prefix complete | Prefix complete và sống / tất cả85 |
|---|---:|---:|---:|
| Evader không đặt bom | 62/85 | 62/62 | 62/85 |
| RuleBot | 56/85 | 51/56 | 51/85 |

Đối với **10 best hai bom**, evader chỉ có3/10 complete và sống, RuleBot chỉ1/10; các trường hợp còn lại bị mask chặn trong prefix. Mẫu rất nhỏ; đây là stress theo kịch bản, không phải ActiveWR của CNN. Full-horizon alive chưa biết trong branch bị chặn, không được tính là tự nổ hoặc sống an toàn. Escape suffix của audit dùng dự báo bom đã biết và không ép mask ở mọi bước suffix.

Các case đã dựng lại:

| Case / tick | Quan sát |
|---|---|
| `original-28000-A0` /50 | Best cũ bị chặn trước bom đầu; candidate5 một bom qua cả ba loại đối thủ, pressure61,9%. Nhãn đầu vẫn LEFT nên không đồng nghĩa thay nhãn học. |
| `original-28000-A0` /250 | Best hợp lệ ở schedule rank21, ngoài top8; pressure22,5%, một bom. |
| `original-28001-A1` /150 | Best hai bom qua static; evader/RuleBot chặn bom thứ hai ở delay4 dù không đặt bom mới. Sự di chuyển của enemy cũng thay đổi mask. |
| `two_bomb_final-28006-A1` /350 | Hai bom complete và sống với RuleBot, nhưng enemy vẫn còn refuge trong forecast, không có owned kill trong branch; evader lại chặn bom đầu ở delay1. |
| `original-28004-A0` /150 | Prefix một bom complete, nhưng suffix chết trước counter-bombs RuleBot. Static guard không bảo đảm mọi diễn biến tương lai. |

## 5. Bước tiếp theo và gate

**T3.1 là benchmark trước, chưa phải train curriculum.** Fixture `combo_arena(2)` và test `test_genuine_combo_that_passes_mask_is_retained` đã có trong repo, cho thấy kiểm tra combo có positive control. Cần mở rộng thành nhiều tình huống thực thi được để đo chỗ CNN/teacher thiếu; số0 trên corpus ngẫu nhiên không có nghĩa luật/mask luôn cấm combo.

| Bước nhỏ | Công việc | Bằng chứng / điều kiện đi tiếp |
|---|---|---|
| T3.1a — chuẩn hóa tình huống | Reuse fixture; bốn nhóm: đặt một bom và thoát, tiếp cận mục tiêu đứng yên, mục tiêu biết né, combo hai bom có nhiều refuges. Kiểm tra legal state, ammo/timer/range/gỗ/lửa/collision. Có controls một bom đủ thắng và tình huống không có trap. | Scenario JSON và action tapes có thể dựng lại; giữ luật/12 kênh/mask/reward. Không xếp một tình huống không có phương án qua mask vào positive combo. |
| T3.1b — chứng nhận combo | Cùng horizon: baseline>0, từng bom riêng>0, combo=0; thực thi lịch/capacity qua `env.step`; prefix và một đường thoát tới hết nguy hiểm đều qua production mask, learner sống. Làm static trước rồi ghi khả năng thực thi riêng với evader/RuleBot. | Có positive controls ở nhiều bố cục; ammo1 không giả định hai bom đồng thời. Nếu fail, ghi rõ physics/mask/placement, sửa fixture/certificate trước. Reference controller chỉ xác nhận bài toán có lời giải, không chọn action cho CNN trong benchmark. |
| T3.1c — audit teacher trên tình huống | Kiểm tra planner hiện tại có tìm được lời giải được chứng nhận trong ≤64 lịch hay không; lưu chất lượng nhãn, full-path pass, trap, first-actions/tied paths và thời gian. | Nếu certificate có combo nhưng teacher không tìm được: mở một sửa search/scoring/guard riêng, rồi test lại. Nếu teacher tìm được mà CNN không thực hiện: hướng supervision/curriculum, chưa đổi GAE/PPO. |
| T3.1d — benchmark B | Chạy CNN/mask hiện tại trên cảA0/A1; giữ policy action selection của evaluator. Đo owned-kill success, suicide, hai placement, survival và steps theo nhóm. Final200 có thể thêm làm diagnostic, B vẫn baseline. | Cố định config/nguồn/checkpoint và lưu từng trận/replay; xoay/phản chiếu cùng layout ở cùng split, tách family train/validation/test trước khi dùng để train. |
| T3.1e — chọn một nhánh | Certificate pass + teacher pass nhưng B thiếu kỹ năng: pilot T3.2 curriculum đối chứng cùng ngân sách. Teacher fail: sửa teacher trước. Task không qua mask: chưa mở train để bù. | L2.1/L2.2 chỉ mở lại khi valid combos/chi phí/nhãn đã được chứng minh đủ dùng; không gộp curriculum và optimizer/reward trong một ablation. |

Bạn chưa cần chạy lại L2 hay train thêm từ final200. Commands train curriculum chưa được đưa ra vì scenario runner/certificate chưa triển khai; bước triển khai kế tiếp có scope cụ thể trong bảng trên. Nếu chỉ muốn tái lập review đã hoàn tất:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
& $py -u .\runs\l2-full-20261014-120348\analyze_l2_results.py
if ($LASTEXITCODE -ne 0) { throw 'L2 review không khớp dữ liệu hoặc source/checkpoint.' }
```

## 6. Phạm vi kết luận

Đây là audit mô tả trên10 seed bàn đã xem trong T2, không phải149 trials độc lập hay bộ test mới. Gốc/final tạo các trạng thái khác nhau và số state khác nhau; không coi chênh coverage của hai model là hiệu ứng nhân quả của phương pháp. Không có lượt train mới hoặc phép đo ActiveWR mới từ L2. Kết luận giữ B dựa trên T1/T2; L2 giúp chọn bước sửa tiếp theo.

Nguồn quy trình: skill [scientific-critical-thinking](C:/Users/ADMIN/.codex/skills/scientific-critical-thinking/SKILL.md) để giữ đúng câu hỏi, đơn vị seed, mẫu số và phạm vi suy luận. Kassis et al. (2026), [Scientific Agent Skills](https://arxiv.org/abs/2609.00065), là nguồn quy trình đã kiểm tra metadata ở T2, không phải bằng chứng Bomberman. Không bổ sung paper hay thuật toán mới trong review này.
