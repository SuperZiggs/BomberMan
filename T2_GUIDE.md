# Thực hiện T2 đầy đủ: kiểm tra teacher và khả năng CNN làm combo

T2 là **audit/chẩn đoán**, chưa phải train thêm hay đổi thuật toán. Mục tiêu là phân biệt: nhãn yếu, nhãn không khả thi qua mask, CNN chưa học nhãn, hoặc teacher đổi kế hoạch trước khi CNN hoàn thành.

Đã có [audit_t2.py](/D:/fpt/subject/BomberMan/audit_t2.py) và [test_audit_t2.py](/D:/fpt/subject/BomberMan/test_audit_t2.py). Runner chỉ chẩn đoán: CNN sample qua mask như evaluator; teacher cung cấp nhãn để log, không chọn action. Core teacher/train, reward và checkpoint giữ nguyên. Không lấy log200 episode cũ làm dữ liệu completion vì nó chưa chứa trace cần thiết.

## Chạy ngay: test → smoke → audit đầy đủ

Mở **PowerShell**, dán theo thứ tự. Dùng Python của môi trường `torch` đã có trên máy:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$t2Smoke = "runs\t2-smoke-$stamp"
$t2Full = "runs\t2-full-$stamp"

& $py -m unittest test_audit_t2 test_tactical_teacher test_combat test_visualize test_two_bomb_complete -q 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "runs\t2-tests-$stamp.log"
if ($LASTEXITCODE -ne 0) { throw 'Tests failed: gửi log trước khi chạy tiếp.' }

& $py -u .\audit_t2.py --boards 10 --seed 28000 --max-steps 500 --check-every 10 --max-checks 50 --output $t2Smoke 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "runs\t2-smoke-$stamp-console.log"
if ($LASTEXITCODE -ne 0) { throw 'Smoke failed: gửi status.json và console log.' }
$smoke = Get-Content -LiteralPath (Join-Path $t2Smoke 'summary.json') -Raw | ConvertFrom-Json
if ($smoke.status -ne 'complete' -or -not $smoke.parity_passed -or -not $smoke.hashes_unchanged -or $smoke.games_completed -ne 40) { throw 'Smoke chưa đủ điều kiện chạy audit đầy đủ.' }
```

Smoke = **40 trận**: 10 board seeds × 2 góc × 2 model. Có thể mở `report.md` và `replays.html` trong thư mục smoke. `full_protocol=false` là đúng cho smoke. Xem `elapsed_seconds` để ước lượng chi phí; full có số bàn gấp10 nhưng độ dài và số branch có thể khác.

Windows PowerShell 5.1 có thể hiển thị `python.exe ... NativeCommandError` cho output của unittest dù cuối log là `Ran 74 tests ... OK`: unittest ghi báo cáo vào stderr. Pipeline trên chuyển output thành chuỗi trước khi Tee-Object hiển thị/lưu, tránh báo đỏ gây nhầm. Lỗi thật vẫn được bắt bằng `$LASTEXITCODE -ne 0`; không suy thành công chỉ từ màu chữ.

Sau khi block trên thành công, chạy trong **cùng cửa sổ PowerShell**:

```powershell
& $py -u .\audit_t2.py --boards 100 --seed 28000 --max-steps 500 --check-every 10 --max-checks 50 --output $t2Full 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath "runs\t2-full-$stamp-console.log"
if ($LASTEXITCODE -ne 0) { throw 'Audit chưa hoàn tất: gửi status.json và console log.' }
$full = Get-Content -LiteralPath (Join-Path $t2Full 'summary.json') -Raw | ConvertFrom-Json
if ($full.status -ne 'complete' -or -not $full.full_protocol -or -not $full.parity_passed -or -not $full.hashes_unchanged -or $full.games_completed -ne 400 -or $full.replay_cases -lt 10) { throw 'Dữ liệu chưa đủ protocol T2.' }
Write-Host "Gửi: ${t2Full}-send.zip"
Write-Host "Gửi thêm: runs\t2-tests-$stamp.log và runs\t2-full-$stamp-console.log"
```

### Đọc kết quả sau khi mở lại PowerShell hoặc khởi động lại máy

Biến PowerShell mất khi đóng cửa sổ. `Get-Date` tạo timestamp mới; khi đọc kết quả cũ, gán `$t2Full` về **thư mục đã chạy**, không tạo tên lượt mới. Liệt kê các lượt đã có:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
Get-ChildItem -LiteralPath .\runs -Directory -Filter 't2-full-*' | Sort-Object Name -Descending | Select-Object Name
```

Lượt `20261014-060320` đã kiểm tra: complete400 trận, full_protocol/parity/hashes đều true, 10 replay. Đọc lại bằng:

```powershell
$t2Full = 'D:\fpt\subject\BomberMan\runs\t2-full-20261014-060320'
$full = Get-Content -LiteralPath (Join-Path $t2Full 'summary.json') -Raw -ErrorAction Stop | ConvertFrom-Json
if ($full.status -ne 'complete' -or -not $full.full_protocol -or -not $full.parity_passed -or -not $full.hashes_unchanged -or $full.games_completed -ne 400 -or $full.replay_cases -lt 10) { throw 'Dữ liệu chưa đủ protocol T2.' }
$full | Select-Object status, full_protocol, parity_passed, hashes_unchanged, games_completed, replay_cases
```

Đây chỉ là đọc/check kết quả; không khởi chạy audit. Với lượt khác, thay đường dẫn bằng tên thư mục tương ứng. Nếu thiếu summary, đọc `status.json` và console log của chính lượt đó trước. File ZIP của lượt hoàn tất nằm cạnh thư mục với hậu tố `-send.zip`.

Full = **400 trận tổng cộng**, 200/model: seeds28000–28099, mỗi seed cả A0/A1. Đây là corpus chẩn đoán T2, khác bài gate T1 400 trận/mỗi model. Model gồm B=`runs/pressure-C-next/final_phaseC.pt` và final200=`runs/two-bomb-complete-20261014-031626/two_bomb/final_phaseC.pt`. Best two_bomb được kiểm tra bằng tensor; hiện bằng B nên ghi alias, không audit lần thứ ba.

Trước khi chạy, runner khóa source theo config T1 và hash checkpoint gốc; cuối lượt kiểm tra lại mọi file đã khóa. Không sửa source/checkpoint trong lúc chạy. Mỗi output phải là thư mục mới; nếu đã tồn tại runner dừng, không ghi đè. Muốn chạy lại từ đầu, chạy block đầu để tạo timestamp mới. Runner chưa có resume.

**Gửi để đánh giá:** ZIP `runs/t2-full-<timestamp>-send.zip` cùng test log và full console log. ZIP đã chứa:

| File | Dùng để đánh giá |
|---|---|
| `config.json`, `status.json`, `parity.json` | Protocol, checkpoint/source hashes, đã chạy đủ chưa, action/outcome/RNG parity cả hai góc |
| `summary.json`, `report.md` | Coverage, agreement/probability mass, completion, churn, pressure/trap, feasibility, timing, outcome tổng/từng góc |
| `games.jsonl`, `plan_starts.jsonl` | Kết quả từng trận và từng plan window; kiểm tra mẫu số/terminal/death |
| `trace.jsonl.gz` | Mọi bước CNN: before/after, mask/prob/action, tất cả tied plans, tracker, clone checks |
| `plan_checks.jsonl.gz` | Sáu branch/trạng thái sampled: physical/masked × static/evader/RuleBot; prefix, suffix, action tape và so combo với từng bom riêng |
| `replay_cases.json`, `replays.html` | 10 game cases theo rule cố định; player offline có prefix dự báo và đường CNN thật |
| `source/` | Snapshot runner, helper và bản instrument teacher để đọc lại dữ liệu sau này |

Nếu ZIP quá lớn, gửi trước `config.json`, `status.json`, `parity.json`, `summary.json`, `report.md`, `games.jsonl`, `plan_starts.jsonl`, test log và console log. Giữ trace/clone/replay; khi đọc replay để chọn nguyên nhân sẽ cần chúng. Không cần gửi lại checkpoint nếu còn ở đúng thư mục chung này.

Nếu Ctrl+C/lỗi: giữ thư mục, gửi `status.json`, console log và trace/plan_checks đã có. Trạng thái `incomplete` không được dùng để đóng T2. Có thể nén output dở bằng `Compress-Archive` sang một tên ZIP mới; không chạy lại vào thư mục dở.

`status=complete` chỉ xác nhận thu đủ dữ liệu theo config. T2 được đóng sau khi đọc kết quả/replay và chọn một nhánh L1/L2/L3/L4 hoặc giữ local; runner ghi `review_pending_keep_original`, chưa tự promote model.

**Review lượt060320 đã hoàn tất:** [T2_decision.md](/D:/fpt/subject/BomberMan/runs/t2-full-20261014-060320/T2_decision.md) và [analysis_results.json](/D:/fpt/subject/BomberMan/runs/t2-full-20261014-060320/analysis_results.json). Giữ B; ưu tiên L2.0 kiểm tra scored candidates qua future mask offline. L2 chưa implement/train. Raw summary và ZIP gốc là snapshot trước review; quyết định sau review ở tệp riêng. 10 replay đã review, thêm hai game đối chiếu, tổng12 games/5.608 frames dựng lại.

## 1. Chuẩn bị, khóa model và kiểm tra cái đã có

Trong PowerShell:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
& 'D:\anaconda\envs\torch\python.exe' -m unittest test_tactical_teacher test_combat test_visualize -q
& 'D:\anaconda\envs\torch\python.exe' .\runs\two-bomb-complete-20261014-031626\analyze_results.py
```

Chạy lệnh thứ hai khi core source/checkpoint chưa đổi; nó audit lại T1, không tạo trace T2. Nếu một lệnh fail thì xử lý lỗi trước khi tiếp tục. Trong terminal đã active `torch`, có thể dùng `python` thay đường dẫn Python trên.

| Nhãn trong audit | File thật | Vai trò |
|---|---|---|
| B/original | `runs/pressure-C-next/final_phaseC.pt` | Chính sách đang giữ |
| final200 | `runs/two-bomb-complete-20261014-031626/two_bomb/final_phaseC.pt` | Chính sách bị giảm validation, dùng tìm nguyên nhân |
| best | `runs/two-bomb-complete-20261014-031626/two_bomb/best_phaseC.pt` | Bằng B ở toàn bộ tensor; ghi alias và hash, không cần chơi lại cùng chính sách |

Giữ các checkpoint này. Chọn final200 trong audit không có nghĩa đổi model đang dùng hoặc chọn lại best. Có thể thêm local200 sau, nhưng hai chính sách khác nhau B/final200 đã đủ cho đợt chẩn đoán đầu.

Audit dùng 9×9, hai agents, max_steps500, CPU một thread, learner cả A0/A1. Dùng seed28000–28099 đã đề xuất trong plan, kiểm tra registry trước khi chạy. Đây là **audit set**, không phải test cuối để promote.

Thứ tự ngân sách: kiểm tra parity vài trận → smoke10 board seeds × hai góc × hai model =40 trận → nếu dữ liệu đúng mới mở100 seeds =400 trận. Đăng ký số seeds trước, giữ cả trận thành công lẫn thất bại. Đo tốc độ smoke để ước lượng thời gian; clone checks sẽ thêm chi phí ngoài planner.

## 2. Công cụ cần triển khai: một runner chẩn đoán riêng

Không cần đổi trainer, reward hay CNN để lấy trace đầu tiên. Một file runner mới có thể chia thành ba phần: thu trace, kiểm tra plan trên clone, tổng hợp và chọn replay.

| Việc | Reuse trong repo |
|---|---|
| Load CNN | `ActorCriticNet`, `smart_load_weights`, `obs_to_tensor` trong [trainer](/D:/fpt/subject/BomberMan/train_actor_critic.py) |
| Action mask | `get_action_mask` hiện tại |
| Teacher | `get_two_bomb_plan` trong [tactical_teacher.py](/D:/fpt/subject/BomberMan/tactical_teacher.py) |
| RuleBot đối thủ | `make_rule_based_fn`/`RuleBasedAgent` |
| Clone state | `deepcopy(env)`, gồm env.rng; không clone chỉ grid |
| Frame xem lại | `capture_state` và `BoardView` trong [visualize.py](/D:/fpt/subject/BomberMan/visualize.py) |
| Ví dụ replay có parity | [replay_timeouts_17000.py](/D:/fpt/subject/BomberMan/runs/escape-C-6600/replay_timeouts_17000.py) |

File replay17000 chỉ dành cho sáu case cũ, không chạy nó rồi đánh dấu T2 mới hoàn tất. Runner xuất player offline `replays.html`; GUI chơi game vẫn dùng riêng như hiện tại.

### 2.1. Log trước và sau action, sample đúng một lần

Load model trước khi seed mỗi game, `net.eval()` và no_grad. Dùng đúng cách tạo Categorical qua mask của SnapshotOpponent, không chuyển argmax. Model chạy mới trên audit set là replay **fixed policy**, không tái tạo chính xác policy thay đổi suốt200 episode train trước.

```text
cho mỗi model, seed bàn, aid:
    seed NumPy/Torch giống evaluator; reset env; tạo bot cho game
    cho mỗi decision tick t:
        lấy obs và mask trước action
        forward CNN, tạo distribution qua mask, sample CNN một lần
        gọi teacher chỉ để lấy nhãn/plan/stats; không thay action đã sample
        lấy action đối thủ theo thứ tự evaluator
        log before + logits/probabilities + action + plan
        env.step(actions thật)
        log after + bombs_placed + kills/self_blast/alive/done
        nếu game done: kết thúc
```

Với hai agents, evaluator chọn action theo thứ tự A0 rồi A1; giữ thứ tự đó và RNG trong parity test. Không gọi `policy.select_action()` thêm lần nữa chỉ để lấy probability: lần sample thêm sẽ đổi trận.

### 2.2. Những trường bắt buộc

| Nhóm | Trường cần có |
|---|---|
| Danh tính | model/hash, board seed, aid, game_id, decision_tick, state_id, plan_id |
| Trước step | vị trí/capacity/free slots/range của agents, bombs/flames/powerups, grid hoặc frame/reference tới action tape |
| CNN | raw logits, masked probabilities, mask, action đã sample, số action hợp lệ |
| Teacher | first_actions, số action nhãn, representative path, schedule bom, pressure, trapped, before/after reachable counts, horizon, plans, rejection counters, ms |
| Sau step | action intents cả hai agents, vị trí mới, placement thành công, bom mới, kills/self_blast, alive/done, frame |
| Tracker | plan đang theo, hành động mong đợi, prefix mismatch, first/second-bomb status, terminal/censor reason |

`decision_tick = env.step_count` trước hành động. Delay0 của plan là **action đang chuẩn bị thực hiện**; đặt ở delay d tương ứng decision_tick `t+d`, sau step thành `env.step_count=t+d+1`. Lưu tick đặt bom tuyệt đối để tránh lệch một bước.

`state_id` là hash state trước action; `plan_id` duy nhất cho một plan-start. Giữ cả metadata source/checkpoint hashes để trace còn đọc được sau khi sửa method.

### 2.3. Hai giới hạn dữ liệu của teacher hiện tại

**Nhãn nhiều action nhưng chỉ một path:** teacher giữ `first_actions` của các best plan tương đương, nhưng `path`/`bombs` chỉ thuộc một đại diện. Agreement dùng toàn bộ first_actions là đúng; completion theo đại diện khi CNN chọn một action tương đương khác sẽ dễ bị đếm sai.

Runner instrument hàm teacher bằng AST trong process audit, giữ tất cả best paths/schedules tương đương và kiểm tra tập first actions khớp API gốc. Production `tactical_teacher.py` không bị sửa. Khi CNN chọn một nhãn, tracker lấy path tương ứng; canonical order chọn đại diện nếu cùng first action có nhiều path. Test đối chiếu plan/scoring/rejection với teacher gốc và parity action/RNG. Strict-prefix metric vẫn là đối chiếu một path đại diện, không khẳng định mọi path khác đều sai.

**Path chỉ tới bom cuối:** `plan['path']` chưa chứa suffix chạy thoát; planner chỉ kiểm tra suffix có đường sống. Oracle cần tìm/chạy suffix thoát theo forecast hoặc một escape controller đã kiểm chứng. Không cho learner STOP tới lúc nổ rồi gọi đó là lỗi survival của planner.

## 3. Theo một plan từ đầu đến cuối

Chỉ giữ một active plan-start/game để thống kê completion không bị mỗi tick tạo một mẫu trùng. First-action agreement vẫn đo ở mọi state có nhãn.

1. Khi chưa có active plan, teacher có nhãn và CNN chọn một first action hợp lệ của plan, mở tracker. Chọn path tương ứng nếu có; không dùng path của action khác.
2. Ghi lịch bom `(absolute_tick, row, col, owner, range)` và prefix kỳ vọng; bom đã có từ trước là baseline bombs, không tính lại thành bom đầu của combo.
3. Sau mỗi step, kiểm tra intent, vị trí thực và `info['bombs_placed']`. ACTION_BOMB được chọn chưa chắc bom đã đặt thành công; collision có thể làm action di chuyển không đạt đích.
4. Ghi lần đặt bom đầu, bom hai; nếu lệch tick/vị trí thì strict-completion thất bại, nhưng tiếp tục ghi outcome và các thay đổi plan. Không reset tracker ngay để che failure.
5. Kết thúc observation window theo horizon đã đăng ký hoặc terminal; ghi owned kill/active win/death/draw và early-terminal status.

Tách **strict prefix** (đúng action/tick/path) với **placement completion** (đúng lịch/vị trí bom). Có thể đi đường tương đương nhưng vẫn đặt đúng; không chỉ dùng equality của action tape để quyết định mọi loại success.

Owned kill trong horizon là một outcome đi kèm plan, chưa chứng minh kill do chính combo: bom cũ có thể góp phần. Không đặt tên metric là “combo-caused kill” khi chưa có attribution/counterfactual đủ mạnh.

### Phân loại outcome, không làm đẹp denominator

- Success: đặt được những bom đã đăng ký đúng tiêu chí đang đo.
- Failure: action/path lệch, placement không thành, bị mask chặn, hoặc learner chết trước hoàn thành.
- Early terminal: trận kết thúc trước bom hai, tách thắng chủ động sớm, thua, timeout. Thắng trước bom hai không phải lỗi chiến thuật, nhưng cũng không phải bằng chứng đã hoàn thành combo hai bom.
- Censored: trace dừng/hủy hoặc thiếu dữ liệu nên chưa biết kết quả. Learner chết là failure quan sát được, không phải censored.

Báo tất cả số mẫu và denominator: first-bomb success/all eligible starts; second-bomb success/starts hai bom có first placement đúng; thêm strict overall completion/all two-bomb starts. Giữ early-terminal/censored counts riêng. Nếu báo complete-case rate, ghi rằng nó đã bỏ censored; không chỉ đưa một phần trăm thành công cao.

## 4. Kiểm tra future masks và physical plan trên clone

Không ép action oracle vào trận CNN đang audit. Lấy clone của **state trước action** và cô lập RNG của từng branch:

```python
numpy_state = np.random.get_state()
try:
    with torch.random.fork_rng(devices=[]):  # Audit CPU
        branch = deepcopy(env)
        # Chạy các bước oracle / opponent ở branch
finally:
    np.random.set_state(numpy_state)
```

Env.rng được deepcopy riêng. Nếu opponent/controller có state riêng, tạo/copy đúng state của nó cho branch. Không chỉ reset seed rồi làm ảnh hưởng trận chính.

### 4.1. Hai phép thử riêng

| Phép thử | Thực hiện | Kết quả |
|---|---|---|
| Physical prefix | Chạy plan prefix trên clone, cho phép action vật lý ngay cả khi mask chiến thuật chặn; kiểm tra vị trí, capacity, placement, flame và collision | physically feasible / lỗi tại tick cụ thể |
| Masked prefix | Tại mỗi state clone mới, lấy lại mask, kiểm tra action prefix có được phép; dừng ở action đầu bị chặn | feasible through current mask / blocked, tick/action/mask |

Action bị mask chặn không nhất thiết vi phạm physics: mask còn xét lợi ích đặt bom và safety margin. Ghi riêng hai kết quả, không nới mask để làm combo pass.

Sau prefix, kiểm tra survival tới hết lửa bằng suffix thoát thực hoặc timed reachability/forecast đã kiểm chứng. Nếu game kết thúc sớm, không tiếp tục trận chính ngoài API rồi gọi đó là outcome thật; báo full-horizon survival chưa quan sát đủ hoặc kiểm tra riêng bằng mô phỏng physics/forecast trên clone.

`_reachable_cells` chỉ trả tập reachable, không trả action path. Để oracle thật sự chạy suffix, dùng timed BFS `(tick,row,col)` có parent pointers theo cùng burns/blockers: được đứng trên bom hiện tại nhưng không đi vào ô bom chặn, không vào flame ở tick tương ứng. Backtrack từ một state còn sống ở cuối horizon, rồi chạy action sequence đó trên clone và đối chiếu forecast với physics. Không suy action suffix bằng cách chọn tùy ý một ô reachable cuối.

### 4.2. Ba loại đối thủ chẩn đoán

1. **Static:** đứng yên, không bom; kiểm tra cơ chế, chưa chứng minh chiến thuật chống đối thủ biết né.
2. **Evader:** wrapper trong runner dùng mask né và `_bfs_safe_cell`, cấm BOMB; test kiểm tra không đặt bom. Đây là một đối thủ chẩn đoán, chưa đại diện mọi chiến thuật né tối ưu.
3. **RuleBot:** policy hiện tại có thể đặt bom; đo sự khác biệt khi xuất hiện counter-bomb/movement/collision.

Giữ cùng initial state cho các branch, log policy/seed/opponent intents. “Đối thủ cố định” phải ghi rõ là fixed position, fixed action tape hay fixed policy; RuleBot cố định policy vẫn phản ứng khác khi state thay đổi.

Clone checks tốn thời gian: smoke có thể kiểm tra plan mỗi10 ticks, tối đa50 sampled states/game; đăng ký sampling rule trước. Coverage/agreement trên trận chính vẫn lấy mọi bước. Kết luận feasibility chỉ áp dụng corpus sampled, không trình bày như mọi nhãn trong cả train.

Timing teacher trong summary đã gồm observer tied paths; clone có elapsed_ms riêng, ghi trace có trace_write_seconds. Timing này không so trực tiếp với19,31 ms/query của T1 để kết luận planner nhanh/chậm hơn.

### 4.3. Đo plan churn đúng nghĩa

Chuyển schedule cũ/mới sang tick tuyệt đối; bỏ phần bom đã đặt khi so future schedule, và giảm prefix cũ theo số bước đã qua. So mục tiêu/lịch còn lại, không so hai tuple tương đối nguyên xi.

Ghi riêng: đổi trước bom đầu; đổi mục tiêu bom hai sau bom đầu; teacher không còn plan; và change reason quan sát được (enemy move/bomb mới, collision, item/range, wood/flame thay đổi, action CNN lệch prefix). Timer giảm tự nhiên hoặc path đã ngắn đi không được tự tính là churn.

Muốn nói đổi plan “không do state mới”, cần đối chiếu state thực với state tiến triển kỳ vọng trên clone; `state_id` thay mỗi tick chưa chứng minh việc đổi là vô lý. Nếu không xác định được nguyên nhân thì ghi unknown, không suy đoán.

## 5. Metric phải có trong summary

| Metric | Numerator/denominator hoặc dữ liệu gốc |
|---|---|
| Coverage | state nhận nhãn / state learner sống đã query |
| Agreement | action CNN nằm trong first_actions / state có nhãn |
| Target probability mass | tổng probability qua mask dành cho first_actions, mean/distribution |
| Nhãn có thông tin không? | label-support size K, mask-support size M; chance uniform K/M và phân bố BOMB/move/STOP |
| First/second placement | các denominator ở phần3, success/failure/early-terminal/censored |
| Trap | nhãn có trapped>0 / nhãn nhận; tách one/two bomb và pressure distribution |
| Physical/mask feasibility | prefix pass / sampled plan checks; tick/action bị chặn và loại đối thủ |
| Churn | số active-plan transitions đổi future target/schedule / transitions được quan sát; reason/unknown |
| Actual outcome | owned kills trong window, learner alive, active wins, draws; không suy nhân quả |
| Cost | query count, plans/query, mean/p95/max ms; clone-check cost riêng |

Nếu first_actions gần bằng mask-support thì agreement100% có thể gần như không có thông tin; xem cả K/M và probability mass. Combat loss của soft target nhiều actions có minimum khác nhãn đơn, nên không so loss raw như tỷ lệ đã học.

Báo B và final200 trên cùng seeds/corners; báo số frame/plan-start/game, không coi hàng nghìn frame liên tiếp là hàng nghìn trận độc lập. Tổng hợp uncertainty theo board seed, giữ A0/A1 chung; trace exploratory không đủ làm gate promote model.

## 6. Test công cụ trước khi chạy audit lớn

- [ ] Bật/tắt trace cho cùng policy/seed phải giữ action tape, state/frame và game result giống nhau; logits/prob logging không sample thêm.
- [ ] Clone checks không đổi env chính, NumPy/Torch RNG và các state của opponent.
- [ ] First labels nằm trong mask; plans≤64 mỗi query.
- [ ] Delay0/tick tương đối/absolute tick khớp placement thật; test path có STOP/wait/chain release.
- [ ] Tied first actions dùng đúng path tương ứng; không đếm nhãn tương đương thành failure giả.
- [ ] Đúng ammo1/2, gỗ, lửa còn sót, pickup/range, collision và đã doomed từ bom cũ; tận dụng fixture [test_tactical_teacher.py](/D:/fpt/subject/BomberMan/test_tactical_teacher.py).
- [ ] Physical prefix và masked prefix có fixture cho kết quả khác nhau; predictor có đường thoát suffix, oracle không chỉ STOP.
- [ ] Tracker không bỏ learner death/early terminal/censor; không mở plan-start chồng nhau để inflate counts.
- [ ] Best==B được ghi alias; checkpoint/source hashes giữ đúng trước/sau audit.

Đã kiểm tra runner với **74 tests** (12 test audit mới và 62 test cũ), mini12 trận max80 bước xuất10 replay, và4 trận max500 bước với sampling thật. Traced/untraced cả hai góc và evaluator khớp; hash giữ nguyên; JSON/gzip/ZIP đọc lại được. Đây là kiểm tra công cụ, chưa phải kết quả audit đủ100 boards. Chạy lại block test/smoke ở đầu file trên code hiện tại trước full.

## 7. Chọn ít nhất10 replay có ích

Chọn bằng rule cố định từ trace, không chỉ chọn trận xấu nhất hoặc đẹp nhất. Ví dụ lấy các case đầu theo seed/corner cho từng nhóm sau, loại trùng game, bổ sung tới ít nhất10 games và có cả A0/A1:

- Agreement tốt nhưng bom thứ hai không được đặt.
- Prefix vật lý được nhưng mask chặn bước sau.
- Churn trước bom hai không giải thích được bằng event đã log.
- Nhãn có pressure nhưng không complete trap; cuối trận timeout.
- Forecast/plan thất bại khi opponent đặt bom hoặc gây collision.
- Có plan/placement success và owned kill để đối chiếu failure.

Mỗi replay có timeline `tick | vị trí | mask | action CNN | nhãn | plan target | event/reason`, thêm frames quanh bom đầu/bom hai/điểm bị chặn/kết thúc. Cho xem plan overlay và actual path tách rõ.

10 replay dùng giải thích cơ chế, không đủ chứng minh phương pháp tốt/xấu toàn cục. Đừng dùng checkpoint final200 trong một ván GUI ngẫu nhiên để thay parity/trace audit có seed.

## 8. Quyết định từ kết quả, chỉ chọn một nhánh

| Bằng chứng audit | Nhánh kế tiếp | Điều chưa nên làm |
|---|---|---|
| Nhiều nhãn pressure yếu/ít trap dù feasible | L1: thử lọc pressure≥0.5 **hoặc** trap-only | Bật cả hai/tăng reward bom |
| Agreement cao nhưng future mask chặn prefix | L2: loại plan prefix không thực thi được qua mask | Nới mask để ép combo |
| Prefix feasible, CNN bắt đầu theo nhưng teacher đổi mục tiêu liên tục | L3: giữ mục tiêu ngắn, hủy khi nguy hiểm/state invalid | Khóa action CNN theo oracle |
| Nhãn/plan tốt nhưng agreement thấp | Kiểm tra label ambiguity và gradient/loss balance; nhánh T4W có đối chứng | Tăng combat coefficient chỉ dựa trên loss scalar |
| Khả thi đối với static, không thắng evader | T3 curriculum giao tranh, hoặc sửa scoring từ phản ứng thực | Kết luận static kill là chiến thuật tốt |
| Nhãn/behavior đủ tốt nhưng budget bị planner chiếm | L4: tối ưu giữ semantics | Đổi scoring và cache cùng lúc |

Nếu trace chưa đủ denominator/parity thì kết luận INCONCLUSIVE và sửa công cụ trước. Dù thấy một lỗi rõ, giữ bản audit của teacher hiện tại trước khi sửa để có đối chiếu.

## 9. File cần giữ/gửi và điều kiện đóng T2

Thư mục mới đề xuất `runs/t2-audit-<timestamp>/`; không ghi vào lượt T1 cũ.

- `config.json`: checkpoints/hash, source hash, seeds, policies đối thủ, sampling rule, các thresholds/definition.
- `trace.jsonl` hoặc `.jsonl.gz`: các bước before/after, plan/prob/mask/actions; action tape đủ để dựng lại state.
- `plan_checks.jsonl`: physical/mask/oracle outcomes trên clone, policy đối thủ và failure tick.
- `summary.json`: counts/denominators và metrics per model/corner/opponent; giới hạn sampled corpus.
- `replay_cases.json` cùng timeline/frames: ít nhất10 cases đã chọn theo rule.
- `report.md`: nguyên nhân có bằng chứng, chưa biết gì, và một next branch được chọn.

T2 đầy đủ khi trace runner qua parity/RNG tests, metrics có denominator đúng, plan prefix và suffix được kiểm tra, replay minh họa đủ, và có quyết định bằng chứng. Chỉ đọc/plot `metrics.jsonl`200 episode hoặc chạy test cũ chưa đủ đóng T2.

Đợt này không train, không chuyển model, không đổi reward/mask/optimizer. Sau khi chọn L1/L2/L3/L4 mới mở lượt triển khai và pilot riêng từ cùng baseline, có đối chứng và test seeds mới.
