# D-S2 — Phase D từ S2 với đối chứng và opponent pool

Ngày: 22/10/2026. Trạng thái: **APPROVED_DESIGN — người dùng duyệt written spec ngày 22/10/2026; đã cho phép viết implementation plan, chưa yêu cầu triển khai hoặc train**.

## 1. Mục tiêu và phạm vi

Người dùng đã đồng ý bước tiếp theo là viết spec Phase D: warm-start từ S2, 1v1 trên 9×9, học với CNN đông cứng và snapshot, giữ RuleBot làm mỏ neo; chọn checkpoint bằng validation trên CNN và RuleBot, rồi test một lần trên seed mới. Written spec với ngân sách và luật quyết định dưới đây đã được người dùng duyệt; implementation plan tại `docs/superpowers/plans/2026-10-22-phase-d-s2-pool.md` cần được review trước implementation.

Câu hỏi chính: **opponent pool có làm CNN thắng chủ động trước các CNN tốt hơn S2 ban đầu và tốt hơn việc chỉ train thêm với RuleBot không?**

Phạm vi chỉ gồm một pilot D-S2. Không thay engine, RuleBot, reward, observation, safety mask, kiến trúc mạng hoặc thuật toán tối ưu. Phase E là lượt riêng sau khi đọc kết quả D. Không dùng search trong rollout train hoặc inference của pilot.

Các lựa chọn đã cân nhắc:

| Cách | Lợi ích | Giới hạn |
|---|---|---|
| Một nhánh pool so với S2 ban đầu | Ít chi phí train nhất | Không tách được lợi ích của pool khỏi train thêm |
| **Hai nhánh: RuleBot đối chứng và pool; cùng từ S2** | Tách được tác dụng của đối thủ; tái dùng trainer hiện tại | 4.000 episode tổng cộng; được chọn cho pilot này |
| Train đồng thời D và FFA E | Có kết quả FFA sớm | Mở thêm bài toán chuyển kích thước/số agent; để lượt sau |

## 2. Bằng chứng và giới hạn đầu vào

- PDE: `runs/pde-50000-resumable/report.json`, 3.360 bản ghi, 200 shard. E2 G / G+search3 ActiveWR = 22,5 / 82,5%; E1 = 32,22 / 91,67%. Trong D, G+search3 đấu G/F/S/B chỉ đạt WR 17,5–47,5%; luật đề xuất train D kích hoạt.
- PDE2: `runs/pde2-51000/report.json`, 5.600 bản ghi, 1.100 shard. FFA 9×9: G+search3 / S2 ActiveWR = 9 / 5%, CI95 của hiệu = [−1; 9] điểm phần trăm; chưa chứng minh tương đương. Hòa FFA 9×9 / 11×11 = 73,5 / 88%.
- S2 vs G: WR 31 / 8,5%, ActiveWR 7,5 / 4%, hòa 60,5%. WR cao hơn chưa đủ chứng minh giao tranh chủ động tốt hơn.
- Mirror S2 11×11 chỉ có 20/400 trận phân thắng bại. Kết quả lệch góc kích hoạt luật PDE2 nhưng còn ít mẫu.
- Seed nền PDE E1 và PDE2 FFA_9 trùng 51000–51049; PDE D và PDE2 FFA_11 trùng 52000–52049. Không dùng lại các dải này cho pilot mới.
- Đã tính lại bảng tổng hợp và 37 so sánh bootstrap từ raw records; khớp báo cáo. Các hash đã đăng ký và các shard khớp tại lúc review. Không dùng các con số cũ làm baseline số học của gate D-S2: mọi đối chứng được đo lại trong runtime của lượt mới.

Nguồn thiết kế: `2026-10-20-pde-check-design.md`, `2026-10-20-pde2-ffa-mix-design.md`, `PROJECT_SPEC.md` phần D/E.

## 3. Checkpoint và vai trò

| ID | Đường dẫn | Vai trò trong lượt mới |
|---|---|---|
| S2 | `runs/sd8a-20261020/train/epoch_16.pt` | Warm-start cả hai nhánh, baseline, một đối thủ CNN cố định |
| G | `runs/sd75-20261018/full/best_phaseC.pt` | Đối thủ cố định ở train/validation/test |
| F | `runs/phaseC-sd-best-fallback/best_phaseC.pt` | Đối thủ cố định ở train/validation/test |
| S | `runs/sd53-20261018/train/epoch_8.pt` | Đối thủ cố định ở train/validation/test |
| B | `runs/pressure-C-next/final_phaseC.pt` | Chỉ mở ở test; không train hoặc chọn checkpoint với B trong D-S2 |
| RB | RuleBot của source đã đăng ký | Mỏ neo train và đánh giá giữ năng lực cũ |

SHA-256 đã đọc trực tiếp ngày 22/10/2026:

```text
S2 d222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814
G  bb0c775c58161aa1bbdd229b9528170aab174d5aabf8f4a6739b848919cdc7ac
F  0315a56f09079333d936db5f55587093f1b6415ed6355d2ef6ae2ab63f8b1483
S  8f7116d290593fa10de1753c4f0f96366e220a193e9860123ed72e0db46bfd4a
B  93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60
```

S2 phải tiếp tục khớp `selected.json` và `test_opened.json.models.S2` của SD8A. G/F/S/S2 có 13 input channels; B có 12. Chỉ B được phép mở rộng kênh 13 bằng trọng số 0 theo loader hiện tại; phải kiểm tra logits trước/sau mở rộng. Mọi tensor khác phải đúng shape; không nội suy kích thước bàn. B chỉ là đối thủ ngoài lịch tối ưu/chọn model của lượt mới, không phải bằng chứng chưa từng chịu ảnh hưởng từ lịch sử phát triển dự án.

## 4. Hai nhánh và cấu hình chung

- `control`: 2.000 episode, đối thủ luôn là RB.
- `pool`: 2.000 episode, lịch đối thủ tại mục 5.
- Cả hai dùng cùng training loop Phase D; khác duy nhất lịch đối thủ. Mỗi nhánh khởi tạo lại model từ đúng S2 và Adam/scheduler mới; không kế thừa optimizer của lượt distillation.
- Train tuần tự hai nhánh trên cùng một GPU; baseline, validation và test chạy CPU trên cùng máy và cùng môi trường phần mềm. Nếu chuyển máy trước khi bắt đầu, đăng ký lại runtime trên máy đó. Không gộp một nhánh được chạy bằng torch khác vào lượt đã đăng ký.

| Tham số | Giá trị cố định cho cả hai nhánh |
|---|---|
| Bàn / số agent / giới hạn | 9×9 / 2 / 500 bước |
| Lịch sudden death | E: start 475, interval 25, warning 40; xác suất bật 1,0 |
| Policy | Lấy mẫu qua safety mask chung; đối thủ CNN ở eval mode, không gradient |
| Objective / advantage | Actor-Critic hiện tại, MC, `normalize_advantage=False` |
| Adam / gamma / value coefficient | lr 2,5e-5 / 0,99 / 0,5 |
| LR schedule | CosineAnnealingLR 2.000 episode, eta_min = lr × 0,05 |
| Entropy | 0,0075; lịch giảm hiện có của trainer |
| Teacher | `attack_teacher=local`, `teacher_version=v2` |
| Teacher attack | lookahead 2; min_pressure_gain 0,15; trap_sequence_depth 2; sequence_pressure_gain 0,35 |
| Auxiliary coefficients | escape 0,5; combat 0,9; navigation 0,4; ring_safety 0,6 |
| Cuối trận | late_game_start 350; late_game_attack_multiplier 1,75 |
| Wall bonus | 0,15; mọi reward khác giữ theo source đã đăng ký |
| Tính năng tắt | search_teacher, group_rl, PPO, anchor KL; anchor_coef = 0 |
| Checkpoint ứng viên | episode 0, 500, 1.000, 1.500, 2.000 |
| Snapshot cho pool | Mỗi 200 episode, chỉ sau khi episode đã hoàn thành |
| Validation nội bộ cũ | Tắt; runner thực hiện validation riêng theo mục 7 |
| Seed RNG train | 2026102201 cho cả hai nhánh; scheduler đối thủ có RNG/state riêng |

Đây là lựa chọn thiết kế của pilot, chưa phải hyperparameter đã chứng minh tối ưu. Các teacher heuristic hiện có được giữ cố định; nếu chúng không phù hợp với CNN thì cần chẩn đoán và một spec khác, không sửa giữa lượt.

## 5. Lịch đối thủ và góc của train

Mỗi block 20 episode gồm 10 cặp. Mỗi cặp chơi cùng một bàn, cùng đối thủ đông cứng: episode đầu learner ở góc 0, episode sau ở góc 1. Với chỉ số cặp j bắt đầu từ 0, seed bàn = 610000 + j. Policy tiếp tục được cập nhật sau từng episode; đây là cân bằng lịch train, không phải cặp độc lập dùng tính CI.

`control` dùng RB trong cả 10 cặp. `pool` dùng các slot sau:

```text
RB, G, F, S, S2, RB, recent, recent, recent, recent
```

Tổng mỗi nhánh pool: **400 episode RB (20%), 800 episode CNN cố định (40%; 200 cho mỗi G/F/S/S2), 800 episode recent (40%)**.

`recent` lấy round-robin trong hai snapshot gần nhất của chính nhánh pool đã được commit vào training state trước khi block bắt đầu. Chưa có snapshot thì dùng S2; mới có một thì dùng snapshot đó. Cursor recent phải được lưu khi resume. Snapshot phát sinh trong một block chỉ có hiệu lực từ block tiếp theo. Cố định đối thủ cho đủ hai episode của từng cặp; không chọn lại theo kết quả episode đầu.

Mỗi episode log `board_seed`, learner corner, loại đối thủ, SHA đối thủ, block/slot, learner steps và optimizer steps. Các trận dùng S2 thay cho recent ban đầu phải được ghi rõ; không báo chúng là snapshot mới. Snapshot thuộc control không được đưa vào pool hoặc chọn model của nhánh pool. Chỉ model learner được cập nhật.

## 6. Seed, đăng ký và provenance

Các dải dưới đây đã được duyệt ở mức thiết kế; **chưa được đặt chỗ hoặc mở**, và vẫn phải qua kiểm tra lịch sử ở preflight:

| Mục đích | Dải seed bàn | Số bàn / cách dùng |
|---|---|---|
| Train | 610000–610999 | 1.000 bàn × 2 góc cho mỗi nhánh |
| Validation | 620000–620039 | 40 bàn × 2 góc / model / đối thủ |
| Test reference pool + B + RB | 630000–630099 | 100 bàn × 2 góc / model / đối thủ |
| Test snapshot cũ | 640000–640099 | 100 bàn × 2 góc / snapshot |
| Smoke | 690000–690003 | Lượt riêng, không mở dải full |

13 file `registration.json` hiện có đã được rà, chưa thấy seed tường minh nằm trong 610000–640099. Đây không phải chứng nhận toàn bộ lịch sử đã sạch: preflight bắt buộc dựng inventory các dải train/collect/validation/test từ metadata, raw board IDs và registry sẵn có; từ chối nếu phát hiện giao với dải đề xuất, kể cả ở bối cảnh khác. Metadata cũ không đủ để xác minh phải được ghi rõ và giải quyết trước registration. Không chỉ kiểm tra seed gốc như PDE/PDE2.

Registry mới phải đặt chỗ atomically toàn bộ các khoảng đã duyệt; cho phép dùng chung các bàn trong chính lượt này để so cặp giữa model/đối thủ, không cho một experiment khác dùng lại. Registry không thay nội dung PDE/PDE2.

`registration.json` khóa config, spec hash, Git HEAD + trạng thái tracked/untracked có liên quan, hash đầy đủ source thực sự được import, hash checkpoint, seed manifest và machine/runtime. Dependency closure phải bao gồm engine, RB/mask, trainer, training_state, evaluator, teacher, progress và runner/helper mới; không bỏ qua file chỉ vì đang untracked. Chụp source vào output trước train; worker chỉ dùng bản đã đóng băng. Lưu Python/torch/numpy/CUDA/cuDNN, GPU, device, thread counts, cờ deterministic/TF32 và OS. Máy hiện tại được đọc là Python 3.11.9, torch 2.11.0+cu128, numpy 2.4.6; đây chỉ là runtime quan sát khi soạn spec, registration lúc chạy mới là nguồn chính thức.

Từ chối resume nếu source/config/checkpoint/runtime không khớp registration. Không viết lại hash để hợp thức hóa thay đổi. Lỗi runner sau registration cần một output/protocol version mới và quyết định riêng về phần dữ liệu đã mở.

## 7. Validation và chọn checkpoint

Chỉ thực hiện khi cả hai nhánh đã hoàn thành đủ 2.000 episode. Không dùng validation để đổi ngân sách, LR hoặc lịch pool giữa lượt.

Đối thủ validation: **G, F, S, S2, RB**; 80 trận / model / đối thủ, đủ hai góc của từng bàn. Các ứng viên: S2 episode 0 dùng chung, cộng episode 500/1.000/1.500/2.000 của từng nhánh, tối đa 9 model khác SHA. Baseline S2 được đo lại cùng runtime; kết quả PDE/SD8A không được thay vào.

`pool_ActiveWR`, `pool_WR`, `pool_draw`, `pool_suicide` là trung bình đều của bốn đối thủ CNN G/F/S/S2, tách RB. Chọn riêng cho từng nhánh:

1. Chỉ nhận ứng viên có RB ActiveWR >= S2 − 2pp, RB suicide <= S2 + 1pp và pool suicide <= S2 + 1pp trên validation.
2. Trong các ứng viên đạt guard, tối đa hóa tuple: `(pool_ActiveWR, -pool_draw, pool_WR, -pool_suicide, -episode)`. So sánh metric với dung sai 1e-12; hòa lấy episode sớm hơn.
3. Episode 0 luôn là ứng viên fallback hợp lệ. Nếu nó thắng thì ghi giữ S2; không quảng bá file final thành best.
4. `selected.json` phải khóa episode, SHA, bảng metric, guard, thứ tự tie-break và manifest test. Không sửa selection sau khi mở test.

Luật mở test: checkpoint pool được chọn có episode > 0, validation pool ActiveWR hơn S2 ít nhất 3pp và không thấp hơn control được chọn; mọi guard ở trên đạt. Nếu không đạt, kết thúc `VALIDATION_STOP`, đóng gói kết quả validation; không mở test và không tự train thêm.

## 8. Held-out test và budget đánh giá

Test ba learner: **S2**, **control được chọn**, **pool được chọn**. Mỗi learner đấu G/F/S/S2/B/RB, 200 trận / cặp, cùng 100 bàn và hai góc. B chỉ được mở ở bước này. SHA learner trùng nhau được đánh giá một lần và giữ alias trong report. S2-vs-S2 phải phân biệt vai learner/opponent, không gộp nhầm hai ghế.

Thêm pool được chọn đấu **mọi snapshot của chính nhánh pool có episode nhỏ hơn episode được chọn**; không chọn tối đa vài bản thuận lợi. Mỗi snapshot 200 trận trên dải 640000. Không đấu với chính checkpoint được chọn hoặc snapshot của tương lai để tính gate snapshot cũ. Manifest snapshot theo quy tắc này phải khóa trước test; không có snapshot cũ hợp lệ thì chưa thể đạt gate Phase D.

| Ngân sách tối đa | Số trận |
|---|---:|
| Train control + pool | 4.000 episode, tối đa 2.000.000 learner steps |
| Validation: 9 model × 5 đối thủ × 80 | 3.600 |
| Test reference: 3 learner × 6 đối thủ × 200 | 3.600 |
| Test snapshot cũ: tối đa 9 × 200 | 1.800 |
| Tổng validation + test | **9.000** |

Số thực tế có thể thấp hơn vì trùng SHA, chọn checkpoint sớm hoặc dừng tại validation. Giới hạn learner steps tính theo 500 bước/episode; không cộng validation vào train. Chưa có đo tốc độ end-to-end của runner mới nên chưa cam kết thời gian chạy. Preflight đo ở smoke, dự báo riêng train/validation/test và RAM rồi ghi vào registration; không mở seed full để đo thử. Evaluation mặc định 2 worker CPU, cho phép chốt 1–5 trước registration; train chỉ một process GPU.

Đây là **một lần mở held-out**, có thể tiếp tục cùng manifest qua `--resume`. Không xóa `test_opened.json` để chạy lại hoặc mở thêm test vì kết quả xấu.

## 9. Metric, CI và quyết định

Mỗi matchup lưu WR, ActiveWR, draw, passive win, sudden-death win, suicide, sudden-death death, kills/game, bombs/game, steps, theo góc và thời gian. WR = số thắng / toàn bộ trận, có hòa trong mẫu số. ActiveWR = thắng và có ít nhất một kill bằng bom của learner. `passive_win` hiện tại không gồm thắng nhờ vòng; báo riêng để tránh diễn giải sai.

So cặp giữ đủ hai góc của một bàn. Bootstrap 10.000 lần, seed 20261014. Với macro CNN, trước hết lấy trung bình hai góc và bốn đối thủ trên từng bàn rồi bootstrap theo bàn; không coi mọi record/đối thủ là mẫu độc lập. Các CI là theo những bàn test của một training seed, không đại diện biến thiên giữa nhiều lần train. B và RB có CI riêng, không trộn vào macro CNN.

**Gate pilot: `PILOT_PASS` chỉ khi tất cả điều kiện sau đạt trên test:**

1. Pool − S2 macro CNN ActiveWR >= +5pp và CI95 dưới > 0.
2. Pool − control macro CNN ActiveWR có CI95 dưới > 0, để có bằng chứng lợi ích riêng của lịch pool.
3. Pool − S2 macro CNN draw <= 0pp và suicide <= +1pp.
4. RB: pool − S2 ActiveWR >= −2pp và suicide <= +1pp.

Các guard dạng pp ở điều kiện 3–4 dùng point estimate, không gọi chúng là kiểm định tương đương hay không kém hơn. B là chẩn đoán đối thủ ngoài lịch train/selection, báo riêng và không dùng để chọn lại checkpoint. Nếu bất kỳ điều kiện pilot không đạt: `PILOT_NOT_PASS`, kèm từng điều kiện trượt; CI chứa 0 được diễn giải là chưa đủ bằng chứng, không chứng minh ngang nhau. Giữ S2 làm checkpoint mặc định.

**Gate hoàn thành Phase D: `PHASE_D_PASS` cần thêm:**

- `PILOT_PASS`;
- WR >= 60% trước từng đối thủ CNN trong pool train cố định và trước từng snapshot cũ hợp lệ, mỗi cặp đo đủ 200 trận;
- WR trước RB không thấp hơn S2 đo cùng lượt;
- Có ít nhất một snapshot cũ hợp lệ và không thiếu matchup nào.

WR của Phase D dùng mẫu số toàn bộ trận, không đổi sang tỉ lệ thắng chỉ trong trận phân thắng bại. Gate Phase D là gate điểm theo PROJECT_SPEC; luôn xuất CI để thể hiện độ chắc chắn. `PILOT_PASS` nhưng chưa `PHASE_D_PASS` chỉ chứng minh cải thiện trong pilot, chưa cho phép đánh dấu xong D hoặc tự chuyển E. Nếu control tốt hơn thì ghi nhận tác dụng của train thêm, không kết luận pool thành công.

## 10. Runner, artifact và resume phải có trước lượt thật

Giao diện dự kiến: `run_phase_d_s2.py` có các stage `register`, `train --arm control|pool`, `validate`, `test`, `package`, và `--resume` ở stage có công việc từng phần. Đây là contract của spec, **runner chưa tồn tại, chưa có lệnh chạy thật**.

Output dự kiến: `runs/phaseD-s2-20261022/`, ZIP `runs/phaseD-s2-20261022-send.zip`; không ghi đè output có sẵn, checkpoint S2 hay bất kỳ run cũ. Cấu trúc tối thiểu:

```text
registration.json, seed_manifest.json, sources/, runtime.json
arms/control/ và arms/pool/: checkpoints/, snapshots/, training_state.pt,
                             metrics.jsonl, opponent_manifest.json, completed.json
validation/tasks/, validation.json, selected.json
test_opened.json, test/tasks/, report.json, decision_report.md
```

- Train resume từ cuối episode đã commit: model, optimizer, scheduler, Python/numpy/torch/CUDA/env RNG, learner corner, board index, histories, lịch pool/cursor và manifest/hash snapshot. State phải có định dạng Phase D riêng; không đổi tên inference weights thành training state. Khôi phục cùng runtime phải được kiểm tra với một lượt liên tục; CUDA chỉ được tuyên bố tương đương nếu đã kiểm trên CUDA thật.
- Checkpoint/state/shard được ghi file tạm + flush/fsync + atomic replace. State commit là nguồn xác định episode/snapshot nào đã hoàn thành; snapshot/log còn dang dở không được đưa vào pool khi resume. Chỉ tái thực hiện phần chưa commit, không nhân đôi metric. Không giữ hai writer cùng output.
- Validation/test có ledger atomically theo matchup-board; kiểm schema, SHA, source/config/runtime, seed, góc, đúng số record, không trùng/thiếu và quan hệ win/draw trước khi reuse. `test_opened.json` khóa selection và manifest trước trận đầu. Lượt resume đọc lại shard hợp lệ, không mở test lần hai. Chỉ xuất summary cuối khi mọi task trong manifest đủ.
- Bắt buộc progress qua `progress.py`, thể hiện cả stage và số việc đã xong; vẫn hiện khi redirect stdout vào log. Bản bàn giao sau implementation phải có lệnh CMD cho register/train/resume/validate/test/package, chọn đúng Python có torch trên máy chạy. Chỉ chạy dài khi người dùng yêu cầu; không tự theo dõi sau launch nếu họ chỉ yêu cầu khởi chạy.
- Package có status `VALIDATION_STOP`, `PILOT_NOT_PASS`, `PILOT_PASS`/`PHASE_D_PASS` hoặc `INCOMPLETE`; kèm registration, source, logs, state/checkpoint liên quan và raw ledger. Không ghi thành PASS nếu thiếu dữ liệu.

## 11. Khoảng thiếu của code hiện tại và kiểm chứng cần có

Đã đọc trực tiếp: `train_actor_critic.train()` chặn full-state resume ngoài Phase C; Phase D đang chọn khoảng 50% RB và 50% snapshot ngẫu nhiên; validation mặc định gọi RuleBot; `best_phaseD.pt` hiện phụ thuộc selection đó. Bởi vậy không dùng launcher Phase D v1 để chạy spec này và không chỉ bỏ guard resume rồi coi là hỗ trợ.

Implementation plan sau khi spec được duyệt phải xác định phần mở rộng tối thiểu cho lịch đối thủ, seed bàn, snapshot và state Phase D; tái dùng training math hiện có, không sao chép toàn bộ training loop. Runner sở hữu validation/selection/test mới, không thay các báo cáo PDE/PDE2 hoặc giao thức đã đăng ký.

Yêu cầu acceptance để chuyển sang chạy thật:

1. Lịch đủ 2.000 episode/nhánh, đúng tỷ lệ, hai góc/cặp, seed và fallback recent; CNN đông cứng không nhận gradient/cập nhật.
2. Hai nhánh bắt đầu cùng SHA S2, optimizer mới; giữ nguyên config/engine/reward/mask. Baseline/B load đúng shape, zero-pad B giữ logits.
3. Chọn model đúng macro CNN, guard và tie-break; episode 0 thắng thì giữ S2. RB tốt nhưng CNN tệ không được chọn nhầm; validation fail không mở test.
4. Bootstrap theo bàn giữ đủ đối thủ/góc; kiểm test tổng hợp có delta/CI biết trước. Các nhãn pilot/Phase D đúng các biên ngưỡng; không lẫn conditional WR hay passive/sudden-death wins.
5. Gián đoạn train và đánh giá rồi resume không lặp metric, không mất optimizer/RNG/pool; từ chối nguồn/hash/runtime sai, thiếu snapshot, shard sai/trùng, writer thứ hai hoặc mở lại test. Kiểm riêng cửa sổ gián đoạn trước/sau atomic commit.
6. Smoke end-to-end riêng không mở seed full; progress và CMD redirect hoạt động. Hồi quy Phase C resume, SD8A và evaluator PDE/PDE2 vẫn giữ contract. Không chạy lại các seed held-out cũ để kiểm hồi quy.

Đây là yêu cầu hành vi, chưa phải test hoặc code đã được viết/chạy. Sau khi duyệt spec mới viết implementation plan và dùng workflow acceptance tests độc lập trước production edits; không giả định các khả năng trên đã có.

## 12. Trạng thái và bước tiếp theo

- [x] Đọc hai báo cáo và tính lại summary/CI.
- [x] Kiểm SHA/input channels, code train/selection/resume và đề xuất seed tách dải.
- [x] Soạn spec để review.
- [x] Người dùng duyệt written spec, gồm ngân sách hai nhánh và đánh giá tối đa 9.000 trận (22/10/2026).
- [x] Viết implementation plan theo spec được duyệt; plan đang PLAN_REVIEW, chưa phải code đã được thực hiện.
- [ ] Implement/kiểm thử runner và resume; bàn giao lệnh CMD cụ thể.
- [ ] Người dùng yêu cầu chạy train/validation/test trên máy đã đăng ký.
- [ ] Đọc gate; quyết định tiếp tục D hay thiết kế E 3 agent → 4 agent.

Không có seed full nào được mở, không tạo output experiment hoặc chạy train/test trong lượt soạn spec này.
