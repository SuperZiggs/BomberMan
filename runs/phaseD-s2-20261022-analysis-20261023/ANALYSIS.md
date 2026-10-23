# Phân tích D-S2 sau VALIDATION_STOP — 23/10/2026

Kết luận: dữ liệu cho thấy S2 mất một phần kỹ năng quyết định đã chưng cất khi được fine-tune bằng công thức RL hiện tại. Hiện tượng có ở cả control 100% RuleBot và pool, trong khi chưa có bằng chứng pool cải thiện khả năng thắng chủ động trước CNN. Cơ chế gây mất kỹ năng chưa được tách bằng thí nghiệm đối chứng; không coi giả thuyết về critic/teacher là nguyên nhân đã chứng minh.

Giữ checkpoint S2. Lượt D-S2 đã kết thúc VALIDATION_STOP; không thay tiêu chí, chọn checkpoint ngoài luật, mở test hoặc train thêm trong cùng protocol.

## 1. Phạm vi và kiểm chứng dữ liệu

- Nguồn: `runs/phaseD-s2-20261022`, digest `474530836ca88ee9be260f8ca3af7f823e2fe3ec9ecba7f2f7a5c086d4de1951`.
- `verify_run` xác nhận source/spec/runtime/checkpoint; training state và completion SHA của hai arm khớp, mỗi arm2000episode. Metrics JSONL bằng context trong state, không thiếu/lặp episode; Adam step và cosine scheduler đều2000.
- Tái dựng lịch toàn bộ2000episode từ snapshots cuối, có kiểm điều kiện snapshot phải tồn tại trước block; khớp từng trường audit, board/corner/cursor/opponent/hash/fallback. Pool400RB/800fixed/800recent, có80episode recent dùng S2 lúc chưa có snapshot. Hai episode đầu của control/pool giống nhau.
-13 entry audit đối thủ CNN đều eval, không gradient, before/after hash bằng nhau. Dòng console legacy `pool: 0` phản ánh biến `snapshot_pool` cũ không được dùng trong context D-S2; không phải pool rỗng.
- Validation1800task/3600trận,40board ×2góc ×5đối thủ ×9SHA learner khác nhau. Raw-ledger schema, task identity, selection và report khớp; test chưa mở.
- Không chạy trận mới, search, backward/optimizer step hoặc ghi checkpoint. Offline chỉ forward CPU trên quan sát cũ thuộc TRAIN partition của SD8A; không đánh giá partition held-out của SD8A.

## 2. Kết quả validation và độ bất định

Mọi số dưới đây là validation. CNN là macro đều G/F/S/S2. Mỗi đối thủ80trận; các góc dùng cùng40board. CI95 dùng paired bootstrap10000lần, gộp hai góc và các CNN trong mỗi board trước khi resample, seed20261014.

| Model | CNN ActiveWR | RB ActiveWR | RB WR | RB hòa | RB tự nổ |
|---|---:|---:|---:|---:|---:|
| S2 | 6,25% | 83,75% | 93,75% | 5,00% | 0,00% |
| control-1000 | 9,69% | 76,25% | 82,50% | 13,75% | 0,00% |
| control-2000 | 6,25% | 65,00% | 77,50% | 16,25% | 0,00% |
| pool-2000 | 7,19% | 57,50% | 81,25% | 15,00% | 0,00% |

| So sánh ActiveWR | Delta pp | CI95 pp |
|---|---:|---:|
| control-2000 − S2, RB | −18,75 | [−31,25; −6,25] |
| pool-2000 − S2, RB | −26,25 | [−38,75; −13,75] |
| pool-2000 − S2, CNN | +0,94 | [−3,13; +4,69] |
| pool-2000 − control-2000, CNN | +0,94 | [−2,81; +4,69] |
| pool-2000 − control-2000, RB | −7,50 | [−20,00; +5,00] |
| control-1000 − S2, CNN | +3,44 | [−0,63; +7,81] |

Suy giảm RB của hai endpoint có CI dưới0 trong phân tích này; lợi ích CNN chưa rõ. Riêng chênh lệch RB giữa pool/control chưa đủ chắc để quy mức giảm thêm cho lịch pool. control-1000 là checkpoint CNN tốt nhất sau khi xem dữ liệu, nhưng không đạt guard RB; không được đổi luật để chọn nó. CI ở đây là hậu kiểm, chưa hiệu chỉnh việc so nhiều checkpoint/metric và không thay thế gate held-out.

Theo từng CNN, ActiveWR S2→pool-2000: G7,50→7,50%; F6,25→5,00%; S6,25→6,25%; S2-self5,00→10,00%. Mức tăng macro nhỏ chủ yếu đến từ S2-self, chưa phải cải thiện trải đều trước pool.

Suy giảm RB xuất hiện ở cả hai góc: S2 là85,0/82,5%; control-2000 là65,0/65,0%; pool-2000 là52,5/62,5%. Tự nổ trước RB vẫn0%; không có bằng chứng tỷ lệ tự nổ tăng là nguyên nhân chính. Số trận hòa và thắng không chủ động tăng, khả năng kết liễu giảm.

## 3. Đo trực tiếp khả năng giữ kỹ năng search

Lấy ngẫu nhiên1024decision states và1024anchor states từ collection cũ của SD8A, seed lấy mẫu20261023; loại toàn bộ board thuộc held-out split gốc (seed530,10%). Định nghĩa nhãn giống SD8A: tập những hành động hợp lệ có search-score cao nhất; nhận mọi hành động đồng hạng. Model chạy với mask đã lưu trên cùng quan sát, không dựng lại environment.

| Model | Argmax thuộc tập nước tốt của search | Xác suất dành cho tập nước tốt | KL(S2 || model) trên anchor |
|---|---:|---:|---:|
| S2 | 97,36% | 91,83% | 0,0000 |
| control-500 | 75,59% | 69,21% | 0,1546 |
| control-1000 | 77,15% | 71,24% | 0,0835 |
| control-2000 | 75,59% | 68,66% | 0,1389 |
| pool-500 | 76,66% | 71,81% | 0,0716 |
| pool-2000 | 73,14% | 66,92% | 0,1082 |

Đây là bằng chứng kỹ năng trên các trạng thái cũ giảm mạnh ở cả hai nhánh, đã thấy tại checkpoint500, trước khi kết thúc2000episode. Chênh lệch control-2000/pool-2000 so S2 lần lượt −21,78/−24,22pp về argmax. Các trạng thái bình thường thay đổi ít hơn: anchor argmax đổi8,30% ở control-2000 và7,23% ở pool-2000. Chỉ xem KL trung bình ở trạng thái bình thường có thể bỏ sót suy giảm tại quyết định chiến thuật quan trọng.

Đây là TRAIN data được dùng lại để chẩn đoán retention; không phải độ chính xác tổng quát hoặc WR độc lập. Tỷ lệ97,36% ban đầu có thể cao vì S2 từng học trên dữ liệu này. Không thể dùng bảng này để chứng minh một model mới sẽ thắng trong game.

## 4. Vì sao công thức hiện tại đáng nghi

**Bằng chứng chắc:** S2 được train bằng soft-target CE của G+search2 và KL neo G trong SD8A. D-S2 chỉ lấy weights S2 để warm-start rồi tối ưu actor-critic MC, các loss teacher local v2 và safety; `search_teacher=false`, `group_rl=false`, `anchor_coef=0`. Vì vậy không còn ràng buộc giữ phân phối hành động của S2 hoặc các quyết định chưng cất cũ. Đối thủ S2 trong pool chỉ tạo hành động phía bên kia; nó không cung cấp nhãn search cho learner.

**Giả thuyết được hỗ trợ mạnh:** đổi mục tiêu fine-tune trong khi không neo giữ policy cho phép ghi đè kỹ năng search. Việc cả control100%RB lẫn pool cùng giảm retention làm giả thuyết này hợp lý hơn lời giải thích chỉ dựa vào20%RB. Tuy nhiên chưa có ablation chỉ thay anchor nên không tuyên bố anchor=0 là nguyên nhân duy nhất, hoặc thêm anchor chắc chắn giải quyết.

**Rủi ro critic cần tách riêng:** optimizer SD8A loại `critic_head.*`, trong khi shared conv/trunk vẫn được cập nhật bởi distillation. Kiểm tensor thực tế xác nhận critic head S2 bằng hệt G. Khi D-S2 bắt đầu, Adam tối ưu cả actor/critic/shared trunk; value loss có thể tác động tới đặc trưng actor dùng chung. Mean value MSE trong control là23,28 và pool11,05, nhưng độ lớn loss không chứng minh độ lớn/hướng gradient. Chưa đo critic calibration, explained variance hay gradient từng thành phần, nên critic chưa phải nguyên nhân đã xác nhận.

**Không phải cứ kéo dài train là tốt:** control-1000 có CNN ActiveWR9,69% rồi endpoint trở về6,25%; đồng thời RB tiếp tục giảm. Retention đã giảm ở500episode. Chưa có dữ liệu để kết luận checkpoint trước500 đạt guard hoặc một lịch LR khác sẽ tốt hơn.

## 5. Pool tạo nhiều bước hơn nhưng ít thắng chủ động trước CNN

| Thống kê toàn train | Control | Pool |
|---|---:|---:|
| Episode / optimizer update | 2000 | 2000 |
| Tổng learner steps | 476492 | 832810 |
| Steps/episode | 238,25 | 416,41 |
| Episode có learner steps>=475 | 18,50% | 77,15% |
| ActiveWR train | 74,35% | 17,80% |
| Hòa train | 9,25% | 53,90% |
| Bom/episode | 23,05 | 42,56 |
| Bom/100 learner steps | 9,68 | 10,22 |

Pool chạy1,748lần số bước của control dù cùng số update. Trong pool, RB chiếm20%episode nhưng chỉ12,68%learner steps; các CNN chiếm87,32%quan sát. Không suy tỷ lệ bước thành tỷ lệ gradient: loss trajectory được lấy mean và mỗi episode chỉ update một lần.

Pool thắng chủ động256/400trận train trước RB và100/1600trước các CNN/recent, tương ứng64,0% và6,25%. CNN/recent tạo ít episode có kết liễu chủ động, nhiều trận dài/hòa; thêm dữ liệu như vậy chưa làm policy học được cách kết liễu CNN. Đây là mô tả dữ liệu on-policy trên các board khác nhau qua thời gian, không phải đánh giá độc lập hoặc bằng chứng nhân quả về reward.

Số bom/episode pool cao hơn phần lớn đi cùng trận dài: mật độ bom/100steps chỉ tăng khoảng5,6%. Trên offline decision states, xác suất BOMB trung bình S2/pool-2000 là33,70/33,74%; trên anchor9,28/9,84%. Chưa đủ bằng chứng gọi đây là policy spam bom toàn cục.

## 6. Giả thuyết reward đã bị loại và lỗi phụ phát hiện

Ban đầu chỉ đọc `_combat_reward` có thể hiểu nhầm SD win giữ lại thưởng+10. Truy vết lên engine cho thấy điều kiện `not sudden_death_deaths` ở `bomberman_env.py:316` đã loại bonus này. Probe event thuần tính toán với raw reward đúng theo engine xác nhận:

- Thắng thụ động bình thường: raw9,999 → shaped−0,001 sau khi hủy bonus.
- Đối thủ chết do vòng, learner không có kill: raw−0,001 → shaped−0,001; không có+10.
- Có owned kill và thắng: raw10,999 → shaped13,999.

Do đó loại giả thuyết “chờ vòng vẫn được thưởng thắng lớn”. Mean tổng reward theo các episode train thắng nhờ vòng chỉ khoảng1,03(control)/1,42(pool), có cả phần thưởng tường/vật phẩm từ trước đó; phù hợp với việc không nhận bonus+10.

Phát hiện phụ: helper `_reward_components` chưa xử lý ngoại lệ SD giống engine, nên optional reward diagnostics sẽ báo mismatch với event SD hợp lệ. Flag `reward_diagnostics` tắt trong lượt này; helper không tham gia loss, không giải thích thất bại D-S2. Cần sửa/kiểm helper ở source version mới trước khi bật log đó; chưa sửa production đã đóng băng.

## 7. Bước tiếp theo được đề xuất

1. Giữ S2; không mở test hoặc tiếp tục2000episode nữa cho lượt hiện tại. Không chỉ tăng tỷ lệ RB rồi chạy lại full: control100%RB cũng mất kỹ năng.
2. Ưu tiên một pilot ngắn mới để tách việc giữ policy: cùng S2, cùng đối thủ RB và các điều kiện khác; đối chứng anchor0 với một nhánh có KL neo S2. Trainer AC hiện đã có đường `anchor_coef`; tránh đồng thời đổi reward, LR, teacher và lịch pool. Hệ số/ngân sách/seed/gate cần chốt trong spec riêng; chưa chọn hoặc chạy ở phân tích này.
3. Theo dõi riêng retention trên decision states, RB ActiveWR và critic/gradient diagnostics. Kiểm việc neo ở trạng thái bình thường có giữ được cả quyết định chiến thuật quan trọng; nếu không, cân nhắc replay nhãn distillation trong một ablation khác. Không mặc định KL rollout thay thế được replay.
4. Nếu retention/control ổn mới thử pool. Nếu critic vẫn bất ổn, tách một phép thử critic warm-up hoặc hạn chế cập nhật shared trunk; không gộp vào cùng thay đổi anchor đầu tiên.

Các bước trên là đề xuất nghiên cứu, không phải cam kết cải thiện hay lệnh đã chạy. Một training seed,40board validation và các so sánh hậu kiểm chưa đủ để kết luận tổng quát về Phase D. Giữ seed test đã đặt chỗ chưa mở; mọi lượt mới cần inventory/registration riêng.

## Tái tạo và artifact

```bat
cd /d E:\Bomberman
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\runs\phaseD-s2-20261022-analysis-20261023\analyze.py
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\runs\phaseD-s2-20261022-analysis-20261023\check_reward.py
```

- `summary.json`: kết luận số và giới hạn; `comparisons.json`: paired CI theo board.
- `checkpoints.csv`, `matchups.csv`, `corners.csv`: checkpoint/đối thủ/góc.
- `training_audit.json`, `training_windows.csv`, `training_opponents.csv`: lịch và diễn biến train.
- `offline_sample_manifest.json`: file SHA, split, chỉ số sample tái lập; `offline_retention.csv`, `parameter_drift.csv`: đo forward-only.
- `reward_probe.json`: giả thuyết bị loại và event arithmetic đúng theo engine.

Thư mục analysis tách khỏi output đã đăng ký và ZIP gốc. Không có kết quả held-out mới hoặc quyết định thăng hạng model.
