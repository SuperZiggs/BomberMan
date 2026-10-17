# T1 hoàn tất: two_bomb 200 episode chưa cải thiện model

Kết luận: **giữ `runs/pressure-C-next/final_phaseC.pt`**. Lượt two_bomb đã train đủ 200 episode và hoàn tất 1.200 trận đánh giá, nhưng không đạt gate cải tiến. T1 hoàn thành về thực thi; kết quả chọn candidate là **FAIL**, không còn là lượt bị thiếu episode như pilot đầu.

## Cấu hình và kiểm tra dữ liệu

Phase C, board 9×9, hai agents, max_steps 500, lr=1e-4, seed train 2026101303; warm-start từ gốc. Local đối chứng là lượt 200 episode đã hoàn tất, được dùng lại với cùng source/config/checkpoint. Two_bomb chạy lại từ gốc, không nối thêm từ final 146. Validation 100 trận seed21000 ở episode 0/100/200.

Đánh giá CPU qua mask hiện tại, không gọi teacher mới: mỗi model 400 trận, seed bàn 26000–26199, mỗi bàn A0/A1. Chọn best theo validation trước test; không chọn final bằng điểm test. Thời gian chạy ghi nhận 5.118,984 giây, khoảng **85,32 phút**.

Audit đã kiểm tra:

- Metrics đúng episode 1–200, learner luân phiên A0/A1; đủ validation 0/100/200.
- Mỗi evaluation đủ 400 cặp seed/corner duy nhất; summary, rate/average tổng và từng corner khớp từng game.
- Source hiện tại và snapshot năm file core khớp manifest; model gốc và local đối chứng còn đúng checksum.
- Best two_bomb bằng gốc ở toàn bộ tensor; tất cả 400 game records two_bomb/original bằng nhau. SHA256 hai file checkpoint có thể khác dù tensor giống, nên đã so tensor thay vì chỉ so hash file.
- Final two_bomb khác gốc ở cả 12 tensor: learner thực sự đã được cập nhật. Final không được chấm riêng trên bộ 400 trận này.
- Tính lại paired bootstrap 10.000 lần theo 200 seed bàn, giữ A0/A1 cùng nhau, seed bootstrap20261014; kết quả khớp summary.

Code/self-check và dữ liệu audit: [analyze_results.py](/D:/fpt/subject/BomberMan/runs/two-bomb-complete-20261014-031626/analyze_results.py), [analysis_results.json](/D:/fpt/subject/BomberMan/runs/two-bomb-complete-20261014-031626/analysis_results.json). Audit chỉ đọc dữ liệu/checkpoint, không train hay chơi thêm trận.

## Test cuối: best đã chọn

| Model | Best episode | WR | ActiveWR | Tự nổ | Hòa | Kills/trận | Thắng thụ động |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gốc | gốc | 43,00% | 36,50% (146/400) | 0,50% (2/400) | 54,75% | 0,365 | 6,50% |
| Local | 200 | 44,00% | 38,00% (152/400) | 0,25% (1/400) | 54,50% | 0,380 | 6,00% |
| Two_bomb | 0 | 43,00% | 36,50% (146/400) | 0,50% (2/400) | 54,75% | 0,365 | 6,50% |

**36,5% của best two_bomb là hiệu năng của warm-start giữ lại**, không phải bằng chứng CNN sau 200 episode đã học combo tốt. Model gốc 33% trên test25000 trước và 36,5% trên test26000 hiện tại dùng cùng trọng số/source; chênh lệch đến từ bộ đánh giá khác, không phải nâng cấp model. Tương tự, tự nổ 0% ở bộ trước không bảo đảm mọi bộ seed đều 0%.

| Chênh lệch ActiveWR | Điểm phần trăm | CI95% theo seed bàn |
|---|---:|---:|
| Two_bomb − gốc | 0,00 | [0,00; 0,00] |
| Two_bomb − local | -1,50 | [-7,25; 4,50] |
| Local − gốc | +1,50 | [-4,50; 7,25] |

Two_bomb không đạt +5 điểm hay cận dưới CI>0 so với hai đối chứng; self-blast quan sát cũng cao hơn local (2 so với 1 trận). Local tăng sáu active wins trên mẫu này, nhưng CI gồm0 và mức tăng <5 điểm, nên cũng chưa đạt tiêu chí thay model. Không kết luận local tốt hơn trên mọi seed, hoặc two_bomb kém mọi phương pháp.

### Từng góc, 200 trận/model/góc

| Model/góc | WR | ActiveWR | Tự nổ | Hòa | Kills/trận | PassiveWR |
|---|---:|---:|---:|---:|---:|---:|
| Gốc A0 | 44,00% | 36,50% | 1,00% | 52,00% | 0,365 | 7,50% |
| Gốc A1 | 42,00% | 36,50% | 0,00% | 57,50% | 0,365 | 5,50% |
| Local A0 | 51,00% | 42,50% | 0,50% | 46,50% | 0,425 | 8,50% |
| Local A1 | 37,00% | 33,50% | 0,00% | 62,50% | 0,335 | 3,50% |
| Two_bomb A0 | 44,00% | 36,50% | 1,00% | 52,00% | 0,365 | 7,50% |
| Two_bomb A1 | 42,00% | 36,50% | 0,00% | 57,50% | 0,365 | 5,50% |

Local tăng6 điểm ở A0 nhưng giảm3 điểm ở A1; không bỏ qua lệch góc để chỉ báo tổng38%. Two_bomb bằng gốc vì chính sách best không đổi.

## Validation: model sau train đang kém hơn warm-start

| Episode two_bomb | ActiveWR | Tự nổ | Hòa | Bom/trận | Steps/trận | Chọn best |
|---|---:|---:|---:|---:|---:|---|
| 0 | 29,00% | 0,00% | 62,00% | 40,89 | 393,38 | Có |
| 100 | 18,00% | 1,00% | 77,00% | 27,52 | 434,99 | Không |
| 200 | 15,00% | 0,00% | 81,00% | 27,09 | 446,32 | Không |

Guard loại episode100 vì tự nổ vượt mốc0%; episode200 qua guard tự nổ nhưng ActiveWR vẫn thấp hơn mốc. Vì vậy best0 là lựa chọn đúng theo rule đã khóa. Không thể giải thích kết quả bằng việc teacher mới chỉ train tới146 episode nữa: đã đủ200 mà validation vẫn giảm14 điểm so với đầu lượt.

Tuy vậy, đây vẫn là một train seed; final200 có validation chứ chưa có held-out400 trận riêng. Không dùng validation100 trận để kết luận final chính xác15% trên mọi seed hoặc cả phương pháp search teacher vô ích.

## Chẩn đoán sơ bộ từ log

Teacher gọi **81.718 lần**, đánh giá **3.308.269 kế hoạch**, nhận **77.355 nhãn (94,66% query)**; trong đó **59.064 nhãn hai bom (76,35% nhãn nhận)**. Pressure dự báo trung bình trên nhãn nhận là **0,3624**; thời gian planner trung bình **19,31 ms/query**, tổng khoảng26,31 phút. Các con số này không đo tỷ lệ combo thực thi hay kill thật.

| Train two_bomb | Episode1–50 | Episode151–200 |
|---|---:|---:|
| ActiveWR train | 20,00% | 16,00% |
| Hòa train | 72,00% | 76,00% |
| Bom/trận | 36,92 | 25,78 |
| Combat loss trung bình | 1,9241 | 0,9510 |

Trên các cửa sổ train này, agent đặt ít bom hơn và active wins không tăng dù combat loss giảm. Các cửa sổ gặp bàn/trải nghiệm khác nhau và các loại nhãn khác local, nên chưa thể quy nguyên nhân cho một coefficient hay đọc loss thành tỷ lệ theo nhãn. Không dùng `combat_bombs/combat_labels` làm agreement: nhãn combo có thể là di chuyển.

Các metric số học trong log đều hữu hạn. Peak loss ep51 là69,55, với value loss132,11; riêng phần0,5×value đóng góp66,05. Loss scalar lớn chưa chứng minh gradient bị critic/teacher áp đảo; gradient theo nhóm chưa được log.

Hiện chưa biết từ các file này:

- CNN có chọn các first actions được teacher gợi ý không, hay chỉ tăng probability mà vẫn sample action khác.
- Sau bom đầu, CNN đặt được bom thứ hai đúng vị trí/tick bao nhiêu lần.
- Prefix vật lý có qua mask tại các bước sau không.
- Teacher đổi kế hoạch bao nhiêu lần; bao nhiêu nhãn thật sự dự báo complete trap.

## Quyết định và bước kế tiếp

1. Đóng T1: thực thi hoàn tất, gate cải tiến FAIL; giữ model gốc, giữ mặc định local và giữ các candidate làm đối chiếu.
2. Không cấp ngay một lượt dài hơn với nguyên teacher này, không nới guard hay trộn PPO/reward để cứu pilot.
3. Làm T2 trace trên bản sao/replay: first-action agreement, khả thi qua future masks, plan churn, first/second-bomb placement, planned trap và actual owned kill. Final200 dùng chẩn đoán, không chọn làm best bằng audit.
4. Chọn một nhánh sửa từ bằng chứng T2: lọc nhãn pressure yếu, loại prefix không qua mask hoặc ổn định mục tiêu. Không mở cả ba cùng lúc.
5. T3 curriculum và T4 advantage/GAE/PPO/reward giữ trong backlog; chỉ mở với baseline/teacher đã khóa và đối chứng riêng.

Các seed test26000–26199 đã được đọc; thay method tiếp thì dùng test mới. Không sửa các JSON/metrics/checkpoint đầu vào; lượt phân tích này chỉ thêm audit/report và cập nhật tracker/kết quả.
