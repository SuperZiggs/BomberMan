# Phân tích kết quả S2-KL — 23/10/2026

**Kết luận: `RB_RETENTION_FAIL`. Hoàn tất 1.400/1.400 trận và đóng gói thành công. Giữ S2 gốc làm mặc định.**

Ở endpoint 500 đã đăng ký trước, KL β=1 giữ R/Q nhỉnh hơn control β=0, nhưng mức cải thiện chưa đạt ngưỡng và cả hai vẫn giảm mạnh so với S2. ActiveWR của anchor-500 chỉ68%, so với84% của S2 và76% của control-500. Các lệnh đều hoàn tất với exit0; status này là kết quả khoa học không đạt gate.

**Thiết lập và mẫu số**

- Hai nhánh khởi tạo từ cùng S2 gốc, optimizer/scheduler mới,500 episode/nhánh, cùng seed/lịch bàn; controlβ=0 và anchorβ=1. Đối thủ chỉ là RuleBot, PhaseD1v1,9×9. Không có pool CNN trong lượt này.
- Mỗi checkpoint được chấm200 trận:100 board validation720000–720099, mỗi board đủ hai góc. Bảy SHA riêng biệt cho tổng1.400trận; S2 và hai checkpoint episode0 được tính một lần qua alias.
- Offline dùng1024 decision states và1024 anchor states cố định từ TRAIN SD8A cho mỗi model. Tổng7168+7168 là số lượt chấm model-state; không phải14.336state độc lập. Đây là phép đo giữ kỹ năng trên dữ liệu từng dùng để train S2.
- CI95% dùng10.000cluster-bootstrap ghép theo board,seed20261023. Offline decision có722 board clusters; validation có100 board clusters, giữ hai góc của cùng board trong một cluster.
- Chỉ checkpoint500 quyết định pilot. Checkpoint100/250 dùng chẩn đoán, không thay endpoint sau khi thấy kết quả.

**Validation đầy đủ — mọi tỷ lệ là % trên200 trận/model**

| Model | WR | ActiveWR | Hòa | Thua | Tự sát | Thắng SD | Chết SD | Thắng thụ động khác |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 | 93.00 | 84.00 | 4.50 | 2.50 | 0.50 | 2.00 | 1.50 | 7.00 |
| control-100 | 84.50 | 69.50 | 10.50 | 5.00 | 1.50 | 7.00 | 5.00 | 8.00 |
| control-250 | 81.50 | 70.00 | 14.00 | 4.50 | 1.00 | 5.00 | 4.50 | 6.50 |
| control-500 | 88.00 | 76.00 | 8.00 | 4.00 | 1.00 | 4.50 | 2.50 | 7.50 |
| anchor-100 | 80.00 | 67.00 | 15.00 | 5.00 | 0.50 | 5.50 | 5.00 | 7.50 |
| anchor-250 | 87.50 | 76.50 | 8.00 | 4.50 | 1.00 | 5.50 | 3.00 | 5.50 |
| anchor-500 | 82.50 | 68.00 | 10.50 | 7.00 | 0.50 | 8.00 | 4.00 | 6.50 |

WR = ActiveWR + thắng sudden death + thắng thụ động khác. Thắng thụ động khác là phần còn lại ngoài hai nhóm trước theo phân loại runner. Tự sát/chếtSD là sự kiện và có thể thuộc trận hòa; các cột sự kiện không cộng thành100%. Tỷ lệ thua được suy ra100−WR−hòa.

| Model | Số thắng/200 | Thắng chủ động/200 | Hòa/200 | Tự sát/200 | Thắng SD/200 | Chết SD/200 |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 186 | 168 | 9 | 1 | 4 | 3 |
| control-500 | 176 | 152 | 16 | 2 | 9 | 5 |
| anchor-500 | 165 | 136 | 21 | 1 | 16 | 8 |

**Retention offline**

R là tỷ lệ argmax thuộc tập nước hợp lệ có điểm tốt nhất, cho phép đồng hạng. Q là tổng xác suất dành cho tập nước đó, lấy trung bình theo state. R/Q dưới đây chấm1024 decision states; tập1024 anchor states bổ sung đo độ lệch chính sách trên các tình huống khác. KL offline có chiều S2||model; KL trong training có chiều learner||S2.

| Model | R (%) | Q (%) | KL decision | KL anchor | Đổi argmax decision (%) |
|---|---:|---:|---:|---:|---:|
| S2 | 97.3633 | 91.8252 | 0.0000 | 0.0000 | 0.0000 |
| control-100 | 78.7109 | 75.1068 | 0.7530 | 0.0873 | 26.7578 |
| control-250 | 74.9023 | 70.3139 | 1.6594 | 0.1523 | 29.7852 |
| control-500 | 75.0000 | 69.9993 | 1.4437 | 0.1234 | 29.9805 |
| anchor-100 | 76.7578 | 72.7294 | 1.1053 | 0.1288 | 28.6133 |
| anchor-250 | 82.8125 | 76.7509 | 0.5695 | 0.0675 | 22.6562 |
| anchor-500 | 77.5391 | 73.2168 | 0.9075 | 0.0897 | 27.2461 |

**Các so sánh endpoint 500 và CI95%**

Mỗi ô là chênh lệch điểm phần trăm [cận dưới; cận trên]. CI phản ánh độ biến động theo các board/state đã chấm, không bao gồm độ biến động giữa nhiều lần train độc lập.

| Chỉ số | Control-500 − S2 | Anchor-500 − S2 | Anchor-500 − control-500 |
|---|---:|---:|---:|
| R | -22.36 [-25.17; -19.62] | -19.82 [-22.44; -17.24] | +2.54 [+1.34; +3.87] |
| Q | -21.83 [-24.23; -19.47] | -18.61 [-20.76; -16.52] | +3.22 [+2.67; +3.81] |
| WR | -5.00 [-11.00; +1.00] | -10.50 [-17.00; -4.00] | -5.50 [-11.50; +0.50] |
| ActiveWR | -8.00 [-16.00; -0.50] | -16.00 [-24.00; -8.00] | -8.00 [-16.00; +0.00] |
| Hòa | +3.50 [-2.00; +8.50] | +6.00 [+1.00; +11.50] | +2.50 [-3.00; +8.00] |
| Tự sát | +0.50 [-1.00; +2.50] | +0.00 [-1.50; +1.50] | -0.50 [-1.50; +0.00] |
| Thắng SD | +2.50 [-1.00; +6.00] | +6.00 [+2.00; +10.01] | +3.50 [-1.00; +8.00] |
| Chết SD | +1.00 [-1.50; +4.00] | +2.50 [-0.50; +6.00] | +1.50 [-2.00; +5.50] |
| Thắng thụ động khác | +0.50 [-4.50; +5.50] | -0.50 [-5.00; +4.00] | -1.00 [-5.50; +3.50] |

**Sáu gate đã chốt trước**

| Điều kiện | Ngưỡng | Quan sát endpoint 500 | Kết quả |
|---|---|---|---|
| DRIFT_REPRODUCED | Control R giảm ít nhất10pp | Giảm22.3633pp | Đạt |
| DECISION_RETAINED | Anchor R ≥92.3633% | 77.5391% | Không đạt |
| PROBABILITY_RETAINED | Anchor Q ≥86.8252% | 73.2168% | Không đạt |
| ANCHOR_BENEFIT | Anchor−control R≥10pp và CI lower>0 | +2.5391pp;CI[+1.3448;+3.8658] | Không đạt |
| RB_ACTIVE_RETAINED | Anchor ActiveWR≥82% | 68% | Không đạt |
| RB_SUICIDE_RETAINED | Anchor tự sát≤1.5% | 0.5% | Đạt |

Hai trong sáu điều kiện đạt. Nhánh control tái hiện drift như dự kiến. Nhánh anchor không giữ được R,Q hoặc ActiveWR. Lợi ích R tuy có CI hoàn toàn dương nhưng+2.54pp vẫn nhỏ hơn ngưỡng10pp. Nhánh anchor giữ được tỷ lệ tự sát trong giới hạn. Status RB_RETENTION_FAIL xuất hiện vì gate RuleBot được xét trước các gate retention còn lại; các gate sai khác vẫn được ghi đầy đủ.

**Diễn giải kết quả**

1. Cả hai nhánh đều mất kỹ năng so với S2. Control-500 có R75% so với97.36% của S2; anchor-500 có R77.54%. Điều này tái hiện drift ngay cả khi chỉ train trước RuleBot.
2. KL có lợi ích hạn chế trên tập retention: anchor−control R+2.54pp,CI95%[+1.34;+3.87]; Q+3.22pp,CI[+2.67;+3.81]. KL offline decision giảm từ1.4437 ở control xuống0.9075 ở anchor, nhưng vẫn cách S2 và chưa giữ được hành động tốt.
3. Hiệu quả trước RuleBot của anchor-500 giảm rõ so với S2: ActiveWR−16pp,CI[−24;−8]. So với control-500, quan sát là−8pp,CI[−16;0]; khoảng này chạm0 nên không khẳng định chênh lệch có ý nghĩa theo tiêu chí CI loại trừ0 nghiêm ngặt.
4. WR tổng che bớt mức giảm thắng chủ động: anchor-500 có68% ActiveWR+8% thắngSD+6.5% thắng thụ động khác=82.5% WR. S2 có84% ActiveWR+2% thắngSD+7% thắng thụ động khác=93% WR. Đây là mô tả phân loại kết quả, không phải bằng chứng khai thác lỗi reward.
5. Anchor-250 hồi phục một phần: R82.81%,ActiveWR76.5%, rồi giảm ở episode500. Đây là tín hiệu chẩn đoán; endpoint không được đổi sang250 để tuyên bố pilot đạt. Anchor-250 cũng vẫn thấp hơn S2.
6. Một seed training và một β cố định chưa cho phép kết luận mọi biến thể KL đều thất bại. Kết quả đủ để kết luận cấu hìnhβ=1 này chưa đạt mục tiêu giữ kỹ năng ở endpoint đã đăng ký. Nguyên nhân cụ thể như ảnh hưởng critic/shared trunk, teacher hoặc độ phủ state vẫn cần ablation riêng.

**Tách train metrics khỏi validation**

| Nhánh | WR train50episode cuối | ActiveWR train50episode cuối | WR validation200 trận | ActiveWR validation200 trận |
|---|---:|---:|---:|---:|
| Control-500 |86%|70%|88%|76%|
| Anchor-500 |72%|60%|82.5%|68%|

Train metrics là cửa sổ50episode với policy đang cập nhật và bàn train. Validation dùng checkpoint đóng băng trên200 trận riêng. Không dùng các dòng cuối terminal train thay cho kết quả validation. Log xác nhận anchor_enabled ở500/500 episode của nhánh anchor và0/500 episode của control; anchor_kl episode500 là0.0432371. Các con số này chỉ xác minh cơ chế đã bật, không đo tỷ lệ đóng góp gradient từng thành phần.

**Đối chiếu hai góc — mỗi hàng100trận**

| Model | Góc | WR | ActiveWR | Hòa | Tự sát | Thắng SD | Chết SD | Thụ động khác |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 | 0 | 94.00 | 88.00 | 4.00 | 0.00 | 1.00 | 1.00 | 5.00 |
| S2 | 1 | 92.00 | 80.00 | 5.00 | 1.00 | 3.00 | 2.00 | 9.00 |
| control-100 | 0 | 82.00 | 66.00 | 10.00 | 3.00 | 7.00 | 6.00 | 9.00 |
| control-100 | 1 | 87.00 | 73.00 | 11.00 | 0.00 | 7.00 | 4.00 | 7.00 |
| control-250 | 0 | 78.00 | 69.00 | 17.00 | 2.00 | 3.00 | 4.00 | 6.00 |
| control-250 | 1 | 85.00 | 71.00 | 11.00 | 0.00 | 7.00 | 5.00 | 7.00 |
| control-500 | 0 | 88.00 | 74.00 | 6.00 | 2.00 | 5.00 | 0.00 | 9.00 |
| control-500 | 1 | 88.00 | 78.00 | 10.00 | 0.00 | 4.00 | 5.00 | 6.00 |
| anchor-100 | 0 | 78.00 | 65.00 | 15.00 | 1.00 | 6.00 | 5.00 | 7.00 |
| anchor-100 | 1 | 82.00 | 69.00 | 15.00 | 0.00 | 5.00 | 5.00 | 8.00 |
| anchor-250 | 0 | 88.00 | 76.00 | 5.00 | 2.00 | 5.00 | 3.00 | 7.00 |
| anchor-250 | 1 | 87.00 | 77.00 | 11.00 | 0.00 | 6.00 | 3.00 | 4.00 |
| anchor-500 | 0 | 81.00 | 68.00 | 12.00 | 1.00 | 7.00 | 2.00 | 6.00 |
| anchor-500 | 1 | 84.00 | 68.00 | 9.00 | 0.00 | 9.00 | 6.00 | 7.00 |

Ở anchor-500,ActiveWR cả hai góc đều68%, thấp hơn S2 góc0=88% và góc1=80%; mức giảm không chỉ xuất hiện tại một góc.

**Các chỉ số hành vi bổ sung trên validation**

| Model | Kills/trận | Bom/trận | Bước/trận |
|---|---:|---:|---:|
| S2 | 0.840 | 19.930 | 204.210 |
| control-100 | 0.695 | 24.125 | 254.625 |
| control-250 | 0.700 | 26.120 | 261.180 |
| control-500 | 0.760 | 23.175 | 241.110 |
| anchor-100 | 0.670 | 26.910 | 265.455 |
| anchor-250 | 0.765 | 23.385 | 245.395 |
| anchor-500 | 0.680 | 26.865 | 269.860 |

Anchor-500 đặt nhiều bom hơn và có trận dài hơn S2 trong bộ validation này, trong khi số kill trung bình thấp hơn. Đây là quan sát hành vi, chưa tách được nguyên nhân.

**Quyết định và giới hạn**

Giữ S2 gốc làm mặc định. Pilot chưa đạt nên chưa có cơ sở chuyển sang pool pilot theo gate này. Không mở test D-S2, không có held-out test mới trong S2-KL, và không tuyên bố PhaseD đạt. Không tự thay β,train thêm hoặc đổi endpoint sau kết quả. Một thí nghiệm tiếp theo cần thiết kế và đăng ký riêng trước khi chạy.

**Kiểm tra tính toàn vẹn và tệp bàn giao**

Đã tái dựng report từ700 raw board shards và7 diagnostic shards, đối chiếu bằng report.json; kiểm tra source/runtime, full training states và validation projection; kiểm CRC toàn bộ ZIP. ZIP còn nguyên SHA256476ac4b1fb59fa24a51682d33c37f54bbff0cb23b33736f48258040a73cbbd1a,64.511.876byte. Báo cáo phân tích này nằm riêng ngoài output đã đóng gói.

- [Report gốc](E:/Bomberman/runs/s2-kl-20261023/report.json)
- [Decision report](E:/Bomberman/runs/s2-kl-20261023/decision_report.md)
- [Validation](E:/Bomberman/runs/s2-kl-20261023/validation.json)
- [Gói kết quả](E:/Bomberman/runs/s2-kl-20261023-send.zip)
- [Bằng chứng kiểm tra](E:/Bomberman/runs/s2-kl-20261023-analysis-20261023/verification.json)
