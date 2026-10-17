# T4-R0 — audit reward hoàn tất

**R0 DONE/REVIEWED. Giữ checkpoint B và các hệ số reward hiện tại.** Không phát hiện sai lệch accounting trong các transition đã kiểm tra. Có cơ sở thử R1 như một ablation riêng; chưa chứng minh reward là nguyên nhân chính của hòa.

Đã audit 300 trận trên validation đã dùng:100/model,50 board seeds21000–21049 và cả A0/A1. CPU một thread, 9×9/2agents/max500,RuleBot,CNN lấy mẫu qua mask cũ. Thời gian runner 16.25 phút. Không train, chọn model, mở test mới hoặc thay checkpoint.

## 1. Cohort và parity

| Checkpoint | Active win | Passive win | Timeout | Chết do đối thủ | Tự nổ |
|---|---:|---:|---:|---:|---:|
| B | 29 | 8 | 62 | 1 | 0 |
| MC final200 | 25 | 7 | 67 | 1 | 0 |
| GAE final200 | 29 | 8 | 62 | 1 | 0 |

Các tỷ lệ win/active-win/draw/suicide và trung bình kills/bombs/items/steps khớp validation B1 ở đúng checkpoint: B dùngepisode 0; hai final dùngepisode 200. Đây là dữ liệu validation để chẩn đoán, không phải test mới hoặc bằng chứng thay model. Các model dùng chungboard/corner;300 game không phải300board độc lập.

## 2. Vì sao timeout vẫn có reward dương?

| Checkpoint | Timeout n | Reward mean / median / p95 | Time mean | Wall mean | Item mean | Reward dương |
|---|---:|---|---:|---:|---:|---:|
| B | 62 | 0.917 / 0.850 / 1.545 | -0.500 | 1.335 | 0.081 | 62/62 |
| MC final200 | 67 | 0.966 / 0.900 / 1.590 | -0.500 | 1.368 | 0.098 | 67/67 |
| GAE final200 | 62 | 1.006 / 0.950 / 1.598 | -0.500 | 1.410 | 0.096 | 62/62 |

Mọi timeout trong cohort có reward dương. Với B, time−0,5 bị vượt bởi wall≈+1,335 và item≈+0,081. Đây là vấn đề cần kiểm tra về mức thưởng phụ; kết quả không có nghĩa policy thích hòa hơn mọi lần thắng. Thắng chủ động vẫn nhận tổng reward lớn hơn rõ rệt trong cohort.

## 3. Hành vi: đã tới đối thủ nhưng chưa hạ được

| Checkpoint | Timeout từng tới khoảng cách≤3 | Bom pressure>0, toàn timeout | Bom/100 bước cuối | Pressure>0 trong bom cuối | Wall reward 100 bước cuối | Tỷ lệ STOP cuối |
|---|---:|---:|---:|---:|---:|---:|
| B | 62/62 | 93.99% | 10.403 | 100.0% | 0.000 | 3.16% |
| MC final200 | 67/67 | 93.92% | 11.075 | 100.0% | 0.000 | 3.39% |
| GAE final200 | 62/62 | 93.92% | 11.032 | 100.0% | 0.000 | 3.44% |

Trong100 bước cuối timeout, wall reward bằng0 ở cả ba model; mean reward khoảng−0,1. MC final có duy nhất một lần nhặt item trong toàn bộ các cửa sổ cuối, khiến mean reward −0,0993. CNN vẫn đặt khoảng10–11bom/100 bước cuối, tất cả có pressure dương theo phép đo hiện tại. Dữ liệu không ủng hộ mô tả agent chỉ đứng farm gỗ cuối trận; các chỉ số này gợi ý kiểm tra chất lượng thế bom và khả năng chuyển sức ép thành kill; R0 chưa chỉ ra nguyên nhân chiến thuật cụ thể.

Pressure chỉ đo giảm refuges với bom đã biết và vị trí đối thủ hiện tại. Pressure>0 không chứng minh trap, kill hoặc combo thực thi được trước RuleBot. Khoảng cách Manhattan≤3 và hướng BFS là các proxy; không chứng minh an toàn toàn bộ đường đi. Bảng chi tiết và nguyên liệu: [timeout_last100.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/timeout_last100.json), [games_original.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/games_original.json).

## 4. Return đã chiết khấu và thứ tự kết quả

R0 tính thêm return tại bước đầu G0 bằng đúng gamma 0,99 từ B1. Đây là MC return của critic/control; không giả định GAE actor dùng cùng target advantage. Tổng component returns được kiểm tra khớp G0.

| Checkpoint | G0 mean thắng chủ động | G0 mean timeout | Wall G0 mean timeout |
|---|---:|---:|---:|
| B | 3.467 | 0.847 | 0.901 |
| MC final200 | 3.698 | 0.869 | 0.917 |
| GAE final200 | 3.354 | 0.892 | 0.939 |

Trung bình G0 của thắng vẫn cao hơn timeout. Tuy nhiên có hai ví dụ trên cùng seed/góc mà return timeout cao hơn trajectory thắng, khi tính cả reward phụ và thời điểm nhận reward. Đây là ví dụ accounting, chưa chỉ ra nguyên nhân thay đổi policy. Toàn bộ88 cặp thắng/timeout đủ điều kiện giữa các checkpoint được giữ, không chỉ hai ví dụ:

| Seed/góc | Model thắng / G0 | Model timeout / G0 | Gap timeout−win sau giảm nửa wall |
|---|---|---|---:|
| 21034/A1 | B / 0.819162 | MC final200 / 0.929953 | 0.002644 |
| 21030/A0 | GAE final200 / 0.817132 | MC final200 / 0.832626 | -0.037679 |

Hai cặp không phải hai thí nghiệm độc lập; cùng board được dùng qua nhiều model. Giảm nửa wall xóa một đảo thứ tự và làm nhỏ đảo còn lại; giảm nửa item không xóa hai ví dụ này. Không suy ra thay đổi ActiveWR từ phép tính đó. Dữ liệu đầy đủ: [discounted_returns.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/discounted_returns.json), [paired_outcome_rankings.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/paired_outcome_rankings.json).

## 5. What-if trên cùng action tape

**Không train hoặc đổi hành động.** Chỉ tính lại reward nếu wall0,15→0,075 hoặc item0,05→0,025. Xem đây là độ lớn thay đổi mục tiêu, không phải dự báo hành vi hay win-rate. Dùng tolerance 1e−10 khi phân loại dương/0/âm để không tính nhiễu số thực là thưởng.

| Checkpoint | Mean timeout hiện tại | Nếu wall0,075 | Dương /≈0/âm | Nếu item0,025 |
|---|---:|---:|---|---:|
| B | 0.917 | 0.249 | 57 / 1 / 4 | 0.876 |
| MC final200 | 0.966 | 0.282 | 62 / 1 / 4 | 0.917 |
| GAE final200 | 1.006 | 0.301 | 61 / 1 / 0 | 0.958 |

Wall bonus chiếm phần lớn reward phụ ở timeout; giảm item một nửa tác động nhỏ hơn nhiều. Giảm nửa wall vẫn để phần lớn timeout có reward dương, nên R1 là phép thử mức thưởng, không phải cách ép agent thắng hoặc giải quyết toàn bộ chiến thuật.

## 6. Mean/median/p95 của mọi thành phần

Bảng dưới dùng toàn100 game/model, gồm cả0 khi sự kiện không xảy ra. Tất cả nhóm kết quả, sum/min/max và tỷ trọng tuyệt đối nằm trong [component_statistics.csv](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/component_statistics.csv) và [statistics.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/statistics.json). Nhóm tự nổ có n=0 nên CSV để trống thống kê có điều kiện; không coi thiếu mẫu là penalty0. Mỗi model chỉ có1 trận chết do đối thủ và đó là cùng seed/góc, không đủ để đề xuất chỉnh death penalty.

### B

| Component | Mean | Median | P95 |
|---|---:|---:|---:|
| time | -0.3934 | -0.5000 | -0.0458 |
| item | 0.0785 | 0.0750 | 0.2000 |
| wall | 1.3020 | 1.2000 | 1.9500 |
| owned_kill | 1.1600 | 0.0000 | 4.0000 |
| active_win | 2.9000 | 0.0000 | 10.0000 |
| passive_win_env | 0.8000 | 0.0000 | 10.0000 |
| passive_win_cancellation | -0.8000 | 0.0000 | 0.0000 |
| death | -0.0100 | 0.0000 | 0.0000 |
| self_blast | 0.0000 | 0.0000 | 0.0000 |

### MC final200

| Component | Mean | Median | P95 |
|---|---:|---:|---:|
| time | -0.3993 | -0.5000 | -0.0457 |
| item | 0.0850 | 0.1000 | 0.2000 |
| wall | 1.2990 | 1.2750 | 1.9500 |
| owned_kill | 1.0000 | 0.0000 | 4.0000 |
| active_win | 2.5000 | 0.0000 | 10.0000 |
| passive_win_env | 0.7000 | 0.0000 | 10.0000 |
| passive_win_cancellation | -0.7000 | 0.0000 | 0.0000 |
| death | -0.0100 | 0.0000 | 0.0000 |
| self_blast | 0.0000 | 0.0000 | 0.0000 |

### GAE final200

| Component | Mean | Median | P95 |
|---|---:|---:|---:|
| time | -0.3896 | -0.5000 | -0.0457 |
| item | 0.0805 | 0.1000 | 0.2000 |
| wall | 1.3065 | 1.3500 | 1.9500 |
| owned_kill | 1.1600 | 0.0000 | 4.0000 |
| active_win | 2.9000 | 0.0000 | 10.0000 |
| passive_win_env | 0.8000 | 0.0000 | 10.0000 |
| passive_win_cancellation | -0.8000 | 0.0000 | 0.0000 |
| death | -0.0100 | 0.0000 | 0.0000 |
| self_blast | 0.0000 | 0.0000 | 0.0000 |

Không chia reward phụ cho tổng reward có thể gần0/âm. Gross shares dùng tổng trị tuyệt đối các component theoepisode. Net shares loại cặp passive+10/−10 triệt tiêu; các khoản raw vẫn được lưu để kiểm chứng.

## 7. Biểu đồ và kiểm chứng

![Reward theo kills và active-win indicator](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/reward_scatter.png)

300 điểm được giữ nguyên ở y; jitter x xác định≤0,035 chỉ tránh chồng điểm. Active-win ở scatter là biến0/1; ActiveWR là tỷ lệ tổng hợp. Mỗi model có100 game/50 board, không coi hai góc là độc lập. Không smoothing, loại outlier hoặc CI. Reward có bonus kill/win theo định nghĩa nên tương quan giữa reward và win không chứng minh nguyên nhân hay chất lượng thuật toán.

[figure_data.csv](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/figure_data.csv); [reward_scatter.svg](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/reward_scatter.svg); [figure_manifest.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/figure_manifest.json). PNG3600×2100, nền trắng, cùng thang y, dùng cả marker và màu. Đã xem PNG để kiểm tra clipping và chữ. Palette audit có0 cảnh báo contrast màu đặc trên trắng và2 cặp cần phân biệt khi grayscale; marker khác nhau cung cấp mã hóa bổ sung. Không tuyên bố chứng nhận accessibility hoặc chuẩn journal.

Review kiểm tra 118,226transition: shaped reward mỗi bước khớp bit-exact công thức cũ; tổng component sai số tối đa1.78e-15, dưới ngưỡng1e−10. Replay lại12 timeout/6,012frames;300 hàng dữ liệu biểu đồ khớp game rows. ZIP raw37 files/CRC/bytes,source11, checkpoint vàparent input hashes qua. Nguồn thô không sửa sau review.

Bộ regression188 PASS/1 SKIP(CUDA không khả dụng), tổng189 tests. Có test default/logger CPU bit-parity với source-before cho actions,rewards,metrics vàweights; useful items/cap,late kill/chain,multiple kills,passive cancellation,mixed lingering self-flames,death+wall vàpost-deathFFA. Smoke6 game/max20 qua; auditfull300 game khớp validation B1 và không cập nhật optimizer. CUDA logger vẫn chưa được chạy trên runtime hiện tại.

## 8. Quyết định và bước tiếp theo

Giữ checkpoint B và toàn bộ coefficient hiện tại. Đánh dấu R0 DONE/REVIEWED. **Mở R1 như ablation một biến:** wall bonus0,15 so0,075, giữ item/kill/win/death/time/mask/teacher/MC raw. Cần runner đối chứng, cùng warm-startB,validation chọnbest vàtest mới đã đăng ký; chưa trainR1 trong R0. Không ghép item,terminal penalty,GAE/PPO hoặc curriculum vào cùng lượt.

R1 có cơ sở để kiểm tra mức thưởng phụ và vài trường hợp thứ tự return lệch mục tiêu ActiveWR. Bằng chứng hiện tại không chứng minh farm gỗ là nguyên nhân chính của hòa, hoặc giảm wall bonus sẽ nâng ActiveWR. Agent vẫn giao tranh; chất lượng trap/combo là điểm nghẽn riêng còn tồn tại.

Giới hạn: audit mô tả trên50validation board đã xem và3 checkpoint, không phải causal ablation. Có1 death case/model,0 self-blast; không suy ra an toàn hay penalty tối ưu. Runtime phụ thuộc máy; không dùng thời gian này để so tốc độ MC/GAE. Không mở thêm heldout hoặc dùng35000 chọn model. Các model tham chiếu vẫn nằm trongworkspace; ZIP raw không đóng gói checkpoint weights.

Files: [summary.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/summary.json), [config.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/config.json), [review_results.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/review_results.json), [timeout_cases.json](/D:/fpt/subject/BomberMan/runs/r0-full-20261015-153908/timeout_cases.json). Script review và report writer lưu cùng thư mục; output phân tích được thêm sau ZIP raw.
