# Insight tổng hợp (Bomberman RL, hướng SD)

Cập nhật: 18/10/2026. File này chỉ giữ những điều **đã có bằng chứng** và còn ảnh hưởng tới các thí nghiệm sau. Mỗi mục ghi nguồn; số liệu đầy đủ nằm trong `RESULTS.md`, cách cài đặt trong `IMPROVEMENTS.md`. Khi có insight mới, thêm vào đúng nhóm. Insight bị bác bỏ thì ghi lại vào mục "Đã bác bỏ", không xóa.

Ký hiệu: F = `runs/phaseC-sd-best-fallback/best_phaseC.pt` (model đang giữ); lịch E = SD engine 475/25/40; ActiveWR = thắng và có hạ gục bằng bom của mình.

## 1. Trận đấu được quyết định ở đâu

- **ActiveWR được quyết định ở giữa trận, trước bước 475.**
  - Trong 190 trận F sống tới bước 475, 174 trận chưa có kill nào (RESULTS SD6.0).
  - Trên B: 250/400 trận kéo tới bước 400, nhưng chỉ 19 trận trong số đó là thắng chủ động (PLAN SD4.2a).
  - Hệ quả: đòn bẩy cho giai đoạn cuối (thoát vòng, teacher endgame) không tăng được ActiveWR.
- **F chạm mặt đối thủ rất sớm (khoảng bước 14–19) ở 100% trận.** Hòa không phải do né tránh mà do không kết liễu được (RESULTS SD5.0).
- **F chết vì vòng nhiều, đúng lúc vòng đầu đóng.** 51/400 trận chết vì vòng, trong đó 39 trận đúng bước 475; 32 trong số đó là thua. Điều này ảnh hưởng WR và hòa, ít ảnh hưởng ActiveWR (RESULTS SD6.0).

## 2. Metric: thắng thụ động làm WR "đẹp" giả

- Nhiều cách can thiệp tăng WR mà ActiveWR đứng yên, vì chỉ đổi kết cục thụ động:
  - SD2 và biên mask k=15: WR +10.75 (SD4.1).
  - Luật đứng giữ ô cửa của đối thủ: WR +3.75 [1.75, 6.0], hòa −2.75, nhưng thắng thụ động 9.5 → 15.25 và ActiveWR −1.75 (RESULTS SD6.1b).
- **Ghi công kill theo chủ của quả bom gây chết.**
  - Đối thủ bị kẹt giữa bom của nó và bom của learner thường chết vì bom của chính nó, nên trận tính là thắng thụ động.
  - Luật chặn đường đã đổi 16 trận F vốn thắng chủ động thành thắng thụ động, vì đối thủ chết sớm hơn do bom của nó (SD6.1b).
  - Mọi teacher hay luật nhắm vào ActiveWR phải tối ưu **kill bằng bom của learner**, không chỉ "làm đối thủ chết".
- `kill_search` chỉ chấm +1 khi bom của learner gây chết. Đây là một lý do nó tăng được ActiveWR, còn luật chặn đường thì không.

## 3. Hành vi của F

- **Lấy mẫu ngẫu nhiên là cần thiết** (RESULTS SD6.0, validation 21000):
  - argmax: ActiveWR −9.25 [−16.0, −2.5], hòa +7.5; policy tất định bị kẹt trước RuleBot tất định.
  - τ = 0.5 và 0.25: không khác biệt có ý nghĩa.
  - Không dùng argmax khi đánh giá hay thu thập dữ liệu.
- Ở tình huống có nước giết, argmax của F đúng 90.5% nhưng lấy mẫu chỉ trúng 62%. Lỡ một bước thường không mất cơ hội khi cửa sổ giết dài, **nhưng 31/37 cửa sổ giết thất bại chỉ kéo dài 1 bước**, và ở 28 cửa sổ trong số đó nước giết duy nhất là đặt bom ngay, F chỉ gán P ≈ 0.58. Lỡ bước này là mất cơ hội (RESULTS SD7.6a).
- **38/57 trận F không thắng chủ động không có cơ hội giết nào** trong tầm 12 tick (seed 6800). Trần của mọi cách cải thiện kết liễu khoảng +19pp; vượt nữa phải tạo thêm cơ hội (RESULTS SD7.6a).
- **Điểm mù 1: đứng giữ ô cửa.**
  - Ở các nước giết bằng di chuyển mà search tìm ra, F gán xác suất trung vị **0.001**.
  - 28/38 trường hợp, nước đúng là STOP: learner đang đứng ở lối thoát duy nhất của đối thủ, trong khi F bỏ chạy (DOWN hoặc RIGHT) và mở đường cho đối thủ (RESULTS SD6.1).
- **Điểm mù 2: không rời vành ngoài trước bước 475.** Search sửa F để sống, tập trung quanh bước 471, với nước mà F gán xác suất ≈ 0 (SD6.0).
- **Nguyên nhân khả dĩ (chưa chứng minh nhân quả)** là teacher escape trong `train_actor_critic.train`:
  - Mỗi bước learner trong vùng nguy hiểm, nó gán nhãn đúng một lối thoát `_bfs_safe_cell → exits[0]` (chỗ trú gần nhất, không xét đối thủ), với hệ số 0.5 và hàng trăm nhãn mỗi episode.
  - Nhãn này đè mọi lối thoát khác, kể cả đứng giữ ô cửa.
- Mask an toàn (`_safe_action_mask`, `avoid_contests`) cấm đi vào ô kề đối thủ. Fixture test về chặn đường phải tôn trọng điều này (IMPROVEMENTS §42).

## 4. Train và chưng cất

- **Công thức của bạn đã bão hòa.** Train thêm 3000 ep từ F không cải thiện: nhánh control trên test là −6.5 [−12.75, 0.0] so với F; ActiveWR khi train phẳng quanh 45–49% (RESULTS SD5.1).
- **Kill-search lúc chơi là trần tham chiếu mạnh:** F + search đạt ActiveWR +22.5 [18.25, 26.75] trên validation (SD5.0), với mô hình RuleBot hoàn hảo. Search đồng ý với F ở khoảng 99.7% số bước.
- **Thông tin rất thưa.** Mỗi trận khoảng 230 bước, nhưng chỉ khoảng 2.2 bước có thông tin (các hành động có điểm khác nhau), khoảng 1.25 bước có nước giết, và search sửa F khoảng 0.9–1.1 lần (RESULTS SD5.3 chuẩn bị, SD6.0).
  - Gắn nhãn search ở mọi bước không thêm thông tin.
  - Nhãn thưa trộn vào RL với lr 2.5e-5 (0.9 nhãn/ep, học một lần) không đổi được CNN: SD5.1 search − F −2.75.
  - Hướng giải: gom nhiều trạng thái quyết định rồi học có giám sát nhiều epoch (spec SD5.3).
- Trong số nước search sửa để giết, khoảng một nửa là đặt bom (F vốn gán xác suất khoảng 0.58), một nửa là di chuyển hoặc đứng yên (F gán khoảng 0.001). Phần F thực sự không biết là nhóm thứ hai. *(Đã sửa bởi SD7.6a: 0.001 là xác suất của đúng nước search chọn; ở cấp tập nước giết, F đặt 0.958 lên nước di chuyển. Điểm nghẽn theo số lần thất bại là đặt bom ngay, xem mục 3.)*

- **Tại các trạng thái quyết định, F có phương sai lớn** (spike SD7.0, 81 trạng thái, 8 lần chơi lại × 20 tick): ở trạng thái search nói có nước giết, F giết được trung bình 60%, và 86% trạng thái cho kết quả lúc được lúc không. Ở trạng thái search nói không có nước giết, F chỉ giết được 0.3%. Chơi lại gần như không chết. Kill-search là bộ lọc trạng thái quyết định rất chính xác; tiêu chí rẻ (bom cháy và đối thủ ở gần) thì không (1.6%).
- Phần thưởng RL đã khớp với ActiveWR: kill của mình +4 (+1 env và +3 shaping), thắng chủ động +10, thắng thụ động bị trừ lại đúng 10 (`_reward_components`). RL bão hòa không phải vì mục tiêu sai.

- **Advantage theo nhóm cần tín hiệu đủ mạnh mới đổi được CNN.** `group_coef` 1, K 8 (SD7): group − control +0.34pp tỉ lệ giết sau 3000 ep; `group_coef` 4, K 16 (SD7.4): +7.12pp [+4.86, +9.72] chỉ sau 1000 ep. Nhưng không thành ActiveWR cả trận: SD7.5 (3000 ep) full − control +0.75 [−5.75, +7.25], vì ~1.2 trạng thái quyết định/trận × +7pp ≈ +1pp, nhỏ hơn CI ±6pp của 400 trận. Validation giảm sau ep 1000. Nguyên nhân (SD7.6a): 79% lần chơi lại đã giết được, chỉ 52% nhóm có chênh lệch; mẫu rơi vào cửa sổ dài đã chắc thắng, còn khoảnh khắc đặt bom một bước (nơi có gần hết thất bại) ít được train và bị bộ lọc rẻ loại nếu chưa có bom cháy. Train lâu làm policy nhọn hơn (entropy 0.432 → 0.396) và đặt nhiều bom hơn. RESULTS SD7.4, SD7.5, SD7.6a.
- **Đẩy BOMB ở khoảnh khắc "đặt bom ngay" lan sang mọi trạng thái** (SD7.6-probe): P(nước giết) ở trạng thái hẹp 0.72 → 0.97 và tỉ lệ giết lúc chơi lại +18pp, nhưng kills/trận không đổi (0.45 vs 0.46), bom/trận +20%, hòa +6, ActiveWR −0.5. Tăng tỉ lệ giết ở trạng thái chọn lọc, đo trên trạng thái của model khác, **không phải** chỉ báo đủ cho ActiveWR; mọi probe cơ chế sau này phải kèm kills/trận và bom/trận trên trận của chính model đó.
- **Phạt KL về G không tách được hai việc đó** (SD7.7-calibrate, 150 ep): P(BOMB | hợp lệ) và P(nước giết) ở trạng thái hẹp tăng gần bằng nhau ở mọi β (β 0: +0.31/+0.26; β 16: +0.19/+0.18); lan bom xuất hiện chỉ sau 150 ep. CNN không thấy được điều kiện "chỉ BOMB mới giết" (cần lookahead), nên mọi tín hiệu "đặt bom ở đây" đều thành "đặt bom". Muốn tách thì CNN phải được nhìn thấy tín hiệu đó (ví dụ một kênh input từ search lúc chơi), hoặc phải có ví dụ ngược "bom bừa thì tệ".

- **Chưng cất dày có giám sát chuyển được search vào CNN** (SD5.3 của bạn, test 43000: S − F +16.25 [+9.75, +22.5]; CNN thuần S đạt ~94% ActiveWR của F+search). Khác SD5.1 (nhãn thưa, học một lần, lẫn RL) và SD7.x (RL chỉ đẩy BOMB ở trạng thái chọn lọc): SD5.3 có ~6 200 trạng thái quyết định gồm cả ví dụ ngược ("ở đây bom không phải nước tốt nhất"), đích mềm theo điểm search, 8 epoch, kèm KL neo về F. Kết quả: P(nước tốt nhất) ở held-out 0.52 → 0.91, nhóm "đứng ở ô cửa" 0.28 → 0.93, mà P(BOMB) ở trạng thái thường không đổi (0.088 → 0.095). Vậy kết luận SD7.7 "CNN không thấy được điều kiện" chỉ đúng với tín hiệu RL một phía; với nhãn có cả ví dụ ngược thì CNN học được. Validation tăng đều qua epoch 2/4/8, chưa bão hòa. RESULTS SD5.3 review.
- **G (group RL) gần như không đổi CNN ở trạng thái quyết định:** P(nước tốt nhất) F 0.52, G 0.55; nhóm ô cửa 0.28 vs 0.30. Giải thích vì sao SD7.5 tăng ít. RESULTS SD5.3 review.
- **Search và RuleBot không đối xứng dưới phép quay/lật bàn** (audit SD5.3: 79% trạng thái giữ nguyên tập nước tốt nhất), nên không được tăng cường dữ liệu bằng đối xứng cho nhãn search.
- **Search một nước có trần chung ~65% ActiveWR, bất kể nền CNN** (F+search 64.25, G+search 64.75/66.25/61.0, S+search 65.5 trên các seed khác nhau). CNN tốt hơn ở trạng thái quyết định (S) không thêm gì khi đã có search, vì search đã sửa đúng các trạng thái đó. Phần còn lại là trận không bao giờ có nước giết một bước. RESULTS SD7.9a.
- **Search hai nước phá trần đó:** G + `search_action2` trên test 47000 ActiveWR 61.0 → 82.0, +21.0 [+17.0, +25.25], hòa 18.0 → 5.75, thắng thụ động −2.5, tự nổ 0, trận ngắn hơn ~90 bước. Đòn bẩy lớn nhất hiện tại là **tạo** cơ hội giết (đi một bước rồi đặt bom), không phải kết liễu tốt hơn. Cơ chế (ghép cặp theo bàn, test 47000): 87 trận thành thắng chủ động (45 từ hòa, 23 từ thắng nhờ vòng, 15 từ thắng thụ động), chỉ 3 trận mất; lịch off toàn bộ 90 trận hòa đổi được đều thành thắng chủ động; trung vị bước thắng chủ động 205 → 158, thắng chủ động trước bước 300 từ 42.0% lên 65.75%. Giá: chậm ~4 lần mỗi trận. RESULTS SD7.9b–c và mục tổng hợp insight 20/10/2026.
- **G chơi cuối trận tốt hơn S** (G+search vs S+search: hòa 12.25 vs 19.5, thắng nhờ vòng 10.75 vs 6.75, validation 21000), có lẽ nhờ G được train dưới SD 3000+1000 ep; S chưng cất từ F. RESULTS SD7.9a.
- **G + kill-search lúc chơi là agent mạnh nhất đã kiểm trên test** (SD7.8, seed 46000): ActiveWR 48.25 → 64.75, +16.5 [+12.5, +20.5]; thắng thụ động không đổi, chết vì vòng 11 → 1, bom/trận giảm. Search chỉ can thiệp đúng chỗ nên không có tác dụng phụ "đặt bom bừa" như các cách train (SD7.6/SD7.7). RESULTS SD7.8.

## 5. Quy trình và hạ tầng

- **Chọn best trong nhiều điểm validation 400 trận thiên lạc quan vài pp.** Control SD5.1: validation 51.0, test 42.0. Nên dùng ít ứng viên, chốt trước.
- **Một seed test 400 trận dao động khoảng ±5pp cho cùng một cặp model.** Control SD5.1 − F: −6.5 (42000), +3.5 (44000), +3.0 (45000); gộp 600 cặp bàn 0.0 [−3.75, +3.75]. Mức hơn vài pp trên một seed (như G − F +3.75) không phân biệt được với dao động này (RESULTS KC1).
- **Validation không so được giữa hai phiên bản torch.** Cùng F, cùng seed: 47.0 (torch 2.5.1) và 43.75 (torch 2.14.1), vì policy lấy mẫu. Mọi stage của một thí nghiệm (train, validate, test, kể cả đánh giá F) chạy trên **cùng một máy** (SD5.1).
- **Clone từ GitHub trên Windows đổi xuống dòng thành CRLF**, nên SHA source khác dù code giống hệt. Phải so bằng `diff --strip-trailing-cr` hoặc tính SHA trên nội dung đã chuẩn hóa (IMPROVEMENTS §39).
- **Không chỉnh luật nhiều lần trên cùng một seed validation.** Sau số lần thử đã chốt (SD6.1 và bản sửa SD6.1b), dừng lại; muốn thử tiếp thì dùng seed chẩn đoán khác và chốt luật trước.
- **Probe nhiều tiến trình:**
  - Lưu kết quả thô từng shard trước khi gộp (SD6.1 mất 1200 trận vì lỗi gộp).
  - Chia shard theo bàn (2 góc mỗi bàn), không theo trận.
  - Smoke test phải chạy cả bước gộp.
- Resume train chính xác cần lưu cả RNG chọn episode có SD (`curriculum_rng`); state được lưu sau mỗi episode, nên Ctrl+C chỉ mất episode đang chạy (IMPROVEMENTS §36).

## 6. Đã bác bỏ

| Giả thuyết | Bằng chứng | Nguồn |
|---|---|---|
| Teacher giai đoạn cuối (local / two_bomb / RuleBot) hạ đối thủ tốt hơn B | Đều kém B | SD4.2a |
| Search chỉ ở giai đoạn cuối tăng ActiveWR | search@360 −0.25 | SD4.6 |
| F biết nước giết, chỉ lỡ vì lấy mẫu; hạ nhiệt độ sẽ giúp | τ 0.25: +1.75 [−5.5, 8.75]; argmax −9.25 | SD6.0 |
| Train RL thêm từ F bằng công thức của bạn | control −6.5 trên test | SD5.1 |
| Nhãn kill-search thưa trong RL | search − F −2.75 [−9.75, 4.0] | SD5.1 |
| Luật đứng chặn đường lúc chơi tăng ActiveWR | −1.75 [−4.25, 0.75]; WR tăng nhờ thắng thụ động | SD6.1b |

- Advantage theo nhóm với tín hiệu yếu (SD7: K = 8, `group_coef` 1, ~0.8 trạng thái/ep, 3000 ep) không nâng tỉ lệ giết của F (group − control +0.34pp) và không nâng ActiveWR (+0.25 [−5.75, 6.25], seed 44000). Lưu ý: SD7.4 cho thấy cơ chế có tác dụng khi tín hiệu đủ mạnh, nên chỉ cấu hình yếu bị bác bỏ. RESULTS SD7.1–SD7.3.

## 7. Câu hỏi còn mở

- Sau SD7.5: cải thiện ở trạng thái quyết định cần nhân lên bao nhiêu (nhiều trạng thái hơn/trận, hay hiệu ứng lớn hơn/trạng thái) để thấy được trên test 400 trận? Vì sao validation giảm khi train lâu với `group_coef` 4?

- Chưng cất search **hai nước** vào CNN (thu bằng G+search2, train từ G theo công thức SD5.3) có cho CNN thuần gần 82% ActiveWR không, và có giữ được lối chơi cuối trận của G không?
- Search ba nước / dùng CNN làm chính sách tiếp diễn thay vì "chỉ né" có tăng tiếp không, với chi phí chấp nhận được?
- Thay nhãn `exits[0]` của teacher escape bằng đích mềm trên mọi lối thoát an toàn có làm F tự tìm ra nước đứng giữ ô cửa, và có tăng ActiveWR không? (Chưa thử; SD6.1b cho thấy giữ ô cửa thường dẫn tới thắng thụ động, nên kỳ vọng thấp.)
- Luật chặn đường cho WR +3.75 và hòa −2.75 không cần train. Có đáng kiểm chứng trên seed test mới nếu mục tiêu là WR và hòa? (Người dùng chưa quyết.)
