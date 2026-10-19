# SD7 — RL với advantage theo nhóm từ các khoảnh khắc quyết định

Ngày: 18/10/2026. Trạng thái: người dùng yêu cầu viết và chạy ("viết và chạy đồng thời kiểm định giả thuyết và thí nghiệm"). Mọi luật dưới đây được chốt **trước** khi train.
Liên quan: `INSIGHTS.md`, RESULTS SD5.0, SD5.1, SD6.0, SD6.1b, SD7.0.

## 1. Lý do

- Phần thưởng RL đã khớp với ActiveWR: kill bằng bom của mình +4, thắng chủ động +10, thắng thụ động bị trừ lại đúng 10. Vậy RL bão hòa không phải vì mục tiêu sai, mà vì tín hiệu quá thưa: khoảng 0.5 kill và khoảng 2 bước quyết định mỗi trận 230 bước. Mỗi episode chỉ ra một kết quả, phải chia công cho mọi bước.
- Kill-search lúc chơi nâng ActiveWR +22.5 (SD5.0). Như vậy các cơ hội giết có tồn tại, nhưng F không tận dụng hết.
- Spike SD7.0: ở trạng thái kill-search thấy có nước giết bằng bom của learner, F chơi lại 8 lần thì giết được trung bình 60%, và 86% trạng thái cho kết quả lúc được lúc không. Ở trạng thái search thấy không có nước giết, F chỉ giết được 0.3%.
- Từ **cùng một trạng thái**, các lần chơi lại chỉ khác nhau ở hành động của policy. Advantage của mỗi lần là kết quả trừ trung bình nhóm: nhiễu thấp, không cần value function, không cần teacher. Kết cục lấy từ ghi công của env, nên tự phân biệt kill của mình với thắng thụ động, đúng chỗ luật SD6 đã vấp.

## 2. Giả thuyết và cách kiểm (chốt trước)

- **H1, cơ chế: cập nhật theo nhóm nâng tỉ lệ biến cơ hội thành kill.**
  - Đo: lấy tập trạng thái quyết định cố định từ các trận F tự chơi trên seed chẩn đoán mới 6650–6699 (100 trận), theo đúng cách chọn của spike SD7.0. Cho từng model chơi lại K = 8 lần × 20 tick từ cùng các trạng thái đó; so theo cặp trạng thái.
  - Đạt nếu: tỉ lệ giết của group − F ≥ **+10pp**, với CI95 bootstrap theo trạng thái có cận dưới > 0.
  - Báo cáo thêm control − F để tách phần do train thêm.
- **H2, kết quả: ActiveWR cả trận tăng.** Đây là gate chính, đo trên test seed **44000** (mục 6).
- **H3, không gây hại:** tự nổ không tăng quá 1pp so với F. Hòa và chết vì vòng chỉ báo cáo.
- Cách đọc kết hợp:
  - H1 đạt nhưng H2 không đạt: cơ chế đúng, nhưng tỉ lệ biến cơ hội thành kill không phải nút thắt của ActiveWR.
  - H1 không đạt: thiết kế cập nhật chưa truyền được tín hiệu, ví dụ lr hoặc hệ số quá nhỏ.

## 3. Thiết kế

**Nhánh:**
- **control:** tái dùng nhánh control của SD5.1 (`runs/sd5-20261017/control`, best ep 2500, SHA `c3b4e37c…`, train trên chính máy này với torch 2.5.1).
- **group:** cùng config với control, cộng thêm phần cập nhật theo nhóm.
- Runner kiểm config của group trùng control, chỉ khác các khóa `group_*` và `checkpoint_dir`. Hai nhánh chỉ khác nhau ở cập nhật theo nhóm.
- Điểm xuất phát F, 3000 episode, train seed 2026101701, công thức của bạn, SD lịch E ở 75% episode.

**Cập nhật theo nhóm, trong mỗi episode train:**
- Ứng viên: learner còn sống, có bom đang cháy, đối thủ nằm trong vùng nguy hiểm và cách learner Manhattan ≤ 4.
- Với mỗi ứng viên, gọi `kill_search.search_action`. Nếu có nước giết bằng bom của learner, chụp trạng thái (`deepcopy(env)`). Tối đa 3 trạng thái mỗi episode, hai trạng thái cách nhau ≥ 8 bước.
- Cuối episode, từ mỗi trạng thái đã chụp, policy hiện tại chơi lại K = 8 lần, mỗi lần tối đa H = 20 tick, gặp RuleBot (`RuleBasedAgent` mới cho mỗi lần chơi lại).
- Kết cục mỗi lần chơi lại: +1 nếu bom của learner giết đối thủ khi learner còn sống, −1 nếu learner chết, 0 nếu không có gì.
- Advantage của lần thứ k = R_k − trung bình(R). Nhóm có mọi kết cục bằng nhau thì không đóng góp gì.
- Loss nhóm = −trung bình trên mọi bước của learner trong các lần chơi lại [A_k · log π(a | s)], với logits đã che mask như trainer. Loss tổng = loss hiện có + `group_coef` · loss nhóm, với **group_coef = 1.0**. Vẫn chỉ một bước optimizer mỗi episode, cùng cách clip gradient.
- Không có teacher trong các lần chơi lại, value head không học từ chúng, và không đổi gì khác trong trainer.

**Cách ly RNG:**
- Các lần chơi lại lấy mẫu hành động bằng `torch.Generator` riêng.
- RuleBot dùng RNG numpy toàn cục, nên trạng thái này được lưu và khôi phục quanh mỗi lần chơi lại; seed mỗi lần lấy từ `group_rng` riêng.
- Env sao chép mang theo `rng` riêng của nó.
- Trận chính vì thế không bị đổi. Kiểm bằng test: `group_rl=True` với `group_coef=0` phải cho trọng số giống hệt `group_rl=False`.
- `group_rng` được lưu trong training state để resume chính xác.

**Chọn checkpoint:** giống control: validation 21000, 400 trận, mỗi 500 episode, chọn ActiveWR cao nhất với giới hạn tự nổ của warm-start. Giữ nguyên để so công bằng, dù đã biết cách chọn này thiên lạc quan (INSIGHTS §5).

## 4. Không phải mục tiêu, phản mục tiêu

- Không đổi RuleBot, `kill_search`, env, lịch E, reward hay các teacher hiện có.
- Không chạy search lúc chơi; agent cuối cùng là CNN thuần.
- Phản mục tiêu:
  - WR tăng bằng thắng thụ động;
  - tự nổ tăng;
  - chọn checkpoint sau khi xem test;
  - chạy test lần hai.

## 5. Chi phí ước tính

- Train trên GPU RTX 3050 của máy này: khoảng 2.8 s/ep (control SD5.1), cộng search ở các bước ứng viên, cộng tối đa 3 × 8 lần chơi lại × 20 tick. Ước khoảng 4–6 s/ep, tức khoảng 4–6 giờ cho 3000 ep, cộng 7 lần validation.
- Smoke đo thời gian thật trước khi chạy full.

## 6. Gate (test một lần, seed 44000, 400 trận/model, lịch E; báo thêm khi tắt SD)

- Model: F, control (best SD5.1), group (best).
- Gate:
  1. group − F ActiveWR ≥ **+5pp**;
  2. CI95 dưới của delta > 0;
  3. group − control ≥ 0;
  4. tự nổ group − F ≤ **+1pp**.
- Đạt cả bốn thì group thay F. Không đạt thì giữ F.

## 7. Thứ tự chạy

1. Code và test (TDD), smoke.
2. Train nhánh group ở nền (`run_sd7_pilot.py train --arm group`), có `--resume`.
3. Chạy `mechanism` (H1) trên seed 6650–6699 cho F, control và group.
4. Chạy `test` một lần (H2, H3).
5. Ghi RESULTS, IMPROVEMENTS, INSIGHTS, PLAN, CLAUDE.md.

## 8. Quyết định của tôi (người dùng có thể đổi)

- K = 8, H = 20, tối đa 3 trạng thái mỗi episode, giãn cách 8 bước, group_coef = 1.0.
- Tái dùng control của SD5.1.
- Ngưỡng H1 là +10pp.
- Seed test 44000; seed cơ chế 6650–6699.
