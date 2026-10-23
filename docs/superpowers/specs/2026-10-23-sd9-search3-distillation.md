# SD9 — chưng cất S2 + search ba nước thành CNN thuần (S3)

Ngày: 23/10/2026. Người dùng đồng ý hướng "chưng cất search3 vào CNN" (sau PDE2). SD9.0 (validation 21000) chọn thầy theo luật chốt trước: S2 + search3. Mọi luật dưới đây chốt trước khi chạy. Runner: `run_sd9_distill.py`, bản chép của `run_sd8a_distill.py` với các thay đổi liệt kê bên dưới. File SD8A giữ nguyên vì lượt SD8A đã xong kiểm SHA source của chính nó.

## Câu hỏi

SD8A chưng cất G + search2 (86.5 vs RuleBot) thành S2 (83.0). Chưng cất S2 + search3 thành S3 có làm CNN thuần mạnh thêm, đủ vượt gate +5pp so với S2 không?

## Khác SD8A

| Mục | SD8A | SD9 |
|---|---|---|
| Thầy | G + `search_action2` | S2 + `search_action3` |
| Học trò xuất phát / neo KL | G | S2 (`runs/sd8a-20261020/train/epoch_16.pt`, SHA `d222d81f…6814`) |
| Nhãn | điểm một nước, ghi đè bằng kế hoạch hai nước | thêm ghi đè bằng kế hoạch ba nước (`deep_plans`) |
| Bàn thu dữ liệu | 210000–211499 | **212000–213499** (chưa ai dùng) |
| Seed test | 48000 | **57000** (mới, dùng một lần) |
| Đối chứng gate | G | S2 |

Giữ nguyên:
- Thu dữ liệu: 3000 trận đấu RuleBot, lịch E với xác suất 0.75. Trạng thái quyết định là trạng thái mà điểm search khác nhau. Neo: 5% trạng thái còn lại, dùng logits của S2.
- Train: CE mục tiêu mềm + KL về S2 (λ = 1), Adam 1e-4, batch 256, 16 epoch, ứng viên 4/8/12/16, held-out 10% bàn.
- Validation: seed 21000, 400 trận/ứng viên, lịch E; chọn ActiveWR cao nhất với tự nổ ≤ S2 + 1pp.

## Test (một lần, seed 57000, 400 trận/agent = 200 bàn × 2 góc, 9×9 1v1, max_steps 500)

- CNN thuần S3, S2, G: lịch E và off.
- Mốc trần chỉ báo cáo: S2 + search3 và G + search3 (lịch E).
- Bootstrap theo cặp bàn 10 000 lần, seed 20261014.

## Gate (chốt trước)

- **ĐẠT** nếu đủ ba điều kiện (lịch E):
  - S3 − S2 ActiveWR ≥ +5pp;
  - CI95 dưới > 0;
  - tự nổ tăng không quá +1pp.
- Khi ĐẠT, S3 thay S2 làm CNN thuần mạnh nhất và làm điểm xuất phát cho Phase D/E (nếu người dùng duyệt).
- Khi FAIL, giữ S2. Không chạy thêm test để cứu gate.
- Báo kèm (không quyết định gì): S3 so với S2 + search3 và G + search3.

## Chi phí và người chạy

- Thu dữ liệu: ~40 s/trận mỗi tiến trình (smoke), nên khoảng 6–7 giờ với 5 tiến trình. Có `--resume`.
- Train nhỏ (~3700 bước): chạy được trên CPU, GPU thì nhanh hơn.
- Test: khoảng 1 giờ (phần search chiếm gần hết).
- Ai chạy cũng được (người dùng hoặc bạn). Lệnh có trong `SD9_GUIDE.md`.

## Giới hạn

- Một seed train/test.
- Thầy và đối thủ đánh giá đều dựa trên RuleBot. SD9 không nhắm vào vấn đề hòa khi đấu CNN (PDE/PDE2).
- SD9.0: S2 + search3 vs RuleBot có ActiveWR thấp hơn G + search3 5 điểm (CI [−11, +1]). Thầy có thể không mạnh hơn G + search3.
