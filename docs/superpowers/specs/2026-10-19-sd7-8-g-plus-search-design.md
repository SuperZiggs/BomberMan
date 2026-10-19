# SD7.8 — G + kill-search lúc chơi, test một lần seed 46000

Ngày: 19/10/2026. Người dùng chọn hướng này sau khi SD7.7 hiệu chỉnh không đạt. Luật chốt **trước** khi chạy.

## Lý do

- SD5.0: F + kill-search lúc chơi tăng ActiveWR +22.5 [18.25, 26.75] trên validation 21000.
- SD7.6a/SD7.7: điểm nghẽn là khoảnh khắc "đặt bom ngay"; CNN không tự nhận ra khoảnh khắc này, và mọi cách đẩy BOMB qua train đều lan thành đặt bom bừa. Search lúc chơi chỉ can thiệp đúng ở những khoảnh khắc nó thấy có nước giết, nên không lan.

## Thiết kế

- Hai agent, cùng luồng RNG torch theo seed bàn: `G` (lấy mẫu như cũ) và `G+search` (hành động lấy mẫu của G, trừ khi `kill_search.search_action` tìm được nước có điểm cao hơn hẳn; giống SD5.0, chỉ đổi F thành G).
- Seed **46000** (chưa dùng; trước giữ cho SD7.6/SD7.7, hai hướng đó không chạy test), 400 trận mỗi agent, mọi góc, lịch E (gate) và tắt SD (chỉ báo cáo). Chạy một lần (`test_opened.json`).
- Không chọn tham số nào; search dùng nguyên cấu hình SD5.0 (tầm 12 tick, phạm vi 6, đối thủ RuleBot).

## Gate (lịch E)

`G+search` − `G`: ActiveWR ≥ +5pp, CI95 dưới > 0 (bootstrap theo cặp bàn), tự nổ ≤ +1pp. Báo kèm WR, hòa, thắng thụ động, kills, bom/trận.

## Giới hạn đã biết

Search giả định đối thủ là RuleBot (mô hình hoàn hảo trong bài này), nên lợi thế có thể không chuyển sang đối thủ khác. Agent cuối cùng cần chạy search khi chơi (~19 ms mỗi bước có đối thủ trong phạm vi 6).
