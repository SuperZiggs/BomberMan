# SD7.9 — nền CNN cho search và search hai nước

Ngày: 19–20/10/2026. Bối cảnh: bạn của người dùng chạy xong SD5.3 (CNN chưng cất S = `runs/sd53-20261018/train/epoch_8.pt`, SHA `8f7116d2…fd4a`). Người dùng yêu cầu xem kỹ SD5.3 rồi cải thiện agent G + kill-search. Mọi luật dưới đây chốt trước khi chạy stage tương ứng.

## SD7.9a — S hay G làm nền cho search (validation)

- Ba nhánh trên validation 21000, 400 trận, lịch E: G+search, S, S+search (`run_sd79_probe.py`).
- Luật: nếu S+search − G+search ActiveWR ≥ +2pp thì S+search là ứng viên test 47000; nếu không, không dùng seed test. S − G+search chỉ báo cáo.

## SD7.9b — search hai nước (validation)

- `kill_search.search_action2`: nếu search một nước không thấy nước giết, thử mọi nước đầu an toàn (điểm ≥ 0), giả lập 1 tick với RuleBot, rồi thử mọi nước thứ hai hợp lệ (sau đó chỉ né). Chọn nước đầu của kế hoạch giết; nếu nước của CNN đã nằm trong một kế hoạch giết thì giữ nước của CNN. Chậm hơn khoảng 4 lần mỗi trận (17.8 so với 4.5 s).
- Nhánh: G+search2, so với các trận G+search của SD7.9a trên cùng bàn.
- Luật (chốt sau SD7.9a, trước SD7.9b): G+search2 − G+search ActiveWR ≥ +2pp thì mở test 47000.

## SD7.9c — test một lần seed 47000

- `run_sd79_test.py`: G+search vs G+search2, 400 trận mỗi agent, lịch E (gate) và off (báo cáo), `test_opened.json`.
- Gate (lịch E), giống SD7.8: ActiveWR ≥ +5pp, CI95 dưới > 0, tự nổ ≤ +1pp. Đạt thì agent khuyến nghị là G + search hai nước.

## Giới hạn đã biết

Cả hai search dùng RuleBot làm mô hình đối thủ (đúng với đối thủ của bài này). Search hai nước tốn khoảng 4 lần thời gian mỗi trận. Một train seed cho G.
