# T2 fixed-policy audit

Audit thu đủ dữ liệu theo config; full_protocol=True. Quyết định sửa teacher chờ đọc trace/replay. Giữ checkpoint gốc.

| Model | Trận | ActiveWR | Hòa | Tự nổ | Coverage | Agreement | Bom hai / bom đầu |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 200 | 33.00% (66/200) | 57.00% (114/200) | 1.00% (2/200) | 92.86% (69774/75136) | 61.23% (42726/69774) | 2.54% (23/905) |
| two_bomb_final | 200 | 16.00% (32/200) | 75.50% (151/200) | 1.00% (2/200) | 95.18% (79147/83157) | 52.98% (41931/79147) | 1.17% (5/427) |

## Cách đọc để chọn một nhánh

- L1: xem pressure, trap và hiệu quả clone với evader trước khi thử lọc nhãn.
- L2: xem physical_pass_but_mask_blocked, tick/action bị chặn; chưa nới mask.
- L3: xem churn và replay. Event association chỉ là proxy; unknown chưa chứng minh teacher đổi vô lý.
- T4W: nếu plan khả thi nhưng agreement/mass thấp, kiểm tra nhãn và cân bằng gradient trước.
- L4: xem timing instrument/clone/write riêng; observer không tương đương timing train T1.

Replay: 10 games, xem replays.html. Mỗi model có breakdown A0/A1 trong summary.json.

## Giới hạn

- Fixed-policy audit: không tái tạo các state/policy thay đổi trong train 200 episode.
- Chỉ clone các state sampled theo config; đối thủ clone là fixed policy phản ứng, không fixed position/tape (trừ static).
- Masked check chỉ kiểm tra prefix tới bom cuối. Suffix là timed BFS vật lý dựa trên bom đã biết, không CNN hay safety mask.
- Suffix chưa dự báo các bom đối thủ đặt thêm; full-horizon alive chỉ là kết quả của branch đã chạy.
- Post-terminal physics tách riêng, không dùng làm ActiveWR/owned kill thật của trận CNN.
- Churn event association là proxy; unknown chưa đủ chứng minh đổi kế hoạch không do state mới.
- Tied first actions đều được giữ; completion theo một path tương ứng, strict path khác có thể vẫn hợp lý.
- Timing teacher đã gồm observer; không so trực tiếp với timing train T1. Kill trong plan window chưa có causal attribution.
- Best two_bomb bằng original ở từng tensor nên alias, không chạy lại trận giống hệt.
- Audit set là exploratory; không dùng để chọn/promote checkpoint hoặc tune rồi báo như test độc lập.
