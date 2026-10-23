# SD8 — chưng cất search hai nước vào CNN (A) và search ba nước (B)

Ngày: 20/10/2026. Bối cảnh: SD7.9 cho agent mạnh nhất G + `search_action2` (test 47000 ActiveWR 82.0). Người dùng yêu cầu làm cả hai hướng: A cho agent CNN nhanh, B cho trần ActiveWR cao hơn. Mọi luật dưới đây chốt trước khi chạy. Hướng dẫn chạy: `SD8_GUIDE.md`.

## SD8A — chưng cất G + search hai nước vào CNN (`run_sd8a_distill.py`)

Công thức SD5.3 (`run_sd53_pilot.py` của bạn, tái dùng helper), với ba thay đổi:

1. **Teacher là G + `search_action2`** (không còn F + search một nước).
   - Nhãn ở mỗi trạng thái là điểm search một nước.
   - Khi search một nước không thấy nước giết và search hai nước đã lập kế hoạch, điểm của các nước đầu được thay bằng điểm kế hoạch hai nước (`teacher_scores`).
   - Trạng thái quyết định là trạng thái mà các điểm này khác nhau.
2. **Student khởi đầu từ G**, và mẫu neo dùng logits của G (KL λ = 1).
3. **Không tăng cường dữ liệu bằng đối xứng**: audit SD5.3 cho thấy nhãn search không đối xứng (79% < 90%).

Cấu hình:

| Mục | Giá trị |
|---|---|
| Bàn thu thập | 210000–211499 (1500 bàn × 2 góc = 3000 trận, không trùng SD5.3) |
| Lịch vòng | E với xác suất 0.75 theo bàn |
| Mẫu neo | 5% các trạng thái còn lại |
| Train | Adam 1e-4, batch 256 (nửa trạng thái quyết định, nửa mẫu neo), 16 epoch; ứng viên epoch 4/8/12/16 |
| Held-out | 10% số bàn (rng 530) |
| Chọn epoch | Validation 21000, 400 trận, lịch E. Lấy ActiveWR cao nhất trong các ứng viên có tự nổ ≤ G + 1pp; hòa thì lấy epoch sớm hơn |

SD5.3 vẫn tăng ở epoch 8 nên lượt này mở tới 16 epoch.

Test seed **48000**, chạy một lần:
- Ba model chạy CNN thuần trên lịch E và lịch off: S2 (epoch được chọn), G, S.
- G + search2 chạy trên lịch E, chỉ để báo cáo làm trần tham chiếu.

Luật quyết định:
- **Gate** (lịch E): S2 − G ActiveWR ≥ +5pp, CI95 dưới > 0, tự nổ ≤ +1pp.
- **Luật phụ**: S2 thay S làm CNN thuần tốt nhất nếu CI95 dưới của S2 − S ActiveWR > 0.
- **Chỉ báo cáo**: S2 so với G + search2.

## SD8B — search ba nước (`run_sd8b_search3.py`)

`kill_search.search_action3` (= `search_action_deep(depth=3)`):
- Nếu search hai nước không thấy kế hoạch giết, nó thử thêm kế hoạch ba nước: mỗi nước đầu an toàn, rồi mỗi nước thứ hai an toàn, rồi mỗi nước thứ ba hợp lệ. Sau đó learner chỉ né; đối thủ là RuleBot.
- Khi search hai nước đã thấy kế hoạch giết, kết quả giống hệt search hai nước.
- Nếu nước của CNN đã nằm trong một kế hoạch giết thì giữ nước đó.

Đo trên 24 trận (seed 6600/6610/6620, 6 tiến trình): search2 khoảng 12 s/trận, search3 khoảng 23 s/trận. Mỗi bước search3 chậm khoảng 4 lần, nhưng trận ngắn hơn.

Validation 21000:
- G + search3 chơi 400 trận, lịch E.
- So cặp với G + search2 trên cùng bàn. Có thể tái dùng `runs/sd79b-search2-21000` khi chạy trên cùng máy và cùng phiên bản torch.
- **Luật**: search3 − search2 ActiveWR ≥ +2pp thì mở test.

Test seed **49000**, chạy một lần, chỉ lịch E: G + search3 và G + search2, 400 trận mỗi agent.
- **Gate**: ActiveWR ≥ +5pp, CI95 dưới > 0, tự nổ ≤ +1pp.
- Không chạy lịch off để giảm chi phí; SD7.9c cho thấy lịch off cùng chiều với lịch E.

## Giới hạn đã biết

- Search vẫn dùng RuleBot làm mô hình đối thủ.
- G chỉ có một train seed; SD8A cũng chỉ một train seed.
- Search ba nước chậm khoảng 2 lần mỗi trận so với hai nước.
- Nhãn của SD8A là nước đầu của một kế hoạch hai nước (thường là một bước đi), nên có thể khó học hơn nhãn "đặt bom ngay" của SD5.3.

## SD8B-off — ablation lịch sudden death (chốt trước khi chạy, 20/10/2026)

Người dùng yêu cầu sau khi SD8B ĐẠT. Câu hỏi: G + search3 phụ thuộc vào sudden death đến mức nào?

Cách làm:
- `run_sd8b_search3.py ablate`: G + search3 và G + search2 chạy lại với lịch **off** trên đúng các bàn của seed 49000, 400 trận mỗi agent.
- Ghép cặp với các trận lịch E đã có trong `runs/sd8b-20261020`.
- Dùng lại 49000 thay vì seed mới, vì ablation chỉ để mô tả: không có gate, không chọn model, không quyết định nào dựa vào nó. Ghép cặp trên cùng bàn khử được dao động giữa các bàn và tiết kiệm 800 trận lịch E.

Báo cáo, không có luật đạt/không đạt:
- Với mỗi agent: hiệu lịch E − off (bootstrap ghép cặp theo bàn) của ActiveWR, WR, hòa, thắng thụ động, tự nổ.
- Lịch off: search3 − search2.
- Tương tác: (E − off của search3) − (E − off của search2) theo ActiveWR. Con số này cho biết search3 có bớt cần sudden death hơn search2 không.
- Tỉ lệ thắng chủ động sau bước 475 (lúc vòng bắt đầu) dưới lịch E.
