# PDE2 — nhiều agent đánh nhau: FFA hỗn hợp, self-play gương, đối đầu 1v1 (không train)

Ngày: 20/10/2026. Người dùng hỏi vì sao Phase D/E không cho nhiều agent đánh nhau, và yêu cầu viết spec này trước. **Chỉ chạy khi người dùng báo đã có kết quả SD8A** (checkpoint S2). Mọi luật dưới đây chốt trước khi chạy. Runner dự kiến: `run_pde2_ffa_mix.py` (viết khi có S2).

## Bối cảnh

- PDE (`docs/superpowers/specs/2026-10-20-pde-check-design.md`, seed 50000) chủ yếu đo agent đấu RuleBot (E1/E2) và đấu một CNN đông cứng (D). Kết quả đó so được với các mốc cũ nhưng không cho biết các agent mạnh xếp hạng thế nào khi đánh trực tiếp với nhau.
- Thiết kế gốc của Phase D/E (PROJECT_SPEC) là self-play: các bản sao của agent đánh nhau, RuleBot chỉ là mỏ neo. Gate cũ của Phase E là "mỗi góc spawn có tỉ lệ thắng khoảng 20–30%".
- SD8A của bạn (báo qua ảnh 20/10/2026, chưa kiểm định): S2 = CNN thuần chưng cất G + search2, test 48000 ActiveWR 83.0 (G 48.0, S 57.5; G + search2 86.5). S2 nhanh như CNN thường, nên dùng được làm ghế chơi trong FFA và làm điểm xuất phát cho self-play.

## Agent

| Tên | Định nghĩa |
|---|---|
| G+s3 | G (`runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c…c7ac`) + `kill_search.search_action3` |
| S2 | Checkpoint SD8A đã chọn (`runs/sd8a-20261020/selected.json`). Runner kiểm SHA khớp `runs/sd8a-20261020/test_opened.json`; không khớp thì từ chối chạy. |
| G | G, CNN thuần |
| RB | RuleBot |

Mọi CNN lấy mẫu hành động qua mask chung. Search mô phỏng mọi đối thủ bằng RuleBot (giới hạn đã biết, cũng là điều cần đo).

## Thiết kế

Mọi phần dùng lịch E (475/25/40), max_steps 500, bàn bắt đầu từ seed **51000** (seed mới, dùng một lần cho cả PDE2).

**Phần 1: FFA hỗn hợp 4 agent.**
- Bốn ghế G+s3, S2, G, RB cùng một bàn.
- Mỗi bàn chơi 4 lượt xoay vòng: ở lượt r, góc i do agent (i + r) mod 4 ngồi. Như vậy agent nào cũng ngồi đủ 4 góc trên cùng bàn.
- 9×9: 50 bàn × 4 lượt = 200 trận. Đây là cấu hình chính, vì mọi CNN được train trên 9×9.
- 11×11: 50 bàn × 4 lượt = 200 trận. Đây là cấu hình Phase E2 gốc; CNN dùng trọng số nội suy.

**Phần 2: self-play gương.**
- 4 bản sao S2 đánh nhau. 9×9: 400 bàn; 11×11: 400 bàn; mỗi bàn một trận.
- Đo tỉ lệ thắng theo góc (gate công bằng góc của Phase E cũ).
- Không chạy gương G+s3, vì 4 search mỗi bước quá chậm.

**Phần 3: đối đầu 1v1** (Phase D, 9×9), mỗi cặp 100 bàn × 2 góc = 200 trận:
- G+s3 vs S2;
- S2 vs G.

G+s3 vs G đã có trong PDE.

Ước tính chi phí: ~400 trận FFA có ghế search3 (~150–190 s/trận) + 200 trận 1v1 có search3 (~30 s) + phần CNN rẻ. Tổng khoảng 20 giờ CPU, tức khoảng 4–5 giờ với 5 tiến trình. Runner có `--resume` theo shard như `run_pde_check.py`.

## Metric và báo cáo

- **Phần 1:** với mỗi agent, WR (là người sống sót duy nhất), ActiveWR (thắng và có kill bằng bom của mình), kills/trận, tự nổ, chết vì vòng; thêm tỉ lệ hòa của cả trận. Mức công bằng là 25%.
  - So cặp WR và ActiveWR giữa các agent, bootstrap theo bàn: mỗi bàn giữ đủ 4 lượt xoay; 10 000 lần, seed 20261014.
  - Các cặp so: G+s3 − S2, S2 − G, G+s3 − G, S2 − RB, G − RB.
- **Phần 2:** tỉ lệ thắng theo góc, tính trên tổng trận và trên trận có người thắng; tỉ lệ hòa.
- **Phần 3:** WR, ActiveWR, hòa, tự nổ của agent đứng trước trong cặp; CI bootstrap theo bàn.

## Luật đề xuất (chốt trước; chỉ đề xuất, không tự train)

1. **Agent khuyến nghị cho FFA:**
   - Là agent có ActiveWR cao nhất ở Phần 1, 9×9, nếu CI của hiệu với agent đứng thứ hai không chứa 0.
   - Nếu CI chứa 0, ghi "ngang nhau" và khuyến nghị agent nhanh hơn (S2 nếu S2 nằm trong nhóm ngang).
2. **Đề xuất train Phase E (self-play FFA) từ S2** nếu có ít nhất một điều sau:
   - Phần 1 ở 11×11: CI dưới của WR(S2) − WR(RB) ≤ 0.
   - Phần 2: có góc nào nằm ngoài 20–30% số trận có người thắng (gate cũ của Phase E), ở 9×9 hoặc 11×11.
3. **Đề xuất train Phase D (self-play pool) từ S2** nếu có ít nhất một điều sau:
   - Phần 3: G+s3 thua S2 về ActiveWR (CI trên của hiệu < 0).
   - G+s3 có WR < 60% trong một cặp đối đầu (ngưỡng gate Phase D của PROJECT_SPEC).
   - Luật Phase D của PDE đã kích hoạt.
4. Không điều kiện nào thỏa thì ghi nhận các agent hiện tại dùng được cho D/E, không train thêm.

Train (nếu có) cần spec riêng và người dùng duyệt; điểm xuất phát là S2, không phải G+s3, vì search quá chậm để train.

## Giới hạn đã biết

- 200 trận cho mỗi cấu hình FFA, nên CI khoảng ±7–10pp.
- Một seed. S2 chưa được Claude kiểm định lại tại thời điểm viết spec. Khi có file, kiểm định SD8A trước rồi mới chạy PDE2.
- Kết quả FFA hỗn hợp phụ thuộc vào ai ngồi cùng bàn. Thứ hạng chỉ đúng với bộ 4 agent này.
- Đối thủ CNN không thích nghi (đông cứng). Search giả định đối thủ là RuleBot.
