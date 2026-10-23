# SDM — kill-search mô phỏng đối thủ bằng CNN (probe, không train)

Ngày: 20/10/2026. Người dùng yêu cầu sau PDE: "thử search mô phỏng đối thủ bằng CNN". Luật dưới đây chốt trước khi chạy. Runner: `probe_search_cnn_model.py`.

## Câu hỏi

PDE (seed 50000) cho thấy G + search3 đấu CNN đông cứng chỉ đạt WR 19–44, hòa 50–70% (gần hết là hết 500 bước), trong khi đấu RuleBot đạt ActiveWR 94. Search mô phỏng đối thủ bằng RuleBot. Nếu cho search mô phỏng đúng CNN của đối thủ, search có giết được nhiều hơn và hòa ít hơn không?

- Nếu có: nguyên nhân chính là search đoán sai đối thủ. Có thể chưa cần train Phase D cho agent có search.
- Nếu không: CNN phòng thủ tốt hơn RuleBot, hoặc search ba nước không đủ sâu trước đối thủ CNN. Điều này ủng hộ đề xuất train Phase D.

## Thiết kế

- Learner: G (`runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c…c7ac`) + `kill_search.search_action3`. Mặc định của search không đổi.
- Hai mô hình đối thủ trong search:
  - `rulebot`: như cũ.
  - `oracle`: CNN thật của đối thủ, lấy argmax của policy đã mask. Đây là mô hình tất định, không dùng RNG.
- Đối thủ thật lấy mẫu policy, nên mô hình argmax vẫn lệch một phần. Đây là giới hạn đã biết.
- Hook: `kill_search.OPPONENT_MODEL`, mặc định None (RuleBot). Đã kiểm bit-exact với bản cũ trên 4 trận (seed 6600, 120 bước).
- Đối thủ thật: S (khó nhất trong PDE) và G (gương).
- Bàn: 9×9, 1v1, lịch E 475/25/40, max_steps 500. Seed **21000 (validation)**, không phải seed test. 50 bàn × 2 góc = 100 trận mỗi nhánh, ghép cặp theo bàn giữa hai mô hình.
- Bootstrap theo bàn 10 000 lần, seed 20261014.

## Luật (chốt trước)

Với mỗi đối thủ, lấy hiệu (oracle − rulebot):
- **Mô hình CNN giúp**: ActiveWR tăng ≥ +5pp với CI dưới > 0 ở ít nhất một đối thủ, và không đối thủ nào có ActiveWR giảm với CI trên < 0. Khi đó đề xuất xác nhận trên seed test mới 52000 và đề xuất dùng mô hình CNN trong search thay vì train Phase D ngay.
- **Không giúp**: CI của ActiveWR chứa 0 hoặc âm ở cả hai đối thủ. Khi đó ghi nhận nguyên nhân không chỉ là đoán sai đối thủ, và giữ đề xuất train Phase D của PDE.
- Báo kèm: WR, hòa, hòa vì hết 500 bước, tỉ lệ thua, tự nổ, thời gian mỗi trận.

## Giới hạn

- Đây là oracle: search biết chính xác checkpoint của đối thủ. Thực tế không phải lúc nào cũng biết.
- 100 trận mỗi nhánh, nên CI khoảng ±8–10pp.
- Một seed validation.
