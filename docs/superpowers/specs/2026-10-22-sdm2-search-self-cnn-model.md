# SDM2 — kill-search đoán đối thủ bằng CNN G của mình (probe validation, không train)

Ngày: 22/10/2026. Người dùng đồng ý đề xuất sau SDM. Luật dưới đây chốt trước khi chạy. Runner: `probe_search_cnn_model.py` (mở rộng từ SDM).

## Câu hỏi

SDM (validation 21000) cho thấy khi search biết chính xác CNN của đối thủ (oracle), ActiveWR tăng khi đấu G (+14.0 [+3, +25]) nhưng không rõ khi đấu S (+4.0 [−6, +15]). Oracle không dùng được khi đấu thật vì thường không biết checkpoint của đối thủ.

Phiên bản dùng được là search **luôn đoán đối thủ bằng CNN G của mình** (mô hình `self`), bất kể đối thủ thật là ai. Hai câu hỏi:
- Mô hình này có giúp khi đấu CNN lạ (S, F) không?
- Nó có làm hỏng mức ActiveWR ~94 khi đấu RuleBot không?

## Thiết kế

- Learner: G (`runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c…c7ac`) + `kill_search.search_action3`, mặc định không đổi.
- Hai mô hình đối thủ trong search:
  - `rulebot`: như cũ.
  - `self`: argmax policy đã mask của G (`cnn_model`, tất định).
- Đối thủ thật: RuleBot, S (`8f7116d2…fd4a`), F (`0315a56f…1483`). CNN lấy mẫu policy.
- Bàn: 9×9, 1v1, lịch E 475/25/40, max_steps 500. Seed **validation 21000**, 50 bàn × 2 góc = 100 trận mỗi nhánh. Hai mô hình ghép cặp theo bàn.
- Tái dùng shard `rulebot_vs_S` của SDM (`runs/sdm-21000/shards`). Đó là cùng code, cùng bàn và cùng máy, nên không chạy lại.
- Bootstrap theo bàn 10 000 lần, seed 20261014.

## Luật (chốt trước)

Lấy hiệu (self − rulebot) cho mỗi đối thủ:
- **Giữ mô hình `self`** khi thỏa cả hai điều kiện:
  - ActiveWR tăng ≥ +5pp với CI dưới > 0 ở ít nhất một trong S, F.
  - Không đối thủ nào (kể cả RuleBot) có ActiveWR giảm với CI trên < 0.
- Khi giữ: đề xuất xác nhận `self` vs `rulebot` trên seed test mới 52000. Seed này dùng một lần, nên chỉ chạy khi người dùng đồng ý.
- **Không giữ** khi không thỏa luật trên: search vẫn dùng RuleBot. Hướng tiếp theo là Phase D, quyết sau khi có SD8A/PDE2.
- Báo kèm: WR, hòa, hòa vì hết 500 bước, thua, tự nổ, s/trận.

## Giới hạn

- Một seed validation, 100 trận mỗi nhánh, nên CI khoảng ±10–14pp.
- Mô hình `self` là argmax, còn đối thủ thật lấy mẫu.
- Shard tái dùng chỉ hợp lệ khi `kill_search.py` không đổi kể từ SDM. Runner ghi SHA của `kill_search.py` vào `opened.json`.
