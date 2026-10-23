# SD9.0 — search ba nước trên S2 so với trên G (probe validation, không train)

Ngày: 22/10/2026. Người dùng đồng ý hướng "chưng cất search3 vào CNN" sau PDE2. Đây là bước 1. Luật dưới đây chốt trước khi chạy. Runner: `probe_s2_search3.py`.

## Câu hỏi

S2 (SD8A, CNN thuần, ActiveWR 83.0 vs RuleBot trên test 48000) mạnh hơn G rất nhiều khi chơi một mình. Câu hỏi:
- Gắn `kill_search.search_action3` lên S2 có mạnh hơn G + search3 (94.0 trên test 49000) không?
- Có giảm hòa khi đấu CNN không?

Kết quả quyết định thầy của SD9 (chưng cất search3 vào CNN, train từ S2).

## Thiết kế

- Learner:
  - G + search3: G `runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c…c7ac`.
  - S2 + search3: S2 `runs/sd8a-20261020/train/epoch_16.pt`, SHA `d222d81f…6814`.
  - Search mặc định, mô phỏng đối thủ bằng RuleBot như SD8B. CNN lấy mẫu policy.
- Đối thủ: RuleBot, S2, G.
- Bàn: 9×9, 1v1, lịch E 475/25/40, max_steps 500. Seed **validation 21000**, 50 bàn × 2 góc = 100 trận mỗi nhánh, ghép cặp theo bàn giữa hai learner.
- Tái dùng shard đã có (cùng code `kill_search.py` SHA `7c8b9f49…7991`, cùng bàn, cùng cách chia 5 bàn mỗi shard):
  - G + search3 vs RuleBot từ SDM2 (`rulebot_vs_rulebot`).
  - G + search3 vs G từ SDM (`rulebot_vs_G`).
- Bootstrap theo bàn 10 000 lần, seed 20261014.

## Luật (chốt trước)

Hiệu (S2 + search3) − (G + search3), theo từng đối thủ:

1. **Thầy cho SD9:**
   - S2 + search3, trừ khi ActiveWR vs RuleBot giảm có ý nghĩa (CI trên < 0); khi đó thầy là G + search3.
   - Lý do chọn S2 + search3 khi hòa: SD9 train từ S2, nên thầy dùng cùng nền thì nhãn chỉ sửa đúng chỗ S2 sai.
2. **Ứng viên agent mạnh nhất mới:**
   - Điều kiện: ActiveWR tăng ≥ +5pp với CI dưới > 0 ở ít nhất một đối thủ, và không đối thủ nào có CI trên < 0.
   - Khi đạt: đề xuất xác nhận trên seed test mới 57000. Seed này dùng một lần, nên chỉ chạy khi người dùng đồng ý.
   - Khi không đạt: G + search3 vẫn là agent mạnh nhất.
3. Báo kèm: WR, hòa, hòa vì hết 500 bước, thua, tự nổ, s/trận.

## Giới hạn

- Một seed validation, 100 trận mỗi nhánh. Đấu RuleBot thì cả hai gần trần (~96%), nên khó tách.
- Search vẫn giả định đối thủ là RuleBot (SDM2: đổi mô hình đoán làm hỏng đấu RuleBot).
