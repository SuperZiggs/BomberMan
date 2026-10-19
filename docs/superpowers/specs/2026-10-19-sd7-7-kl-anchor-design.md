# SD7.7 — Buffer khoảnh khắc bom + phạt KL về G

Ngày: 19/10/2026. Người dùng chọn "sửa tác dụng phụ trước" sau SD7.6-probe. Luật dưới đây chốt **trước** khi chạy.

## Bằng chứng (RESULTS SD7.6-probe và đo thêm trên 40 trận G, seed 6600)

- Nhánh buffer SD7.6: P(BOMB | BOMB hợp lệ) **0.943**, so với G 0.637 và control 0.656. KL(π‖G) trung bình trên mọi bước 0.049, control 0.003.
- Lý do: bộ lọc hẹp chỉ chọn trạng thái có BOMB hợp lệ và thường BOMB là nước giết, nên dữ liệu một chiều. Ở trạng thái không có nước giết, mọi lần chơi lại đều ra 0, nên group RL không sinh tín hiệu ngược. Policy học thành "cứ được đặt bom là đặt".

## Thay đổi

- `train(anchor_coef=β)`: cộng β · mean_t KL(π_θ(·|s_t) ‖ π_ref(·|s_t)) trên mọi bước learner của episode (phân phối đã che mask). π_ref là bản đông cứng của trọng số warm start (`resume_from` = G), tạo trước khi khôi phục training_state nên resume vẫn đúng. β = 0 giữ nguyên hành vi cũ, bit-exact.
- Còn lại như SD7.6 buffer (K 16, H 20, coef 4, narrow, buffer 1000, chỉ bước đầu, config control SD5.1, từ G).

## Hiệu chỉnh β (trước probe; seed chẩn đoán 6600, không phải validation/test)

- Bốn lượt ngắn 150 ep từ G, train seed 2026101701, β ∈ {0, 1, 4, 16}.
- Đo trên 40 trận G chơi (seed 6600, lịch E): P(BOMB | hợp lệ), KL(π‖G), và P(tập nước giết) ở trạng thái hẹp.
- **Luật chọn:** trong các β > 0 có P(BOMB | hợp lệ) ≤ P_G + 0.05, chọn β có P(nước giết) ở trạng thái hẹp cao nhất (hòa thì chọn β nhỏ hơn). Không β nào thỏa thì dừng và báo lại.
- Nếu β = 0 ở 150 ep chưa thấy lan bom (P(BOMB | hợp lệ) ≤ P_G + 0.05), ghi lại rằng 150 ep chưa đủ để hiệu chỉnh và dừng để người dùng quyết.

## Probe SD7.7 (1000 ep)

- Một nhánh `anchor` (buffer + β đã chọn), 1000 ep, không validation khi train. Đối chứng dùng lại `runs/sd76-20261019/control/final_phaseC.pt` (cùng config, cùng train seed, cùng điểm xuất phát G).
- Cơ chế: seed chẩn đoán mới 6900–6949 (100 trận, G chơi), trạng thái hẹp, K 8 × H 20, tỉ lệ giết và P(nước giết). Validation 21000 (400 trận, lịch E) cho anchor; control lấy lại kết quả SD7.6.
- **Luật đạt (H1⁗):** anchor − control tỉ lệ giết ≥ +10pp, CI95 dưới > 0; chết lúc chơi lại ≤ control + 1pp; trên validation: kills/trận anchor ≥ control, bom/trận anchor ≤ control × 1.10, và ActiveWR anchor − control ≥ 0 (điểm ước lượng, bootstrap theo cặp bàn).
- Đạt → full 3000 ep + test một lần seed 46000, gate full − G ActiveWR ≥ +5pp, CI dưới > 0, tự nổ ≤ +1pp.

## Giới hạn

Một train seed; hiệu chỉnh β dựa trên 150 ep và 40 trận; validation 21000 đã được dùng ở SD7.6 cho control (dùng lại kết quả, không chạy lại).
