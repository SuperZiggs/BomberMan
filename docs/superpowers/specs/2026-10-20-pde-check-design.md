# PDE — agent mạnh nhất trong bối cảnh Phase D và Phase E (không train)

Ngày: 20/10/2026. Người dùng yêu cầu chạy agent mạnh nhất qua các phase sau Phase C (D, E), theo đề xuất "kiểm tra rẻ trước rồi mới train". Mọi luật dưới đây chốt trước khi chạy. Runner: `run_pde_check.py`.

## Bối cảnh

- Phase D/E cũ (04–05/10, code ban đầu, 8 kênh) báo PASS bằng WR/reward. Kiểm lại ngày 10/10: `best_phaseE_4a.pt` đấu 3 RuleBot có ActiveWR 0%.
- Từ đó mọi thí nghiệm ở Phase C (1v1, 9×9, RuleBot). Agent mạnh nhất hiện tại là G + `kill_search.search_action3`: test 49000 ActiveWR 94.0.
- Search mô phỏng mọi đối thủ bằng RuleBot. Chưa biết search có còn mạnh khi có nhiều đối thủ hơn (FFA) hoặc khi đối thủ không phải RuleBot (self-play) hay không.

## Thiết kế

Mọi nhánh: lịch E (475/25/40), max_steps 500, policy lấy mẫu, learner luân phiên mọi góc của mỗi bàn, bàn bắt đầu từ seed **50000** (seed mới, dùng một lần cho cả thí nghiệm).

| Bối cảnh | Bàn / số agent | Learner | Đối thủ | Số bàn × góc |
|---|---|---|---|---|
| E2 (Phase E 4 agent) | 11×11, 4 | RuleBot (mốc), `best_phaseE_4a.pt` cũ, G, G + search3 | 3 RuleBot | 40 × 4 = 160 trận/nhánh |
| E1 (Phase E 3 agent) | 9×9, 3 | RuleBot (mốc), `best_phaseE_3a.pt` cũ, G, G + search3 | 2 RuleBot | 60 × 3 = 180 trận/nhánh |
| D (Phase D self-play) | 9×9, 2 | G, G + search3 | CNN đông cứng: G, F, S, B, `best_phaseD.pt` cũ | 100 × 2 = 200 trận/cặp |

- G trên bàn 11×11 dùng trọng số nội suy (`smart_load_weights`) vì G được train trên 9×9. Đây là điều kiện bất lợi đã biết.
- Mốc "RuleBot ở ghế learner" là mức của một bot không học. Mức ngẫu nhiên công bằng giữa các ghế là 25% (4 agent) hoặc 33% (3 agent).
- Số trận nhỏ hơn 400 vì search3 trong FFA chậm: đo trên seed 6600 được ~190 s/trận ở E2 và ~124 s/trận ở E1, so với ~23 s ở 1v1. Ước tính ~20 giờ CPU, chạy 5 tiến trình.

## Báo cáo (mô tả, không gate)

So cặp theo bàn (mọi góc của một bàn đi cùng nhau), bootstrap 10 000 lần, seed 20261014. Các metric: ActiveWR, WR, hòa, thắng thụ động, tự nổ.
- E2, E1:
  - G + search3 − G;
  - G − mốc RuleBot;
  - G + search3 − mốc RuleBot;
  - G − checkpoint Phase E cũ.
- D: với từng đối thủ, G + search3 − G.
- Thêm ActiveWR / WR theo góc.

## Luật đề xuất bước 2 (train), chốt trước

Bước này không tự train. Chỉ đề xuất, và người dùng duyệt một spec riêng:
- **Đề xuất train Phase E (FFA 4 agent, 11×11) từ G** nếu ở E2, CI dưới của (G − mốc RuleBot) ActiveWR ≤ 0. Nghĩa là CNN không chuyển được sang FFA, và agent nhanh không search vẫn cần học FFA.
- **Đề xuất train Phase D (self-play pool) từ G** nếu có một đối thủ CNN mà:
  - (G + search3 − G) ActiveWR có CI dưới ≤ 0, nghĩa là search mất tác dụng khi đối thủ không phải RuleBot; hoặc
  - WR của G + search3 < 60%, là ngưỡng gate Phase D trong PROJECT_SPEC.
- Nếu không điều kiện nào thỏa: ghi nhận agent hiện tại đã dùng được cho D/E, không train thêm.

## Giới hạn đã biết

- Số trận mỗi nhánh 160–200, nên CI rộng khoảng ±7–10pp.
- Một seed; checkpoint cũ là 8 kênh của code ban đầu.
- Đối thủ CNN trong D là policy lấy mẫu, đông cứng, không thích nghi.
