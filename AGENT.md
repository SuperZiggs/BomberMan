# AGENT.md

Quy định cho mọi coding agent (Codex, Claude, …) làm việc trong repo này. Hướng dẫn đầy đủ (môi trường, kiến trúc, quy ước seed và checkpoint) nằm ở `CLAUDE.md`.

## Ghi nhật ký (bắt buộc)

- Mọi task, thay đổi code hay thí nghiệm đều được ghi vào **cả** `RESULTS.md` (kết quả) và `IMPROVEMENTS.md` (đã đổi gì, vì sao, kiểm thử ra sao). Áp dụng cả cho smoke, lượt bị ngắt, gate FAIL và các lượt chỉ sửa test.
- Viết bằng tiếng Việt và ghi đúng sự thật. FAIL, incomplete hay chưa chạy full đều phải ghi rõ.
- Cập nhật checkbox trong `PLAN_SUDDEN_DEATH.md` / `PLAN_BOMBERMAN.md`. Có insight mới thì cập nhật `INSIGHTS.md`.

## Thông số đầy đủ (người dùng yêu cầu 20/10/2026)

Mỗi thí nghiệm trong `RESULTS.md` và `IMPROVEMENTS.md` phải có một dòng `**Thông số:**` đủ để người khác chạy lại:

- **Train:**
  - checkpoint xuất phát (đường dẫn và SHA rút gọn), phase;
  - số episode mỗi nhánh, train seed;
  - lr và các hệ số khác mặc định;
  - bàn / số agent / max_steps, lịch sudden death;
  - thiết bị (CPU/CUDA), thời gian chạy.
- **Đánh giá:**
  - model hoặc agent, đối thủ;
  - seed và dải bàn, số trận mỗi model (số bàn × số góc);
  - max_steps, bàn / số agent, lịch SD (E = 475/25/40, off, …);
  - cách chọn nước (lấy mẫu / argmax / search một, hai hay ba nước);
  - số lần bootstrap.
- **Môi trường:** ai chạy (người dùng / bạn / agent), máy, phiên bản torch, số tiến trình song song.
- Thông số không có trong `runs/` thì ghi "không lưu lại", không đoán.

Ví dụ:

> **Thông số:** G (`runs/sd75-20261018/full/best_phaseC.pt`, SHA `bb0c775c…c7ac`) + `search_action3` vs RuleBot; test seed 49000, bàn 49000–49199 × 2 góc = 400 trận/agent; 9×9, 2 agent, max_steps 500; lịch E 475/25/40; policy lấy mẫu; bootstrap ghép cặp theo bàn 10 000 lần (seed 20261014); người dùng chạy trên máy chính, torch 2.5.1, 5 tiến trình.

## Quy ước khác

- Mỗi seed test chỉ dùng một lần. Luật chọn model chốt trước khi đọc kết quả. Không ghi đè checkpoint gốc.
- Lệnh đưa cho người dùng viết theo cú pháp cmd (`> log 2>&1`). Script dài có thanh progress (`progress.py`).
