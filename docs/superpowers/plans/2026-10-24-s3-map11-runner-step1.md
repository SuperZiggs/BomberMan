# S3 Map11 Step 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chuẩn bị runner riêng cho chưng cất S3 trên 11×11, chưa thực hiện thí nghiệm dài.

**Architecture:** Module mới dùng các helper thuần có sẵn, mạng 11×11 khởi tạo từ S3. Cấu hình được đăng ký trước; CLI tách từng stage, không thay source đã khóa của SD9.

**Tech Stack:** Python, PyTorch, NumPy, unittest, progress.py.

**Spec:** `docs/superpowers/specs/2026-10-24-s3-map11-runner-step1.md`.

## Global Constraints

- 11×11, 13 kênh, 2 agent, RuleBot; giữ nguyên source và checkpoint cũ.
- Chỉ thêm runner/test/guide; ghi nối nhật ký/tracker hiện có.
- Không chạy baseline/collect/train/test dài; seed test do người chạy khai báo ở init.
- Làm tại workspace hiện có để dùng checkpoint; chỉ thêm file riêng và nối docs, không chuyển nhánh hoặc tạo worktree.

## Review Focus

- Hardcode 9×9 còn lọt trong helper, collection hoặc evaluation.
- Resume nhận shard/config khác, hoặc thiếu source trong provenance.
- Seed test giao nhau hoặc smoke vô tình dùng seed test thật.
- Train mở lại dữ liệu cũ, sai shape hoặc cập nhật critic ngoài hợp đồng.
- Ghi đè checkpoint/run/selection; test chạy khi validation chưa cải thiện.

## Task 1: Runner và nghiệm thu bước 1

**Files:** tạo `run_s3_map11_distill.py`, `test_run_s3_map11.py`, `S3_MAP11_GUIDE.md`; cập nhật nhật ký/tracker.

**Interfaces:** API và CLI trong spec là hợp đồng giữa test độc lập, runner và các bước chạy sau.

- [x] Agent test đọc spec/current code, tạo matrix và acceptance suite; chạy RED trước production code.
- [x] Triển khai runner theo hợp đồng; dùng helper hiện có khi không cố định 9×9.
- [x] Chạy acceptance suite tới GREEN; không sửa yếu test độc lập.
- [x] Chạy regression và unittest toàn repo; ghi lỗi có sẵn nếu có.
- [x] Reviewer độc lập đọc source/test/guide, xử lý lỗi quan trọng.
- [x] Ghi kết quả, cập nhật tracker và bàn giao bước 1. Không tự chạy bước 2.

## Execution record

Người dùng đã yêu cầu thực hiện bước 1; thực thi trong chat hiện tại, không cần hỏi lại lựa chọn đã xác nhận. Không commit tự động. Các tùy chọn CLI cụ thể là quyết định triển khai trong phạm vi đã yêu cầu; seed và thời điểm chạy dài vẫn chưa được chọn.

Kết quả: acceptance33 độc lập giữ nguyên SHA, 2 regression provenance và 27 regression cũ đều PASS (targeted62/62); smoke2worker thật PASS. Full discovery486: 8 lỗi thiếu fixture cũ tái hiện riêng, 1skip, không có assertion failure. Source cũ và S3 không đổi; source closure được bổ sung theo reviewer sau supporting RED→GREEN. Chi tiết ở RESULTS `S3-MAP11-STEP1`, IMPROVEMENTS §91.
