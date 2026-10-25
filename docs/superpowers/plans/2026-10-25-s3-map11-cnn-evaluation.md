# S3-11 CNN Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chạy benchmark2400trận CNN11×11 theo yêu cầu và kiểm tra45phút/lần.

**Architecture:** Runner mới sử dụng evaluator/opponent_fn, loader11×11 và helper thống kê/atomic đã có; registration+shard+lock giữ provenance và resume an toàn. Không sửa module đã khóa của run trước.

**Tech Stack:** Python/PyTorch/NumPy/unittest/progress.py, Windows native multiprocessing, app heartbeat.

**Spec:** `docs/superpowers/specs/2026-10-25-s3-map11-cnn-evaluation.md`.

## Global Constraints

- Hai learner S3_11/S3, pool S3/S2/G,6cặp,400trận/cặp,11×11/max500/E475-25-40, lấy mẫu+mask không search/RuleBot/gradient.
- Source/checkpoint gốc giữ nguyên; làm in-place bằng file mới trong branch hiện tại, không Git mutation/commit/push.
- Chỉ native background jobs; lịch kiểm tra45phút, không poll. Seed mới/marker/raw/provenance phải khóa trước run.

## Review Focus

- Cả learner và opponent phải nhận11×11 và đúng agent_id khi đổi góc.
- Corrupt/partial/duplicate seed-corner shard không được skip và không được ghi đè.
- Marker/lock/rerun/resume phải ngăn chạy chồng hoặc mở test lần nữa.
- Macro CI phải resample200bàn sau trung bình3opponent, không pseudo-replicate.
- Bao gồm mọi dependency/hash, config và4checkpoint; báo cả hòa/loss/tự nổ, không tự promote.

## Task1: Runner/test/guide

**Files:** tạo `probe_s3_map11_cnn.py`, `test_probe_s3_map11_cnn.py`, `S3_MAP11_CNN_GUIDE.md`; nhật ký/tracker.

**Interfaces:** API/CLI trong spec; JSON registration/marker/shard/summary là hợp đồng giữa runner, test và heartbeat.

- [x] Independent test-author tạo matrix, runnable tests và RED; giữ nguyên suite được giao.
- [x] Implement runner nhỏ dùng helper có sẵn; GREEN32/32targeted, suite độc lập26test giữ SHA.
- [x] Native smoke2worker24game PASS; review fixes RED→GREEN. Suite rộng518test/8lỗi fixture cũ/1skip/0assertion failure,32test mới PASS.
- [x] Ghi guide/journal/tracker, source cũ và4checkpoint giữ nguyên SHA.

## Task2: Đăng ký/chạy/check45phút

**Files:** artifact run mới `runs/s3-map11-cnn-20261025-v1/`, không sửa checkpoint.

- [x] Kiểm seed/source/checkpoint còn fresh, init registration trước data.
- [x] Lưu AUTORUN/state và tạo heartbeat45phút `s3-11-u-cnn-ki-m-tra-45-ph-t` trong chat này.
- [x] Launch full native/hidden lúc01:39:46ngày25/10, PID26792/interpreter26232, marker/hash/start đã xác minh. Lưu PID/start/log/launch snapshot; không restart.
- [x] Heartbeat25/10 xác thực summary/raw120shard/2400game/hash bằng publicAPI, chặn chơi thêm; không cần resume. MacroActive+3.83ppCI[1.92,5.83] chưa đạt gate≥5pp; FINAL_REPORT/RESULTS/IMPROVEMENTS/haiPLAN/INSIGHTS/seedledger cập nhật, state completed_no_qualified_gain. Kết thúc và gỡ lịch45phút.

## Execution ruling

Người dùng đã yêu cầu thực hiện các bước và giữ cadence45phút; thực thi trực tiếp, không hỏi lại thiết kế đã chọn. Không có thay đổi production ngoài runner mới. Phương án khác là sửa runner9×9 hoặc evaluator lõi; không chọn vì làm stale source hash các run đã hoàn tất.
