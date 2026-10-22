# D-S2 — bằng chứng kiểm chứng, 22/10/2026

Được người dùng duyệt implementation plan, triển khai Native từ `5f8ed19`. Đây là kiểm chứng code; chưa có kết quả thí nghiệm full D-S2.

- Requirements và 62 acceptance được tác giả độc lập khóa trước implementation tương ứng; commit `30083da`. Hash/RED/giới hạn nằm trong `2026-10-22-phase-d-s2.md`. Implementer không sửa bốn file acceptance.
- Protocol/training: 21 test OK, 1 CUDA opt-in skip (54,262 s). CUDA thật + regression state: 13 test OK (59,173 s); continuous/resume khớp model/Adam/scheduler/RNG/context.
- Selection/CI/gates/load: 21 test OK (2,096 s).
- Full acceptance + hai test recovery đầu: 64 test OK, 1 CUDA skip (291,096 s).
- Recovery mở rộng: 8 test OK (21,098 s), sau RED đúng hành vi ở JSONL overlap, PENDING sau publish, partial prepare, worker CPU flags, history CRLF, selection projection và package integrity. Test reserve trước publish đã qua như một regression.
- Regression state/SD8/PDE: 42 test, một failure giống baseline, tại `test_pde_evaluator_contracts.py:155`; test xóa `_` trước khi tìm `test_opened`.
- Baseline tracked trước production: 431 test, 20 error, 1 skip (155,311 s). Mtime log hoàn tất trước production write đầu tiên; không coi lỗi thiếu behavior mới là baseline.

Full discover, smoke hai worker và review cuối đang được hoàn tất; số cuối được bổ sung bên dưới.

## Các quyết định trong execution ledger

1. Native worktree bị sandbox chặn ghi ngoài workspace: triển khai trên nhánh hiện có `Loc/sd8-huong-a`, commit phạm vi rõ và giữ mọi PDE file có trước. Chi phí nếu sai: thay đổi local có thể tương tác với công việc khác trước commit.
2. Git Bash helper bị MSYS chặn: dùng Python trích task briefs và ghi cùng ledger/command/result. Chi phí nếu sai: thiếu bookkeeping; đã đối chiếu task headings.
3. Implement từng phần chỉ sau slice acceptance đã khóa RED; test author tiếp tục phần khác với packet/original code, không đọc source mới hoặc plan. Chi phí nếu sai: ảnh hưởng tính độc lập; tác giả xác nhận không có đọc source mới.
4. Evaluation dùng `progress.bar` + public spawn `multiprocessing.Pool` vì `progress.parallel_map` đợi mọi future khi Ctrl+C. Pool terminate công việc chưa commit; atomic board shard là nguồn resume. Chi phí nếu sai: lỗi vòng đời worker/interruption; kiểm bằng atomic ledger và smoke spawn thật.

Các source hash của lượt mới luôn dùng raw bytes. Normalization LF/CRLF chỉ phục vụ kiểm kê hai bản source lịch sử SD7.4 bị Git đổi xuống dòng; inventory vẫn ghi SHA raw và giải ra đúng dải 6700–6749. Không cập nhật registration cũ hoặc bỏ kiểm hash mới.
