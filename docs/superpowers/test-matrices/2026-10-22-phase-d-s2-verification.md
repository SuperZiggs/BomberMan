# D-S2 — bằng chứng kiểm chứng, 22/10/2026

Được người dùng duyệt implementation plan, triển khai Native từ `5f8ed19`. Đây là kiểm chứng code; chưa có kết quả thí nghiệm full D-S2.

- Requirements và 62 acceptance được tác giả độc lập khóa trước implementation tương ứng; commit `30083da`. Hash/RED/giới hạn nằm trong `2026-10-22-phase-d-s2.md`. Implementer không sửa bốn file acceptance.
- Protocol/training: 21 test OK, 1 CUDA opt-in skip (54,262 s). CUDA thật + regression state: 13 test OK (59,173 s); continuous/resume khớp model/Adam/scheduler/RNG/context.
- Selection/CI/gates/load: 21 test OK (2,096 s).
- Full acceptance + hai test recovery đầu: 64 test OK, 1 CUDA skip (291,096 s).
- Recovery mở rộng: 8 test OK (21,098 s), sau RED đúng hành vi ở JSONL overlap, PENDING sau publish, partial prepare, worker CPU flags, history CRLF, selection projection và package integrity. Test reserve trước publish đã qua như một regression.
- Regression state/SD8/PDE: 42 test, một failure giống baseline, tại `test_pde_evaluator_contracts.py:155`; test xóa `_` trước khi tìm `test_opened`.
- Baseline tracked trước production: 431 test, 20 error, 1 skip (155,311 s). Mtime log hoàn tất trước production write đầu tiên; không coi lỗi thiếu behavior mới là baseline.

Full discover: **526 test, 20 error, 1 failure, 2 skip, 471,437 s**. So sánh tên từng error với baseline: không thêm hoặc mất error nào (20/20 khớp). Failure PDE trùng batch baseline riêng. Hai skip gồm skip cũ và CUDA opt-in. Bộ acceptance D-S2 nằm trong discover không có failure/error. Test one-ULP bổ sung sau collection: 1 test OK (1,318 s).

Hai lượt CLI smoke `runs/d-s2-code-smoke-20261022-cpu` và `...-cuda`: register → train control → train pool → validate → package, exit0; 8 episode/160 learner steps và 40 trận validation mỗi lượt (SHA trùng được reuse). Cả hai **VALIDATION_STOP**, không có `test_opened.json`. Đã kiểm mọi payload worker: CPU/threads1/source_root đúng `output/sources`. Success path mở/resume test được kiểm bằng fixture độc lập đủ gate, không ép mở seed khoa học. CUDA resume rerun trên source cuối:1 test OK13,674s.

| Stage | CPU wall s | CUDA wall s | Peak RSS tree MiB CPU / CUDA |
|---|---:|---:|---:|
| Register |3,27|4,28|577 /589|
| Train control 4ep |7,30|9,84|1199 /2199|
| Train pool 4ep |7,57|9,84|1195 /2201|
| Validation40games,2workersCPU |12,12|13,84|2379 /2374|
| Package |11,61|12,33|1165 /1164|

RSS lấy mẫu250ms theo process tree, có đếm lặp shared pages, không gồm tiến trình ngoài cây. Theo callback từ bước đầu đến episode commit (source frozen, thêm output profile riêng):CPU6,35ms/learner-step,CUDA15,44ms. Dự phóng tuyến tính với trần2triệu learner steps:3,5h CPU /8,6h GPU; evaluation tối đa3600/5400trận trên2worker CPU khoảng1,5–1,7h /2,2–2,6h. Đây **không phải ETA hoặc cận trên**: smoke chỉ20bước, chưa đo trạng thái cuối trận hoặc I/O history lớn; episode thật có thể kết thúc sớm, teacher cuối trận có thể tốn hơn. GPU không được chứng minh nhanh hơn CPU ở workload này. File JSON kèm raw stage/profile/runtime/source hashes được nhúng vào registration mới; snapshot registration đã tạo không bị viết lại.

Review độc lập cuối đang thực hiện.

## Các quyết định trong execution ledger

1. Native worktree bị sandbox chặn ghi ngoài workspace: triển khai trên nhánh hiện có `Loc/sd8-huong-a`, commit phạm vi rõ và giữ mọi PDE file có trước. Chi phí nếu sai: thay đổi local có thể tương tác với công việc khác trước commit.
2. Git Bash helper bị MSYS chặn: dùng Python trích task briefs và ghi cùng ledger/command/result. Chi phí nếu sai: thiếu bookkeeping; đã đối chiếu task headings.
3. Implement từng phần chỉ sau slice acceptance đã khóa RED; test author tiếp tục phần khác với packet/original code, không đọc source mới hoặc plan. Chi phí nếu sai: ảnh hưởng tính độc lập; tác giả xác nhận không có đọc source mới.
4. Evaluation dùng `progress.bar` + public spawn `multiprocessing.Pool` vì `progress.parallel_map` đợi mọi future khi Ctrl+C. Pool terminate công việc chưa commit; atomic board shard là nguồn resume. Chi phí nếu sai: lỗi vòng đời worker/interruption; kiểm bằng atomic ledger và smoke spawn thật.

Các source hash của lượt mới luôn dùng raw bytes. Normalization LF/CRLF chỉ phục vụ kiểm kê hai bản source lịch sử SD7.4 bị Git đổi xuống dòng; inventory vẫn ghi SHA raw và giải ra đúng dải 6700–6749. Không cập nhật registration cũ hoặc bỏ kiểm hash mới.
