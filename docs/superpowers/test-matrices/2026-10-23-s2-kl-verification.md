# S2-KL implementation verification — 23/10/2026

Status: FINAL_REVIEW_AND_REGRESSION_PENDING. Chưa chạy/reserve pilot full. Hướng dẫn: `S2_KL_GUIDE.md`.

## Phạm vi và provenance

Plan được người dùng duyệt từ commit `ed7c749`; Native execution trên branch `Loc_s2-kl-retention`. Năm module/runner mới tái dùng nguyên trainer, KL/reference và full-state hiện có. Không thêm dependency, không sửa D-S2/engine/RuleBot hoặc checkpoint gốc.

Input S2 SHA256 `d222d81f86e5188951c1327f60cdb46ec770e5e1ba060e42c1f24efbfe8f6814`; sample manifest `9b36b8e07ed8f65175e5c9fbf53a18eca16ae791f0073d19730b28e75529e333`. File JSON kèm theo giữ các hash bảo vệ, lệnh/test counts, lỗi baseline và bằng chứng smoke/performance.

## Acceptance độc lập và RED/GREEN

Tác giả acceptance nhận riêng frozen requirements/public contracts/current code, không nhận implementation plan. Commit `58e5321` giữ packet/matrix/tests ban đầu. RED:40methods,39assertion failure records kể cả subtest,0errors,1CUDA opt-in skip; hai method toán KL/observer đã có sẵn pass và chỉ được ghi là regression.

Một fixture JSONL ban đầu bị ghi pretty JSON nhiều dòng và từ chối do parse error. Tác giả độc lập đã chứng minh lỗi fixture, sửa đúng một nhánh thành JSONL hợp lệ và tăng assertion thành overlap. Bản gốc còn ở `58e5321`; matrix acceptance ghi proof/old/newSHA. Không thay requirements hoặc làm yếu assertion. Các test còn lại và packet giữ SHA ban đầu.

| Check | Kết quả đã đo |
|---|---|
| T1 protocol | 15/15 PASS,26.999s |
| T2/T3 CPU | 20methods PASS,34.807s,1CUDA opt-in skip |
| T2 actual CUDA | 8/8 PASS,32.617s |
| T4 CLI journey | 5/5 PASS,132.306s |
| Acceptance actual CUDA | 40/40 PASS,199.804s,không skip |
| Integration cả S2-KL và D-S2/state/SD8 | 119/119 PASS,468.573s; bật cả hai cờ CUDA |
| Seed supporting regressions | RED4methods/2assertion failures/0errors → protocol+supporting21/21 PASS,28.720s |
| Acceptance cuối +8 supporting regressions | 48/48 PASS,215.774s,actual CUDA,không skip |

Bằng chứng raw giữ tại `runs/s2-kl-code-verification-20261023/`. Các regression bổ sung thuộc implementer, tách khỏi acceptance đã freeze.

## Full regression

Baseline trước production: `venv\Scripts\python.exe -m unittest discover -p "test_*.py" -q`,535tests,419.283s,20errors/1failure/2skips,exit1. Danh tính từng lỗi nằm trong JSON kèm theo và `baseline-summary.json`; nhiều lỗi do Windows sandbox/system Temp atomic replace, loader thiếu module lịch sử và PDE assertion lịch sử. Không coi toàn suite xanh hoặc sửa lỗi ngoài phạm vi.

Lượt581tests sau seed fixes:643.264s,20errors/1failure/3skips; so chính xác tên qualified error/failure với baseline, không có tên mới. Sau đó tìm/sửa thêm lỗi mất alias endpoint khi dedupSHA và malformed seed count bị coi là1board, có RED riêng rồi48/48 GREEN. Full regression lặp trên source cuối đang chạy trong `full-regression-final.log`; không dùng lượt581 để khẳng định source sau đó đã qua toàn suite.

## Smoke và hiệu năng

Smoke CPU và CUDA thực qua frozen CLI, workers2: cả hai PASS đủ6stage. Mỗi smoke4episode/arm,max20bước,5uniqueSHA,5diagnostics,20/20game,5120decision+5120anchor rows. Cả hai SMOKE_ONLY/complete, không test marker/reservation full; workerCPU1thread/from snapshot, ZIPCRC và protected SHA trước-sau đều hợp lệ. Exact resume CPU/CUDA trước/sau episode commit cho cả hai arm nằm trong48tests cuối. CON redirect check qua TTY hiển thị0/2→2/2, redirected log chỉ có stdout marker.

| Stage | CPU CLI wall s | CUDA CLI wall s | CPU process peak MiB | CUDA process peak MiB |
|---|---:|---:|---:|---:|
| register | 3.07 | 2.99 | 593.7 | 593.4 |
| train control | 6.26 | 7.76 | 625.0 | 1652.1 |
| train anchor | 6.26 | 7.56 | 622.1 | 1654.6 |
| diagnose | 21.31 | 21.76 | 637.9 | 636.9 |
| validate | 13.33 | 14.31 | 628.2 | 629.9 |
| package | 11.83 | 14.31 | 627.5 | 628.9 |

Worker peaks CPU613.0/614.8MiB, CUDA-run614.9/614.9MiB (validation vẫnCPU). Smoke có overlap với bounded acceptance suite; đây là thời gian quan sát dưới tải đó, không phải benchmark độc lập. Không suy GPU nhanh/chậm hơn từ4episode ngắn có startup.

Stage log đo thời gian sau import và peak RSS từng process; driver đo walltime ngoài CLI. Worker có PID/peak RSS riêng. Không cộng các peak độc lập thành RAM đỉnh toàn cây. Train20bước không đủ suy ETA500bước; full tối đa1000episode/500000learner steps. Validation tối đa1400game/700000env steps, chưa có ETA đại diện. Offline cùng2048state/model, có thể ngoại suy tối đa7SHA với giới hạn startup/I/O/runtime.

Ngoại suy đơn giản offline7/5model từ21–22s là khoảng30s trên máy/runtime/tải tương tự, chỉ để dự trù; phần bootstrap/report/startup không tuyến tính hoàn toàn. Train và validation chưa có ETA full đáng tin cậy từ smoke này.

## Audit seed thật trước full

Audit chỉ đọc, chưa đăng ký output hoặc reservation: REJECTED theo fail-closed. `runs/sd6-diagnostics-6600/overrides.json` có seed=null; hai `runs/t2-{full,smoke}-20261014-060320/replay_cases.json` lớn hơn32MiB cần seed index được kiểm chứng. Metadata/ranges chưa đủ chứng minh sạch để launch full; không bỏ qua hoặc sửa log cũ. Đây là prerequisite về lịch sử, không phải kết quả chất lượng hay lỗi smoke.

## Rulings khi thực thi

1. Dùng workspace/branch đã được duyệt, chỉ stage file thuộc S2-KL; giữ UI/PDE/user changes. Nếu sai có thể lẫn thay đổi ngoài scope; giảm rủi ro bằng explicit paths và hash bảo vệ.
2. Thay bash workflow scripts bằng PowerShell/Python tương đương trên Windows. Nếu sai có thể lệch bookkeeping; đối chiếu ledger với từng checkbox của plan.
3. Chạy T2 core rồi T3 offline trước khi chốt T2 vì frozen acceptance gọi diagnose giữa stop/resume. Commit T2/T3 chung; chi phí là diff qua hai task lớn hơn, giữ log/suite riêng để truy vết.
4. Cho tác giả độc lập sửa fixture JSONL kỹ thuật sai và tăng assertion overlap; giữ bản gốc/hashes/proof. Nếu sai có thể ảnh hưởng tính độc lập; implementer không sửa acceptance và mọi hash khác được đối chiếu.

## Review cuối

Đang chờ reviewer độc lập trên toàn range từ `ed7c749`, tập trung registry/UI metadata, reference sau resume, locked samples/ties/held-out, CI micro, stale projection và endpoint500. Chưa merge/push hoặc mở full pilot.
