# S2-KL implementation verification — 23/10/2026

Status: CODE_VERIFIED_WITH_KNOWN_BASELINE_FAILURES; full launch blocked by historical seed audit. Chưa chạy/reserve pilot full. Hướng dẫn: `S2_KL_GUIDE.md`.

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
| Acceptance trước review +8 supporting regressions | 48/48 PASS,215.774s,actual CUDA,không skip |
| Acceptance sau review | 50/50 PASS,235.902s,actual CUDA,không skip |
| Protocol và supporting sau audit inventory | 26/26 PASS,43.218s |
| S2-KL source cuối | 51/51 PASS trong full-root586tests,actual CUDA bật |

Bằng chứng raw giữ tại `runs/s2-kl-code-verification-20261023/`. Các regression bổ sung thuộc implementer, tách khỏi acceptance đã freeze.

## Full regression

Baseline trước production: `venv\Scripts\python.exe -m unittest discover -p "test_*.py" -q`,535tests,419.283s,20errors/1failure/2skips,exit1. Danh tính từng lỗi nằm trong JSON kèm theo và `baseline-summary.json`; nhiều lỗi do Windows sandbox/system Temp atomic replace, loader thiếu module lịch sử và PDE assertion lịch sử. Không coi toàn suite xanh hoặc sửa lỗi ngoài phạm vi.

Lượt581tests sau seed fixes:643.264s,20errors/1failure/3skips; so chính xác tên qualified error/failure với baseline, không có tên mới. Sau đó tìm/sửa thêm lỗi mất alias endpoint khi dedupSHA và malformed seed count bị coi là1board, có RED riêng rồi48/48 GREEN. Lượt583tests/631.627s tiếp tục giữ nguyên21qualified errors/failures. Sau review/P2 và audit inventory, full regression source cuối chạy trong `full-regression-reviewed.log` với S2_KL_ACCEPTANCE_CUDA=1; không dùng lượt581 để khẳng định source sau đó đã qua toàn suite.

Kết quả cuối sau toàn bộ fixes:586tests,681.386s,20errors/1failure/2skips,exit1; tên21qualified errors/failures trùng chính xác baseline, không phát sinh lỗi mới. Bộ S2-KL51/51PASS trong cùng lệnh, gồm exact-resumeCUDA bật qua S2_KL_ACCEPTANCE_CUDA=1; hai skip còn lại thuộc legacy. Gate đối chiếu và live input/source/runtime/smoke kiểm bằng `runs/s2-kl-code-verification-20261023/finalize.py` exit0. Đây là code verified với lỗi repo có sẵn, không phải tuyên bố toàn repo xanh.

## Smoke và hiệu năng

Smoke CPU và CUDA thực qua frozen CLI, workers2: cả hai PASS đủ6stage. Mỗi smoke4episode/arm,max20bước,5uniqueSHA,5diagnostics,20/20game,5120decision+5120anchor rows. Cả hai SMOKE_ONLY/complete, không test marker/reservation full; workerCPU1thread/from snapshot, ZIPCRC và protected SHA trước-sau đều hợp lệ. Exact resume CPU/CUDA trước/sau episode commit cho cả hai arm được kiểm trong bộ S2-KL gồm CUDA; source cuối đã kiểm51/51methods trong full-root. CON redirect check qua TTY hiển thị0/2→2/2, redirected log chỉ có stdout marker.

| Stage | CPU CLI wall s | CUDA CLI wall s | CPU process peak MiB | CUDA process peak MiB |
|---|---:|---:|---:|---:|
| register | 6.05 | 4.00 | 593.6 | 594.0 |
| train-control | 8.95 | 9.68 | 624.0 | 1673.2 |
| train-anchor | 7.90 | 9.37 | 621.1 | 1653.7 |
| diagnose | 25.11 | 24.86 | 638.2 | 637.3 |
| validate | 16.76 | 16.76 | 628.4 | 628.7 |
| package | 11.29 | 14.17 | 626.4 | 628.2 |

Worker peaks CPU614.6/614.9MiB, CUDA-run614.3/614.9MiB (validation vẫnCPU). Đây là smoke source cuối tại các output hậu tố `-final`; các lượt trước và hậu tố `-review` giữ làm lịch sử. Smoke có overlap với bounded supporting tests; đây là thời gian quan sát dưới tải đó, không phải benchmark độc lập. Không suy GPU nhanh/chậm hơn từ4episode ngắn có startup.

Stage log đo thời gian sau import và peak RSS từng process (RAM host, không phải VRAM); driver đo walltime ngoài CLI. Worker có PID/peak RSS riêng. Không cộng các peak độc lập thành RAM đỉnh toàn cây. Train20bước không đủ suy ETA500bước; full tối đa1000episode/500000learner steps. Validation tối đa1400game/700000env steps, chưa có ETA đại diện. Offline cùng2048state/model, có thể ngoại suy tối đa7SHA với giới hạn startup/I/O/runtime.

Ngoại suy đơn giản offline7/5model từ khoảng25s là khoảng35s trên máy/runtime/tải tương tự, chỉ để dự trù; phần bootstrap/report/startup không tuyến tính hoàn toàn. Train và validation chưa có ETA full đáng tin cậy từ smoke này.

## Audit seed thật trước full

Audit chỉ đọc, chưa đăng ký output hoặc reservation: REJECTED theo fail-closed. `runs/sd6-diagnostics-6600/overrides.json` có seed=null; hai `runs/t2-{full,smoke}-20261014-060320/replay_cases.json` lớn hơn32MiB cần seed index được kiểm chứng. Metadata/ranges chưa đủ chứng minh sạch để launch full; không bỏ qua hoặc sửa log cũ. Đây là prerequisite về lịch sử, không phải kết quả chất lượng hay lỗi smoke.

## Rulings khi thực thi

1. Dùng workspace/branch đã được duyệt, chỉ stage file thuộc S2-KL; giữ UI/PDE/user changes. Nếu sai có thể lẫn thay đổi ngoài scope; giảm rủi ro bằng explicit paths và hash bảo vệ.
2. Thay bash workflow scripts bằng PowerShell/Python tương đương trên Windows. Nếu sai có thể lệch bookkeeping; đối chiếu ledger với từng checkbox của plan.
3. Chạy T2 core rồi T3 offline trước khi chốt T2 vì frozen acceptance gọi diagnose giữa stop/resume. Commit T2/T3 chung; chi phí là diff qua hai task lớn hơn, giữ log/suite riêng để truy vết.
4. Cho tác giả độc lập sửa fixture JSONL kỹ thuật sai và tăng assertion overlap; giữ bản gốc/hashes/proof. Nếu sai có thể ảnh hưởng tính độc lập; implementer không sửa acceptance và mọi hash khác được đối chiếu.

5. Giữ commit dữ liệu80bafa7 xuất hiện đồng thời trong lịch sử; executor này không tạo/push commit đó và không suy đoán ủy quyền xuất bản của workflow khác. Không revert/viết lại history/push. Nếu sai, diff toàn nhánh gồm payload cũ và cần kiểm phạm vi/ủy quyền riêng trước tích hợp.

6. Giữ UI/PDE/journal và so lỗi baseline theo plan đã duyệt, không sửa code lịch sử ngoài phạm vi. Nếu sai, lỗi cũ có thể che vấn đề khác; toàn repo vẫn chưa xanh và không merge/push.
7. Giữ full registration fail-closed với lịch sử SD6/T2 chưa ánh xạ; không bỏ qua/xóa log. Chi phí là chưa thể launch full đến khi xác minh xong lịch sử.
8. Chỉ kết luận về phần mềm/resume giới hạn, không suy chất lượng/generalization/ETA full. Nếu sai, pilot sau này có thể thất bại hoặc lâu hơn.
9. Giữ trainer và phạm vi pilot một biến, không thêm gradient-component instrumentation/sửa helper reward ngoài scope. Chi phí là giới hạn khả năng quy nguyên nhân/quan sát, không chứng minh helper bị loại là đúng.

## Review cuối

Reviewer độc lập gpt-6-astra kiểm range ed7c749..2dbcdd3, cả năm Review Focus:0Critical/2Important(P2)/0Minor. Hai finding output identity/stage order được sửa trong một lượt ở52b407e, regression2methods/bốn trường hợp RED→GREEN14.803s. Audit bổ sung inventory RNG được sửa ởdc8b08b, RED1method/hai trường hợp → GREEN trong26test. Không re-review; full-root cuối đã kiểm51/51methods S2-KL gồm CUDA thật, không có lỗi mới. Chi tiết: `2026-10-23-s2-kl-final-review.md`. Chưa merge/push hoặc mở full pilot.
