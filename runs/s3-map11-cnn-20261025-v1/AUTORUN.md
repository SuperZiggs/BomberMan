# S3-11 đấu CNN — theo dõi45phút, 25/10/2026

Người dùng đã yêu cầu thực hiện bộ đánh giá theo từng bước và quay lại mỗi45phút khi chờ, không theo dõi liên tục. Không có ủy quyền train thêm, thay model phục vụ hoặc chạy một bộ thử khác để cứu kết quả.

## Hợp đồng đã khóa

- Workspace `E:\Bomberman`; Python `E:\Bomberman\venv\Scripts\python.exe`; runner `E:\Bomberman\probe_s3_map11_cnn.py`.
- Output duy nhất `E:\Bomberman\runs\s3-map11-cnn-20261025-v1`; registration là nguồn thật, không init lại hoặc sửa config/hash/source/model sau khi mở run.
- Learner S3_11=`runs/s3-map11-20261024-auto-v1/train/epoch_16.pt` SHA `17748471…a95f`, baseline S3=`runs/sd9-20261023/train/epoch_12.pt` SHA `4185375f…69ed7`. Cùng pool S3/S2/G,4checkpoint đều khóa SHA đầy đủ trong registration.
- 200bàn830000–830199×2góc=400trận/cặp;6cặp=2400trận. Map11×11/13kênh/2agent/max500/E475-25-40, CNN CPU lấy mẫu+mask, không RuleBot/search/gradient.
- Shard10bàn,20trận/shard,120shard tổng; tối đa5worker, mỗi worker1torch thread. Hai learner và mọi đối thủ đều được nạp11×11 bằng loader đã có; các đối thủ9×9 là bản chuyển map, không phải CNN native11×11.
- CI95 paired theo từng đối thủ và macro: trung bình đều3opponent trước tại mỗi(seed,corner), rồi resample200bàn/10000lần/seed20261014. Không coi1200game pool là1200mẫu độc lập.
- Ngưỡng báo cáo qualified gain: macro ActiveWR≥+5pp, CI95 dưới>0, macro tự nổ tăng≤1pp. Báo cả WR/draw/loss/timeout/passive/SDdeath dù PASS/FAIL. Đây chỉ là kết quả benchmark, không cho phép promote/train tự động.

## Trước launch full

Đọc `autorun-state.json`. Runner đã được kiểm acceptance32test và native smoke2worker24game/max12; review độc lập đã xử lý tính nhất quán kết quả và source bundle. Chỉ launch full sau khi `verification_ready=true` trong state; nếu main bị ngắt trước khi cập nhật, đọc `.superpowers/sdd/2026-10-25-s3-map11-cnn/full-result.json` và log để xác nhận: toàn bộ test mới/targeted phải pass; chỉ8lỗi thiếu fixture cũ đã tái hiện trong `old-fixture-baseline.log` và1skip được ghi riêng, không được coi lỗi mới là bình thường. Nếu kiểm chứng chưa đủ thì dừng báo cần xử lý, không đoán.

## Launch/resume

Dùng Start-Process với `-WindowStyle Hidden`, WorkingDirectory `E:\Bomberman`, PassThru, stdout/stderr riêng dưới run; native ngoài sandbox cho Windows named pipe. Môi trường chỉ cho process được khởi động: TEMP/TMP=`<output>\tmp`, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1, PYTHONUNBUFFERED=1. Không sửa môi trường hệ thống.

Arguments: `E:\Bomberman\probe_s3_map11_cnn.py run --output E:\Bomberman\runs\s3-map11-cnn-20261025-v1 --workers 5`. Sau launch ghi PID/startUTC/command/log/attempt vào state. Lần đầu không `--resume`; marker test tạo duy nhất trước game. Không mở thêm process khi đã có run đang sống.

**Sandbox có thể không nhìn thấy PID native.** Không kết luận chết từ Get-Process trong sandbox. Khi cần phải kiểm bằng native CIM/Get-Process: root venv PID trong state, command line và creation/start-time; interpreter PID trong `run.lock.json`; các child/grandchild có ParentProcessId thuộc các PID trên. PID tái sử dụng hoặc không xác định chắc thì không giết/xóa lock/launch chồng.

PowerShell `ConvertFrom-Json` có thể đã chuyển chuỗi ISO thành DateTime. So thời gian bằng `([DateTime]$state.processes.run.start_time_utc).ToUniversalTime()` với `CreationDate.ToUniversalTime()`; không parse lại DateTime qua DateTimeOffset.Parse vì có thể mất timezone. Child Python gồm resource tracker ngoài5worker, không coi đó là worker thứ6.

Nếu run còn hoạt động đúng định danh và summary chưa hoàn tất, kết thúc lượt yên lặng, chờ heartbeat45phút sau; không poll/log-stream/sleep vòng lặp.

Nếu process thực sự chết và không còn worker cũ, chỉ resume tự động tối đa1lần của cùng phép đo. Kiểm mọi shard hiện có bằng runner, không xóa raw/marker/summary để chạy lại. Nếu hard kill để lại `run.lock.json`, chỉ sau xác minh native owner/children đã chết mới xóa đúng file lock trong output; không xóa file khác. Resume dùng log mới `run-resume1.stdout.log`/stderr và thêm `--resume`; lưu attempt2. Marker/seed/hash giữ nguyên, chỉ chơi shard thiếu. Nếu không có marker và chưa có shard thì first launch chưa mở phép đo, có thể khởi động first-run một lần sau kiểm chứng; không giả vờ đây là resume.

Shard lỗi, config/source/model/runtime/bundle khác, khóa không xác minh được, hoặc resume vẫn chết: dừng/incomplete, lưu lỗi và báo; không chỉnh file để hợp thức hóa hay đổi seed/hyperparameter. Không khởi động training.

## Khi có summary

Xác minh process/lock không còn active. Dùng API `probe_s3_map11_cnn.run(output, workers=5, resume=True)` để kiểm chứng report hoàn tất: API này kiểm raw/hash và trả report cũ, không chơi game/ghi đè khi đủ120shard. Không chỉ tin PID, số file hoặc câu in console. Đối chiếu seed/corner400/cặp,6cặp/2400game,4model/source/bundle, marker và macro/pairedCI. Nếu report không khớp raw, dừng báo integrity error.

Viết `FINAL_REPORT.md`, ghi RESULTS.md + IMPROVEMENTS.md với đầy đủ tham số/thời gian và cập nhật PLAN_SUDDEN_DEATH.md + PLAN_BOMBERMAN.md; thêm insight khi có bằng chứng mới, cập nhật ledger seed830000–830199 đã dùng. Không sửa S3/S3-11 phục vụ. Kết luận chỉ11×11/1v1 với pool CNN đã chuyển map, một benchmark mới; không suy FFA/native11×11 CNN hoặc ổn định nhiều train seed.

State cuối `completed_pass` hoặc `completed_no_qualified_gain` theo gate, hay `stopped_incomplete` nếu lỗi. Khi lý do theo dõi kết thúc, **xóa chính heartbeat bằng automation_update(mode=delete)** với ID trong state, lưu xác nhận và thông báo ngắn tiếng Việt kèm kết quả, nêu rõ lịch đã gỡ. Không archive chat hoặc gửi nơi khác. Những snapshot đang chạy/không có thay đổi hành động thì DONT_NOTIFY, không phát prose progress cho người dùng.
