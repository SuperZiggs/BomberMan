# S3 map11 — tự động bước 2–5, 24/10/2026

Người dùng đã cho phép tự thực hiện tuần tự bước 2–5 và chỉ quay lại kiểm tra mỗi 45 phút khi phải chờ. Không poll liên tục, không mở chat khác. Quy trình nằm trong chat hiện tại và dùng heartbeat của app.

## Đăng ký đã khóa

- Workspace `E:\Bomberman`, Python `E:\Bomberman\venv\Scripts\python.exe`, runner `E:\Bomberman\run_s3_map11_distill.py`.
- Output duy nhất `E:\Bomberman\runs\s3-map11-20261024-auto-v1`; đọc `registration.json` làm nguồn thật.
- S3 `runs/sd9-20261023/train/epoch_12.pt`, SHA `4185375f8b5f5cf070d3eef6ce98bec27e23fe06c64d97b500c939e36e669ed7`.
- Collect bàn 812000–812099 × 2 góc = 200 trận; validation 21000–21199 × 2 góc = 400 trận mỗi agent. Baseline dùng 50 bàn đầu 21000–21049 × 2 góc = 100 trận mỗi agent (S3 và S3+search3, tổng200).
- Test820000–820199 × 2góc=400trận/agent **đã mở đúng một lần và hoàn tất PASS** ngày24/10/2026; không mở lại. Seed được đối chiếu ledger/registration/opened trước init.
- 11×11/13 kênh/2 agent/RuleBot, max500, E475/25/40; collect trộn75%E/25%off; policy lấy mẫu+mask, thầy search3. Collection/evaluation CPU, yêu cầu tối đa5worker; baseline/test chỉ có2job nên thực tế tối đa2worker, collect4shard nên tối đa4worker.
- Train CUDA RTX4060 Laptop 8GB, Adam1e-4, batch256, KLneo1, held-out10% theo bàn,16epoch, candidate4/8/12/16. Phép tính CUDA đã kiểm tra thành công trước khởi động. Không đổi source/config/hyperparameter trong run.

## Các bước và gate chốt trước kết quả

1. Bước2 `baseline`: chỉ tiếp tục khi `teacher_has_advantage=true`, paired ActiveWR của thầy−S3 >0pp, và paired suicide tăng không quá1pp. Đây là gate thăm dò100trận/agent, không tuyên bố ý nghĩa thống kê khi CI còn chứa0. Nếu không đạt, dừng, không thu dữ liệu để cố cứu gate.
2. Bước3 `collect`: 200trận mới. Chỉ `--resume` khi tiến trình cũ đã chết, thiếu shard và chưa có summary hoàn chỉnh. Kiểm metadata/shape/coverage theo runner. Dữ liệu không đủ quyết định/neo hoặc thiếu train/held-out thì dừng báo giới hạn; không tự mở rộng dải bàn hoặc đổi cấu hình.
3. Bước4 `train --device cuda`: chưng cất, không RL; không ghi đè thư mục train. Candidate mới giữ riêng, không thay S3 đang phục vụ.
4. Bước5 `validate`:400trận/agent; chọn ActiveWR cao nhất trong các candidate tự nổ≤S3+1pp, tie lấy epoch sớm. `proceed_to_test=false` thì dừng VALIDATION_STOP, giữ test kín.
5. Bước5 `test`: chỉ khi validation cho phép; đúng một lần trên820000–820199. Gate cuối ActiveWR≥+5pp, CI95 dưới>0, tự nổ≤+1pp; bootstrap ghép cặp theo bàn10000lần, seed20261014. FAIL vẫn báo đúng, không train/test lại để đổi kết luận.

## Cách kiểm tra mỗi lần heartbeat

Đọc `autorun-state.json` và các artifact hoàn tất, không đọc log tăng trưởng liên tục. Chỉ lấy một snapshot trạng thái mỗi lần thức dậy. Khi một stage hoàn tất thì xác thực artifact/coverage/source, cập nhật journal/tracker và khởi động stage kế tiếp đã được phép. Nếu tiến trình vẫn chạy và chưa có artifact hoàn tất thì kết thúc lượt yên lặng, chờ lần45phút sau.

`autorun-state.json` chứa PID, thời điểm bắt đầu UTC và đường dẫn stdout/stderr từng stage. Xác minh PID cùng start-time để tránh nhầm PID tái sử dụng; artifact đầy đủ mới là bằng chứng hoàn tất. **Sandbox có thể trả về không thấy PID native dù tiến trình vẫn chạy** (đã xác nhận với baseline PID4856); khi Get-Process trong sandbox không thấy PID, phải kiểm tra native bằng Get-CimInstance/Get-Process ngoài sandbox trước khi kết luận chết hoặc launch lại. Đọc cả process con của venv redirector nếu cần; đối chiếu command line và thời gian tạo. Nếu PID thực sự chết mà thiếu artifact, xem tail log một lần, ghi lỗi/incomplete, dừng; không khởi động chồng tiến trình. Có thể resume collect đúng một lần sau khi xác nhận không còn worker cũ; không resume train hoặc test đã mở mà chưa có summary. Không sửa hash registration hoặc hạ gate.

## Cách khởi động stage đã được phép

Dùng `Start-Process` với `-WindowStyle Hidden`, `-WorkingDirectory E:\Bomberman`, `-PassThru`; stdout/stderr tách file riêng dưới output. Ghi process ID/start-time/command vào state ngay sau launch. Process phải độc lập với PTY của chat để không mất khi lượt chat kết thúc. Chạy native ngoài sandbox khi cần Windows named pipe (đã có bằng chứng sandbox chặn multiprocessing; người dùng đã cho phép các stage này). Không mở terminal hiển thị hoặc dùng vòng lặp chờ.

Đặt các biến môi trường chỉ cho tiến trình được khởi động: `TEMP/TMP=<output>\tmp`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`. Không thay môi trường hệ thống. Tạo `<output>\tmp` nếu thiếu.

Đối số Python: `E:\Bomberman\run_s3_map11_distill.py <stage> --output E:\Bomberman\runs\s3-map11-20261024-auto-v1`; thêm `--workers 5` cho baseline/collect/validate/test, hoặc `--device cuda` cho train. Không tự gọi init lại. Chỉ thêm `--resume` cho collect bị ngắt theo điều kiện ở trên.

## Kết thúc

Ghi kết quả và đầy đủ thông số/thời gian vào RESULTS.md + IMPROVEMENTS.md, cập nhật PLAN_SUDDEN_DEATH.md + PLAN_BOMBERMAN.md. Khi hoàn tất hoặc gate/lỗi yêu cầu dừng, ghi state trạng thái cuối, báo kết quả bằng tiếng Việt (WR, ActiveWR, hòa, tự nổ, delta/CI và checkpoint), rồi gỡ chính heartbeat này bằng automation_update theo quy định heartbeat hiện hành; ID được lưu trong state. Không archive chat, không đổi model phục vụ, không gửi tin nhắn nơi khác.

Trạng thái chốt24/10/2026: hoàn tất bước2–5, test PASS với epoch16, ActiveWR87.25% so46.25%, delta+41pp[34.5,47.25], tự nổ+.5pp. Không còn stage phải chạy; xem `FINAL_REPORT.md` và state cuối.
