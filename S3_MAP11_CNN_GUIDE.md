# Đánh giá S3-11 với CNN trên11×11

Runner riêng `probe_s3_map11_cnn.py`, không train/không đổi checkpoint phục vụ. Hai learner S3-11(epoch16) và S3 gốc, cùng pool đóng băng S3/S2/G. Cả hai bên được nạp11×11 bằng loader hiện có, lấy mẫu+mask, không search/RuleBot.

## Cấu hình đã chọn

- Map11×11/13kênh/2agent/max500/E475-25-40.
- 200bàn×2góc=400trận/cặp;2learner×3opponent=6cặp,2400trận tổng.
- Cùng bàn/góc cho hai learner, shard10bàn,120shard tổng. CPU tối đa5worker, mỗi worker1torch thread; có progress.
- WR, ActiveWR, hòa, hòa hết giờ, loss, tự nổ/passive/SDdeath; lưu raw games cùng metadata/hash.
- So sánh paired theo từng đối thủ và macro trung bình đều3đối thủ tại mỗi(seed,corner), rồi bootstrap theo200bàn/10000lần/seed20261014. Không dùng1200trận pool như các mẫu độc lập.
- Ngưỡng báo cáo cải thiện: macro ActiveWR≥+5pp, CI95 dưới>0, macro tự nổ tăng≤1pp. Luôn báo hòa/loss; kết quả benchmark không tự cho phép train thêm hoặc thay model phục vụ.
- Seed830000–830199 là lựa chọn cho run mới ngày25/10/2026, phải còn fresh khi init; không dùng lại820000–820199 đã test trước đó.

## Lệnh cmd tại E:\Bomberman

```cmd
venv\Scripts\python.exe -m unittest test_probe_s3_map11_cnn -q
venv\Scripts\python.exe probe_s3_map11_cnn.py --help

REM Init mot lan, khong ghi de run da co
venv\Scripts\python.exe probe_s3_map11_cnn.py init --output runs\s3-map11-cnn-20261025-v1 --seed 830000

REM Danh gia, khong train
venv\Scripts\python.exe probe_s3_map11_cnn.py run --output runs\s3-map11-cnn-20261025-v1 --workers 5 > runs\s3-map11-cnn-20261025-v1-console.log 2>&1

REM Chi resume neu tien trinh cu da dung; giu cac shard hop le
venv\Scripts\python.exe probe_s3_map11_cnn.py run --output runs\s3-map11-cnn-20261025-v1 --workers 5 --resume > runs\s3-map11-cnn-20261025-v1-resume.log 2>&1
```

Không đổi source/config/checkpoint giữa init/run/resume. Shard hỏng bị từ chối, không được âm thầm skip hoặc ghi đè. Nếu bị hard kill để lại `run.lock.json`, phải xác minh PID native/start-time/command line đã chết rồi mới xóa đúng stale lock; không xóa raw data/marker để chạy lại. Run đã có summary hoàn tất chỉ được xác thực/đọc lại qua resume, không chơi thêm.

Smoke dùng `init --smoke` với output/seed giả riêng:2bàn/cặp,max12,24trận tổng. Smoke không phải bằng chứng về sức chơi hoặc kết quả full. Benchmark2400trận dùng seed đã đăng ký và chỉ mở marker một lần; resume là tiếp phần thiếu của cùng phép đo.

## Chạy tự động và chờ45phút

Sau khi test/review runner, Codex tạo run, launch tiến trình native/hidden độc lập PTY, lưu PID/start-time/log/state và đặt heartbeat45phút trong chat hiện tại. Chỉ xem snapshot khi thức dậy; vẫn chạy thì yên lặng. Khi xong, xác thực toàn bộ shard/hash/raw/counts/CI, ghi báo cáo/journal/tracker rồi gỡ heartbeat. Không suy tiến độ chỉ từ PID hoặc không thấy PID trong sandbox: process native phải được kiểm bằng native API trước khi kết luận đã chết hoặc khởi động resume.

Máy cần bật và app đang chạy để lịch làm việc với file local. Không tái tạo lịch liên tục hoặc mở chat khác. Không train bổ sung, chỉnh gate hoặc chọn lại model theo kết quả benchmark này.
