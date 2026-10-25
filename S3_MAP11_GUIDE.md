# S3 → 11×11: runner chưng cất riêng

Runner: `run_s3_map11_distill.py`. Phạm vi hiện tại là **bước 1: chuẩn bị và kiểm tra runner**; chưa chạy các stage dài. Mục tiêu là CNN thuần trên 11×11, 1v1 RuleBot, khởi tạo từ S3. Không thay thế model 9×9 đang dùng.

## Kiểm tra bước 1 (cmd tại E:\Bomberman)

```cmd
venv\Scripts\python.exe -m unittest test_run_s3_map11 -v
venv\Scripts\python.exe run_s3_map11_distill.py --help
venv\Scripts\python.exe run_s3_map11_distill.py init --help
```

Các test dùng thư mục tạm và workload nhỏ; không chạy benchmark hoặc train dài. `--help` không tạo run.

## Cấu hình cho các bước sau

- Checkpoint xuất phát: `runs/sd9-20261023/train/epoch_12.pt`, SHA `4185375f…69ed7`.
- Mạng thầy và mạng học đều 13 kênh, 11×11; nạp S3 qua loader chuyển kích thước hiện có. Thầy dùng search3; stage baseline so S3 thuần sau chuyển map với S3+search3 trên cùng bàn/góc.
- Mặc định: 100 bàn collect × 2 góc = 200 trận pilot; baseline 100 trận mỗi agent (S3 và S3+search3), tổng 200 trận; validation/test 400 trận mỗi agent. Max_steps 500; E = 475/25/40, collect trộn 75% E và 25% off.
- Train chưng cất: Adam 1e-4, batch 256, 16 epoch, ứng viên 4/8/12/16; KL neo 1; held-out 10% theo seed bàn. Đây là cấu hình xuất phát, chưa được xác nhận có lợi trên 11×11.
- `init` bắt buộc khai báo seed collect và seed test mới; phải kiểm tra sổ seed dự án trước. Không dùng lại seed test SD9 57000, SD10 58000 đã đặt chỗ hoặc các seed đã dùng/đặt chỗ của hướng khác. Các stage sau lấy thông số từ registration; không sửa source/config giữa các stage.
- Dùng output mới, ví dụ `runs/s3-map11-pilot-v1`. Source và checkpoint được kiểm SHA. Giữ nguyên thư mục run sau init.

Lệnh init dưới đây là **mẫu**, thay `SEED_THU_MOI` và `SEED_TEST_MOI` bằng seed đã kiểm tra/đăng ký trước khi chạy:

```cmd
venv\Scripts\python.exe run_s3_map11_distill.py init --output runs\s3-map11-pilot-v1 --collect-first-seed SEED_THU_MOI --collect-boards 100 --validation-seed 21000 --test-seed SEED_TEST_MOI
```

## Các stage công việc dài — chưa chạy ở bước 1

```cmd
REM Buoc 2: baseline S3 va thay S3+search3 tren cung ban validation
venv\Scripts\python.exe run_s3_map11_distill.py baseline --output runs\s3-map11-pilot-v1 --workers 5 > runs\s3-map11-pilot-v1-baseline.log 2>&1

REM Doc baseline.json: thay phai cho thay loi the truoc khi thu du lieu
REM Buoc 3: thu du lieu 11x11 moi
venv\Scripts\python.exe run_s3_map11_distill.py collect --output runs\s3-map11-pilot-v1 --workers 5 > runs\s3-map11-pilot-v1-collect.log 2>&1

REM Neu collect bi ngat: chi tiep tuc cac shard thieu
venv\Scripts\python.exe run_s3_map11_distill.py collect --output runs\s3-map11-pilot-v1 --workers 5 --resume > runs\s3-map11-pilot-v1-collect-resume.log 2>&1

REM Buoc 4: train, khong phai RL Phase C
venv\Scripts\python.exe run_s3_map11_distill.py train --output runs\s3-map11-pilot-v1 --device cpu > runs\s3-map11-pilot-v1-train.log 2>&1

REM Buoc 5: chon checkpoint tren validation, sau do moi test mot lan
venv\Scripts\python.exe run_s3_map11_distill.py validate --output runs\s3-map11-pilot-v1 --workers 5 > runs\s3-map11-pilot-v1-validate.log 2>&1
venv\Scripts\python.exe run_s3_map11_distill.py test --output runs\s3-map11-pilot-v1 --workers 5 > runs\s3-map11-pilot-v1-test.log 2>&1
```

Có thể dùng `--device cuda` cho train nếu CUDA hoạt động. Collection/evaluation dùng CPU, tối đa 5 worker, có thanh tiến trình. Đọc số nhãn thực tế trước khi train: pilot quá ít trạng thái quyết định hoặc thiếu nhãn trong held-out sẽ bị từ chối, cần một run lớn hơn với đăng ký mới.

Validation chọn ActiveWR cao nhất trong các ứng viên có tự nổ ≤ baseline + 1 điểm phần trăm; hòa điểm chọn epoch sớm. Không có ứng viên hơn baseline thì không mở test thật. Gate test: ActiveWR hơn baseline ít nhất 5 điểm phần trăm, CI95 dưới > 0, tự nổ tăng không quá 1 điểm phần trăm.

Smoke được đăng ký bằng `init --smoke` trong **output khác**. Cấu hình ngắn (12 bước/trận, baseline 4 trận mỗi agent = 8 lượt env) chỉ để kiểm tra kỹ thuật, có thể chưa có nhãn quyết định đủ để train; không phải bằng chứng về sức chơi và không mở seed test thật. Train chỉ tiếp tục khi dữ liệu đủ cho cả train/held-out.
