# SD9 — lệnh chạy (chưng cất S2 + search3 thành S3)

Spec: `docs/superpowers/specs/2026-10-23-sd9-search3-distillation.md`. Chạy trong **cmd** tại `D:\fpt\subject\BomberMan`, theo đúng thứ tự. Không đổi tên hoặc di chuyển `runs\sd9-20261023` giữa các bước.

```
REM 0. Kiểm tra nhanh (vài giây)
D:\anaconda\envs\torch\python.exe -m unittest test_run_sd9 -q

REM 1. Thu dữ liệu: 3000 trận S2+search3, khoảng 6-7 giờ với 5 tiến trình
D:\anaconda\envs\torch\python.exe run_sd9_distill.py collect --output runs\sd9-20261023 --workers 5 > runs\sd9-20261023-collect-console.log 2>&1

REM    Nếu bị ngắt: chạy lại với --resume (giữ các shard đã xong)
REM D:\anaconda\envs\torch\python.exe run_sd9_distill.py collect --output runs\sd9-20261023 --workers 5 --resume > runs\sd9-20261023-collect2-console.log 2>&1

REM 2. Train 16 epoch từ S2 (có GPU thì tự dùng; CPU cũng chạy được)
D:\anaconda\envs\torch\python.exe run_sd9_distill.py train --output runs\sd9-20261023 > runs\sd9-20261023-train-console.log 2>&1

REM 3. Validation seed 21000, chọn epoch
D:\anaconda\envs\torch\python.exe run_sd9_distill.py validate --output runs\sd9-20261023 --workers 5 > runs\sd9-20261023-validate-console.log 2>&1

REM 4. Test MỘT LẦN seed 57000 (khoảng 1 giờ)
D:\anaconda\envs\torch\python.exe run_sd9_distill.py test --output runs\sd9-20261023 --workers 5 > runs\sd9-20261023-test-console.log 2>&1
```

- Không sửa source giữa các bước: mỗi bước kiểm SHA source với `registration.json`.
- Bước 4 chỉ chạy được một lần (`test_opened.json`).
- Xong bước nào cũng có thể báo Claude đọc kết quả; bắt buộc báo sau bước 4.
