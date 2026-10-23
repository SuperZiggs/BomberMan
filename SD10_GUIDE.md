# SD10 — lệnh chạy (chưng cất S3 + search3 đoán đối thủ đúng thành S4)

Spec: `docs/superpowers/specs/2026-10-24-sd10-cnn-aware-distillation.md`.

**Cảnh báo chi phí:** bước thu dữ liệu tốn khoảng **53 giờ** với 5 tiến trình, vì mỗi trận đấu CNN mất khoảng 10 phút. Lợi ích tối đa ước lượng khoảng +10pp. Chỉ chạy khi đã chấp nhận chi phí này.

Chạy trong **cmd** tại `D:\fpt\subject\BomberMan`, theo đúng thứ tự. Không đổi tên hoặc di chuyển `runs\sd10-20261024` giữa các bước.

```
REM 0. Kiểm tra nhanh (khoảng 1-2 phút)
D:\anaconda\envs\torch\python.exe -m unittest test_run_sd10 -q

REM 1. Thu dữ liệu: 2000 trận S3+search3 (500 đấu RuleBot, 1500 đấu S2/G/S3), khoảng 53 giờ với 5 tiến trình
D:\anaconda\envs\torch\python.exe run_sd10_distill.py collect --output runs\sd10-20261024 --workers 5 > runs\sd10-20261024-collect-console.log 2>&1

REM    Nếu bị ngắt (tắt máy, ...): chạy lại với --resume, các shard 20 bàn đã xong được giữ
REM D:\anaconda\envs\torch\python.exe run_sd10_distill.py collect --output runs\sd10-20261024 --workers 5 --resume > runs\sd10-20261024-collect2-console.log 2>&1

REM 2. Train 16 epoch từ S3 (có GPU thì tự dùng, vài phút)
D:\anaconda\envs\torch\python.exe run_sd10_distill.py train --output runs\sd10-20261024 > runs\sd10-20261024-train-console.log 2>&1

REM 3. Validation seed 21000 (S3 + 4 ứng viên đấu S2/G/S3/RuleBot, khoảng 1 giờ)
D:\anaconda\envs\torch\python.exe run_sd10_distill.py validate --output runs\sd10-20261024 --workers 5 > runs\sd10-20261024-validate-console.log 2>&1

REM 4. Test MỘT LẦN seed 58000 (khoảng 40 phút); tự từ chối nếu validation không hơn S3
D:\anaconda\envs\torch\python.exe run_sd10_distill.py test --output runs\sd10-20261024 --workers 5 > runs\sd10-20261024-test-console.log 2>&1
```

- Không sửa source giữa các bước: mỗi bước kiểm SHA source với `registration.json`.
- Bước 4 chỉ chạy được một lần (`test_opened.json`).
- Xong bước nào cũng có thể báo Claude đọc kết quả; bắt buộc báo sau bước 3 và bước 4.
