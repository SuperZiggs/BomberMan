# SD8 — hướng dẫn chạy hai hướng A và B

Spec (luật chốt trước): `docs/superpowers/specs/2026-10-20-sd8-distill-and-deep-search-design.md`.

| | Hướng B: search ba nước | Hướng A: chưng cất search hai nước vào CNN |
|---|---|---|
| Ai chạy | **Bạn (máy này)**, vì nhẹ hơn | **Bạn của bạn** (máy có GPU), vì nặng hơn |
| Có train không | Không | Có (train trên GPU vài phút) |
| Khối lượng | ~800 trận search3 + 400 trận search2 ≈ 7–8 giờ CPU | 3000 trận G+search2 để thu dữ liệu ≈ 10–15 giờ CPU, cộng validate/test ≈ 3 giờ CPU |
| Thời gian thực (5 tiến trình) | khoảng 1.5–2 giờ | khoảng 3–4 giờ |
| Seed test (một lần) | 49000 | 48000 |
| Kết quả mong đợi | ActiveWR trên 82% nếu search sâu còn tăng | CNN thuần chạy nhanh, mạnh hơn G và S |

Mọi lệnh dưới đây dùng cú pháp **cmd**. Các script có thanh progress hiện trên console, kể cả khi output được chuyển vào log. Thư mục output không được tồn tại trước; script từ chối ghi đè.

---

## Hướng B (bạn chạy trên máy này)

Python: `D:\anaconda\envs\torch\python.exe`. Không nên chạy việc nặng khác cùng lúc, vì script dùng 5 tiến trình.

```cmd
cd /d D:\fpt\subject\BomberMan

REM 1) Kiểm tra nhanh (khoảng 1 phút)
D:\anaconda\envs\torch\python.exe -m unittest test_kill_search test_run_sd8 -q

REM 2) Validation seed 21000: G+search3, 400 trận. Tái dùng 400 trận G+search2 đã chạy ở SD7.9b trên máy này (khoảng 30–45 phút)
D:\anaconda\envs\torch\python.exe run_sd8b_search3.py validate --output runs\sd8b-val-21000 --base runs\sd79b-search2-21000 > runs\sd8b-val-21000-console.log 2>&1

REM 3) Xem luật: "open_test_49000": true thì làm bước 4; false thì dừng và báo lại
type runs\sd8b-val-21000\summary.json

REM 4) Test một lần seed 49000: G+search3 vs G+search2, 400 trận mỗi bên (khoảng 1 giờ)
D:\anaconda\envs\torch\python.exe run_sd8b_search3.py test --output runs\sd8b-20261020 --validation runs\sd8b-val-21000 > runs\sd8b-20261020-test-console.log 2>&1
```

- Bước 4 tự từ chối nếu luật validation không đạt, và chỉ chạy được một lần (`test_opened.json`).
- Kết quả nằm ở `runs\sd8b-20261020\summary.json` (mục `gate`), kèm file ZIP `runs\sd8b-20261020-send.zip`.
- Nếu muốn chạy lại cả G+search2 thay vì tái dùng, bỏ `--base ...` ở bước 2. Khi đó chạy thêm khoảng 15 phút.

---

## Hướng A (bạn của bạn chạy)

Dùng Python có torch trên máy bạn ấy (ví dụ `venv\Scripts\python.exe`; dưới đây viết tắt là `python`). Nên có CUDA để train nhanh; collect, validate và test chạy trên CPU.

```cmd
REM 0) Lấy code và checkpoint (G và S nằm trong Git LFS)
cd /d <thư mục repo BomberMan>
git fetch origin
git checkout danh/sd5-friend-merge
git pull
git lfs pull

REM 1) Kiểm tra
python -m unittest test_run_sd8 test_kill_search test_run_sd53_pilot -q

REM 2) Smoke trọn bốn bước, ghi vào thư mục tạm (khoảng 3–5 phút)
python run_sd8a_distill.py collect  --output %TEMP%\sd8a-smoke --smoke --workers 2
python run_sd8a_distill.py train    --output %TEMP%\sd8a-smoke --smoke
python run_sd8a_distill.py validate --output %TEMP%\sd8a-smoke --smoke --workers 2
python run_sd8a_distill.py test     --output %TEMP%\sd8a-smoke --smoke --workers 5
rmdir /s /q %TEMP%\sd8a-smoke

REM 3) Thu dữ liệu: 3000 trận G+search2 (phần nặng nhất, khoảng 2–3 giờ với 5 tiến trình; máy mạnh có thể dùng --workers 8)
python run_sd8a_distill.py collect --output runs\sd8a-20261020 --workers 5 > runs\sd8a-20261020-collect-console.log 2>&1

REM    Nếu bị ngắt giữa chừng: chạy lại với --resume (các shard đã xong được giữ lại)
REM python run_sd8a_distill.py collect --output runs\sd8a-20261020 --workers 5 --resume > runs\sd8a-20261020-collect2-console.log 2>&1

REM 4) Train 16 epoch từ G (GPU vài phút; CPU khoảng 20–40 phút)
python run_sd8a_distill.py train --output runs\sd8a-20261020 > runs\sd8a-20261020-train-console.log 2>&1

REM 5) Validation seed 21000: G và các epoch 4/8/12/16, chọn epoch theo luật (khoảng 15–30 phút)
python run_sd8a_distill.py validate --output runs\sd8a-20261020 --workers 5 > runs\sd8a-20261020-validate-console.log 2>&1

REM 6) Test một lần seed 48000: S2, G, S (CNN thuần, lịch E và off) + G+search2 (trần tham chiếu) (khoảng 1 giờ)
python run_sd8a_distill.py test --output runs\sd8a-20261020 --workers 5 > runs\sd8a-20261020-test-console.log 2>&1
```

Gửi lại cho bạn:
- `runs\sd8a-20261020-send.zip`, gồm dữ liệu, các checkpoint `train\epoch_*.pt`, `validation.json` và `summary.json`;
- các file `runs\sd8a-20261020-*-console.log`.

Hoặc commit và push lên một nhánh mới; file `.pt` tự đi qua LFS.

Lưu ý:
- Không đổi tên hoặc di chuyển thư mục `runs\sd8a-20261020` giữa các bước. Đường dẫn output được ghi trong `registration.json`, và mọi bước kiểm tra lại cấu hình cùng SHA của source.
- Không sửa source giữa các bước, vì bước sau sẽ từ chối chạy.
- Bước test chỉ chạy được một lần (`test_opened.json`).
