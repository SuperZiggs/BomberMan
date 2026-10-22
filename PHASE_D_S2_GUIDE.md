# Chạy Phase D từ S2 trên Windows

Runner: `E:\Bomberman\run_phase_d_s2.py`. Python có torch: `E:\Bomberman\venv\Scripts\python.exe`.
Thiết kế: `docs/superpowers/specs/2026-10-22-phase-d-s2-pool-design.md`.

Hai nhánh control/pool cùng bắt đầu từ S2, mỗi nhánh 2.000 episode; train một nhánh tại một thời điểm trên GPU. Validation/test chạy CPU với số worker đã đăng ký (mặc định 2; cho phép 1–5). Ngân sách tối đa 4.000 episode train và 9.000 trận đánh giá. Không tự train thêm nếu validation dừng.

## Lệnh CMD cho lượt thật

Các lệnh dưới đây chỉ là hướng dẫn; chưa đăng ký hoặc chạy lượt full. Chạy lần lượt, kiểm exit code và status trước bước sau. Không dùng PowerShell `*>` trong CMD.

```bat
cd /d E:\Bomberman
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\run_phase_d_s2.py register --output E:\Bomberman\runs\phaseD-s2-20261022 --input-root E:\Bomberman --device cuda:0 --workers 2 > E:\Bomberman\runs\phaseD-s2-register.log 2>&1
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\run_phase_d_s2.py train --output E:\Bomberman\runs\phaseD-s2-20261022 --arm control > E:\Bomberman\runs\phaseD-s2-20261022\control.log 2>&1
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\run_phase_d_s2.py train --output E:\Bomberman\runs\phaseD-s2-20261022 --arm pool > E:\Bomberman\runs\phaseD-s2-20261022\pool.log 2>&1
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\run_phase_d_s2.py validate --output E:\Bomberman\runs\phaseD-s2-20261022 > E:\Bomberman\runs\phaseD-s2-20261022\validation.log 2>&1
```

Chỉ khi `selected.json` có `open_test: true` / report `TEST_READY`, chạy:

```bat
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\run_phase_d_s2.py test --output E:\Bomberman\runs\phaseD-s2-20261022 > E:\Bomberman\runs\phaseD-s2-20261022\test.log 2>&1
```

`VALIDATION_STOP` là kết quả khoa học hợp lệ: giữ S2, đóng gói và đọc kết quả. Không chạy test. Package không ngầm train hoặc đánh giá:

```bat
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\run_phase_d_s2.py package --output E:\Bomberman\runs\phaseD-s2-20261022
```

## Dừng và tiếp tục

Ctrl+C giữ state episode/board đã commit. Với stage đã bắt đầu, thêm `--resume` vào chính lệnh đó, cùng output/arm/máy/runtime. Ví dụ:

```bat
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\run_phase_d_s2.py train --output E:\Bomberman\runs\phaseD-s2-20261022 --arm pool --resume > E:\Bomberman\runs\phaseD-s2-20261022\pool-resume.log 2>&1
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\run_phase_d_s2.py validate --output E:\Bomberman\runs\phaseD-s2-20261022 --resume > E:\Bomberman\runs\phaseD-s2-20261022\validation-resume.log 2>&1
E:\Bomberman\venv\Scripts\python.exe E:\Bomberman\run_phase_d_s2.py test --output E:\Bomberman\runs\phaseD-s2-20261022 --resume > E:\Bomberman\runs\phaseD-s2-20261022\test-resume.log 2>&1
```

Nếu register bị ngắt, chạy lại register cùng toàn bộ tham số và `--resume`. Không xóa reservation PENDING, sửa digest hoặc xóa `test_opened.json`. Task đánh giá đã commit được tái dùng; task dở đánh lại cả hai góc. Train resume từ cuối episode đã commit, bao gồm Adam/scheduler/RNG/pool; inference checkpoint đơn lẻ không đủ để resume.

## Bất biến và kết quả

Registration khóa checkpoint, config, spec, toàn bộ source dependency và runtime. CLI xác minh source hiện tại rồi thực thi bản `sources/` đã chụp; worker Windows spawn cũng dùng bản đó. Sau register, không sửa code/spec, cập nhật torch/numpy, đổi device hay chuyển máy trong cùng lượt. Git commit chỉ sửa nhật ký không đổi source được phép. Nếu phát hiện drift, runner từ chối; không có cờ bỏ kiểm tra hash hay ép mở test.

`PILOT_PASS` chỉ là cải thiện trong pilot; `PHASE_D_PASS` còn đòi WR ≥60% trước từng CNN/snapshot cũ, đủ 200 trận/cặp và giữ WR RuleBot. `PILOT_NOT_PASS` giữ S2. Report tách thắng chủ động/thụ động/vòng, CI theo board và chẩn đoán B. Lượt smoke không được quảng bá thành PASS thực nghiệm.

ZIP chỉ chứa output của lượt: registration/source/runtime/state/checkpoint/manifests/raw ledger/report. Log train/evaluation trong output được gói cùng ZIP; gửi kèm log register nằm cạnh output. Báo cáo incomplete không có quyền kết luận gate.

Smoke dùng output riêng với `register --smoke --device cpu` hoặc `cuda:0`, 4 episode/nhánh, max20 bước; seed 690000–690003. Nó không đặt chỗ dải full. Không dùng tốc độ 20 bước để cam kết thời gian trận 500 bước; đo và dự báo từng stage được ghi cùng bằng chứng kiểm chứng.

Bằng chứng kiểm chứng: `docs/superpowers/test-matrices/2026-10-22-phase-d-s2-verification.md` và JSON cùng tên. CPU/GPU smoke đều hoàn tất, peak RSS validation hai worker khoảng2,4GiB (đếm shared pages). GPU hoạt động nhưng chưa nhanh hơn CPU trong smoke batch1/20bước; môi trường và teacher chạy CPU. Dự phóng tuyến tính riêng từng stage có trong bằng chứng, không phải ETA lượt full.
