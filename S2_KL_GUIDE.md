# S2-KL — hướng dẫn pilot giữ kỹ năng S2

Thiết kế: `docs/superpowers/specs/2026-10-23-s2-kl-retention-design.md`.
Runner: `run_s2_kl_probe.py`. Việc triển khai/kiểm chứng code không khởi chạy lượt full.

## Phép thử cố định

Hai nhánh đều bắt đầu từ S2 `runs/sd8a-20261020/train/epoch_16.pt`, Adam/scheduler mới, cùng seed và lịch bàn/góc, chỉ đấu RuleBot. `control` có KL0; `anchor` có KL1 theo chiều learner||S2 với cùng safety mask. Reference S2 được dựng trước khi restore learner, giữ eval/no-gradient.

Mỗi nhánh 500 episode, train tuần tự trên CUDA:0. Checkpoint 0/100/250/500; chỉ endpoint500 quyết định. Lịch cosine và entropy có tổng500episode; không coi control này là prefix của D-S2 lịch2000 cũ. Các tham số khác đã khóa trong registration, CLI không cho đổi số episode/lr/beta.

Train710000–710249; validation720000–720099,100board×2góc/model; tối đa7SHA=1400trận. Offline1024decision+1024anchor TRAIN states SD8A theo manifest đã khóa; forward-only CPU, không replay/gradient/search. Validation CPU một thread/worker, tối đa2worker. Không có stage test hoặc tự thay S2 mặc định.

## Chuẩn bị CMD

Các lệnh dưới đây dùng **Command Prompt (CMD)** trên cùng máy/runtime. Output full phải là thư mục mới. Chỉ chạy khi đã có yêu cầu khởi chạy pilot; không mở lại held-out hoặc tiếp tục output D-S2 cũ.

```bat
cd /d E:\Bomberman
set "PY=E:\Bomberman\venv\Scripts\python.exe"
set "RUNNER=E:\Bomberman\run_s2_kl_probe.py"
set "OUT=E:\Bomberman\runs\s2-kl-20261023"
```

Input cần còn nguyên S2/metadata SD8A, collection và manifest tại `runs/phaseD-s2-20261022-analysis-20261023/offline_sample_manifest.json`. SHA manifest `9b36b8e07ed8f65175e5c9fbf53a18eca16ae791f0073d19730b28e75529e333`. Không chép toàn bộ runs, junction hoặc đổi checkpoint gốc để vượt kiểm SHA.

## Đăng ký rồi chạy từng stage

```bat
"%PY%" -u "%RUNNER%" register --output "%OUT%" --input-root E:\Bomberman --device cuda:0 --workers 2 >> E:\Bomberman\runs\s2-kl-20261023-register.log 2>&1
```

Registration rà lịch sử cả runs chính và game UI/runs, dùng cùng registry/OS lock với D-S2, đặt chỗ PENDING→COMPLETE và copy source/manifest. Va chạm hoặc metadata không ánh xạ được thì dừng; không tự đổi seed. CUDA thiếu sẽ báo lỗi, không tự chuyển CPU.

Audit ban đầu ngày23/10 từ chối full do `runs/sd6-diagnostics-6600/overrides.json` có seed=null và hai `runs/t2-{full,smoke}-20261014-060320/replay_cases.json` lớn hơn32MiB. Sau yêu cầu khởi chạy, đã đọc toàn bộ hai replay (chỉ có seed28000–28003), kiểm source generator của70 dòng null và lập index `s2_kl_history.py` gắn SHA chính xác của cả ba tệp cùng generator. SD6 được phủ bảo thủ6600–6649, không gán seed giả cho từng dòng. Tệp khác hoặc bất kỳ SHA/evidence thay đổi vẫn bị từ chối; không bỏ qua/xóa log. Index là dependency được đóng băng cùng source. Bằng chứng đọc-only và kiểm thử: `runs/s2-kl-launch-preflight-20261023`.

Sau mỗi lệnh kiểm `echo %ERRORLEVEL%` ngay lập tức; chỉ chạy stage tiếp theo khi bằng0.

```bat
"%PY%" -u "%RUNNER%" train --output "%OUT%" --arm control >> "%OUT%\logs\control-console.log" 2>&1
```

```bat
"%PY%" -u "%RUNNER%" train --output "%OUT%" --arm anchor >> "%OUT%\logs\anchor-console.log" 2>&1
```

```bat
"%PY%" -u "%RUNNER%" diagnose --output "%OUT%" >> "%OUT%\logs\diagnose-console.log" 2>&1
```

```bat
"%PY%" -u "%RUNNER%" validate --output "%OUT%" >> "%OUT%\logs\validate-console.log" 2>&1
```

```bat
"%PY%" -u "%RUNNER%" package --output "%OUT%" >> "%OUT%\logs\package-console.log" 2>&1
```

Train dùng GPU cho forward/backward; môi trường, RuleBot và teacher vẫn chạy CPU nên GPU có thể hoạt động ngắt quãng. Offline và validation chủ động dùng CPU. Progress từ `progress.py` ưu tiên CON khi có Windows console, nên vẫn hiện khi stdout/stderr chuyển vào log; khi console không khả dụng thì dùng stderr/log. `>>` giữ log khi resume. Không cần tạo monitor hoặc polling sau launch.

## Dừng và resume

Ctrl+C trả mã130. Full-state lưu model/Adam/scheduler/Python/numpy/torch/CUDA/env RNG và context tại cuối episode đã commit. Episode chưa commit được chạy lại từ state cũ. Không xóa state rồi warm-start lại nếu đã có artifacts committed.

```bat
"%PY%" -u "%RUNNER%" train --output "%OUT%" --arm control --resume >> "%OUT%\logs\control-console.log" 2>&1
"%PY%" -u "%RUNNER%" train --output "%OUT%" --arm anchor --resume >> "%OUT%\logs\anchor-console.log" 2>&1
"%PY%" -u "%RUNNER%" diagnose --output "%OUT%" --resume >> "%OUT%\logs\diagnose-console.log" 2>&1
"%PY%" -u "%RUNNER%" validate --output "%OUT%" --resume >> "%OUT%\logs\validate-console.log" 2>&1
```

Chọn đúng stage/arm bị ngắt và kiểm errorlevel sau từng lệnh. Arm đã đủ500 có thể resume để tái dựng projection; không train thêm. Diagnostic/modelSHA và task/board có đủ hai góc đã commit được kiểm rồi tái dùng, không cộng lặp. Shard/summary/source/runtime sai bị từ chối.

Validation yêu cầu đủ diagnostics của tất cả SHA trước khi chạy bất kỳ trận nào; diagnostics thiếu hoặc mới có một phần sẽ báo lỗi để bạn chạy/resume `diagnose` trước. Identity của lượt gắn với đúng đường dẫn output đã đăng ký: bản copy ở thư mục khác không thể tiếp tục cùng reservation. Backup chỉ được phục hồi về đúng output ban đầu trên cùng máy/runtime.

Nếu ngắt lúc đăng ký, dùng nguyên lệnh register và thêm `--resume`, giữ đúng output/input-root/device/workers/mode. Giữ cả thư mục `.registration-pending` và registry để phục hồi cùng identity. Không xóa reservation của lượt khác. Cùng một output chỉ một writer.

Không chuyển máy, nâng torch/numpy, sửa source/spec đã đăng ký hoặc đổi config giữa lượt. Source worker chạy từ `OUT\sources`; live source cũng được kiểm SHA trước dispatch.

## Đọc báo cáo

`report.json` và `decision_report.md` tái dựng từ raw diagnostics và raw board ledger. `validation.json` gồm tổng và theo góc. Có WR/ActiveWR/hòa/tự nổ/thắngSD/chếtSD/thắng thụ động khác, kills/bombs/steps và CI paired theo board. Mọi tỷ lệ dùng tổng số trận làm mẫu số. WR=ActiveWR+thắngSD+thắng thụ động khác; các sự kiện chết có thể thuộc hòa.

Offline R là tỷ lệ argmax thuộc tập nước tốt đồng hạng, Q là xác suất dành cho tập đó; CE và KL(S2||model) là diagnostics. KL offline có chiều ngược KL train. Dữ liệu này từng train S2, không phải độ chính xác độc lập. Value mean không đo calibration khi không có target. Global grad_norm/loss không cho biết gradient từng thành phần.

CI retention:10000cluster-bootstrap board, seed20261023, giữ micro theo state. CI RB giữ cả hai góc/board và ghép cùng board giữa model. Không làm tròn trước gate. Endpoint500 cố định;100/250 chỉ chẩn đoán.

| Status | Ý nghĩa |
|---|---|
| INCOMPLETE | Thiếu raw/inventory; không kết luận chất lượng |
| INCONCLUSIVE_NO_DRIFT | Control không mất R ít nhất10pp; pilot chưa tái hiện điều kiện kiểm |
| RB_RETENTION_FAIL | Anchor không giữ RB ActiveWR trong2pp hoặc tự nổ tăng quá1pp |
| RETENTION_NOT_PRESERVED | R hoặc Q anchor giảm quá5pp so S2 |
| NO_CLEAR_ANCHOR_BENEFIT | Delta R anchor−control chưa đạt10pp hoặc CI lower<=0 |
| RETENTION_PILOT_PASS | Đạt toàn bộ sáu điều kiện; chỉ cho phép đề xuất pool pilot mới |
| SMOKE_ONLY | Smoke kiểm đường chạy; không có kết luận khoa học |

Mã CLI0 nghĩa stage hoàn tất, kể cả kết quả khoa học không đạt;1 là invalid/drift;130 là ngắt. `package` được phép đóng output chưa đủ thành INCOMPLETE, không tự train/evaluate. ZIP ở `<OUT>-send.zip`, chỉ chứa output mới và không chép collection gốc. Giữ S2 mặc định sau mọi status.

## Kiểm chứng code và giới hạn dự phóng

Bằng chứng: `docs/superpowers/test-matrices/2026-10-23-s2-kl-verification.md` vàJSON kèm theo; acceptance cố định trong `docs/superpowers/test-matrices/2026-10-23-s2-kl-retention.md`.

```bat
set S2_KL_ACCEPTANCE_CUDA=1
"%PY%" -m unittest test_s2_kl_protocol test_s2_kl_training test_s2_kl_evaluation test_run_s2_kl_probe -q
set S2_KL_ACCEPTANCE_CUDA=
```

Smoke riêng4episode/arm,max20bước; train790000–790001 và validation790002–790003, chỉ audit trong output, không đặt chỗ dải full. Code smoke đầu dùng `runs/s2-kl-code-smoke-20261023-cpu` và `...-cuda`; bản sau review dùng hậu tố `-review`, bản chốt sau audit dùng hậu tố `-final`, có đủ register/train/diagnose/validate/package. Giữ các output trước làm bằng chứng lịch sử; source thay đổi thì output cũ sẽ bị từ chối resume theo SHA.

`logs/stages.jsonl` ghi thời gian stage sau import và peak RSS của tiến trình stage; raw board payload ghi PID/peak RSS riêng từng worker. Hai số này không cộng thành peak RAM toàn cây tiến trình. Bản verification còn ghi walltime CLI có startup/dispatch. Smoke20bước kiểm tính đúng và resume; không suy ETA500bước bằng phép nhân25. Ngân sách full tối đa500000learner steps train và700000environment steps validation; cần đo thêm throughput500bước trên dải smoke riêng trước khi hứa thời gian chạy. Offline có cùng2048quan sát/model, tối đa7model; dự phóng offline từ sốSHA là ước lượng có giới hạn startup/I/O/runtime.
