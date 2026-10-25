# Bằng chứng RED và tính hợp lệ của fixture — 24/10/2026

## RED đầu tiên

Lệnh cmd chính xác:

```bat
E:\Bomberman\venv\Scripts\python.exe -m unittest test_run_s3_map11 -v > docs\superpowers\specs\2026-10-24-s3-map11-acceptance-red.log 2>&1
```

Exit1, `Ran 33 tests in 0.022s`, `FAILED (errors=33)`. Cả33 lỗi đều là `ModuleNotFoundError: No module named 'run_s3_map11_distill'` trong setUp. Module đã được xác nhận chưa tồn tại. `py_compile` và import numpy/torch đều thành công; numpy2.4.6, torch2.11.0+cu128. Không có failure do dependency/syntax được tính là RED.

## Sửa fixture trước production, không hạ yêu cầu

Tác giả test độc lập thực hiện, không đọc implementation plan hay source runner mới. Giữ nguyên log RED đầu tiên, không ghi đè.

1. Giá trị cũ `COLLECT, VALIDATION, TEST = 1900000000, 1900000100, 1900000200` không hợp lệ ở full: validation400game/2góc chiếm1900000100–1900000299, giao nhau với TEST1900000200. Đổi duy nhất TEST thành1900001000; yêu cầu từ chối giao nhau được giữ nguyên. Đây là sửa lỗi setup, không nới kiểm tra.
2. Callback evaluation cũ phân loại checkpoint chỉ bằng stem: `last['model'] = name if name.startswith('epoch_') else 'S3'`. S3 gốc cũng tên `epoch_12.pt`, khiến fixture nhầm thành candidate12. Callback mới so đường dẫn resolve với đường dẫn S3 gốc trước khi dùng stem candidate. Không thay toán chọn model.
3. Packet được parent bổ sung rõ baseline hai arm đã được người dùng chọn: S3 thuần và S3+search3, mỗi arm `baseline_games`, cùng bàn/góc/lịch E. Assertion cũ `len(envs)==4` đổi thành8 ở smoke (4mỗi arm). Stub search3 được áp dụng cho baseline để giữ chạy ngắn; env/network/RuleBot vẫn thật.
4. Stub search cũ dùng `scores[legal[-1]]=1; return legal[-1],...`, thường đặt BOMB và dẫn tới trạng thái chỉ còn1nước hợp lệ. Probe fixture độc lập trên env11 thật cho mode quyết định đã tạo2/1anchor ở hai shard, nên không đạt tiền điều kiện “không anchor”. Stub đổi sang `scores[legal[0]]=1; return legal[0],...` để ở yên và giữ fixture có thông tin. Probe sau sửa: mode decisions `[48decision,0anchor]` cho cả hai shard; mode anchors `[0,3]`/`[0,6]`; mode mixed `[17,3]`/`[13,4]`. Probe gồm24game ngắn max12 trên seed giả, CPU một torch thread, RuleBot thật, S3 chuyển map11 bằng loader hiện có; không có search thật, gradient hoặc seed test thật.

## RED đóng băng sau sửa fixture

Lệnh cmd chính xác:

```bat
E:\Bomberman\venv\Scripts\python.exe -m unittest test_run_s3_map11 -v > docs\superpowers\specs\2026-10-24-s3-map11-acceptance-red-final.log 2>&1
```

Module vẫn chưa tồn tại khi chạy. Exit1, `Ran 33 tests in 0.021s`, `FAILED (errors=33)`; cả33 đều `ModuleNotFoundError: No module named 'run_s3_map11_distill'`. SHA256 suite đóng băng: `3cdd4d52037beca0e2a4dc68a1198c3316f94eb68f11d169b484deb5dc0af8cf`.

Không sửa production, nhật ký hoặc tracker trong subtask này; parent chịu trách nhiệm nhật ký theo AGENT.md.
