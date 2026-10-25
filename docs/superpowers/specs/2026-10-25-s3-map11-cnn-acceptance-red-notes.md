# S3-map11 CNN — bằng chứng RED độc lập, 25/10/2026

Tác giả: subagent `cnn11_acceptance`, chỉ nhận packet đóng băng `2026-10-25-s3-map11-cnn-evaluation.md`, AGENT/CLAUDE và code/test hiện tại; không đọc implementation plan hoặc production mới. Parent ghi nhật ký/tracker theo phân công. Không tạo `probe_s3_map11_cnn.py`, không chỉnh runner/source/checkpoint cũ.

Ma trận được tạo trước test executable. Suite cuối gồm26test; RED cuối chạy trước production mới. Tất cả26error là `ModuleNotFoundError: No module named 'probe_s3_map11_cnn'` tại `setUp`, exit1, `Ran 26 tests in 0.022s`. Không lỗi syntax/setup/dependency. AST parse test hợp lệ; torch2.11.0+cu128/numpy2.4.6 có sẵn trong venv. Cả4checkpoint trên đĩa được kiểm SHA256 và khớp packet. Không chơi game trong lượt RED.

Lệnh thực chạy (PowerShell, trong `E:\Bomberman`):

```powershell
$env:TEMP='E:\Bomberman\.superpowers\sdd\2026-10-25-s3-map11-cnn\tmp'
$env:TMP=$env:TEMP
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'E:\Bomberman\venv\Scripts\python.exe' -m unittest test_probe_s3_map11_cnn -v 2>&1 |
    Tee-Object -FilePath 'docs\superpowers\specs\2026-10-25-s3-map11-cnn-acceptance-red-final.log'
```

SHA256 của suite đã khóa: `49948d835e9bb8d9a1b8c3c2e54656ea43218d487ae3789b3e23f9bf10d018f1`.

SHA256 log RED cuối: `e020a7961d6c48ad03afc311fb390ef471375c1563ca1d52cc38fb114c2a7254`.

Log RED đầu giữ nguyên tại `2026-10-25-s3-map11-cnn-acceptance-red.log`:26error cùng nguyên nhân. Sau đó suite được bổ sung AST local-import closure của source bundle và mock thay đổi byte source thật, vẫn26test. Log cuối tương ứng suite cuối; log đầu không được gọi là baseline của phiên bản cuối.

**Thông số:** RED chưa chơi game/train. Suite GREEN dự kiến dùng seed smoke1900410000–1900410001,2learner(S3_11,S3)×3CNN(S3,S2,G),2góc,24game thật max12/11×11/2agent,CPU lấy mẫu+mask/no_grad,E475/25/40. Chỉ scheduler tuần tự thay cho process pool; loader/model/environment thật và guard cấm RuleBot/search/backward/optimizer. Smoke snapshot tái dùng tại đúng output đã đăng ký cho integrity cases; ca resume chơi thêm đúng1shard(2game ngắn). Synthetic metrics sửa raw hợp lệ rồi dùng public run/resume tổng hợp; bootstrap10000/seed20261014. Người chạy: agent trên máy workspace hiện tại,CPU/1worker/torch1thread; không dùng seedfull830000, không mở benchmark/full test.

Suite chấp nhận một số alias key hợp lệ cho config/summary để không khóa cách đặt tên không nêu trong packet. Shard theo convention JSON object `games`+metadata lồng hoặc top-level; source registration theo convention `sources_normalized_lf_sha256`. No gradient và hai mạng thật được quan sát ở real24game; synthetic metric tests không thay implementation nội bộ.

Giới hạn cần parent bổ sung trước launch: native Windows2worker smoke; full-gate unit boundaries5pp/CI dưới>0/suicide≤1pp; broad regression; independent review. Suite này kiểm arithmetic macro/CI theo whole-board trên2board synthetic và smoke không thểPASSfull, không tạo bằng chứng kết quả full200board. Không có training/promotion hoặc kết luận benchmark.
