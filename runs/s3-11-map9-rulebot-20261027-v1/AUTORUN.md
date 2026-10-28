# S3-11 thử lại trên 9×9 với RuleBot — 27/10/2026

Người dùng trực tiếp yêu cầu test S3-11 trên9×9 với RuleBot. Đánh giá cùng S3 gốc trên cùng bàn là đối chứng để đo giữ kỹ năng. Không train, không thay model phục vụ, không sửa source đã khóa, không push/commit Git trong task này.

## Hợp đồng

- Output duy nhất `E:\Bomberman\runs\s3-11-map9-rulebot-20261027-v1`; đọc registration.json và autorun-state.json trước hành động.
- S3_11 là epoch16 SHA `17748471…a95f`; S3 là epoch12 SHA `4185375f…69ed7`. SHA đầy đủ/config/runtime/source và bundle được khóa trong registration.
- Seed840000–840199 mới,200bàn×2góc=400trận/model,800trận tổng. Hai model cùng9×9/13kênh/2agent/max500/E475-25-40, CNN lấy mẫu+mask, CPU1torchthread/arm, tối đa2arm song song. Loader nội suy trọng số S3-11 từ11 xuống9; không cập nhật model.
- Thống kê mô tả: WR/ActiveWR/hòa/hòa hết bước/thua/tự nổ/passive/SDwin/SDdeath, chênh lệch S3_11−S3 và CI95 bootstrap ghép cặp theo200bàn,10000lần/seed20261014. Không đặt gate hoặc chọn model sau khi xem data; không tuyên bố noninferiority chỉ vì CI chứa0. Không dùng lại820000/830000 hoặc test840000 sau khi hoàn tất.
- Script trong run `evaluate_registered.py <S3_11|S3>` chỉ gọi evaluator và progress có sẵn, lock từngarm; output `evaluation-S3_11.json`, `evaluation-S3.json` ghi nguyên tử, đủ400rawgame mới đánh dấu complete. Không có automatic resume/replay nếu chết thiếu kết quả.

## Kiểm tra 45 phút

Đây là cadence tiếp tục từ yêu cầu trước. Mỗi lượt một snapshot, không poll/sleep-loop/log-stream. Query process native ngoài sandbox, so rootPID/start-time/command và child interpreter/owner lock. PowerShell ConvertFrom-Json có thể đã parse DateTime; dùng `[DateTime]... .ToUniversalTime()` để so, không parse lại làm mất timezone. Sandbox có thể không thấy process native; không kết luận chết bằng snapshot sandbox.

Nếu một trong haiarm còn chạy đúng định danh, chưa đủ2artifact, kết thúc DONT_NOTIFY yên lặng; không restart hoặc tạo process chồng. Nếu cảhai hoàn tất thì chốt. Nếu arm chết thiếu artifact, đọc tail log một lần, ghi stopped_incomplete và báo; không chạy lại hoặc đổi seed để cứu kết quả.

## Chốt kết quả

- Xác nhận native root/interpreter/children/lock đã kết thúc, không đánh giá thêm game. Import script qua runpy.run_path (không gọi main) và `verify_registration()` để kiểm19source/live+bundle/2model/config; đối chiếu runtime Python/NumPy/torch hiện tại với registration.
- Kiểm marker/configSHA, mỗiartifact complete/model/checkpointSHA/seed9×9/2agent/max500/sudden deathE,400trận/đủseed-góc. Raw field hữu hạn, outcome binary/nhất quán, steps1..500, counter nguyên không âm; win+draw≤1, active+passive+SDwin=win, active cần kills≥1, win không đồng thời suicide/SDdeath. Không sửa raw để hợp thức hóa.
- Dùng `probe_s3_map11_cnn._summarize(rows,500)` và `_paired(candidate_rows,reference_rows,config)` (các helper thống kê không phụ thuộc map) để tính lại9chỉ số và pairedCI từ raw; không gọi `probe_s3_map11_cnn.run` vì API đó riêngmap11. Lưu summary.json/rawSHA/proof, FINAL_REPORT.md; số liệu mỗigóc cũng lưu. Marker một lần, không ghi đè artifact arm hoặc chơi thêm game.
- Ghi kết quả thật và đủ thông số vào RESULTS.md, IMPROVEMENTS.md, hai PLAN, seedledger CLAUDE; cập nhật INSIGHTS nếu có bằng chứng mới. Giới hạn chỉ9×9/1v1RuleBot/lịchE/mộtcheckpoint; không suy CNN/FFA hoặc mọi map.
- State cuối completed hoặc stopped_incomplete. Xóa chính heartbeat với automation_update(mode=delete), ID trong state; lưu xác nhận và báo rõ đã gỡ lịch. Không tạo chat mới, nhắn nơi khác hoặc archive chat.
