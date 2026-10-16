# Phase C Sudden Death Design

## Mục tiêu

Bổ sung cơ chế vòng bo thu nhỏ bản đồ để giảm các trận hòa kéo dài và buộc hai agent tiếp cận nhau, đồng thời giữ nguyên khả năng né bom và giao tranh đã học từ checkpoint Phase C hiện tại.

Checkpoint khởi đầu được khuyến nghị là `runs/pressure-C-next/final_phaseC.pt`. Không ghi đè checkpoint, metrics hoặc kết quả huấn luyện cũ.

## Phạm vi

- Thêm vòng bo tùy chọn vào `BombermanFFAEnv`.
- Thêm một kênh quan sát cảnh báo vòng sắp đóng, nâng observation từ 12 lên 13 kênh.
- Migrate checkpoint 12 kênh bằng cách giữ nguyên trọng số cũ và khởi tạo trọng số kênh mới bằng `0`.
- Đưa vòng bo vào safety forecast/action mask để agent không chọn đường thoát sẽ bị đóng trước khi tới nơi.
- Thêm cấu hình curriculum Phase C-SD vào trainer và cấu hình vòng bo vào evaluator.
- Ghi riêng chết do vòng bo, thắng bằng bom và thắng thụ động.
- Thêm launcher PowerShell chạy fine-tune trong terminal, không ghi log terminal xuống file.

Không tự động chạy lượt fine-tune dài trong thay đổi này. Chỉ chạy test và smoke training ngắn để xác nhận pipeline.

## Luật vòng bo

`BombermanFFAEnv` nhận các tham số mới:

- `sudden_death: bool = False`
- `sudden_death_start: int = 350`
- `sudden_death_interval: int = 20`
- `sudden_death_warning_steps: int = 40`

Mặc định `sudden_death=False` để các lệnh và checkpoint hiện tại giữ nguyên luật chơi.

Khi bật, vòng trong đầu tiên đóng tại `sudden_death_start`. Mỗi `sudden_death_interval` bước, một vòng trong kế tiếp biến thành `RIGID_WALL`. Vòng biên ngoài có sẵn không được tính là một lần đóng. Vòng đóng ở đầu bước, trước khi xử lý hành động.

Khi một vòng đóng:

- Agent trên ô bị đóng chết và được ghi vào `info["sudden_death_deaths"]`.
- Bom trên ô bị đóng bị xóa và bộ đếm bom của chủ sở hữu được hoàn trả.
- Lửa, power-up và tường gỗ trên ô bị đóng bị xóa trước khi ô trở thành `RIGID_WALL`.
- Cái chết do vòng bo không tạo kill credit và không được tính là `ActiveWR`.

Tham số thời gian không hợp lệ phải bị từ chối: `start >= 1`, `interval >= 1`, `warning_steps >= 0`.

## Quan sát và tương thích checkpoint

Kênh 12 mới chứa mức cảnh báo đóng vòng:

- `0` cho ô không nằm trong vòng sắp đóng hoặc khi Sudden Death tắt.
- Giá trị tăng tuyến tính từ gần `0` tới `1` trong `sudden_death_warning_steps` bước trước khi vòng đóng.
- `1` ở bước cuối trước khi đóng.

`OBS_CHANNELS` tăng từ 12 lên 13. `smart_load_weights` phải tiếp tục nạp checkpoint cũ bằng cách copy đủ 12 kênh cũ vào `conv.0.weight[:, :12]` và đặt toàn bộ trọng số kênh 13 bằng `0`. Với Sudden Death tắt, logits và value của checkpoint đã migrate phải khớp checkpoint 12 kênh trong sai số số thực hiện có.

## Safety forecast

Environment cung cấp lịch đóng vòng tương đối cho vòng kế tiếp. Safety mask dùng lịch này như các ô bị chặn theo thời gian:

- Không chọn bước di chuyển nếu đường sống duy nhất yêu cầu đứng trên ô đã đóng tại tick tương ứng.
- Không cho đặt bom nếu đường thoát chỉ tồn tại qua vòng sẽ đóng trước khi agent thoát khỏi vùng nổ.
- Không thay đổi mask khi Sudden Death tắt hoặc vòng kế tiếp nằm ngoài cửa sổ cảnh báo.

RuleBot, trainer, evaluator và GUI tiếp tục dùng chung `get_action_mask`, nên không tạo một bộ luật an toàn riêng cho từng caller.

## Curriculum Phase C-SD

Trainer nhận thêm:

- `--sudden_death_probability`, số thực trong `[0, 1]`, mặc định `0.0`.
- `--sudden_death_start`, mặc định `350`.
- `--sudden_death_interval`, mặc định `20`.
- `--sudden_death_warning_steps`, mặc định `40`.

Mỗi episode bật Sudden Death theo `sudden_death_probability` bằng RNG huấn luyện đã seed. Cấu hình khuyến nghị cho pilot là xác suất `0.75`, start `350`, interval `20`, warning `40`, 300–500 episode, warm-start từ `runs/pressure-C-next/final_phaseC.pt` và lưu sang thư mục mới.

Lượt dài chỉ được chạy sau khi pilot đạt tiêu chí nghiệm thu. Curriculum tiếp theo giảm `start` xuống 300 rồi 250; không giảm thẳng xuống 200 trước khi có benchmark độc lập.

## Metrics và chọn checkpoint

Metrics bổ sung:

- `sudden_death_enabled`
- `sudden_death_deaths`
- `sudden_death_wins`
- `active_wins`
- `passive_wins`
- `draws`

Một trận thắng chỉ là `active_win` khi learner có ít nhất một kill bằng bom. Đối thủ chết vì vòng bo tạo `sudden_death_win`, không tạo kill và không mở khóa terminal win reward. Logic chống thắng thụ động hiện tại vẫn giữ nguyên.

Checkpoint tốt nhất tiếp tục ưu tiên ActiveWR và guard tự nổ. WR tổng hoặc giảm hòa do vòng bo không đủ để chọn model.

## Evaluator và launcher

`evaluate.py` nhận cờ `--sudden_death` cùng ba tham số thời gian. JSON kết quả phải ghi cấu hình và tách `active_win_rate`, `sudden_death_win_rate`, `passive_win_rate`, `suicide_rate`, `sudden_death_death_rate` và `draw_rate`.

Tạo `run_train_phaseC_sd.ps1` để fine-tune từ checkpoint hiện tại sang một thư mục mới. Script in trực tiếp ra PowerShell, không dùng transcript, redirect hoặc `Tee-Object`.

## Tiêu chí nghiệm thu

1. Khi Sudden Death tắt, cùng seed và action sequence tạo trạng thái/reward/info giống trước thay đổi, ngoại trừ observation có thêm một kênh toàn `0`.
2. Vòng đóng đúng bước, đúng thứ tự và không để lại bom/lửa/power-up trong tường mới.
3. Chết do vòng bo không tạo kill credit hoặc ActiveWR.
4. Warning channel phản ánh đúng vòng kế tiếp và thời gian còn lại.
5. Safety mask tránh đường thoát bị vòng bo đóng nhưng không đổi hành động hợp lệ khi chế độ tắt.
6. Checkpoint 12 kênh migrate sang 13 kênh và giữ nguyên đầu ra khi warning channel bằng `0`.
7. Trainer tái lập được chuỗi episode bật/tắt vòng bo với cùng seed.
8. Evaluator và JSON báo cáo tách riêng kết quả do bom, vòng bo, tự nổ và hòa.
9. Test hiện có vẫn qua; smoke training tạo checkpoint/metrics hữu hạn trong thư mục tạm.

## Ngưỡng trước khi chạy fine-tune dài

Pilot được coi là đủ an toàn để chạy tiếp khi benchmark cố định cho thấy:

- Self-death không tăng quá 1 điểm phần trăm so với checkpoint warm-start trên cùng luật.
- ActiveWR không giảm quá 2 điểm phần trăm.
- Draw rate giảm ít nhất 5 điểm phần trăm.
- `sudden_death_win_rate` được báo riêng và không được dùng để bù cho ActiveWR giảm.

Nếu pilot không đạt, giữ checkpoint hiện tại làm model chính và chỉ điều chỉnh lịch vòng bo hoặc safety forecast; không tăng thưởng đặt bom và không train lại Phase C từ đầu.

## Ngoài phạm vi

- Không thay đổi reward đặt bom, reward phá tường, CNN ngoài kênh input mới hoặc RuleBot chiến thuật.
- Không thay model chính bằng checkpoint mới chỉ dựa trên rolling training metrics.
- Không đánh giá Phase D/E hoặc FFA 3–4 agent trong lượt đầu.
- Không xóa hoặc ghi đè artifacts hiện có.
