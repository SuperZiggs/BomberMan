# SD9.3: probe vòng sudden death sớm hơn có biến hòa thành thắng chủ động không

Ngày 24/10/2026. Người dùng đồng ý ("ok") sau SD9.2. Mọi luật dưới đây được chốt trước khi chạy. Runner: `probe_sd_schedule.py`. Probe không train và không đổi luật chính thức.

## Câu hỏi

SD9.2 cho thấy hòa giữa S3 và CNN chủ yếu do cấu trúc: chỉ 15.9% trận hòa có kế hoạch giết. Lịch E (475/25/40) chỉ đóng vòng 1 ở bước 475 và vòng 2 ở bước 500, đúng lúc trận hết giờ. Nếu cho vòng đóng sớm hơn, bàn chơi hẹp lại trong khi còn nhiều bước, thì:
- các trận hòa có chuyển thành **thắng chủ động** không;
- hay chỉ chuyển thành thắng nhờ vòng hoặc thắng thụ động (WR đẹp lên nhưng agent không giỏi hơn)?

## Thiết kế

- **Learner:** S3 (`4185375f…69ed7`), lấy mẫu.
- **Đối thủ:** S2, G và S3 (CNN, lấy mẫu), cộng thêm RuleBot làm cột kiểm tra.
- **Lịch vòng** (cùng khoảng cách 25 bước và cảnh báo 40 bước như E):

| Lịch | Bắt đầu | Các vòng đóng ở bước |
|---|---|---|
| E | 475 | 475, 500 (tham chiếu) |
| S400 | 400 | 400, 425, 450 |
| S350 | 350 | 350, 375, 400 |
| S300 | 300 | 300, 325, 350 |

  Trên bàn 9×9, vòng 3 lấp kín mọi ô trống. Vì vậy với lịch bắt đầu ở bước t, mọi trận kết thúc trước bước t + 50 (test đã kiểm với S300). Hai bên còn sống khi vòng 3 đóng thì cùng chết, tính là hòa.
- **Bàn:** seed validation 21000, 200 bàn (21000–21199) × 2 góc = 400 trận cho mỗi cặp (lịch, đối thủ), tổng 6400 trận.
- **Luật chơi:** 9×9 1v1, max_steps 500.
- **Kiểm tra tái lập:** với lịch E, các trận đấu CNN chính là trận của SD9.1 chơi lại, nên runner so lại với shard SD9.1.
- **So sánh:**
  - Lịch X − E, ghép cặp theo bàn, bootstrap theo bàn 10 000 lần, seed 20261014.
  - Macro CNN: trung bình ba đối thủ CNN theo từng (bàn, góc), giống SD9.1.
  - Báo đủ các chỉ số: ActiveWR, WR, thắng nhờ vòng, thắng thụ động, hòa, thua, tự nổ, chết vì vòng.

## Luật chốt trước

Lịch X được coi là **"giúp"** nếu đủ sáu điều kiện, xét so với E:
1. Hòa (macro CNN) giảm ≥ 15pp;
2. CI trên của hiệu hòa < 0;
3. ActiveWR (macro CNN) tăng ≥ 5pp;
4. CI dưới của hiệu ActiveWR > 0;
5. Tự nổ (macro CNN) tăng không quá 1pp;
6. ActiveWR khi đấu RuleBot giảm không quá 3pp (ước lượng điểm).

**Chọn lịch:**
- Trong các lịch "giúp", chọn lịch có mức tăng ActiveWR macro lớn nhất.
- Nếu các lịch chênh nhau trong vòng 1pp, chọn lịch bắt đầu muộn nhất, tức gần E nhất.

**Nếu có lịch được chọn:**
- Đề xuất kiểm lại trên seed test mới 58000: S3 đấu CNN dưới lịch đó so với E.
- Bàn với bạn trước khi đổi luật chính thức, vì mọi kết quả cũ đều đo bằng lịch E.

**Nếu không lịch nào "giúp":**
- Vòng sớm hơn không tạo thắng chủ động.
- Giữ E và chấp nhận hòa khi hai CNN đấu nhau là tính chất của game với mask an toàn chung.
- Khi đấu CNN, báo ActiveWR thay vì WR.

**Khác:** không dùng seed test nào trong probe này.

## Chi phí

- SD9.1 đo được khoảng 1.7 s/trận CNN với lịch E. Lịch sớm hơn làm trận ngắn hơn, trận đấu RuleBot nhanh.
- Ước tính tổng 30–45 phút với 5 tiến trình. Có `--resume`.

## Lệnh (cmd)

```
D:\anaconda\envs\torch\python.exe probe_sd_schedule.py --output runs/sd93-21000 --workers 5 > runs\sd93-21000-console.log 2>&1
REM Nếu bị ngắt:
D:\anaconda\envs\torch\python.exe probe_sd_schedule.py --output runs/sd93-21000 --workers 5 --resume > runs\sd93-21000-resume-console.log 2>&1
```

## Giới hạn

- S3 được train dưới lịch E. Lịch sớm hơn nằm ngoài phân bố train, nên S3 có thể chết vì vòng nhiều hơn. Kết quả đo S3 như hiện tại, không phải một model được train cho lịch mới.
- Chỉ dùng seed validation và một bộ checkpoint.
- Đối thủ lấy mẫu nên có nhiễu.
- Chỉ thử khoảng cách 25 bước và max_steps 500.
