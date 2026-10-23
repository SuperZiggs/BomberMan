# SD9.1 — probe: S3 có bớt hòa khi đấu CNN không

Ngày: 23/10/2026. Người dùng đồng ý ("ok viết probe S3 đi"). Mọi luật dưới đây chốt trước khi chạy. Runner: `probe_s3_cnn.py`. Không train.

## Câu hỏi

SD9 cho S3 (CNN thuần) đạt ActiveWR 92.25 khi đấu RuleBot, nhưng mọi số đó đều là đấu RuleBot. PDE/PDE2/D-S2 cho thấy S2 hòa khoảng 60–65% khi đấu CNN, ActiveWR chỉ khoảng 6%, và Phase D bằng RL (D-S2) không sửa được việc này. Câu hỏi: **S3 có bớt hòa rõ rệt khi đấu CNN so với S2 không?** Kết quả quyết định có cần một hướng train mới để đấu CNN hay không.

## Thiết kế

- **Learner:** S3 (`runs/sd9-20261023/train/epoch_12.pt`, `4185375f…69ed7`) và S2 (`runs/sd8a-20261020/train/epoch_16.pt`, `d222d81f…6814`). Cả hai là CNN thuần, lấy mẫu.
- **Đối thủ:**
  - Ba CNN lấy mẫu: S2, G (`bb0c775c…c7ac`) và S3. Mỗi learner gặp chính nó một lần và hai model còn lại.
  - G + `search_action3` (đoán đối thủ bằng RuleBot), chỉ để mô tả.
- **Bàn:** seed validation 21000. Đấu CNN dùng 200 bàn × 2 góc = 400 trận mỗi cặp (bàn 21000–21199). Đấu G + search3 dùng 50 bàn × 2 góc = 100 trận (21000–21049), vì mỗi trận mất khoảng 140–250 giây.
- **Luật chơi:** 9×9, 1v1, max_steps 500, lịch E 475/25/40.
- **So sánh:** S3 − S2 ghép cặp theo bàn với từng đối thủ. Bootstrap theo bàn 10 000 lần, seed 20261014.
- **Macro CNN:** trung bình ba đối thủ CNN theo từng (bàn, góc), rồi mới bootstrap theo bàn.
- Báo thêm: hòa do hết 500 bước, chết vì vòng, thua, s/trận.

## Luật chốt trước

**"S3 bớt hòa rõ"** khi đủ bốn điều kiện trên macro CNN:
1. Hòa giảm ≥ 15pp;
2. CI trên của hiệu hòa < 0;
3. ActiveWR tăng ≥ 5pp;
4. CI dưới của hiệu ActiveWR > 0.

- **Nếu đạt:** đề xuất kiểm lại trên seed test mới 58000 (S3 so với S2 khi đấu CNN). Chưa cần hướng train mới để đấu CNN.
- **Nếu không đạt:** S3 vẫn hòa nhiều khi đấu CNN. Chọn giữa hai hướng sau khi có kết quả S2-KL của bạn:
  - A: chưng cất search3 khi đấu CNN, search đoán đối thủ bằng CNN đó;
  - B: RL có KL neo.
- Kết quả đấu G + search3 chỉ để mô tả, không quyết định gì.
- Không dùng seed test nào trong probe này.

## Chi phí

Chạy thử 1 bàn × 8 cặp với 5 tiến trình:
- trận CNN đấu CNN mất ~2 s;
- trận đấu G + search3 mất ~140–250 s.

Tổng cộng 2400 trận CNN và 200 trận có search, ước tính khoảng 2–3 giờ với 5 tiến trình. Có `--resume`.

## Lệnh (cmd)

```
D:\anaconda\envs\torch\python.exe probe_s3_cnn.py --output runs/sd91-21000 --workers 5 > runs\sd91-21000-console.log 2>&1
REM Nếu bị ngắt:
D:\anaconda\envs\torch\python.exe probe_s3_cnn.py --output runs/sd91-21000 --workers 5 --resume > runs\sd91-21000-resume-console.log 2>&1
```

## Giới hạn

- Chỉ dùng seed validation và một bộ checkpoint.
- Đối thủ CNN lấy mẫu nên kết quả có nhiễu.
- G + search3 chỉ có 100 trận mỗi cặp.
- Probe không tách được nguyên nhân hòa, chẳng hạn mask an toàn cấm đi vào ô sát đối thủ. Nó chỉ báo hòa do hết giờ và các tỉ lệ.
