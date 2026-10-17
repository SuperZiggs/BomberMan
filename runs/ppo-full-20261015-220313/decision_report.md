# T4-D1 — U4/P4 có đối chứng

**DONE/REVIEWED. Gate FAIL; giữ checkpoint B.**

Hai nhánh đủ 200 episode từ B, validation0/100/200 và test400/model. Thời gian từ đăng ký đến hoàn tất 106.89 phút, gồm cả khoảng dừng. Giữ model mặc định B; test dùng 200 seed bàn chung, mỗi bàn cả A0/A1, không coi1.200game là1.200bàn độc lập.

## Thiết kế

Cả hai warm-start từ B với Adam mới, seed2026101507, C9×9/2agents/RuleBot/max500, MC advantage raw và critic MC, local teacher, wall0,15, LR1e-4. Full batch một learner episode, tối đa4epochs. U4 dùng ratio chưa clip; P4 clip epsilon0,20. Cùng targetKL0,02: nếu exact categorical KL sau update>0,03 thì dừng các epoch còn lại. Step vượt ngưỡng vẫn giữ; không coi guard là bảo đảm KL không bao giờ vượt ngưỡng. Scheduler và entropy schedule theo episode, teacher losses dùng cùng nhãn và hệ số trong mọi epoch.

P4 dừng tại episode36/Adam144, rồi resume từ full state sau khi người dùng giải phóng VRAM. Episode37 nối tiếp với Adam148; cả10file U4 đã hoàn tất giữ nguyên hash. Deadline ban đầu không được gia hạn. `verification/before-resume.json` và `verification/resume_review.json` lưu kiểm tra này.

Không thêm GAE/normalization, reward/observation/mask mới hoặc opponent pool. B là policy baseline giữ nguyên; D1 này chưa train thêm đối chứng AC một-update. Epoch/Adam counts thực có thể khác vì cùng guard phản ứng với mỗi policy.

## Best chọn bằng validation

| Nhánh | Episode | ActiveWR % | Tự nổ % | Hòa % | Chọn |
| --- | --- | --- | --- | --- | --- |
| U4 chưa clip | 0 | 29.00 | 0.00 | 62.00 | True |
| U4 chưa clip | 100 | 36.00 | 0.00 | 55.00 | True |
| U4 chưa clip | 200 | 37.00 | 0.00 | 51.00 | True |
| P4 clip 0,20 | 0 | 29.00 | 0.00 | 62.00 | True |
| P4 clip 0,20 | 100 | 30.00 | 0.00 | 63.00 | True |
| P4 clip 0,20 | 200 | 31.00 | 0.00 | 59.00 | True |

Khóa best/hash trước test37000–37199. Final200 chỉ dùng diagnostics/skill validation, không chọn lại model sau test.

## Test400/model

| Model | Best ep | ActiveWR % | Passive WR % | Hòa % | Tự nổ % | Kills/game |
| --- | --- | --- | --- | --- | --- | --- |
| B | gốc | 37.50 | 6.25 | 53.50 | 0.75 | 0.375 |
| U4 chưa clip | 200 | 37.00 | 6.25 | 54.25 | 1.00 | 0.370 |
| P4 clip 0,20 | 200 | 32.00 | 6.75 | 58.75 | 1.00 | 0.320 |

| P4 so với | Delta ActiveWR pp | CI95% pp | Board pairs |
| --- | --- | --- | --- |
| B | -5.50 | [-11.25; 0.50] | 200 |
| U4 chưa clip | -5.00 | [-11.00; 1.00] | 200 |

Bootstrap10.000 lần theo cặp seed bàn, giữ cả hai góc trong cùng cụm. Gate cần delta≥5pp so với cảB/U4, hai cận dướiCI>0 và tự nổ quan sát không cao hơn. Kết quả điều kiện: {'gain': False, 'ci': False, 'suicide': False}.

Cả hai khoảng tin cậy của P4 so với B/U4 đều chứa0 trong lượt này. Điểm quan sát của P4 thấp hơn, nhưng dữ liệu chưa chứng minh chênh lệch thực khác0; kết luận vận hành là không đạt gate thay model, không phải chứng minh PPO nói chung kém hơn.

## Số updates và tác động KL/clipping

| Nhánh | Learner steps | Adam steps | Epoch counts/200ep | Early-stop episodes | KL vượt ngưỡng | PPO clip fraction, weighted | Max post-KL | Runtime min | Peak CUDA MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U4 chưa clip | 76920 | 800 | {'4': 200} | 0 | 0 | 0.0442% | 0.004992 | 29.05 | 348.63 |
| P4 clip 0,20 | 75999 | 800 | {'4': 200} | 0 | 0 | 0.0507% | 0.004997 | 39.41 | 347.64 |

Clip fraction đo ratio ngoài[0,8;1,2] trước mỗi update, tính trọng số theo sample-updates; U4 chỉ log, không clip. Nó không đồng nghĩa mọi sample ngoài khoảng đều bị chặn gradient vì dấu advantage quyết định surrogate nào được chọn. Các epoch thực giữ cùng labels/targets; gradient clip5 là cơ chế khác với PPO ratio clipping. Critic EV/raw advantage trong metrics tính từ policy cũ; value_loss từng epoch từ prediction hiện tại. Peak CUDA là bộ nhớ tensor allocated mà PyTorch ghi nhận trong tiến trình, không phải tổng VRAM theo nvidia-smi; với P4 đây là peak của đoạn sau resume. Runtime mỗi nhánh gồm train/validation/skill probes, cộng các đoạn chạy trước/sau resume và không cộng quãng tiến trình tắt. Điều kiện VRAM thay đổi giữa lượt P4 nên chênh runtime giữa hai nhánh chưa đo riêng chi phí của clipping.

## Từng góc

| Model | Góc | ActiveWR % | Passive WR % | Hòa % | Tự nổ % | Kills/game |
| --- | --- | --- | --- | --- | --- | --- |
| B | A0 | 36.50 | 6.00 | 52.50 | 1.50 | 0.365 |
| B | A1 | 38.50 | 6.50 | 54.50 | 0.00 | 0.385 |
| U4 chưa clip | A0 | 34.00 | 8.00 | 53.50 | 2.00 | 0.340 |
| U4 chưa clip | A1 | 40.00 | 4.50 | 55.00 | 0.00 | 0.400 |
| P4 clip 0,20 | A0 | 28.50 | 7.50 | 59.00 | 2.00 | 0.285 |
| P4 clip 0,20 | A1 | 35.50 | 6.00 | 58.50 | 0.00 | 0.355 |

## Mở đường và tiếp cận enemy

| Model | Tường/game | Tới gần % | Bước đầu có điều kiện |
| --- | --- | --- | --- |
| B | 8.90 | 99.75 | 19.76 |
| U4 chưa clip | 8.95 | 99.75 | 19.28 |
| P4 clip 0,20 | 8.95 | 99.75 | 19.61 |

Tới gần dùng Manhattan≤3, đọc tỷ lệ cùng thời gian có điều kiện; chưa phải chứng minh có đường đi an toàn hoặc trap.

## Skill probes

| Nhánh | Episode | Escape owned-kill % | Approach owned-kill % |
| --- | --- | --- | --- |
| U4 chưa clip | 0 | 100.00 | 87.50 |
| U4 chưa clip | 100 | 100.00 | 87.50 |
| U4 chưa clip | 200 | 100.00 | 93.75 |
| P4 clip 0,20 | 0 | 100.00 | 87.50 |
| P4 clip 0,20 | 100 | 100.00 | 93.75 |
| P4 clip 0,20 | 200 | 100.00 | 87.50 |

## Kiểm chứng và giới hạn

Regression 241/241 PASS; sau sửa metadata, 25 focused metadata/compatibility tests PASS. Các nhóm test có trùng nhau, không cộng thành 266 test riêng biệt. Smoke CPU/CUDA qua đủ 20 episode/nhánh. Có test clipping dấu advantage, fixed targets/group weighting, KL sau update, nonfinite gradient, cancel giữa epoch khôi phục Adam và resume PPO ở tiến trình mới trên CPU/CUDA. Fixture500bước/4epochs kiểm tra bộ nhớ/tensor finite; không dùng làm đánh giá chất lượng policy.

Đã kiểm tra ZIP raw 59 files/CRC/bytes, source/B/selected/full-state hashes, replay lại 384 skill games/8794 frames, validation selection và các metric tổng/từng góc. Adam counts bằng tổng epochs thực, scheduler 200, context/history và final/best weights khớp full state; RNG Python/NumPy/env/Torch CPU/CUDA có đủ trạng thái.

Một train seed, một cấu hìnhK/epsilon/LR và RuleBot hiện tại. CI theo board cho các policy đã chọn chưa đo biến thiên train. Chưa triển khai D2/D3, minibatch/rollout nhiều episode hoặc thay model. KL guard không cắt epoch nào ở cả hai nhánh; điều kiện mở D2 vì KL thường chặn còn1–2epochs chưa xảy ra. Không dùng test37000 đã mở để chỉnh LR/epsilon rồi chọn lại candidate.
