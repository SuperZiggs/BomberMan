# T4-A1: pilot hoàn tất, giữ checkpoint gốc

15/10/2026. Đủ hai lượt train 200 episode và 400 trận test cho mỗi model. Normalization chưa đạt tiêu chí cải tiến; giữ B và mặc định `normalize_advantage=False`. Không chọn final lại sau khi xem test.

## Kết quả trên cùng seed bàn

Test 34000–34199, mỗi seed cả A0/A1, 400 trận/model, Phase C/RuleBot/9×9/2 agents/max500. CNN sample categorical qua mask hiện tại; không gọi attack teacher.

| Chỉ số | B gốc | Control best100 | Normalized best100 |
|---|---:|---:|---:|
| ActiveWR | 33,25% (133/400) | 36,5% (146/400) | 33,5% (134/400) |
| WR | 38,75% (155) | 42,5% (170) | 39,75% (159) |
| Hòa | 58% (232) | 54,5% (218) | 57% (228) |
| Tự nổ | 0,25% (1) | 0,25% (1) | 0,25% (1) |
| Kills trung bình | 0,3325 | 0,365 | 0,335 |
| Thắng thụ động | 5,5% (22) | 6% (24) | 6,25% (25) |
| Bom trung bình | 40,495 | 40,3875 | 40,4425 |
| Item trung bình | 1,825 | 1,82 | 1,925 |
| Bước trung bình | 379,3525 | 376,61 | 368,87 |
| ActiveWR A0/A1 | 32% / 34,5% | 35% / 38% | 33% / 34% |
| Hòa A0/A1 | 57% / 59% | 53,5% / 55,5% | 54,5% / 59,5% |
| Tự nổ A0/A1 | 0,5% / 0% | 0,5% / 0% | 0,5% / 0% |

Bootstrap theo cặp **seed bàn**, giữ hai góc cùng nhau: 200 cặp, 10.000 resamples, RNG seed20261014.

| So sánh | Chênh ActiveWR | CI95% | Gate +5 điểm / CI lower>0 |
|---|---:|---:|---|
| Normalized − B | +0,25 điểm | [-5,5; 6] | FAIL / FAIL |
| Normalized − Control | −3 điểm | [-9,25; 3,25] | FAIL / FAIL |

Điều kiện tự nổ quan sát không cao hơn hai đối chứng: PASS. Toàn bộ gate: FAIL, `eligible_for_replication=false`, `decision=retain_original`. Control cũng chỉ hơn B 3,25 điểm, chưa đạt ngưỡng +5 điểm để đề xuất thay B. CI của hai so sánh candidate đều chứa 0; dữ liệu chưa chứng minh normalization tốt hơn, tương đương hay kém hơn ổn định.

## Đã triển khai và kiểm tra

Trainer/CLI thêm `--normalize_advantage`, mặc định tắt, và diagnostics opt-in. Advantage của actor detach và chuẩn hóa trên toàn learner trajectory bằng population std; <2 mẫu hoặc std<1e-8 giữ raw và ghi skip. MC returns/critic targets/reward/teacher local/mask/CNN 12 kênh/loss phụ/Adam/schedule giữ nguyên. Không dùng MOVE teacher mới, GAE hay PPO trong pilot này.

150 tests PASS, gồm tính tay, trajectory một bước/hằng/gần hằng/dài, detach, critic gradient/target isolation, baseline và diagnostics-only bit parity trên CPU với source trước sửa, actual update CPU/CUDA, source/fresh-seed/budget/clobber/partial-run guards. CPU/CUDA smoke20ep/arm/max20 đủ 20 update và 400 learner steps/arm, không NaN/Inf, 46,60/55,36s. Deadline smoke0,001phút trả incomplete/no train/no test. Guard `test_opened.json` được thêm và test sau smoke, không đổi code số học đã smoke.

Hai nhánh warm-start từ B, train seed2026101502, LR1e-4, gamma0,99, một Adam và một update/episode. Diagnostics bật như nhau. Ngoài đường dẫn output, hai config nhánh chỉ khác cờ normalization. Source snapshot 18 file, B và selected weights giữ nguyên qua toàn lượt. Best khóa bằng checksum trước mở test; marker giữ seed đã dùng kể cả test bị gián đoạn.

| Train/diagnostic | Control | Normalized |
|---|---:|---:|
| Episodes / updates / Adam | 200 / 200 / 1 | 200 / 200 / 1 |
| Learner steps | 75.646 | 75.957 |
| Active wins trên train | 72 | 67 |
| Tự nổ trên train | 3 | 3 |
| Update bị clipping | 95 (47,5%) | 79 (39,5%) |
| Norm gradient trước clip: mean / p95 | 34,339 / 129,576 | 32,282 / 131,898 |
| Mean raw advantage std | 1,1964 | 1,1064 |
| Mean policy advantage std | 1,1964 | ≈1 |
| Skip normalization khi bật | N/A (tắt) | 0/200 |
| Mean explained variance theo episode | 0,4520 | 0,4789 |
| Mean entropy theo episode | 0,2829 | 0,2884 |
| Thời gian train gồm validation/skill | 29,51 phút | 29,33 phút |

Clipping giảm nhưng p95 norm không giảm, và ActiveWR không qua gate. Norm này của **tổng gradient policy/value/entropy/auxiliary**, không riêng policy. Hai nhánh có trajectory khác nhau; các diagnostics không chứng minh nguyên nhân hay optimizer là bottleneck. Loss policy raw và normalized khác scale, không so trực tiếp trị số. Normalization còn đổi tương quan với các loss phụ cố định.

| Mốc | Validation control / normalized | Skill approach control / normalized |
|---|---:|---:|
| B / ep0 | 29% / 29% | 28/32 / 28/32 |
| ep100, best cả hai | 35% / 36% | 30/32 / 28/32 |
| ep200 | 30% / 32% | 30/32 / 28/32 |

Validation100 trận seed21000, suicide0 mọi mốc; chọn best bằng thứ tự score và guard hiện có. Skill escape32/32 mọi mốc; chỉ diagnostic, không chọn best. 384 stored games gồm baseline64 được ghi ở cả hai nhánh; chỉ 320 skill rollouts thực. Skill repeats/orientations có phụ thuộc, không xem chúng như 384 mẫu thống kê độc lập.

## Thời gian, provenance và dữ liệu

Runner full: **5.816,06s = 96,93 phút**, bắt đầu20:04:54UTC. Triển khai/test/smoke bắt đầu19:53:44UTC; hạn người dùng22:23:44UTC (05:23:44 giờ Việt Nam). Full xong21:41:50UTC, khoảng108,1phút sau khi bắt đầu công việc; review và báo cáo hoàn tất trước hạn. Ngân sách runner134phút được giảm từ giới hạn150 để tính cả phần triển khai và chừa thời gian review.

Kiểm tra độc lập: ZIP45files/CRC/byte equality; source+B+best/final hashes; 400 training rows/updates,6 validations và6 skill probes; replay384games/**8.864frames** khớp masks/positions/placements/kills/alive/labels/purity; 1.200 game rows/rates/averages/corners và hai bootstrap CI khớp summary. `analysis_results.json` chứa phép kiểm tra và số liệu đầy đủ; script `review_results.py` tái tính chúng. ZIP raw được giữ nguyên; analysis/report được thêm sau ZIP.

B SHA256: `93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60`.
Control best100: `8324ca8d389daa19c27af7879c4ca6d2855c6d381f7ca89a3b4ff021bfdb7cfd`.
Normalized best100: `2657cfa9495ef63b7874e1993591a6577fa9319b4575b9407ca875e6745a6f5d`.

## Bước tiếp theo và giới hạn

Giữ B, không bật normalization làm mặc định và không cần chạy lại A1. 34000–34199 đã dùng; lần sửa tiếp theo cần protocol và seed mới, không thêm trận để cứu gate hoặc chọn final sau test. 31000/wood_branch vẫn không được mở trong lượt này.

Theo tracker, bước độc lập tiếp theo là **T4-B0**: kiểm tra boundary và GAE lambda=1 tương đương MC trên frozen trajectories trước khi mở B1 (actor GAE, critic MC). B0/B1 chưa triển khai hoặc train ở đây; teacher/reward/coefficients vẫn phải khóa riêng. Reward audit, probe gradient từng nhóm, PPO, resume Adam/RNG và guard phản bom là task khác.

Một train seed chỉ đủ pilot. Kết quả không chứng minh phương pháp luôn kém, không chứng minh optimizer gây validation giảm, và cùng một ca tự nổ/400 không chứng minh rủi ro thật bằng nhau. Mask hiện tại vẫn chỉ dự báo phạm vi nguy hiểm đã biết; normalization không sửa giới hạn đối thủ đặt bom tương lai.
