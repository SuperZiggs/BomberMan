# S3-map11 CNN — ma trận chấp nhận độc lập, 25/10/2026

Nguồn duy nhất: `2026-10-25-s3-map11-cnn-evaluation.md`, `AGENT.md` và code/test hiện tại. Không đọc kế hoạch triển khai. Test-author không sửa production. Các tên field theo convention registration/summary của runner hiện tại; raw shard là JSON object có `games` và metadata (lồng `metadata` hoặc top-level).

Lệnh: `venv\Scripts\python.exe -m unittest test_probe_s3_map11_cnn -v`; TEMP/TMP trỏ `.superpowers/sdd/2026-10-25-s3-map11-cnn/tmp`, OMP/MKL=1. Chỉ seed smoke 1900410000; không dùng seed benchmark830000. Tất cả game thật tối đa12bước.

| Yêu cầu / rủi ro | Mức / scenario | Quan sát mong đợi | Test |
|---|---|---|---|
| API, import, help | Contract/CLI | Bốn API; import/help không game/gradient | public_api, import_and_help |
| Full cố định | Config unit | 11x11/13kênh/2agent;6cặp;200bàn/400game mỗi cặp;shard10;500bước;E | full_config |
| Smoke riêng | Config + integration | 2bàn/1bàn-shard;24game;max12;smoke=true | smoke_config, real_smoke |
| Seed type/range | Boundary | Âm/bool/float/string/overflow từ chối trước tạo output | invalid_seed |
| Fresh seed | Guard | Seed đã dùng/đặt chỗ và dải overlap từ chối | used_reserved_seed |
| Output tồn tại | Safety | Không tạo/ghi đè output hiện hữu | existing_output |
| Provenance | Integration | Checkpoint đúng SHA; source bundle + hash +runtime; init không game | registration |
| Config/source/checkpoint đóng băng | Tamper | Mismatch từ chối trước game; dữ liệu cũ nguyên vẹn | changed_config, changed_source, changed_checkpoint |
| Hai CNN thật | Integration24game | Cả hai ghế có mạng13x11x11, CPU,mask,sampling,no_grad; không RuleBot/search/train | real_smoke |
| Marker trước game | Integration | test_opened.json có trước env.step; đúng seed/4/24/hash/smoke | marker |
| Raw/summary coverage | Integration |6cặp×2seed×2góc,12shard;metric đủ/đúng;smoke không fullPASS | raw_and_summary |
| Chống ghi đè | Error | non-resume từ chối shard/marker/summary có trước | nonresume_artifacts |
| Shard integrity | Corrupt cases | metadata/count/duplicate/missingcorner/badmetric từ chối, giữ bytes | malformed_shards |
| Incomplete | Fault injection scheduler | Không đủ shard không summary;lock của mình giải phóng | incomplete |
| Resume | Integration | Chỉ shard thiếu chạy; marker/shard cũ giữ nguyên | resume_missing |
| Complete resume | Integration | Chỉ xác thực, trả summary cũ, không game/ghi lại | completed_resume |
| Summary integrity | Tamper | Summary lệch raw từ chối, không overwrite | corrupted_summary |
| Lock ownership | Integration | ExclusivePID lock; caller khác không thay/xóa;ownlock giải phóng khi lỗi | foreign_lock, own_lock_error |
| Worker bound | Boundary | Chỉ int1–5, bool bị từ chối trước game | workers |
| Paired macro | Synthetic raw through run/resume | Trung bình đều3opponent ở mỗi(seed,corner), bootstrap theo board,10000/20261014; CI không pool independent | macro_bootstrap |
| Gate và metric | Synthetic raw through run/resume | Smoke không đủPASSfull ngay cả delta100pp;luônloss/timeout/draw | smoke_never_qualifies, raw_and_summary, macro_bootstrap |

Các ranh giới full gate (5pp,CI dưới strictly>0,tự nổ≤1pp) được packet khóa nhưng chưa được kiểm trực tiếp bằng fixture full trong suite này. Parent cần thêm supporting unit tests cho chúng; không chạy full game chỉ để kiểm số học. Native2worker và regression cũng thuộc bước parent theo packet. Tên test trong cột cuối là nhãn rút gọn của test method trong `test_probe_s3_map11_cnn.py`.

Không có ngoại lệ hạ mức tích hợp: model loader/env/network thực thi thật; chỉ scheduler thay tuần tự để giới hạn tài nguyên. Native Windows2worker smoke và broad regression do parent thực hiện theo packet. Chưa có full kết quả, chưa train/promotion.
