# T4-B1: lượt đầu chưa đủ protocol — giữ B

Ngày 15/10/2026. Runner hoàn tất việc dừng/lưu dữ liệu đúng deadline, nhưng phép thử MC/GAE **chưa hoàn tất**: control đủ200 episode, GAE184/200. Chưa chạy400 trận/model, chưa có paired CI, không kết luận GAE tốt hơn hay kém hơn checkpoint cũ.

## Kết quả đã có

| Nội dung | MC control | Actor GAE(0,95) |
|---|---:|---:|
| Episode / Adam updates | 200 / 200 | 184 / 184 |
| Learner steps | 73.101 | 71.429 |
| Validation ActiveWR ep0 | 29% | 29% |
| Validation ActiveWR ep100 | 28% | 29% |
| Validation ActiveWR ep200 | 25% | Chưa chạy |
| Tự nổ validation đã chấm | 0% | 0% |
| Best được chọn | Episode0 | Episode0 |
| Approach ep0 /100 /200 | 28/32;30/32;28/32 | 28/32;28/32;chưa chạy |
| Clip gradient | 81/200 | 66/184 |
| Tự nổ train | 3 | 0 |
| Thời gian nhánh | 41,78 phút | 43,22 phút |

Các chỉ số train không phải đánh giá thắng trên seed mới. Mẫu184 và200 khác nhau; không dùng tỷ lệ tự nổ/clip trên đây làm bằng chứng GAE an toàn hơn. Escape đạt32/32 tại cả5 probe đã hoàn tất. Best episode0 có cùng weights B; hash file có thể khác B do định dạng lưu Torch.

Cả hai từ B SHA256 `93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60`, seed2026101504, PhaseC/9×9/2/max500/LR1e-4/gamma0,99. Teacher local, normalization tắt, critic MC, reward/mask/observation giữ nguyên. Một Adam mới mỗi nhánh. Validation100 trận seed21000 ở0/100/200 dùng score và guard tự nổ cũ; GAE thiếu mốc200.

## Nguyên nhân dừng và bản sửa sau lượt chạy

Mình phân bổ trần85 phút cho cả hai nhánh train, quá chặt trên máy này. Timer dừng GAE sau184 episode, bỏ partial trajectory và không cập nhật Adam cho phần chưa hoàn tất. Tổng runner85,11 phút; budget tổng150 phút chưa dùng hết. Đây là thiếu thời gian thực thi, không phải non-finite loss, lỗi CUDA hay kết luận GAE không hiệu quả.

Sau khi runner kết thúc và review dữ liệu gốc, đã đổi riêng trần train từ85 lên100 phút. Tổng runner vẫn bị chặn150 phút; evaluation chỉ nhận thời gian còn lại, tối đa50 phút. Unit test mô phỏng preflight12 phút + train100 phút xác nhận evaluation chỉ được38 phút. Không thay estimator, critic hay game. **Chưa chạy lại full bằng bản sửa.** Snapshot trong ZIP giữ bản85 phút đã thực sự chạy; không sửa config/summary/raw cũ thành100 phút.

Muốn hoàn tất phép thử cần output mới, train lại đủ hai nhánh từ B rồi test400/model nếu mọi guard qua. Không chạy thêm16 episode từ `gae/final_phaseC.pt`: chỉ có weights, thiếu Adam/LR/RNG để tiếp tục đúng protocol. Không khởi động một full mới trong phần budget còn lại vì không đủ thời gian cho cả train và evaluation đã đăng ký.

## Kiểm chứng và file

- Bộ regression trước lượt chạy:174 tests PASS; smoke CPU57,49s/CUDA73,66s,20episode/nhánh/max20, mỗi ZIP46files, full_protocol=false.
- Sau sửa trần thời gian:49 tests liên quan PASS, gồm test deadline mới. Đây không phải chứng minh full mới sẽ hoàn tất đúng thời gian.
- Review ZIP raw41files:CRC/bytes/source19/B/best/final hashes khớp. Source và B không đổi trong lượt chạy; hai file runner/test được sửa budget sau đó, snapshot vẫn giữ nguyên.
-384 completed training rows,5 validations,5 skill probes;320stored skill games/7.140frames được replay lại, summary nhóm khớp. Baseline64game được ghi trong cả hai nhánh, nên có256rollouts kỹ năng thực. Loss hữu hạn, critic MSE khớp MC residual, boundary GAE/normalization đúng.
- Không có `test_opened.json`, không có `evaluation_*.json`, không chọn model dựa trên test.35000–35199 còn chưa mở;31000/wood_branch không dùng trong lượt này.

Kết quả thô: [summary.json](summary.json), [config.json](config.json), [analysis_results.json](analysis_results.json), console log nằm cạnh thư mục; ZIP `../gae-full-20261015-112853-send.zip`. Script review lưu cùng báo cáo. Protocol/lệnh mới: `../../GAE_PILOT_GUIDE.md`.

**Quyết định:** giữ checkpoint B, mặc định MC. T4-B1 INCONCLUSIVE/WAIT rerun; chưa mở B2, PPO, reward hoặc normalization ghép. Một lượt train seed chưa đủ kết luận phương pháp ổn định ngay cả khi lần chạy đầy đủ sau này qua gate.

Cập nhật sau lượt thiếu: đã train lại cả hai từ B và hoàn tất full tại ../gae-full-20261015-125942/decision_report.md. Gate cải thiện FAIL, giữ B; dữ liệu thô lượt thiếu này không bị sửa.
