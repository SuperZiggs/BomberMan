# L2.0 candidate audit

Status: incomplete; full_protocol=False.

| Model | States checked | Old representative pass | Any candidate pass | Any two-bomb pass | Chosen best two-bomb | Genuine combo trap |
|---|---:|---:|---:|---:|---:|---:|
| original | 2 | 0/2 | 2/2 | 0/2 | 0/2 | 0/2 |
| two_bomb_final | 0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |

## Dùng để quyết định

- Nếu alternatives khả thi/hữu ích đủ đa dạng: cân nhắc L2.1 chọn trong các candidate đã qua guard; xem coverage và chi phí top-K trước.
- Nếu có nhiều candidate nhưng chọn best vẫn là một bom, chưa thể gọi guard là cải tiến phối hợp hai bom.
- Nếu combo qua mask/trap vẫn rất hiếm: ưu tiên T3.1 benchmark tình huống trước train trên map ngẫu nhiên dài.
- K=1/4/8 là số schedules, mỗi schedule giữ tất cả scored paths. Best-score recall chưa bảo đảm first-actions exact; xem cả hai.
- Status incomplete hoặc corpus thiếu thì giữ kết quả để chẩn đoán, chưa kết luận toàn bộ corpus.

## Phạm vi

- Saved fixed-policy T2 corpus; diagnostic only, not new training/test for model promotion.
- Only scored positive-pressure candidates within original <=64 schedules, not exhaustive game planning.
- Static opponent; robust mask still models counter-bombs. Passing it does not guarantee future opponent reactions.
- Viable requires prefix mask/physics pass and actual forecast suffix survival; post-terminal physics is not a game win.
- Top-K measured through all paths in K schedules; timing includes clone/suffix overhead, not a production guard benchmark.
- Frame/branch samples are correlated within board seeds; counts here are descriptive, not independent trials.
