# Chạy đủ two_bomb và hướng nghiên cứu tiếp theo

Ngày đối chiếu nguồn: 14/10/2026. Đây là tìm kiếm có chủ đích, do một người sàng lọc, không phải tổng quan hệ thống hay danh sách đầy đủ mọi nghiên cứu.

Plan triển khai và checklist theo dõi từng thử nghiệm, đặc biệt các nhánh GAE/PPO/reward: [PLAN_BOMBERMAN.md](/D:/fpt/subject/BomberMan/PLAN_BOMBERMAN.md).

Cập nhật kết quả: lượt đủ200 episode và1.200 trận đã hoàn tất. Best two_bomb vẫn ep0 bằng gốc, ActiveWR36,5%; local38% nhưng chênh lệch chưa đạt gate. Không cần chạy lại nguyên lượt chỉ vì hướng dẫn bên dưới còn mô tả cách chạy; bước tiếp theo là audit T2. [Báo cáo T1 đầy đủ](/D:/fpt/subject/BomberMan/runs/two-bomb-complete-20261014-031626/T1_report.md).

## Việc nên làm ngay

Chạy lại **200 episode two_bomb từ checkpoint gốc**, cùng seed train và cấu hình pilot. Dùng lại đối chứng local đã hoàn tất 200 episode; script kiểm tra SHA256 của code, checkpoint và cấu hình trước khi cho phép dùng lại. Không nối thêm 54 episode từ final 146: checkpoint hiện chỉ lưu trọng số, không lưu Adam, scheduler và RNG để tiếp tục đúng lượt cũ.

Chấm lại original, best local và best two_bomb trên **400 trận/model**, seed bàn **26000–26199**, mỗi bàn cả A0/A1. Bộ seed 25000 đã được xem, nên bộ mới này được cố định trước khi chạy. Validation vẫn 100 trận seed 21000 ở episode 0/100/200; cách chọn best và guard tự nổ vẫn giữ nguyên. Final 200 được lưu, nhưng phép so sánh chính dùng best chọn bằng validation.

Script mới không giới hạn thời gian. Nó tái sử dụng trainer, mask và bootstrap hiện có; không đổi teacher, reward, observation hay optimizer. Nó không tự thay checkpoint đang dùng. Nếu có tín hiệu đạt tiêu chí, kết luận chỉ là nên lặp lại trên seed train khác.

Chạy trong PowerShell:

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
& $py -m unittest test_two_bomb_complete test_tactical_teacher test_combat test_visualize -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& $py -u .\run_two_bomb_complete.py
if ($LASTEXITCODE -ne 0) { throw 'Run incomplete; preserve the printed output directory and error' }
```

Mỗi lần chạy tạo một thư mục mới `runs/two-bomb-complete-YYYYMMDD-HHMMSS`. Nếu cần chạy lại sau khi bị ngắt, bắt đầu một lượt mới từ gốc; không coi warm-start từ final giữa chừng là tiếp tục chính xác.

Gửi lại thư mục kết quả hoặc các file:

- `summary.json`, `config.json`.
- `two_bomb/metrics.jsonl`, `two_bomb/validation_metrics.jsonl`, `two_bomb/best_validation.json`.
- `evaluation_original_26000.json`, `evaluation_local_26000.json`, `evaluation_two_bomb_26000.json`.

Các file đánh giá chứa từng seed/corner để tính bootstrap theo cặp, ngoài số tổng. Khi cần xem replay, giữ cả best/final trong thư mục đó. Không chỉ gửi training reward: reward tăng chưa chứng minh hạ đối thủ tốt hơn.

## Những tài liệu gần đây thực sự liên quan

| Tài liệu và năm | Phương pháp đã đọc | Liên hệ với dự án và giới hạn |
|---|---|---|
| Huynh, Cao, Wu: [Multi-Agent Training for Pommerman](https://arxiv.org/html/2407.00662v2), bản đầu 2024, sửa 08/01/2025 | PPO, curriculum đối thủ, giảm thưởng khám phá theo hiệu suất, population self-play và Elo | Ưu tiên tham khảo. Luật 2v2/11×11 khác Phase C. Bài dùng tới 200 triệu timestep self-play; kết quả không chuyển trực tiếp sang pilot 200 episode. [Lịch sử bản](https://arxiv.org/abs/2407.00662). |
| Sun và cộng sự: [Decision-making with Speculative Opponent Models / DOMAC](https://arxiv.org/html/2211.11940v3), sửa 2024; preprint đầu 2022 | Dự đoán hành động đối thủ từ thông tin của learner, kết hợp actor và distributional critic; có Pommerman FFA/team | Gợi ý kiểm tra phản ứng của đối thủ khi đặt bom. Chuyển đầy đủ DOMAC là thay đổi lớn. Không gọi đây là ý tưởng mới xuất hiện năm 2024 chỉ vì bản sửa mới. |
| Guo, Xiao: [Multi-Agent RL for Bomberland Game](https://guochenmeinian.github.io/bomberland_rl_report/), dự án môn học NYU 2025 | CNN/entity attention, thử temporal attention, PPO+GAE và frozen-snapshot self-play | Nguồn triển khai, chưa phải kết quả paper phản biện. Bản temporal ~4.000 episode vẫn tự nổ; bản đơn giản ~2.000 episode đang train. Không có đánh giá đối chứng đủ mạnh để khuyến nghị đổi architecture. |
| Helfenstein và cộng sự: [M2CTS / Checkmating One, by Using Many](https://www.dfki.de/en/web/research/projects-and-publications/publication/17385), IEEE Transactions on Games 06/2026 | Chọn neural expert theo giai đoạn rồi dùng MCTS; thí nghiệm Pommerman FFA đấu ba simple agents | Có thí nghiệm bom thật dù tiêu đề nói chess. Bản đọc là [arXiv v3, 17/06/2025](https://arxiv.org/html/2401.16852v3); bản đầu 2024. Search được dùng lúc inference, khác CNN chỉ chơi qua mask của ta. |

Nguồn cũ như PPO, GAE và Action Guidance là nguồn gốc thuật toán. Việc viện dẫn nguồn gốc không có nghĩa không tồn tại nghiên cứu mới; kế hoạch trước có bài curriculum 2024 nhưng phần đối chiếu phương pháp mới chưa đủ rộng. Cần ghi đúng năm bản đầu, bản sửa và xuất bản để tránh biến một phương pháp cũ thành “mới 2026”.

## Thứ tự thử sau khi khép lại pilot

Các đề xuất dưới đây là suy luận cho code này, không phải tuyên bố đã được các paper chứng minh trong môi trường này.

1. **Kiểm tra chất lượng nhãn và việc thực hiện combo.** Đếm CNN có chọn hành động được teacher gợi ý không, đi được tới bom thứ hai không, và có thật sự tạo thế hết đường sống không. Preflight trước chỉ có 1 nhãn bẫy hoàn toàn trong 147 trạng thái được lấy mẫu, không có bẫy hoàn toàn bằng hai bom. Đây không đại diện cho toàn bộ train, nhưng đáng kiểm tra trước khi thêm hàng nghìn episode. Đặc biệt cần đo việc đổi kế hoạch mỗi bước và tính khả thi của hành động tiếp theo qua mask.
2. **Curriculum tập trung giao tranh.** Thử riêng bản đồ nhỏ có tình huống đặt bom/thoát thân rõ ràng; đối thủ từ đứng yên đến di chuyển biết né, rồi biết đặt bom. Sau đó quay lại map ngẫu nhiên và RuleBot hiện tại. Giữ tập đánh giá riêng để biết agent học kỹ năng hay chỉ nhớ bản đồ. Đây là cách áp dụng tinh thần curriculum, không sao chép nguyên hệ thống của bài.
3. **Pool đối thủ.** Tận dụng Phase D/snapshot pool đã có, đưa nhiều phong cách đối thủ vào thay vì chỉ một RuleBot. Trước hết đánh giá chéo với RuleBot và snapshot cũ; chỉ thêm Elo matchmaking nếu chọn đối thủ đang là nút thắt thật.
4. **Tách reward phụ khỏi mục tiêu giao tranh.** Chỉ giảm thưởng phá tường/nhặt đồ khi log cho thấy reward phụ chi phối. Đo kills và ActiveWR; tránh lấy thắng do đối thủ tự nổ làm bằng chứng chiến thuật tốt.
5. **Thử GAE/chuẩn hóa advantage hoặc PPO như thí nghiệm riêng.** Giữ đối chứng và teacher cố định. PPO vẫn được nghiên cứu mới sử dụng; tuổi của paper gốc không làm thuật toán mất giá trị. Không đồng thời thay teacher, reward, curriculum và optimizer rồi quy công cho một yếu tố.

Opponent modelling và phase-aware experts là hướng sau: bản hiện tại cần chứng minh việc học chuỗi hành động trước. Với chỉ hai agents và observation đủ thông tin về bom hiện tại, chưa có bằng chứng bắt buộc phải thêm Transformer/LSTM/MoE.

## Phạm vi tìm kiếm và nguồn chưa đánh giá đủ

Các truy vấn web gồm `Pommerman Bomberman reinforcement learning 2024 2025 2026 paper`, `"Pommerman" "2025" site:arxiv.org`, `"Pommerman" "2026" site:arxiv.org`, `"Pommerman" "2025" site:openreview.net`, `"Checkmating One, by Using Many" arxiv`, và `"Decision-Making With Speculative Opponent Models" Pommerman`. Các bản đầy đủ ưu tiên từ arXiv, tác giả, trường/viện và publisher.

OpenAlex: `Bomberman OR Pommerman`, title/abstract/keywords, 2024–2026, trả 14 records; nhiều records là dataset, luận văn hoặc ứng dụng giáo dục. Đây không phải 14 nghiên cứu RL. [Truy vấn có thể tái lập](https://api.openalex.org/?oql=get%20works%20where%20it's%20not%20retracted%20and%20published%20from%202024%20through%202026%20and%20title-abstract-keywords%20has%20(Bomberman%20OR%20Pommerman)). Semantic Scholar truy vấn Pommerman/2024–2026 bị rate-limit, nên không dùng để tuyên bố bao phủ đầy đủ.

Tìm thấy thêm [Transformer-Based Opponent Modeling in Pommerman, luận văn KTH 2025](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1987589), nhưng chưa lấy được full text qua công cụ. Bản thảo [DSOMIX dự ICLR 2025](https://openreview.net/forum?id=yZdPpKTO9R) cũng bị chặn truy cập đầy đủ; chưa xác nhận quyết định chấp nhận, không trình bày nó như paper đã được ICLR chấp nhận. Không lấy ngày crawl, năm luận văn hay năm tải lên lại làm bằng chứng một thuật toán vừa được phát minh.

Quy trình tìm và ghi giới hạn sử dụng skill literature-review từ Scientific Agent Skills. Nguồn công cụ, không phải nghiên cứu Bomberman: Kassis, Agarwal, He, Patel, Brueckner (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065).
