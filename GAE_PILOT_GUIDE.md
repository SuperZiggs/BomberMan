# T4-B1: MC so với actor GAE(0,95)

**Đã hoàn tất/review:** `runs/gae-full-20261015-125942`, đủ200episode/nhánh và400test/model,54,87phút. GateFAIL,giữB:best cảhai làepisode0,ActiveWR cảba31,75%,delta0/CI[0;0].35000–35199 đã mở; không chạy lại full các lệnh dưới đây trên cùng dải seed. [Báo cáo](/D:/fpt/subject/BomberMan/runs/gae-full-20261015-125942/decision_report.md).

Runner `run_gae_pilot.py` dùng GAE đã kiểm tra ở B0. Cả hai nhánh warm-start từ `runs/pressure-C-next/final_phaseC.pt`, một Adam mới cho mỗi nhánh; không tiếp tục optimizer của checkpoint cũ.

| Tham số | Đăng ký trước |
|---|---|
| Nhánh | `control`: actor MC; `gae`: actor GAE, lambda 0,95 |
| Giữ cố định | Critic MC returns; normalization tắt; reward/mask/teacher local/12 kênh không đổi |
| Train | Seed 2026101504, 200 episode/nhánh, Phase C, 9×9, 2 agents, max500, LR 1e-4, gamma0,99 |
| Chọn best | Validation100 trận seed21000 ở episode0/100/200, guard tự nổ hiện có |
| Kiểm tra kỹ năng | 32 escape + 32 approach mỗi lần validation; lưu replay và model digest |
| Test cuối | B, best control, best GAE; 400 trận/model; seeds35000–35199, cả A0/A1, CPU một thread |
| Inference | CNN lấy mẫu qua mask hiện có, không gọi teacher/planner |
| Ngân sách runner | Tổng tối đa 150 phút; trần preflight 15, hai lượt train 100, evaluation 50 phút. Mỗi giai đoạn còn bị chặn bởi thời gian tổng còn lại; thiếu phần nào là incomplete |

Timeout của trò chơi là terminal với giá trị kế tiếp bằng0. GAE chỉ thay advantage của actor; critic và explained variance vẫn dùng MC. Không cắt rollout giữa trận. GAE có thể giảm phương sai và thêm bias; chưa mặc định tốt hơn MC.

Runner khóa source/config/hash B trước train, chọn và khóa hash best trước khi mở test. Từ chối ghi đè output và từ chối dải test đã mở, kể cả lượt evaluation bị ngắt. Không sửa source trong lúc chạy. Smoke dùng seed39850, 20episode/nhánh/max20, validation2 trận và test2/model, luôn `full_protocol=false`.

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
& $py -m unittest test_gae test_gae_pilot test_advantage_pilot test_combat_move_pilot -q
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py -u .\run_gae_pilot.py --smoke --device cpu --output-dir "runs\gae-smoke-cpu-$stamp"
& $py -u .\run_gae_pilot.py --smoke --device cuda --output-dir "runs\gae-smoke-cuda-$stamp"
& $py -u .\run_gae_pilot.py --device cuda --budget-minutes 150 --output-dir "runs\gae-full-$stamp"
```

Chỉ chạy lệnh full khi seed35000–35199 chưa dùng. Không chạy lại nguyên lệnh full sau khi đã mở test; muốn lặp phải đăng ký train seed/test seed mới bằng một thử nghiệm riêng. Xem `runs/gae-last-path.txt` để lấy output sau khi mở lại PowerShell; không tạo timestamp mới để tìm kết quả cũ.

Lượt `runs/gae-full-20261015-112853` ngày 15/10 **chưa hoàn tất**: trần train ban đầu 85 phút khiến control đủ 200 episode nhưng GAE dừng ở 184. Validation GAE mới có episode 0/100, cả hai ActiveWR 29%; best cả hai nhánh vẫn episode 0. Chưa mở test35000, chưa có CI hoặc kết luận về hiệu quả GAE. Giữ B.

Sau lượt này runner đã tăng trần train lên100 phút, giữ trần tổng150 phút. Test deadline kiểm tra evaluation chỉ nhận thời gian còn lại. Full bản sửa đã hoàn tất tại `runs/gae-full-20261015-125942`:train lại cảhai từB,không đổi hyperparameter. Không lấy `gae/final_phaseC.pt` của lượt thiếu để chạy thêm16episode:checkpoint chỉ chứa weights,không cóAdam/LR/RNG để tiếp tục đúng protocol.

Kết quả gồm `config.json`, source snapshot, `preflight.json`, baseline skill, config/metrics/validation/skill replay/best/final từng nhánh, `selected_models.json`, `test_opened.json`, ba `evaluation_*.json`, `summary.json` và ZIP `*-send.zip`. Console log đặt cạnh thư mục chạy. ZIP được tạo khi runner kết thúc; báo cáo phân tích thêm sau đó được lưu riêng.

Gate yêu cầu GAE tăng ActiveWR ít nhất5 điểm phần trăm so với **cả B lẫn control**, cận dưới CI95% bootstrap theo cặp seed bàn lớn hơn0, tự nổ quan sát không cao hơn hai đối chứng. PASS chỉ cho phép lặp nhiều train seed trước khi thay B; FAIL/incomplete giữ B. Báo thêm hòa, kills, passive wins và từng góc.

Tham khảo định nghĩa estimator: [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). Thiết kế đối chứng và ngưỡng5 điểm là quy ước của dự án.
