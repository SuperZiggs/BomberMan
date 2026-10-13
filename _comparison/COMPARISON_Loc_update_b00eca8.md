# So sánh Loc_update với code hiện tại

Ngày kiểm tra: 13/10/2026. Đã clone thật repository vào `D:\fpt\subject\BomberMan\_comparison\BomberMan-Loc_update`, checkout nhánh `Loc_update`.

**Kết luận:** Có những phần đáng lấy riêng, nhất là lưu checkpoint an toàn và thống kê chẩn đoán. Chưa có căn cứ để thay cả engine hoặc pipeline hiện tại bằng nhánh này. Code hiện tại đã có các sửa về kill credit, observation, validation và thoát bom mà Loc_update chưa có.

## 1. Phiên bản và phạm vi

- Nhánh tải về: `Loc_update`, commit `b00eca8a533733f02f698c2056282fc1f0af06bb`.
- Nhánh hiện tại: `codex/visualizer-assets`, HEAD `25b1571a92b3a2eb9b548180083c3e7a99526e25`, **có tính cả thay đổi chưa commit** tại thời điểm kiểm tra.
- Tổ tiên chung: `d76c9aee398335395e0e8764db258f7277198f6f`. Hai nhánh phát triển từ cùng bản nền; Loc_update không bao gồm những sửa chưa commit của bạn.
- So với bản nền, Loc_update thay đổi 13 file, thêm 3.273 dòng và xóa 109 dòng; chủ yếu ở trainer, demo Pygame, sudden death và test.
- SHA256 của **cả 10 checkpoint trong `checkpoints/` đều trùng** với thư mục checkpoint gốc hiện tại. `RESULTS.md`, `IMPROVEMENTS.md`, `README.md` của Loc_update cũng không được cập nhật so với bản nền.
- Các script v12–v19 có cấu hình train mới nhưng thiếu các checkpoint nguồn tương ứng, ví dụ v19 cần `checkpoints_v18/best_phaseC.pt`. Không có trọng số v19 hoặc log thực nghiệm mới trong các file được commit để xác nhận chất lượng sau train.

Code chính, checkpoint và log hiện tại được giữ nguyên. Chỉ tạo bản clone và các artifact so sánh trong `_comparison/`.

## 2. Khác biệt có ý nghĩa

| Phần | Code hiện tại | Loc_update | Ý nghĩa khi tích hợp |
|---|---|---|---|
| Observation | 12 kênh: chỉ số bom, range và nguy hiểm theo thời gian | 9 kênh, kênh 8 là cảnh báo vòng bo | Kênh 8 hai bên có nghĩa khác nhau; không chép thẳng trọng số |
| Dự báo bom | Mô phỏng mỗi tick, cập nhật tường bị phá và lửa | Hazard trên hình học bàn hiện tại | Giữ forecast hiện tại; đã tái hiện trường hợp forecast bên kia bỏ sót lửa |
| Gán kill | Một credit mỗi nạn nhân; giữ nguồn lửa và cờ tự nổ riêng | Gán theo mỗi bom vừa nổ | Giữ sửa kill credit hiện tại |
| Hướng dẫn tấn công | Reachability đến tick nổ; nhãn vị trí, bom chiến đấu và thoát riêng | Thưởng chu kỳ bom an toàn, pressure ở vị trí và ô kề đối thủ | Đây là hai phương pháp khác nhau; pressure của hai bên không cùng thang đo |
| Chọn best | Policy cố định trên validation seed, luân phiên góc, guard tự nổ | Rolling train 200 ván, nhiều gate WR/bom/kills | Giữ validation hiện tại để chọn model |
| Lưu checkpoint | `torch.save` trực tiếp, state dict; metadata ở JSON | File tạm + `os.replace`, retry khóa file Windows, payload metadata | Có thể lấy helper lưu an toàn và thêm hỗ trợ payload có kiểm tra |
| Vòng bo | Chưa có | Thu hẹp bàn từ bước 200, mặc định mỗi 20 bước khi bật | Một thử nghiệm phù hợp với vấn đề hòa; phải đánh giá thành luật chơi riêng |
| GUI | Tkinter, pixel assets, train trực tiếp, xem đấu và chơi | Pygame, 4 chế độ demo, gồm toàn RL và người đấu 3 RL | Có thể thêm chế độ vào GUI hiện có; chưa cần chuyển thư viện |
| Test | 48 test qua | 39 test qua | Test hiện có chưa bao phủ hết các lỗi được tái hiện bên dưới |

## 3. Benchmark chung, tránh so nhầm luật và đối thủ

Mỗi cấu hình chạy **100 trận**, tổng cộng 400 trận. Dùng cùng engine hiện tại, RuleBot hiện tại, bàn **9×9, 2 tác tử, tối đa 500 bước**, CPU một thread. Seed bàn **18000–18049**, mỗi bàn chạy ở cả A0 và A1; seed policy được đặt lại cho mỗi trận. Policy lấy mẫu Categorical như train/evaluate hiện tại. Không bật sudden death.

Ở các hàng sử dụng Loc_update, script dùng một view chỉ đọc của trạng thái engine chung để chạy observation/mask của nhánh. Hai policy nạp cùng checkpoint 8 kênh được kiểm tra cho logits/value trước mask khớp nhau sau mở rộng kênh bằng trọng số 0. Hàng thử mask trên checkpoint 12 kênh giữ nguyên observation và trọng số hiện tại.

| Trọng số / policy | WR | ActiveWR | Tự nổ | Hòa | Kill/trận | Bom/trận |
|---|---:|---:|---:|---:|---:|---:|
| Checkpoint 8 kênh chung + policy hiện tại | 2% | 0% | 0% | 93% | 0.00 | 6.68 |
| Cùng checkpoint 8 kênh + policy Loc_update | 22% | 0% | 8% | 64% | 0.00 | 2.04 |
| `pressure-C-next/final_phaseC.pt` + policy hiện tại | **45%** | **41%** | **0%** | 54% | **0.41** | 40.57 |
| Cùng `pressure-C-next/final_phaseC.pt`, chỉ thay mask Loc_update | 18% | 6% | 56% | 2% | 0.06 | 18.53 |

**Không thay safety mask hiện tại bằng mask Loc_update.** Cùng trọng số đang sử dụng, mask bên kia giảm 35 thắng có hạ gục và tăng 56 trận tự nổ trên mẫu 100 trận. Hòa giảm trong trường hợp này chủ yếu đi cùng việc trận kết thúc bằng cái chết, không phải khả năng tấn công an toàn được cải thiện. Không suy rộng kết quả này thành chất lượng của model được train lại dưới mask/vòng bo khác.

`ActiveWR` là tỷ lệ thắng có ít nhất một kill do bom của learner. WR cao do RuleBot tự nổ không chứng minh learner tấn công tốt. Các hàng dùng checkpoint khác nhau không tách được tác động của train khỏi policy; hai cặp cùng checkpoint mới dùng để đánh giá riêng tác động của mask/policy.

Đây là đánh giá policy và trọng số đã có trên engine chung, **không phải kết quả train v19**, cũng không phải benchmark native của Loc_update với RuleBot cũ hoặc vòng bo bật. 100 trận/cấu hình chưa đủ để đảm bảo ưu thế trên mọi seed. Sau khi dùng bộ seed này để chọn hướng sửa, lần test cuối tiếp theo cần seed mới.

## 4. Những rủi ro đã tái hiện

### 1. Checkpoint có thể bị nạp sai mà không báo lỗi

**Phần liên quan:** [loader hiện tại](D:/fpt/subject/BomberMan/train_actor_critic.py:201), [loader Loc_update](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/train_actor_critic.py:732), [payload](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/train_actor_critic.py:293).

**Vấn đề:** Khi đưa payload `{model_state_dict, episode, metrics, config}` của Loc_update vào loader hiện tại, toàn bộ trọng số bị bỏ qua và model vẫn là ngẫu nhiên. Ngược lại, nạp checkpoint 12 kênh hiện tại vào mạng 9 kênh bên kia giữ nguyên convolution đầu ngẫu nhiên nhưng nạp các lớp khác. Cả hai thử nghiệm đều xác nhận loader không báo lỗi. Kể cả unwrap payload, kênh 8 của Loc_update là vòng bo còn kênh 8 hiện tại là chỗ đặt bom trống.

**Sửa nhỏ nhất khi tích hợp:** Unwrap state dict rõ ràng, kiểm tra đủ key/shape, báo lỗi cho các trường hợp không hỗ trợ; ánh xạ kênh theo ý nghĩa thay vì chỉ theo số kênh. Khi thêm vòng bo vào model hiện tại, thêm kênh thứ 13 và khởi tạo trọng số kênh mới bằng 0 để giữ 12 kênh cũ.

**Nếu bỏ qua:** GUI có thể hiển thị một model được tưởng là đã train, trong khi một phần hoặc toàn bộ mạng vẫn ngẫu nhiên.

### 2. Engine Loc_update tính trùng kill và mất kill trong lửa còn sót

**Phần liên quan:** [resolve damage/reward](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/bomberman_env.py:232), [gán kill trong mỗi bom](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/bomberman_env.py:510).

**Vấn đề:** Hai bom A0 và A2 cùng trúng A1 cho `kills=[(0,1),(2,1)]`, cả hai được tăng kill dù chỉ một nạn nhân chết. Khi A1 bước vào lửa ở tick sau, A1 chết nhưng `kills=[]`. Engine hiện tại cho đúng một credit trong trường hợp đầu và ghi `(0,1)` trong trường hợp sau.

**Sửa nhỏ nhất khi tích hợp:** Giữ logic nguồn lửa và credit hiện tại; chỉ bổ sung telemetry của Loc_update từ những cái chết đã resolve thật.

**Nếu bỏ qua:** Reward, kills và phân loại nguyên nhân chết có thể sai; những gate chọn best dựa vào kills không đáng tin.

### 3. Forecast Loc_update bỏ sót tia nổ qua tường vừa bị phá

**Phần liên quan:** [project hazards](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/train_actor_critic.py:379), [forecast hiện tại](D:/fpt/subject/BomberMan/bomberman_env.py:373).

**Vấn đề:** Bom ở `(3,1)` phá gỗ `(3,3)` tại tick 1. Bom `(3,5)` nổ tại tick 4 và có thể cháy tới `(3,2)` sau khi gỗ đã mất. Forecast hiện tại ghi tick `[1,2,4,5]`; Loc_update chỉ ghi `[1,2]` vì tính tia nổ với tường ban đầu.

**Sửa nhỏ nhất khi tích hợp:** Dùng lại forecast hiện tại, bổ sung vòng bo vào các mảng burns/blocked theo tick nếu cần.

**Nếu bỏ qua:** Safety mask có thể coi một vị trí là an toàn dù nó sẽ cháy sau khi bàn thay đổi.

### 4. Rolling train chưa đủ để xác nhận checkpoint tốt hơn

**Phần liên quan:** [chọn best từ rolling metrics](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/train_actor_critic.py:1319), [validation cố định hiện tại](D:/fpt/subject/BomberMan/train_actor_critic.py:381).

**Vấn đề:** Rolling metrics là kết quả của nhiều phiên bản policy trong lúc train, không phải 200 trận của checkpoint được lưu. Nhánh còn chọn `best` chính theo WR và số bom trước các gate combat riêng. Trên benchmark chung, policy Loc_update với checkpoint cũ có WR 22% nhưng ActiveWR 0%.

**Sửa nhỏ nhất khi tích hợp:** Giữ đánh giá model cố định, bộ validation riêng và guard tự nổ hiện tại; thêm telemetry vào metadata để giải thích quyết định chọn model.

**Nếu bỏ qua:** WR hoặc số bom có thể tăng mà khả năng hạ đối thủ không tăng.

## 5. Những phần nên lấy và thứ tự thực hiện

1. **Lưu checkpoint qua file tạm:** lấy ý tưởng [atomic_torch_save](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/train_actor_critic.py:266), áp dụng tại các chỗ lưu snapshot/best/final hiện tại. Có thể giữ state dict cũ ở bước đầu để tránh đổi định dạng; metadata JSON hiện tại đã đủ dùng. Mục tiêu là giảm nguy cơ checkpoint dở dang khi tiến trình dừng trong lúc ghi và xử lý khóa file Windows.
2. **Thống kê nguyên nhân chết và vòng đời bom:** thêm nguyên nhân chết bởi đối thủ, đóng vòng bo, bom dây chuyền; đếm cơ hội mask bị chặn và những bom đã thoát an toàn nhưng không tạo kill. Giữ `self_blast_deaths` riêng vì tự nổ và bom đối thủ có thể cùng góp phần gây chết. Bắt đầu bằng log, chưa thêm reward mới; các số này giúp giải thích hòa/tự nổ còn lại.
3. **Vòng bo dưới cờ tùy chọn:** lấy cơ chế [thu hẹp bàn](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/bomberman_env.py:331) và [cảnh báo](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/bomberman_env.py:361), tích hợp với forecast/GUI hiện tại và kênh observation bổ sung. Chạy đối chứng bật/tắt trên cùng checkpoint và seed; báo riêng kill bằng bom, chết bởi bo, ActiveWR, tự nổ, hòa. Giảm hòa do bo chưa chứng minh đã học đặt bom ép chết đối thủ trên luật cũ.
4. **Thêm chế độ demo:** 4 RL cùng đấu, hoặc người đấu 3 RL, vào Tkinter hiện có bằng wrapper policy đã có. Đây là cải thiện phần trình diễn đồ án; không cần sao chép toàn bộ `demo_gui.py` hoặc thêm Pygame ngay.

Không có bằng chứng từ clone này để đổi CNN/thuật toán, lấy toàn bộ reward mới hoặc thay tiêu chí best bằng mục tiêu số bom. Nút thắt hiện tại vẫn là lựa chọn vị trí/phối hợp bom để chặn đường sống, như chẩn đoán timeout trong `RESULTS.md` của bạn; vòng bo là một cách thay đổi luật để tạo giao tranh, cần được đánh giá tách biệt.

## 6. Artifact và tái chạy

- [Benchmark theo từng seed/góc, hash code/checkpoint](D:/fpt/subject/BomberMan/_comparison/benchmark_18000.json).
- [Thử riêng việc thay mask trên checkpoint hiện tại](D:/fpt/subject/BomberMan/_comparison/mask_transplant_18000.json).
- [Các trường hợp lỗi và tương thích đã assert](D:/fpt/subject/BomberMan/_comparison/compatibility_checks.json).
- [Script benchmark ba cấu hình](D:/fpt/subject/BomberMan/_comparison/compare_branches.py).
- [Script thử thay mask](D:/fpt/subject/BomberMan/_comparison/benchmark_mask_transplant.py).
- [Script tái hiện rủi ro](D:/fpt/subject/BomberMan/_comparison/check_compatibility.py).

```powershell
cd D:\fpt\subject\BomberMan
& 'D:\anaconda\envs\torch\python.exe' _comparison/compare_branches.py
& 'D:\anaconda\envs\torch\python.exe' _comparison/benchmark_mask_transplant.py
& 'D:\anaconda\envs\torch\python.exe' _comparison/check_compatibility.py
```

Validation đã chạy: 48 test hiện tại, 39 test nhánh clone và các assert của script kiểm tra đều qua. Chưa train lại, chưa chạy GUI Pygame, chưa benchmark vòng bo bật hoặc FFA 3–4 tác tử/11×11. Vì chưa có checkpoint v19/log mới, không kết luận về hiệu quả sau khi train bằng pipeline mới của Loc_update.
