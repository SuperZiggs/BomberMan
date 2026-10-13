# So sánh cập nhật Loc_update với dự án hiện tại

Ngày kiểm tra: 13/10/2026. Bản clone ở D:\fpt\subject\BomberMan\_comparison\BomberMan-Loc_update, commit afbf6f9.

**Kết luận:** Các checkpoint mới đã học được hành vi tấn công, nhưng chưa cải thiện model đang dùng khi đưa vào điều kiện của dự án hiện tại. Giữ runs/pressure-C-next/final_phaseC.pt và safety mask hiện tại. Phần đáng lấy là lưu checkpoint an toàn và chẩn đoán theo kết quả của bom; vòng bo và các model mới phù hợp làm thử nghiệm riêng.

Đã kiểm tra 129 checkpoint và chạy **1.300 trận mới**: 1.000 trận trên engine/RuleBot hiện tại, 300 trận trên engine/RuleBot của Loc_update. Không sửa code chính, không ghi đè checkpoint và không train lại.

## 1. Điều gì đã thay đổi sau pull

Commit afbf6f9 thêm 119 checkpoint cùng nội dung mới trong RESULTS.md và IMPROVEMENTS.md. Các file Python và README.md không thay đổi so với b00eca8. Vì vậy, kết luận cũ “chưa có checkpoint v19 để kiểm chứng” đã hết hiệu lực; các phát hiện về engine/loader vẫn áp dụng cho cùng phiên bản code.

Đã nạp và kiểm tra toàn bộ 129 file: 10 checkpoint nền có 8 kênh; 119 checkpoint mới có 9 kênh; mọi tensor trọng số đều hữu hạn. Hai ứng viên chính được chọn trước benchmark theo khuyến nghị của chính nhánh:

| Checkpoint | Episode được lưu | WR rolling trong metadata | Kills rolling | Tự nổ rolling |
|---|---:|---:|---:|---:|
| v18 best | 701 | 86,5% | 0,060/trận | 5,5% |
| v19 final | 2.000 | 69,5% | 0,155/trận | 24% |
| v19 best, chỉ kiểm tra metadata | 200 | 83,5% | 0,065/trận | 11% |
| v19 combat, chỉ kiểm tra metadata | 1.747 | 78,5% | 0,175/trận | 17% |

Các số này khớp tài liệu mới và payload. Chúng là rolling train, không phải kết quả đánh giá model cố định. Không dùng WR 86,5% hoặc 83,5% để kết luận model đó thắng trên test mới.

## 2. Kết quả bạn của bạn đã báo cáo

Theo [RESULTS.md, phần v19](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/RESULTS.md:683), bộ đánh giá 400 trận dùng bàn 9×9, hai tác tử, vòng bo, giới hạn 300 bước và RuleBot của nhánh:

| Checkpoint, đều dùng mask v19 | WR | Kills/trận | Tự nổ |
|---|---:|---:|---:|
| v18 best | 77,25% | 0,0775 | 43/400 = 10,75% |
| v19 final | 71,25% | 0,1275 | 84/400 = 21% |

Tài liệu cho thấy v19 đổi thêm tấn công lấy thêm rủi ro. Gate nghiệm thu của v19 không đặt giới hạn riêng cho tự nổ, nên “PASS mọi gate” không đồng nghĩa an toàn theo mục tiêu hiện tại của bạn.

[Phần audit v20](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/RESULTS.md:717) xác nhận RNG của môi trường từng để seed=None. np.random.seed không điều khiển np.random.default_rng(None), nên các bảng v17–v20 là các mẫu độc lập, không phải A/B trên chính xác cùng bàn. Các số cũ vẫn mô tả lượt chạy đó; không thể coi chênh lệch giữa các bảng là tác động đã được tách riêng hoàn toàn.

Prototype v20 cấm quay lại tia bom của mình đã làm WR/kills giảm và tự nổ tăng, rồi được rollback. Đây là bằng chứng thực nghiệm để tránh thêm một luật cấm rộng như vậy vào dự án hiện tại.

## 3. So sánh trên cùng điều kiện của dự án bạn

Mỗi hàng **200 trận**, cùng 100 bàn seed 19000–19099, mỗi bàn learner ở A0/A1, CPU một thread, 9×9, hai tác tử, tối đa 500 bước, không vòng bo. Dùng cùng engine và RuleBot hiện tại; seed policy được đặt lại từng trận.

Checkpoint hiện tại là runs/pressure-C-next/final_phaseC.pt, SHA256 93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60. Checkpoint clone được nạp đầy đủ bằng strict=True vào CNN 9 kênh. Tám kênh chung được đối chiếu từng bước; kênh cảnh báo bo bằng 0 khi bo tắt. Không nạp qua loader hiện tại vốn chưa hỗ trợ payload của nhánh.

| Model / mask | WR | ActiveWR | Tự nổ | Hòa | Kills/trận | Bom/trận |
|---|---:|---:|---:|---:|---:|---:|
| Model hiện tại / mask hiện tại | 40% | **34,5%** | **0%** | 59% | **0,345** | 40,67 |
| v18 best / mask Loc_update | 36% | 2% | 4,5% | 55,5% | 0,020 | 4,135 |
| v19 final / mask Loc_update | 48% | 6% | 29% | 13% | 0,060 | 12,23 |
| v18 best / mask hiện tại | 6% | 1,5% | 0% | 91,5% | 0,015 | 8,885 |
| v19 final / mask hiện tại | 12% | 5% | 0,5% | 86,5% | 0,050 | 12,46 |

ActiveWR là tỷ lệ thắng có ít nhất một đối thủ bị hạ bởi bom của learner. Trong hàng v19/mask Loc_update, WR 48% gồm 6% thắng có hạ gục và **42% thắng thụ động**. Hòa ít hơn đi cùng nhiều tự nổ hơn; chưa đáp ứng mục tiêu tấn công an toàn.

Trên từng cặp bàn/góc, v19 dùng mask hiện tại thêm năm thắng có hạ gục nhưng mất 64 so với model hiện tại, giảm ròng 59/200. So riêng cùng v19: đổi sang mask hiện tại giữ ActiveWR gần nhau, 6% xuống 5%, nhưng giảm tự nổ từ 58 xuống một trận. Vì vậy mask hiện tại bảo vệ tốt hơn trên mẫu này; trọng số mới vẫn không thay thế được trọng số đang dùng.

Đây là **đánh giá khả năng chuyển vào game hiện tại**. Loc_update train với vòng bo, observation khác và item không có trần giống hiện tại. Kết quả chuyển kém không chứng minh model không có ích trong luật chơi gốc của nó. Tuy nhiên, nó là bằng chứng trực tiếp để chưa thay model/engine của bạn.

## 4. Kiểm tra riêng engine và RuleBot của Loc_update

Mỗi hàng **100 trận**, cùng 50 bàn seed 19000–19049 và hai góc, giới hạn 300 bước, mask v19. Chỉ quan sát nguồn nổ để tính một credit cho mỗi nạn nhân và cả chết bởi lửa còn sót; giữ nguyên chuyển động, reward, observation và hành vi của engine.

Observer có assert cho kill trực tiếp, lửa còn sót, bom chồng nhau, tự nổ cùng bom đối thủ; kiểm tra 45 bước xác nhận quan sát không thay đổi trạng thái/reward/observation của engine.

| Model / luật | WR | ActiveWR đã audit | Thắng thụ động | Tự nổ | Hòa |
|---|---:|---:|---:|---:|---:|
| v18 best, bo bật | 93% | 8% | 85% | 7% | 2% |
| v19 final, bo bật | 78% | 16% | 62% | 18% | 3% |
| v19 final, bo tắt | 77% | 16% | 61% | 19% | 3% |

V19 có hạ gục thật và tấn công nhiều hơn v18 trong thử nghiệm này. Tuy nhiên, 62/78 trận thắng của v19 khi bo bật vẫn là thắng thụ động. Với v19, engine gốc ghi 14 kill trong khi audit ghi 16: hai kill do lửa còn sót bị mất credit trong số liệu gốc.

Chỉ hai trận v19/bo bật đi tới bước 200. Vì vậy mẫu này chưa đủ để đánh giá sâu cơ chế thu hẹp bàn, và không có bằng chứng vòng bo là nguyên nhân chính của WR cao. RuleBot và engine khác nhau giữa phần này và phần 3; không lấy WR 78% so trực tiếp với 40% để xếp hạng chất lượng tấn công.

Các phép đánh giá này dùng seed thật, có thể tái chạy. Chúng không tái hiện nguyên bộ 400 bàn cũ vì nhánh không lưu seed bàn/từng trận của các benchmark đó.

## 5. Có thể giúp gì cho code hiện tại

### Nên lấy trước: lưu checkpoint an toàn

Lấy ý tưởng [atomic_torch_save](D:/fpt/subject/BomberMan/_comparison/BomberMan-Loc_update/train_actor_critic.py:266): ghi file tạm rồi os.replace, retry lỗi khóa file Windows. Áp dụng tại snapshot/best/final của trainer hiện tại. Có thể giữ state dict và JSON metadata như đang dùng để tránh đổi định dạng ngay.

Đây là cải thiện độ tin cậy khi ghi file; không được coi là cải thiện ActiveWR.

### Nên bổ sung: chẩn đoán theo kết quả của bom

Ý tưởng hữu ích ở v16 là tách “đã thoát bom” khỏi “bom sau đó tạo tác động và người đặt vẫn sống”. Bổ sung log bom không tạo kill, chết sau khi đã thoát, chết bởi đối thủ và những bước mask can thiệp; dùng nguồn lửa/kill credit đúng của engine hiện tại.

Model hiện tại vẫn hòa 59% và đặt trung bình 40,67 bom/trận. Các log này giúp xác định bom chưa chặn đủ lối thoát, agent đi sai sau khi đặt, hay tín hiệu phá tường đang lấn át mục tiêu giao tranh. Chưa tự động thêm reward cycle/pressure: Loc_update từng có v11 farm reward và v20 tăng tự nổ khi thêm constraint.

### Phần cải thiện hành vi cần ưu tiên: chọn vị trí và bom nối tiếp

Dùng lại [forecast theo tick](D:/fpt/subject/BomberMan/bomberman_env.py:373), [pressure theo số đường sống](D:/fpt/subject/BomberMan/rule_based_bot.py:65) và [teacher hiện tại](D:/fpt/subject/BomberMan/train_actor_critic.py:105) để thử kế hoạch chặn lối thoát bằng bom nối tiếp. Hiện teacher xét vị trí hiện tại/một bước; đây vẫn là giới hạn được ghi trong chẩn đoán timeout của dự án.

Nhánh mới chưa cung cấp một planner mạnh hơn phần này. Điểm học được là đánh giá hành vi sau cả vòng đời bom, giữ mask sống sót và đo ActiveWR trên model cố định. Không chỉ tăng số episode, thưởng đặt bom hoặc giảm thưởng tường theo cấu hình của nhánh.

### Có thể thử riêng: vòng bo và đối thủ bổ sung

Vòng bo có thể làm chế độ demo/curriculum tùy chọn, nhưng thay đổi luật. Nếu tích hợp, giữ 12 kênh hiện tại và thêm kênh cảnh báo thứ 13, cập nhật forecast theo thứ tự bo đóng trước chuyển động; đánh giá riêng kill bằng bom, chết bởi bo, hòa và tự nổ. Kết quả hiện tại chưa xác nhận bo cải thiện model của bạn.

V18/v19 có thể là ứng viên đối thủ bổ sung để tạo kiểu chơi khác trong opponent pool. Cần adapter đúng observation và đánh giá xem learner có cải thiện trên RuleBot/test riêng hay không. Chưa có bằng chứng dùng chúng làm đối thủ sẽ tăng chất lượng; không thay model chính bằng các checkpoint này.

## 6. Những phần nên giữ từ code hiện tại

- Giữ kill credit theo từng nguồn lửa, ghi tự nổ riêng và một credit mỗi nạn nhân. Engine clone vẫn đếm trùng kill khi bom chồng nhau và mất credit ở lửa còn sót.
- Giữ forecast cập nhật tường bị phá. Forecast clone vẫn có trường hợp bỏ sót tia bom muộn đi qua gỗ đã mất.
- Giữ best theo validation cố định, ActiveWR và guard tự nổ. Rolling train của nhánh không thay thế được đánh giá checkpoint độc lập.
- Giữ giới hạn item và observation 12 kênh. Kênh 8 bên clone là bo, còn hiện tại là chỗ đặt bom trống; không chép trọng số theo số thứ tự mà bỏ qua ý nghĩa.
- Giữ seed môi trường rõ ràng trong train/evaluate hiện tại. Dự án đã truyền seed vào BombermanFFAEnv; không mắc lỗi RNG đã ghi ở audit của nhánh.
- Giữ loại bỏ toàn bộ terminal bonus của thắng không có kill. Nhánh phạt tối thiểu 8 trong khi thưởng thắng của engine là 10; mục tiêu chống thắng thụ động hiện tại đã chặt hơn.
- Không dùng trực tiếp checkpoint v19 với loader hiện tại. Đã kiểm tra file thật: loader bỏ qua payload và giữ nguyên toàn bộ model ngẫu nhiên, không báo lỗi. Muốn nhập model phải unwrap payload, kiểm tra đủ key/shape và ánh xạ observation rõ ràng.

## 7. Artifact, tái chạy và giới hạn

[Bản tổng hợp tám cấu hình](D:/fpt/subject/BomberMan/_comparison/updated_results/summary_19000.json) chứa số liệu và thay đổi theo cặp. Trong [thư mục kết quả](D:/fpt/subject/BomberMan/_comparison/updated_results) có JSON từng trận, hash code/checkpoint, inventory 129 file và kiểm tra loader với checkpoint v19 thật.

[Script đánh giá](D:/fpt/subject/BomberMan/_comparison/compare_new_checkpoints.py) tái chạy bằng Python D:\anaconda\envs\torch\python.exe với tên cấu hình:

- current_current_mask
- v18_current_mask, v19_current_mask
- v18_Loc_mask, v19_Loc_mask
- v18_native_on, v19_native_on, v19_native_off

Các nguồn Python và checkpoint có hash giữ nguyên từ đầu tới cuối benchmark. Các assert nạp đầy đủ trọng số, mapping tám kênh, số trận, seed/góc và observer đều qua. Bộ 48/39 unit test đã qua ở lần so sánh trước; Python không thay đổi nên không chạy lại chỉ vì thêm file trọng số.

[Báo cáo cũ đã lưu riêng](D:/fpt/subject/BomberMan/_comparison/COMPARISON_Loc_update_b00eca8.md). Các dữ liệu seed 18000 cũ không bị ghi đè.

Chưa train lại, chưa chạy thử các thay đổi được đề xuất, chưa đánh giá 11×11/FFA 3–4 tác tử hoặc mọi snapshot. V18 best/v19 final được chọn trước test từ khuyến nghị của nhánh, không tìm model tốt nhất trong cả 119 file trên cùng test seed. Không suy rộng số liệu mẫu này thành bảo đảm trên mọi bàn/đối thủ. Khi chỉnh code hoặc chọn mô hình tiếp từ kết quả này, dùng bộ seed cuối mới.
