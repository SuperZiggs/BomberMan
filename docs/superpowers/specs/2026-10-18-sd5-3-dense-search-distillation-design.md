# SD5.3 — Chưng cất kill-search dày vào CNN (hướng b)

Ngày: 18/10/2026. Trạng thái: **bản thiết kế, chờ người dùng duyệt**; chưa có code.
Liên quan: `PLAN_SUDDEN_DEATH.md` (SD5), `RESULTS.md` (SD5.0, SD5.1 kết quả), `IMPROVEMENTS.md` §35–§39.

Tài liệu này dành cho hai người đọc:
- người cài đặt (viết code runner, test);
- người chạy hộ trên máy khác (mục 8: chỉ cần làm theo lệnh, rồi gửi lại file zip).

---

## 1. Lý do

Mục tiêu của hướng SD là tăng **ActiveWR** (thắng và có hạ gục bằng bom của mình) của agent CNN trước RuleBot, đồng thời giảm hòa. Thắng thụ động không tính là giỏi hơn.

Hai kết quả gần nhất mâu thuẫn nhau, và chính chỗ mâu thuẫn đó là lý do có SD5.3:

| Thí nghiệm | Kết quả |
|---|---|
| SD5.0: F + kill-search **lúc chơi** (validation 21000, 400 trận, lịch E) | ActiveWR 47.0 → **69.5**, delta **+22.5 [18.25, 26.75]**; hòa −5.75 |
| SD5.1: F + nhãn kill-search **khi train RL** 3000 ep (test 42000) | search − F **−2.75 [−9.75, 4.0]**; control − F −6.5; gate FAIL |

Như vậy kill-search biết những nước hạ đối thủ mà F không biết, nhưng cách chưng cất của SD5.1 không truyền được hiểu biết đó vào CNN. Người dùng chọn giữ agent là CNN thuần, không chạy search lúc chơi (hướng a bị loại vì search dùng chính code RuleBot làm mô hình đối thủ và chậm hơn khoảng 6 lần). Vì vậy cần một cách chưng cất mạnh hơn.

## 2. Hiện trạng

**Model và luật:**
- F = `runs/phaseC-sd-best-fallback/best_phaseC.pt` (SHA256 `0315a56f09079333d936db5f55587093f1b6415ed6355d2ef6ae2ab63f8b1483`), CNN 13 kênh, là mốc và điểm xuất phát.
- Luật chính là lịch E (SD engine 475/25/40). Báo cáo thêm khi tắt SD.

**`kill_search.search_action(env, aid, mask, default_action)`:**
- Chỉ chạy khi có đối thủ trong khoảng Manhattan ≤ 6.
- Với mỗi hành động mask cho phép, nó mô phỏng 12 bước: learner làm hành động đó rồi chỉ né, đối thủ chơi bằng RuleBot.
- Điểm của mỗi hành động: +1 nếu bom của learner giết đối thủ mà learner còn sống, −1 nếu learner chết, 0 nếu không có gì.
- Hành động của CNN được giữ, trừ khi có hành động khác điểm cao hơn hẳn.

**Vì sao SD5.1 thất bại** (số đo trong `runs/sd5-20261017`):
- Search đã được gọi ở gần như mọi bước: 914 403 lần gọi, nhưng chỉ ra 2 675 nhãn, tức 0.89 nhãn mỗi episode.
- Nhãn chỉ được sinh khi search sửa nước của CNN. Những lần CNN đã đi đúng nước giết được thì không được củng cố.
- Mỗi nhãn chỉ được học **một lần**, với lr 2.5e-5, giữa nhiễu của gradient RL. Tín hiệu tổng cộng quá nhỏ.
- Train RL thêm 3000 ep từ F (nhánh control) cũng không cải thiện: công thức này đã bão hòa.

**Đo mật độ khoảnh khắc quyết định** (18/10/2026, 20 trận F + search, seed chẩn đoán 6600, lịch E):

| Đại lượng | Tổng (20 trận) | Mỗi trận |
|---|---:|---:|
| Bước của learner | 4603 | 230 |
| Bước có đối thủ trong tầm (search chạy) | 3975 | 199 |
| Bước **có thông tin** (điểm các hành động không bằng nhau) | 44 | 2.2 |
| — trong đó có nước giết được (điểm tốt nhất = +1) | 25 | 1.25 |
| — search sửa nước của CNN | 22 | 1.1 (8 lần là đặt bom) |
| — CNN đã chọn đúng nước giết | 12 | 0.6 |
| Thời gian | 5.0 s/trận, 1 tiến trình, 2 luồng | |

Hai điều rút ra từ bảng này:
- Ở 99% số bước, mọi hành động cùng điểm, nên search không có ý kiến. Gắn nhãn search ở mọi bước chỉ là bắt CNN học lại chính nó, không thêm thông tin gì.
- Thông tin nằm trong khoảng 2 bước mỗi trận. Muốn CNN học được, phải **gom thật nhiều trạng thái như vậy, học nhiều lần, có trọng số rõ ràng**, và giữ nguyên phần hành vi còn lại của F.

## 3. Mục tiêu, không phải mục tiêu, phản mục tiêu

**Mục tiêu:** một checkpoint CNN thuần (cùng kiến trúc 13 kênh, không gọi search lúc chơi) có ActiveWR trên test seed 43000 cao hơn F ít nhất 5pp (CI95 dưới > 0), tự nổ không tăng quá 1pp. Gate chốt ở mục 6.

**Không phải mục tiêu:**
- Không đổi RuleBot, `kill_search.py`, luật env hay lịch E.
- Không train RL thêm (đã bão hòa ở SD5.1).
- Không đổi kiến trúc mạng.
- Không giải quyết riêng chuyện hòa (đó là SD5.2, kiểu PLR).

**Phản mục tiêu** (kỹ thuật thì chạy, nhưng vẫn tính là thất bại):
- WR tăng nhờ thắng thụ động trong khi ActiveWR không tăng.
- Model học thuộc lòng các bàn của tập thu thập (trùng seed bàn với validation hoặc test).
- Đánh mất kỹ năng sống sót của F (tự nổ tăng) để đổi lấy vài kill.
- Chọn checkpoint sau khi đã xem test, hoặc chạy test lần hai.

## 4. Hướng xử lý

Hướng chọn là **behaviour cloning có trọng tâm**: cho agent F + search tự chơi, chỉ thu các trạng thái có thông tin, rồi fine-tune F bằng học có giám sát trên tập đó. Một loss neo giữ cho mọi trạng thái khác vẫn giống F.

Các phương án đã cân nhắc và bỏ:

| Phương án | Vì sao bỏ |
|---|---|
| Nhãn search ở mọi bước | 99% bước là hòa điểm, nhãn trùng hành động CNN; không có thông tin mới (mục 2). |
| SD5.1 với `search_loss_coef` cao hơn | Vẫn 0.9 nhãn/ep, mỗi nhãn học một lần, lẫn trong nhiễu RL; tăng hệ số chỉ làm bước gradient của vài mẫu to hơn. |
| DAgger nhiều vòng | Agent F + search chỉ lệch khỏi F khoảng 1 nước mỗi trận, nên trạng thái hai agent gặp gần như cùng phân phối. Một vòng là đủ cho lượt đầu; vòng 2 chỉ mở có điều kiện (mục 7). |
| Chạy search lúc chơi (hướng a) | Người dùng loại. |

## 5. Giải quyết (thiết kế)

### 5.1. Giai đoạn `collect` — thu dữ liệu

- Agent thu thập: F lấy mẫu hành động như bình thường, rồi `search_action` sửa nếu có nước tốt hơn hẳn (giống arm `search` của `probe_sd5_search.evaluate_arm`). Đối thủ là RuleBot.
- Số trận: **3000** (đúng quy ước 3000 episode mỗi lượt), bàn 9×9, 2 agent, 500 bước. Learner luân phiên hai góc như `evaluate_seeded`.
- Seed bàn: dải riêng **200000–201499** (1500 bàn × 2 góc). Không trùng validation (21000, 6600, 15000) hay bất kỳ seed test nào (16000–43000, 60000).
- SD: trận có lịch E với xác suất 0.75, quyết định theo seed bàn để chia shard vẫn tái lập được (như cách train SD5.1 dùng 75%).
- Ghi lại:
  1. **Mẫu quyết định**: mọi bước mà search chạy và điểm các hành động không bằng nhau. Ghi obs (13×9×9), mask, điểm của từng hành động, hành động CNN đã lấy mẫu, hành động thực hiện, cùng seed bàn, góc và bước để truy vết.
  2. **Mẫu neo**: một mẫu ngẫu nhiên 5% các bước còn lại, ghi obs, mask và logits của F. Dự kiến khoảng 34 000 mẫu.
- Dự kiến khoảng 6 600 mẫu quyết định, trong đó khoảng 3 750 có nước giết.
- Chạy song song theo shard seed bàn (mặc định 4 tiến trình). Mỗi shard ghi một file riêng. `--resume` bỏ qua các shard đã xong. Có thanh progress (`progress.parallel_map`).
- Kèm thống kê thu thập: ActiveWR, WR, hòa, tự nổ của agent F + search trên các trận này (chỉ báo cáo).

### 5.2. Giai đoạn `train` — fine-tune có giám sát từ F

- Khởi tạo từ F; chỉ cập nhật policy head và thân CNN, value head không dùng trong loss.
- **Đích của mẫu quyết định**: phân phối đều trên các hành động có điểm tốt nhất (trong mask). Ví dụ có hai nước cùng giết được thì mỗi nước 0.5. Loss là cross-entropy với đích mềm, tính trên logits đã che mask (giống cách trainer che mask).
- **Loss neo**: KL(F ‖ model) trên mẫu neo, để hành vi ngoài các khoảnh khắc quyết định vẫn là của F.
- Loss tổng = CE_quyết định + λ · KL_neo, với **λ = 1.0**. Mỗi batch gồm một nửa mẫu quyết định, một nửa mẫu neo.
- **Tăng cường dữ liệu bằng đối xứng bàn**: bàn 9×9 có cột cứng ở ô (chẵn, chẵn) và vòng SD là các vành vuông đồng tâm, nên luật env đối xứng dưới 8 phép quay và lật. Kênh 8–10 là hằng số trên toàn bàn; các kênh khác là ảnh không gian. Áp dụng ngẫu nhiên một phép cho mỗi mẫu và đổi hành động tương ứng (có sẵn `benchmark_t3.transform` và `transform_action`). Người cài đặt phải kiểm hai việc:
  1. Obs biến đổi đúng: so obs của env đã biến đổi với obs đã biến đổi, trên vài trạng thái thật (unit test).
  2. **Nhãn vẫn đúng sau biến đổi**: RuleBot có thể phá tính đối xứng, qua thứ tự xét hành động khi hòa điểm và nhánh ngẫu nhiên. Luật chốt trước: trên ≥ 200 mẫu quyết định, dựng env đã biến đổi và chạy lại search. Nếu tập nước tốt nhất trùng với nhãn đã biến đổi ở ≥ 90% mẫu thì bật tăng cường; dưới 90% thì tắt tăng cường cho cả lượt. Tỉ lệ đo được ghi vào `registration.json` trước khi train.
- Chia 90/10 theo **seed bàn** (không theo mẫu) thành train và held-out.
- Tham số: Adam, lr 1e-4, batch 256, **8 epoch**, lưu checkpoint ở epoch 2, 4 và 8.
- Metric offline trên held-out, mỗi epoch:
  - tỉ lệ argmax nằm trong tập hành động tốt nhất ở mẫu quyết định, tách riêng nhóm có nước giết;
  - KL neo;
  - tỉ lệ đổi argmax so với F ở mẫu neo.
- Chạy trên GPU nếu có. Dự kiến vài phút.

### 5.3. Giai đoạn `validate` — chọn checkpoint (luật chốt trước)

- Ứng viên: epoch 2, 4 và 8. Mỗi ứng viên đánh giá trên validation 21000, 400 trận, lịch E, CNN thuần (không search).
- Chọn ứng viên có ActiveWR cao nhất trong số các ứng viên có tự nổ ≤ tự nổ của F + 1pp (F được đo trong cùng lượt, cùng máy).
- Nếu không ứng viên nào hơn F trên validation, vẫn chọn theo luật trên rồi đi tiếp tới test, không đổi luật. Kết quả lúc đó gần như chắc chắn là FAIL; ghi đúng như vậy.
- Chỉ có 3 ứng viên vì ở SD5.1, chọn best trong 7 điểm validation đã thiên lạc quan (control: validation 51.0, test 42.0).

### 5.4. Giai đoạn `test` — một lần, seed 43000

- Model test: F, SD5.3-best (CNN thuần), và **F + search** (chỉ báo cáo, làm trần tham chiếu, không vào gate).
- Mỗi model 400 trận dưới lịch E (luật gate) và khi tắt SD (báo cáo).
- Mở seed bằng `test_opened.json` (mở file ở mode `'x'`), từ chối chạy lần hai.
- So sánh bằng paired bootstrap theo seed bàn (`pilot_two_bomb.paired_active_bootstrap`) cho ActiveWR, WR, hòa và tự nổ.
- Ghi `summary.json` rồi ZIP thành `runs/<output>-send.zip`.

### 5.5. Hạ tầng (theo protocol sẵn có)

- Runner mới `run_sd53_pilot.py` với các stage `collect [--resume]`, `train`, `validate`, `test` và `--smoke` cho mỗi stage. Một thư mục output, từ chối ghi đè. Ghi config, SHA của F, và source cùng SHA vào `registration.json`.
- **SHA source tính trên nội dung đã chuẩn hóa xuống dòng** (CRLF → LF). Đây là bài học từ SD5.1, khi máy clone từ GitHub cho SHA khác chỉ vì CRLF. Đồng thời ghi provenance runtime (torch, numpy, GPU).
- Mọi stage có thanh progress (`progress.py`).
- **Cả bốn stage chạy trên cùng một máy**, kể cả lần đánh giá F, để không lẫn sai khác giữa phiên bản torch: cùng F mà validation ra 47.0 trên torch 2.5.1 và 43.75 trên torch 2.14.1.
- Test viết trước (TDD):
  - đích mềm đúng khi có hai hành động hòa điểm;
  - đối xứng obs và hành động đúng;
  - chia train/held-out không lẫn seed bàn;
  - collect với resume không ghi trùng shard;
  - test chỉ mở được một lần;
  - gate;
  - smoke đầu-cuối.

## 6. Gate (chốt trước khi chạy)

Trên test seed 43000, lịch E, SD5.3-best so với F:

1. ActiveWR ≥ **+5pp**;
2. CI95 dưới của delta ActiveWR **> 0**;
3. tự nổ tăng không quá **1pp**.

Đạt cả ba thì SD5.3-best thành model mới (thay F). Không đạt thì giữ F. Số liệu khi tắt SD và của F + search chỉ để báo cáo.

## 7. Sau khi có kết quả (điều kiện mở hướng tiếp)

- Đạt: cân nhắc vòng DAgger thứ 2. Thu 3000 trận với SD5.3-best + search, gộp vào dữ liệu cũ, train lại, rồi test trên seed mới 44000.
- Không đạt, nhưng held-out cho thấy model đã học được nước giết (tỉ lệ đúng ở mẫu có nước giết tăng rõ so với F) mà ActiveWR không tăng: lỗi nằm ở chỗ chuyển từ quyết định đơn lẻ sang kết quả trận. Xem lại tầm search 12 bước hoặc cách chấm điểm.
- Không đạt và held-out cũng không học được: dữ liệu hoặc kiến trúc không đủ. Quay sang SD5.2 (đòn bẩy hòa).

## 8. Người chạy hộ cần gì

**Yêu cầu máy:** Windows, có Python kèm torch (đã chạy được SD5.1), git và git-lfs, khoảng 2 GB đĩa trống. Nên có GPU cho stage `train`; các stage còn lại chạy CPU. RAM: mỗi tiến trình nạp torch riêng, mặc định 4 tiến trình, có thể đặt `--workers` thấp hơn.

**Thời gian ước tính** (theo số đo trên máy RTX 3050; máy mạnh hơn sẽ nhanh hơn):

| Stage | Khối lượng | Ước tính |
|---|---|---|
| collect | 3000 trận F + search, khoảng 5 s/trận | ~1.1 giờ với 4 tiến trình |
| train | ~6 600 mẫu quyết định + ~34 000 mẫu neo, 8 epoch | vài phút (GPU) |
| validate | F + 3 ứng viên × 400 trận, khoảng 1.1 s/trận | ~10 phút với 3 tiến trình |
| test | F, SD5.3 × {E, tắt SD} × 400 trận + F + search 400 trận | ~30–45 phút |

**Lệnh** (cmd, trong thư mục repo đã checkout nhánh `danh/sd5-friend-merge` sau khi code được push; đổi `PY` thành đường dẫn python có torch):

```cmd
git pull
git lfs pull
set PY=D:\anaconda\envs\torch\python.exe
%PY% -m unittest test_run_sd53_pilot -q
%PY% -u run_sd53_pilot.py collect  --output runs\sd53-20261018 > runs\sd53-20261018-collect.log 2>&1
%PY% -u run_sd53_pilot.py train    --output runs\sd53-20261018 > runs\sd53-20261018-train.log 2>&1
%PY% -u run_sd53_pilot.py validate --output runs\sd53-20261018 > runs\sd53-20261018-validate.log 2>&1
%PY% -u run_sd53_pilot.py test     --output runs\sd53-20261018 > runs\sd53-20261018-test.log 2>&1
```

**Quy tắc khi chạy:**
- Nếu `collect` bị ngắt, chạy lại đúng lệnh đó thêm `--resume` và đổi `>` thành `>>`.
- Không chạy lại `test`. Nó chỉ chạy được một lần, và đó là chủ ý.
- Không sửa code hay tham số giữa các stage.

**Gửi lại:** file `runs\sd53-20261018-send.zip` (được tạo sau `test`) cùng 4 file `.log`.

## 8b. Ghi chú từ SD6 (18/10/2026, sau khi viết spec)

Các phát hiện sau đến sau bản spec đầu, người cài đặt nên biết (chi tiết ở `INSIGHTS.md`):
- **Không dùng argmax** khi thu thập hay đánh giá: F argmax có ActiveWR −9.25 so với lấy mẫu. Agent thu thập giữ cách lấy mẫu của F, như mục 5.1 đã ghi.
- Trong các mẫu quyết định có nước giết, khoảng một nửa là nước **đứng yên/di chuyển** mà F gán xác suất khoảng 0.001. Chủ yếu là STOP ở ô cửa thoát của đối thủ, trong khi F bỏ chạy. Nên báo riêng metric held-out cho nhóm này, vì đây là phần F thực sự chưa biết.
- Đích chỉ nên lấy từ điểm của `kill_search`: kill bằng bom của learner. Một luật "làm đối thủ hết đường" không phân biệt bom của ai sẽ chủ yếu tạo thắng thụ động (SD6.1b).
- Probe/runner nhiều tiến trình: chia shard theo bàn, lưu kết quả thô trước khi gộp, smoke test chạy cả bước gộp.

## 9. Quyết định đã chốt và phần để người cài đặt

**Người dùng đã chốt:**
- Đi theo hướng b (CNN thuần, chưng cất search).
- Mỗi lượt là 3000 episode.
- Người chạy là người khác, trên máy khác.

**Tôi đề xuất, người dùng có thể đổi khi duyệt:**
- Behaviour cloning một vòng, thay cho DAgger nhiều vòng.
- λ = 1.0, lr 1e-4, 8 epoch, 3 ứng viên.
- Mẫu neo 5%.
- Đích mềm đều trên các nước tốt nhất.
- Có tăng cường bằng đối xứng bàn.
- Dải seed thu thập 200000+.
- Gate giống SD5.1 nhưng so với F.
- Thêm F + search vào test (chỉ báo cáo).
- Cả 4 stage chạy cùng một máy.

**Để người cài đặt quyết:**
- Định dạng file dữ liệu (`.npz` mỗi shard là đủ).
- Cách che logits cho phép bị mask.
- Cấu trúc hàm trong runner.
- Số luồng torch cho mỗi tiến trình.

**Giới hạn đã biết:**
- Một train seed.
- Search dùng mô hình RuleBot hoàn hảo, nên thứ CNN học là cách hạ chính RuleBot; khả năng tổng quát sang đối thủ khác không được đo.
- Mật độ mẫu đo trên 20 trận, có thể lệch vài chục phần trăm.
