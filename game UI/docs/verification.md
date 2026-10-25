# Nghiệm thu Game UI — 23/10/2026

Bản local đã triển khai theo kế hoạch Game Studio, trong `E:\Bomberman\game UI`. Phaser chỉ dựng hình; engine, policy, timer và kết quả vẫn do Python quyết định. Có Chơi 3/4 tác tử, Xem AI 2/3/4 tác tử và huấn luyện C/D/E bằng callback thật.

## Kiểm tra trên mã cuối

Lệnh: `..\venv\Scripts\python.exe tools\verify.py` hoặc `test.cmd` trong folder này. Exit code **0**. Log: `artifacts/verification-final.log`.

| Gate | Kết quả |
|---|---|
| Python: 19 acceptance độc lập + 3 asset + 6 lifecycle | **28/28**, 9,091 giây |
| Input TypeScript | **3/3** |
| TypeScript `tsc --noEmit` | PASS |
| Vite production build | PASS |
| Chrome: 9 journey độc lập + 3 hồi quy | **12/12**, 55,3 giây, không retry/skip |

Acceptance kiểm tra action tape khớp engine từng bước, snapshot là bản copy, tap ngắn/bom ưu tiên, pause/reset/single-step, sequence/ACK trùng, mất và nối lại socket, cấu hình/model lỗi, giới hạn job và output train thật. Browser kiểm tra thao tác phím/touch, kết quả, font/asset local mới, responsive và metric từ train thật.

Python/Chrome dùng các job CPU mới có ngân sách rất ngắn; bài kiểm tra train thật giới hạn 1–2 episode và tối đa 8 bước/ván. Output nằm trong `game UI/runs`. Không có quality evaluation, mở held-out tournament, thay checkpoint gốc hay kết luận AI chơi tốt hơn.

## Rà soát và sửa lỗi

Review độc lập ban đầu ở `docs/code-review.md` đã tìm thấy 3 Important và 1 Minor. Đã sửa và có RED → GREEN cho:

- Nhường lock giữa từng tick khi inference chậm, để pause/state nhận điều khiển.
- Mọi lỗi khởi tạo training worker chuyển sang terminal error; dispatch thất bại không giữ slot.
- Giữ input sequence tăng xuyên suốt reset; xóa cả phím client và touch khi pause/reset/disconnect.
- Checkpoint hỏng (`UnpicklingError`, `EOFError`, `IndexError`) trả lỗi HTTP 422 đọc được.
- Cảnh báo vòng dùng số bước còn lại từ engine đúng một lần; kiểm tra trước cửa sổ cảnh báo, còn 40 bước và còn 1 bước.
- Khóa form setup trong lúc tải catalog, tránh ghi đè số tác tử người dùng vừa chọn; browser xác nhận trận thực sự có 3 tác tử.

Log RED theo từng nhóm: `acceptance-red-parent.log`, `review-regression-red.log`, `review-browser-red.log`, `checkpoint-final-red.log`, `ring-regression-red.log`, `setup-regression-red.log` trong `artifacts/`. GREEN cuối là toàn bộ runner ở trên. Acceptance assertions độc lập không bị sửa bởi người triển khai. Tác giả test đổi tracing sang opt-in vì Chrome software WebGL quá chậm khi ghi trace; giữ nguyên timeout và assertions, giải thích ở `docs/acceptance-author-corrections.md`.

## Hình ảnh và dung lượng

`artifacts/screenshots/` có menu, setup, settings, gameplay thật desktop/mobile, pause, disconnect, training thật và fixture engine cho bom dưới chân, nổ dây chuyền, vòng cảnh báo, kết quả hòa. `fixture-*` là trạng thái engine có kiểm soát phục vụ dựng hình, không phải trận đánh giá policy. Nhân vật, timer bom, lửa, vòng và dấu tiếng Việt đã được xem trực tiếp.

`node tools/visual_qa.mjs` chạy thành công trên server local cuối, không có JavaScript page error. Kiểm tra bàn không tràn ngang tại 1366×768, 1920×1080, 768×1024, 390×844 và 844×390. Browser acceptance kiểm tra thêm các nút không che ô bàn, điều khiển touch và focus bàn phím.

- Runtime asset v2: **26 file, 1.077.354 byte**; 144 frame nhân vật, 12 tile, 68 frame FX, avatar, nền, SVG, font và 8 SFX mới.
- Toàn bộ `dist`: **2.342.135 byte**, dưới ngân sách 8 MB; raw ImageGen không nằm trong build.
- Provenance/prompt và giấy phép font: `docs/asset-prompts.md`, `public/assets/v2/fonts/OFL.txt`.
- 12 bước checkpoint thật: engine P50 **9,26 ms**, P95 **22,35 ms**; click → quan sát state P50 **79,40 ms**, P95 **232,13 ms**. Đây là mẫu local nhỏ có chi phí automation, không phải benchmark production hay độ trễ WebSocket thuần.
- RAF trong JSON đo refresh của trình duyệt, **không đo FPS Phaser**. Chưa xác nhận mục tiêu render gần 60 FPS trên thiết bị người dùng; không dùng số RAF để kết luận tốc độ game/AI.

Chi tiết máy đọc được: `artifacts/verification.json`, `artifacts/visual-qa.json` và `tests/_artifacts/browser-results.json`.

## Phạm vi và giới hạn

Baseline bộ test gốc **trước khi sửa production UI**: 535 test, **20 error, 1 failure, 2 skip**, 701,772 giây (`artifacts/root-baseline.log`). Các lỗi thuộc suite/thí nghiệm cũ; không gọi toàn repo PASS. Sau đó không chạy lại suite gốc tốn thời gian vì engine/trainer/bot/GUI cũ không thay đổi. `git diff` xác nhận không sửa source Python gốc hoặc checkpoint; SHA hiện tại được lưu trong `artifacts/verification.json`. File pointer do test gốc ghi đã phục hồi về đúng HEAD.

Server mới bind loopback, chưa có multiplayer hoặc triển khai Internet. Chưa đo memory soak dài hoặc FPS thiết bị thật. Đã kiểm tra reset/reconnect có giới hạn, không phải bài soak. Không commit/push; giữ các thay đổi và thí nghiệm ngoài phạm vi.

## Chạy

```powershell
cd 'E:\Bomberman'
& '.\game UI\start.cmd'
```

Mở http://127.0.0.1:8000. Nếu server đang chạy, chỉ cần mở trang. Cài lại/build: `game UI\setup.cmd`; hướng dẫn đầy đủ: `game UI/README.md`.

## Cập nhật kích thước bàn theo yêu cầu — 23/10/2026

Sau nghiệm thu toàn tính năng phía trên, tăng khung desktop lên tối đa720px và bỏ cap theo chiều cao cửa sổ. Canvas đo được698×698px tại1366×768,1440×1000,1920×1080; trước đó lần lượt396/578/578px. 768×1024 vẫn538px và390×844 vẫn346px. Bàn giữ tỷ lệ vuông, không tràn ngang hoặc có nút che ô; màn hình desktop thấp có thể cần cuộn dọc.

Đã build/typecheck lại, chạy probe năm viewport và hai ca browser có sẵn cho desktop/asset và mobile/touch. Lượt desktop chạy đồng thời probe bị timeout tải asset; chạy riêng cùng assertion/timeout đạt1/1 trong5,6s, mobile đạt1/1. Không chạy lại toàn bộ suite Python vì chỉ sửa CSS. Ảnh mới: `artifacts/screenshots/enlarged-map-desktop.png`; số đo và log mang prefix `artifacts/map-size-*`. Các con số full-suite/dung lượng bên trên là snapshot của lượt bàn giao trước thay đổi CSS này.
