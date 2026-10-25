# Bomberman · Robot Foundry

Giao diện game 2D dùng **Phaser + TypeScript + Vite**. Engine, AI và huấn luyện chạy bằng Python của repo Bomberman. Mã mới, asset, dependency, bản build, test và output job đều nằm trong folder `game UI`.

## Chạy ngay trên Windows

Máy hiện tại đã được cài dependency và build. Mở PowerShell:

```powershell
cd 'E:\Bomberman'
& '.\game UI\start.cmd'
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000). Dừng server bằng `Ctrl+C`. Nếu port đã được dùng, chạy `& '.\game UI\start.cmd' --port 8001` rồi mở port tương ứng.

## Cài đặt hoặc build lại

Cần Node.js >=22.18 và môi trường `E:\Bomberman\venv\Scripts\python.exe` có Torch/NumPy. Không cần API key để chạy game hoặc dùng các asset đã lưu.

```powershell
cd 'E:\Bomberman\game UI'
.\setup.cmd
.\start.cmd
```

`setup.cmd` cài Python dependency đã khóa vào `.python-deps`, npm dependency vào `node_modules`, cache vào các folder riêng trong `game UI`, sau đó typecheck và build. Không cài lại Torch hoặc sửa môi trường Python hiện có.

Khi phát triển, chạy `start.cmd` ở terminal thứ nhất; chạy `npm run dev` ở terminal thứ hai trong folder này, mở [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite proxy HTTP/WS tới Python port 8000. Sau khi build lại frontend có thể refresh trang; sau khi sửa Python cần khởi động lại server.

## Cách chơi

- **Chơi ngay:** bạn là A0, AI được chọn là A1; còn lại là RuleBot. Hỗ trợ 3 hoặc 4 tác tử.
- **Xem AI thi đấu:** A0 là AI đã huấn luyện, đối đầu RuleBot; hỗ trợ 2–4 tác tử.
- `WASD` hoặc mũi tên di chuyển; `Space`/`E` đặt bom. Mobile có pad điều hướng và nút bom.
- Ván mới bắt đầu ở trạng thái tạm dừng. **Tiếp tục** để chạy, **Một bước** để quan sát một tick, nút vòng tròn để reset.
- `Esc` mở menu tạm dừng. Mất focus hoặc mất kết nối sẽ pause và giải phóng phím; reconnect giữ ván cũ, không tự resume.
- Timer bom tính theo **bước**, không phải số giây thực. Menu Cài đặt có giảm chuyển động, vùng nguy hiểm, mute và âm lượng.

Catalog đọc weights `best_*.pt`/`final_*.pt` trong `checkpoints/`, `runs/*/` và job mới tại `game UI/runs/*/`. Client chỉ gửi ID do server cấp. Mặc định cùng checkpoint `best_phaseE_4a.pt` của GUI cũ; không tự coi mô hình khác là tốt nhất. File full-state optimizer không phải weights để chơi.

## Phòng huấn luyện

Chọn Phase C/D (2 tác tử) hoặc E (2/3/4), số episode và checkpoint warm-start nếu cần. Bàn và biểu đồ lấy từ callback thật của trainer: WR, ActiveWR, reward, loss; mặc định biểu đồ metric tổng hợp trên 50 episode gần nhất, đường biểu diễn 100 episode gần nhất.

Job web chạy CPU và lưu config, JSONL metric, checkpoint vào `game UI/runs/ui-...`. Pause/resume và **Dừng & lưu** chỉ điều khiển job do giao diện tạo. Không gắn vào hoặc sửa lượt thí nghiệm ở `E:\Bomberman\runs`. Giao diện không tự chạy quality evaluation/held-out tournament sau train.

Không khởi chạy train dài khi chỉ muốn thử UI. Trong “Tùy chọn phiên huấn luyện”, có thể đặt tối đa 8 bước/ván và 1–2 episode cho smoke. Các kết quả smoke này không chứng minh chất lượng AI.

## Kiểm tra

```powershell
cd 'E:\Bomberman\game UI'
.\test.cmd
```

Lệnh này chạy toàn bộ test của project web, typecheck/build, tự mở server kiểm thử tại port local trống và kiểm tra bằng Chrome headless, sau đó đóng server kiểm thử. Chrome mặc định: `C:\Program Files\Google\Chrome\Application\chrome.exe`; có thể đặt `BOMBERMAN_CHROME` để đổi executable. Test không điều khiển server bạn đang chơi.

Các kiểm tra riêng:

```powershell
npm run test:unit
npm run build
# Nếu server local đã chạy:
npm run test:browser
```

HTTP/WS acceptance độc lập nằm tại `tests/python/test_acceptance.py`; traceability và baseline RED ở `docs/acceptance-matrix.md`, `docs/acceptance-red-baseline.md`. Bộ test gốc của repo được ghi riêng trong báo cáo nghiệm thu; không gộp các lỗi có sẵn thành kết quả của UI.

Bản mã cuối đạt **28 test Python, 3 test input, 12 test Chrome, typecheck và build**. Báo cáo, ảnh và giới hạn: [docs/verification.md](docs/verification.md). Bật ghi trace khi cần bằng `BOMBERMAN_TRACE=1`; mặc định tắt để tránh overhead trên Chrome software WebGL.

## Source map

| Folder/file | Trách nhiệm |
|---|---|
| `src/main.ts`, `views.ts`, `style.css` | Navigation, form/HUD, cài đặt, training monitor, giao diện responsive |
| `src/arena.ts` | Phaser renderer: tile, sprite, animation, bom/FX; không quyết định luật |
| `src/input.ts`, `transport.ts` | Input có sequence, command ACK, heartbeat/reconnect |
| `bridge/sessions.py` | Owner tuần tự của mỗi engine, tick, queue input và snapshot |
| `bridge/snapshots.py` | Copy dataclass/NumPy sang JSON, ring và kết quả trận |
| `bridge/models.py` | Catalog allowlist và cache weights inference |
| `bridge/training.py` | Job worker CPU có quyền sở hữu riêng và callback metric/frame |
| `bridge/app.py`, `protocol.py` | REST/WS v1, validation/origin checks, static build |
| `public/assets/v2/` | Chỉ asset runtime mới; manifest có đường dẫn và nguồn |
| `assets/source/`, `assets/previews/` | Raw ImageGen và preview; không đưa vào production build |
| `tools/prepare_assets.py` | Chuẩn hóa/pack các sheet, SVG và SFX nguyên bản |
| `tools/sprite-pipeline/` | Helper chuẩn hóa từ Game Studio 0.1.2 |
| `tests/`, `artifacts/` | Acceptance/unit/visual fixtures, screenshot, log và báo cáo |
| `runs/` | Chỉ output các job mới do giao diện tạo |

## Asset và nguồn

Bộ mới gồm **144 frame nhân vật, 12 tile môi trường, 68 frame hiệu ứng, 4 avatar, nền menu, logo/icon SVG, 3 weight font và 8 SFX**. Nhân vật/terrain/FX/background được tạo bằng ImageGen; logo/icon/SFX là code mới. Font Be Vietnam Pro self-hosted kèm OFL. Prompt và nguồn: [docs/asset-prompts.md](docs/asset-prompts.md).

Tái tạo asset runtime từ source đã lưu:

```powershell
..\venv\Scripts\python.exe tools\prepare_assets.py
npm run build
```

Tất cả thế giới gameplay vẫn lấy dữ liệu từ `../bomberman_env.py`; AI dùng `../train_actor_critic.py` và `../rule_based_bot.py`. GUI Tkinter cũ tiếp tục có thể chạy bằng `python visualize.py` ở repo root. Server mới chỉ bind `127.0.0.1`; đây là bản local, chưa có multiplayer hoặc triển khai Internet.
