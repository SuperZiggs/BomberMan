import type { Model, Settings } from './types';

export const colors = ['mint', 'blue', 'coral', 'gold'];
export const names = ['MINT', 'COBALT', 'CORAL', 'AMBER'];
export const escape = (value: unknown) => String(value).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]!));
export const icon = (name: string) => `<svg class="icon" aria-hidden="true"><use href="/assets/v2/ui/icons.svg#${name}"></use></svg>`;
const logo = `<img class="brand-mark" src="/assets/v2/ui/logo.svg" alt="" width="40" height="40"><span>BOMBERMAN<small>ROBOT FOUNDRY</small></span>`;
export const options = (models: Model[]) => models.map(m => `<option value="${escape(m.id)}">${escape(m.name)} · ${escape(m.source.split('/').slice(0, -1).join('/'))}</option>`).join('');

function header(subtitle = '', home = false) {
  return `<header class="topbar"><button class="brand" data-action="home" aria-label="Về menu">${logo}</button>
    ${subtitle ? `<span class="header-divider"></span><span class="section-name">${subtitle}</span>` : '<span class="edition">ARCADE / EXPERIMENT 01</span>'}
    <nav>${home ? `<button class="text-button" data-action="home">${icon('back')}<span>Về menu</span></button>` : `<button class="text-button" data-action="training">${icon('train')}<span>Huấn luyện</span></button>`}
    <button class="icon-button" data-action="settings" aria-label="Cài đặt">${icon('settings')}</button></nav></header>`;
}

export function menu() {
  return `<main class="menu-view">${header()}<section class="hero">
    <div class="hero-art" aria-hidden="true"></div><div class="hero-grid" aria-hidden="true"></div>
    <div class="hero-copy"><div class="eyebrow"><span class="signal-dot"></span>ĐẤU TRƯỜNG ĐÃ SẴN SÀNG</div>
      <h1>BÙNG NỔ<br><span>CHIẾN THUẬT.</span><b class="title-dot">✦</b></h1>
      <p>Mỗi quả bom là một nước đi.<br>Né thật nhanh. Tính thật xa. Sống sót sau cùng.</p>
      <div class="hero-actions"><button class="button primary large" data-action="setup-play">${icon('play')}Chơi ngay${icon('arrow')}</button>
        <button class="button secondary large" data-action="setup-watch">${icon('watch')}Xem AI thi đấu</button></div>
      <div class="key-hint"><span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> di chuyển</span><span><kbd>SPACE</kbd> đặt bom</span></div>
    </div><div class="arena-stamp"><span class="signal-dot"></span>FOUNDRY ARENA<span>01 / 11 × 11</span></div>
    <div class="hero-caption"><span>01—04</span><i></i><span>CHỈ MỘT ROBOT Ở LẠI.</span></div>
  </section><footer class="menu-footer"><div class="crew"><span class="mini-label">CHỌN VỊ TRÍ CỦA BẠN</span>
    <div class="crew-list">${colors.map((color, i) => `<div class="crew-member ${color}"><img src="/assets/v2/characters/${color}-avatar.png" alt="Robot ${names[i]}" width="40" height="40"><span><b>A${i}</b>${names[i]}</span></div>`).join('')}</div></div>
    <button class="lab-link" data-action="training"><span class="lab-icon">${icon('train')}</span><span><b>Phòng huấn luyện</b><small>Quan sát cách AI học chơi</small></span>${icon('arrow')}</button></footer></main>`;
}

export function setup(mode: 'play' | 'watch', models: Model[] = [], loading = false) {
  return `<div class="modal-heading"><div><span class="eyebrow">MATCH / ${mode === 'play' ? 'PLAYER' : 'SPECTATE'}</span><h2 id="setup-title">${mode === 'play' ? 'Vào đấu trường.' : 'Xem trí tuệ thi đấu.'}</h2></div><button class="icon-button" data-action="close-dialog" aria-label="Đóng">${icon('close')}</button></div>
    <p class="modal-intro">${mode === 'play' ? 'Bạn điều khiển A0. Một tác tử AI và các RuleBot sẽ đối đầu với bạn.' : 'AI đã huấn luyện đối đầu RuleBot. Tạm dừng bất cứ lúc nào để xem từng nước đi.'}</p>
    <form id="match-form"><input type="hidden" name="mode" value="${mode}">
      <label class="field">Số tác tử<select name="num_agents" aria-label="Số tác tử" ${loading ? 'disabled' : ''}>${(mode === 'play' ? [3, 4] : [2, 3, 4]).map(n => `<option value="${n}" ${n === 4 ? 'selected' : ''}>${n} tác tử</option>`).join('')}</select></label>
      <label class="field">Checkpoint AI<select name="model_id" aria-label="Checkpoint AI" required ${loading || !models.length ? 'disabled' : ''}>${models.length ? options(models) : '<option>Đang tải danh sách checkpoint…</option>'}</select><small>Mô hình sẽ được nạp khi bắt đầu ván.</small></label>
      <div class="form-row"><label class="field">Tốc độ ban đầu<select name="speed" ${loading ? 'disabled' : ''}><option value="${mode === 'play' ? 12 : 8}">${mode === 'play' ? 12 : 8} bước / giây</option><option value="4">4 bước / giây</option><option value="20">20 bước / giây</option></select></label><div class="board-chip"><b>11 × 11</b><small>KÍCH THƯỚC BÀN</small></div></div>
      <label class="check-line"><input type="checkbox" name="sudden_death" ${loading ? 'disabled' : ''}>Bật vòng thu hẹp cuối ván</label>
      <p class="form-error" data-modal-error role="alert"></p>
      <button type="submit" class="button primary full" ${loading || !models.length ? 'disabled' : ''}>${loading ? '<span class="spinner"></span>Đang tải…' : `${icon('play')}Bắt đầu ván đấu${icon('arrow')}`}</button>
    </form>`;
}

export function board() {
  return `<div class="arena-frame"><i class="corner tl"></i><i class="corner tr"></i><i class="corner bl"></i><i class="corner br"></i><div id="arena-canvas" aria-label="Bàn đấu Bomberman"></div><div id="canvas-loading"><span class="spinner"></span>Đang nạp đấu trường…</div></div>`;
}

export function match(mode: 'play' | 'watch', model: string) {
  return `<main class="game-view">${header(mode === 'play' ? 'CHƠI TRỰC TIẾP' : 'QUAN SÁT AI', true)}
    <div class="game-heading"><div><div class="eyebrow">FOUNDRY ARENA / ${mode === 'play' ? 'PLAYER A0' : 'SPECTATOR'}</div><h1>${mode === 'play' ? 'Sẵn sàng bùng nổ?' : 'Mỗi nước đi, một chiến thuật.'}</h1></div><span id="connection" class="connection">Đang kết nối…</span></div>
    <div id="connection-banner" class="connection-banner" role="status" hidden></div>
    <div class="arena-layout"><section class="playfield"><div class="board-topbar"><span id="play-status" class="state-chip">Tạm dừng</span><span>BƯỚC <b id="tick">000</b><span class="dim"> / <span id="max-steps">500</span></span></span><span>${icon('alive')}<b id="alive">4</b><small>CÒN LẠI</small></span></div>
      ${board()}<div class="board-legend"><span><i class="legend-swatch wall"></i>Tường cứng</span><span><i class="legend-swatch crate"></i>Thùng phá được</span><span>${icon('bomb')}Timer theo bước</span></div>
      <div class="match-controls"><button class="button primary" id="toggle-run" data-action="toggle">${icon('play')}Tiếp tục</button><button class="button secondary" id="step-button" data-action="step">${icon('step')}Một bước</button><button class="icon-button" data-action="reset" aria-label="Ván mới">${icon('reset')}</button><label class="speed-control">Tốc độ<select id="game-speed" aria-label="Tốc độ bước mỗi giây">${[1, 4, 8, 12, 20, 30, 60].map(n => `<option value="${n}">${n}×</option>`).join('')}</select></label></div>
      ${mode === 'play' ? `<div class="touch-pad" aria-label="Điều khiển cảm ứng"><div class="dpad"><button data-direction="ArrowUp" aria-label="Lên">${icon('up')}</button><button data-direction="ArrowLeft" aria-label="Trái">${icon('left')}</button><span class="pad-center"></span><button data-direction="ArrowRight" aria-label="Phải">${icon('right')}</button><button data-direction="ArrowDown" aria-label="Xuống">${icon('down')}</button></div><button class="bomb-button" data-action="bomb" aria-label="Đặt bom">${icon('bomb')}<span>BOM</span></button></div>` : ''}
      <div class="game-key-hint">${mode === 'play' ? '<kbd>WASD</kbd> / mũi tên · <kbd>SPACE</kbd> / <kbd>E</kbd> đặt bom · ' : ''}<kbd>ESC</kbd> menu tạm dừng</div>
    </section><aside class="roster"><div class="roster-title"><span class="eyebrow">CÁC TÁC TỬ</span><span id="roster-count">04</span></div><div id="agent-cards"></div><div id="ring-warning" class="ring-warning" hidden></div><section id="result-card" class="result-card" aria-live="polite" hidden></section>
      <details class="match-details"><summary>Thông số ván đấu</summary><p class="model-label">${escape(model)}</p><div id="agent-details"></div><p class="dim" id="perf-note">Đang chờ bước đầu tiên.</p></details>
      <div class="arena-note">${icon('danger')}<p>Bom nổ theo hình chữ thập.<br>Tường cứng chặn lửa. Hãy chừa đường thoát.</p></div></aside></div></main>`;
}

export function settings(values: Settings) {
  return `<div class="modal-heading"><div><span class="eyebrow">PREFERENCES</span><h2 id="settings-title">Theo cách của bạn.</h2></div><button class="icon-button" data-action="close-dialog" aria-label="Đóng">${icon('close')}</button></div>
    <div class="settings-options"><label class="check-line"><span><b>Giảm chuyển động</b><small>Giảm animation và chuyển cảnh trang trí.</small></span><input id="setting-motion" type="checkbox" ${values.reducedMotion ? 'checked' : ''}></label>
      <label class="check-line"><span><b>Hiện vùng nguy hiểm</b><small>Đánh dấu các ô theo dự báo của engine.</small></span><input id="setting-danger" type="checkbox" ${values.danger ? 'checked' : ''}></label>
      <label class="check-line"><span><b>Tắt âm thanh</b><small>Hiệu ứng bom, nhặt đồ và kết quả.</small></span><input id="setting-muted" type="checkbox" ${values.muted ? 'checked' : ''}></label>
      <label class="volume-field">Âm lượng<input id="setting-volume" type="range" min="0" max="100" value="${Math.round(values.volume * 100)}"></label></div>
    <button class="button primary full" data-action="close-dialog">${icon('check')}Xong</button>`;
}

export function pause() {
  return `<div class="modal-heading"><div><span class="eyebrow">TAKE A BREATH</span><h2 id="pause-title">Đấu trường đang chờ.</h2></div></div><p class="modal-intro">Ván đấu đã tạm dừng. Không có nước đi nào được thực hiện khi bạn mở menu này.</p><div class="pause-actions"><button class="button primary full" data-action="resume-modal">${icon('play')}Tiếp tục</button><button class="button secondary full" data-action="settings">${icon('settings')}Cài đặt</button><button class="text-button" data-action="home">${icon('home')}Về menu</button></div>`;
}

export function training(models: Model[]) {
  return `<main class="training-view">${header('PHÒNG HUẤN LUYỆN', true)}<div class="game-heading"><div><div class="eyebrow">LEARNING LAB / C — D — E</div><h1>Chiến thuật được học từ trải nghiệm.</h1></div><span id="train-status" class="connection">Chưa bắt đầu</span></div>
    <div class="training-layout"><aside class="train-config"><span class="eyebrow">THIẾT LẬP HUẤN LUYỆN</span><form id="training-form">
      <label class="field">Giai đoạn (Phase)<select name="phase" id="train-phase"><option value="C">C · Đối đầu RuleBot</option><option value="D">D · Self-play</option><option value="E">E · Free-for-all</option></select></label>
      <div class="form-row"><label class="field">Số tập (episode)<input name="episodes" type="number" min="1" max="100000" step="1" value="100" required></label><label class="field">Số tác tử<select name="num_agents" id="train-agents" disabled><option value="2">2</option><option value="3">3</option><option value="4">4</option></select></label></div>
      <label class="check-line"><input type="checkbox" name="warm_start" checked>Warm-start từ checkpoint</label>
      <label class="field">Checkpoint khởi đầu<select name="model_id" id="train-model" required>${options(models)}</select></label>
      <details class="advanced"><summary>Tùy chọn phiên huấn luyện</summary><label class="field">Bước tối đa mỗi ván<input type="number" name="max_steps" min="1" max="500" step="1" value="500" required></label><small>Job mới chạy CPU, lưu trong game UI/runs. Không điều khiển các thí nghiệm đã mở bên ngoài.</small></details>
      <button type="submit" id="train-start" class="button primary full">${icon('train')}Bắt đầu huấn luyện</button><p class="form-error" id="train-error" role="alert"></p></form>
      <div class="train-buttons"><button class="button secondary" id="train-pause" data-action="train-pause" disabled>${icon('pause')}Tạm dừng</button><button class="button secondary" id="train-stop" data-action="train-stop" disabled>${icon('check')}Dừng & lưu</button></div><div class="train-ownership">${icon('alive')}<span>Chỉ job do giao diện tạo<br>mới có thể được điều khiển.</span></div></aside>
    <section class="train-monitor"><div class="metric-strip"><div><small>WR</small><b id="metric-wr">—</b></div><div><small>ActiveWR</small><b id="metric-active">—</b></div><div><small>Reward</small><b id="metric-reward">—</b></div><div><small>Loss</small><b id="metric-loss">—</b></div></div>
      <div class="training-progress"><span id="train-progress-label">Đang chờ job mới</span><b id="train-progress-count">0 / 0</b><progress id="train-progress" max="100" value="0"></progress></div>
      <div class="train-live"><div><div class="live-title"><span class="signal-dot"></span>QUAN SÁT MÔI TRƯỜNG<span id="train-frame-label">Chưa có ván đấu</span></div>${board()}<p class="dim monitor-note" id="train-empty">Bàn chơi thật sẽ hiện khi episode đầu tiên bắt đầu.</p></div><div class="charts"><div class="chart-box"><div class="eyebrow">REWARD / 100 TẬP GẦN NHẤT</div><div id="reward-chart" class="chart-empty">Chờ dữ liệu callback…</div></div><div class="chart-box"><div class="eyebrow">LOSS / 100 TẬP GẦN NHẤT</div><div id="loss-chart" class="chart-empty">Chờ dữ liệu callback…</div></div><p class="run-path" id="train-output">Mỗi job có thư mục lưu riêng.</p></div></div>
    </section></div></main>`;
}
