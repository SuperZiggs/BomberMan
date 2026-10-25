import './style.css';
import { createArena } from './arena';
import { InputController } from './input';
import { api, MatchConnection } from './transport';
import type { GameState, Job, Manifest, Model, Settings } from './types';
import * as views from './views';

const $ = <T extends HTMLElement = HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const app = $('#app');
const dialogs = ['setup-dialog', 'settings-dialog', 'pause-dialog'].map(id => $<HTMLDialogElement>('#' + id));
const defaults: Settings = {reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches, danger: false, volume: .35, muted: false};
let settings: Settings = {...defaults};
try { settings = {...defaults, ...JSON.parse(localStorage.getItem('foundry-settings') ?? '{}')}; } catch { /* A bad local preference never prevents startup. */ }
let view: 'menu' | 'match' | 'training' = 'menu';
let models: Model[] = [];
let manifest: Manifest | undefined;
let arena: ReturnType<typeof createArena> | undefined;
let connection: MatchConnection | undefined;
let sessionId: string | undefined;
let frame: GameState | undefined;
let input: InputController | undefined;
let job: Job | undefined;
let pollTimer: ReturnType<typeof setTimeout> | undefined;
let lastEvent = 0;
let soundUnlocked = false;
let returnFocus: HTMLElement | null = null;
const sounds = new Map<string, HTMLAudioElement>();
const pointerKeys = new Map<number, string>();

function text(selector: string, value: unknown) {
  const node = document.querySelector(selector);
  const next = String(value);
  if (node && node.textContent !== next) node.textContent = next;
}
function clearControls() { pointerKeys.clear(); input?.clear(); }
function error(message: string) {
  const toast = $('#toast');
  toast.innerHTML = `${views.icon('danger')}<span>${views.escape(message)}</span><button class="icon-button" aria-label="Đóng thông báo">${views.icon('close')}</button>`;
  toast.hidden = false;
  toast.querySelector('button')!.onclick = () => { toast.hidden = true; };
  for (const dialog of dialogs) if (dialog.open) text(`#${dialog.id} [data-modal-error]`, message);
  if (view === 'training') text('#train-error', message);
}
function busy(value: string | null) {
  $('#busy').hidden = value === null;
  if (value) text('#busy-text', value);
}
function saveSettings() {
  localStorage.setItem('foundry-settings', JSON.stringify(settings));
  document.body.dataset.motion = settings.reducedMotion ? 'reduced' : 'full';
  arena?.scene.refreshSettings();
}
function sound(name: string) {
  if (!soundUnlocked || settings.muted || !settings.volume || !['bomb', 'fuse', 'explosion', 'pickup', 'defeat', 'win', 'draw', 'click'].includes(name)) return;
  let audio = sounds.get(name);
  if (!audio) { audio = new Audio(`/assets/v2/audio/${name}.wav`); sounds.set(name, audio); }
  audio.volume = settings.volume;
  audio.currentTime = 0;
  void audio.play().catch(() => {});
}
function closeDialogs() {
  for (const dialog of dialogs) if (dialog.open) dialog.close();
  if (returnFocus?.isConnected && returnFocus.getClientRects().length) returnFocus.focus();
  else document.querySelector<HTMLElement>('.button.primary:not(:disabled), .brand')?.focus();
}
function openDialog(id: string, content: string) {
  returnFocus = document.activeElement as HTMLElement;
  for (const dialog of dialogs) if (dialog.open) dialog.close();
  const dialog = $<HTMLDialogElement>('#' + id);
  dialog.innerHTML = content;
  dialog.showModal();
}
function cleanupView() {
  clearTimeout(pollTimer);
  arena?.game.destroy(true);
  arena = undefined;
  for (const dialog of dialogs) if (dialog.open) dialog.close();
  $('#toast').hidden = true;
}
async function leaveSession() {
  clearControls(); input = undefined;
  connection?.close(); connection = undefined;
  if (sessionId) {
    const old = sessionId; sessionId = undefined;
    await api(`/sessions/${old}`, undefined, 'DELETE').catch(() => {});
  }
  frame = undefined;
}
async function home() {
  await leaveSession();
  cleanupView(); view = 'menu';
  app.innerHTML = views.menu();
  window.scrollTo(0, 0);
}
async function loadModels() {
  const result = await api<{models: Model[]}>('/models');
  models = result.models;
  if (!models.length) throw new Error('Không có checkpoint hợp lệ. Đặt mô hình best_/final_ trong thư mục checkpoints rồi thử lại.');
  return models;
}
async function setup(mode: 'play' | 'watch') {
  openDialog('setup-dialog', views.setup(mode, [], true));
  try {
    await loadModels();
    if ($<HTMLDialogElement>('#setup-dialog').open) $('#setup-dialog').innerHTML = views.setup(mode, models);
  } catch (exc) { error((exc as Error).message); }
}
async function getManifest() {
  if (!manifest) {
    const response = await fetch('/assets/v2/manifest.json');
    if (!response.ok) throw new Error('Không tải được danh mục asset v2. Hãy kiểm tra bộ build.');
    manifest = await response.json() as Manifest;
  }
  return manifest;
}
async function mountArena(initial?: GameState) {
  const element = $('#arena-canvas');
  const data = await getManifest();
  if (!element.isConnected) return;
  await document.fonts.ready;
  if (!element.isConnected) return;
  arena = createArena(element, data, () => settings, () => {
    const loading = element.parentElement?.querySelector<HTMLElement>('#canvas-loading');
    if (loading) loading.hidden = true;
    const canvas = element.querySelector('canvas');
    if (canvas) { canvas.tabIndex = 0; canvas.setAttribute('aria-label', 'Bàn đấu Bomberman'); }
  }, error);
  if (initial) arena.scene.apply(initial);
}
function connectionStatus(status: string) {
  text('#connection', status);
  const healthy = status === 'Đã kết nối';
  if (!healthy) clearControls();
  $('#connection')?.classList.toggle('online', healthy);
  const banner = document.querySelector<HTMLElement>('#connection-banner');
  if (banner) { banner.hidden = healthy; banner.textContent = status; }
}
async function createMatch(form: HTMLFormElement) {
  if (!form.reportValidity()) return;
  const data = new FormData(form);
  const submit = form.querySelector<HTMLButtonElement>('[type="submit"]')!;
  submit.disabled = true;
  submit.innerHTML = '<span class="spinner"></span>Đang nạp checkpoint…';
  busy('Đang nạp checkpoint và tạo ván đấu…');
  try {
    const config = {mode: data.get('mode') as 'play' | 'watch', model_id: String(data.get('model_id')),
      num_agents: Number(data.get('num_agents')), speed: Number(data.get('speed')), sudden_death: data.has('sudden_death')};
    await leaveSession();
    const result = await api<{session_id: string; snapshot: GameState}>('/sessions', config);
    cleanupView(); view = 'match'; sessionId = result.session_id; lastEvent = 0;
    const name = models.find(m => m.id === config.model_id)?.name ?? 'Checkpoint AI';
    app.innerHTML = views.match(config.mode, name);
    input = new InputController(message => connection?.send(message));
    connection = new MatchConnection(sessionId, showFrame, connectionStatus, error);
    showFrame(result.snapshot);
    await mountArena(result.snapshot);
    connection.connect();
    window.scrollTo(0, 0);
  } catch (exc) { error((exc as Error).message); }
  finally { busy(null); if (submit.isConnected) { submit.disabled = false; submit.innerHTML = `${views.icon('play')}Bắt đầu ván đấu${views.icon('arrow')}`; } }
}
function showFrame(state: GameState) {
  const previous = frame;
  if (previous && (previous.match_id !== state.match_id || (!previous.paused && state.paused))) clearControls();
  frame = state;
  if (view !== 'match') return;
  text('#tick', String(state.tick).padStart(3, '0'));
  text('#max-steps', state.max_steps);
  text('#alive', state.agents.filter(a => a.alive).length);
  text('#roster-count', String(state.agents.length).padStart(2, '0'));
  text('#play-status', state.done ? 'Kết thúc' : state.paused ? 'Tạm dừng' : 'Đang thi đấu');
  $('#play-status').classList.toggle('running', !state.paused && !state.done);
  const toggle = $<HTMLButtonElement>('#toggle-run');
  toggle.innerHTML = `${views.icon(state.done ? 'reset' : state.paused ? 'play' : 'pause')}${state.done ? 'Ván mới' : state.paused ? 'Tiếp tục' : 'Tạm dừng'}`;
  $<HTMLButtonElement>('#step-button').disabled = state.done || !state.paused;
  $<HTMLSelectElement>('#game-speed').value = String(state.speed);
  $('#agent-cards').innerHTML = state.agents.map(a => `<article class="agent-card ${views.colors[a.agent_id]} ${a.alive ? '' : 'eliminated'}"><img src="/assets/v2/characters/${views.colors[a.agent_id]}-avatar.png" alt="" width="48" height="48"><div class="agent-name"><span><b>A${a.agent_id}</b><small>${views.names[a.agent_id]}</small></span><strong>${views.escape(state.labels[a.agent_id])}</strong><span class="agent-state">${a.alive ? '<i></i>Còn sống' : 'Đã bị loại'}</span></div><div class="agent-inventory"><span title="Bom / sức chứa">${views.icon('bomb')}${a.bombs_placed}/${a.ammo}</span><span title="Tầm nổ">${views.icon('range')}${a.blast_range}</span><span title="Hạ gục">${a.kills}<small> K</small></span></div></article>`).join('');
  $('#agent-details').innerHTML = state.agents.map(a => `<div class="detail-row"><span>A${a.agent_id} · ${['Đứng', 'Lên', 'Xuống', 'Trái', 'Phải', 'Bom'][state.actions[a.agent_id]]}</span><span>${(state.rewards[a.agent_id] ?? 0).toFixed(2)} reward</span></div>`).join('');
  text('#perf-note', `Xử lý bước gần nhất: ${(state.tick_ms ?? 0).toFixed(1)} ms · ${state.grid.length}×${state.grid.length}`);
  const warning = $('#ring-warning');
  warning.hidden = state.ring.remaining === null;
  if (!warning.hidden) warning.innerHTML = `${views.icon('danger')}Vòng tiếp theo đóng sau ${state.ring.remaining} bước`;
  const result = $('#result-card');
  result.hidden = !state.done;
  if (state.done && state.result) result.innerHTML = `<span class="eyebrow">KẾT QUẢ VÁN ĐẤU</span><h2>${state.result.winner === null ? 'Không ai lùi bước.' : `A${state.result.winner} thắng trận!`}</h2><p>${views.escape(state.result.title)} · ${state.tick} bước</p><button class="button primary full" data-action="reset">${views.icon('reset')}Chơi lại</button>`;
  arena?.scene.apply(state);
  for (const event of state.events ?? []) {
    if (event.id <= lastEvent) continue;
    lastEvent = event.id;
    if (event.tick < state.tick - 2) continue;
    sound(event.kind === 'end' ? (state.result?.winner === null ? 'draw' : 'win') : event.kind);
  }
  if (state.error) error(state.error);
}
async function showSettings() {
  clearControls();
  if (connection) await connection.command('pause').catch(() => {});
  openDialog('settings-dialog', views.settings(settings));
}
async function showPause() {
  clearControls();
  if (connection) await connection.command('pause').catch(() => {});
  openDialog('pause-dialog', views.pause());
}
async function trainingView() {
  await leaveSession(); cleanupView(); view = 'training';
  busy('Đang mở phòng huấn luyện…');
  try {
    await loadModels();
    app.innerHTML = views.training(models);
    await mountArena();
    const saved = job?.job_id ?? localStorage.getItem('foundry-job');
    if (saved) {
      try { job = await api<Job>(`/training/${saved}`); showJob(job); pollJob(saved); }
      catch { localStorage.removeItem('foundry-job'); }
    }
  } catch (exc) { app.innerHTML = `${views.menu()}`; view = 'menu'; error((exc as Error).message); }
  finally { busy(null); window.scrollTo(0, 0); }
}
async function startTraining(form: HTMLFormElement) {
  if (!form.reportValidity()) return;
  const data = new FormData(form);
  const button = $<HTMLButtonElement>('#train-start');
  button.disabled = true;
  text('#train-error', '');
  try {
    const result = await api<{job_id: string}>('/training', {phase: data.get('phase'), episodes: Number(data.get('episodes')),
      num_agents: Number($<HTMLSelectElement>('#train-agents').value), warm_start: data.has('warm_start'),
      model_id: data.has('warm_start') ? String(data.get('model_id')) : null, max_steps: Number(data.get('max_steps'))});
    localStorage.setItem('foundry-job', result.job_id);
    job = await api<Job>(`/training/${result.job_id}`);
    showJob(job); pollJob(result.job_id);
  } catch (exc) { error((exc as Error).message); button.disabled = false; }
}
const terminal = (status: string) => ['completed', 'cancelled', 'error'].includes(status);
function plot(values: number[], color: string) {
  if (!values.length) return '<span class="dim">Chờ dữ liệu callback…</span>';
  const min = Math.min(...values), max = Math.max(...values), delta = max - min || Math.max(Math.abs(max) * .1, 1);
  const points = values.map((v, i) => `${12 + i * 276 / Math.max(1, values.length - 1)},${86 - (v - min) * 64 / delta}`).join(' ');
  return `<svg viewBox="0 0 300 108" role="img" aria-label="Biểu đồ từ metric huấn luyện thực tế"><path d="M12 24h276M12 56h276M12 88h276" stroke="#2b3d48" stroke-width="1"/><polyline fill="none" stroke="${color}" stroke-width="2.5" points="${points}"/><text x="12" y="105" fill="#879b9f" font-size="9">${min.toFixed(2)}</text><text x="288" y="105" text-anchor="end" fill="#879b9f" font-size="9">${max.toFixed(2)}</text></svg>`;
}
function showJob(value: Job) {
  job = value;
  if (view !== 'training') return;
  const active = !terminal(value.status);
  const labels: Record<string, string> = {initializing: 'Đang khởi tạo', running: 'Đang huấn luyện', paused: 'Tạm dừng', stopping: 'Đang dừng và lưu…', completed: 'Hoàn tất', cancelled: 'Đã dừng và lưu', error: 'Lỗi huấn luyện'};
  text('#train-status', labels[value.status] ?? value.status);
  text('#train-progress-label', `${labels[value.status] ?? value.status} · ${Math.round(value.elapsed)} giây`);
  text('#train-progress-count', `${value.episode} / ${value.episodes}`);
  $<HTMLProgressElement>('#train-progress').max = value.episodes;
  $<HTMLProgressElement>('#train-progress').value = value.episode;
  $<HTMLButtonElement>('#train-start').disabled = active;
  $<HTMLButtonElement>('#train-pause').disabled = !active || value.status === 'stopping';
  $('#train-pause').innerHTML = `${views.icon(value.status === 'paused' ? 'play' : 'pause')}${value.status === 'paused' ? 'Tiếp tục' : 'Tạm dừng'}`;
  $<HTMLButtonElement>('#train-stop').disabled = !active || value.status === 'stopping';
  for (const element of Array.from($('#training-form').querySelectorAll<HTMLInputElement | HTMLSelectElement>('input,select'))) element.disabled = active;
  if (!active) syncPhase();
  const metrics = value.metrics.slice(-50);
  if (metrics.length) {
    const mean = (key: 'win' | 'active_win' | 'reward' | 'loss') => metrics.reduce((sum, row) => sum + row[key], 0) / metrics.length;
    text('#metric-wr', (mean('win') * 100).toFixed(1) + '%');
    text('#metric-active', (mean('active_win') * 100).toFixed(1) + '%');
    text('#metric-reward', mean('reward').toFixed(2));
    text('#metric-loss', mean('loss').toFixed(3));
    $('#reward-chart').innerHTML = plot(value.metrics.slice(-100).map(row => row.reward), '#a6efc8');
    $('#loss-chart').innerHTML = plot(value.metrics.slice(-100).map(row => row.loss), '#f4c95d');
  }
  if (value.frame) {
    arena?.scene.apply(value.frame);
    $('#train-empty').hidden = true;
    text('#train-frame-label', `Bước ${value.frame.tick} / ${value.frame.max_steps}`);
  }
  text('#train-output', `Đã lưu: ${value.output_dir}`);
  if (value.error) error(value.error);
}
function pollJob(id: string) {
  clearTimeout(pollTimer);
  if (view !== 'training') return;
  pollTimer = setTimeout(async () => {
    try { const value = await api<Job>(`/training/${id}`); showJob(value); if (!terminal(value.status)) pollJob(id); }
    catch (exc) { error((exc as Error).message); }
  }, 350);
}
function syncPhase() {
  const phase = $<HTMLSelectElement>('#train-phase'), agents = $<HTMLSelectElement>('#train-agents');
  if (!phase || !agents) return;
  agents.disabled = phase.value !== 'E';
  if (phase.value !== 'E') agents.value = '2';
}

async function action(name: string) {
  soundUnlocked = true;
  if (name !== 'bomb') sound('click');
  switch (name) {
    case 'home': await home(); break;
    case 'setup-play': await setup('play'); break;
    case 'setup-watch': await setup('watch'); break;
    case 'settings': await showSettings(); break;
    case 'close-dialog': closeDialogs(); break;
    case 'training': await trainingView(); break;
    case 'toggle':
      if (frame?.done || !frame?.paused) clearControls();
      await connection?.command(frame?.done ? 'reset' : frame?.paused ? 'resume' : 'pause'); break;
    case 'step': await connection?.command('step'); break;
    case 'reset': clearControls(); closeDialogs(); await connection?.command('reset'); break;
    case 'resume-modal': closeDialogs(); await connection?.command('resume'); break;
    case 'bomb': input?.bomb(); break;
    case 'train-pause': if (job) { showJob(await api<Job>(`/training/${job.job_id}/${job.status === 'paused' ? 'resume' : 'pause'}`, {})); pollJob(job.job_id); } break;
    case 'train-stop': if (job) { showJob(await api<Job>(`/training/${job.job_id}/stop`, {})); pollJob(job.job_id); } break;
  }
}
document.addEventListener('click', event => {
  const button = (event.target as Element).closest<HTMLButtonElement>('button[data-action]');
  if (button && !button.disabled) void action(button.dataset.action!).catch(exc => error(exc.message));
});
document.addEventListener('submit', event => {
  const form = event.target as HTMLFormElement;
  if (form.id === 'match-form' || form.id === 'training-form') {
    event.preventDefault(); soundUnlocked = true;
    void (form.id === 'match-form' ? createMatch(form) : startTraining(form));
  }
});
document.addEventListener('change', event => {
  const target = event.target as HTMLInputElement;
  if (target.id === 'game-speed') void connection?.command('set_speed', {speed: Number(target.value)}).catch(exc => error(exc.message));
  if (target.id === 'train-phase') syncPhase();
  if (target.name === 'warm_start' && view === 'training') $<HTMLSelectElement>('#train-model').disabled = !target.checked;
  if (target.id.startsWith('setting-')) {
    if (target.id === 'setting-motion') settings.reducedMotion = target.checked;
    if (target.id === 'setting-danger') settings.danger = target.checked;
    if (target.id === 'setting-muted') settings.muted = target.checked;
    if (target.id === 'setting-volume') settings.volume = Number(target.value) / 100;
    saveSettings();
  }
});
document.addEventListener('input', event => {
  const target = event.target as HTMLInputElement;
  if (target.id === 'setting-volume') { settings.volume = Number(target.value) / 100; saveSettings(); }
});
for (const dialog of dialogs) dialog.addEventListener('cancel', event => { event.preventDefault(); closeDialogs(); });
function gameKeys(event: KeyboardEvent, down: boolean) {
  if (down && event.repeat) return;
  if (event.ctrlKey || event.metaKey || event.altKey || (event.target as Element)?.closest('input,select,textarea')) return;
  if (event.key === 'Escape' && down && !dialogs.some(d => d.open) && view === 'match') {
    event.preventDefault(); void showPause(); return;
  }
  if (view !== 'match' || frame?.mode !== 'play' || dialogs.some(d => d.open)) return;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (!['w', 'a', 's', 'd', 'e', ' ', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(key)) return;
  event.preventDefault(); soundUnlocked = true;
  if (down) input?.press(key); else input?.release(key);
}
window.addEventListener('keydown', event => gameKeys(event, true));
window.addEventListener('keyup', event => gameKeys(event, false));
document.addEventListener('pointerdown', event => {
  const button = (event.target as Element).closest<HTMLButtonElement>('[data-direction]');
  if (button) {
    event.preventDefault(); soundUnlocked = true;
    pointerKeys.set(event.pointerId, button.dataset.direction!); input?.press(button.dataset.direction!);
    if (event.isTrusted) button.setPointerCapture(event.pointerId);
  }
  const canvas = (event.target as Element).closest<HTMLCanvasElement>('canvas');
  canvas?.focus();
});
for (const kind of ['pointerup', 'pointercancel'] as const) document.addEventListener(kind, event => {
  const key = pointerKeys.get(event.pointerId);
  if (key) { input?.release(key); pointerKeys.delete(event.pointerId); }
});
function blur() {
  clearControls();
  if (connection) void connection.command('pause').catch(() => {});
}
window.addEventListener('blur', blur);
document.addEventListener('visibilitychange', () => { if (document.hidden) blur(); });
window.addEventListener('offline', () => { clearControls(); connection?.offline(); });
window.addEventListener('online', () => connection?.connect());
window.addEventListener('beforeunload', () => {
  connection?.close();
  if (sessionId) void fetch(`/api/sessions/${sessionId}`, {method: 'DELETE', keepalive: true}).catch(() => {});
});
saveSettings();
app.innerHTML = views.menu();
