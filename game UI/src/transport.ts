import type { GameState } from './types';

export async function api<T>(path: string, body?: unknown, method?: string): Promise<T> {
  const response = await fetch('/api' + path, {
    method: method ?? (body === undefined ? 'GET' : 'POST'),
    headers: { 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : {body: JSON.stringify(body)}),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ?
      detail.map((error: {msg: string}) => error.msg).join(' · ') : 'Không thể thực hiện thao tác. Hãy thử lại.');
  }
  return data as T;
}

export class MatchConnection {
  private socket: WebSocket | null = null;
  private retry = 0;
  private timer?: ReturnType<typeof setTimeout>;
  private heartbeat?: ReturnType<typeof setInterval>;
  private disposed = false;
  private current?: GameState;
  private pending = new Map<string, {resolve: () => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout>}>();

  constructor(public readonly id: string, private onState: (state: GameState) => void,
    private onStatus: (status: string) => void, private onError: (message: string) => void) {}

  connect() {
    if (this.disposed || !navigator.onLine) { this.onStatus('Ngoại tuyến · Ván đấu đã tạm dừng'); return; }
    clearTimeout(this.timer);
    this.onStatus(this.retry ? 'Đang kết nối lại…' : 'Đang kết nối…');
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${scheme}//${location.host}/api/sessions/${this.id}/ws`);
    this.socket = socket;
    socket.onopen = () => {
      this.retry = 0;
      clearInterval(this.heartbeat);
      this.heartbeat = setInterval(() => this.send({type: 'ping'}), 650);
    };
    socket.onmessage = event => {
      let message;
      try { message = JSON.parse(event.data); } catch { this.onError('Dữ liệu kết nối không hợp lệ.'); return; }
      if (message.type === 'hello') {
        if (message.protocol_version !== 1) { this.onError('Phiên bản giao thức không tương thích.'); this.close(); return; }
        this.onStatus('Đã kết nối');
      } else if (message.type === 'snapshot') {
        const state = message.state as GameState;
        if (this.current && state.match_id === this.current.match_id &&
          (state.revision ?? 0) < (this.current.revision ?? 0)) return;
        this.current = state;
        this.onState(state);
      } else if (message.type === 'ack') {
        const waiting = this.pending.get(message.command_id);
        if (waiting) { clearTimeout(waiting.timer); waiting.resolve(); this.pending.delete(message.command_id); }
      } else if (message.type === 'error') this.onError(message.message);
    };
    socket.onerror = () => this.onStatus('Kết nối lỗi · Đang kết nối lại…');
    socket.onclose = () => {
      if (this.socket !== socket) return;
      clearInterval(this.heartbeat);
      for (const value of this.pending.values()) { clearTimeout(value.timer); value.reject(new Error('Mất kết nối. Ván đấu đã tạm dừng.')); }
      this.pending.clear();
      if (!this.disposed) {
        this.onStatus('Mất kết nối · Đang kết nối lại…');
        this.timer = setTimeout(() => this.connect(), Math.min(3000, 300 * ++this.retry));
      }
    };
  }
  send(message: unknown) {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(message));
  }
  command(command: string, fields: Record<string, unknown> = {}) {
    if (this.socket?.readyState !== WebSocket.OPEN) return Promise.reject(new Error('Chờ kết nối lại trước khi điều khiển ván đấu.'));
    const command_id = crypto.randomUUID();
    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(command_id); reject(new Error('Python đang xử lý lâu hơn dự kiến. Hãy chờ trạng thái mới.')); }, 15000);
      this.pending.set(command_id, {resolve, reject, timer});
      this.send({type: 'command', command, command_id, ...fields});
    });
  }
  offline() {
    clearTimeout(this.timer);
    clearInterval(this.heartbeat);
    this.socket?.close();
    this.onStatus('Ngoại tuyến · Ván đấu đã tạm dừng');
  }
  close() {
    this.disposed = true;
    clearTimeout(this.timer);
    clearInterval(this.heartbeat);
    for (const value of this.pending.values()) { clearTimeout(value.timer); value.resolve(); }
    this.pending.clear();
    this.socket?.close();
  }
}
