const keys: Record<string, string> = {
  w: 'up', arrowup: 'up', s: 'down', arrowdown: 'down',
  a: 'left', arrowleft: 'left', d: 'right', arrowright: 'right',
};
export class InputController {
  private held: string[] = [];
  private seq = 0;
  private send: (message: {type: string; seq: number; direction: string; bomb?: boolean}) => void;
  constructor(send: (message: {type: string; seq: number; direction: string; bomb?: boolean}) => void) { this.send = send; }
  private emit(bomb = false) {
    const last = [...this.held].reverse().find(key => keys[key]);
    this.send({type: 'input', seq: ++this.seq, direction: last ? keys[last] : 'stop', bomb});
  }
  press(raw: string) {
    const key = raw.toLowerCase();
    if (this.held.includes(key) || !(key in keys || key === ' ' || key === 'e')) return;
    this.held.push(key);
    this.emit(key === ' ' || key === 'e');
  }
  release(raw: string) {
    const key = raw.toLowerCase();
    if (!this.held.includes(key)) return;
    this.held = this.held.filter(value => value !== key);
    this.emit();
  }
  bomb() { this.emit(true); }
  clear() { this.held = []; this.emit(); }
}
