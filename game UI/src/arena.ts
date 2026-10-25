import Phaser from 'phaser';
import type { GameState, Manifest, Settings } from './types';

const CELL = 32;
const colors = ['mint', 'blue', 'coral', 'gold'];
const facing = ['down', 'up', 'down', 'left', 'right'];
const rows: Record<string, number> = { down: 0, up: 6, left: 12, right: 18 };
type Actor = { sprite: Phaser.GameObjects.Sprite; tag: Phaser.GameObjects.Text; row: number; col: number;
  direction: string; walkUntil: number; placeUntil: number; dead: boolean };

export class ArenaScene extends Phaser.Scene {
  private manifest: Manifest;
  private getSettings: () => Settings;
  private onReady: () => void;
  private onError: (message: string) => void;
  private loaded = false;
  private latest: GameState | null = null;
  private terrain!: Phaser.GameObjects.Container;
  private overlays!: Phaser.GameObjects.Graphics;
  private layout = '';
  private match = '';
  private actors = new Map<number, Actor>();
  private objects = new Map<string, Phaser.GameObjects.Sprite>();
  private timers = new Map<string, Phaser.GameObjects.Text>();
  private eventId = 0;

  constructor(manifest: Manifest, settings: () => Settings, ready: () => void, error: (message: string) => void) {
    super('arena');
    this.manifest = manifest; this.getSettings = settings; this.onReady = ready; this.onError = error;
  }
  preload() {
    for (const robot of this.manifest.characters) this.load.spritesheet(robot.id, robot.path, {frameWidth: CELL, frameHeight: CELL});
    this.load.spritesheet('terrain', this.manifest.terrain.path, {frameWidth: CELL, frameHeight: CELL});
    this.load.spritesheet('fx', this.manifest.effects.path, {frameWidth: CELL, frameHeight: CELL});
    this.load.on('loaderror', (file: {src: string}) => this.onError(`Không tải được asset: ${file.src}`));
  }
  create() {
    this.terrain = this.add.container().setDepth(0);
    this.overlays = this.add.graphics().setDepth(1);
    for (const color of colors) {
      for (const [dir, start] of Object.entries(rows)) {
        this.anims.create({key: `${color}-idle-${dir}`, frames: this.anims.generateFrameNumbers(color, {start, end: start + 1}), frameRate: 3, repeat: -1});
        this.anims.create({key: `${color}-walk-${dir}`, frames: this.anims.generateFrameNumbers(color, {start: start + 2, end: start + 5}), frameRate: 12, repeat: -1});
        const index = dir === 'down' ? 24 : dir === 'up' ? 26 : dir === 'left' ? 28 : 30;
        this.anims.create({key: `${color}-place-${dir}`, frames: this.anims.generateFrameNumbers(color, {start: index, end: index + 1}), frameRate: 10, repeat: -1});
      }
      this.anims.create({key: `${color}-defeat`, frames: this.anims.generateFrameNumbers(color, {start: 32, end: 35}), frameRate: 10, repeat: 0});
    }
    for (let i = 0; i < 17; i++) this.anims.create({key: `fx-${i}`, frames: this.anims.generateFrameNumbers('fx', {start: i * 4, end: i * 4 + 3}), frameRate: 13, repeat: -1});
    this.loaded = true;
    this.onReady();
    if (this.latest) this.renderState(this.latest);
  }
  apply(state: GameState) {
    const previous = this.latest;
    this.latest = state;
    if (this.loaded && (!previous || state.tick !== previous.tick || state.match_id !== previous.match_id)) this.renderState(state);
    else if (this.loaded) this.drawOverlays(state);
  }
  refreshSettings() { if (this.loaded && this.latest) { this.drawOverlays(this.latest); this.renderObjects(this.latest); } }
  private resetWorld() {
    for (const actor of this.actors.values()) { actor.sprite.destroy(); actor.tag.destroy(); }
    for (const object of this.objects.values()) object.destroy();
    for (const text of this.timers.values()) text.destroy();
    this.actors.clear(); this.objects.clear(); this.timers.clear(); this.layout = ''; this.eventId = 0;
    this.tweens.killAll();
  }
  private renderState(state: GameState) {
    if (this.match !== state.match_id) { this.resetWorld(); this.match = state.match_id; }
    const boardSize = state.grid.length * CELL;
    if (this.scale.width !== boardSize) this.scale.setGameSize(boardSize, boardSize);
    const signature = JSON.stringify(state.grid);
    if (signature !== this.layout) {
      this.terrain.removeAll(true);
      state.grid.forEach((row, r) => row.forEach((tile, c) => {
        this.terrain.add(this.add.image(c * CELL + 16, r * CELL + 16, 'terrain', (r * 3 + c) % 4));
        if (tile) this.terrain.add(this.add.image(c * CELL + 16, r * CELL + 16, 'terrain', tile === 1 ? 4 + (r + c) % 3 : 7 + (r + c) % 2));
      }));
      this.layout = signature;
    }
    this.drawOverlays(state);
    this.renderObjects(state);
    for (const agent of state.agents) {
      const id = agent.agent_id;
      let actor = this.actors.get(id);
      if (!actor && agent.alive) {
        const sprite = this.add.sprite(agent.col * CELL + 16, agent.row * CELL + 16, colors[id], 0).setDepth(5);
        const tag = this.add.text(sprite.x - 10, sprite.y - 16, `A${id}`, {
          fontFamily: 'Be Vietnam Pro', fontSize: '8px', fontStyle: 'bold', color: '#f5f5e9',
          backgroundColor: '#101820', padding: {x: 2, y: 0},
        }).setDepth(15);
        actor = {sprite, tag, row: agent.row, col: agent.col, direction: 'down', walkUntil: 0, placeUntil: 0, dead: false};
        this.actors.set(id, actor);
      }
      if (!actor) continue;
      if (!agent.alive && !actor.dead) {
        actor.dead = true;
        actor.tag.setVisible(false);
        this.tweens.killTweensOf(actor.sprite);
        actor.sprite.play(`${colors[id]}-defeat`);
        const dead = actor;
        this.time.delayedCall(400, () => { dead.sprite.setVisible(false); });
        continue;
      }
      if (actor.dead) continue;
      const action = state.actions[id];
      if (action >= 1 && action <= 4) actor.direction = facing[action];
      const x = agent.col * CELL + 16, y = agent.row * CELL + 16;
      if (actor.row !== agent.row || actor.col !== agent.col) {
        actor.walkUntil = this.time.now + 180;
        this.tweens.killTweensOf(actor.sprite);
        this.tweens.killTweensOf(actor.tag);
        const duration = this.getSettings().reducedMotion ? 0 : Math.min(95, 700 / (state.speed || 12));
        this.tweens.add({targets: actor.sprite, x, y, duration});
        this.tweens.add({targets: actor.tag, x: x - 10, y: y - 16, duration});
      }
      if (action === 5) actor.placeUntil = this.time.now + 160;
      actor.row = agent.row; actor.col = agent.col;
    }
    for (const event of state.events ?? []) {
      if (event.id <= this.eventId) continue;
      this.eventId = event.id;
      if (event.tick < state.tick - 2 || event.row === undefined || event.col === undefined) continue;
      const clip = event.kind === 'crate' ? 13 : event.kind === 'pickup' ? 15 : event.kind === 'defeat' ? 14 : -1;
      if (clip >= 0) {
        const fx = this.add.sprite(event.col * CELL + 16, event.row * CELL + 16, 'fx', clip * 4).setDepth(12);
        if (!this.getSettings().reducedMotion) fx.play(`fx-${clip}`);
        this.time.delayedCall(260, () => fx.destroy());
      }
    }
  }
  private object(key: string, x: number, y: number, clip: number, depth: number) {
    let sprite = this.objects.get(key);
    if (!sprite) { sprite = this.add.sprite(x, y, 'fx', clip * 4).setDepth(depth); this.objects.set(key, sprite); }
    sprite.setPosition(x, y);
    if (this.getSettings().reducedMotion) { sprite.anims.stop(); sprite.setFrame(clip * 4); }
    else sprite.play(`fx-${clip}`, true);
    return sprite;
  }
  private renderObjects(state: GameState) {
    const active = new Set<string>();
    for (const [pos, kind] of state.powerups) {
      const key = `item-${pos}`; active.add(key);
      this.object(key, pos[1] * CELL + 16, pos[0] * CELL + 16, kind === 'ammo' ? 4 : 5, 3);
    }
    const flame = new Set(state.flames.map(pos => pos.join(',')));
    for (const [r, c] of state.flames) {
      const n = flame.has(`${r - 1},${c}`), s = flame.has(`${r + 1},${c}`);
      const e = flame.has(`${r},${c + 1}`), w = flame.has(`${r},${c - 1}`);
      const degree = Number(n) + Number(s) + Number(e) + Number(w);
      const clip = degree >= 3 || degree === 0 ? 6 : e && w ? 7 : n && s ? 8 : s ? 9 : w ? 10 : n ? 11 : 12;
      const key = `flame-${r},${c}`; active.add(key);
      this.object(key, c * CELL + 16, r * CELL + 16, clip, 9);
    }
    for (const b of state.bombs) {
      const key = `bomb-${b.row},${b.col},${b.owner}`; active.add(key);
      const occupied = state.agents.some(a => a.alive && a.row === b.row && a.col === b.col);
      const x = b.col * CELL + (occupied ? 24 : 16), y = b.row * CELL + (occupied ? 24 : 16);
      const sprite = this.object(key, x, y, b.owner, occupied ? 11 : 4);
      sprite.setScale(occupied ? .56 : .88);
      let timer = this.timers.get(key);
      if (!timer) {
        timer = this.add.text(x, y, '', {fontFamily: 'Be Vietnam Pro', fontSize: '8px', fontStyle: 'bold',
          color: '#fff5db', backgroundColor: '#101820', padding: {x: 2, y: 0}}).setOrigin(.5).setDepth(16);
        this.timers.set(key, timer);
      }
      timer.setText(`${b.timer}`).setPosition(x, y + 3);
    }
    for (const [key, sprite] of this.objects) if (!active.has(key)) { sprite.destroy(); this.objects.delete(key); }
    for (const [key, text] of this.timers) if (!active.has(key)) { text.destroy(); this.timers.delete(key); }
  }
  private drawOverlays(state: GameState) {
    this.overlays.clear();
    if (this.getSettings().danger) {
      this.overlays.fillStyle(0xf4c95d, .25);
      state.danger?.forEach((row, r) => row.forEach((value, c) => {
        if (value && state.grid[r][c] === 0) this.overlays.fillRect(c * CELL + 2, r * CELL + 2, 28, 28);
      }));
    }
    this.overlays.lineStyle(2, 0xf4c95d, .9);
    for (const [r, c] of state.ring?.cells ?? []) this.overlays.strokeRect(c * CELL + 2, r * CELL + 2, 28, 28);
  }
  update(time: number) {
    const reduced = this.getSettings().reducedMotion;
    for (const [id, actor] of this.actors) {
      if (actor.dead) continue;
      if (reduced) { actor.sprite.anims.stop(); actor.sprite.setFrame(rows[actor.direction]); }
      else {
        const mode = time < actor.placeUntil ? 'place' : time < actor.walkUntil ? 'walk' : 'idle';
        actor.sprite.play(`${colors[id]}-${mode}-${actor.direction}`, true);
      }
    }
  }
}

export function createArena(parent: HTMLElement, manifest: Manifest, settings: () => Settings,
  ready: () => void, error: (message: string) => void) {
  const scene = new ArenaScene(manifest, settings, ready, error);
  const game = new Phaser.Game({type: Phaser.AUTO, parent, width: 352, height: 352,
    backgroundColor: '#172732', pixelArt: true, roundPixels: true, antialias: false,
    scale: {mode: Phaser.Scale.FIT, autoCenter: Phaser.Scale.CENTER_BOTH},
    scene: [scene], audio: {noAudio: true}, banner: false, fps: {target: 60},
  });
  return {game, scene};
}
