export interface Agent {
  agent_id: number; row: number; col: number; alive: boolean; ammo: number;
  bombs_placed: number; blast_range: number; kills: number;
}
export interface Bomb { row: number; col: number; owner: number; timer: number; blast_range: number }
export interface GameEvent { id: number; kind: string; tick: number; row?: number; col?: number; agent?: number }
export interface GameState {
  session_id: string; match_id: string; tick: number; step: number; max_steps: number;
  grid: number[][]; agents: Agent[]; bombs: Bomb[]; flames: number[][];
  powerups: [number[], string][]; danger: number[][];
  ring: { enabled: boolean; remaining: number | null; cells: number[][] };
  actions: number[]; rewards: number[]; labels: string[]; done: boolean;
  result: { winner: number | null; outcome: string; title: string } | null;
  paused: boolean; running?: boolean; speed: number; mode: 'play' | 'watch' | 'training';
  revision?: number; events?: GameEvent[]; tick_ms?: number; error?: string | null;
}
export interface Model { id: string; name: string; source: string }
export interface Settings { reducedMotion: boolean; danger: boolean; volume: number; muted: boolean }
export interface Manifest {
  version: number; frameSize: number;
  characters: {id: string; path: string; avatar: string; frames: number}[];
  terrain: {path: string; frames: number}; effects: {path: string; frames: number}; files: string[];
}
export interface Metric { episode: number; reward: number; loss: number; win: number; active_win: number; kills: number; steps: number }
export interface Job {
  job_id: string; status: string; episode: number; episodes: number; metrics: Metric[];
  frame: GameState | null; error: string | null; output_dir: string; elapsed: number;
}
