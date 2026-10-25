# Frozen requirements — Bomberman web UI, 2026-10-22

The user approved implementing the web UI plan and specifically requires **all new code, tests, generated assets, dependencies, build output, and game job outputs inside `E:\Bomberman\game UI`**. Existing root documentation may receive append-only task records. Do not modify the existing engine, trainers, checkpoints, or existing experiment runs.

## User-visible requirements

1. A runnable browser game using Phaser, TypeScript and Vite, with HTML/CSS menus and HUD, backed by the existing Python engine and learned policy. Three working modes: human play, AI spectating, and live training. Default art: the plan's original pixel-art robot foundry, Vietnamese UI, all new runtime visual assets and font.
2. Human play supports 3/4 agents, spectating supports 2/3/4 agents; ordinary matches use an 11×11 board. Rendering also supports 9×9 frames produced by training. WASD/arrows move; Space/E places a bomb; a short movement tap is retained until the next tick, and movement does not replace a queued bomb. Pause, resume, reset, single-step and speed 1–60 are supported. Single-step while paused advances exactly one engine tick.
3. Python owns all game rules, policy inference, timers, rewards, terminal results and RNG. Web UI reflects authoritative snapshots and never steps the simulation locally. Snapshot data is copied and JSON serializable. Bomb timers are labelled in ticks, not falsely presented as real seconds.
4. Pause, blur and disconnect clear held controls. Disconnected sessions pause automatically; reconnect sends current full state. Duplicate command IDs and old input sequences must not execute twice. Invalid messages produce recoverable errors. Unknown protocol version must not silently run a game.
5. New graphics: four identifiable robot characters with animation; floor/rigid wall/destructible crate, bombs by owner, ammo/range pickups, explosion/smoke/crate/pickup/ring FX; menu background, logo/UI/icons and Vietnamese-capable self-hosted font. All asset paths resolve within `game UI`; no legacy PNG or Emulogic font is used by the web build. Distinguish slots with numbers and shapes as well as color. Controls/HUD do not obscure board cells; desktop/mobile, touch pad, focus indication, settings and reduced motion are required.
6. Checkpoint catalog uses opaque IDs mapped to existing allowed files; arbitrary filesystem paths are never accepted as model IDs. Loading failure is readable and recoverable. Existing GUI's default checkpoint is acceptable; no new claim about best model quality. Real-model gameplay must be demonstrated, not just mock simulation.
7. Training supports phase C/D with 2 agents, E with 2/3/4; episode count validation, optional warm start, progress and actual metrics including WR/ActiveWR, reward/loss, latest observed board, pause/resume and stop-and-save. Only a UI-created job may be controlled. Job artifacts are created under `game UI/runs/`; no automatic attachment to existing experiment jobs or long validation/evaluation. A bounded short real smoke job is allowed for verification, max_steps ≤ 8 and episodes ≤ 2. Do not start long training.
8. Menu, setup, gameplay/watch HUD, pause/settings, results, training, loading, connection and error states work. A local Windows launch command and documented install/build/test commands must be verified. No Internet publication.

## Public contract v1 (acceptance boundary)

Python package import: `bridge`, resolved with `game UI` on PYTHONPATH. ASGI entry point: `bridge.app:app`.

- `GET /api/health`: JSON `status: "ok"`, `protocol_version: 1`.
- `GET /api/models`: JSON `models: [...]`, each containing `id` and `name`; clients submit the ID, never a filesystem path.
- `POST /api/sessions`: config `{mode: "play"|"watch", model_id: string, num_agents: int, speed?: int, seed?: int, max_steps?: int, sudden_death?: bool}`. Returns `session_id` and initial `snapshot` (state). Missing/invalid model or unsupported settings fail with a 4xx response and do not start a broken session. Session starts paused.
- `GET /api/sessions/{id}`: session metadata and authoritative `snapshot`. Unknown ID returns 404.
- `DELETE /api/sessions/{id}`: closes this local session.
- `WS /api/sessions/{id}/ws`: sends `{type:"hello", protocol_version:1, session_id}` followed by `{type:"snapshot", state:...}`. Commands `{type:"command", command_id:string, command:"step"|"pause"|"resume"|"reset"|"set_speed", speed?:number}` receive `{type:"ack", command_id,...}` and state updates. Input `{type:"input",seq:int,direction:"up"|"down"|"left"|"right"|"stop",bomb?:bool}`. Heartbeat `{type:"ping"}` → `{type:"pong"}`. Errors `{type:"error",code,message}`; one bad message does not kill a valid session. Reset yields a new match ID, tick 0, paused.
- State contains `session_id`, `match_id`, `tick`/`step`, `grid`, `agents`, `bombs`, `flames`, `powerups`, `labels`, `actions`, `rewards`, `max_steps`, `done`, `result`; danger/ring fields may be added. Agent fields reflect existing dataclasses. Snapshots identify paused/running state. A terminal state never advances further.
- Public snapshot adapter `bridge.snapshots.capture_state(env, labels, actions=None, rewards=None)` mirrors engine data as copied JSON values and includes terminal result. It must not mutate or reseed the engine. Presentation state may include extra fields.
- `POST /api/training`: `{phase,episodes,num_agents,warm_start?:bool,model_id?:string,max_steps?:int}` → `job_id`; invalid phases/counts/episodes/model combinations fail with 4xx. C/D require 2 agents. Only one active UI training job at a time; competing start returns 409.
- `GET /api/training/{id}`: `{job_id,status,episode,episodes,metrics,frame,error,output_dir}`; unknown job returns 404. `metrics` contains actual episode callback data (may be empty while initializing).
- `POST /api/training/{id}/{pause|resume|stop}` controls that job. Unknown job/action fails with 4xx. Stop saves completed work and reports eventual completion/cancellation/error.
- Loopback server by default. Mutating HTTP requests and WebSocket handshakes from explicitly foreign browser origins are rejected; allow same local app/dev origins and nonbrowser tests without Origin.

## Acceptance evidence

Use independent tests at the highest practical level: Python unit/HTTP/WS contract tests against the real engine; focused browser journey for menu → setup → real match → pause/step/reset, settings, training validation and responsive layout. Verify source/build asset references and include gameplay screenshots; static text checks alone are insufficient. Validate bridge-vs-engine action tape equivalence and copied frames. CPU real-checkpoint smoke and a bounded tiny training job are permitted; do not reuse held-out RL seeds or run quality benchmarks.

Test author receives this requirements packet and existing source/test conventions only. Produce a traceability matrix, runnable tests and a recorded valid RED baseline before production code. Do not infer intended algorithms from the design plan. Any technically invalid test needs an author/reviewer-supported correction with a preserved reason, never an implementer's relaxation.
