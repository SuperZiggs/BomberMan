# Fresh whole-feature review — 2026-10-22

Assessment: **changes required before completion**. No Critical finding. Three Important findings were reproduced with bounded, isolated checks; one Minor client-state finding follows directly from the input/event paths. Production source and acceptance tests were not edited. The running application and existing experiments were not stopped or modified.

## Important

### R1 — [P1] Release the session lock between over-budget ticks

Location: `E:\Bomberman\game UI\bridge\sessions.py:169`, especially lines 179–189.

`_loop` holds `self.condition` around the entire loop. Its only normal release while running is `condition.wait` when `delay > 0`. If inference plus engine work costs more than `1 / speed`, line 189 sets `due` to the current time, the next iteration has no positive delay, and another tick starts without releasing the lock. `state`, `input`, `pause`, reset and close all need that same lock. Moving inference to a thread therefore does not preserve control/snapshot responsiveness under load.

Reproduction: instantiate a real `GameSession` with a policy stub that waits 90 ms and returns action 0, speed 60 and max_steps 5; resume, wait until policy inference begins, then call `pause`. Pause returned after **453 ms at tick 5 with terminal=True**, after the whole match, rather than after the current tick. This used the real engine and no training. Longer matches can freeze snapshots and controls for many seconds; disconnect cleanup also waits for the lock.

Required correction: preserve one serialized engine owner, but provide a real lock-release/control opportunity after every completed tick, including when behind schedule. Add a regression that requests pause/state while one slow tick is in flight and verifies it does not wait for all remaining ticks.

### R2 — [P1] Include training startup in the worker error boundary

Location: `E:\Bomberman\game UI\bridge\training.py:65`–74; the `try` starts at line 93.

The trainer import, second checkpoint resolution and config-file write happen before the worker's exception handler. A disappearing checkpoint or config write failure terminates the thread with `status="initializing"` and `error=None`. The manager only checks status to enforce its active-job slot. Stop changes this dead job to `stopping`, but no live worker remains to reach a terminal status, so every later training start returns a conflict until the server restarts.

Reproduction: patch checkpoint resolution to succeed at `TrainingManager.start` and fail at the worker's second lookup, and suppress directory creation. The thread exited, `thread_alive=False`, `status="initializing"`, `error=None`, and its exception reached `threading.excepthook`. After `control("stop")`, the next start still raised the one-active-job `RuntimeError`. No training or output files were created.

Required correction: catch failures around the entire worker entry point and transition every failed startup to a readable terminal error. Also keep manager registration consistent if starting the thread itself fails. Add a startup-failure regression that proves a subsequent UI job can start.

### R3 — [P2] Keep input sequencing monotonic across match reset

Location: `E:\Bomberman\game UI\bridge\sessions.py:57`.

`_reset_locked` resets `input_seq` to -1 although the WebSocket and `InputController` continue using the same session-wide sequence. Old inputs from the preceding match become valid again. The input message contains no match ID or generation that could distinguish an old input from an intended input for the new match.

Reproduction: send input sequence 10, reset, send stale sequence 9 with `bomb=True`, then single-step. The resulting human action was **5**, placing a bomb. The frozen contract requires old input sequences not to execute twice, including across reset/reconnect behavior.

Required correction: initialize the high-water mark only for a new session, retaining it on reset, or explicitly version inputs by match. Add a regression with a stale movement/bomb packet arriving after reset.

## Minor

### R4 — [P2] Clear the client's held-key state on every pause/reset/disconnect path

Locations: `E:\Bomberman\game UI\src\main.ts:129`–135 and 297–300; `E:\Bomberman\game UI\src\transport.ts:60`–67.

Blur, settings and the pause dialog clear `InputController`, but the main HUD pause/reset actions and unexpected WebSocket closure do not. Server pause clears its held input, while the client still remembers the old direction. For example, hold D, pause/resume through the HUD without releasing D, then press E: the next emitted message contains `direction:"right", bomb:true`; after the bomb, the old direction is held again. The same stale direction can cross a reset or an unexpected reconnect. This is inconsistent with the requirement that pausing/disconnecting clears held controls.

Required correction: clear the local controller and touch-pointer bookkeeping on those lifecycle transitions, without restoring stale input after reconnect. Preserve intentional fresh input/single-step while paused. Validate a held direction across HUD pause/reset and a socket-only disconnection, not merely the browser offline event (which already clears input).

## Reviewed scope and evidence

- Read the frozen requirements, approved plan, execution rulings, Python bridge modules, launcher/config, frontend transport/input/view/renderer paths, asset preparation and manifest, and Python/client/browser acceptance tests.
- The engine remains the game-rule authority. Snapshots copy engine dataclasses and grids; normal inference and training execute outside the ASGI event loop. Checkpoint IDs resolve through a catalog; no direct user-supplied path reaches the loader. Training output is constrained to UI-owned run directories.
- Human/watch setup exposes the required agent counts; training supports C/D/E constraints, real callbacks, optional warm start and its owned-job controls. Rendering reads 9x9/11x11 dimensions. Runtime art has four 36-frame character sheets, 12 terrain frames and 68 bomb/pickup/explosion/auxiliary frames; the renderer consumes the new asset paths. The provenance document was added during review and now exists.
- Existing supplied evidence was 19 Python HTTP/WS/engine/training tests, 3 asset tests, 3 client tests and 9 Playwright tests passing, plus inspected screenshots. These suites were read but not rerun wholesale in this review. They do not cover R1–R4's failure scenarios.
- The initially observed npm browser-test config omission was fixed during review; the current package script explicitly selects `tests/browser/playwright.config.ts`, so it is not an open finding.
- Additional attempted malformed-checkpoint endpoint probes did not complete cleanly and were interrupted; they are not used as passing verification or a confirmed finding. No long training, held-out evaluation, repository-wide test run or branch publication was performed.

## Completion recommendation

Fix R1–R3 before declaring the feature complete, and address R4 in the same lifecycle correction pass. Run focused regressions and the already-scoped acceptance/build gates once after the fixes. No broad security or model-quality conclusions are implied by this review.

## Correction closure — 2026-10-23, implementer record

The same reviewer's narrow source reread confirmed the R1–R3 corrections and found no remaining source defect for R1–R4. Its final message also reproduced the installed Torch parser raising IndexError on corrupt in-memory weights; that case is now handled alongside UnpicklingError and EOFError. This addendum records implementation and test evidence, not a separate independent final approval.

- **R1 closed:** session lock is scoped per iteration, with a positive scheduling gap even when behind; slow-policy pause regression passes.
- **R2 closed:** the entire worker entry point has a terminal error boundary; failed thread dispatch removes manager registration; both startup/next-job regressions pass.
- **R3 closed:** sequence high-water mark initializes only for a new session, not reset; stale post-reset input regression passes.
- **R4 closed:** common clearing covers toolbar pause/reset, authoritative pause/reset snapshots, disconnect, blur, dialogs and leaving the session; touch bookkeeping is cleared too. Both actual browser lifecycle regressions pass. The earlier `review-browser-green.log` filename contains timeout attempts and must not be cited as GREEN. Successful evidence is `review-browser-no-trace.log` and the final full runner.

Additional visual inspection found an incorrect ring countdown (the engine already returns remaining steps) and a loading-form selection race. Both have valid RED logs and passing regressions in the final suite. `artifacts/verification-final.log`: 28 Python + 3 input + typecheck/build + 12 browser tests, exit 0. No initial review finding remains deferred. Limits, baseline root failures and visual evidence are in `docs/verification.md`.
