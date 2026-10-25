# Independent acceptance matrix

Frozen source: `docs/requirements.md`, 2026-10-22. The author inspected only that packet and existing `bomberman_env.py`, `visualize.py`, `train_actor_critic.py`, `test_visualize.py` and test conventions. No implementation plan, progress ledger or new production implementation was supplied.

Decision: exercise the public boundary using a disposable loopback ASGI process and real engine/checkpoints; use standard-library unittest/urllib plus the approved websockets package. Browser tests use accessible labels, public HTTP state and network frames. They never invoke private bridge internals. Human slot 0 follows the existing GUI convention. Tests create only bounded jobs and write only under `game UI`.

| Requirement | Level / scenario | Observable outcome | Test / command |
|---|---|---|---|
| 1, 8 | HTTP startup/catalog | Healthy protocol v1, nonempty opaque catalog | `HealthCatalogTests`; Python command below |
| 2, 6 | HTTP valid mode/agent boundaries and real checkpoint | Human 3/4 and watch 2/3/4 create paused 11x11 matches with correct agent counts | `SessionTests.test_supported_modes_and_counts` |
| 2, 4, 6 | HTTP invalid configs, traversal/absolute model IDs, failed load recovery | 4xx and next valid create works | `SessionTests.test_invalid_session_configs_are_recoverable` |
| 3 | Snapshot adapter on 9x9/11x11; mutable env/metadata | JSON values mirror dataclasses, independent nested copies, no mutation/reseed | `SnapshotTests` |
| 3 | Real engine action tape replay | Snapshot actions replay against seeded engine with identical grid/agents/bombs/flames/powerups/rewards/terminal | `SessionTests.test_authoritative_action_tape_matches_real_engine` |
| 2, 4 | WS lifecycle, heartbeat, duplicate IDs | hello+full state, pong, step exactly once, old ID never repeats | `SessionTests.test_ws_handshake_ping_step_and_duplicate_command` |
| 2, 4 | WS input tap/release, stale sequence, queued bomb | Tap retained, stale input ignored, movement cannot overwrite bomb | `SessionTests.test_tap_and_old_input_sequence`, `test_bomb_is_not_replaced_by_movement` |
| 2, 4 | WS pause/resume/reset/speed, held-input clear | Automatic ticks only while running; pause clears controls; reset new match at zero paused; 1/60 accepted, 0/61 rejected | `SessionTests.test_pause_resume_speed_and_reset`, `test_pause_clears_controls` |
| 3 | Max-steps terminal | Result present; later step/resume never advances terminal frame | `SessionTests.test_terminal_never_advances` |
| 4 | Disconnect/reconnect and explicit foreign WS/HTTP Origin | Disconnect pauses and clears controls; reconnect full state; foreign browser writes rejected | `SessionTests.test_disconnect_pauses_and_reconnects`, `test_foreign_origins_rejected` |
| 4 | Invalid JSON/message/version | Recoverable errors, same connection stays usable, unsupported version cannot advance | `SessionTests.test_invalid_messages_and_unknown_version` |
| 6 | Unknown/deleted sessions | 404; local delete closes target only | `SessionTests.test_unknown_and_deleted_session` |
| 7 | HTTP phase/count/episode/model validation | 4xx for bad combinations; unknown job/action cannot control existing experiment | `TrainingTests.test_validation_and_job_isolation` |
| 7 | Actual tiny phase-C worker, callbacks/frame/save | <=2 episodes, <=8 steps; actual loss/reward/win/active-win API callback data, 9x9 latest frame, checkpoint under UI runs | `TrainingTests.test_real_bounded_training_metrics_and_artifacts` |
| 7 | Actual bounded job control, one-active constraint | UI-created job only; pause/resume/stop accepted, competing active start 409, eventual saved terminal state | `TrainingTests.test_job_controls_and_competing_start` |
| 1, 2, 8 | Browser critical journey | Vietnamese menu → setup → real paused match → step/resume/pause/reset → results; no console/runtime failures | `journey.spec.ts` |
| 2, 4, 5 | Browser keys, blur and touch; responsive layout | Real HTTP state follows input; blur clears held direction; visible unobscured board and touch controls at mobile viewport | `journey.spec.ts`, `presentation.spec.ts` |
| 5, 8 | Browser keyboard/settings/reduced motion/loading/error/disconnect | Accessible focus, working settings, reduced motion available; recoverable loading/connection/error states | `presentation.spec.ts`, `journey.spec.ts` |
| 5 | Browser visual assets, font, sprites and effects | Same-origin successful asset requests, no legacy files, self-hosted Vietnamese font, animated four-slot identity with numbers/shapes; screenshots for visual review | `presentation.spec.ts`; screenshots + documented visual review |
| 7, 8 | Browser training setup validation and bounded real submission | C/D constrain 2 agents, E 2/3/4; invalid episode cannot start; bounded UI job shows metrics/frame/control states | `training.spec.ts` |
| 8 | Windows commands/build/assets | Install/build/test/start documented and verified; packaged asset refs stay in UI; no publication | parent verification plus browser asset assertions |

Commands from repo root (PowerShell):

```powershell
$env:PYTHONPATH = 'E:\Bomberman\game UI\.python-deps;E:\Bomberman\game UI;E:\Bomberman'
& 'E:\Bomberman\venv\Scripts\python.exe' -m unittest discover -s 'game UI/tests/python' -p 'test_*.py' -v
```

Browser command from `game UI` after normal app server is running on loopback (defaults `http://127.0.0.1:8000`, overridable by `BOMBERMAN_BASE_URL`):

```powershell
npx playwright test --config tests/browser/playwright.config.ts
```

The browser suite's first assertion is a readable rendered frontend readiness check. A missing frontend is a behavior failure; missing Playwright/browser tooling is recorded as blocked setup, never mislabeled RED. Visual screenshots supplement executable geometry/network/state assertions; they do not prove animation/art identity without visual inspection.
