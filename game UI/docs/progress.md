# Execution ledger — Bomberman Game Studio web UI

Plan: `E:\Bomberman\docs\game-studio\2026-10-22-bomberman-web-ui-plan.md`.
Frozen requirements: `game UI/docs/requirements.md`.

## Rulings

- User explicitly requests work in `E:\Bomberman\game UI`; implement in that folder in the current checkout rather than creating a separate worktree whose output would violate the requested destination. Branch is `Loc/sd8-huong-a`, not main. Preserve unrelated changes. No commit/push requested.
- All new source/tests/assets/dependencies/builds/job outputs live under `game UI`; original plan's `web/` and `web_bridge/` become local frontend and `bridge/` beneath that folder.
- User approved implementation of the plan. Adopt proposed art A (robot foundry pixel art), perform sample/preview inspection before generating full animation strips. No additional platform question is needed.
- Implementation is performed inline; independent acceptance author and final reviewer are required by the applicable skills.
- New package dependencies install into `game UI/node_modules` and `game UI/.python-deps`; existing Python environment provides Torch/NumPy. Existing environment and experiment state are not modified.

## Milestones

- [x] W0: independent acceptance RED baseline, contracts, environment.
- [x] W1: approved-direction seed assets and layout.
- [x] W2: authoritative Python match rendered by Phaser.
- [x] W3: full new asset manifest and sprite/FX pipeline.
- [x] W4: polished menus/play/watch, input/reconnect/mobile/settings/results.
- [x] W5: real training UI, owned worker lifecycle and bounded smoke.
- [x] W6: tests, screenshots, fresh review, fixes and handover.

## Completion — 2026-10-23

Final runner exit 0: 28 Python tests, 3 client input tests, typecheck/build and 12 Chrome journeys. Independent acceptance RED/GREEN retained; review lifecycle findings fixed with additional regression RED/GREEN. Visual inspection found and corrected a double-subtraction in the ring countdown. Setup now disables editable fields during catalog loading to prevent selection replacement. All new runtime art is v2; source prompts/normalization helpers/provenance are retained beneath this folder.

See `docs/verification.md` for evidence, screenshots, measurements and limits. The root suite was already red (535 tests, 20 errors/1 failure/2 skips) before UI production work; do not label the whole repository green. No existing engine/trainer/checkpoint was changed, no long training or quality evaluation was run. Current checkout is retained without commit/push.

## Desktop map enlargement — 2026-10-23

User requested a larger displayed map. The smallest change updates three CSS rules: desktop frame/topbar cap 720px, playfield cap 760px, removing the viewport-height cap. At 1366×768 the canvas grew from 396px to 698px; at 1440×1000 and 1920×1080 from 578px to 698px. Tablet/mobile measurements are unchanged. Narrow screens retain their existing responsive rules; short desktop screens may scroll vertically to reach controls.

Verified typecheck/build, a five-viewport Chrome probe without horizontal overflow or overlapping HUD, existing mobile/touch test and existing desktop/asset test. One desktop test timed out during simultaneous browser work; its unchanged standalone rerun passed in 5.6s. Evidence: `artifacts/map-size-before.json`, `map-size-after.json`, `map-size-build.log`, `map-size-browser.log`, `map-size-browser-recheck.log`, and `screenshots/enlarged-map-desktop.png`. No engine/AI changes or new dependencies/tests.
