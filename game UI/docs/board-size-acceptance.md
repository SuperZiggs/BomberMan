# Board-size acceptance evidence

Independent test author: requirements and current code only; no implementation plan received.
Frozen source: `docs/board-size-requirements.md` (2026-10-24).

| Requirement | Level / scenario | Observable outcome | Command |
|---|---|---|---|
| Accessible choices in Play and Watch; initial 11 | Real browser, both modes | Visible labelled control offers 9 and 11; 11 is initially selected | Browser command below |
| Explicit 9/11 and omitted-size compatibility | HTTP integration | Real Python snapshots have the requested square grid; omitted size remains 11 | Python command below |
| Existing supported agent counts | HTTP + real inference | Play 3/4 and Watch 2/3/4 work at both sizes and can step | Python command below |
| Selected model is loaded at the requested size | Session integration, real checkpoint/engine | Policy network and engine sizes equal the requested size; inference steps successfully | Python command below |
| Reject unsupported sizes/types; preserve validation | HTTP contract | Only integer 9/11 accepted; invalid model IDs and existing invalid configuration remain 4xx | Python command below |
| Single step/reset retain actual size | HTTP/WebSocket + browser | Tick advances once, reset starts a new paused match at tick zero, grid and canvas retain size | Both commands below |
| Actual rendered 9 and regression 11 | Real browser, Play/Watch x 9/11 | Submitted size equals selected size; real snapshot grid and canvas dimensions agree | Browser command below |

Commands (PowerShell, from `E:/Bomberman/game UI`):

```powershell
$env:PYTHONPATH='E:/Bomberman/game UI/.python-deps;E:/Bomberman/game UI;E:/Bomberman'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& 'E:/Bomberman/venv/Scripts/python.exe' -m unittest discover -s tests/python -p test_board_size.py -v
& 'E:/Bomberman/venv/Scripts/python.exe' tests/run_board_size_browser.py
```

The browser runner builds the UI and starts/stops its own loopback server on an ephemeral port.
The Python suite reuses the existing disposable-server helper. Neither command uses port 8000.
No mocks replace API responses, the engine, or inference. No training or checkpoint writes are involved.

## Baseline before adding tests

- `node --experimental-strip-types --test tests/client.test.ts`: 3 passed.
- Existing Python `SessionTests`: first sandboxed run failed due to loopback `WinError 10013`; this is environment evidence, not feature RED.
- Rerun with local-loopback sandbox escalation: all 13 existing Python `SessionTests` passed in 5.823s.

## RED evidence

Python command above, executed with local-loopback sandbox escalation: **6 tests, 15 assertion failures, 0 errors, 3.535s**.

- Explicit integer 9 and 11 fail at HTTP creation with `422`, `extra_forbidden`, location `body.board_size`; the field is missing from the contract.
- Real-model integration reports the same missing field through a deliberate assertion rather than an uncaught validation error.
- Omitted-size 11 and rejection of invalid sizes/types pass as already-existing compatibility/guard behavior. They are regression coverage, not claimed RED.
- Step/reset and preserved-validation scenarios are blocked by the same valid board-size rejection; they will proceed once the contract accepts the requested field.

Browser command above, executed with local-loopback/Chrome sandbox escalation: **4 failed, 0 setup or syntax errors**.

- Vite build succeeded (10 modules, 409ms).
- The disposable server answered health checks at `http://127.0.0.1:49400`.
- Play and Watch at both sizes fail at the assertion `Match setup must expose a labelled map-size choice`: element not found after 8000ms.
- Browser pages rendered the existing setup dialogs; screenshots/error contexts are under `tests/_artifacts/browser/board-size-*` and machine-readable results at `tests/_artifacts/browser-results.json` (later browser runs overwrite those ordinary artifacts).
- The runner exited with code 1 and terminated its own server. It was not restarted after production work began.

All feature failures are missing-behavior assertions. Compatibility/invalid-type cases that already pass are explicitly retained as regressions. The suite is handed off unchanged after this RED record; production implementation and final GREEN verification belong to the implementer.

## Frozen test handoff

| File | SHA-256 |
|---|---|
| `tests/python/test_board_size.py` | `B3FDA2A66BDD1DD48E9E4C9B10DA0EF297FEA8EC8809E211897CAA4C9D58160A` |
| `tests/browser/board-size.spec.ts` | `4131A6300AA8AA3FF1D896C05566DE4DDFA24EB852D499BB2B2DD04689DB6F0C` |
| `tests/run_board_size_browser.py` | `BDBC62E5804BFE542D8578DB122664328E19B8CDCF4EA7B53A758AEFC14F342B` |

No existing tests were edited. This author changed only these new test/helper files and this evidence document; Vite and tests also generated their ordinary build/cache/artifact outputs.
