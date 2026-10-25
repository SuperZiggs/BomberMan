# Match board-size verification — 2026-10-24

## Change

Play and Watch setup now offer a labelled native select for 9 × 9 and 11 × 11. The default remains 11. The API strictly accepts only integer 9/11, and both policy loading and engine creation use the requested size. Reset reuses that session configuration. The existing Phaser renderer already sizes the canvas from the snapshot grid, so no renderer, asset, training, engine-rule, or dependency change was needed.

Requirements: [board-size-requirements.md](board-size-requirements.md). Independent RED baseline, test matrix, commands, and frozen hashes: [board-size-acceptance.md](board-size-acceptance.md). The three handed-off test/helper files retain their recorded hashes.

## Results

- New Python acceptance: **6 passed**, including both sizes/modes, supported agent counts, strict validation, real model loading/inference, single step, and reset.
- Web-project Python suite: **34 passed** on the first complete Python run (`tools/verify.py`).
- Existing client unit tests: **3 passed**.
- TypeScript `tsc --noEmit` and Vite production build: **passed**; `dist/` contains the new option.
- Chrome suite with default tracing disabled: **15 passed, 1 timed out**; all four new Play/Watch × 9/11 cases passed. The remaining existing catalog-loading regression passed independently afterward (**1 passed, 3.3s**) without code or test changes.
- A repeated Python run encountered a Windows loopback connection abort during training-test cleanup; running that test during Chrome verification also timed out. The same isolated two-episode training test passed after Chrome closed (**1 passed, 3.654s**). No training code or test was changed.
- An exploratory browser run with trace recording enabled was interrupted after page-loading timeouts. It is not reported as passing. The final browser results above use the project's default tracing-disabled configuration.
- Independent read-only review: **no actionable findings** across the five modified source/documentation files.

Browser screenshots from the four passing map-size journeys are retained under `artifacts/board-size-validation/`. Every verification server used an ephemeral loopback port and was stopped afterward; the user's port-8000 server was not restarted.

## Apply to the currently open UI

The running server on port 8000 was confirmed to have the old API schema (no `board_size`). Stop it with Ctrl+C, then run from the repository root:

```powershell
cd 'E:\Bomberman'
& '.\game UI\start.cmd'
```

Refresh the browser with Ctrl+F5, open Play or Watch setup, and choose **Kích thước bản đồ → 9 × 9**.
