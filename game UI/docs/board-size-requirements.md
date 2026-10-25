# Match board-size selection

Requested on 2026-10-24: add a 9x9 map option to the existing web Game UI.

## Acceptance requirements

- Both Play and Watch match setup expose an accessible, visibly labelled map-size control with 9x9 and 11x11 options.
- 11x11 remains the initial/default option and the default for existing API callers that omit board size.
- Choosing 9x9 creates a real 9x9 Python-engine match and loads the selected model for that size; choosing 11x11 continues to create an 11x11 match.
- The session API accepts only integer board sizes 9 and 11, rejects unsupported sizes/types, and preserves existing model-ID and configuration validation.
- Single step and reset work for a 9x9 match. Reset preserves the chosen size, and snapshots and rendered canvas reflect the actual grid.
- Both sizes remain usable with the existing supported agent counts. No new asset or dependency is required.
- Scope is match Play/Watch, not training settings, model discovery, checkpoint files, experiment results, or changes to Python engine/AI rules.

## Verification

Use requirements-derived Python contract/integration tests and targeted real-browser journeys with a disposable local server. Keep inference/engine real at integration/E2E boundaries where practical. Run existing web-project checks after implementation.
