# Phase C-SD Best Fallback Marker

This branch preserves the source configuration that produced the strongest verified Phase C-SD checkpoint before temporal planning and resumable evaluation were introduced.

## Frozen artifact

- Folder: `runs/phaseC-sd-best-fallback`
- Best checkpoint: `best_phaseC.pt`
- SHA-256: `0315A56F09079333D936DB5F55587093F1B6415ED6355D2EF6AE2AB63F8B1483`
- Evaluation: 200 seeded games
- Win rate: `67.5%`
- Active win rate: `51.0%`
- Draw rate: `21.5%`
- Suicide rate: `0.5%`
- Sudden Death death rate: `13.0%`

## Source boundary

The fallback keeps Sudden Death, the 13-channel observation, static pressure/two-position attack supervision, target-aware checkpoint selection, and the one-command 750-episode train plus 200-game evaluation launcher. It removes the later temporal planner/follower, delayed bomb placement forecast, resumable evaluation lifecycle, and their launchers/tests/docs.

No new training or evaluation was run while creating this marker.

## Verification

- Fallback acceptance: `9/9` passed.
- Target/attack/Sudden Death acceptance suites passed.
- Full historical suite: `114` tests ran with `3` failures in `test_combat.py`.
- The failures reproduce target-era attack-teacher behavior and are intentionally preserved rather than changing the source that generated the checkpoint:
  - `CombatChecks.test_danger_targets_escape_instead_of_attack`
  - `CombatChecks.test_safe_followup_traps_opponent_and_survives_simultaneous_counter_bomb` for learner corners `0` and `1`
