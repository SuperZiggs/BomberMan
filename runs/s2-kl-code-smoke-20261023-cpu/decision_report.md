# S2-KL decision

Status: SMOKE_ONLY

Complete: True; RB games: 20/20

S2 remains the default. This pilot has no held-out stage or Phase D qualification.

| Model | R | Q | WR | ActiveWR | Draw | Suicide | SD win | SD death | Passive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 | 97.3633 | 91.8252 | 0.0000 | 0.0000 | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| control-2 | 97.3633 | 91.7878 | 0.0000 | 0.0000 | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| control-4 | 97.3633 | 91.7583 | 0.0000 | 0.0000 | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| anchor-2 | 97.3633 | 91.7878 | 0.0000 | 0.0000 | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| anchor-4 | 97.3633 | 91.7580 | 0.0000 | 0.0000 | 100.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Conditions: {'DRIFT_REPRODUCED': False, 'DECISION_RETAINED': True, 'PROBABILITY_RETAINED': True, 'ANCHOR_BENEFIT': False, 'RB_ACTIVE_RETAINED': True, 'RB_SUICIDE_RETAINED': True}

- One training seed; reused SD8A TRAIN observations measure retention only.
- Only fixed endpoint 4 decides the pilot; intermediate checkpoints are diagnostic.
- No held-out stage and no automatic replacement of S2 or Phase D qualification.
- Smoke with 20-step episodes does not establish full-run throughput or model quality.
