# S2-KL decision

Status: RB_RETENTION_FAIL

Complete: True; RB games: 1400/1400

S2 remains the default. This pilot has no held-out stage or Phase D qualification.

| Model | R | Q | WR | ActiveWR | Draw | Suicide | SD win | SD death | Passive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 | 97.3633 | 91.8252 | 93.0000 | 84.0000 | 4.5000 | 0.5000 | 2.0000 | 1.5000 | 7.0000 |
| control-100 | 78.7109 | 75.1068 | 84.5000 | 69.5000 | 10.5000 | 1.5000 | 7.0000 | 5.0000 | 8.0000 |
| control-250 | 74.9023 | 70.3139 | 81.5000 | 70.0000 | 14.0000 | 1.0000 | 5.0000 | 4.5000 | 6.5000 |
| control-500 | 75.0000 | 69.9993 | 88.0000 | 76.0000 | 8.0000 | 1.0000 | 4.5000 | 2.5000 | 7.5000 |
| anchor-100 | 76.7578 | 72.7294 | 80.0000 | 67.0000 | 15.0000 | 0.5000 | 5.5000 | 5.0000 | 7.5000 |
| anchor-250 | 82.8125 | 76.7509 | 87.5000 | 76.5000 | 8.0000 | 1.0000 | 5.5000 | 3.0000 | 5.5000 |
| anchor-500 | 77.5391 | 73.2168 | 82.5000 | 68.0000 | 10.5000 | 0.5000 | 8.0000 | 4.0000 | 6.5000 |

Conditions: {'DRIFT_REPRODUCED': True, 'DECISION_RETAINED': False, 'PROBABILITY_RETAINED': False, 'ANCHOR_BENEFIT': False, 'RB_ACTIVE_RETAINED': False, 'RB_SUICIDE_RETAINED': True}

- One training seed; reused SD8A TRAIN observations measure retention only.
- Only fixed endpoint 500 decides the pilot; intermediate checkpoints are diagnostic.
- No held-out stage and no automatic replacement of S2 or Phase D qualification.
