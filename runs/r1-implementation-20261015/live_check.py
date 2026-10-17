"""Read completed metrics without touching the training process or held-out data."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path((ROOT / 'runs/reward-last-path.txt').read_text(encoding='utf-8-sig').strip())
result = {}
for arm, coefficient in (('control', .15), ('half_wall', .075)):
    path = OUT / arm / 'metrics.jsonl'
    if not path.exists():
        continue
    text = path.read_text(encoding='utf-8-sig')
    lines = text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    rows = [json.loads(line) for line in lines]
    for row in rows:
        assert math.isclose(sum(row['reward_components'].values()), row['reward'], rel_tol=0, abs_tol=1e-10)
        assert row['reward_component_max_error'] <= 1e-10
        walls = row['reward_components'].get('wall', 0) / coefficient
        assert math.isclose(walls, round(walls), rel_tol=0, abs_tol=1e-9)
        assert 0 <= round(walls) <= row['walls']
        assert all(math.isfinite(row[key]) for key in ('loss', 'policy_loss', 'value_loss', 'reward'))
    result[arm] = {'episodes_logged': len(rows), 'last_optimizer_steps': rows[-1]['optimizer_steps'] if rows else 0,
                   'minutes': round(rows[-1]['elapsed_seconds'] / 60, 2) if rows else 0,
                   'accounting_and_finite': True}
print(json.dumps(result))
