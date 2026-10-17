"""Check resumed progress without opening the active checkpoint file."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[1]
out = Path((root / 'runs/ppo-last-path.txt').read_text().strip())
before = json.loads((here / 'before-resume.json').read_text())
for name, expected in before['control_files'].items():
    assert hashlib.sha256((out / 'control' / name).read_bytes()).hexdigest() == expected, name
rows = []
for line in (out / 'ppo/metrics.jsonl').read_text().splitlines():
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        break
assert [r['episode'] for r in rows] == list(range(1, len(rows) + 1))
assert rows[before['paused_at_ppo_episode']]['episode'] == 37
assert rows[35]['optimizer_steps'] == 144
assert rows[36]['optimizer_steps'] == 148
recovery = next((out / 'ppo/recovery').glob('*/metrics.jsonl'))
old_rows = [json.loads(line) for line in recovery.read_text().splitlines()]
assert len(old_rows) >= 36 and old_rows[:36] == rows[:36]
audit = {
    'checked_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'resumed',
    'control_files_unchanged': len(before['control_files']),
    'paused_episode': 36,
    'first_resumed_episode': 37,
    'adam_steps_before': 144,
    'adam_steps_after_first_resumed_episode': 148,
    'episodes_observed': len(rows),
    'committed_metric_rows_unchanged': 36,
    'registered_utc': json.loads((out / 'config.json').read_text())['registered_utc'],
    'training_deadline': json.loads((out / 'training_deadline.json').read_text()),
    'active_checkpoint_not_opened': True,
}
(here / 'resume_review.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit))
