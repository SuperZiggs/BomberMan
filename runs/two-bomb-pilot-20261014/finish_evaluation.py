"""Finish the reserved evaluation stage after the original runner's train timeout."""
from pathlib import Path
import sys
import json
import hashlib
import shutil
from datetime import datetime, timezone

output = Path(__file__).resolve().parent
root = output.parents[1]
sys.path.insert(0, str(root))
import torch
from pilot_two_bomb import final_evaluation, paired_active_bootstrap, complete_validation, save_json

torch.set_num_threads(1)
status = json.loads((output / 'status.json').read_text())
manifest = json.loads((output / 'config.json').read_text())
assert status['state'] == 'inconclusive', 'Wait for the training process to finish'
assert not list(output.glob('evaluation_*_25000.json')), 'Do not overwrite an existing evaluation'
checkpoint = Path(manifest['training']['resume_from'])
started = datetime.fromisoformat(manifest['started_utc'])
elapsed = lambda: (datetime.now(timezone.utc) - started).total_seconds()
seconds = min(40 * 60, 120 * 60 - elapsed())
assert seconds > 0, 'Overall pilot budget is exhausted'
snapshot = output / 'evaluation_source'
snapshot.mkdir()
for filename in ('pilot_two_bomb.py', 'test_tactical_teacher.py'):
    shutil.copy2(root / filename, snapshot / filename)
manifest['evaluation_runner_sha256'] = hashlib.sha256((root / 'pilot_two_bomb.py').read_bytes()).hexdigest()
save_json(output / 'config.json', manifest)
status['validations_complete'] = {}
for name in ('local', 'two_bomb'):
    rows = [json.loads(line) for line in (output / name / 'validation_metrics.jsonl').read_text().splitlines()]
    status['validations_complete'][name] = complete_validation(rows)
status['training_complete'] = (status['completed_episodes'] == {'local': 200, 'two_bomb': 200}
                               and all(status['validations_complete'].values()))
status.update(state='evaluation', promote=False)
save_json(output / 'status.json', status)
results = final_evaluation(output, checkpoint, seconds)
status.update(state='inconclusive', promote=False, decision='retain_original',
              evaluated_models=list(results), evaluation_complete=len(results) == 3,
              elapsed_seconds=elapsed())
if len(results) == 3:
    status['comparisons'] = {name: paired_active_bootstrap(results['two_bomb'], results[name])
                             for name in ('original', 'local')}
else:
    status['reason'] += '; final evaluations also incomplete'
save_json(output / 'status.json', status)
print(json.dumps(status, indent=2), flush=True)
