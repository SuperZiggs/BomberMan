"""Read completed log lines only; never open a live checkpoint on Windows."""
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
out=Path((ROOT/'runs/ppo-last-path.txt').read_text(encoding='utf-8-sig').strip())
result={}
for arm in ('control','ppo'):
    p=out/arm/'metrics.jsonl'
    if not p.exists(): continue
    text=p.read_text()
    rows=[json.loads(line) for line in text[:text.rfind('\n')+1].splitlines()]
    if not rows: continue
    assert [r['episode'] for r in rows]==list(range(1,len(rows)+1))
    assert rows[-1]['optimizer_steps']==sum(r['ratio_update']['epochs'] for r in rows)
    result[arm]={'episodes':len(rows),'updates':rows[-1]['optimizer_steps'],
        'epochs':dict(Counter(str(r['ratio_update']['epochs']) for r in rows)),
        'early_stop':sum(r['ratio_update']['early_stop'] for r in rows),
        'best_episode':rows[-1]['best_episode']}
    epochs=[e for row in rows for e in row['ratio_update']['epoch_metrics']]
    result[arm]['clip_fraction_percent']=round(100*sum(e['clip_fraction']*e['samples'] for e in epochs)/sum(e['samples'] for e in epochs),3)
    result[arm]['max_post_kl']=round(max(e['exact_kl_after_update'] for e in epochs),6)
    v=out/arm/'validation_metrics.jsonl'
    if v.exists():
        text=v.read_text()
        vals=[json.loads(s) for s in text[:text.rfind('\n')+1].splitlines()]
        result[arm]['validations']=[{k:r[k] for k in ('episode','active_win_rate','suicide_rate','draw_rate','selected')} for r in vals]
print(json.dumps(result))
