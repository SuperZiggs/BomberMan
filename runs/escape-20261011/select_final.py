"""Recheck candidates after the placement-order fix, using validation seeds only."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from evaluate import make_ac_fn, evaluate_seeded
torch.set_num_threads(1)
folder = Path(__file__).parent / 'pilot-500'
deadline = time.monotonic() + 1800
while not (folder/'training_summary.json').is_file():
    if time.monotonic() >= deadline:
        raise TimeoutError('The 500-episode pilot did not finish')
    time.sleep(1)
rows = [json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
assert len(rows) == 500
training_selection = json.loads((folder/'best_validation.json').read_text())
shutil.copyfile(folder/'best_phaseC.pt', folder/'training_selected_phaseC.pt')
shutil.copyfile(folder/'best_validation.json', folder/'training_best_validation.json')
original = ROOT/'runs/20261010-135449-587798/best_phaseC.pt'
original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
candidates = [('original',original,0),
              ('training_best',folder/'training_selected_phaseC.pt',training_selection['episode']),
              ('final',folder/'final_phaseC.pt',500)]
results = []
for name, path, episode in candidates:
    print('Post-validation',name,flush=True)
    result = evaluate_seeded(make_ac_fn(str(path),9,'cpu'),num_episodes=48,seed=6000)
    assert result['num_episodes'] == 48 and result['active_win_rate'] <= result['win_rate']
    assert all(result['by_corner'][str(a)]['num_episodes'] == 24 for a in [0,1])
    result.update(episode=episode,source='post-training CPU validation',source_checkpoint=str(path),device='cpu')
    (folder/f'post_validation_{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    score = (result['active_win_rate'],-result['suicide_rate'],-result['draw_rate'],result['avg_kills'],result['win_rate'])
    results.append((score,name,path,result))
    print(name,json.dumps({k:v for k,v in result.items() if k not in ['games','by_corner']}),flush=True)
winner = max(results,key=lambda r:r[0])
shutil.copyfile(winner[2],folder/'best_phaseC.pt')
selected = {k:v for k,v in winner[3].items() if k != 'games'}
selected['candidate'] = winner[1]
selected['runtime_hashes'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                            for name in ['bomberman_env.py','rule_based_bot.py','train_actor_critic.py','evaluate.py']}
(folder/'best_validation.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
with (folder/'validation_metrics.jsonl').open('a',encoding='utf-8') as handle:
    for _,name,_,result in results:
        handle.write(json.dumps({k:v for k,v in result.items() if k != 'games'} | {'selected':name==winner[1]})+'\n')
assert hashlib.sha256(original.read_bytes()).hexdigest() == original_hash
print('SELECTED',winner[1],'episode',selected['episode'],flush=True)
