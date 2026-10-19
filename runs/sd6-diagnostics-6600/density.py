import sys, time, json
sys.path.insert(0, r'D:\fpt\subject\BomberMan')
import torch; torch.set_num_threads(2)
import evaluate, train_actor_critic as trainer
from kill_search import search_action
F = r'D:\fpt\subject\BomberMan\runs\phaseC-sd-best-fallback\best_phaseC.pt'
policy = evaluate.make_ac_fn(F, 9)
c = dict(steps=0, near=0, informative=0, override=0, agree_kill=0, override_bomb=0)
def agent(env, obs, aid):
    a = policy(env, obs, aid)
    choice, info = search_action(env, aid, trainer.get_action_mask(env, aid), a)
    c['steps'] += 1
    if not info['skipped']:
        c['near'] += 1
        s = info['scores']
        if len(set(s.values())) > 1:
            c['informative'] += 1
            if choice != a:
                c['override'] += 1; c['override_bomb'] += choice == 5
            elif s[a] == 1.0: c['agree_kill'] += 1
            c['has_kill'] = c.get('has_kill',0) + (max(s.values()) == 1.0)
    return choice
t = time.time()
r = evaluate.evaluate_seeded(agent, 9, 2, 20, 6600, 500, sudden_death=True, sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
print(json.dumps(c), 'games 20', 'AWR', r['active_win_rate'], 'sec/game', round((time.time()-t)/20, 1))
