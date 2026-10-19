import sys, time, json
sys.path.insert(0, r'D:\fpt\subject\BomberMan')
import numpy as np, torch; torch.set_num_threads(2)
import evaluate, train_actor_critic as trainer
from train_actor_critic import ActorCriticNet, smart_load_weights, obs_to_tensor, get_action_mask
from kill_search import search_action
F = r'D:\fpt\subject\BomberMan\runs\phaseC-sd-best-fallback\best_phaseC.pt'
net = ActorCriticNet(in_channels=13, board_size=9); smart_load_weights(net, F, 'cpu'); net.eval()
rows = []
def agent(env, obs, aid):
    mask = get_action_mask(env, aid)
    with torch.no_grad():
        logits, _ = net(obs_to_tensor(obs[aid], 'cpu'))
    logits = logits.masked_fill(~torch.tensor(mask, dtype=torch.bool), -1e9)
    p = torch.softmax(logits, -1)[0].numpy()
    a = int(torch.distributions.Categorical(logits=logits).sample())
    _, info = search_action(env, aid, mask, a)
    if not info['skipped']:
        s = info['scores']
        if len(set(s.values())) > 1:
            best = max(s.values()); bs = [k for k, v in s.items() if v == best]
            rows.append(dict(step=env.step_count, best=best, pbest=float(p[bs].sum()), argmax_ok=int(np.argmax(p)) in bs,
                             sample_ok=a in bs, sampled_score=s[a], bomb_best=5 in bs, ent=float(-(p*np.log(p+1e-12)).sum())))
    return a
t = time.time()
r = evaluate.evaluate_seeded(agent, 9, 2, 40, 6600, 500, sudden_death=True, sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
for name, sel in (('kill', [x for x in rows if x['best'] == 1.0]), ('escape', [x for x in rows if x['best'] == 0.0])):
    if sel:
        print(name, 'n', len(sel), 'sample_ok', round(np.mean([x['sample_ok'] for x in sel]), 3), 'argmax_ok', round(np.mean([x['argmax_ok'] for x in sel]), 3),
              'pbest mean', round(np.mean([x['pbest'] for x in sel]), 3), 'pbest quartiles', np.round(np.percentile([x['pbest'] for x in sel], [25, 50, 75]), 3).tolist(),
              'bomb_best', round(np.mean([x['bomb_best'] for x in sel]), 2), 'sampled -1', sum(x['sampled_score'] < 0 for x in sel))
print('F alone AWR', r['active_win_rate'], 'WR', r['win_rate'], 'draw', r['draw_rate'], 'kill states/game', round(sum(x['best']==1 for x in rows)/40, 2), 's/game', round((time.time()-t)/40, 1))
json.dump(rows, open(S.replace('.py', '.json'), 'w'))
