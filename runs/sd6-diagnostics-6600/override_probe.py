import sys, json
sys.path.insert(0, r'D:\fpt\subject\BomberMan')
import numpy as np, torch; torch.set_num_threads(1)
from concurrent.futures import ProcessPoolExecutor
def run(seed):
    import evaluate
    from train_actor_critic import ActorCriticNet, smart_load_weights, obs_to_tensor, get_action_mask
    from kill_search import search_action
    F = r'D:\fpt\subject\BomberMan\runs\phaseC-sd-best-fallback\best_phaseC.pt'
    net = ActorCriticNet(in_channels=13, board_size=9); smart_load_weights(net, F, 'cpu'); net.eval()
    ov, games = [], []
    def agent(env, obs, aid):
        mask = get_action_mask(env, aid)
        with torch.no_grad():
            logits, _ = net(obs_to_tensor(obs[aid], 'cpu'))
        logits = logits.masked_fill(~torch.tensor(mask, dtype=torch.bool), -1e9)
        p = torch.softmax(logits, -1)[0].numpy()
        a = int(torch.distributions.Categorical(logits=logits).sample())
        c, info = search_action(env, aid, mask, a)
        if c != a:
            opp = [o for o in env.agents if o.agent_id != aid][0]
            me = env.agents[aid]
            ov.append(dict(seed=env.seed if hasattr(env,'seed') else None, step=env.step_count, choice=c, p_choice=float(p[c]), argmax=int(p.argmax()),
                           best=max(info['scores'].values()), dist=abs(opp.row-me.row)+abs(opp.col-me.col),
                           opp_bombs_live=sum(b.owner != aid for b in env.bombs), my_bombs_live=sum(b.owner == aid for b in env.bombs)))
        return c
    r = evaluate.evaluate_seeded(agent, 9, 2, 20, seed, 500, sudden_death=True, sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
    return ov, r['active_win_rate']
if __name__ == '__main__':
    with ProcessPoolExecutor(4) as pool:
        out = list(pool.map(run, [6600, 6610, 6620, 6630]))
    ov = [o for part, _ in out for o in part]
    print('AWR per shard', [a for _, a in out], 'overrides', len(ov), 'per game', len(ov)/80)
    kill = [o for o in ov if o['best'] == 1.0]; esc = [o for o in ov if o['best'] < 1.0]
    for name, sel in (('kill-overrides', kill), ('escape-overrides', esc)):
        if not sel: continue
        bomb = [o for o in sel if o['choice'] == 5]; move = [o for o in sel if o['choice'] != 5]
        print(name, len(sel), '| bomb', len(bomb), 'p_choice median', np.median([o['p_choice'] for o in bomb]).round(3) if bomb else '-',
              '| move/stop', len(move), 'p_choice median', np.median([o['p_choice'] for o in move]).round(3) if move else '-',
              '| argmax==choice', sum(o['argmax']==o['choice'] for o in sel))
        print('   with opponent bomb live', sum(o['opp_bombs_live']>0 for o in sel), 'with my bomb live', sum(o['my_bombs_live']>0 for o in sel),
              'dist quartiles', np.percentile([o['dist'] for o in sel],[25,50,75]).tolist(), 'step median', np.median([o['step'] for o in sel]))
        mv = [o for o in move]
        print('   move-kill with my bomb live', sum(o['my_bombs_live']>0 for o in mv), 'with no bombs at all', sum(o['my_bombs_live']==0 and o['opp_bombs_live']==0 for o in mv))
    json.dump(ov, open(r'C:\Users\ADMIN\AppData\Local\Temp\claude\D--fpt-subject-BomberMan\ff20192f-36a6-44f0-9ed3-798070a95674\scratchpad\overrides.json','w'))
