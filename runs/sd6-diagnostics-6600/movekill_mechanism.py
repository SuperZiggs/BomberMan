"""SD6.1 follow-up: what are kill-search's killing moves? (diagnostic seed 15000, F + search)."""
import sys, json
sys.path.insert(0, r'D:\fpt\subject\BomberMan')
from concurrent.futures import ProcessPoolExecutor
def run(seed, games=50):
    import numpy as np, torch; torch.set_num_threads(1)
    from copy import deepcopy
    import evaluate
    from bomberman_env import ACTION_BOMB, _MOVE_DELTA, ACTION_STOP
    from kill_search import search_action
    from rule_based_bot import RuleBasedAgent, _survival_map
    from train_actor_critic import get_action_mask
    policy = evaluate.make_ac_fn(r'D:\fpt\subject\BomberMan\runs\phaseC-sd-best-fallback\best_phaseC.pt', 9)
    rows = []
    def surv(env, o, contests):
        s, _, _ = _survival_map(env, o.row, o.col, avoid_contests=contests); return bool(s[0, o.row, o.col])
    def rb_action(env, oid):
        st = np.random.get_state()
        try: return RuleBasedAgent(oid).select_action(env)
        finally: np.random.set_state(st)
    def agent(env, obs, aid):
        mask = get_action_mask(env, aid); a = policy(env, obs, aid)
        c, info = search_action(env, aid, mask, a)
        if c != a and c != ACTION_BOMB and max(info['scores'].values()) == 1.0:
            me = env.agents[aid]; o = [x for x in env.agents if x.agent_id != aid][0]
            moved = deepcopy(env); dr, dc = _MOVE_DELTA.get(c, (0, 0)); moved.agents[aid].row += dr; moved.agents[aid].col += dc
            mo = moved.agents[o.agent_id]
            rows.append(dict(choice=c, sampled=a, sampled_score=info['scores'].get(a), n_minus=sum(v < 0 for v in info['scores'].values()), n_kill=sum(v == 1.0 for v in info['scores'].values()), d_before=abs(me.row-o.row)+abs(me.col-o.col),
                d_after=abs(me.row+dr-o.row)+abs(me.col+dc-o.col),
                phys_before=surv(env, o, False), phys_after=surv(moved, mo, False),
                rb_before=surv(env, o, True), rb_after=surv(moved, mo, True),
                rb_act_before=rb_action(env, o.agent_id), rb_act_after=rb_action(moved, o.agent_id),
                opp_in_danger=bool(env._danger_map()[o.row, o.col] > 0), my_bomb=any(b.owner == aid for b in env.bombs)))
        return c
    evaluate.evaluate_seeded(agent, 9, 2, games, seed, 500, sudden_death=True, sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
    return rows
if __name__ == '__main__':
    with ProcessPoolExecutor(4) as pool:
        rows = [r for part in pool.map(run, [15000, 15025, 15050, 15075]) for r in part]
    json.dump(rows, open(__file__.replace('.py', '.json'), 'w'))
    n = len(rows); f = lambda k: sum(r[k] for r in rows)
    print('move-kills', n)
    print('opp in danger', f('opp_in_danger'), '| my bomb live', f('my_bomb'))
    print('physically survivable before', f('phys_before'), 'after', f('phys_after'))
    print('RuleBot-view survivable before', f('rb_before'), 'after', f('rb_after'))
    print('RuleBot action changes because of the move', sum(r['rb_act_before'] != r['rb_act_after'] for r in rows))
    from collections import Counter
    print('score of F sampled action', Counter(r['sampled_score'] for r in rows), '| actions scoring -1 per state', Counter(r['n_minus'] for r in rows), '| killing actions per state', Counter(r['n_kill'] for r in rows))
    print('distance before->after', Counter((r['d_before'], r['d_after']) for r in rows).most_common(8))
