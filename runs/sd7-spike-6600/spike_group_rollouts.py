"""SD7 spike (diagnostic seeds 6600-6649, no training): is there learnable variance at decisive states?

F plays 100 games (lich E). Candidate state: a bomb is live, the opponent is in the danger map and
within Manhattan 4 of the learner, >= 8 steps after the last kept state; kept: up to 3 per game
where kill_search finds a learner kill, plus 1 control per game where it does not (v2; v1 kept the
first 3 candidates and found almost no kill states). Each candidate is labelled
by kill_search (a learner-credited kill exists within 12 ticks or not), then F replays it K=8 times
for H=20 ticks against RuleBot (own torch/numpy seeds per replay). Outcome per replay: +1 learner's
bomb kills the opponent while the learner lives, -1 learner dies, 0 otherwise.
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(r'D:\fpt\subject\BomberMan'); sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
K, H, PER_GAME, GAP = 8, 20, 3, 8
E = dict(sudden_death=True, sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)

def replay(env0, aid, policy, rule_fns, seed):
    import numpy as np, torch
    from copy import deepcopy
    env = deepcopy(env0)
    np.random.seed(seed); torch.manual_seed(seed)
    obs = [env._get_obs(a.agent_id) for a in env.agents]
    for _ in range(H):
        acts = [0] * env.num_agents
        for a in env.agents:
            if a.alive:
                acts[a.agent_id] = policy(env, obs, aid) if a.agent_id == aid else rule_fns[a.agent_id](env, obs, a.agent_id)
        obs, _, _, info = env.step(acts)
        if any(k == aid and v != aid for k, v in info['kills']) and env.agents[aid].alive:
            return 1
        if not env.agents[aid].alive:
            return -1
        if info['episode_done']:
            return 0
    return 0

def run(seed, games):
    import numpy as np, torch; torch.set_num_threads(1)
    from copy import deepcopy
    import evaluate
    from kill_search import search_action
    from train_actor_critic import get_action_mask
    policy = evaluate.make_ac_fn(str(ROOT / 'runs/phaseC-sd-best-fallback/best_phaseC.pt'), 9)
    rule_fns = [evaluate.make_rule_based_fn(i) for i in range(2)]
    states, last = [], {}
    def agent(env, obs, aid):
        key = (env.seed if hasattr(env, 'seed') else id(env), aid)
        opp = [a for a in env.agents if a.agent_id != aid][0]
        me = env.agents[aid]
        kills_taken = sum(s['game'] == key and s['kill_available'] for s in states)
        controls_taken = sum(s['game'] == key and not s['kill_available'] for s in states)
        if (env.bombs and opp.alive and env._danger_map()[opp.row, opp.col] > 0
                and abs(opp.row - me.row) + abs(opp.col - me.col) <= 4
                and env.step_count - last.get(key, -99) >= GAP):
            mask = get_action_mask(env, aid)
            _, info = search_action(env, aid, mask, 0)
            kill = (not info['skipped']) and max(info['scores'].values()) == 1.0
            if (kill and kills_taken < PER_GAME) or (not kill and controls_taken < 1):
                last[key] = env.step_count
                snap = deepcopy(env)
                rng = (np.random.get_state(), torch.get_rng_state())
                outcomes = [replay(snap, aid, policy, rule_fns, 1000 * len(states) + k + seed) for k in range(K)]
                np.random.set_state(rng[0]); torch.set_rng_state(rng[1])
                states.append(dict(game=key, step=env.step_count, kill_available=kill, outcomes=outcomes))
        return policy(env, obs, aid)
    evaluate.evaluate_seeded(agent, 9, 2, games, seed, 500, **E)
    for s in states:
        s['game'] = list(map(str, s['game']))
    return states

if __name__ == '__main__':
    with ProcessPoolExecutor(4) as pool:
        states = [s for part in pool.map(run, [6600, 6612, 6625, 6637], [24, 26, 24, 26]) for s in part]
    (OUT / 'states.json').write_text(json.dumps(states), encoding='utf-8')
    import numpy as np
    def report(name, sel):
        if not sel:
            return
        wins = np.array([sum(o == 1 for o in s['outcomes']) for s in sel])
        deaths = np.array([sum(o == -1 for o in s['outcomes']) for s in sel])
        mixed = ((wins > 0) & (wins < K)).mean()
        learn = ((wins >= 1) & (wins <= K - 1) & (wins / K >= 0.1) & (wins / K <= 0.9)).mean()
        print(f'{name}: n={len(sel)} | success rate mean {wins.mean() / K:.3f} | states with 0/{K} wins {(wins == 0).mean():.2f}, '
              f'{K}/{K} {(wins == K).mean():.2f}, mixed {mixed:.2f} | replays dying {deaths.sum() / (len(sel) * K):.3f} | '
              f'win-count histogram {np.bincount(wins, minlength=K + 1).tolist()}')
    print('states', len(states), 'per game', len(states) / 100)
    report('all candidates', states)
    report('kill available (search)', [s for s in states if s['kill_available']])
    report('no kill per search', [s for s in states if not s['kill_available']])
