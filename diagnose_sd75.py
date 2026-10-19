"""SD7.6 diagnosis: why SD7.5 (group advantage) gained little ActiveWR. No training.

F plays seeds 6800-6849 (100 games, schedule E). At every learner step with an opponent within
kill_search.REACH, run kill-search; for each state where a kill exists, record whether the SD7
decisive filter (live bomb, opponent in danger within 4) would have kept it, whether the kill
needs the bomb or a move/STOP, and the probability F / SD7.5 best / SD7.5 final put on the kill
actions. Policy entropy is averaged over every learner step.
  H-coverage: most kill states fail the cheap filter, so group RL never trains on them.
  H-explore:  full raises P(kill) on bomb kills but not on move kills (F ~0.001).
  H-sharpen:  entropy falls F > best > final (validation dropped after ep 1000).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from progress import parallel_map

ROOT = Path(__file__).resolve().parent
MODELS = {'F': ROOT / 'runs/phaseC-sd-best-fallback/best_phaseC.pt',
          'best': ROOT / 'runs/sd75-20261018/full/best_phaseC.pt',
          'final': ROOT / 'runs/sd75-20261018/full/final_phaseC.pt'}
SEED, GAMES = 6800, 100
SCHEDULE_E = dict(sudden_death=True, sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)


def shard(seed, games, max_steps, ticks):
    import torch
    torch.set_num_threads(1)
    import evaluate
    import group_rl
    import kill_search
    from bomberman_env import ACTION_BOMB, ACTION_STOP
    from run_sd7_pilot import _net
    from train_actor_critic import get_action_mask
    nets = {k: _net(p) for k, p in MODELS.items()}
    policy = evaluate.make_ac_fn(str(MODELS['F']), 9)
    states, entropy, steps, game = [], {k: 0.0 for k in nets}, [0], {'env': None, 'index': -1}

    def agent(env, obs, aid):
        if game['env'] is not env:
            game.update(env=env, index=game['index'] + 1)
        mask = get_action_mask(env, aid)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs[aid], dtype=np.float32)).unsqueeze(0)
            m = torch.tensor(mask, dtype=torch.bool)
            probs = {k: torch.softmax(n(x)[0].masked_fill(~m, -1e9), -1)[0].numpy() for k, n in nets.items()}
        for k, p in probs.items():
            entropy[k] += float(-(p * np.log(p + 1e-12)).sum())
        steps[0] += 1
        _, info = kill_search.search_action(env, aid, mask, ACTION_STOP)
        if not info['skipped'] and max(info['scores'].values(), default=0.0) == 1.0:
            me, danger = env.agents[aid], env._danger_map()
            cheap = bool(env.bombs) and any(
                a.alive and a.agent_id != aid and danger[a.row, a.col] > 0
                and abs(a.row - me.row) + abs(a.col - me.col) <= group_rl.REACH for a in env.agents)
            kill = [a for a, s in info['scores'].items() if s == 1.0]
            states.append({'game': game['index'], 'step': env.step_count,
                           'cheap': cheap, 'bomb_kill': ACTION_BOMB in kill, 'n_kill': len(kill),
                           'p_kill': {k: float(p[kill].sum()) for k, p in probs.items()}})
        return policy(env, obs, aid)

    def advance(done, total):
        if done:
            ticks.put(1)
    result = evaluate.evaluate_seeded(agent, 9, 2, games, seed, max_steps, on_progress=advance, **SCHEDULE_E)
    games_out = [{k: g[k] for k in ('seed', 'aid', 'win', 'active_win', 'passive_win', 'draw')} for g in result['games']]
    ticks.put(1)
    return {'states': states, 'entropy': entropy, 'steps': steps[0], 'games': games_out}


def summarize(parts):
    states = [s for p in parts for s in p['states']]
    steps = sum(p['steps'] for p in parts)
    out = {'games': sum(len(p['games']) for p in parts), 'learner_steps': steps, 'kill_states': len(states),
           'entropy': {k: sum(p['entropy'][k] for p in parts) / steps for k in MODELS},
           'active_win_F': 100 * float(np.mean([g['active_win'] for p in parts for g in p['games']]))}
    for name, sel in (('all', lambda s: True), ('cheap_pass', lambda s: s['cheap']), ('cheap_fail', lambda s: not s['cheap']),
                      ('bomb_kill', lambda s: s['bomb_kill']), ('move_kill', lambda s: not s['bomb_kill']),
                      ('cheap_pass_move', lambda s: s['cheap'] and not s['bomb_kill']),
                      ('cheap_fail_bomb', lambda s: not s['cheap'] and s['bomb_kill'])):
        group = [s for s in states if sel(s)]
        out[name] = {'n': len(group), **{f'p_kill_{k}': float(np.mean([s['p_kill'][k] for s in group])) if group else None
                                         for k in MODELS}}
    return out


def main(output, workers=4, smoke=False):
    output = Path(output)
    output.mkdir(parents=True)  # FileExistsError: never overwrite
    games, steps = (4, 60) if smoke else (GAMES, 500)
    cuts = [games // 2 * i // workers for i in range(workers + 1)]
    jobs = [(SEED + cuts[i], 2 * (cuts[i + 1] - cuts[i]), steps) for i in range(workers) if cuts[i + 1] > cuts[i]]
    parts = parallel_map(shard, jobs, total=games + len(jobs), desc='SD7.6 diagnosis', workers=workers)
    (output / 'raw.json').write_text(json.dumps(parts), encoding='utf-8')
    summary = summarize(parts)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    main(a.output, a.workers, a.smoke)
