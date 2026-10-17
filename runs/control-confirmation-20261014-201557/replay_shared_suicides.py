"""Diagnose the two already-viewed deaths with frozen policies; no new test seeds."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8-sig'))
for name, expected in protocol['source_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
for checkpoint in protocol['checkpoints'].values():
    assert hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest() == checkpoint['sha256']
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from bomberman_env import BombermanFFAEnv, _MOVE_DELTA
from evaluate import make_ac_fn, make_rule_based_fn
from rule_based_bot import _survival_map
from train_actor_critic import get_action_mask

torch.set_num_threads(1)


def state(env):
    return {'grid': env.grid.tolist(), 'agents': [asdict(a) for a in env.agents],
            'bombs': [asdict(b) for b in env.bombs],
            'flames': [{'position': list(pos), 'ticks': duration, 'sources': env._flame_sources.get(pos, {})}
                       for pos, duration in env.flames.items()]}


def known_safe_moves(env, aid):
    branch = deepcopy(env)
    # This diagnostic removes opponent occupancy, not its already-placed bombs.
    for agent in branch.agents:
        if agent.agent_id != aid:
            agent.alive = False
    agent = branch.agents[aid]
    safe, blocked, _ = _survival_map(branch, agent.row, agent.col, avoid_contests=False, safety_margin=0)
    moves = [0] if safe[0, agent.row, agent.col] else []
    for action, (dr, dc) in _MOVE_DELTA.items():
        nr, nc = agent.row + dr, agent.col + dc
        if 0 <= nr < 9 and 0 <= nc < 9 and not blocked[0, nr, nc] and safe[0, nr, nc]:
            moves.append(action)
    return moves


replays = []
for model, checkpoint in protocol['checkpoints'].items():
    policy = make_ac_fn(checkpoint['path'], 9, 'cpu')
    evaluation = json.loads((OUT / f'evaluation_{model}_32000.json').read_text())
    selected = [g for g in evaluation['games'] if g['suicide']]
    assert {(g['seed'], g['aid']) for g in selected} == {(32080, 0), (32133, 0)}
    for expected in selected:
        seed, aid = expected['seed'], expected['aid']
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed)
        obs = env.reset()
        opponent = make_rule_based_fn(1 - aid)
        tape, bombs, items, suicide = [], 0, 0, False
        for tick in range(500):
            before = state(env)
            fingerprint = pickle.dumps(env.__dict__), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()
            mask = get_action_mask(env, aid)
            safe_moves = known_safe_moves(env, aid)
            assert pickle.dumps(env.__dict__) == fingerprint[0]
            assert pickle.dumps(np.random.get_state()) == fingerprint[1]
            assert torch.equal(torch.get_rng_state(), fingerprint[2])
            actions = [0, 0]
            for index in range(2):
                actions[index] = policy(env, obs, aid) if index == aid else opponent(env, obs, index)
            assert mask[actions[aid]]
            obs, _, _, info = env.step(actions)
            suicide |= aid in info['self_blast_deaths'] or (aid, aid) in info['kills']
            bombs += int(aid in info['bombs_placed'])
            items += len(info['items_collected'].get(aid, []))
            tape.append({'tick': tick, 'before': before, 'mask': mask, 'known_safe_moves_without_opponent_occupancy': safe_moves,
                         'actions': actions, 'info': info, 'after': state(env)})
            if info['episode_done']:
                break
        agent = env.agents[aid]
        win = int(agent.alive and not env.agents[1 - aid].alive)
        result = {'seed': seed, 'aid': aid, 'win': win, 'active_win': int(win and agent.kills > 0),
                  'suicide': int(suicide), 'draw': int(sum(info['alive']) != 1), 'kills': agent.kills,
                  'bombs': bombs, 'items': items, 'steps': env.step_count}
        assert result == expected, (model, result, expected)
        record = {'model': model, **result, 'first_no_known_safe_move_tick': next((r['tick'] for r in tape
                  if not r['known_safe_moves_without_opponent_occupancy']), None), 'tape': tape}
        replays.append(record)
        print(f'{model} seed{seed}/A{aid}: replay outcome matches; death step{env.step_count}', flush=True)
(OUT / 'shared_suicide_replays.json').write_text(json.dumps(replays, indent=2) + '\n', encoding='utf-8')
summary = [{k: v for k, v in r.items() if k != 'tape'} for r in replays]
for index in (0, 1):
    assert [r['actions'] for r in replays[index]['tape']] == [r['actions'] for r in replays[index + 2]['tape']]
(OUT / 'shared_suicide_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
