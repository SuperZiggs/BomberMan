"""Replay six shared timeouts without changing checkpoint weights or policy RNG."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bomberman_env import BombermanFFAEnv, ACTION_BOMB, _MOVE_DELTA
from evaluate import make_ac_fn, make_rule_based_fn
from train_actor_critic import get_action_mask, get_attack_action
from rule_based_bot import _bomb_escape_pressure

torch.set_num_threads(1)
folder = Path(__file__).parent
available = json.loads((folder / 'diagnostic_cases_17000.json').read_text())['common_timeouts']
cases = [g for aid in (0, 1) for g in [x for x in available if x['aid'] == aid][:3]]
assert len(cases) == 6
paths = {'control': ROOT / 'runs/pressure-C-next/final_phaseC.pt', 'candidate': folder / 'best_phaseC.pt'}
hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
output = {'sample': 'First three shared timeouts per corner, not an overall benchmark',
          'model_sha256': hashes, 'games': [], 'totals': {}}
for label, path in paths.items():
    policy = make_ac_fn(str(path), 9, 'cpu')
    totals = Counter()
    for case in cases:
        seed, aid = case['seed'], case['aid']
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed)
        obs = env.reset()
        bot = make_rule_based_fn(1 - aid)
        counts = Counter()
        bomb_events = []
        self_death = False
        for tick in range(500):
            agent = env.agents[aid]
            enemy = env.agents[1 - aid]
            mask = get_action_mask(env, aid)
            danger = env._danger_map()
            actions = [0, 0]
            for index in range(2):
                actions[index] = policy(env, obs, index) if index == aid else bot(env, obs, index)
            counts['steps'] += 1
            counts['close_steps'] += int(abs(agent.row - enemy.row) + abs(agent.col - enemy.col) <= agent.blast_range + 1)
            if danger[agent.row, agent.col] == 0:
                counts['quiet_steps'] += 1
                target = get_attack_action(env, aid, danger, mask)
                if target in _MOVE_DELTA:
                    counts['teacher_moves'] += 1
                    counts['teacher_moves_taken'] += int(actions[aid] == target)
                blast = env._blast_cells(agent.row, agent.col, agent.blast_range)
                if mask[ACTION_BOMB] and (enemy.row, enemy.col) in blast:
                    counts['direct_bomb_opportunities'] += 1
                    counts['direct_bombs_taken'] += int(actions[aid] == ACTION_BOMB)
            else:
                counts['danger_steps'] += 1
            if actions[aid] == ACTION_BOMB:
                pressure = _bomb_escape_pressure(env, aid, agent.row, agent.col)
                kind = 'zero' if pressure <= 0 else 'full' if pressure == 1 else 'partial'
                counts['bomb_pressure_' + kind] += 1
                bomb_events.append({'step': tick + 1, 'agent': [agent.row, agent.col],
                                    'enemy': [enemy.row, enemy.col], 'pressure': pressure,
                                    'range': agent.blast_range, 'ammo': agent.ammo})
            obs, _, _, info = env.step(actions)
            counts['bombs'] += int(aid in info['bombs_placed'])
            self_death |= aid in info.get('self_blast_deaths', []) or (aid, aid) in info['kills']
            if info['episode_done']:
                break
        actual = {'win': int(agent.alive and not enemy.alive),
                  'active_win': int(agent.alive and not enemy.alive and agent.kills > 0),
                  'suicide': int(self_death), 'draw': int(sum(info['alive']) != 1),
                  'kills': agent.kills, 'bombs': counts['bombs'], 'steps': info['step']}
        assert all(actual[k] == case[label][k] for k in actual), (label, seed, aid, actual, case[label])
        output['games'].append({'model': label, 'seed': seed, 'aid': aid, 'result': actual,
                                'counts': dict(counts), 'bomb_events': bomb_events,
                                'final_grid': env.grid.tolist()})
        totals.update(counts)
        print(label, seed, aid, json.dumps(dict(counts)), flush=True)
    output['totals'][label] = dict(totals)
assert all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name] for name, path in paths.items())
(folder / 'replay_timeouts_17000.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
print('TOTALS', json.dumps(output['totals']), flush=True)
