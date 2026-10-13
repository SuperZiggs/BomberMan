"""Replay the existing test cases without changing policy RNG or original artifacts."""
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from bomberman_env import BombermanFFAEnv, ACTION_BOMB, ACTION_STOP
from evaluate import make_ac_fn, make_rule_based_fn
from rule_based_bot import _bfs_toward_opponent, _can_bomb_and_escape
from train_actor_critic import get_action_mask, get_attack_action

torch.set_num_threads(1)
folder = Path(__file__).parent
snapshot = folder / 'before'
snapshot.mkdir(exist_ok=True)
for name in ('bomberman_env.py', 'rule_based_bot.py', 'train_actor_critic.py', 'evaluate.py'):
    target = snapshot / name
    if not target.exists():
        target.write_bytes((ROOT / name).read_bytes())
checkpoint = ROOT / 'runs/20261010-135449-587798/best_phaseC.pt'
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
policy = make_ac_fn(str(checkpoint), 9, 'cpu')
old = json.loads((ROOT/'runs/escape-20261011/test_baseline.json').read_text())['games']
selected = []
for aid in (0, 1):
    selected += [g for g in old if g['aid'] == aid and g['draw'] and g['steps'] == 500][:10]
selected += [g for g in old if g['suicide']][:10]
assert len(selected) == 30
games = []
for idx, expected in enumerate(selected):
    seed, aid = expected['seed'], expected['aid']
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed)
    obs = env.reset()
    bot = make_rule_based_fn(1-aid)
    counts = Counter()
    trace = []
    recent = deque(maxlen=20)
    suicide = False
    bombs = 0
    for tick in range(500):
        a, enemy = env.agents[aid], env.agents[1-aid]
        danger = env._danger_map()
        mask = get_action_mask(env, aid)
        teacher = get_attack_action(env, aid, danger, mask)
        ray = env._blast_cells(a.row, a.col, a.blast_range)
        inline = (enemy.row, enemy.col) in ray
        quiet = danger[a.row, a.col] == 0
        approach = _bfs_toward_opponent(env, aid, a.row, a.col, danger) if quiet else None
        self_safe = _can_bomb_and_escape(env, aid, a.row, a.col, danger, robust=False) if quiet and inline else None
        actions = [0, 0]
        for i in range(2):
            actions[i] = policy(env, obs, i) if i == aid else bot(env, obs, i)
        action = actions[aid]
        counts['steps'] += 1
        counts['danger_steps' if not quiet else 'quiet_steps'] += 1
        counts['bomb_allowed'] += int(mask[ACTION_BOMB])
        counts['inline_quiet'] += int(inline and quiet)
        counts['inline_bomb_allowed'] += int(inline and quiet and mask[ACTION_BOMB])
        counts['inline_self_safe_strict_blocked'] += int(inline and quiet and bool(self_safe) and not mask[ACTION_BOMB])
        counts['teacher_none_quiet'] += int(quiet and teacher is None)
        counts['approach_masked'] += int(approach is not None and not mask[approach])
        counts['teacher_chosen'] += int(teacher is not None and action == teacher)
        counts['attack_opportunity_taken'] += int(quiet and inline and mask[ACTION_BOMB] and action == ACTION_BOMB)
        counts['stop_quiet'] += int(quiet and action == ACTION_STOP)
        counts['near_enemy'] += int(abs(a.row-enemy.row)+abs(a.col-enemy.col) <= 3)
        record = dict(step=tick+1, agents=[[b.row,b.col] for b in env.agents],
                      action=action, bot_action=actions[1-aid], mask=mask, teacher=teacher,
                      danger=int(danger[a.row,a.col]), inline=inline,
                      bombs=[[b.row,b.col,b.owner,b.timer,b.blast_range] for b in env.bombs])
        recent.append(record)
        if action == ACTION_BOMB or (quiet and inline and mask[ACTION_BOMB]) or tick >= 450:
            trace.append(record)
        obs, rewards, dones, info = env.step(actions)
        bombs += aid in info['bombs_placed']
        suicide |= (aid,aid) in info['kills']
        if aid in info['bombs_placed']:
            counts['bombs_at_current_enemy'] += int(inline)
            counts['bombs_with_wood'] += int(any(env.grid[pos] == 2 for pos in ray))
        if info['episode_done']:
            break
    win = int(env.agents[aid].alive and not env.agents[1-aid].alive)
    actual = dict(win=win, active_win=int(win and env.agents[aid].kills > 0),
                  suicide=int(suicide), draw=int(sum(info['alive']) != 1),
                  kills=env.agents[aid].kills, bombs=bombs, steps=info['step'])
    assert all(actual[k] == expected[k] for k in actual), (seed,aid,actual,expected)
    games.append(dict(seed=seed, aid=aid, kind='suicide' if suicide else 'timeout',
                      result=actual, counts=dict(counts), attack_trace=trace,
                      tail=list(recent), final_grid=env.grid.tolist()))
    print('Replay', idx+1, '/ 30', seed, aid, dict(counts), flush=True)
summary = {}
for kind in ('timeout','suicide'):
    rows = [g for g in games if g['kind'] == kind]
    total = Counter()
    for g in rows:
        total.update(g['counts'])
    summary[kind] = dict(num_games=len(rows), **total)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
(folder/'diagnostics.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
    checkpoint_sha256=digest,summary=summary,games=games),indent=2),encoding='utf-8')
print('SUMMARY',json.dumps(summary),flush=True)
