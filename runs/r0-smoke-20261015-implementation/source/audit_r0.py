"""R0: fixed-policy reward audit on the already-used validation boards; no training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch

from audit_t2 import digest, dump
from bomberman_env import BombermanFFAEnv, ACTION_BOMB, ACTION_STOP, WOODEN_WALL
from evaluate import make_rule_based_fn
from pilot_two_bomb import stage_timer
from rule_based_bot import _bfs_toward_opponent, _bomb_escape_pressure
from train_actor_critic import (ActorCriticNet, SnapshotOpponent, smart_load_weights,
    get_action_mask, _shaped_reward, _reward_components)
from visualize import capture_state

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / 'runs/gae-full-20261015-125942'
B = ROOT / 'runs/pressure-C-next/final_phaseC.pt'
B_SHA = '93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60'
COMPONENTS = ('time', 'item', 'wall', 'owned_kill', 'active_win', 'passive_win_env',
              'passive_win_cancellation', 'death', 'self_blast')
GROUPS = ('active_win', 'passive_win', 'timeout', 'self_blast', 'enemy_death')
GAME_FIELDS = ('seed', 'aid', 'win', 'active_win', 'suicide', 'draw', 'kills', 'bombs', 'items', 'steps')
SOURCES = ('audit_r0.py', 'test_reward_audit.py', 'train_actor_critic.py', 'bomberman_env.py',
    'rule_based_bot.py', 'evaluate.py', 'visualize.py', 'audit_t2.py', 'pilot_two_bomb.py',
    'tactical_teacher.py', 'combat_move_teacher.py')


def observe(env, aid, mask, action):
    """Descriptive proxies only; current-position pressure is not a future-kill guarantee."""
    agent, enemy = env.agents[aid], env.agents[1 - aid]
    pos, foe = (agent.row, agent.col), (enemy.row, enemy.col)
    approach = _bfs_toward_opponent(env, aid, *pos, env._danger_map(), action_mask=mask)
    blast = env._blast_cells(*pos, agent.blast_range) if action == ACTION_BOMB else []
    return {'distance': abs(pos[0] - foe[0]) + abs(pos[1] - foe[1]),
        'position': pos, 'enemy_position': foe, 'approach_action': approach,
        'pressure': _bomb_escape_pressure(env, aid, *pos) if action == ACTION_BOMB else None,
        'projected_wood': int(sum(env.grid[p] == WOODEN_WALL for p in blast)),
        'enemy_in_blast': foe in blast}


@torch.no_grad()
def play_game(net, seed, aid, max_steps, stop, observed=True):
    numpy_state, training = np.random.get_state(), net.training
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=max_steps, seed=seed)
            obs = env.reset()
            policy, opponent = SnapshotOpponent(net, 'cpu'), make_rule_based_fn(1 - aid)
            tape, frames, totals, stats = [], [], Counter(), Counter()
            rewards, env_rewards, distances = [], [], []
            first_close, max_error = None, 0.0
            labels = ['CNN' if i == aid else 'RuleBot' for i in range(2)]
            for _ in range(max_steps):
                if stop.is_set():
                    return None
                actions = [ACTION_STOP, ACTION_STOP]
                was_alive = env.agents[aid].alive
                for index in range(2):
                    if env.agents[index].alive:
                        actions[index] = (policy.select_action(obs[index], env, index) if index == aid
                                          else opponent(env, obs, index))
                if observed and was_alive:
                    mask = list(map(bool, get_action_mask(env, aid)))
                    seen = observe(env, aid, mask, actions[aid])
                    frames.append(capture_state(env, labels, actions))
                    distances.append(seen['distance'])
                    if seen['distance'] <= 3 and first_close is None:
                        first_close = env.step_count
                    stats['close_steps'] += seen['distance'] <= 3
                    stats['approach_available_steps'] += seen['approach_action'] is not None
                    stats['approach_chosen_steps'] += seen['approach_action'] is not None and actions[aid] == seen['approach_action']
                    stats['mask_violations'] += not mask[actions[aid]]
                obs, raw, _, info = env.step(actions)
                stats['bombs'] += aid in info['bombs_placed']
                stats['items'] += len(info['items_collected'].get(aid, []))
                stats['suicide'] += aid in info['self_blast_deaths'] or (aid, aid) in info['kills']
                if observed and was_alive:
                    shaped = _shaped_reward(raw[aid], aid, info, env.agents[aid].kills)
                    parts = _reward_components(raw[aid], aid, info, env.agents[aid].kills, was_alive)
                    error = abs(sum(parts.values()) - shaped)
                    max_error = max(max_error, error)
                    totals.update(parts)
                    rewards.append(shaped)
                    env_rewards.append(raw[aid])
                    placed = aid in info['bombs_placed']
                    if placed:
                        stats['pressure_bombs'] += seen['pressure'] is not None and seen['pressure'] > 0
                        stats['wood_only_bombs'] += not (seen['pressure'] is not None and seen['pressure'] > 0) and seen['projected_wood'] > 0
                        stats['neither_bombs'] += not (seen['pressure'] is not None and seen['pressure'] > 0) and seen['projected_wood'] == 0
                        stats['enemy_in_blast_bombs'] += seen['enemy_in_blast']
                    stats['walls_all'] += info['walls_destroyed'].get(aid, 0)
                    stats['walls_rewarded'] += info['walls_destroyed'].get(aid, 0) if info['alive'][aid] else 0
                    after = env.agents[aid].row, env.agents[aid].col
                    foe = seen['enemy_position']
                    stats['moved_closer_steps'] += after != seen['position'] and (
                        abs(after[0] - foe[0]) + abs(after[1] - foe[1]) < seen['distance'])
                    tape.append({'step': info['step'], 'actions': actions, 'mask': mask, **seen,
                        'position_after': after, 'info': info, 'env_reward': raw[aid],
                        'shaped_reward': shaped, 'reward_components': parts, 'reward_error': error})
                if info['episode_done']:
                    break
            alive = info['alive']
            win = int(alive[aid] and sum(alive) == 1)
            row = {'seed': seed, 'aid': aid, 'win': win,
                'active_win': int(win and env.agents[aid].kills > 0),
                'suicide': int(stats['suicide'] > 0), 'draw': int(sum(alive) != 1),
                'kills': env.agents[aid].kills, 'bombs': stats['bombs'], 'items': stats['items'], 'steps': info['step']}
            if observed:
                frames.append(capture_state(env, labels, actions, raw))
                timeout = int(sum(alive) > 1 and info['step'] >= max_steps)
                group = ('active_win' if row['active_win'] else 'passive_win' if win else
                         'timeout' if timeout else 'self_blast' if row['suicide'] else 'enemy_death')
                total = sum(rewards)
                assert math.isclose(sum(totals.values()), total, rel_tol=0, abs_tol=1e-10)
                row.update(outcome=group, timeout=timeout, mutual_death=int(sum(alive) == 0),
                    learner_steps=len(tape), env_reward=sum(env_rewards), shaped_reward=total,
                    reward_components={k: totals[k] for k in COMPONENTS}, component_max_error=max_error,
                    initial_distance=distances[0], min_distance=min(distances),
                    first_close_step=first_close, **{k: stats[k] for k in (
                        'close_steps', 'approach_available_steps', 'approach_chosen_steps', 'moved_closer_steps',
                        'mask_violations', 'pressure_bombs', 'wood_only_bombs', 'neither_bombs',
                        'enemy_in_blast_bombs', 'walls_all', 'walls_rewarded')})
            return row, tape, frames
    finally:
        np.random.set_state(numpy_state)
        net.train(training)


def distribution(values):
    if not values:
        return None
    return {name: float(value) for name, value in zip(('mean', 'median', 'p95', 'min', 'max'),
        (np.mean(values), np.median(values), np.percentile(values, 95), min(values), max(values)))}


def describe(rows):
    n = len(rows)
    if not n:
        return {'games': 0}
    components = {k: distribution([r['reward_components'][k] for r in rows]) for k in COMPONENTS}
    absolute = {k: sum(abs(r['reward_components'][k]) for r in rows) for k in COMPONENTS}
    denom = sum(absolute.values())
    # Gross passive +10/-10 are reported separately; net shares remove that cancelling pair.
    net_absolute = {k: v for k, v in absolute.items() if k not in ('passive_win_env', 'passive_win_cancellation')}
    net_denom = sum(net_absolute.values())
    fields = ('shaped_reward', 'env_reward', 'kills', 'bombs', 'items', 'steps', 'walls_all', 'walls_rewarded',
              'initial_distance', 'min_distance', 'close_steps', 'approach_available_steps',
              'approach_chosen_steps', 'moved_closer_steps', 'pressure_bombs', 'wood_only_bombs', 'neither_bombs')
    return {'games': n, **{k + '_rate': 100 * sum(r[k] for r in rows) / n for k in ('win', 'active_win', 'draw', 'suicide')},
        'stats': {k: distribution([r[k] for r in rows]) for k in fields}, 'components': components,
        'component_sums': {k: sum(r['reward_components'][k] for r in rows) for k in COMPONENTS},
        'gross_abs_shares': {k: v / denom if denom else None for k, v in absolute.items()},
        'net_abs_shares': {k: v / net_denom if net_denom else None for k, v in net_absolute.items()},
        'positive_reward_games': sum(r['shaped_reward'] > 0 for r in rows),
        'reached_close_games': sum(r['first_close_step'] is not None for r in rows),
        'pressure_bomb_fraction': sum(r['pressure_bombs'] for r in rows) / sum(r['bombs'] for r in rows) if sum(r['bombs'] for r in rows) else None,
        'mask_violations': sum(r['mask_violations'] for r in rows),
        'max_component_error': max(r['component_max_error'] for r in rows)}


def plot_results(output, models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = [{**r, 'model': name} for name, rows in models.items() for r in rows]
    columns = ('model', 'seed', 'aid', 'outcome', 'kills', 'active_win', 'shaped_reward')
    with (output / 'figure_data.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in data)
    palette = {'active_win': ('#009E73', '^'), 'passive_win': ('#0072B2', 's'),
        'timeout': ('#D55E00', 'o'), 'self_blast': ('#000000', 'X'), 'enemy_death': ('#882255', 'v')}
    with plt.rc_context({'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True, layout='constrained')
        for column, (name, rows) in enumerate(models.items()):
            for group, (color, marker) in palette.items():
                subset = [r for r in rows if r['outcome'] == group]
                jitter = np.array([((r['seed'] % 13) - 6) * .005 + (.005 if r['aid'] else -.005) for r in subset])
                for index, key in enumerate(('kills', 'active_win')):
                    axes[index, column].scatter(np.array([r[key] for r in subset]) + jitter,
                        [r['shaped_reward'] for r in subset], label=group, color=color, marker=marker,
                        s=32, alpha=.65, edgecolors='black', linewidths=.4)
            axes[0, column].set_title(f'{name} (n={len(rows)} games)')
            for index, label in enumerate(('Owned kills', 'Active win (0/1)')):
                axes[index, column].set_xlabel(label + '\nDisplay jitter ≤0.035')
                axes[index, column].axhline(0, color='#666666', linewidth=.7, linestyle='--')
                axes[index, column].set_xticks((0, 1))
                axes[index, column].grid(axis='y', alpha=.2)
        axes[0, 0].set_ylabel('Shaped reward per learner trajectory')
        axes[1, 0].set_ylabel('Shaped reward per learner trajectory')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=5)
        fig.suptitle('R0: fixed-policy validation diagnostics (not training curves)')
        fig.savefig(output / 'reward_scatter.png', dpi=300, facecolor='white')
        fig.savefig(output / 'reward_scatter.svg', facecolor='white')
        plt.close(fig)
    dump(output / 'figure_manifest.json', {'data_sha256': digest(output / 'figure_data.csv'),
        'points': len(data), 'unit': 'one game/corner; two corners share a board seed',
        'transforms': 'y unchanged; deterministic x jitter at most0.035, solely to expose overlap',
        'exclusions': [], 'uncertainty': 'none; descriptive plot on reused validation boards',
        'alt': 'Shaped reward versus owned kills and active-win indicator, split by checkpoint and outcome.',
        'matplotlib': matplotlib.__version__, 'files': ['reward_scatter.png', 'reward_scatter.svg']})


def validation_parity(rows, expected):
    for key in ('win', 'active_win', 'draw', 'suicide'):
        assert math.isclose(100 * sum(r[key] for r in rows) / len(rows), expected[key + '_rate'], abs_tol=1e-9)
    for key in ('kills', 'bombs', 'items', 'steps'):
        assert math.isclose(sum(r[key] for r in rows) / len(rows), expected['avg_' + key], abs_tol=1e-9)


def run(output, smoke=False, budget_minutes=60):
    if not 0 < budget_minutes <= 60:
        raise ValueError('Budget must be in (0,60] minutes')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    parent = json.loads((PARENT / 'summary.json').read_text(encoding='utf-8'))
    parent_config = json.loads((PARENT / 'config.json').read_text(encoding='utf-8'))
    if parent['status'] != 'complete' or digest(B) != B_SHA:
        raise ValueError('Parent run or B does not match the registered audit')
    models = {'original': {'path': str(B), 'sha256': B_SHA},
        'mc_final': parent['arms']['control']['checkpoints']['final'],
        'gae_final': parent['arms']['gae']['checkpoints']['final']}
    for record in models.values():
        if digest(Path(record['path'])) != record['sha256']:
            raise ValueError('A checkpoint changed')
    for name in ('bomberman_env.py', 'rule_based_bot.py', 'evaluate.py'):
        if digest(ROOT / name) != parent_config['source_sha256'][name]:
            raise ValueError('Validation runtime changed since B1')
    expected = {}
    for name, arm, episode in (('original', 'control', 0), ('mc_final', 'control', 200), ('gae_final', 'gae', 200)):
        vals = [json.loads(line) for line in (PARENT / arm / 'validation_metrics.jsonl').read_text().splitlines()]
        expected[name] = next(v for v in vals if v['episode'] == episode)
    parent_inputs = {str(path): digest(path) for path in (PARENT / 'summary.json', PARENT / 'config.json',
        PARENT / 'control/validation_metrics.jsonl', PARENT / 'gae/validation_metrics.jsonl')}
    torch.set_num_threads(1)
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    output.mkdir(parents=True)
    (output / 'source').mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / name, output / 'source' / name)
    count, maximum = (2, 20) if smoke else (100, 500)
    dump(output / 'config.json', {'experiment': 'T4-R0', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'full_protocol': not smoke, 'seed': 21000, 'games_per_model': count, 'max_steps': maximum,
        'board_size': 9, 'num_agents': 2, 'device': 'cpu', 'threads': 1, 'checkpoints': models,
        'source_sha256': hashes, 'parent_input_sha256': parent_inputs, 'expected_validation': expected,
        'wall_destroy_bonus': .15, 'near_distance': 3, 'timeout_replays_per_model': 4,
        'components': COMPONENTS, 'reward_tolerance': 1e-10, 'training_updates': 0,
        'budget_minutes': budget_minutes, 'torch': torch.__version__, 'numpy': np.__version__,
        'limits': 'Known validation only; descriptive proxies, no training, model selection or new held-out tests'})
    (ROOT / 'runs/r0-last-path.txt').write_text(str(output), encoding='utf-8')
    started, all_rows, reports, cases = perf_counter(), {}, {}, []
    reason = None
    stop, timer = stage_timer(budget_minutes * 60)
    try:
        for name, checkpoint in models.items():
            net = ActorCriticNet(board_size=9)
            smart_load_weights(net, checkpoint['path'], 'cpu')
            net.eval()
            initial_weights = {k: v.clone() for k, v in net.state_dict().items()}
            rows = all_rows.setdefault(name, [])
            saved = 0
            with gzip.open(output / f'steps_{name}.jsonl.gz', 'wt', encoding='utf-8') as tape:
                for index in range(count):
                    result = play_game(net, 21000 + index // 2, index % 2, maximum, stop)
                    if result is None:
                        reason = 'budget_or_cancellation'
                        break
                    row, steps, frames = result
                    rows.append(row)
                    for step in steps:
                        tape.write(json.dumps({'seed': row['seed'], 'aid': row['aid'], **step}) + '\n')
                    if row['timeout'] and saved < 4:
                        path = f'timeout_{name}_{row["seed"]}_A{row["aid"]}.json.gz'
                        with gzip.open(output / path, 'wt', encoding='utf-8') as f:
                            json.dump({'model': name, 'game': row, 'frames': frames, 'trace': steps}, f)
                        cases.append({'model': name, 'seed': row['seed'], 'aid': row['aid'], 'path': path})
                        saved += 1
                    dump(output / f'games_{name}.json', rows)
                    if (index + 1) % 10 == 0 or index + 1 == count:
                        print(f'[{name}] {index + 1}/{count} games; timeout replays {saved}', flush=True)
            assert all(torch.equal(value, net.state_dict()[key]) for key, value in initial_weights.items())
            if len(rows) != count:
                break
            if not smoke:
                validation_parity(rows, expected[name])
            reports[name] = {'all': describe(rows), 'by_outcome': {
                group: describe([r for r in rows if r['outcome'] == group]) for group in GROUPS},
                'by_corner': {str(a): describe([r for r in rows if r['aid'] == a]) for a in (0, 1)}}
        if len(reports) == 3:
            plot_results(output, all_rows)
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        raise
    finally:
        timer.cancel()
        try:
            unchanged = all(digest(Path(r['path'])) == r['sha256'] for r in models.values()) and hashes == {
                name: digest(ROOT / name) for name in SOURCES} and all(
                digest(Path(path)) == sha for path, sha in parent_inputs.items())
        except FileNotFoundError:
            unchanged = False
        complete = reason is None and len(reports) == 3 and unchanged
        if not smoke:
            complete &= len(cases) >= 10
            if reason is None and len(reports) == 3 and len(cases) < 10:
                reason = 'insufficient_timeout_replays'
        dump(output / 'timeout_cases.json', cases)
        dump(output / 'statistics.json', reports)
        summary = {'status': 'complete' if complete else 'incomplete', 'full_protocol': not smoke,
            'reason': reason, 'games_completed': sum(map(len, all_rows.values())), 'timeout_cases': len(cases),
            'validation_parity': complete and not smoke, 'hashes_unchanged': unchanged,
            'training_updates': 0, 'decision': 'retain_original', 'elapsed_seconds': perf_counter() - started,
            'models': {name: report['all'] for name, report in reports.items()}}
        dump(output / 'summary.json', summary)
        shutil.make_archive(str(output) + '-send', 'zip', root_dir=output)
        print(json.dumps({k: v for k, v in summary.items() if k != 'models'}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--budget-minutes', type=float, default=60)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / ('r0-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.smoke, args.budget_minutes)
