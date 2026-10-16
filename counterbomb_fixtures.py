"""Diagnostic fixtures for a delayed counter-bomb closing a single exit."""
from copy import deepcopy
from dataclasses import asdict
import pickle

import numpy as np

from benchmark_t3 import transform, transform_action
from bomberman_env import BombermanFFAEnv, Bomb, ACTION_BOMB, _MOVE_DELTA
from train_actor_critic import get_action_mask

VARIANTS = ('returned_capacity', 'occupied_tile', 'no_counter', 'alternate_exit',
            'capacity_unavailable', 'unreachable_door')


def make_fixture(variant, rotation=0, mirror=False, aid=0):
    if variant not in VARIANTS:
        raise ValueError(variant)
    env = BombermanFFAEnv(board_size=9, num_agents=2, wooden_wall_density=0,
                         powerup_prob=0, seed=20261015)
    env.reset()
    env.grid[:] = 1
    cells = [(3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (2, 3)]
    if variant == 'alternate_exit':
        cells += [(4, 1), (5, 1), (5, 2), (5, 3)]
    for pos in cells:
        env.grid[transform(pos, rotation, mirror)] = 0
    for pos in ((3, 2), (3, 3)):
        env.grid[transform(pos, rotation, mirror)] = 2
    enemy = 1 - aid
    env.agents[aid].row, env.agents[aid].col = transform((3, 1), rotation, mirror)
    env.agents[enemy].row, env.agents[enemy].col = transform((2, 3), rotation, mirror)
    env.agents[enemy].ammo = 1 if variant in ('returned_capacity', 'capacity_unavailable') else 2
    env.agents[enemy].bombs_placed = 1
    timer = 4 if variant == 'returned_capacity' else 2
    bomb_pos = (2, 3)
    if variant in ('capacity_unavailable', 'unreachable_door'):
        bomb_pos, timer = (7, 7), 20
        env.grid[transform(bomb_pos, rotation, mirror)] = 0
    if variant == 'unreachable_door':
        env.grid[transform((1, 3), rotation, mirror)] = 1
    env.bombs = [Bomb(*transform(bomb_pos, rotation, mirror), enemy, timer, 2)]
    delay = 5 if variant == 'returned_capacity' else 3
    own = Bomb(*transform((3, 1), rotation, mirror), aid, env.bomb_timer, 2)
    counter = Bomb(*transform((1, 2), rotation, mirror), enemy, env.bomb_timer, 2)
    schedule = [(0, own)]
    if variant in ('returned_capacity', 'occupied_tile', 'alternate_exit'):
        schedule.append((delay, counter))
    route = [1, 3] + [0] * (delay - 2) + [5, 4, 4, 4, 2]
    if variant == 'no_counter':
        route[delay] = 0
    if variant == 'unreachable_door':
        route = [0] * len(route)
    return env, schedule, [transform_action(a, rotation, mirror) for a in route]


def surviving_path(env, aid, schedule, enemy_positions=None):
    """Timed reachability for fixed bombs and an optional fixed occupancy timeline.

    A returned path still needs a physics replay to certify collisions/masks.
    The initial action is BOMB, and subsequent actions contain no new bombs.
    """
    burns, blocked = env._forecast_blasts(scheduled_bombs=schedule)
    pos = env.agents[aid].row, env.agents[aid].col
    reachable = {pos: [ACTION_BOMB]} if not burns[0, *pos] else {}
    for tick in range(1, len(burns)):
        following = {}
        for (r, c), path in reachable.items():
            for action, (dr, dc) in ((0, (0, 0)), *_MOVE_DELTA.items()):
                dest = r + dr, c + dc
                if (0 <= dest[0] < env.board_size and 0 <= dest[1] < env.board_size
                        and not burns[tick, *dest] and (action == 0 or not blocked[tick, *dest])
                        and not any(delay == tick and bomb.owner != aid and (bomb.row, bomb.col) == dest
                                    for delay, bomb in schedule)
                        and (enemy_positions is None or dest != enemy_positions[tick])):
                    following.setdefault(dest, path + [action])
        reachable = following
    return next(iter(reachable.values()), None)


def parity_replay(env, aid, schedule, learner_actions, enemy_actions, require_alive=False):
    """Compare forecast with step at damage time, before flames decay."""
    burns, blocked, slots = env._forecast_blasts(scheduled_bombs=schedule, return_slots=True)
    clone, actual, tape = deepcopy(env), {}, []
    tick_bombs = clone._tick_bombs_and_explode

    def capture_damage():
        actual['blocked'] = clone.grid != 0
        for bomb in clone.bombs:
            actual['blocked'][bomb.row, bomb.col] = True
        actual['slots'] = [a.ammo - a.bombs_placed for a in clone.agents]
        tick_bombs()
        actual['burns'] = np.zeros_like(clone.grid, dtype=bool)
        for pos in clone.flames:
            actual['burns'][pos] = True

    clone._tick_bombs_and_explode = capture_damage
    placements = {(tick, b.owner): (b.row, b.col) for tick, b in schedule}
    for tick in range(len(burns)):
        actions = [0, 0]
        actions[aid] = learner_actions[tick] if tick < len(learner_actions) else 0
        actions[1 - aid] = enemy_actions[tick] if tick < len(enemy_actions) else 0
        mask = get_action_mask(clone, aid) if clone.agents[aid].alive else [False] * 6
        if require_alive:
            assert mask[actions[aid]], (tick, actions, mask)
        _, _, _, info = clone.step(actions)
        assert np.array_equal(actual['burns'], burns[tick]), ('damage', tick)
        assert np.array_equal(actual['blocked'], blocked[tick]), ('blocking', tick)
        assert actual['slots'] == slots[tick].tolist(), ('capacity', tick)
        for owner in info['bombs_placed']:
            assert (tick, owner) in placements, ('unexpected placement', tick, owner)
            assert (clone.agents[owner].row, clone.agents[owner].col) == placements[tick, owner]
        assert set(info['bombs_placed']) == {owner for delay, owner in placements if delay == tick}
        tape.append({'tick': tick, 'actions': actions, 'mask': mask, 'alive': info['alive'],
                     'bombs_placed': info['bombs_placed'], 'self_blast_deaths': info['self_blast_deaths']})
        if require_alive:
            assert clone.agents[aid].alive, (tick, tape[-1])
    assert not clone.bombs and not clone.flames
    return tape


def masked_surviving_path(env, aid, schedule, enemy_actions):
    """Finite fixture search: public steps, every learner action through the mask."""
    horizon = len(env._forecast_blasts(scheduled_bombs=schedule)[0])
    frontier = [(deepcopy(env), [])]
    placements = {(tick, b.owner) for tick, b in schedule}
    for tick in range(horizon):
        following = {}
        for branch, path in frontier:
            mask = get_action_mask(branch, aid)
            for action in ([ACTION_BOMB] if tick == 0 else range(ACTION_BOMB)):
                if not mask[action]:
                    continue
                child = deepcopy(branch)
                actions = [0, 0]
                actions[aid] = action
                actions[1 - aid] = enemy_actions[tick] if tick < len(enemy_actions) else 0
                _, _, _, info = child.step(actions)
                if (child.agents[aid].alive and set(info['bombs_placed']) ==
                        {owner for delay, owner in placements if delay == tick}):
                    following.setdefault(pickle.dumps(child.__dict__), (child, path + [action]))
        assert len(following) < 4096, 'Fixture search exceeded its finite-state budget'
        frontier = list(following.values())
    return next((path for branch, path in frontier if not branch.bombs and not branch.flames), None)


def certify_fixtures():
    results = []
    for variant in VARIANTS:
        for rotation in range(4):
            for mirror in (False, True):
                for aid in (0, 1):
                    env, schedule, route = make_fixture(variant, rotation, mirror, aid)
                    assert get_action_mask(env, aid)[ACTION_BOMB], (variant, rotation, mirror, aid)
                    shadow = deepcopy(env)
                    enemy_positions = []
                    for tick in range(len(env._forecast_blasts(scheduled_bombs=schedule)[0])):
                        actions = [0, 0]
                        actions[aid] = ACTION_BOMB if tick == 0 else 0
                        actions[1 - aid] = route[tick] if tick < len(route) else 0
                        shadow.step(actions)
                        enemy = shadow.agents[1 - aid]
                        enemy_positions.append((enemy.row, enemy.col) if enemy.alive else None)
                    path = surviving_path(env, aid, schedule, enemy_positions)
                    unsafe = variant in ('returned_capacity', 'occupied_tile')
                    assert (path is None) == unsafe, (variant, path)
                    if unsafe:
                        assert masked_surviving_path(env, aid, schedule, route) is None
                        # Mask-valid movement until the fallback becomes unavoidable.
                        actions = [5, 1, 1, 2, 1, 2, 1, 0]
                        actions = [transform_action(a, rotation, mirror) for a in actions]
                    else:
                        actions = masked_surviving_path(env, aid, schedule, route)
                        assert actions is not None, (variant, rotation, mirror, aid)
                    tape = parity_replay(env, aid, schedule, actions, route, require_alive=not unsafe)
                    assert any(aid in row['self_blast_deaths'] for row in tape) == unsafe
                    results.append({'variant': variant, 'rotation': rotation, 'mirror': mirror, 'aid': aid,
                        'initial_bomb_allowed': True, 'surviving_path_with_fixed_enemy_route': path,
                        'relaxed_path_ignoring_enemy': surviving_path(env, aid, schedule),
                        'scheduled_bombs': [(d, asdict(b)) for d, b in schedule], 'tape': tape})
    return results


def replay_seen_counterbombs(replays):
    results = []
    for saved in replays:
        if saved['model'] != 'original':
            continue
        aid, seed = saved['aid'], saved['seed']
        start = 41 if seed == 32080 else 37
        delay = 5 if seed == 32080 else 3
        env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed)
        env.reset()
        for row in saved['tape'][:start]:
            env.step(row['actions'])
        before = saved['tape'][start]['before']
        assert env.grid.tolist() == before['grid']
        assert [asdict(a) for a in env.agents] == before['agents']
        assert [asdict(b) for b in env.bombs] == before['bombs']
        own, enemy = env.agents[aid], env.agents[1 - aid]
        counter_pos = saved['tape'][start + delay]['before']['agents'][1 - aid]
        schedule = [(0, Bomb(own.row, own.col, aid, env.bomb_timer, own.blast_range)),
                    (delay, Bomb(counter_pos['row'], counter_pos['col'], 1 - aid, env.bomb_timer, enemy.blast_range))]
        assert get_action_mask(env, aid)[ACTION_BOMB]
        horizon = len(env._forecast_blasts(scheduled_bombs=schedule)[0])
        # Only the observed approach/occupancy up to counter placement is fixed.
        enemy_positions = [tuple(row['after']['agents'][1 - aid][key] for key in ('row', 'col'))
                           for row in saved['tape'][start:start + delay + 1]]
        enemy_positions += [None] * (horizon - len(enemy_positions))
        known_only_path = surviving_path(env, aid, schedule[:1], enemy_positions)
        assert known_only_path is not None
        assert surviving_path(env, aid, schedule, enemy_positions) is None
        results.append({'seed': seed, 'aid': aid, 'placement_tick': start, 'counter_delay': delay,
                        'initial_mask': get_action_mask(env, aid), 'known_only_path': known_only_path,
                        'relaxed_counter_path_ignoring_enemy': surviving_path(env, aid, schedule),
                        'with_counter_path': None, 'scheduled_bombs': [(d, asdict(b)) for d, b in schedule]})
    assert {r['seed'] for r in results} == {32080, 32133}
    return results
