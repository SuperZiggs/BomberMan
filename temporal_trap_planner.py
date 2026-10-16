from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from bomberman_env import (
    ACTION_BOMB,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STOP,
    ACTION_UP,
    Bomb,
    BombermanFFAEnv,
    _MOVE_DELTA,
)
from rule_based_bot import (
    _bomb_escape_pressure,
    _can_bomb_and_escape,
    _reachable_cells,
    _survival_map,
)


@dataclass(frozen=True)
class TemporalTrapConfig:
    horizon: int = 8
    max_bombs: int = 2
    min_pressure_gain: float = 0.35


@dataclass(frozen=True)
class TemporalTrapPlan:
    first_action: int
    bomb_positions: tuple[tuple[int, int], ...]
    pressure_gain: float
    completion_tick: int
    escape_margin: int
    predicts_elimination: bool

    _action_sequence = ()

    @property
    def action_sequence(self) -> tuple[int, ...]:
        return self._action_sequence


@dataclass(frozen=True)
class _PathNode:
    position: tuple[int, int]
    tick: int
    first_action: int
    path: tuple[tuple[int, int], ...]


_ACTION_TIE_ORDER = {
    ACTION_BOMB: 5,
    ACTION_LEFT: 4,
    ACTION_RIGHT: 3,
    ACTION_UP: 2,
    ACTION_DOWN: 1,
    ACTION_STOP: 0,
}


def _validate_config(config: TemporalTrapConfig) -> None:
    if config.horizon < 1:
        raise ValueError("temporal trap horizon must be positive")
    if config.max_bombs not in (1, 2):
        raise ValueError("temporal trap max_bombs must be 1 or 2")
    if config.min_pressure_gain < 0.0:
        raise ValueError("temporal trap min_pressure_gain must be non-negative")


def _cell_closes_by(env: BombermanFFAEnv, position: tuple[int, int], tick: int) -> bool:
    if not env.sudden_death:
        return False
    for ring in range(env._sudden_death_ring + 1, env.board_size // 2):
        if position not in env._ring_cells(ring):
            continue
        closure_step = env.sudden_death_start + (ring - 1) * env.sudden_death_interval
        return env.step_count + tick >= closure_step
    return False


def _path_nodes(env: BombermanFFAEnv, agent_id: int, danger: np.ndarray,
                mask: list[bool], horizon: int) -> list[_PathNode]:
    agent = env.agents[agent_id]
    start = (agent.row, agent.col)
    occupied = {
        (other.row, other.col)
        for other in env.agents
        if other.alive and other.agent_id != agent_id
    }
    burns, blocked = env._forecast_blasts(min_horizon=horizon)
    max_move_ticks = horizon - 1
    queue = deque()
    visited = {start}
    nodes = []

    for action, (delta_row, delta_col) in _MOVE_DELTA.items():
        if action >= len(mask) or not mask[action]:
            continue
        position = (start[0] + delta_row, start[1] + delta_col)
        if position in occupied or not env._is_walkable(*position):
            continue
        if danger[position] > 0 or burns[0, position[0], position[1]]:
            continue
        if _cell_closes_by(env, position, 1):
            continue
        visited.add(position)
        node = _PathNode(position, 1, action, (start, position))
        nodes.append(node)
        queue.append(node)

    while queue:
        node = queue.popleft()
        if node.tick >= max_move_ticks:
            continue
        for delta_row, delta_col in _MOVE_DELTA.values():
            position = (node.position[0] + delta_row, node.position[1] + delta_col)
            tick = node.tick + 1
            if position in visited or position in occupied or not env._is_walkable(*position):
                continue
            forecast_tick = min(tick - 1, len(burns) - 1)
            if blocked[forecast_tick, position[0], position[1]]:
                continue
            if burns[forecast_tick, position[0], position[1]]:
                continue
            if _cell_closes_by(env, position, tick):
                continue
            visited.add(position)
            child = _PathNode(position, tick, node.first_action, node.path + (position,))
            nodes.append(child)
            queue.append(child)
    return nodes


def _immediately_contested(env: BombermanFFAEnv, agent_id: int,
                           position: tuple[int, int], tick: int) -> bool:
    if tick != 1:
        return False
    for other in env.agents:
        if not other.alive or other.agent_id == agent_id:
            continue
        if abs(other.row - position[0]) + abs(other.col - position[1]) == 1:
            return True
    return False


def _has_closing_continuation(env: BombermanFFAEnv, node: _PathNode) -> bool:
    next_tick = node.tick + 1
    return any(
        env._is_walkable(node.position[0] + delta_row, node.position[1] + delta_col)
        and _cell_closes_by(
            env,
            (node.position[0] + delta_row, node.position[1] + delta_col),
            next_tick,
        )
        for delta_row, delta_col in _MOVE_DELTA.values()
    )


def _advanced_copy(env: BombermanFFAEnv, agent_id: int,
                   node: _PathNode) -> BombermanFFAEnv:
    simulated = deepcopy(env)
    for previous, position in zip(node.path, node.path[1:]):
        delta = (position[0] - previous[0], position[1] - previous[1])
        action = next(a for a, move in _MOVE_DELTA.items() if move == delta)
        actions = [ACTION_STOP] * simulated.num_agents
        actions[agent_id] = action
        simulated.step(actions)
        learner = simulated.agents[agent_id]
        if not learner.alive or (learner.row, learner.col) != position:
            break
    return simulated


def _placement_safety(env: BombermanFFAEnv, agent_id: int,
                      node: _PathNode) -> tuple[bool, int]:
    simulated = _advanced_copy(env, agent_id, node)
    agent = simulated.agents[agent_id]
    if not agent.alive or (agent.row, agent.col) != node.position:
        return False, 0
    danger = simulated._danger_map()
    if not _can_bomb_and_escape(
            simulated, agent_id, node.position[0], node.position[1], danger, robust=True):
        return False, 0
    proposed = Bomb(
        node.position[0], node.position[1], agent_id,
        simulated.bomb_timer, agent.blast_range,
    )
    safe, blocked, _ = _survival_map(
        simulated,
        node.position[0],
        node.position[1],
        proposed,
        safety_margin=2,
        reserve_exits=True,
    )
    if not safe[0, node.position[0], node.position[1]]:
        return False, 0
    return True, _reachable_escape_margin(safe, blocked, node.position)


def _reachable_escape_margin(safe: np.ndarray, blocked: np.ndarray,
                             start: tuple[int, int], start_tick: int = 0) -> int:
    if start_tick >= len(safe) or not safe[start_tick, start[0], start[1]]:
        return 0
    frontier = {start}
    reached = {start}
    for tick in range(start_tick + 1, len(safe)):
        next_frontier = set()
        for row, col in frontier:
            if safe[tick, row, col]:  # staying on a bomb is legal
                next_frontier.add((row, col))
            for delta_row, delta_col in _MOVE_DELTA.values():
                nr, nc = row + delta_row, col + delta_col
                if (0 <= nr < safe.shape[1] and 0 <= nc < safe.shape[2]
                        and not blocked[tick, nr, nc] and safe[tick, nr, nc]):
                    next_frontier.add((nr, nc))
        frontier = next_frontier
        reached.update(frontier)
        if not frontier:
            break
    return len(reached)


def _candidate_plan(env: BombermanFFAEnv, agent_id: int,
                    node: _PathNode) -> TemporalTrapPlan | None:
    if _immediately_contested(env, agent_id, node.position, node.tick):
        return None
    safe, escape_margin = _placement_safety(env, agent_id, node)
    if not safe:
        return None
    pressure = _scheduled_bomb_pressure(env, agent_id, node)
    plan = TemporalTrapPlan(
        first_action=node.first_action,
        bomb_positions=(node.position,),
        pressure_gain=pressure,
        completion_tick=node.tick,
        escape_margin=escape_margin,
        predicts_elimination=pressure >= 1.0 - 1e-9,
    )
    object.__setattr__(plan, "_action_sequence", _actions_for_placements(node, None))
    return plan


def _actions_for_placements(first: _PathNode,
                            second: _PathNode | None) -> tuple[int, ...]:
    """Include the placement tick between the shared path prefix and suffix."""
    for node in (first,) if second is None else (first, second):
        if (not node.path or len(node.path) != node.tick + 1
                or node.path[-1] != node.position):
            raise ValueError("placement path must match its position and tick")
    if second is not None and (
            second.tick <= first.tick
            or second.path[:len(first.path)] != first.path):
        raise ValueError("second placement must extend the first placement path")

    path = first.path if second is None else second.path
    action_by_delta = {delta: action for action, delta in _MOVE_DELTA.items()}
    movements = []
    for previous, position in zip(path, path[1:]):
        delta = (position[0] - previous[0], position[1] - previous[1])
        if delta not in action_by_delta:
            raise ValueError("placement path must use adjacent movement cells")
        movements.append(action_by_delta[delta])
    actions = tuple(movements)
    if second is None:
        return actions + (ACTION_BOMB,)
    return (actions[:first.tick] + (ACTION_BOMB,)
            + actions[first.tick:] + (ACTION_BOMB,))


def _scheduled_bomb_pressure(env: BombermanFFAEnv, agent_id: int,
                             node: _PathNode) -> float:
    if node.tick == 0:
        return float(_bomb_escape_pressure(
            env, agent_id, node.position[0], node.position[1]
        ))
    agent = env.agents[agent_id]
    bomb = Bomb(node.position[0], node.position[1], agent_id,
                env.bomb_timer, agent.blast_range, placement_tick=node.tick)
    burns, blocked = env._forecast_blasts(bomb)
    explosion_ticks = np.flatnonzero(burns[node.tick:, *node.position])
    if len(explosion_ticks) == 0:
        return 0.0
    ticks = node.tick + int(explosion_ticks[0]) + 1
    baseline_burns, baseline_blocked = env._forecast_blasts(min_horizon=ticks)
    pressure = 0.0
    for enemy in env.agents:
        if not enemy.alive or enemy.agent_id == agent_id:
            continue
        start = (enemy.row, enemy.col)
        before = int(np.count_nonzero(_reachable_cells(
            start, baseline_burns, baseline_blocked, ticks
        )))
        if before:
            after = int(np.count_nonzero(_reachable_cells(
                start, burns, blocked, ticks
            )))
            pressure = max(pressure, 1.0 - after / before)
    return float(pressure)


def _combined_pressures(env: BombermanFFAEnv, agent_id: int,
                        first: _PathNode, second: _PathNode,
                        completion_tick: int) -> tuple[float, float]:
    agent = env.agents[agent_id]
    first_bomb = Bomb(
        first.position[0], first.position[1], agent_id,
        env.bomb_timer, agent.blast_range, placement_tick=first.tick,
    )
    second_bomb = Bomb(
        second.position[0], second.position[1], agent_id,
        env.bomb_timer, agent.blast_range, placement_tick=completion_tick,
    )
    pair_burns, pair_blocked = env._forecast_blasts(
        [first_bomb, second_bomb],
        min_horizon=env.bomb_timer + completion_tick + env.flame_duration,
    )
    first_burns, first_blocked = env._forecast_blasts(
        first_bomb, min_horizon=len(pair_burns)
    )
    baseline_burns, baseline_blocked = env._forecast_blasts(
        min_horizon=len(pair_burns)
    )
    first_pressure = 0.0
    pair_pressure = 0.0
    for enemy in env.agents:
        if not enemy.alive or enemy.agent_id == agent_id:
            continue
        start = (enemy.row, enemy.col)
        before = int(np.count_nonzero(_reachable_cells(
            start, baseline_burns, baseline_blocked, len(pair_burns)
        )))
        if before:
            after_first = int(np.count_nonzero(_reachable_cells(
                start, first_burns, first_blocked, len(pair_burns)
            )))
            after_pair = int(np.count_nonzero(_reachable_cells(
                start, pair_burns, pair_blocked, len(pair_burns)
            )))
            first_pressure = max(first_pressure, 1.0 - after_first / before)
            pair_pressure = max(pair_pressure, 1.0 - after_pair / before)
    return float(first_pressure), float(pair_pressure)


def _pair_timeline(first: _PathNode, second: _PathNode,
                   horizon: int) -> tuple[list[tuple[int, tuple[int, int]]], int] | None:
    if second.tick <= first.tick:
        return None
    if second.path[:len(first.path)] != first.path:
        return None
    completion_tick = second.tick + 1
    if completion_tick > horizon:
        return None

    timeline = []
    for path_index, position in enumerate(second.path[1:], start=1):
        action_tick = path_index - 1 if path_index <= first.tick else path_index
        timeline.append((action_tick, position))
    timeline.append((first.tick, first.position))
    timeline.append((completion_tick, second.position))
    timeline.sort(key=lambda item: item[0])
    return timeline, completion_tick


def _pair_survival(env: BombermanFFAEnv, agent_id: int,
                   first: _PathNode, second: _PathNode,
                   horizon: int) -> tuple[bool, int, int]:
    timeline_result = _pair_timeline(first, second, horizon)
    if timeline_result is None:
        return False, 0, 0
    timeline, completion_tick = timeline_result
    agent = env.agents[agent_id]
    placements = [
        Bomb(first.position[0], first.position[1], agent_id,
             env.bomb_timer, agent.blast_range, placement_tick=first.tick),
        Bomb(second.position[0], second.position[1], agent_id,
             env.bomb_timer, agent.blast_range, placement_tick=completion_tick),
    ]
    safe, blocked, burns = _survival_map(
        env,
        second.position[0],
        second.position[1],
        placements,
        safety_margin=2,
        reserve_exits=True,
        ignore_agent_id=agent_id,
    )
    if completion_tick >= len(safe):
        return False, 0, completion_tick
    previous = (agent.row, agent.col)
    for tick, position in timeline:
        if tick >= len(burns) or burns[tick, position[0], position[1]]:
            return False, 0, completion_tick
        if position != previous and blocked[tick, position[0], position[1]]:
            return False, 0, completion_tick
        if _cell_closes_by(env, position, tick + 1):
            return False, 0, completion_tick
        previous = position
    if not safe[completion_tick, second.position[0], second.position[1]]:
        return False, 0, completion_tick
    margin = _reachable_escape_margin(
        safe, blocked, second.position, start_tick=completion_tick
    )
    return margin > 0, margin, completion_tick


def _pair_plan(env: BombermanFFAEnv, agent_id: int, first: _PathNode,
               second: _PathNode, horizon: int
               ) -> tuple[TemporalTrapPlan | None, str]:
    if first.position == second.position:
        return None, "duplicate"
    first_blast = set(env._blast_cells(
        first.position[0], first.position[1], env.agents[agent_id].blast_range
    ))
    second_blast = set(env._blast_cells(
        second.position[0], second.position[1], env.agents[agent_id].blast_range
    ))
    if not second_blast - first_blast:
        return None, "duplicate"
    survives, escape_margin, completion_tick = _pair_survival(
        env, agent_id, first, second, horizon
    )
    if not survives:
        return None, "unsafe"
    first_pressure, combined = _combined_pressures(
        env, agent_id, first, second, completion_tick
    )
    if combined <= first_pressure + 1e-9:
        return None, "duplicate"
    plan = TemporalTrapPlan(
        first_action=first.first_action,
        bomb_positions=(first.position, second.position),
        pressure_gain=combined,
        completion_tick=completion_tick,
        escape_margin=escape_margin,
        predicts_elimination=combined >= 1.0 - 1e-9,
    )
    object.__setattr__(plan, "_action_sequence", _actions_for_placements(first, second))
    return plan, "selected"


def _score(plan: TemporalTrapPlan) -> tuple:
    return (
        int(plan.predicts_elimination),
        plan.pressure_gain,
        -plan.completion_tick,
        -len(plan.bomb_positions),
        plan.escape_margin,
        _ACTION_TIE_ORDER.get(plan.first_action, -1),
    )


def find_temporal_trap_plan(
    env: BombermanFFAEnv,
    agent_id: int,
    danger: np.ndarray,
    mask: list[bool],
    config: TemporalTrapConfig = TemporalTrapConfig(),
    diagnostics: dict | None = None,
) -> TemporalTrapPlan | None:
    _validate_config(config)
    agent = env.agents[agent_id]
    remaining_capacity = agent.ammo - agent.bombs_placed
    if not agent.alive or remaining_capacity <= 0:
        return None

    existing_owned_bombs = any(bomb.owner == agent_id for bomb in env.bombs)
    pair_search = (config.max_bombs == 2 and remaining_capacity >= 2
                   and not existing_owned_bombs)
    move_limit = (config.horizon - 1 if pair_search
                  else max(1, config.horizon - env.bomb_timer))
    nodes = _path_nodes(env, agent_id, danger, mask, move_limit + 1)
    if not pair_search:
        path_prefixes = {node.path[:-1] for node in nodes if len(node.path) > 2}
        nodes = [node for node in nodes if node.path not in path_prefixes]
    nodes = [
        node for node in nodes
        if not _has_closing_continuation(env, node)
    ]
    if (config.max_bombs == 2 and mask[ACTION_BOMB]
            and danger[agent.row, agent.col] != 1):
        nodes.insert(0, _PathNode(
            (agent.row, agent.col), 0, ACTION_BOMB,
            ((agent.row, agent.col),),
        ))

    plans = []
    single_candidates = []
    for node in nodes:
        plan = _candidate_plan(env, agent_id, node)
        if plan is None:
            if diagnostics is not None:
                diagnostics["temporal_trap_rejected_unsafe"] = diagnostics.get(
                    "temporal_trap_rejected_unsafe", 0
                ) + 1
            continue
        single_candidates.append((node, plan))
        single_search = (config.max_bombs == 1 or existing_owned_bombs
                         or (not pair_search and not mask[ACTION_BOMB]))
        if (single_search
                and plan.pressure_gain >= config.min_pressure_gain):
            plans.append(plan)

    if pair_search:
        duplicate_actions = set()
        for first_node, first_plan in single_candidates:
            for second_node, _ in single_candidates:
                if first_node.position not in second_node.path:
                    continue
                pair, reason = _pair_plan(
                    env, agent_id, first_node, second_node, config.horizon
                )
                if pair is None:
                    if diagnostics is not None:
                        field = (
                            "temporal_trap_rejected_duplicate"
                            if reason == "duplicate"
                            else "temporal_trap_rejected_unsafe"
                        )
                        if reason != "duplicate" or first_node.first_action not in duplicate_actions:
                            diagnostics[field] = diagnostics.get(field, 0) + 1
                            if reason == "duplicate":
                                duplicate_actions.add(first_node.first_action)
                elif pair.pressure_gain >= config.min_pressure_gain:
                    plans.append(pair)

    selected = max(plans, key=_score) if plans else None
    if diagnostics is not None and selected is not None:
        diagnostics["predicted_refuge_reduction"] = diagnostics.get(
            "predicted_refuge_reduction", 0.0
        ) + selected.pressure_gain
        diagnostics["predicted_trap_completion_ticks"] = diagnostics.get(
            "predicted_trap_completion_ticks", 0.0
        ) + selected.completion_tick
    return selected


def _reachable_region(env: BombermanFFAEnv, start: tuple[int, int],
                      blocked_cell: tuple[int, int], horizon: int) -> set[tuple[int, int]]:
    reached = {start}
    frontier = {start}
    for _ in range(horizon):
        next_frontier = set()
        for row, col in frontier:
            for delta_row, delta_col in _MOVE_DELTA.values():
                position = (row + delta_row, col + delta_col)
                if position in reached or position == blocked_cell:
                    continue
                if env._is_walkable(*position):
                    reached.add(position)
                    next_frontier.add(position)
        frontier = next_frontier
    return reached


def find_interception_action(env: BombermanFFAEnv, agent_id: int,
                             danger: np.ndarray, mask: list[bool],
                             horizon: int = 4) -> int | None:
    if horizon < 1:
        raise ValueError("interception horizon must be positive")
    agent = env.agents[agent_id]
    opponents = [other for other in env.agents
                 if other.alive and other.agent_id != agent_id]
    if not agent.alive or not opponents:
        return None

    start = (agent.row, agent.col)
    baseline = max(
        len(_reachable_region(env, (enemy.row, enemy.col), start, horizon))
        for enemy in opponents
    )
    candidates = []
    for action, (delta_row, delta_col) in _MOVE_DELTA.items():
        if action >= len(mask) or not mask[action]:
            continue
        position = (start[0] + delta_row, start[1] + delta_col)
        if danger[position] > 0 or _cell_closes_by(env, position, 1):
            continue
        degree = sum(
            env._is_walkable(position[0] + dr, position[1] + dc)
            for dr, dc in _MOVE_DELTA.values()
        )
        if degree > 2:
            continue
        future = max(
            len(_reachable_region(env, (enemy.row, enemy.col), position, horizon))
            for enemy in opponents
        )
        gain = baseline - future
        if gain > 0 and gain / baseline >= 1.0 / horizon:
            candidates.append((gain, -degree, _ACTION_TIE_ORDER[action], action))
    return max(candidates)[-1] if candidates else None
