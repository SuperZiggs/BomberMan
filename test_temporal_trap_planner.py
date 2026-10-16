"""Independent acceptance tests for the temporal trap planner teacher."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from bomberman_env import (
    ACTION_BOMB,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STOP,
    ACTION_UP,
    Bomb,
    BombermanFFAEnv,
    EMPTY,
    POWERUP_AMMO,
    RIGID_WALL,
    _MOVE_DELTA,
)
from rule_based_bot import _can_bomb_and_escape, _survival_map
import train_actor_critic as training


MODULE_NAME = "temporal_trap_planner"
TRAINER_METRICS = (
    "temporal_trap_labels",
    "temporal_trap_bombs",
    "temporal_trap_rejected_unsafe",
    "temporal_trap_rejected_duplicate",
    "interception_labels",
    "predicted_refuge_reduction",
    "predicted_trap_completion_ticks",
)
MIRROR_ACTION = {
    ACTION_STOP: ACTION_STOP,
    ACTION_UP: ACTION_UP,
    ACTION_DOWN: ACTION_DOWN,
    ACTION_LEFT: ACTION_RIGHT,
    ACTION_RIGHT: ACTION_LEFT,
    ACTION_BOMB: ACTION_BOMB,
}


def arena(*, board_size=7, bomb_timer=6, sudden_death=False,
          sudden_death_start=20, sudden_death_interval=4):
    env = BombermanFFAEnv(
        board_size=board_size,
        num_agents=2,
        max_steps=40,
        wooden_wall_density=0,
        bomb_timer=bomb_timer,
        flame_duration=2,
        powerup_prob=0,
        sudden_death=sudden_death,
        sudden_death_start=sudden_death_start,
        sudden_death_interval=sudden_death_interval,
        sudden_death_warning_steps=8,
        seed=73,
    )
    env.reset()
    return env


def carve(env, cells):
    env.grid[1:-1, 1:-1] = RIGID_WALL
    for position in cells:
        env.grid[position] = EMPTY


def put_bomb(env, row, col, owner, *, timer=6, blast_range=2):
    env.bombs.append(Bomb(row, col, owner, timer, blast_range))
    env.agents[owner].bombs_placed += 1


def legal_mask(env, agent_id):
    agent = env.agents[agent_id]
    occupied = {
        (other.row, other.col)
        for other in env.agents
        if other.alive and other.agent_id != agent_id
    }
    mask = [False] * 6
    mask[ACTION_STOP] = True
    for action, (delta_row, delta_col) in _MOVE_DELTA.items():
        position = (agent.row + delta_row, agent.col + delta_col)
        mask[action] = env._is_walkable(*position) and position not in occupied
    mask[ACTION_BOMB] = (
        agent.bombs_placed < agent.ammo
        and not any((bomb.row, bomb.col) == (agent.row, agent.col) for bomb in env.bombs)
    )
    return mask


def followup_trap_env():
    env = arena()
    carve(
        env,
        {
            (3, 3), (3, 4), (3, 5), (3, 2), (3, 1),
            (2, 1), (1, 1), (1, 3), (2, 3),
        },
    )
    env.agents[0].row, env.agents[0].col = 3, 3
    env.agents[0].ammo = 2
    env.agents[0].blast_range = 2
    env.agents[1].row, env.agents[1].col = 3, 5
    env.agents[1].ammo = 1
    env.agents[1].blast_range = 2
    put_bomb(env, 1, 3, 0, timer=6, blast_range=2)
    return env


def redundant_followup_env():
    env = arena()
    carve(env, {(3, 1), (3, 2), (3, 3), (3, 4), (2, 1), (1, 1)})
    env.agents[0].row, env.agents[0].col = 3, 2
    env.agents[0].ammo = 2
    env.agents[0].blast_range = 1
    env.agents[1].row, env.agents[1].col = 3, 4
    return env


def counter_bomb_unsafe_env():
    env = arena()
    carve(
        env,
        {
            (1, 2), (1, 4), (1, 5), (2, 1), (2, 3), (3, 1),
            (3, 2), (4, 1), (4, 3), (5, 3), (5, 4), (5, 5),
        },
    )
    env.agents[0].row, env.agents[0].col = 3, 2
    env.agents[0].ammo = 2
    env.agents[1].row, env.agents[1].col = 4, 1
    env.agents[1].ammo = 1
    return env


def sudden_death_route_env(*, enabled):
    env = arena(
        sudden_death=enabled,
        sudden_death_start=2,
        sudden_death_interval=4,
    )
    carve(env, {(3, 3), (2, 3), (1, 3), (1, 4), (1, 5), (3, 2), (3, 1)})
    env.agents[0].row, env.agents[0].col = 3, 3
    env.agents[0].ammo = 1
    env.agents[0].blast_range = 2
    env.agents[1].row, env.agents[1].col = 1, 5
    env.agents[1].ammo = 0
    return env


def interception_corridor_env():
    env = arena()
    carve(env, {(3, col) for col in range(1, 6)} | {(2, 1), (1, 1)})
    env.agents[0].row, env.agents[0].col = 3, 1
    env.agents[0].blast_range = 2
    env.agents[1].row, env.agents[1].col = 3, 5
    return env


def pressure_ranking_env():
    env = arena()
    carve(
        env,
        {
            (1, 1), (1, 5), (2, 1), (2, 3), (2, 5), (3, 1),
            (3, 2), (3, 5), (4, 1), (4, 5), (5, 1), (5, 2),
            (5, 3), (5, 4), (5, 5),
        },
    )
    env.agents[0].row, env.agents[0].col = 5, 3
    env.agents[0].ammo = 1
    env.agents[0].blast_range = 2
    env.agents[1].row, env.agents[1].col = 2, 5
    env.agents[1].ammo = 0
    return env


def completion_ranking_env():
    env = arena()
    carve(
        env,
        {
            (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 3),
            (2, 5), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1),
            (4, 3), (5, 2), (5, 4), (5, 5),
        },
    )
    env.agents[0].row, env.agents[0].col = 3, 2
    env.agents[0].ammo = 1
    env.agents[0].blast_range = 2
    env.agents[1].row, env.agents[1].col = 4, 3
    env.agents[1].ammo = 0
    return env


def escape_margin_ranking_env():
    env = arena()
    carve(
        env,
        {
            (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 3),
            (2, 5), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
            (4, 1), (4, 3), (4, 5), (5, 2), (5, 3), (5, 4),
        },
    )
    env.agents[0].row, env.agents[0].col = 3, 3
    env.agents[0].ammo = 1
    env.agents[0].blast_range = 2
    env.agents[1].row, env.agents[1].col = 1, 3
    env.agents[1].ammo = 0
    return env


def adjacent_second_bomb_unsafe_env():
    env = arena(board_size=7, bomb_timer=7)
    env.grid[:] = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=env.grid.dtype,
    )
    env.agents[0].row, env.agents[0].col = 5, 3
    env.agents[0].ammo = 2
    env.agents[0].blast_range = 3
    env.agents[1].row, env.agents[1].col = 1, 2
    return env


def mirror_env(env):
    mirrored = deepcopy(env)
    mirrored.grid = np.fliplr(env.grid).copy()
    for source, target in zip(env.agents, mirrored.agents):
        target.row = source.row
        target.col = env.board_size - 1 - source.col
    for source, target in zip(env.bombs, mirrored.bombs):
        target.row = source.row
        target.col = env.board_size - 1 - source.col
    mirrored.flames = {
        (row, env.board_size - 1 - col): duration
        for (row, col), duration in env.flames.items()
    }
    mirrored.powerups = {
        (row, env.board_size - 1 - col): kind
        for (row, col), kind in env.powerups.items()
    }
    mirrored._flame_owners = {
        (row, env.board_size - 1 - col): owner
        for (row, col), owner in env._flame_owners.items()
    }
    mirrored._flame_sources = {
        (row, env.board_size - 1 - col): deepcopy(owners)
        for (row, col), owners in env._flame_sources.items()
    }
    return mirrored


def mirror_mask(mask):
    mirrored = [False] * len(mask)
    for action, allowed in enumerate(mask):
        mirrored[MIRROR_ACTION[action]] = allowed
    return mirrored


def numpy_state_equal(first, second):
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


class PlannerContractCase(unittest.TestCase):
    def planner_module(self):
        try:
            return importlib.import_module(MODULE_NAME)
        except ModuleNotFoundError as error:
            if error.name == MODULE_NAME:
                self.fail(
                    "temporal_trap_planner public module is missing; "
                    "the legacy static two-position teacher is not this contract"
                )
            raise

    def config(self, **overrides):
        planner = self.planner_module()
        values = {"horizon": 8, "max_bombs": 2, "min_pressure_gain": 0.35}
        values.update(overrides)
        return planner.TemporalTrapConfig(**values)

    def find_plan(self, env, *, mask=None, **config_overrides):
        planner = self.planner_module()
        if mask is None:
            mask = legal_mask(env, 0)
        return planner.find_temporal_trap_plan(
            env,
            0,
            env._danger_map(),
            mask,
            self.config(**config_overrides),
        )

    def reachable_positions(self, env, horizon):
        start = (env.agents[0].row, env.agents[0].col)
        occupied = {
            (agent.row, agent.col)
            for agent in env.agents
            if agent.alive and agent.agent_id != 0
        }
        reached = {start}
        frontier = {start}
        for _ in range(horizon):
            next_frontier = set()
            for row, col in frontier:
                for delta_row, delta_col in _MOVE_DELTA.values():
                    position = (row + delta_row, col + delta_col)
                    if position not in reached and position not in occupied and env._is_walkable(*position):
                        reached.add(position)
                        next_frontier.add(position)
            frontier = next_frontier
        return reached


class ActionSequenceContractCase(PlannerContractCase):
    def test_move_first_pair_keeps_both_placements_in_order(self):
        from test_temporal_trap_regressions import open_pair_arena
        env = open_pair_arena()
        mask = [False] * 6
        mask[ACTION_RIGHT] = True
        plan = self.find_plan(env, mask=mask, min_pressure_gain=0.01)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.action_sequence, (
            ACTION_RIGHT, ACTION_BOMB, ACTION_UP, ACTION_LEFT, ACTION_BOMB,
        ))
        self.assertEqual(plan.action_sequence.count(ACTION_BOMB), 2)
        self.assertEqual(plan.first_action, plan.action_sequence[0])
        self.assertEqual(plan.completion_tick, len(plan.action_sequence) - 1)
        with self.assertRaises(FrozenInstanceError):
            plan.action_sequence = (ACTION_STOP,)
        self.assertEqual(pickle.loads(pickle.dumps(plan)).action_sequence,
                         plan.action_sequence)

    def test_move_first_singleton_stops_at_its_placement(self):
        from test_temporal_trap_regressions import open_pair_arena
        env = open_pair_arena(bomb_timer=5, enemy=(1, 3))
        env.agents[0].ammo = 1
        mask = [False] * 6
        mask[ACTION_RIGHT] = True
        plan = self.find_plan(env, mask=mask, horizon=6, max_bombs=1,
                              min_pressure_gain=0.01)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.action_sequence, (ACTION_RIGHT, ACTION_BOMB))
        self.assertEqual(plan.action_sequence.count(ACTION_BOMB), 1)
        self.assertEqual(plan.action_sequence[-1], ACTION_BOMB)
        self.assertEqual(plan.first_action, plan.action_sequence[0])

    def test_stationary_first_placement_precedes_followup_movements(self):
        planner = self.planner_module()
        first = planner._PathNode((3, 2), 0, ACTION_BOMB, ((3, 2),))
        second = planner._PathNode((4, 3), 2, ACTION_RIGHT,
                                   ((3, 2), (3, 3), (4, 3)))
        self.assertEqual(planner._actions_for_placements(first, None),
                         (ACTION_BOMB,))
        self.assertEqual(planner._actions_for_placements(first, second),
                         (ACTION_BOMB, ACTION_RIGHT, ACTION_DOWN, ACTION_BOMB))

    def test_placement_actions_reject_malformed_paths(self):
        planner = self.planner_module()
        first = planner._PathNode((3, 3), 1, ACTION_RIGHT, ((3, 2), (3, 3)))
        invalid_nodes = (
            planner._PathNode((3, 3), 1, ACTION_RIGHT, ()),
            planner._PathNode((3, 3), 2, ACTION_RIGHT, ((3, 2), (3, 3))),
            planner._PathNode((3, 4), 1, ACTION_RIGHT, ((3, 2), (3, 3))),
            planner._PathNode((3, 4), 1, ACTION_RIGHT, ((3, 2), (3, 4))),
            planner._PathNode((4, 3), 1, ACTION_RIGHT, ((3, 2), (4, 3))),
            planner._PathNode((3, 2), 1, ACTION_RIGHT, ((3, 2), (3, 2))),
        )
        for node in invalid_nodes:
            with self.subTest(node=node), self.assertRaises(ValueError):
                planner._actions_for_placements(node, None)
        second = planner._PathNode((4, 3), 2, ACTION_DOWN,
                                   ((3, 2), (4, 2), (4, 3)))
        for node in (first, second):
            with self.subTest(second=node), self.assertRaises(ValueError):
                planner._actions_for_placements(first, node)


class PublicApiAcceptance(PlannerContractCase):
    def test_frozen_dataclasses_defaults_and_function_signatures(self):
        planner = self.planner_module()
        self.assertTrue(is_dataclass(planner.TemporalTrapConfig))
        self.assertTrue(is_dataclass(planner.TemporalTrapPlan))
        self.assertTrue(planner.TemporalTrapConfig.__dataclass_params__.frozen)
        self.assertTrue(planner.TemporalTrapPlan.__dataclass_params__.frozen)

        config = planner.TemporalTrapConfig()
        self.assertEqual((config.horizon, config.max_bombs, config.min_pressure_gain), (8, 2, 0.35))
        with self.assertRaises(FrozenInstanceError):
            config.horizon = 9

        expected_plan_fields = (
            "first_action",
            "bomb_positions",
            "pressure_gain",
            "completion_tick",
            "escape_margin",
            "predicts_elimination",
        )
        self.assertEqual(tuple(field.name for field in fields(planner.TemporalTrapPlan)), expected_plan_fields)

        plan_parameters = inspect.signature(planner.find_temporal_trap_plan).parameters
        self.assertEqual(tuple(plan_parameters)[:4], ("env", "agent_id", "danger", "mask"))
        self.assertIsInstance(plan_parameters["config"].default, planner.TemporalTrapConfig)
        interception_parameters = inspect.signature(planner.find_interception_action).parameters
        self.assertEqual(tuple(interception_parameters)[:4], ("env", "agent_id", "danger", "mask"))
        self.assertEqual(interception_parameters["horizon"].default, 4)


class TemporalPlanAcceptance(PlannerContractCase):
    def test_safe_counter_bomb_resistant_elimination_plan_is_selected(self):
        env = followup_trap_env()
        plan = self.find_plan(env)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.first_action, ACTION_BOMB)
        self.assertEqual(plan.bomb_positions, ((3, 3),))
        self.assertGreaterEqual(plan.pressure_gain, 0.35)
        self.assertGreaterEqual(plan.completion_tick, 0)
        self.assertLessEqual(plan.completion_tick, 8)
        self.assertGreaterEqual(plan.escape_margin, 0)
        self.assertIs(plan.predicts_elimination, True)

    def test_pressure_threshold_boundary_is_inclusive(self):
        env = followup_trap_env()
        baseline = self.find_plan(env, min_pressure_gain=0.0)
        self.assertIsNotNone(baseline)

        at_boundary = self.find_plan(env, min_pressure_gain=baseline.pressure_gain)
        above_boundary = self.find_plan(
            env,
            min_pressure_gain=float(np.nextafter(baseline.pressure_gain, np.inf)),
        )
        self.assertEqual(at_boundary, baseline)
        self.assertIsNone(above_boundary)

    def test_ranking_prefers_larger_pressure_before_other_non_eliminating_plans(self):
        env = pressure_ranking_env()
        plan = self.find_plan(
            env,
            horizon=4,
            max_bombs=1,
            min_pressure_gain=0.2,
        )
        self.assertIsNotNone(plan)
        self.assertIs(plan.predicts_elimination, False)
        self.assertEqual(plan.first_action, ACTION_RIGHT)
        self.assertEqual(plan.bomb_positions, ((5, 4),))
        from test_temporal_trap_regressions import refuge_count
        actions = [ACTION_RIGHT, ACTION_BOMB] + [ACTION_STOP] * (env.bomb_timer - 1)
        baseline = actions.copy()
        baseline[1] = ACTION_STOP
        before = refuge_count(env, baseline)
        after = refuge_count(env, actions)
        self.assertGreater(before, 0)
        self.assertLess(after, before)
        self.assertAlmostEqual(plan.pressure_gain, 1.0 - after / before)

    def test_ranking_prefers_earlier_completion_when_pressure_is_equal(self):
        env = completion_ranking_env()
        plan = self.find_plan(
            env,
            horizon=4,
            max_bombs=1,
            min_pressure_gain=0.2,
        )
        self.assertIsNotNone(plan)
        self.assertIs(plan.predicts_elimination, False)
        self.assertEqual(plan.first_action, ACTION_LEFT)
        self.assertEqual(plan.bomb_positions, ((3, 1),))
        self.assertAlmostEqual(plan.pressure_gain, 0.5)

    def test_ranking_prefers_larger_escape_margin_after_equal_pressure_and_ticks(self):
        env = escape_margin_ranking_env()
        mask = legal_mask(env, 0)
        mask[ACTION_BOMB] = False
        plan = self.find_plan(
            env,
            mask=mask,
            horizon=2,
            max_bombs=1,
            min_pressure_gain=0.3,
        )
        self.assertIsNotNone(plan)
        self.assertIs(plan.predicts_elimination, False)
        self.assertEqual(plan.first_action, ACTION_LEFT)
        self.assertEqual(plan.bomb_positions, ((3, 2),))
        self.assertAlmostEqual(plan.pressure_gain, 1.0 / 3.0)
        self.assertGreater(plan.escape_margin, 0)

    def test_two_new_bombs_include_movement_and_second_placement_ticks(self):
        env = adjacent_second_bomb_unsafe_env()
        mask = [False] * 6
        mask[ACTION_BOMB] = True
        mask[ACTION_UP] = True
        plan = self.find_plan(
            env,
            mask=mask,
            horizon=8,
            max_bombs=2,
            min_pressure_gain=0.0,
        )
        if plan is None or len(plan.bomb_positions) < 2:
            return

        first, second = plan.bomb_positions
        if plan.first_action == ACTION_BOMB:
            movement_ticks = abs(first[0] - second[0]) + abs(first[1] - second[1])
            self.assertGreaterEqual(
                plan.completion_tick,
                movement_ticks + 1,
                "The second bomb action needs a tick after reaching its distinct cell",
            )

    def test_two_new_bombs_are_survivable_under_one_combined_forecast(self):
        env = adjacent_second_bomb_unsafe_env()
        mask = [False] * 6
        mask[ACTION_BOMB] = True
        mask[ACTION_UP] = True
        first_position = (5, 3)
        second_position = (4, 3)
        agent = env.agents[0]

        first_bomb = Bomb(
            first_position[0], first_position[1], 0,
            env.bomb_timer, agent.blast_range,
        )
        first_safe, _, _ = _survival_map(
            env,
            first_position[0],
            first_position[1],
            first_bomb,
            safety_margin=2,
            reserve_exits=True,
        )

        second_tick = 2
        second_only = deepcopy(env)
        second_only.agents[0].row, second_only.agents[0].col = second_position
        second_only.step_count += second_tick
        second_bomb = Bomb(
            second_position[0], second_position[1], 0,
            second_only.bomb_timer, agent.blast_range,
        )
        second_safe, _, _ = _survival_map(
            second_only,
            second_position[0],
            second_position[1],
            second_bomb,
            safety_margin=2,
            reserve_exits=True,
        )

        combined = deepcopy(second_only)
        combined.bombs.append(Bomb(
            first_position[0], first_position[1], 0,
            env.bomb_timer - second_tick, agent.blast_range,
        ))
        combined.agents[0].bombs_placed = 1
        combined_safe, _, _ = _survival_map(
            combined,
            second_position[0],
            second_position[1],
            second_bomb,
            safety_margin=2,
            reserve_exits=True,
        )
        self.assertTrue(first_safe[0, first_position[0], first_position[1]])
        self.assertTrue(second_safe[0, second_position[0], second_position[1]])
        self.assertFalse(combined_safe[0, second_position[0], second_position[1]])

        plan = self.find_plan(
            env,
            mask=mask,
            horizon=8,
            max_bombs=2,
            min_pressure_gain=0.0,
        )
        if plan is not None and plan.bomb_positions == (first_position, second_position):
            actual_second_tick = max(plan.completion_tick, second_tick)
            if actual_second_tick < env.bomb_timer:
                simulated = deepcopy(env)
                simulated.agents[0].row, simulated.agents[0].col = second_position
                simulated.step_count += actual_second_tick
                simulated.bombs.append(Bomb(
                    first_position[0], first_position[1], 0,
                    env.bomb_timer - actual_second_tick, agent.blast_range,
                ))
                simulated.agents[0].bombs_placed = 1
                proposed = Bomb(
                    second_position[0], second_position[1], 0,
                    simulated.bomb_timer, agent.blast_range,
                )
                safe, _, _ = _survival_map(
                    simulated,
                    second_position[0],
                    second_position[1],
                    proposed,
                    safety_margin=2,
                    reserve_exits=True,
                )
                self.assertTrue(
                    safe[0, second_position[0], second_position[1]],
                    "A returned pair must survive the forecast containing both active bombs",
                )

    def test_escape_margin_never_counts_walls_or_disconnected_cells(self):
        env = completion_ranking_env()
        plan = self.find_plan(
            env,
            horizon=4,
            max_bombs=1,
            min_pressure_gain=0.2,
        )
        self.assertIsNotNone(plan)
        start = plan.bomb_positions[-1]
        reachable = {start}
        frontier = [start]
        while frontier:
            row, col = frontier.pop(0)
            for delta_row, delta_col in _MOVE_DELTA.values():
                position = (row + delta_row, col + delta_col)
                if position in reachable or not env._is_walkable(*position):
                    continue
                reachable.add(position)
                frontier.append(position)
        self.assertLessEqual(
            plan.escape_margin,
            len(reachable),
            "Escape margin cannot exceed the learner's connected walkable region",
        )

    def test_timing_or_joint_safety_rejection_does_not_increment_duplicate_diagnostics(self):
        planner = self.planner_module()
        env = adjacent_second_bomb_unsafe_env()
        config = planner.TemporalTrapConfig(
            horizon=8,
            max_bombs=2,
            min_pressure_gain=0.0,
        )

        bomb_only = [False] * 6
        bomb_only[ACTION_BOMB] = True
        baseline = {}
        planner.find_temporal_trap_plan(
            env, 0, env._danger_map(), bomb_only, config, diagnostics=baseline
        )

        pair_mask = list(bomb_only)
        pair_mask[ACTION_UP] = True
        diagnostics = {}
        plan = planner.find_temporal_trap_plan(
            env, 0, env._danger_map(), pair_mask, config, diagnostics=diagnostics
        )
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.bomb_positions), 2)
        self.assertNotEqual(plan.bomb_positions, ((5, 3), (4, 3)),
                            "The independently unsafe adjacent pair must stay rejected")
        from test_temporal_trap_regressions import real_pair_gain_exists
        self.assertTrue(real_pair_gain_exists(env, plan),
                        "A selected alternative must have a real safe pressure gain")
        self.assertGreaterEqual(diagnostics.get("temporal_trap_rejected_unsafe", 0), 1)
        self.assertEqual(
            diagnostics.get("temporal_trap_rejected_duplicate", 0),
            baseline.get("temporal_trap_rejected_duplicate", 0) + 1,
            "Only the additional same-position candidate is a duplicate",
        )

    def test_planner_and_interception_do_not_mutate_state_or_rng(self):
        planner = self.planner_module()
        env = followup_trap_env()
        env.flames[(1, 1)] = 2
        env._flame_owners[(1, 1)] = 1
        env._flame_sources[(1, 1)] = {1: 2}
        env.powerups[(2, 1)] = POWERUP_AMMO
        mask = legal_mask(env, 0)

        environment_before = pickle.dumps(env, protocol=5)
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()

        planner.find_temporal_trap_plan(
            env, 0, env._danger_map(), mask, planner.TemporalTrapConfig()
        )
        planner.find_interception_action(env, 0, env._danger_map(), mask)

        self.assertEqual(pickle.dumps(env, protocol=5), environment_before)
        self.assertTrue(numpy_state_equal(np.random.get_state(), numpy_before))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))

    def test_determinism_and_horizontal_mirror_equivalence(self):
        env = followup_trap_env()
        mask = legal_mask(env, 0)
        plans = [self.find_plan(env, mask=mask) for _ in range(4)]
        self.assertIsNotNone(plans[0])
        self.assertTrue(all(plan == plans[0] for plan in plans[1:]))

        mirrored = mirror_env(env)
        mirrored_plan = self.find_plan(mirrored, mask=mirror_mask(mask))
        self.assertIsNotNone(mirrored_plan)
        self.assertEqual(mirrored_plan.first_action, MIRROR_ACTION[plans[0].first_action])
        self.assertEqual(
            mirrored_plan.bomb_positions,
            tuple((row, env.board_size - 1 - col) for row, col in plans[0].bomb_positions),
        )
        self.assertAlmostEqual(mirrored_plan.pressure_gain, plans[0].pressure_gain)
        self.assertEqual(mirrored_plan.completion_tick, plans[0].completion_tick)
        self.assertEqual(mirrored_plan.escape_margin, plans[0].escape_margin)
        self.assertEqual(mirrored_plan.predicts_elimination, plans[0].predicts_elimination)

    def test_plan_positions_are_unique_reachable_and_within_capacity(self):
        env = followup_trap_env()
        plan = self.find_plan(env)
        self.assertIsNotNone(plan)
        remaining_capacity = env.agents[0].ammo - env.agents[0].bombs_placed
        self.assertLessEqual(len(plan.bomb_positions), 2)
        self.assertLessEqual(len(plan.bomb_positions), remaining_capacity)
        self.assertEqual(len(plan.bomb_positions), len(set(plan.bomb_positions)))
        self.assertTrue(set(plan.bomb_positions) <= self.reachable_positions(env, 8))

        exhausted = followup_trap_env()
        exhausted.agents[0].bombs_placed = exhausted.agents[0].ammo
        self.assertIsNone(self.find_plan(exhausted))

    def test_unreachable_or_duplicate_pressure_followup_is_rejected(self):
        env = redundant_followup_env()
        mask = legal_mask(env, 0)
        self.assertTrue(mask[ACTION_BOMB])
        self.assertTrue(mask[ACTION_LEFT])
        self.assertIsNone(
            self.find_plan(
                env,
                mask=mask,
                horizon=8,
                max_bombs=2,
                min_pressure_gain=0.9,
            )
        )

    def test_unsafe_first_placement_is_rejected_even_if_mask_allows_bomb(self):
        env = followup_trap_env()
        env.bombs[0].timer = 1
        danger = env._danger_map()
        self.assertEqual(danger[3, 3], 1)
        mask = [False] * 6
        mask[ACTION_BOMB] = True
        planner = self.planner_module()
        self.assertIsNone(
            planner.find_temporal_trap_plan(
                env, 0, danger, mask, planner.TemporalTrapConfig(min_pressure_gain=0.0)
            )
        )

    def test_simultaneous_counter_bomb_unsafe_candidate_is_rejected(self):
        env = counter_bomb_unsafe_env()
        danger = env._danger_map()
        position = (env.agents[0].row, env.agents[0].col)
        self.assertTrue(_can_bomb_and_escape(env, 0, *position, danger, robust=False))
        self.assertFalse(_can_bomb_and_escape(env, 0, *position, danger, robust=True))
        mask = [False] * 6
        mask[ACTION_BOMB] = True
        self.assertIsNone(
            self.find_plan(
                env,
                mask=mask,
                max_bombs=1,
                min_pressure_gain=0.0,
            )
        )

    def test_path_closing_under_sudden_death_is_rejected_before_arrival(self):
        safe_env = sudden_death_route_env(enabled=False)
        closing_env = sudden_death_route_env(enabled=True)
        mask = [False] * 6
        mask[ACTION_UP] = True

        safe_plan = self.find_plan(
            safe_env,
            mask=mask,
            horizon=8,
            max_bombs=1,
            min_pressure_gain=0.35,
        )
        self.assertIsNotNone(safe_plan)
        self.assertEqual(safe_plan.first_action, ACTION_UP)
        self.assertIn((1, 3), safe_plan.bomb_positions)
        self.assertEqual(closing_env._next_sudden_death_closure()[0], 2)
        self.assertIsNone(
            self.find_plan(
                closing_env,
                mask=mask,
                horizon=8,
                max_bombs=1,
                min_pressure_gain=0.35,
            )
        )


class InterceptionAcceptance(PlannerContractCase):
    def test_interception_requires_strict_safe_choke_improvement(self):
        planner = self.planner_module()
        corridor = interception_corridor_env()
        corridor_mask = legal_mask(corridor, 0)
        corridor_mask[ACTION_BOMB] = False
        action = planner.find_interception_action(
            corridor, 0, corridor._danger_map(), corridor_mask, horizon=4
        )
        self.assertEqual(action, ACTION_RIGHT)
        self.assertNotEqual(action, ACTION_BOMB)
        self.assertTrue(corridor_mask[action])

        open_env = arena()
        open_env.agents[0].row, open_env.agents[0].col = 3, 1
        open_env.agents[1].row, open_env.agents[1].col = 3, 5
        open_mask = legal_mask(open_env, 0)
        open_mask[ACTION_BOMB] = False
        self.assertIsNone(
            planner.find_interception_action(
                open_env, 0, open_env._danger_map(), open_mask, horizon=4
            )
        )

        unsafe = interception_corridor_env()
        put_bomb(unsafe, 3, 3, 1, timer=1, blast_range=2)
        unsafe_mask = [False] * 6
        unsafe_mask[ACTION_RIGHT] = True
        self.assertEqual(unsafe._danger_map()[3, 2], 1)
        self.assertIsNone(
            planner.find_interception_action(
                unsafe, 0, unsafe._danger_map(), unsafe_mask, horizon=4
            )
        )


class AttackTeacherPriorityAcceptance(PlannerContractCase):
    def integration_mocks(self, *, temporal_action=None, interception_action=None,
                          ring_action=None, item_action=ACTION_DOWN,
                          opponent_action=ACTION_LEFT):
        planner = self.planner_module()
        temporal_result = (
            None
            if temporal_action is None
            else SimpleNamespace(first_action=temporal_action)
        )
        temporal_mock = Mock(return_value=temporal_result)
        interception_mock = Mock(return_value=interception_action)
        stack = [
            patch.object(planner, "find_temporal_trap_plan", new=temporal_mock),
            patch.object(planner, "find_interception_action", new=interception_mock),
            patch.object(training, "find_temporal_trap_plan", new=temporal_mock, create=True),
            patch.object(training, "find_interception_action", new=interception_mock, create=True),
            patch.object(training, "_ring_safety_action", return_value=ring_action),
            patch.object(training, "_sequence_attack_action", return_value=None),
            patch.object(training, "_pressure_attack_action", return_value=None),
            patch.object(training, "_bfs_toward_item", return_value=item_action),
            patch.object(training, "_bfs_toward_opponent", return_value=opponent_action),
            patch.object(training, "_bfs_toward_wood", return_value=None),
            patch.object(training, "_bfs_safe_cell", return_value=ACTION_LEFT),
        ]
        return planner, temporal_mock, interception_mock, stack

    def run_with_mocks(self, env, mask, danger, **mock_options):
        planner, temporal_mock, interception_mock, stack = self.integration_mocks(**mock_options)
        entered = []
        try:
            for manager in stack:
                entered.append(manager)
                manager.start()
            action = training.get_attack_action(env, 0, danger, mask)
        finally:
            for manager in reversed(entered):
                manager.stop()
        return planner, temporal_mock, interception_mock, action

    def test_urgent_escape_precedes_all_attack_teachers(self):
        env = arena()
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 5
        danger = np.zeros_like(env.grid, dtype=np.int32)
        danger[3, 3] = 1
        mask = [False] * 6
        mask[ACTION_LEFT] = True
        _, temporal_mock, interception_mock, action = self.run_with_mocks(
            env,
            mask,
            danger,
            temporal_action=ACTION_RIGHT,
            interception_action=ACTION_DOWN,
            ring_action=ACTION_UP,
        )
        self.assertEqual(action, ACTION_LEFT)
        temporal_mock.assert_not_called()
        interception_mock.assert_not_called()

    def test_ring_safety_precedes_temporal_and_interception(self):
        env = arena()
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 5
        danger = np.zeros_like(env.grid, dtype=np.int32)
        mask = [True] * 6
        _, _, _, action = self.run_with_mocks(
            env,
            mask,
            danger,
            ring_action=ACTION_RIGHT,
            temporal_action=ACTION_DOWN,
            interception_action=ACTION_LEFT,
        )
        self.assertEqual(action, ACTION_RIGHT)

    def test_temporal_plan_precedes_interception_and_existing_guidance(self):
        env = arena()
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 5
        mask = [True] * 6
        _, temporal_mock, interception_mock, action = self.run_with_mocks(
            env,
            mask,
            np.zeros_like(env.grid, dtype=np.int32),
            temporal_action=ACTION_RIGHT,
            interception_action=ACTION_LEFT,
            item_action=ACTION_DOWN,
        )
        self.assertEqual(action, ACTION_RIGHT)
        temporal_mock.assert_called_once()
        interception_mock.assert_not_called()

    def test_interception_precedes_guidance_and_fallbacks_remain_available(self):
        env = arena()
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 5
        mask = [True] * 6
        mask[ACTION_BOMB] = False
        danger = np.zeros_like(env.grid, dtype=np.int32)

        _, _, interception_mock, interception = self.run_with_mocks(
            env,
            mask,
            danger,
            temporal_action=None,
            interception_action=ACTION_LEFT,
            item_action=ACTION_DOWN,
        )
        self.assertEqual(interception, ACTION_LEFT)
        interception_mock.assert_called_once()

        _, _, _, upgrade = self.run_with_mocks(
            env,
            mask,
            danger,
            temporal_action=None,
            interception_action=None,
            item_action=ACTION_DOWN,
            opponent_action=ACTION_LEFT,
        )
        self.assertEqual(upgrade, ACTION_DOWN)

        _, _, _, legacy_navigation = self.run_with_mocks(
            env,
            mask,
            danger,
            temporal_action=None,
            interception_action=None,
            item_action=None,
            opponent_action=ACTION_LEFT,
        )
        self.assertEqual(legacy_navigation, ACTION_LEFT)


class _ValidationReachedEnvironment(Exception):
    pass


class TemporalTrainerAcceptance(unittest.TestCase):
    expected_defaults = {
        "temporal_trap_horizon": 8,
        "temporal_trap_max_bombs": 2,
        "interception_horizon": 4,
    }

    def test_train_api_exposes_frozen_temporal_defaults(self):
        parameters = inspect.signature(training.train).parameters
        missing = set(self.expected_defaults) - parameters.keys()
        self.assertFalse(missing, f"train is missing temporal options: {sorted(missing)}")
        for name, expected in self.expected_defaults.items():
            with self.subTest(name=name):
                self.assertEqual(parameters[name].default, expected)
        self.assertEqual(parameters["sequence_pressure_gain"].default, 0.35)

    def test_training_cli_exposes_temporal_flags(self):
        root = Path(__file__).resolve().parent
        completed = subprocess.run(
            [sys.executable, str(root / "train_actor_critic.py"), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for name in self.expected_defaults:
            with self.subTest(name=name):
                self.assertIn(f"--{name}", completed.stdout)

    def test_training_cli_persists_frozen_temporal_defaults(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=root) as directory:
            checkpoint_dir = Path(directory) / "tiny-cli-run"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "train_actor_critic.py"),
                    "--episodes", "1",
                    "--board_size", "7",
                    "--num_agents", "2",
                    "--max_steps", "1",
                    "--log_every", "1",
                    "--validation_every", "0",
                    "--seed", "20261015",
                    "--checkpoint_dir", str(checkpoint_dir),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        for name, expected in self.expected_defaults.items():
            with self.subTest(name=name):
                self.assertIn(name, config)
                self.assertEqual(config[name], expected)
        self.assertEqual(config["sequence_pressure_gain"], 0.35)

    def call_until_environment(self, **overrides):
        options = {
            "episodes": 1,
            "board_size": 7,
            "num_agents": 2,
            "max_steps": 1,
            "log_every": 1,
            "snapshot_every": 1,
            "validation_every": 0,
            "run_report": False,
            "temporal_trap_horizon": 8,
            "temporal_trap_max_bombs": 2,
            "interception_horizon": 4,
            "sequence_pressure_gain": 0.35,
        }
        options.update(overrides)
        with patch.object(
            training,
            "BombermanFFAEnv",
            side_effect=_ValidationReachedEnvironment("validation accepted"),
        ):
            try:
                training.train(**options)
            except Exception as error:
                return error
        self.fail("train returned before constructing the environment")

    def test_training_rejects_invalid_temporal_configuration_before_environment(self):
        cases = (
            ({"temporal_trap_horizon": 0}, r"temporal.*horizon|horizon"),
            ({"temporal_trap_max_bombs": 1}, r"max.*bomb|bomb"),
            ({"temporal_trap_max_bombs": 3}, r"max.*bomb|bomb"),
            ({"interception_horizon": 0}, r"interception.*horizon|horizon"),
            ({"sequence_pressure_gain": -1e-9}, r"pressure"),
        )
        for options, message_pattern in cases:
            with self.subTest(options=options):
                error = self.call_until_environment(**options)
                self.assertIsInstance(error, ValueError, str(error))
                self.assertRegex(str(error), message_pattern)

    def test_pressure_threshold_accepts_zero_and_values_above_one(self):
        for threshold in (0.0, 1.25):
            with self.subTest(threshold=threshold):
                error = self.call_until_environment(sequence_pressure_gain=threshold)
                self.assertIsInstance(error, _ValidationReachedEnvironment, str(error))

    def tiny_metrics_run(self, attack_action, metrics_path):
        observed = []
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            def write_metrics(payload):
                observed.append(payload)
                metrics_file.write(json.dumps(payload, allow_nan=False) + "\n")
                metrics_file.flush()

            with patch.object(training, "get_attack_action", return_value=attack_action), \
                 patch.object(training.torch, "save"), \
                 patch.object(training.os, "makedirs"), \
                 patch("builtins.print"):
                _, rewards, _ = training.train(
                    episodes=1,
                    board_size=7,
                    num_agents=2,
                    max_steps=1,
                    device="cpu",
                    seed=20261015,
                    checkpoint_dir="unused-temporal-trainer-acceptance",
                    run_report=False,
                    validation_every=0,
                    on_episode=write_metrics,
                )
        self.assertEqual(len(observed), 1)
        logged = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(logged, observed)
        return observed[0], rewards

    def test_episode_and_metrics_jsonl_publish_numeric_temporal_diagnostics_without_reward_change(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=root) as directory:
            directory = Path(directory)
            without_label, rewards_without = self.tiny_metrics_run(
                None, directory / "without-label.metrics.jsonl"
            )
            with_label, rewards_with = self.tiny_metrics_run(
                ACTION_RIGHT, directory / "with-label.metrics.jsonl"
            )

        self.assertEqual(rewards_with, rewards_without)
        self.assertEqual(with_label["reward"], without_label["reward"])
        for payload in (without_label, with_label):
            json.dumps(payload, allow_nan=False)
            for field in TRAINER_METRICS:
                with self.subTest(field=field, label=payload is with_label):
                    self.assertIn(field, payload)
                    self.assertIn(type(payload[field]), (int, float))

    def test_interception_source_remains_navigation_when_legacy_action_matches(self):
        metrics = []

        def interception_teacher(*_args, diagnostics=None, **_kwargs):
            diagnostics["source"] = "interception"
            diagnostics["interception_labels"] += 1
            return ACTION_RIGHT

        def no_danger(env):
            return np.zeros((env.board_size, env.board_size), dtype=np.int32)

        with patch.object(training, "get_attack_action", side_effect=interception_teacher), \
             patch.object(training, "_sequence_attack_action", return_value=ACTION_RIGHT), \
             patch.object(training, "_pressure_attack_action", return_value=None), \
             patch.object(training, "_ring_safety_action", return_value=None), \
             patch.object(training, "_build_danger_map", side_effect=no_danger), \
             patch.object(training.torch, "save"), \
             patch.object(training.os, "makedirs"), \
             patch("builtins.print"):
            training.train(
                episodes=1,
                board_size=7,
                num_agents=2,
                max_steps=1,
                device="cpu",
                seed=20261016,
                checkpoint_dir="unused-interception-source-acceptance",
                run_report=False,
                validation_every=0,
                on_episode=metrics.append,
            )

        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["interception_labels"], 1)
        self.assertEqual(metrics[0]["combat_labels"], 0)
        self.assertEqual(metrics[0]["combat_loss"], 0.0)
        self.assertGreater(metrics[0]["navigation_loss"], 0.0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
