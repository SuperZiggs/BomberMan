"""Independent behavioral regressions for delayed temporal trap placements."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import subprocess
import unittest

import numpy as np

from bomberman_env import (
    ACTION_BOMB, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP,
    ACTION_UP, Bomb, EMPTY, _MOVE_DELTA,
)
from rule_based_bot import _can_bomb_and_escape, _reachable_cells, _survival_map
from temporal_trap_planner import (
    TemporalTrapConfig, _PathNode, _combined_pressures, _pair_survival,
    _placement_safety,
    _reachable_escape_margin, find_temporal_trap_plan,
)
from test_temporal_trap_planner import arena, carve, legal_mask, mirror_env, mirror_mask
import test_phase_c_sd_temporal_250 as launcher_fixtures


def open_pair_arena(*, bomb_timer=6, enemy=(2, 5)):
    env = arena(board_size=7, bomb_timer=bomb_timer)
    env.grid[1:-1, 1:-1] = EMPTY
    env.agents[0].row, env.agents[0].col = 3, 2
    env.agents[0].ammo = 2
    env.agents[0].blast_range = 2
    env.agents[1].row, env.agents[1].col = enemy
    env.agents[1].ammo = 0
    return env


def real_tick_forecast(env, actions):
    """Observe each next tick in a real environment; only place on BOMB actions."""
    simulated = deepcopy(env)
    burns = []
    blocked = []
    positions = []
    for action in actions:
        agent = simulated.agents[0]
        proposed = (
            Bomb(agent.row, agent.col, 0, simulated.bomb_timer, agent.blast_range)
            if action == ACTION_BOMB else None
        )
        future_burns, future_blocked = simulated._forecast_blasts(proposed)
        burns.append(future_burns[0].copy())
        blocked.append(future_blocked[0].copy())
        simulated.step([action, ACTION_STOP])
        positions.append((agent.row, agent.col))
    return np.stack(burns), np.stack(blocked), simulated, positions


def refuge_count(env, actions):
    burns, blocked, _, _ = real_tick_forecast(env, actions)
    opponent = env.agents[1]
    return int(np.count_nonzero(_reachable_cells(
        (opponent.row, opponent.col), burns, blocked, len(actions)
    )))


def real_pair_gain_exists(env, plan):
    """Search real legal timelines matching public plan fields, without assuming its path."""
    first, second = plan.bomb_positions
    action_count = plan.completion_tick + 1
    if action_count > 8 or action_count < 4:
        return False
    frontier = [(deepcopy(env), [], 0, -1)]
    for tick in range(action_count):
        next_frontier = {}
        for state, actions, stage, first_tick in frontier:
            position = (state.agents[0].row, state.agents[0].col)
            if tick == 0:
                choices = [plan.first_action]
            elif tick == action_count - 1:
                choices = [ACTION_BOMB] if stage == 1 and position == second else []
            else:
                choices = [ACTION_STOP, *_MOVE_DELTA]
                if stage == 0 and position == first:
                    choices.append(ACTION_BOMB)
            for action in choices:
                next_state = deepcopy(state)
                next_state.step([action, ACTION_STOP])
                next_position = (next_state.agents[0].row, next_state.agents[0].col)
                if not next_state.agents[0].alive:
                    continue
                if action in _MOVE_DELTA and next_position == position:
                    continue
                if (action == ACTION_BOMB and not any(
                        bomb.owner == 0 and (bomb.row, bomb.col) == position
                        for bomb in next_state.bombs)):
                    continue
                next_stage = stage + (action == ACTION_BOMB)
                next_first_tick = tick if stage == 0 and action == ACTION_BOMB else first_tick
                next_actions = actions + [action]
                if next_stage == 2:
                    safe, _, _ = _survival_map(
                        next_state, *next_position, safety_margin=0,
                    )
                    if not safe[0, *next_position]:
                        continue
                    detonation_step = len(next_actions) - 1 + env.bomb_timer
                    for horizon in range(detonation_step,
                                         detonation_step + env.flame_duration):
                        padded = next_actions + [ACTION_STOP] * (horizon - len(next_actions))
                        first_only = padded.copy()
                        first_only[len(next_actions) - 1] = ACTION_STOP
                        before = refuge_count(env, first_only)
                        after = refuge_count(env, padded)
                        if before > 0 and after < before:
                            return True
                    continue
                if next_stage > 1:
                    continue
                active_bombs = tuple((b.row, b.col, b.timer) for b in next_state.bombs)
                key = (next_position, next_stage, next_first_tick, active_bombs,
                       next_state.agents[1].alive)
                next_frontier.setdefault(key,
                                         (next_state, next_actions, next_stage, next_first_tick))
        frontier = list(next_frontier.values())
    return False


def assert_real_second_gain(case, env, plan):
    case.assertTrue(
        real_pair_gain_exists(env, plan),
        "A real safe action timeline must match the plan and gain pressure from bomb two",
    )


class TemporalPlacementRegression(unittest.TestCase):
    def test_expired_existing_bomb_is_absent_at_future_placement(self):
        env = open_pair_arena(bomb_timer=6, enemy=(5, 5))
        env.agents[0].ammo = 1
        env.bombs.append(Bomb(1, 1, 1, 1, 2))
        env.agents[1].bombs_placed = 1
        node = _PathNode(
            (1, 3), 3, ACTION_UP,
            ((3, 2), (2, 2), (2, 3), (1, 3)),
        )

        replay = deepcopy(env)
        for action in (ACTION_UP, ACTION_RIGHT, ACTION_UP):
            replay.step([action, ACTION_STOP])
        self.assertTrue(replay.agents[0].alive)
        self.assertEqual((replay.agents[0].row, replay.agents[0].col), (1, 3))
        self.assertEqual(replay.bombs, [])
        self.assertEqual(replay.flames, {})
        self.assertTrue(_can_bomb_and_escape(
            replay, 0, 1, 3, replay._danger_map(), robust=True,
        ))

        never_bombed = deepcopy(env)
        never_bombed.bombs.clear()
        never_bombed.agents[1].bombs_placed = 0
        self.assertTrue(_placement_safety(never_bombed, 0, node)[0])
        self.assertTrue(_placement_safety(env, 0, node)[0],
                        "An exploded bomb cannot threaten a later placement")

    def test_expired_enemy_bomb_restores_counter_bomb_capacity(self):
        env = open_pair_arena(bomb_timer=6, enemy=(2, 3))
        env.agents[0].ammo = 1
        env.agents[1].ammo = 1
        env.bombs.append(Bomb(3, 1, 1, 1, 2))
        env.agents[1].bombs_placed = 1
        node = _PathNode(
            (1, 3), 3, ACTION_UP,
            ((3, 2), (2, 2), (1, 2), (1, 3)),
        )

        replay = deepcopy(env)
        for action in (ACTION_UP, ACTION_UP, ACTION_RIGHT):
            replay.step([action, ACTION_STOP])
        self.assertTrue(replay.agents[0].alive)
        self.assertEqual((replay.agents[0].row, replay.agents[0].col), (1, 3))
        self.assertEqual(replay.bombs, [])
        self.assertEqual(replay.agents[1].bombs_placed, 0)
        self.assertTrue(_can_bomb_and_escape(
            replay, 0, 1, 3, replay._danger_map(), robust=False,
        ))
        self.assertFalse(_can_bomb_and_escape(
            replay, 0, 1, 3, replay._danger_map(), robust=True,
        ))
        self.assertFalse(_placement_safety(env, 0, node)[0],
                         "Future safety must account for restored enemy bomb capacity")

    def test_future_placement_applies_sudden_death_closure(self):
        def closing_arena(enabled):
            env = arena(
                board_size=9, bomb_timer=6, sudden_death=enabled,
                sudden_death_start=475, sudden_death_interval=25,
            )
            carve(env, {(2, 2), (2, 3), (1, 2), (7, 7)})
            env.agents[0].row, env.agents[0].col = 2, 2
            env.agents[0].blast_range = 1
            env.agents[1].row, env.agents[1].col = 7, 7
            env.agents[1].ammo = 0
            env.step_count = 474
            return env

        ordinary = closing_arena(False)
        closing = closing_arena(True)
        node = _PathNode((2, 3), 1, ACTION_RIGHT, ((2, 2), (2, 3)))
        self.assertTrue(_placement_safety(ordinary, 0, node)[0])
        real_tick = deepcopy(closing)
        real_tick.step([ACTION_RIGHT, ACTION_STOP])
        self.assertEqual((real_tick.agents[0].row, real_tick.agents[0].col), (2, 3))
        self.assertTrue(real_tick.agents[0].alive)
        self.assertFalse(real_tick._is_walkable(1, 2))
        self.assertFalse(_can_bomb_and_escape(
            real_tick, 0, 2, 3, real_tick._danger_map(), robust=True,
        ))
        self.assertFalse(_placement_safety(closing, 0, node)[0],
                         "Future safety must include the ring that closes on arrival")

        mask = [False] * 6
        mask[ACTION_RIGHT] = True
        config = TemporalTrapConfig(horizon=8, max_bombs=1, min_pressure_gain=0.0)
        self.assertIsNotNone(find_temporal_trap_plan(
            ordinary, 0, ordinary._danger_map(), mask, config,
        ))
        self.assertIsNone(find_temporal_trap_plan(
            closing, 0, closing._danger_map(), mask, config,
        ))

    def test_future_pair_survival_ignores_old_learner_position(self):
        env = arena(board_size=7, bomb_timer=8)
        carve(env, {(2, 2), (2, 3), (3, 3), (3, 2), (3, 1), (4, 1), (5, 5)})
        env.agents[0].row, env.agents[0].col = 2, 2
        env.agents[0].ammo = 2
        env.agents[0].blast_range = 1
        env.agents[1].row, env.agents[1].col = 5, 5
        env.agents[1].ammo = 0

        route = [
            ACTION_BOMB, ACTION_RIGHT, ACTION_BOMB, ACTION_DOWN,
            ACTION_LEFT, ACTION_LEFT, ACTION_STOP, ACTION_STOP,
            ACTION_STOP, ACTION_STOP,
        ]
        _, _, replay, positions = real_tick_forecast(env, route)
        self.assertEqual(positions[:6],
                         [(2, 2), (2, 3), (2, 3), (3, 3), (3, 2), (3, 1)])
        self.assertTrue(replay.agents[0].alive,
                        "The real two-placement route survives both explosions")

        placements = [Bomb(2, 2, 0, 8, 1), Bomb(2, 3, 0, 10, 1)]
        future_state = deepcopy(env)
        future_state.agents[0].row, future_state.agents[0].col = 2, 3
        future_safe, _, _ = _survival_map(
            future_state, 2, 3, placements, safety_margin=2, reserve_exits=True,
        )
        self.assertTrue(future_safe[2, 2, 3],
                        "The future-position safety forecast accepts the route")

        first = _PathNode((2, 2), 0, ACTION_BOMB, ((2, 2),))
        second = _PathNode((2, 3), 1, ACTION_BOMB, ((2, 2), (2, 3)))
        survives, margin, completion = _pair_survival(env, 0, first, second, 8)
        self.assertEqual(completion, 2)
        self.assertTrue(survives,
                        "The old learner position must not veto its future escape route")
        self.assertGreater(margin, 0)

    def test_move_then_single_pressure_matches_real_placement_tick(self):
        env = open_pair_arena(bomb_timer=5, enemy=(1, 3))
        env.agents[0].ammo = 1
        mask = [False] * 6
        mask[ACTION_RIGHT] = True
        plan = find_temporal_trap_plan(
            env, 0, env._danger_map(), mask,
            TemporalTrapConfig(horizon=6, max_bombs=1, min_pressure_gain=0.0),
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.first_action, ACTION_RIGHT)
        self.assertEqual(plan.bomb_positions, ((3, 3),))
        self.assertEqual(plan.completion_tick, 1)

        delayed = [ACTION_RIGHT, ACTION_BOMB] + [ACTION_STOP] * 4
        baseline = delayed.copy()
        baseline[1] = ACTION_STOP
        before = refuge_count(env, baseline)
        after = refuge_count(env, delayed)
        self.assertEqual((before, after), (25, 16))
        self.assertAlmostEqual(plan.pressure_gain, 1.0 - after / before,
                               msg="Single-bomb pressure must use its actual placement tick")

    def test_two_bomb_budget_uses_safe_singleton_when_ammo_is_one(self):
        env = open_pair_arena(bomb_timer=5, enemy=(1, 3))
        env.agents[0].ammo = 1
        mask = [False] * 6
        mask[ACTION_RIGHT] = True
        delayed = [ACTION_RIGHT, ACTION_BOMB] + [ACTION_STOP] * 4
        baseline = delayed.copy()
        baseline[1] = ACTION_STOP
        real_gain = 1.0 - refuge_count(env, delayed) / refuge_count(env, baseline)
        self.assertGreaterEqual(real_gain, 0.35)
        safe_route = [
            ACTION_RIGHT, ACTION_BOMB, ACTION_DOWN,
            ACTION_RIGHT, ACTION_STOP, ACTION_STOP,
        ]
        _, _, replay, _ = real_tick_forecast(env, safe_route)
        self.assertTrue(replay.agents[0].alive)

        plan = find_temporal_trap_plan(
            env, 0, env._danger_map(), mask,
            TemporalTrapConfig(horizon=6, max_bombs=2, min_pressure_gain=0.35),
        )
        self.assertIsNotNone(plan,
                             "One free ammo slot must still allow a qualifying singleton")
        self.assertEqual(plan.first_action, ACTION_RIGHT)
        self.assertEqual(plan.bomb_positions, ((3, 3),))
        self.assertAlmostEqual(plan.pressure_gain, real_gain)

    def test_future_placement_matches_real_tick_timeline(self):
        env = open_pair_arena(bomb_timer=3)
        route = [
            ACTION_RIGHT, ACTION_BOMB, ACTION_DOWN, ACTION_RIGHT,
            ACTION_DOWN, ACTION_LEFT, ACTION_BOMB, ACTION_RIGHT,
            ACTION_UP, ACTION_STOP, ACTION_STOP,
        ]
        first_only = route.copy()
        first_only[6] = ACTION_STOP
        first_burns, first_blocked, _, _ = real_tick_forecast(env, first_only)
        pair_burns, pair_blocked, pair_env, positions = real_tick_forecast(env, route)

        self.assertEqual(positions[5], (5, 3))
        self.assertTrue(pair_env.agents[0].alive, "The real timed pair must be survivable")
        np.testing.assert_array_equal(pair_burns[:6], first_burns[:6])
        np.testing.assert_array_equal(pair_blocked[:6], first_blocked[:6])
        self.assertTrue(first_burns[3, 5, 3], "The earlier bomb reaches the future placement cell")
        self.assertFalse(pair_blocked[0, 5, 3], "The future bomb must not block before placement")
        self.assertTrue(pair_blocked[6, 5, 3], "The placed bomb blocks on its action tick")
        self.assertNotEqual(pair_burns[8].tolist(), first_burns[8].tolist())

        first = _PathNode((3, 3), 1, ACTION_RIGHT, ((3, 2), (3, 3)))
        second = _PathNode(
            (5, 3), 5, ACTION_RIGHT,
            ((3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (5, 3)),
        )
        first_pressure, pair_pressure = _combined_pressures(env, 0, first, second, 6)
        before = refuge_count(env, [ACTION_STOP if a == ACTION_BOMB else a for a in route])
        after_first = refuge_count(env, first_only)
        after_pair = refuge_count(env, route)
        self.assertGreater(before, 0)
        self.assertEqual(after_first, before)
        self.assertLess(after_pair, after_first)
        self.assertAlmostEqual(first_pressure, 1.0 - after_first / before)
        self.assertAlmostEqual(
            pair_pressure, 1.0 - after_pair / before,
            msg="A delayed placement must match the real-tick pressure timeline",
        )

    def test_move_then_two_distinct_placements_gain_pressure(self):
        env = open_pair_arena()
        route = [
            ACTION_RIGHT, ACTION_BOMB, ACTION_DOWN, ACTION_RIGHT,
            ACTION_BOMB, ACTION_DOWN, ACTION_STOP, ACTION_STOP,
            ACTION_LEFT, ACTION_STOP,
        ]
        first_only = route.copy()
        first_only[4] = ACTION_STOP
        _, _, replay, positions = real_tick_forecast(env, route)
        self.assertTrue(replay.agents[0].alive)
        self.assertEqual((positions[0], positions[1], positions[3], positions[4]),
                         ((3, 3), (3, 3), (4, 4), (4, 4)))
        self.assertGreater(refuge_count(env, first_only), refuge_count(env, route))

        mask = [False] * 6
        mask[ACTION_RIGHT] = True
        before = pickle.dumps(env, protocol=5)
        config = TemporalTrapConfig(horizon=8, max_bombs=2, min_pressure_gain=0.01)
        plans = [find_temporal_trap_plan(env, 0, env._danger_map(), mask, config)
                 for _ in range(2)]
        self.assertIsNotNone(plans[0], "The legal, safe two-placement arena needs a plan")
        self.assertEqual(plans[0], plans[1], "Selection must be deterministic")
        self.assertEqual(pickle.dumps(env, protocol=5), before,
                         "Planning must not mutate the live environment")
        plan = plans[0]
        self.assertEqual(plan.first_action, ACTION_RIGHT)
        self.assertEqual(len(plan.bomb_positions), 2)
        self.assertEqual(len(set(plan.bomb_positions)), 2)
        self.assertLessEqual(len(plan.bomb_positions), env.agents[0].ammo)
        self.assertGreaterEqual(plan.completion_tick, 3)
        self.assertLessEqual(plan.completion_tick, config.horizon)
        self.assertGreater(plan.pressure_gain, 0.0)
        self.assertGreater(plan.escape_margin, 0)
        assert_real_second_gain(self, env, plan)

    def test_existing_pair_scenarios_require_a_pair(self):
        for label, env in (
            ("original", open_pair_arena()),
            ("mirrored", mirror_env(open_pair_arena())),
        ):
            with self.subTest(label=label):
                mask = [False] * 6
                first_move = ACTION_RIGHT if label == "original" else ACTION_LEFT
                mask[first_move] = True
                plan = find_temporal_trap_plan(
                    env, 0, env._danger_map(), mask,
                    TemporalTrapConfig(horizon=8, max_bombs=2, min_pressure_gain=0.01),
                )
                self.assertIsNotNone(plan, f"{label}: None cannot pass a pair scenario")
                self.assertEqual(plan.first_action, first_move)
                self.assertEqual(len(plan.bomb_positions), 2)
                self.assertNotEqual(*plan.bomb_positions)
                self.assertGreater(plan.pressure_gain, 0.0)
                assert_real_second_gain(self, env, plan)

    def test_escape_margin_excludes_temporally_unreachable_refuge(self):
        safe = np.zeros((4, 5, 5), dtype=bool)
        blocked = np.ones_like(safe)
        start, crossing, refuge = (2, 1), (2, 2), (2, 3)
        safe[:, start[0], start[1]] = True
        safe[:, crossing[0], crossing[1]] = True
        safe[:, refuge[0], refuge[1]] = True
        for position in (start, crossing, refuge):
            blocked[:, position[0], position[1]] = False
        blocked[1:, crossing[0], crossing[1]] = True
        self.assertEqual(
            _reachable_escape_margin(safe, blocked, start), 1,
            "The only route closes before the first possible arrival",
        )

    def test_launcher_accepts_fresh_output_and_recovers_same_directory(self):
        fixture = launcher_fixtures.PhaseCSDTemporal250LauncherAcceptance()
        staged, root, default_output = fixture.stage_launcher(warm_start=True)
        try:
            chosen = root / "runs" / "phaseC-sd-temporal-250-v2"
            call_log = root / "calls.jsonl"
            environment = os.environ.copy()
            environment.update({
                "ACCEPTANCE_CALL_LOG": str(call_log),
                "FAKE_TRAIN_EXIT": "0",
                "FAKE_CREATE_BEST": "1",
                "FAKE_EVAL_EXIT": "0",
                "FAKE_EVAL_ARTIFACT": "write",
                "FAKE_EVAL_PAYLOAD": json.dumps(fixture.valid_evaluation_payload()),
            })
            command = [
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(root / fixture.launcher_name),
                "-OutputDirectory", str(chosen),
            ]
            def invoke(*extra):
                return subprocess.run(command + list(extra), cwd=root, env=environment,
                                      capture_output=True, text=True, check=False)

            first = invoke()
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            calls = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([call["kind"] for call in calls],
                             ["train_actor_critic", "evaluate"])
            fixture.assert_option(calls[0]["argv"], "--checkpoint_dir", chosen)
            fixture.assert_option(calls[1]["argv"], "--ac_model", chosen / "best_phaseC.pt")
            fixture.assert_option(calls[1]["argv"], "--output", chosen / "evaluation-200.json")
            self.assertTrue((chosen / "best_phaseC.pt").is_file())
            self.assertTrue((chosen / "evaluation-200.json").is_file())
            self.assertFalse(default_output.exists())

            sentinel = chosen / "do-not-overwrite.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            retry = invoke()
            self.assertNotEqual(retry.returncode, 0, "Normal mode must not overwrite")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(len(call_log.read_text(encoding="utf-8").splitlines()), 2)

            recovery = invoke("-EvaluateOnly")
            self.assertEqual(recovery.returncode, 0, recovery.stdout + recovery.stderr)
            recovered_calls = [json.loads(line) for line in
                               call_log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([call["kind"] for call in recovered_calls],
                             ["train_actor_critic", "evaluate", "evaluate"])
            fixture.assert_option(recovered_calls[-1]["argv"], "--ac_model",
                                  chosen / "best_phaseC.pt")
            fixture.assert_option(recovered_calls[-1]["argv"], "--output",
                                  chosen / "evaluation-200.json")
            self.assertFalse(default_output.exists())
        finally:
            staged.cleanup()


if __name__ == "__main__":
    unittest.main()
