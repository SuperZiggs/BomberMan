"""Focused trainer-follower checks; no model updates or artifact writes."""

import unittest
from unittest.mock import patch

import numpy as np

from bomberman_env import ACTION_BOMB, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP, ACTION_UP, Bomb
from rule_based_bot import _survival_map
import train_actor_critic as training
from temporal_trap_planner import TemporalTrapConfig
from test_temporal_plan_follower import ROUTE, control_loop, fixture_plan
from test_temporal_trap_planner import carve, legal_mask
from test_temporal_trap_regressions import open_pair_arena


class TemporalFollowerStateTests(unittest.TestCase):
    def test_fallback_mask_cannot_certify_a_doomed_temporal_step(self):
        env = open_pair_arena(bomb_timer=12)
        carve(env, {(3, 1), (3, 2), (3, 3), (1, 5)})
        env.agents[1].row, env.agents[1].col = 1, 5
        env.bombs.append(Bomb(3, 1, 1, 3, 4))
        diagnostics = {}
        follower = training.TemporalPlanFollower(
            fixture_plan((ACTION_RIGHT, ACTION_BOMB)), (3, 2), diagnostics,
        )
        mask = training.get_action_mask(env, 0)
        danger = env._danger_map()
        self.assertEqual(mask, [True, False, False, False, True, False])
        self.assertEqual((danger[3, 2], danger[3, 3]), (3, 3))
        safe, _, _ = _survival_map(env, 3, 2, avoid_contests=False, safety_margin=0)
        self.assertFalse(safe[0, 3, 3])
        label, pending = training._select_temporal_label(
            env, 0, danger, mask, follower, TemporalTrapConfig(), diagnostics,
        )
        self.assertIsNone(label)
        self.assertIsNone(pending)
        self.assertEqual(diagnostics.get("temporal_trap_labels", 0), 0)
        self.assertEqual(diagnostics["temporal_plan_aborts_unsafe"], 1)

    def test_pending_bomb_requires_escape_after_its_hypothetical_placement(self):
        env = open_pair_arena(bomb_timer=12)
        carve(env, {(3, 2), (1, 5)})
        env.agents[1].row, env.agents[1].col = 1, 5
        diagnostics = {}
        follower = training.TemporalPlanFollower(
            fixture_plan((ACTION_BOMB,)), (3, 2), diagnostics,
        )
        mask = legal_mask(env, 0)
        self.assertTrue(mask[ACTION_BOMB])
        label, pending = training._select_temporal_label(
            env, 0, env._danger_map(), mask, follower, TemporalTrapConfig(), diagnostics,
        )
        self.assertIsNone(label)
        self.assertIsNone(pending)
        self.assertEqual(training._temporal_metrics(diagnostics)["temporal_plan_aborts_safety"], 1)

    def test_matching_action_advances_and_mismatch_aborts_once(self):
        diagnostics = {}
        follower = training.TemporalPlanFollower(
            fixture_plan((ACTION_RIGHT, ACTION_BOMB)), (3, 2), diagnostics,
        )
        self.assertEqual(follower.expected_action(), ACTION_RIGHT)
        self.assertTrue(follower.advance(ACTION_RIGHT))
        self.assertEqual(follower.expected_action(), ACTION_BOMB)
        self.assertEqual(follower.expected_position, (3, 3))
        self.assertFalse(follower.advance(ACTION_STOP))
        follower.abort("mismatch")
        self.assertEqual(diagnostics["temporal_plan_aborts_mismatch"], 1)
        self.assertEqual(training._temporal_metrics(diagnostics)["temporal_plan_aborts"], 1)

    def test_proposal_does_not_count_execution_or_completion(self):
        env = open_pair_arena(bomb_timer=12)
        diagnostics = {}
        with patch.object(training, "find_temporal_trap_plan", return_value=fixture_plan()), \
             patch.object(training, "_ring_safety_action", return_value=None):
            label, follower = training._select_temporal_label(
                env, 0, np.zeros_like(env.grid), legal_mask(env, 0), None,
                TemporalTrapConfig(), diagnostics,
            )
        self.assertEqual(label, ACTION_BOMB)
        self.assertEqual(follower.expected_action(), ACTION_BOMB)
        self.assertEqual(diagnostics["temporal_trap_labels"], 1)
        self.assertEqual(diagnostics.get("temporal_trap_bombs", 0), 0)
        self.assertEqual(diagnostics.get("temporal_trap_pair_completions", 0), 0)

    def test_mask_and_stale_position_cancel_before_label(self):
        for reason in ("mask", "stale_position"):
            with self.subTest(reason=reason):
                env = open_pair_arena(bomb_timer=12)
                diagnostics = {}
                follower = training.TemporalPlanFollower(
                    fixture_plan((ACTION_RIGHT, ACTION_BOMB)), (3, 2), diagnostics,
                )
                mask = legal_mask(env, 0)
                if reason == "mask":
                    mask[ACTION_RIGHT] = False
                else:
                    env.agents[0].col = 3
                with patch.object(training, "_ring_safety_action", return_value=None):
                    label, pending = training._select_temporal_label(
                        env, 0, np.zeros_like(env.grid), mask, follower,
                        TemporalTrapConfig(), diagnostics,
                    )
                self.assertIsNone(label)
                self.assertIsNone(pending)
                self.assertEqual(diagnostics["temporal_plan_aborts_" + reason], 1)

    def test_safety_preempts_even_a_physically_legal_pending_bomb(self):
        for reason in ("danger", "ring"):
            with self.subTest(reason=reason):
                env = open_pair_arena(bomb_timer=12)
                diagnostics = {}
                follower = training.TemporalPlanFollower(fixture_plan(), (3, 2), diagnostics)
                danger = np.zeros_like(env.grid)
                if reason == "danger":
                    danger[3, 2] = 1
                with patch.object(training, "_ring_safety_action",
                                  return_value=ACTION_UP if reason == "ring" else None):
                    label, pending = training._select_temporal_label(
                        env, 0, danger, legal_mask(env, 0), follower,
                        TemporalTrapConfig(), diagnostics,
                    )
                self.assertIsNone(label)
                self.assertIsNone(pending)
                self.assertEqual(diagnostics["temporal_plan_aborts_" + reason], 1)

    def test_sequence_step_labels_are_counted_separately_from_proposals(self):
        result = control_loop()
        self.assertEqual(result.metrics["temporal_trap_labels"], 4)
        self.assertEqual(result.metrics["temporal_trap_plans"], 1)
        self.assertEqual(result.metrics["temporal_trap_bombs"], 2)
        self.assertEqual(result.metrics["temporal_trap_pair_completions"], 1)

    def test_rejected_placement_cancels_instead_of_teaching_movement(self):
        result = control_loop(samples=(ACTION_BOMB, ACTION_STOP), rejected_bomb_tick=0)
        self.assertEqual(result.labels, [(0, ACTION_BOMB)])
        self.assertEqual(result.metrics["temporal_plan_aborts_placement"], 1)

    def test_pending_route_reuses_plan_instead_of_proposing_each_tick(self):
        result = control_loop()
        self.assertEqual(result.planner_calls, 1)
        self.assertEqual(result.labels, list(enumerate(ROUTE)))

    def test_temporal_navigation_steps_keep_combat_supervision(self):
        result = control_loop()
        self.assertEqual(result.metrics["combat_labels"], 4)
        self.assertGreater(result.metrics["combat_loss"], 0.0)
        self.assertEqual(result.metrics["navigation_loss"], 0.0)

    def test_interception_stays_navigation_when_legacy_combat_action_matches(self):
        def interception(*args, diagnostics, **kwargs):
            diagnostics["source"] = "interception"
            diagnostics["interception_labels"] += 1
            training._sequence_attack_action.return_value = ACTION_RIGHT
            return ACTION_RIGHT

        with patch.object(training, "get_attack_action", side_effect=interception):
            result = control_loop(samples=(ACTION_RIGHT,), no_plan=True)
        self.assertEqual(result.metrics["interception_labels"], 1)
        self.assertEqual(result.metrics["combat_labels"], 0)
        self.assertEqual(result.metrics["combat_loss"], 0.0)
        self.assertGreater(result.metrics["navigation_loss"], 0.0)


class TemporalObservedMetricsTests(unittest.TestCase):
    def test_zero_steps_publish_zero_agreement_and_zero_abort_groups(self):
        metrics = control_loop(samples=(ACTION_STOP,), no_plan=True).metrics
        self.assertEqual(metrics["temporal_policy_agreement_rate"], 0.0)
        self.assertIs(type(metrics["temporal_policy_agreement_rate"]), float)
        for suffix in ("", "_safety", "_stale", "_masked", "_mismatch"):
            self.assertEqual(metrics["temporal_plan_aborts" + suffix], 0)

    def test_episode_counters_cover_zero_singleton_pair_and_mismatch(self):
        cases = (
            ({"samples": (ACTION_STOP,), "no_plan": True}, (0, 0, 0, 0, 0, 0, 0.0)),
            ({"samples": (ACTION_RIGHT, ACTION_BOMB),
              "plan_actions": (ACTION_RIGHT, ACTION_BOMB)}, (1, 0, 2, 2, 1, 0, 1.0)),
            ({}, (1, 1, 4, 4, 2, 1, 1.0)),
            ({"samples": (ACTION_BOMB, ACTION_STOP, ACTION_STOP)},
             (1, 1, 2, 1, 1, 0, 0.5)),
            ({"samples": (ACTION_STOP,), "plan_actions": (ACTION_RIGHT, ACTION_BOMB),
              "masked_tick": 0}, (1, 0, 0, 0, 0, 0, 0.0)),
        )
        names = (
            "temporal_plans_started", "temporal_pair_plans_started",
            "temporal_steps_supervised", "temporal_policy_action_matches",
            "temporal_bomb_actions_executed", "temporal_pair_completions",
            "temporal_policy_agreement_rate",
        )
        for options, expected in cases:
            with self.subTest(options=options):
                metrics = control_loop(**options).metrics
                self.assertEqual(tuple(metrics[name] for name in names), expected)
                for canonical, alias in (
                    ("temporal_plans_started", "temporal_trap_plans"),
                    ("temporal_steps_supervised", "temporal_trap_labels"),
                    ("temporal_bomb_actions_executed", "temporal_trap_bombs"),
                    ("temporal_pair_completions", "temporal_trap_pair_completions"),
                ):
                    self.assertEqual(metrics[canonical], metrics[alias])

    def test_abort_totals_group_safety_stale_masked_and_mismatch(self):
        cases = (
            ({"samples": (ACTION_BOMB, ACTION_STOP)}, "mismatch"),
            ({"samples": (ACTION_BOMB, ACTION_STOP), "masked_tick": 1}, "masked"),
            ({"samples": (ACTION_BOMB, ACTION_RIGHT), "blocked_tick": 1}, "stale"),
            ({"samples": (ACTION_BOMB,), "rejected_bomb_tick": 0}, "stale"),
            ({"samples": (ACTION_BOMB, ACTION_UP), "ring_tick": 1}, "safety"),
            ({"samples": (ACTION_RIGHT, ACTION_LEFT), "hazard": "fire",
              "plan_actions": (ACTION_RIGHT, ACTION_BOMB, ACTION_RIGHT, ACTION_BOMB)},
             "safety"),
            ({"samples": (ACTION_BOMB, ACTION_STOP), "hazard": "destination"}, "safety"),
        )
        for options, expected in cases:
            with self.subTest(options=options):
                metrics = control_loop(**options).metrics
                self.assertEqual(metrics["temporal_plan_aborts"], 1)
                for group in ("safety", "stale", "masked", "mismatch"):
                    self.assertEqual(metrics["temporal_plan_aborts_" + group],
                                     int(group == expected))

    def test_rejected_transition_still_records_policy_agreement(self):
        result = control_loop(samples=(ACTION_BOMB,), rejected_bomb_tick=0)
        self.assertEqual(result.metrics["temporal_policy_action_matches"], 1)
        self.assertEqual(result.metrics["temporal_policy_agreement_rate"], 1.0)
        self.assertEqual(result.metrics["temporal_bomb_actions_executed"], 0)
        self.assertEqual(result.metrics["temporal_pair_completions"], 0)

    def test_aggregate_agreement_uses_total_matches_over_total_steps(self):
        totals = {
            "temporal_trap_plans": 2, "temporal_pair_plans_started": 2,
            "temporal_trap_labels": 6, "temporal_policy_action_matches": 5,
            "temporal_trap_bombs": 3, "temporal_trap_pair_completions": 1,
            "temporal_plan_aborts_mismatch": 1,
            "predicted_refuge_reduction": 1.0,
            "predicted_trap_completion_ticks": 6,
        }
        metrics = training._temporal_metrics(totals)
        self.assertEqual(metrics["temporal_plans_started"], 2)
        self.assertEqual(metrics["temporal_pair_plans_started"], 2)
        self.assertEqual(metrics["temporal_policy_agreement_rate"], 5 / 6)
        self.assertEqual(metrics["temporal_bomb_actions_executed"], 3)
        self.assertEqual(metrics["temporal_pair_completions"], 1)
        self.assertEqual(metrics["temporal_plan_aborts"], 1)

    def test_aggregate_log_path_runs_with_learning_and_artifact_writes_disabled(self):
        actual_train = training.train

        def log_each_episode(**options):
            return actual_train(**{**options, "log_every": 1})

        with patch.object(training, "train", side_effect=log_each_episode):
            result = control_loop()
        self.assertEqual(result.metrics["temporal_policy_action_matches"], 4)
        self.assertEqual(result.metrics["temporal_pair_completions"], 1)


if __name__ == "__main__":
    unittest.main()
