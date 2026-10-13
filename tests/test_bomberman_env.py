import io
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from evaluate import classify_agent_death, combine_side_swapped_stats, run_match
from train_actor_critic import (
    TacticalBombLifecycle,
    _project_bomb_hazards,
    atomic_torch_save,
    combat_checkpoint_rank,
    classify_learning_agent_death,
    get_action_mask,
    checkpoint_payload,
    checkpoint_state_dict,
    is_combat_safe_checkpoint,
    is_target_bomb_checkpoint,
    noncombat_win_cadence_penalty,
    outcome_reward_signals,
    safe_bomb_cadence_bonus,
    save_snapshot,
    target_checkpoint_rank,
    tactical_bomb_reward,
    learning_agent_id,
)
from bomberman_env import (
    ACTION_BOMB,
    ACTION_DOWN,
    ACTION_RIGHT,
    ACTION_STOP,
    ACTION_UP,
    Bomb,
    BombermanFFAEnv,
    WOODEN_WALL,
)


class SuddenDeathEnvironmentTests(unittest.TestCase):
    def test_warning_channel_marks_the_next_closing_ring(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            sudden_death=True,
            sudden_death_start=2,
            sudden_death_warning_steps=1,
        )
        env.reset()

        observations, _, _, _ = env.step([ACTION_STOP, ACTION_STOP])

        self.assertEqual(observations[0].shape, (9, 7, 7))
        self.assertEqual(observations[0][8, 1, 1], 1.0)

    def test_closing_ring_eliminates_agents_that_do_not_leave_it(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            sudden_death=True,
            sudden_death_start=1,
        )
        env.reset()

        _, _, dones, info = env.step([ACTION_STOP, ACTION_STOP])

        self.assertEqual(info["sudden_death_ring"], 1)
        self.assertEqual(info["sudden_death_kills"], [0, 1])
        self.assertEqual(dones, [True, True])


class TacticalBombTests(unittest.TestCase):
    def test_bomb_next_to_a_wood_wall_is_reported_as_tactical(self):
        env = BombermanFFAEnv(board_size=7, num_agents=2, wooden_wall_density=0.0)
        env.reset()
        env.grid[1, 2] = WOODEN_WALL

        _, _, _, info = env.step([ACTION_BOMB, ACTION_STOP])

        self.assertEqual(info["tactical_bomb_owners"], [0])
        self.assertEqual(info["offensive_bomb_owners"], [])

    def test_bomb_with_an_opponent_in_its_clear_blast_line_is_offensive(self):
        env = BombermanFFAEnv(board_size=7, num_agents=2, wooden_wall_density=0.0)
        env.reset()
        env.agents[1].row, env.agents[1].col = 1, 3

        _, _, _, info = env.step([ACTION_BOMB, ACTION_STOP])

        self.assertEqual(info["offensive_bomb_owners"], [0])

    def test_explosion_reports_the_bomb_outcome_for_its_owner(self):
        env = BombermanFFAEnv(board_size=7, num_agents=2, wooden_wall_density=0.0)
        env.reset()
        env.bomb_timer = 1
        env.grid[1, 2] = WOODEN_WALL

        _, _, _, info = env.step([ACTION_BOMB, ACTION_STOP])

        self.assertEqual(info["bomb_outcomes"], [{
            "owner": 0,
            "position": (1, 1),
            "walls_destroyed": 1,
            "victims": [0],
            "chain_reaction": False,
        }])


class TacticalBombLifecycleTests(unittest.TestCase):
    def test_impact_becomes_rewardable_only_after_safe_escape(self):
        lifecycle = TacticalBombLifecycle()
        lifecycle.register((3, 3))

        self.assertEqual(lifecycle.mark_safe_escape((3, 3), is_safe=True), (0, 0, 0, 0))
        self.assertEqual(lifecycle.resolve_outcome((3, 3)), (False, False, False, 0))

        lifecycle.register((3, 3))
        self.assertEqual(lifecycle.mark_safe_escape((3, 4), is_safe=True), (1, 0, 1, 0))
        self.assertEqual(lifecycle.resolve_outcome((3, 3)), (True, False, True, 0))

    def test_safe_escape_keeps_track_of_whether_the_bomb_was_offensive(self):
        lifecycle = TacticalBombLifecycle()
        lifecycle.register((3, 3), offensive=True, pressure_score=2)

        self.assertEqual(
            lifecycle.mark_safe_escape((3, 4), is_safe=True),
            (1, 1, 1, 0),
        )
        self.assertTrue(lifecycle.has_escaped_pending())
        self.assertEqual(lifecycle.resolve_outcome((3, 3)), (True, True, True, 2))
        self.assertFalse(lifecycle.has_escaped_pending())

    def test_cycle_outside_the_generic_budget_cannot_earn_wall_reward(self):
        lifecycle = TacticalBombLifecycle()
        lifecycle.register((3, 3), generic_reward_eligible=False)

        self.assertEqual(
            lifecycle.mark_safe_escape((3, 4), is_safe=True),
            (1, 0, 0, 0),
        )
        self.assertEqual(lifecycle.resolve_outcome((3, 3)), (True, False, False, 0))

    def test_outcome_reward_requires_living_safe_owner(self):
        self.assertEqual(
            outcome_reward_signals(
                escaped=True,
                offensive=True,
                pressure_score=2,
                owner_alive=True,
                owner_safe=True,
                victims=[1],
                owner_id=0,
            ),
            (1, 1, 2),
        )
        self.assertEqual(
            outcome_reward_signals(
                escaped=True,
                offensive=True,
                pressure_score=2,
                owner_alive=False,
                owner_safe=False,
                victims=[1],
                owner_id=0,
            ),
            (0, 0, 0),
        )
        self.assertEqual(
            outcome_reward_signals(
                escaped=True,
                offensive=True,
                pressure_score=2,
                owner_alive=True,
                owner_safe=False,
                victims=[],
                owner_id=0,
            ),
            (0, 0, 0),
        )

    def test_reward_progresses_from_safe_target_to_escape_to_wall_impact(self):
        self.assertAlmostEqual(
            tactical_bomb_reward(tactical_placements=1),
            0.05,
        )
        self.assertAlmostEqual(
            tactical_bomb_reward(tactical_placements=1, safe_escapes=1),
            0.20,
        )
        self.assertAlmostEqual(
            tactical_bomb_reward(
                tactical_placements=1,
                safe_escapes=1,
                walls_destroyed_after_escape=1,
            ),
            0.22,
        )
        self.assertAlmostEqual(
            tactical_bomb_reward(
                tactical_placements=1,
                safe_escapes=1,
                offensive_safe_escapes=1,
                opponent_kills_after_escape=1,
            ),
            1.60,
        )

    def test_pressure_reward_scales_with_removed_escape_options(self):
        self.assertAlmostEqual(
            tactical_bomb_reward(pressure_score=2, pressure_bonus=0.15),
            0.30,
        )

    def test_cadence_bonus_is_paid_when_the_second_safe_escape_happens(self):
        self.assertEqual(
            safe_bomb_cadence_bonus(previous_safe_escapes=0, newly_safe_escapes=1),
            0.0,
        )

    def test_noncombat_win_penalty_requires_two_safe_cycles(self):
        self.assertEqual(
            noncombat_win_cadence_penalty(
                won=True, kills=0, safe_cycles=0, target_cycles=2
            ),
            -10.0,
        )
        self.assertEqual(
            noncombat_win_cadence_penalty(
                won=True, kills=0, safe_cycles=1, target_cycles=2
            ),
            -5.0,
        )
        self.assertEqual(
            noncombat_win_cadence_penalty(
                won=True, kills=0, safe_cycles=2, target_cycles=2
            ),
            0.0,
        )

    def test_combat_win_and_non_win_are_not_penalized(self):
        self.assertEqual(
            noncombat_win_cadence_penalty(
                won=True, kills=1, safe_cycles=0, target_cycles=2
            ),
            0.0,
        )
        self.assertEqual(
            noncombat_win_cadence_penalty(
                won=False, kills=0, safe_cycles=0, target_cycles=2
            ),
            0.0,
        )
        self.assertEqual(
            safe_bomb_cadence_bonus(previous_safe_escapes=1, newly_safe_escapes=1),
            0.25,
        )
        self.assertEqual(
            safe_bomb_cadence_bonus(previous_safe_escapes=2, newly_safe_escapes=1),
            0.0,
        )

    def test_noncombat_win_can_be_capped_even_after_safe_cycles(self):
        self.assertEqual(
            noncombat_win_cadence_penalty(
                won=True,
                kills=0,
                safe_cycles=2,
                target_cycles=2,
                fixed_penalty=8.0,
            ),
            -8.0,
        )


class BombPressureTests(unittest.TestCase):
    def test_bomb_pressure_counts_opponent_escape_options_in_blast(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
        )
        env.reset()
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 2, 4

        _, _, _, info = env.step([ACTION_BOMB, ACTION_STOP])

        self.assertEqual(info["bomb_pressure_scores"], [(0, 2)])
        self.assertEqual(info["offensive_bomb_owners"], [0])


class BombEscapeMaskTests(unittest.TestCase):
    def test_owner_in_own_blast_zone_can_take_any_shortest_escape_step(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
        )
        env.reset()
        env.step([ACTION_BOMB, ACTION_STOP])

        mask = get_action_mask(env, 0)
        allowed_actions = [action for action, allowed in enumerate(mask) if allowed]

        self.assertEqual(allowed_actions, [ACTION_DOWN, ACTION_RIGHT])

    def test_temporal_mask_blocks_long_timer_blast_with_no_escape(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
        )
        env.reset()
        env.grid[2, 1] = 1
        env.grid[4, 1] = 1
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 1, 5
        env.bombs = [Bomb(row=3, col=3, owner=1, timer=3, blast_range=2)]
        diagnostics = {}

        mask = get_action_mask(env, 0, diagnostics=diagnostics)

        self.assertFalse(mask[ACTION_RIGHT])
        self.assertGreaterEqual(diagnostics["unsafe_reentry_blocks"], 1)
        self.assertEqual(diagnostics["mask_interventions"], 1)

    def test_temporal_mask_allows_long_timer_blast_when_escape_exists(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
        )
        env.reset()
        env.grid[2, 2] = 0
        env.agents[0].row, env.agents[0].col = 2, 2
        env.agents[1].row, env.agents[1].col = 1, 5
        env.bombs = [Bomb(row=3, col=3, owner=1, timer=3, blast_range=2)]

        mask = get_action_mask(env, 0)

        self.assertTrue(mask[ACTION_RIGHT])

    def test_projected_hazards_accelerate_chain_reaction_bombs(self):
        env = BombermanFFAEnv(
            board_size=9,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
        )
        env.reset()
        env.bombs = [
            Bomb(row=3, col=3, owner=0, timer=2, blast_range=2),
            Bomb(row=3, col=5, owner=1, timer=6, blast_range=2),
        ]

        hazards = _project_bomb_hazards(env)

        self.assertIn(2, hazards[(3, 6)])

    def test_bomb_mask_blocks_placement_when_chain_leaves_one_escape_step(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
        )
        env.reset()
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 1, 5
        env.bombs = [Bomb(row=3, col=1, owner=1, timer=2, blast_range=2)]
        diagnostics = {}

        mask = get_action_mask(env, 0, diagnostics=diagnostics)

        self.assertFalse(mask[ACTION_BOMB])
        self.assertEqual(diagnostics["unsafe_bomb_placement_blocks"], 1)

    def test_incoming_threat_overrides_late_chain_bomb_placement(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
        )
        env.reset()
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 1, 5
        env.bombs = [Bomb(row=3, col=1, owner=1, timer=4, blast_range=2)]

        mask = get_action_mask(env, 0)

        self.assertFalse(mask[ACTION_BOMB])


class OpponentAndSuddenDeathMaskTests(unittest.TestCase):
    def test_opponent_bomb_threat_forces_shortest_temporal_escape(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
        )
        env.reset()
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 1, 5
        env.bombs = [Bomb(row=3, col=3, owner=1, timer=3, blast_range=2)]
        diagnostics = {}

        mask = get_action_mask(env, 0, diagnostics=diagnostics)
        allowed_actions = [action for action, allowed in enumerate(mask) if allowed]

        self.assertEqual(allowed_actions, [ACTION_UP, ACTION_DOWN])
        self.assertEqual(diagnostics["forced_escape_steps"], 1)

    def test_sudden_death_horizon_blocks_waiting_and_bombing_on_closing_ring(self):
        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
            sudden_death=True,
            sudden_death_start=10,
            sudden_death_interval=20,
        )
        env.reset()
        env.step_count = 8
        env.agents[0].row, env.agents[0].col = 1, 3
        env.agents[1].row, env.agents[1].col = 5, 5
        diagnostics = {}

        mask = get_action_mask(env, 0, diagnostics=diagnostics)

        self.assertFalse(mask[ACTION_STOP])
        self.assertFalse(mask[ACTION_BOMB])
        self.assertTrue(mask[ACTION_DOWN])
        self.assertGreaterEqual(diagnostics["sudden_death_blocks"], 2)


class ChainReactionDiagnosticsTests(unittest.TestCase):
    def test_bomb_outcome_marks_chain_reaction(self):
        env = BombermanFFAEnv(
            board_size=9,
            num_agents=2,
            wooden_wall_density=0.0,
            seed=7,
        )
        env.reset()
        env.agents[0].row, env.agents[0].col = 1, 1
        env.agents[1].row, env.agents[1].col = 7, 7
        env.bombs = [
            Bomb(row=3, col=3, owner=0, timer=1, blast_range=2),
            Bomb(row=3, col=5, owner=1, timer=6, blast_range=2),
        ]

        _, _, _, info = env.step([ACTION_STOP, ACTION_STOP])

        outcomes = {outcome["position"]: outcome for outcome in info["bomb_outcomes"]}
        self.assertTrue(outcomes[(3, 5)]["chain_reaction"])


class DeathClassificationTests(unittest.TestCase):
    def test_sudden_death_has_priority_over_other_attribution(self):
        info = {
            "alive": [False, True],
            "kills": [(1, 0)],
            "sudden_death_kills": [0],
        }

        self.assertEqual(
            classify_learning_agent_death(info, agent_id=0, was_alive=True),
            "sudden_death",
        )

    def test_classifies_suicide_and_opponent_bomb_deaths(self):
        suicide_info = {"alive": [False, True], "kills": [(0, 0)], "sudden_death_kills": []}
        opponent_info = {"alive": [False, True], "kills": [(1, 0)], "sudden_death_kills": []}

        self.assertEqual(
            classify_learning_agent_death(suicide_info, agent_id=0, was_alive=True),
            "suicide",
        )
        self.assertEqual(
            classify_learning_agent_death(opponent_info, agent_id=0, was_alive=True),
            "opponent_bomb",
        )

    def test_evaluator_classifies_all_new_death_causes(self):
        self.assertEqual(
            classify_agent_death(
                {"alive": [False, True], "kills": [(0, 0)], "sudden_death_kills": []},
                agent_id=0,
                was_alive=True,
            ),
            "self_bomb",
        )
        self.assertEqual(
            classify_agent_death(
                {"alive": [False, True], "kills": [(1, 0)], "sudden_death_kills": []},
                agent_id=0,
                was_alive=True,
            ),
            "opponent_bomb",
        )


class EvaluationConsoleTests(unittest.TestCase):
    def test_verbose_results_are_encodable_by_windows_cp1252_console(self):
        env = BombermanFFAEnv(board_size=7, num_agents=2, max_steps=1)
        cp1252_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

        with patch("sys.stdout", cp1252_stdout):
            run_match(env, [lambda *_: ACTION_STOP, lambda *_: ACTION_STOP],
                      num_episodes=1, verbose=True)
            cp1252_stdout.flush()


class SideSwappedEvaluationTests(unittest.TestCase):
    def test_combines_each_agent_results_across_both_spawn_positions(self):
        first_side = {
            "num_episodes": 10,
            "wins": [8, 1],
            "draws": 1,
            "avg_reward": [4.0, -1.0],
            "avg_bombs": [1.0, 2.0],
            "avg_kills": [0.4, 0.1],
            "avg_offensive_bombs": [0.6, 0.2],
            "avg_pressure_score": [1.2, 0.3],
            "sudden_death_deaths": [2, 5],
            "self_bomb_deaths": [1, 2],
            "opponent_bomb_deaths": [3, 4],
            "other_deaths": [0, 1],
            "chain_bomb_self_deaths": [1, 2],
        }
        second_side = {
            "num_episodes": 10,
            "wins": [3, 5],
            "draws": 2,
            "avg_reward": [-0.5, 3.0],
            "avg_bombs": [2.5, 1.5],
            "avg_kills": [0.2, 0.3],
            "avg_offensive_bombs": [0.4, 0.8],
            "avg_pressure_score": [0.5, 1.5],
            "sudden_death_deaths": [4, 1],
            "self_bomb_deaths": [2, 3],
            "opponent_bomb_deaths": [1, 5],
            "other_deaths": [2, 0],
            "chain_bomb_self_deaths": [2, 3],
        }

        summary = combine_side_swapped_stats(first_side, second_side)

        self.assertEqual(summary["num_episodes"], 20)
        self.assertEqual(summary["wins"], [13, 4])
        self.assertEqual(summary["draws"], 3)
        self.assertEqual(summary["sudden_death_deaths"], [3, 9])
        self.assertEqual(summary["self_bomb_deaths"], [4, 4])
        self.assertEqual(summary["opponent_bomb_deaths"], [8, 5])
        self.assertEqual(summary["chain_bomb_self_deaths"], [4, 4])
        self.assertAlmostEqual(summary["avg_offensive_bombs"][0], 0.7)
        self.assertAlmostEqual(summary["avg_pressure_score"][1], 0.4)


class CurriculumPositionTests(unittest.TestCase):
    def test_phase_c_and_e_cycle_the_learning_agent_across_spawn_positions(self):
        self.assertEqual(
            [learning_agent_id("C", episode, 2) for episode in range(1, 5)],
            [0, 1, 0, 1],
        )
        self.assertEqual(
            [learning_agent_id("E", episode, 4) for episode in range(1, 5)],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [learning_agent_id("D", episode, 2) for episode in range(1, 3)],
            [0, 0],
        )


class CheckpointSaveTests(unittest.TestCase):
    def test_combat_safe_checkpoint_requires_wr_kills_and_low_self_death(self):
        eligible = {"win_rate": 70.0, "kills": 0.025, "self_deaths": 0.05}

        self.assertTrue(is_combat_safe_checkpoint(eligible))
        self.assertFalse(is_combat_safe_checkpoint(dict(eligible, win_rate=69.9)))
        self.assertFalse(is_combat_safe_checkpoint(dict(eligible, kills=0.024)))
        self.assertFalse(is_combat_safe_checkpoint(dict(eligible, self_deaths=0.051)))

    def test_target_checkpoint_requires_bomb_band_and_win_rate_floor(self):
        self.assertTrue(is_target_bomb_checkpoint({"win_rate": 70.0, "bombs": 1.7}))
        self.assertTrue(is_target_bomb_checkpoint({"win_rate": 75.0, "bombs": 2.3}))
        self.assertFalse(is_target_bomb_checkpoint({"win_rate": 69.9, "bombs": 2.0}))
        self.assertFalse(is_target_bomb_checkpoint({"win_rate": 75.0, "bombs": 2.31}))

    def test_target_checkpoint_rank_prioritizes_combat_then_win_rate(self):
        lower_combat = {
            "kills": 0.0,
            "offensive_safe_cycles": 0.4,
            "win_rate": 80.0,
            "bombs": 2.0,
        }
        higher_combat = {
            "kills": 0.1,
            "offensive_safe_cycles": 0.3,
            "win_rate": 70.0,
            "bombs": 2.2,
        }

        self.assertGreater(
            target_checkpoint_rank(higher_combat),
            target_checkpoint_rank(lower_combat),
        )

    def test_combat_checkpoint_rank_prioritizes_kills_then_pressure_and_safety(self):
        safer = {
            "kills": 0.1,
            "pressure_score": 0.8,
            "offensive_safe_cycles": 0.4,
            "win_rate": 72.0,
            "self_deaths": 0.02,
            "sudden_death_deaths": 0.05,
        }
        more_pressure = dict(safer, pressure_score=1.0, self_deaths=0.04)
        more_kills = dict(safer, kills=0.2, win_rate=70.0)

        self.assertGreater(combat_checkpoint_rank(more_pressure), combat_checkpoint_rank(safer))
        self.assertGreater(combat_checkpoint_rank(more_kills), combat_checkpoint_rank(more_pressure))

    def test_checkpoint_payload_preserves_weights_metrics_and_config(self):
        weights = {"layer.weight": torch.tensor([1.0])}

        payload = checkpoint_payload(
            weights,
            episode=400,
            phase="C",
            metrics={"win_rate": 72.0, "bombs": 2.1},
            config={"target_bombs": 2.0},
        )

        self.assertEqual(payload["episode"], 400)
        self.assertEqual(payload["phase"], "C")
        self.assertEqual(payload["metrics"]["bombs"], 2.1)
        self.assertEqual(payload["config"]["target_bombs"], 2.0)
        self.assertIs(checkpoint_state_dict(payload), weights)
        self.assertIs(checkpoint_state_dict(weights), weights)

    def test_phase_c_snapshot_uses_distinct_name_and_metadata_payload(self):
        class DummyNet:
            def state_dict(self):
                return {"weight": torch.tensor([1.0])}

        with tempfile.TemporaryDirectory() as directory, \
             patch("train_actor_critic.atomic_torch_save") as save:
            path = save_snapshot(
                DummyNet(),
                directory,
                episode=200,
                phase="C",
                metrics={"win_rate": 71.0, "bombs": 2.0},
                config={"target_bombs": 2.0},
            )

        self.assertEqual(os.path.basename(path), "snapshot_phaseC_ep000200.pt")
        payload = save.call_args.args[0]
        self.assertEqual(payload["episode"], 200)
        self.assertEqual(payload["metrics"]["bombs"], 2.0)

    def test_retries_when_windows_temporarily_locks_the_checkpoint(self):
        with patch("train_actor_critic.torch.save") as save, \
             patch("train_actor_critic.os.replace", side_effect=[PermissionError(32, "locked"), None]) as replace, \
             patch("train_actor_critic.time.sleep") as sleep:
            atomic_torch_save({"weight": 1}, "checkpoints_v3/best_phaseC.pt", retries=2)

        self.assertEqual(save.call_count, 2)
        self.assertEqual(replace.call_count, 2)
        sleep.assert_called_once()

    def test_retries_when_windows_replace_reports_winerror_five(self):
        locked_error = PermissionError(13, "Access is denied")
        locked_error.winerror = 5
        with patch("train_actor_critic.torch.save"), \
             patch("train_actor_critic.os.replace", side_effect=[locked_error, None]) as replace, \
             patch("train_actor_critic.time.sleep"):
            atomic_torch_save({"weight": 1}, "checkpoints_v3/best_phaseC.pt", retries=2)

        self.assertEqual(replace.call_count, 2)


if __name__ == "__main__":
    unittest.main()
