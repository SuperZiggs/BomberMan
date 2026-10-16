"""Independent acceptance tests for the Phase C best-source fallback.

These tests never launch training or evaluation. They inspect public contracts,
load the frozen checkpoint, and validate the existing 200-game artifact.
"""

from dataclasses import fields
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from bomberman_env import ACTION_BOMB, ACTION_LEFT, Bomb, BombermanFFAEnv, EMPTY
import evaluate
from train_actor_critic import ActorCriticNet, get_attack_action, smart_load_weights, train


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "phaseC-sd-best-fallback"
CONFIG_PATH = RUN_DIR / "config.json"
CHECKPOINT_PATH = RUN_DIR / "best_phaseC.pt"
EVALUATION_PATH = RUN_DIR / "evaluation-200.json"
LAUNCHER_PATH = ROOT / "run_train_phaseC_sd_target_750.ps1"

EXPECTED_CONFIG = {
    "phase": "C",
    "episodes": 750,
    "board_size": 9,
    "num_agents": 2,
    "max_steps": 500,
    "lr": 2.5e-05,
    "entropy_coef": 0.0075,
    "wall_destroy_bonus": 0.15,
    "attack_lookahead": 2,
    "min_pressure_gain": 0.15,
    "trap_sequence_depth": 2,
    "sequence_pressure_gain": 0.35,
    "ring_safety_coef": 0.6,
    "late_game_start": 350,
    "late_game_attack_multiplier": 1.75,
    "escape_loss_coef": 0.5,
    "combat_loss_coef": 0.9,
    "navigation_loss_coef": 0.4,
    "target_win_rate": 70.0,
    "target_active_win_rate": 55.0,
    "target_draw_rate": 15.0,
    "target_sudden_death_death_rate": 10.0,
    "max_suicide_rate": 1.5,
    "snapshot_every": 200,
    "checkpoint_dir": r"E:\Bomberman\runs\phaseC-sd-target-750",
    "resume_from": r"E:\Bomberman\runs\phaseC-sd-attack-750\best_phaseC.pt",
    "log_every": 50,
    "seed": 2026101503,
    "validation_every": 250,
    "validation_episodes": 200,
    "validation_seed": 30000,
    "sudden_death_probability": 0.75,
    "sudden_death_start": 475,
    "sudden_death_interval": 25,
    "sudden_death_warning_steps": 40,
}

TRAINING_OPTIONS = {
    "phase": "C",
    "episodes": "750",
    "board_size": "9",
    "num_agents": "2",
    "max_steps": "500",
    "lr": "0.000025",
    "entropy_coef": "0.0075",
    "wall_destroy_bonus": "0.15",
    "attack_lookahead": "2",
    "min_pressure_gain": "0.15",
    "trap_sequence_depth": "2",
    "sequence_pressure_gain": "0.35",
    "ring_safety_coef": "0.6",
    "late_game_start": "350",
    "late_game_attack_multiplier": "1.75",
    "escape_loss_coef": "0.5",
    "combat_loss_coef": "0.9",
    "navigation_loss_coef": "0.4",
    "seed": "2026101503",
    "validation_every": "250",
    "validation_episodes": "200",
    "validation_seed": "30000",
    "sudden_death_probability": "0.75",
    "sudden_death_start": "475",
    "sudden_death_interval": "25",
    "sudden_death_warning_steps": "40",
    "target_win_rate": "70",
    "target_active_win_rate": "55",
    "target_draw_rate": "15",
    "target_sudden_death_death_rate": "10",
    "max_suicide_rate": "1.5",
}


def launcher_source():
    return LAUNCHER_PATH.read_text(encoding="utf-8-sig")


def assert_launcher_option(test_case, source, name, value):
    pattern = rf"(?im)^\s*--{re.escape(name)}\s+{re.escape(value)}(?:\s+`)?\s*$"
    test_case.assertRegex(source, pattern, f"launcher option --{name} must equal {value}")


class PreservedBestRunAcceptance(unittest.TestCase):
    def test_frozen_config_and_effective_training_parameters(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config, EXPECTED_CONFIG)

        source = launcher_source()
        for name, value in TRAINING_OPTIONS.items():
            with self.subTest(option=name):
                assert_launcher_option(self, source, name, value)

        self.assertRegex(source, r"(?im)^\s*--resume_from\s+\$Checkpoint(?:\s+`)?\s*$")
        self.assertRegex(source, r"(?im)^\s*--checkpoint_dir\s+\$OutputDirectory(?:\s+`)?\s*$")
        parameters = inspect.signature(train).parameters
        self.assertEqual(parameters["snapshot_every"].default, config["snapshot_every"])
        self.assertEqual(parameters["log_every"].default, config["log_every"])

    def test_launcher_keeps_one_command_train_then_exact_200_game_json_evaluation(self):
        source = launcher_source()
        self.assertEqual(source.count('"train_actor_critic.py"'), 1)
        self.assertEqual(source.count('"evaluate.py"'), 1)
        self.assertLess(source.index('"train_actor_critic.py"'), source.index('"evaluate.py"'))
        expected = {
            "mode": "seeded",
            "board_size": "9",
            "num_agents": "2",
            "num_episodes": "200",
            "max_steps": "500",
            "seed": "31000",
            "device": "cpu",
            "sudden_death_start": "475",
            "sudden_death_interval": "25",
            "sudden_death_warning_steps": "40",
        }
        for name, value in expected.items():
            with self.subTest(option=name):
                assert_launcher_option(self, source, name, value)
        self.assertRegex(source, r"(?im)^\s*--sudden_death(?:\s+`)?\s*$")
        self.assertRegex(source, r"(?im)^\s*--ac_model\s+\$BestCheckpoint(?:\s+`)?\s*$")
        self.assertRegex(source, r"(?im)^\s*--output\s+\$EvaluationOutput\s*$")
        self.assertIn('"evaluation-200.json"', source)

    def test_sudden_death_observation_and_checkpoint_are_13_channel_compatible(self):
        env = BombermanFFAEnv(
            board_size=9,
            num_agents=2,
            max_steps=500,
            sudden_death=True,
            sudden_death_start=475,
            sudden_death_interval=25,
            sudden_death_warning_steps=40,
            seed=7,
        )
        observations = env.reset()
        self.assertEqual([observation.shape for observation in observations], [(13, 9, 9)] * 2)

        model = ActorCriticNet(board_size=9)
        smart_load_weights(model, str(CHECKPOINT_PATH), "cpu")
        self.assertEqual(model.conv[0].in_channels, 13)
        with torch.no_grad():
            logits, value = model(torch.from_numpy(observations[0]).unsqueeze(0))
        self.assertEqual(tuple(logits.shape), (1, 6))
        self.assertEqual(tuple(value.shape), (1,))

    def test_frozen_evaluation_artifact_has_exact_200_game_result(self):
        artifact = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
        self.assertEqual(artifact["num_episodes"], 200)
        self.assertEqual(len(artifact["games"]), 200)
        expected_metrics = {
            "win_rate": 67.5,
            "active_win_rate": 51.0,
            "draw_rate": 21.5,
            "sudden_death_death_rate": 13.0,
            "suicide_rate": 0.5,
        }
        for metric, expected in expected_metrics.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(artifact[metric], expected)

        outcome_fields = {
            "win_rate": "win",
            "active_win_rate": "active_win",
            "draw_rate": "draw",
            "sudden_death_death_rate": "sudden_death_death",
            "suicide_rate": "suicide",
        }
        for metric, field in outcome_fields.items():
            with self.subTest(recomputed=metric):
                recomputed = sum(game[field] for game in artifact["games"]) / 200 * 100
                self.assertAlmostEqual(recomputed, artifact[metric])

    def test_static_two_position_pressure_teacher_is_preserved(self):
        expected_parameters = {
            "env",
            "agent_id",
            "danger",
            "mask",
            "attack_lookahead",
            "min_pressure_gain",
            "trap_sequence_depth",
            "sequence_pressure_gain",
        }
        self.assertEqual(set(inspect.signature(get_attack_action).parameters), expected_parameters)

        env = BombermanFFAEnv(
            board_size=7,
            num_agents=2,
            max_steps=20,
            wooden_wall_density=0,
            powerup_prob=0,
            seed=29,
        )
        env.reset()
        env.grid[1:-1, 1:-1] = EMPTY
        env.agents[0].row, env.agents[0].col = 3, 2
        env.agents[0].blast_range = 2
        env.agents[0].ammo = 2
        env.agents[1].row, env.agents[1].col = 3, 5
        danger = np.zeros((7, 7), dtype=np.int32)
        mask = [True] * 6

        def pressure(_env, _agent_id, row, col, base=None):
            return {(3, 2): 0.20, (3, 3): 0.15}.get((row, col), 0.0)

        with patch("train_actor_critic._bomb_escape_pressure", side_effect=pressure), patch(
            "train_actor_critic._bfs_toward_item", return_value=None
        ), patch("train_actor_critic._bfs_toward_opponent", return_value=ACTION_LEFT), patch(
            "train_actor_critic._bfs_toward_wood", return_value=None
        ):
            action = get_attack_action(
                env,
                0,
                danger,
                mask,
                attack_lookahead=2,
                min_pressure_gain=0.5,
                trap_sequence_depth=2,
                sequence_pressure_gain=0.35,
            )
        self.assertEqual(action, ACTION_BOMB)


class RemovedFeatureAcceptance(unittest.TestCase):
    def test_temporal_planner_follower_and_interception_runtime_are_removed(self):
        train_source = (ROOT / "train_actor_critic.py").read_text(encoding="utf-8")
        forbidden = (
            "temporal_trap_planner",
            "TemporalTrapConfig",
            "TemporalTrapPlan",
            "TemporalPlanFollower",
            "find_temporal_trap_plan",
            "find_interception_action",
            "_select_temporal_label",
            "_observe_temporal_transition",
            "temporal_trap_horizon",
            "temporal_trap_max_bombs",
            "interception_horizon",
        )
        present = [token for token in forbidden if token in train_source]
        self.assertEqual(present, [], f"temporal/interception runtime remains: {present}")
        self.assertFalse((ROOT / "temporal_trap_planner.py").exists())

    def test_delayed_bomb_placement_tick_is_removed(self):
        self.assertEqual(
            [field.name for field in fields(Bomb)],
            ["row", "col", "owner", "timer", "blast_range"],
        )
        source = (ROOT / "bomberman_env.py").read_text(encoding="utf-8")
        self.assertNotIn("placement_tick", source)

    def test_resumable_evaluation_flags_and_helpers_are_removed(self):
        evaluate_source = (ROOT / "evaluate.py").read_text(encoding="utf-8")
        forbidden = (
            "resume_partial",
            "partial_every",
            "initial_games",
            "on_checkpoint",
            "run_seeded_evaluation_to_file",
            "_checkpoint_identity",
            "_atomic_write_json",
            "_validate_ordered_games",
            "_load_partial_games",
            "_partial_path_for",
            "_partial_payload",
        )
        present = [token for token in forbidden if token in evaluate_source]
        self.assertEqual(present, [], f"resumable evaluator surface remains: {present}")

        completed = subprocess.run(
            [sys.executable, str(ROOT / "evaluate.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("--resume_partial", completed.stdout)
        self.assertNotIn("--partial_every", completed.stdout)
        source = launcher_source()
        self.assertNotIn("EvaluateOnly", source)
        self.assertNotIn("--resume_partial", source)
        self.assertNotIn("--partial_every", source)

    def test_temporal_and_resumable_feature_artifacts_are_removed(self):
        forbidden = [
            ROOT / "run_train_phaseC_sd_temporal_250.ps1",
            ROOT / "test_phase_c_sd_temporal_250.py",
            ROOT / "test_temporal_plan_follower.py",
            ROOT / "test_temporal_plan_follower_support.py",
            ROOT / "test_temporal_trap_planner.py",
            ROOT / "test_temporal_trap_regressions.py",
            ROOT / "test_resumable_evaluation.py",
            ROOT / "docs" / "superpowers" / "plans" / "2026-10-15-phase-c-temporal-trap-planner.md",
            ROOT / "docs" / "superpowers" / "plans" / "2026-10-15-resumable-seeded-evaluation.md",
            ROOT / "docs" / "superpowers" / "plans" / "2026-10-16-temporal-placement-corrections.md",
            ROOT / "docs" / "superpowers" / "plans" / "2026-10-16-temporal-plan-follower.md",
            ROOT / "docs" / "superpowers" / "specs" / "2026-10-15-phase-c-temporal-trap-resumable-evaluation-design.md",
            ROOT / "docs" / "superpowers" / "specs" / "2026-10-16-phase-c-temporal-plan-follower-design.md",
            ROOT / "docs" / "superpowers" / "test-matrices" / "2026-10-15-resumable-seeded-evaluation.md",
            ROOT / "docs" / "superpowers" / "test-matrices" / "2026-10-15-temporal-trap-planner.md",
            ROOT / "docs" / "superpowers" / "test-matrices" / "2026-10-15-temporal-trap-review-fixes.md",
            ROOT / "docs" / "superpowers" / "test-matrices" / "2026-10-16-temporal-placement-regressions.md",
            ROOT / "docs" / "superpowers" / "test-matrices" / "2026-10-16-temporal-plan-follower-acceptance.md",
        ]
        remaining = [str(path.relative_to(ROOT)) for path in forbidden if path.exists()]
        self.assertEqual(remaining, [], f"post-fallback feature artifacts remain: {remaining}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
