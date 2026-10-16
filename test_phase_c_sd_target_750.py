"""Acceptance tests for the Phase C-SD target-aware 750-episode fine-tune."""

import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np
import torch

from bomberman_env import (
    ACTION_BOMB,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_STOP,
    Bomb,
    BombermanFFAEnv,
    EMPTY,
    RIGID_WALL,
)
from rule_based_bot import RuleBasedAgent
from train_actor_critic import _combat_reward, get_attack_action, train


TARGET_OPTIONS = (
    "target_win_rate",
    "target_active_win_rate",
    "target_draw_rate",
    "target_sudden_death_death_rate",
    "max_suicide_rate",
)


def require_parameters(callable_object, *names):
    parameters = inspect.signature(callable_object).parameters
    missing = set(names) - parameters.keys()
    if missing:
        raise AssertionError(
            f"{callable_object.__name__} is missing required options: {sorted(missing)}"
        )


def arena(num_agents=2, board_size=7, **kwargs):
    env = BombermanFFAEnv(
        board_size=board_size,
        num_agents=num_agents,
        max_steps=20,
        wooden_wall_density=0,
        powerup_prob=0,
        seed=29,
        **kwargs,
    )
    env.reset()
    env.grid[1:-1, 1:-1] = EMPTY
    return env


def put_bomb(env, row, col, owner, timer=1, blast_range=1):
    env.bombs.append(Bomb(row, col, owner, timer, blast_range))
    env.agents[owner].bombs_placed += 1


def validation_metrics(
    *,
    win_rate=70.0,
    active_win_rate=55.0,
    draw_rate=15.0,
    sudden_death_death_rate=10.0,
    suicide_rate=1.0,
    avg_kills=0.5,
):
    return {
        "win_rate": win_rate,
        "active_win_rate": active_win_rate,
        "draw_rate": draw_rate,
        "sudden_death_death_rate": sudden_death_death_rate,
        "suicide_rate": suicide_rate,
        "avg_kills": avg_kills,
        "games": [],
    }


class FrozenBehaviorAcceptance(unittest.TestCase):
    def test_rulebot_and_environment_rewards_remain_frozen(self):
        rule_env = arena(2)
        rule_env.agents[0].row, rule_env.agents[0].col = 3, 1
        rule_env.agents[1].row, rule_env.agents[1].col = 3, 3
        self.assertEqual(RuleBasedAgent(0).select_action(rule_env), ACTION_BOMB)

        warning_env = arena(
            2,
            sudden_death=True,
            sudden_death_start=4,
            sudden_death_interval=4,
            sudden_death_warning_steps=4,
        )
        warning_env.grid[1, 2] = RIGID_WALL
        warning_env.agents[0].row, warning_env.agents[0].col = 1, 1
        warning_env.agents[1].row, warning_env.agents[1].col = 5, 5
        self.assertEqual(RuleBasedAgent(0).select_action(warning_env), ACTION_DOWN)

        kill_env = arena(3)
        for agent, position in zip(kill_env.agents, ((1, 1), (3, 3), (5, 5))):
            agent.row, agent.col = position
        put_bomb(kill_env, 3, 2, owner=0)
        _, rewards, _, info = kill_env.step([ACTION_STOP] * 3)
        self.assertEqual(info["kills"], [(0, 1)])
        self.assertEqual([agent.kills for agent in kill_env.agents], [1, 0, 0])
        self.assertEqual(rewards, [0.999, -1.0, -0.001])
        self.assertAlmostEqual(_combat_reward(rewards[0], 0, info, 1), 3.999)

        suicide_env = arena(3)
        for agent, position in zip(suicide_env.agents, ((3, 3), (1, 1), (5, 5))):
            agent.row, agent.col = position
        put_bomb(suicide_env, 3, 3, owner=0)
        _, rewards, _, info = suicide_env.step([ACTION_STOP] * 3)
        self.assertEqual(info["kills"], [(0, 0)])
        self.assertEqual(info["self_blast_deaths"], [0])
        self.assertEqual(suicide_env.agents[0].kills, 0)
        self.assertEqual(rewards[0], -1.0)
        self.assertEqual(_combat_reward(rewards[0], 0, info, 0), -2.5)


class TrainingConfigurationAcceptance(unittest.TestCase):
    train_defaults = {
        "episodes": 750,
        "lr": 0.000025,
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
        "target_win_rate": None,
        "target_active_win_rate": None,
        "target_draw_rate": None,
        "target_sudden_death_death_rate": None,
        "max_suicide_rate": None,
    }

    def test_train_and_teacher_expose_approved_defaults(self):
        train_parameters = inspect.signature(train).parameters
        missing = set(self.train_defaults) - train_parameters.keys()
        self.assertFalse(missing, f"train is missing required options: {sorted(missing)}")
        for name, expected in self.train_defaults.items():
            with self.subTest(callable="train", name=name):
                self.assertEqual(train_parameters[name].default, expected)

        teacher_parameters = inspect.signature(get_attack_action).parameters
        teacher_defaults = {"trap_sequence_depth": 2, "sequence_pressure_gain": 0.35}
        missing = set(teacher_defaults) - teacher_parameters.keys()
        self.assertFalse(
            missing,
            f"get_attack_action is missing required options: {sorted(missing)}",
        )
        for name, expected in teacher_defaults.items():
            with self.subTest(callable="get_attack_action", name=name):
                self.assertEqual(teacher_parameters[name].default, expected)

    def test_cli_exposes_new_configuration(self):
        root = Path(__file__).resolve().parent
        completed = subprocess.run(
            [sys.executable, str(root / "train_actor_critic.py"), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        required = (
            "trap_sequence_depth",
            "sequence_pressure_gain",
            "ring_safety_coef",
            *TARGET_OPTIONS,
        )
        for name in required:
            with self.subTest(name=name):
                self.assertIn(f"--{name}", completed.stdout)


class SequenceTeacherAcceptance(unittest.TestCase):
    teacher_options = ("trap_sequence_depth", "sequence_pressure_gain")

    def configured_case(self):
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 2
        env.agents[0].blast_range = 2
        env.agents[0].ammo = 2
        env.agents[1].row, env.agents[1].col = 3, 5
        danger = np.zeros((env.board_size, env.board_size), dtype=np.int32)
        mask = [True] * 6
        return env, danger, mask

    def call_teacher(self, env, danger, mask, **overrides):
        options = {
            "attack_lookahead": 2,
            "min_pressure_gain": 0.5,
            "trap_sequence_depth": 2,
            "sequence_pressure_gain": 0.35,
        }
        options.update(overrides)
        return get_attack_action(env, 0, danger, mask, **options)

    def pressure_oracle(self, _env, _agent_id, row, col, base=None):
        return {(3, 2): 0.20, (3, 3): 0.15}.get((row, col), 0.0)

    def test_two_bomb_cumulative_pressure_requires_depth_two(self):
        require_parameters(get_attack_action, *self.teacher_options)
        env, danger, mask = self.configured_case()
        with patch(
            "train_actor_critic._bomb_escape_pressure", side_effect=self.pressure_oracle
        ), patch("train_actor_critic._bfs_toward_item", return_value=None), patch(
            "train_actor_critic._bfs_toward_opponent", return_value=ACTION_LEFT
        ), patch("train_actor_critic._bfs_toward_wood", return_value=None):
            depth_one = self.call_teacher(
                env, danger, mask, trap_sequence_depth=1
            )
            depth_two = self.call_teacher(
                env, danger, mask, trap_sequence_depth=2
            )
            above_threshold = self.call_teacher(
                env,
                danger,
                mask,
                trap_sequence_depth=2,
                sequence_pressure_gain=0.350001,
            )

        self.assertEqual(depth_one, ACTION_LEFT)
        self.assertEqual(depth_two, ACTION_BOMB)
        self.assertEqual(above_threshold, ACTION_LEFT)

    def test_sequence_teacher_rejects_unsafe_placements_and_raw_spam(self):
        require_parameters(get_attack_action, *self.teacher_options)
        env, danger, mask = self.configured_case()
        with patch(
            "train_actor_critic._bomb_escape_pressure", side_effect=self.pressure_oracle
        ), patch("train_actor_critic._bfs_toward_item", return_value=None), patch(
            "train_actor_critic._bfs_toward_opponent", return_value=ACTION_LEFT
        ), patch("train_actor_critic._bfs_toward_wood", return_value=None):
            unsafe_first_mask = mask.copy()
            unsafe_first_mask[ACTION_BOMB] = False
            unsafe_first = self.call_teacher(env, danger, unsafe_first_mask)

            unsafe_followup_danger = danger.copy()
            unsafe_followup_danger[3, 3] = 1
            unsafe_followup = self.call_teacher(
                env, unsafe_followup_danger, mask
            )

        def repeated_first_pressure(_env, _agent_id, row, col, base=None):
            return 0.20 if (row, col) == (3, 2) else 0.0

        with patch(
            "train_actor_critic._bomb_escape_pressure",
            side_effect=repeated_first_pressure,
        ), patch("train_actor_critic._bfs_toward_item", return_value=None), patch(
            "train_actor_critic._bfs_toward_opponent", return_value=ACTION_LEFT
        ), patch("train_actor_critic._bfs_toward_wood", return_value=None):
            repeated_only = self.call_teacher(env, danger, mask)

        self.assertNotEqual(unsafe_first, ACTION_BOMB)
        self.assertEqual(unsafe_followup, ACTION_LEFT)
        self.assertEqual(repeated_only, ACTION_LEFT)


class RingSafetySupervisionAcceptance(unittest.TestCase):
    def run_ring_case(
        self,
        *,
        sudden_death_probability,
        sudden_death_start,
        warning_steps,
        learner_position=(1, 1),
        ring_safety_coef=0.6,
    ):
        require_parameters(train, "ring_safety_coef")
        targets = []
        leaves = []

        def fake_cross_entropy(_logits, target):
            targets.append(int(target.reshape(-1)[0].item()))
            leaf = torch.tensor(1.0, requires_grad=True)
            leaves.append(leaf)
            return leaf

        def prepare(env, _episode, learner, _actions, _opponents):
            if env.step_count != 0:
                return
            env.grid[1:-1, 1:-1] = EMPTY
            env.grid[1, 2] = RIGID_WALL
            env.agents[learner].row, env.agents[learner].col = learner_position
            other = 1 - learner
            env.agents[other].row, env.agents[other].col = 5, 5

        with patch(
            "train_actor_critic.F.cross_entropy", side_effect=fake_cross_entropy
        ), patch("train_actor_critic.get_attack_action", return_value=None), patch(
            "train_actor_critic.torch.save"
        ), patch("builtins.print"):
            train(
                episodes=1,
                board_size=7,
                num_agents=2,
                max_steps=1,
                gamma=0.0,
                value_coef=0.0,
                entropy_coef=0.0,
                device="cpu",
                seed=43,
                checkpoint_dir="unused-ring-safety-acceptance",
                validation_every=0,
                run_report=False,
                sudden_death_probability=sudden_death_probability,
                sudden_death_start=sudden_death_start,
                sudden_death_interval=4,
                sudden_death_warning_steps=warning_steps,
                ring_safety_coef=ring_safety_coef,
                on_step=prepare,
            )

        gradients = [leaf.grad.item() for leaf in leaves if leaf.grad is not None]
        return targets, gradients

    def test_warning_ring_supervises_first_safe_inward_move(self):
        targets, gradients = self.run_ring_case(
            sudden_death_probability=1.0,
            sudden_death_start=4,
            warning_steps=4,
        )
        self.assertEqual(targets, [ACTION_DOWN])
        np.testing.assert_allclose(gradients, [0.6], rtol=0, atol=1e-6)

    def test_ring_supervision_is_inactive_without_warning_threat(self):
        cases = (
            {
                "sudden_death_probability": 0.0,
                "sudden_death_start": 4,
                "warning_steps": 4,
                "learner_position": (1, 1),
            },
            {
                "sudden_death_probability": 1.0,
                "sudden_death_start": 10,
                "warning_steps": 3,
                "learner_position": (1, 1),
            },
            {
                "sudden_death_probability": 1.0,
                "sudden_death_start": 4,
                "warning_steps": 4,
                "learner_position": (3, 3),
            },
        )
        for options in cases:
            with self.subTest(options=options):
                targets, gradients = self.run_ring_case(**options)
                self.assertEqual(targets, [])
                self.assertEqual(gradients, [])


class TargetAwareCheckpointAcceptance(unittest.TestCase):
    target_values = {
        "target_win_rate": 70.0,
        "target_active_win_rate": 55.0,
        "target_draw_rate": 15.0,
        "target_sudden_death_death_rate": 10.0,
        "max_suicide_rate": 1.5,
    }

    def run_validation_sequence(self, results, *, target_mode=True):
        require_parameters(train, *TARGET_OPTIONS)
        events = []
        options = self.target_values if target_mode else {}
        with tempfile.TemporaryDirectory() as checkpoint_dir, patch(
            "evaluate.evaluate_seeded", side_effect=results
        ), patch("train_actor_critic.torch.save"), patch("builtins.print"):
            train(
                episodes=len(results) - 1,
                board_size=7,
                num_agents=2,
                max_steps=1,
                device="cpu",
                seed=47,
                checkpoint_dir=checkpoint_dir,
                validation_every=1,
                validation_episodes=2,
                validation_seed=47000,
                run_report=False,
                sudden_death_probability=0.75,
                sudden_death_start=475,
                sudden_death_interval=25,
                sudden_death_warning_steps=40,
                on_validation=events.append,
                **options,
            )
        completed = [event for event in events if "result" in event]
        self.assertEqual([event["episode"] for event in completed], list(range(len(results))))
        return [event["best_episode"] for event in completed]

    def test_target_count_precedes_worst_margin(self):
        few_targets_small_miss = validation_metrics(
            win_rate=69.3,
            active_win_rate=54.45,
            draw_rate=15.15,
            sudden_death_death_rate=10.1,
        )
        more_targets_large_miss = validation_metrics(
            win_rate=70.0,
            active_win_rate=55.0,
            draw_rate=30.0,
            sudden_death_death_rate=20.0,
        )
        self.assertEqual(
            self.run_validation_sequence(
                [few_targets_small_miss, more_targets_large_miss]
            ),
            [0, 1],
        )

    def test_worst_margin_precedes_metric_tiebreaks(self):
        worse_margin = validation_metrics(
            win_rate=70.0,
            active_win_rate=55.0,
            draw_rate=22.5,
            sudden_death_death_rate=15.0,
        )
        better_margin = validation_metrics(
            win_rate=70.0,
            active_win_rate=55.0,
            draw_rate=18.0,
            sudden_death_death_rate=12.0,
        )
        self.assertEqual(
            self.run_validation_sequence([worse_margin, better_margin]),
            [0, 1],
        )

    def test_metric_tiebreak_order(self):
        results = [
            validation_metrics(
                win_rate=70.0,
                active_win_rate=56.0,
                draw_rate=14.0,
                sudden_death_death_rate=9.0,
            ),
            validation_metrics(
                win_rate=70.0,
                active_win_rate=60.0,
                draw_rate=14.0,
                sudden_death_death_rate=9.0,
            ),
            validation_metrics(
                win_rate=70.0,
                active_win_rate=60.0,
                draw_rate=13.0,
                sudden_death_death_rate=9.0,
            ),
            validation_metrics(
                win_rate=70.0,
                active_win_rate=60.0,
                draw_rate=13.0,
                sudden_death_death_rate=8.0,
            ),
        ]
        self.assertEqual(self.run_validation_sequence(results), [0, 1, 2, 3])

    def test_total_win_rate_is_final_tiebreak(self):
        lower_total = validation_metrics(
            win_rate=71.0,
            active_win_rate=55.0,
            draw_rate=14.0,
            sudden_death_death_rate=9.0,
        )
        higher_total = validation_metrics(
            win_rate=72.0,
            active_win_rate=55.0,
            draw_rate=14.0,
            sudden_death_death_rate=9.0,
        )
        self.assertEqual(
            self.run_validation_sequence([lower_total, higher_total]),
            [0, 1],
        )

    def test_suicide_gate_rejects_otherwise_better_candidate(self):
        eligible = validation_metrics(suicide_rate=1.5)
        ineligible = validation_metrics(
            win_rate=95.0,
            active_win_rate=90.0,
            draw_rate=1.0,
            sudden_death_death_rate=1.0,
            suicide_rate=1.5001,
        )
        self.assertEqual(
            self.run_validation_sequence([eligible, ineligible]),
            [0, 0],
        )

    def test_warm_start_episode_zero_remains_eligible(self):
        warm_start = validation_metrics(
            win_rate=80.0,
            active_win_rate=65.0,
            draw_rate=8.0,
            sudden_death_death_rate=5.0,
            suicide_rate=1.0,
        )
        later = validation_metrics(
            win_rate=70.0,
            active_win_rate=55.0,
            draw_rate=15.0,
            sudden_death_death_rate=10.0,
            suicide_rate=1.0,
        )
        self.assertEqual(
            self.run_validation_sequence([warm_start, later]),
            [0, 0],
        )

    def test_disabled_targets_preserve_legacy_selection(self):
        warm_start = validation_metrics(
            win_rate=61.0,
            active_win_rate=40.0,
            draw_rate=20.0,
            suicide_rate=1.0,
        )
        later = validation_metrics(
            win_rate=1.0,
            active_win_rate=41.0,
            draw_rate=99.0,
            sudden_death_death_rate=99.0,
            suicide_rate=1.0,
            avg_kills=0.0,
        )
        self.assertEqual(
            self.run_validation_sequence([warm_start, later], target_mode=False),
            [0, 1],
        )


class PhaseCSDTarget750LauncherAcceptance(unittest.TestCase):
    launcher_name = "run_train_phaseC_sd_target_750.ps1"
    outcome_fields = (
        "win",
        "active_win",
        "sudden_death_win",
        "passive_win",
        "suicide",
        "sudden_death_death",
        "draw",
        "kills",
        "bombs",
        "items",
        "steps",
    )

    def valid_evaluation_payload(self, **overrides):
        games = []
        for index in range(200):
            win = int(index < 140)
            active_win = int(index < 110)
            sudden_death_win = int(110 <= index < 130)
            passive_win = int(130 <= index < 140)
            games.append(
                {
                    "seed": 31000 + index // 2,
                    "aid": index % 2,
                    "win": win,
                    "active_win": active_win,
                    "sudden_death_win": sudden_death_win,
                    "passive_win": passive_win,
                    "suicide": int(index >= 197),
                    "sudden_death_death": int(index >= 180),
                    "draw": int(index >= 170),
                    "kills": float(active_win),
                    "bombs": 1.0,
                    "items": 0.0,
                    "steps": 100,
                }
            )
        payload = {
            "seed": 31000,
            "board_size": 9,
            "num_agents": 2,
            "max_steps": 500,
            "sudden_death": True,
            "sudden_death_start": 475,
            "sudden_death_interval": 25,
            "sudden_death_warning_steps": 40,
            "num_episodes": 200,
            "games": games,
            "win_rate": 70.0,
            "active_win_rate": 55.0,
            "draw_rate": 15.0,
            "sudden_death_death_rate": 10.0,
            "suicide_rate": 1.5,
        }
        payload.update(overrides)
        return payload

    def launcher_source(self):
        launcher = Path(__file__).resolve().parent / self.launcher_name
        self.assertTrue(launcher.is_file(), f"Missing required launcher: {launcher}")
        return launcher

    def stage_launcher(
        self,
        *,
        checkpoint=True,
        existing_output=False,
        output_best=False,
    ):
        launcher = self.launcher_source()
        temporary = tempfile.TemporaryDirectory(dir=launcher.parent)
        root = Path(temporary.name)
        shutil.copy2(launcher, root / launcher.name)

        scripts_dir = root / "venv" / "Scripts"
        scripts_dir.mkdir(parents=True)
        repository_venv = launcher.parent / "venv"
        shutil.copy2(repository_venv / "Scripts" / "python.exe", scripts_dir / "python.exe")
        shutil.copy2(repository_venv / "pyvenv.cfg", root / "venv" / "pyvenv.cfg")

        checkpoint_path = root / "runs" / "phaseC-sd-attack-750" / "best_phaseC.pt"
        if checkpoint:
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"warm-start")

        output_directory = root / "runs" / "phaseC-sd-target-750"
        if existing_output:
            output_directory.mkdir(parents=True)
            (output_directory / "do-not-overwrite.txt").write_text(
                "preserve", encoding="utf-8"
            )
        if output_best:
            output_directory.mkdir(parents=True, exist_ok=True)
            (output_directory / "best_phaseC.pt").write_bytes(b"existing-best")

        fake_program = textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path
            import sys

            def option(name):
                index = sys.argv.index(name)
                return sys.argv[index + 1]

            kind = Path(__file__).stem
            log_path = Path(os.environ["ACCEPTANCE_CALL_LOG"])
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": kind, "argv": sys.argv[1:]}) + "\\n")
            print(f"FAKE_SUBPROCESS_START:{kind}", flush=True)

            if kind == "train_actor_critic":
                exit_code = int(os.environ.get("FAKE_TRAIN_EXIT", "0"))
                if exit_code == 0 and os.environ.get("FAKE_CREATE_BEST", "1") == "1":
                    output = Path(option("--checkpoint_dir"))
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "best_phaseC.pt").write_bytes(b"best")
                raise SystemExit(exit_code)

            exit_code = int(os.environ.get("FAKE_EVAL_EXIT", "0"))
            if exit_code == 0 and os.environ.get("FAKE_EVAL_ARTIFACT", "write") == "write":
                output = Path(option("--output"))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(os.environ["FAKE_EVAL_PAYLOAD"], encoding="utf-8")
            raise SystemExit(exit_code)
            """
        ).strip()
        (root / "train_actor_critic.py").write_text(fake_program, encoding="utf-8")
        (root / "evaluate.py").write_text(fake_program, encoding="utf-8")
        return temporary, root, output_directory

    def run_launcher(
        self,
        *,
        checkpoint=True,
        existing_output=False,
        output_best=False,
        evaluate_only=False,
        train_exit=0,
        create_best=True,
        evaluation_exit=0,
        evaluation_artifact="write",
        evaluation_payload=None,
        metrics=None,
    ):
        temporary, root, output_directory = self.stage_launcher(
            checkpoint=checkpoint,
            existing_output=existing_output,
            output_best=output_best,
        )
        try:
            call_log = root / "calls.jsonl"
            payload = self.valid_evaluation_payload()
            if evaluation_payload is not None:
                payload = evaluation_payload
            if metrics is not None:
                payload.update(metrics)
            environment = os.environ.copy()
            environment.update(
                {
                    "ACCEPTANCE_CALL_LOG": str(call_log),
                    "FAKE_TRAIN_EXIT": str(train_exit),
                    "FAKE_CREATE_BEST": "1" if create_best else "0",
                    "FAKE_EVAL_EXIT": str(evaluation_exit),
                    "FAKE_EVAL_ARTIFACT": evaluation_artifact,
                    "FAKE_EVAL_PAYLOAD": json.dumps(payload),
                }
            )
            command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(root / self.launcher_name),
            ]
            if evaluate_only:
                command.append("-EvaluateOnly")
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            calls = []
            if call_log.is_file():
                calls = [
                    json.loads(line)
                    for line in call_log.read_text(encoding="utf-8").splitlines()
                ]
            evaluation_path = output_directory / "evaluation-200.json"
            return {
                "completed": completed,
                "calls": calls,
                "output": completed.stdout + completed.stderr,
                "root": root,
                "output_directory": output_directory,
                "evaluation_exists": evaluation_path.is_file(),
                "sentinel": (
                    output_directory / "do-not-overwrite.txt"
                ).read_text(encoding="utf-8")
                if (output_directory / "do-not-overwrite.txt").is_file()
                else None,
            }
        finally:
            temporary.cleanup()

    def assert_option(self, arguments, name, expected):
        self.assertEqual(arguments.count(name), 1, f"Expected exactly one {name}: {arguments}")
        index = arguments.index(name)
        self.assertLess(index + 1, len(arguments), f"Missing value for {name}")
        self.assertEqual(arguments[index + 1], str(expected))

    def option_value(self, arguments, name):
        self.assertEqual(arguments.count(name), 1, f"Expected exactly one {name}: {arguments}")
        return arguments[arguments.index(name) + 1]

    def assert_gate_status(self, output, label, status):
        patterns = {
            "WR": r"(?<![A-Za-z])WR(?![A-Za-z])",
            "ActiveWR": r"ActiveWR",
            "Draw": r"Draw",
            "SD death": r"(?:SD|Sudden Death)\s+death",
            "Suicide": r"Suicide",
        }
        matching = [
            line
            for line in output.splitlines()
            if re.search(patterns[label], line, flags=re.IGNORECASE)
        ]
        self.assertTrue(matching, f"Missing gate output for {label}:\n{output}")
        self.assertTrue(
            any(status.lower() in line.lower() for line in matching),
            f"Expected {status} for {label}: {matching}",
        )

    def test_launcher_training_contract_is_exact(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        self.assertEqual(
            [call["kind"] for call in result["calls"]],
            ["train_actor_critic", "evaluate"],
        )
        arguments = result["calls"][0]["argv"]
        expected = {
            "--phase": "C",
            "--episodes": "750",
            "--board_size": "9",
            "--num_agents": "2",
            "--max_steps": "500",
            "--lr": "0.000025",
            "--entropy_coef": "0.0075",
            "--wall_destroy_bonus": "0.15",
            "--attack_lookahead": "2",
            "--min_pressure_gain": "0.15",
            "--trap_sequence_depth": "2",
            "--sequence_pressure_gain": "0.35",
            "--ring_safety_coef": "0.6",
            "--late_game_start": "350",
            "--late_game_attack_multiplier": "1.75",
            "--escape_loss_coef": "0.5",
            "--combat_loss_coef": "0.9",
            "--navigation_loss_coef": "0.4",
            "--validation_every": "250",
            "--validation_episodes": "200",
            "--sudden_death_probability": "0.75",
            "--sudden_death_start": "475",
            "--sudden_death_interval": "25",
            "--sudden_death_warning_steps": "40",
            "--target_win_rate": "70",
            "--target_active_win_rate": "55",
            "--target_draw_rate": "15",
            "--target_sudden_death_death_rate": "10",
            "--max_suicide_rate": "1.5",
        }
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assert_option(arguments, option, value)
        self.assert_option(
            arguments,
            "--resume_from",
            result["root"] / "runs" / "phaseC-sd-attack-750" / "best_phaseC.pt",
        )
        self.assert_option(arguments, "--checkpoint_dir", result["output_directory"])
        int(self.option_value(arguments, "--validation_seed"))

    def test_success_path_evaluates_generated_best_exactly_once(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        self.assertEqual(
            [call["kind"] for call in result["calls"]],
            ["train_actor_critic", "evaluate"],
        )
        train_arguments = result["calls"][0]["argv"]
        arguments = result["calls"][1]["argv"]
        expected = {
            "--mode": "seeded",
            "--num_episodes": "200",
            "--board_size": "9",
            "--num_agents": "2",
            "--max_steps": "500",
            "--seed": "31000",
            "--device": "cpu",
            "--sudden_death_start": "475",
            "--sudden_death_interval": "25",
            "--sudden_death_warning_steps": "40",
            "--ac_model": result["output_directory"] / "best_phaseC.pt",
            "--output": result["output_directory"] / "evaluation-200.json",
        }
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assert_option(arguments, option, value)
        self.assertEqual(arguments.count("--sudden_death"), 1)
        validation_seed = int(self.option_value(train_arguments, "--validation_seed"))
        evaluation_seed = int(self.option_value(arguments, "--seed"))
        self.assertNotEqual(validation_seed, evaluation_seed)
        self.assertTrue(result["evaluation_exists"])

    def test_fail_fast_and_no_overwrite_contract(self):
        for options in ({"checkpoint": False}, {"existing_output": True}):
            with self.subTest(stage="preflight", options=options):
                result = self.run_launcher(**options)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])
                self.assertEqual(result["calls"], [])
                if options.get("existing_output"):
                    self.assertEqual(result["sentinel"], "preserve")

        for options in ({"train_exit": 17}, {"create_best": False}):
            with self.subTest(stage="training", options=options):
                result = self.run_launcher(**options)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])
                self.assertEqual(
                    [call["kind"] for call in result["calls"]],
                    ["train_actor_critic"],
                )

        result = self.run_launcher(evaluation_exit=19)
        self.assertNotEqual(result["completed"].returncode, 0, result["output"])
        self.assertEqual(
            [call["kind"] for call in result["calls"]],
            ["train_actor_critic", "evaluate"],
        )

    def test_gate_boundaries_report_without_process_failure(self):
        passing = self.run_launcher()
        self.assertEqual(passing["completed"].returncode, 0, passing["output"])
        labels = ("WR", "ActiveWR", "Draw", "SD death", "Suicide")
        for label in labels:
            self.assert_gate_status(passing["output"], label, "PASS")

        failing_payload = self.valid_evaluation_payload()
        for index, game in enumerate(failing_payload["games"]):
            game.update(
                {
                    "win": int(index < 139),
                    "active_win": int(index < 109),
                    "sudden_death_win": int(109 <= index < 129),
                    "passive_win": int(129 <= index < 139),
                    "draw": int(index >= 169),
                    "sudden_death_death": int(index >= 179),
                    "suicide": int(index >= 196),
                    "kills": float(index < 109),
                }
            )
        failing_payload.update(
            {
                "win_rate": 69.5,
                "active_win_rate": 54.5,
                "draw_rate": 15.5,
                "sudden_death_death_rate": 10.5,
                "suicide_rate": 2.0,
            }
        )
        failing = self.run_launcher(evaluation_payload=failing_payload)
        self.assertEqual(failing["completed"].returncode, 0, failing["output"])
        for label in labels:
            self.assert_gate_status(failing["output"], label, "FAIL")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
