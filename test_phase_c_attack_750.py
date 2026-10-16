"""Phase C attack-teacher acceptance tests.

Run with: venv\Scripts\python.exe -m unittest -v test_phase_c_attack_750.py
"""

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
    ACTION_RIGHT,
    ACTION_STOP,
    Bomb,
    BombermanFFAEnv,
    EMPTY,
)
from rule_based_bot import RuleBasedAgent
from train_actor_critic import _combat_reward, get_attack_action, train


def require_parameters(callable_object, *names):
    parameters = inspect.signature(callable_object).parameters
    missing = set(names) - parameters.keys()
    if missing:
        raise AssertionError(f"{callable_object.__name__} is missing required options: {sorted(missing)}")


def arena(num_agents=2, board_size=7):
    env = BombermanFFAEnv(
        board_size=board_size,
        num_agents=num_agents,
        wooden_wall_density=0,
        powerup_prob=0,
        seed=17,
    )
    env.reset()
    env.grid[1:-1, 1:-1] = EMPTY
    return env


def put_bomb(env, row, col, owner, timer=1, blast_range=1):
    env.bombs.append(Bomb(row, col, owner, timer, blast_range))
    env.agents[owner].bombs_placed += 1


class PhaseCAttackTeacherAcceptance(unittest.TestCase):
    required_teacher_options = ("attack_lookahead", "min_pressure_gain")

    def test_frozen_rulebot_and_environment_reward_contract(self):
        rule_env = arena(2)
        rule_env.agents[0].row, rule_env.agents[0].col = 3, 1
        rule_env.agents[1].row, rule_env.agents[1].col = 3, 3
        self.assertEqual(RuleBasedAgent(0).select_action(rule_env), ACTION_BOMB)

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

    def test_two_step_pressure_candidate_requires_lookahead_two(self):
        require_parameters(get_attack_action, *self.required_teacher_options)
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[0].blast_range = 3
        env.agents[1].row, env.agents[1].col = 1, 5
        danger = np.zeros((env.board_size, env.board_size), dtype=np.int32)
        mask = [True] * 6

        def pressure(_env, _agent_id, row, col, base=None):
            return 0.30 if (row, col) == (3, 5) else 0.0

        with patch("train_actor_critic._bomb_escape_pressure", side_effect=pressure), \
             patch("train_actor_critic._bfs_toward_item", return_value=None), \
             patch("train_actor_critic._bfs_toward_opponent", return_value=ACTION_LEFT), \
             patch("train_actor_critic._bfs_toward_wood", return_value=None):
            adjacent_only = get_attack_action(
                env,
                0,
                danger,
                mask,
                attack_lookahead=1,
                min_pressure_gain=0.15,
            )
            two_steps = get_attack_action(
                env,
                0,
                danger,
                mask,
                attack_lookahead=2,
                min_pressure_gain=0.15,
            )

        self.assertEqual(adjacent_only, ACTION_LEFT)
        self.assertEqual(two_steps, ACTION_RIGHT)

    def test_pressure_threshold_is_inclusive_and_preserves_fallback_guidance(self):
        require_parameters(get_attack_action, *self.required_teacher_options)
        env = arena(2)
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[0].blast_range = 3
        env.agents[1].row, env.agents[1].col = 1, 5
        danger = np.zeros((env.board_size, env.board_size), dtype=np.int32)
        mask = [True] * 6

        def threshold_pressure(_env, _agent_id, row, col, base=None):
            return 0.15 if (row, col) == (3, 4) else 0.0

        with patch("train_actor_critic._bomb_escape_pressure", side_effect=threshold_pressure), \
             patch("train_actor_critic._bfs_toward_item", return_value=None), \
             patch("train_actor_critic._bfs_toward_opponent", return_value=ACTION_LEFT), \
             patch("train_actor_critic._bfs_toward_wood", return_value=None):
            self.assertEqual(
                get_attack_action(
                    env,
                    0,
                    danger,
                    mask,
                    attack_lookahead=1,
                    min_pressure_gain=0.15,
                ),
                ACTION_RIGHT,
            )
            self.assertEqual(
                get_attack_action(
                    env,
                    0,
                    danger,
                    mask,
                    attack_lookahead=1,
                    min_pressure_gain=0.150001,
                ),
                ACTION_LEFT,
            )

        env.agents[1].row, env.agents[1].col = 3, 5
        with patch("train_actor_critic._bomb_escape_pressure", return_value=0.0), \
             patch("train_actor_critic._bfs_toward_item", return_value=None), \
             patch("train_actor_critic._bfs_toward_opponent", return_value=ACTION_DOWN), \
             patch("train_actor_critic._bfs_toward_wood", return_value=None):
            self.assertEqual(
                get_attack_action(
                    env,
                    0,
                    danger,
                    mask,
                    attack_lookahead=2,
                    min_pressure_gain=0.15,
                ),
                ACTION_BOMB,
                "A direct blast remains usable even when its pressure gain is below the pressure label threshold",
            )

        env.agents[0].blast_range = 2
        env.agents[1].row, env.agents[1].col = 1, 5
        with patch("train_actor_critic._bomb_escape_pressure", return_value=0.0), \
             patch("train_actor_critic._bfs_toward_item", return_value=None), \
             patch("train_actor_critic._bfs_toward_opponent", return_value=ACTION_DOWN), \
             patch("train_actor_critic._bfs_toward_wood", return_value=None):
            self.assertEqual(
                get_attack_action(
                    env,
                    0,
                    danger,
                    mask,
                    attack_lookahead=2,
                    min_pressure_gain=0.15,
                ),
                ACTION_DOWN,
                "Ordinary opponent navigation remains available without pressure combat",
            )


class PhaseCTrainingConfigurationAcceptance(unittest.TestCase):
    expected_defaults = {
        "attack_lookahead": 2,
        "min_pressure_gain": 0.15,
        "late_game_start": 350,
        "late_game_attack_multiplier": 1.75,
        "escape_loss_coef": 0.5,
        "combat_loss_coef": 0.9,
        "navigation_loss_coef": 0.4,
        "entropy_coef": 0.0075,
        "wall_destroy_bonus": 0.15,
    }

    def test_train_api_exposes_frozen_defaults(self):
        parameters = inspect.signature(train).parameters
        missing = set(self.expected_defaults) - parameters.keys()
        self.assertFalse(missing, f"train is missing required options: {sorted(missing)}")
        for name, expected in self.expected_defaults.items():
            with self.subTest(name=name):
                self.assertEqual(parameters[name].default, expected)

    def test_training_cli_exposes_required_flags(self):
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
            self.assertIn(f"--{name}", completed.stdout)


class PhaseCTrainingLossAcceptance(unittest.TestCase):
    required_options = (
        "attack_lookahead",
        "min_pressure_gain",
        "late_game_start",
        "late_game_attack_multiplier",
        "escape_loss_coef",
        "combat_loss_coef",
        "navigation_loss_coef",
    )

    def _train_pressure_case(self, threshold):
        metrics = []

        def pressure(_env, _agent_id, row, col, base=None):
            return 0.20 if (row, col) == (1, 3) else 0.0

        def prepare(env, _episode, learner, _actions, _opponents):
            if env.step_count == 0:
                env.agents[learner].blast_range = 3
                env.grid[1, 3] = EMPTY

        with patch("train_actor_critic._bomb_escape_pressure", side_effect=pressure), \
             patch("train_actor_critic.torch.save"), \
             patch("train_actor_critic.os.makedirs"), \
             patch("builtins.print"):
            train(
                episodes=1,
                board_size=7,
                num_agents=2,
                max_steps=1,
                device="cpu",
                seed=91,
                checkpoint_dir="unused-attack-acceptance",
                run_report=False,
                validation_every=0,
                attack_lookahead=2,
                min_pressure_gain=threshold,
                on_step=prepare,
                on_episode=metrics.append,
            )
        self.assertEqual(len(metrics), 1)
        return metrics[0]

    def test_min_pressure_gain_controls_combat_labeling(self):
        require_parameters(train, *self.required_options)
        qualifying = self._train_pressure_case(0.15)
        below_threshold = self._train_pressure_case(0.25)

        self.assertEqual(qualifying["attack_labels"], 1)
        self.assertEqual(qualifying["combat_labels"], 1)
        self.assertGreater(qualifying["combat_loss"], 0)
        self.assertEqual(qualifying["navigation_loss"], 0)

        self.assertEqual(below_threshold["attack_labels"], 1)
        self.assertEqual(below_threshold["combat_labels"], 0)
        self.assertEqual(below_threshold["combat_loss"], 0)
        self.assertGreater(below_threshold["navigation_loss"], 0)

    def _loss_probe(self, kind, multiplier):
        leaves = []
        metrics = []

        def fake_cross_entropy(*_args, **_kwargs):
            leaf = torch.tensor(1.0, requires_grad=True)
            leaves.append(leaf)
            return leaf

        def action_mask(_env, _agent_id):
            mask = [True] * 6
            if kind == "escape":
                mask[ACTION_BOMB] = False
            return mask

        def danger_map(env):
            danger = np.zeros((env.board_size, env.board_size), dtype=np.int32)
            if kind == "escape":
                learner = env.agents[0]
                danger[learner.row, learner.col] = 1
            return danger

        target_action = ACTION_BOMB if kind == "combat" else ACTION_RIGHT
        pressure = 1.0 if kind == "combat" else 0.0

        with patch("train_actor_critic.F.cross_entropy", side_effect=fake_cross_entropy), \
             patch("train_actor_critic.get_action_mask", side_effect=action_mask), \
             patch("train_actor_critic._build_danger_map", side_effect=danger_map), \
             patch("train_actor_critic.get_attack_action", return_value=target_action), \
             patch("train_actor_critic._bomb_escape_pressure", return_value=pressure), \
             patch("train_actor_critic._bfs_safe_cell", return_value=ACTION_RIGHT), \
             patch("train_actor_critic.torch.save"), \
             patch("train_actor_critic.os.makedirs"), \
             patch("builtins.print"):
            train(
                episodes=1,
                board_size=7,
                num_agents=2,
                max_steps=2,
                gamma=0.0,
                value_coef=0.0,
                entropy_coef=0.0,
                device="cpu",
                seed=123,
                checkpoint_dir="unused-loss-acceptance",
                run_report=False,
                validation_every=0,
                attack_lookahead=2,
                min_pressure_gain=0.15,
                late_game_start=1,
                late_game_attack_multiplier=multiplier,
                escape_loss_coef=0.2,
                combat_loss_coef=0.6,
                navigation_loss_coef=0.4,
                on_episode=metrics.append,
            )

        self.assertEqual(len(leaves), 2, f"Expected one {kind} teacher loss per step")
        self.assertTrue(all(leaf.grad is not None for leaf in leaves))
        self.assertEqual(len(metrics), 1)
        return [leaf.grad.item() for leaf in leaves], metrics[0]

    def test_configurable_loss_weights_and_late_game_boundary(self):
        require_parameters(train, *self.required_options)
        escape_gradients, _ = self._loss_probe("escape", multiplier=3.0)
        combat_gradients, late_metrics = self._loss_probe("combat", multiplier=3.0)
        navigation_gradients, _ = self._loss_probe("navigation", multiplier=3.0)
        _, base_metrics = self._loss_probe("combat", multiplier=1.0)

        np.testing.assert_allclose(escape_gradients, [0.1, 0.1], rtol=0, atol=1e-6)
        np.testing.assert_allclose(combat_gradients, [0.3, 0.9], rtol=0, atol=1e-6)
        np.testing.assert_allclose(navigation_gradients, [0.2, 0.6], rtol=0, atol=1e-6)
        self.assertEqual(late_metrics["reward"], base_metrics["reward"])
        self.assertAlmostEqual(late_metrics["policy_loss"], base_metrics["policy_loss"], places=6)
        self.assertAlmostEqual(late_metrics["value_loss"], base_metrics["value_loss"], places=6)


class PhaseCAttack750LauncherAcceptance(unittest.TestCase):
    launcher_name = "run_train_phaseC_sd_attack_750.ps1"

    def launcher_source(self):
        launcher = Path(__file__).resolve().parent / self.launcher_name
        self.assertTrue(launcher.is_file(), f"Missing required launcher: {launcher}")
        return launcher, launcher.read_text(encoding="utf-8-sig")

    def configured_relative_path(self, source, variable):
        match = re.search(
            rf'\${variable}\s*=\s*Join-Path\s+\$PSScriptRoot\s+["\']([^"\']+)["\']',
            source,
            flags=re.IGNORECASE,
        )
        self.assertIsNotNone(match, f"Launcher must define ${variable} with Join-Path $PSScriptRoot")
        return Path(match.group(1).replace("\\", os.sep))

    def stage_launcher(self, *, interpreter=True, checkpoint=True, existing_output=False):
        launcher, source = self.launcher_source()
        temporary = tempfile.TemporaryDirectory(dir=launcher.parent)
        root = Path(temporary.name)
        shutil.copy2(launcher, root / launcher.name)

        if interpreter:
            scripts_dir = root / "venv" / "Scripts"
            scripts_dir.mkdir(parents=True)
            repository_venv = launcher.parent / "venv"
            shutil.copy2(repository_venv / "Scripts" / "python.exe", scripts_dir / "python.exe")
            shutil.copy2(repository_venv / "pyvenv.cfg", root / "venv" / "pyvenv.cfg")

        checkpoint_path = root / "runs" / "phaseC-sd-1500-e" / "best_phaseC.pt"
        if checkpoint:
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"warm-start")

        output_relative = self.configured_relative_path(source, "OutputDirectory")
        output_directory = root / output_relative
        if existing_output:
            output_directory.mkdir(parents=True)
            (output_directory / "do-not-overwrite.txt").write_text("preserve", encoding="utf-8")

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

            if kind == "train_actor_critic":
                exit_code = int(os.environ.get("FAKE_TRAIN_EXIT", "0"))
                if exit_code == 0 and os.environ.get("FAKE_CREATE_BEST", "1") == "1":
                    output = Path(option("--checkpoint_dir"))
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "best_phaseC.pt").write_bytes(b"best")
                raise SystemExit(exit_code)

            exit_code = int(os.environ.get("FAKE_EVAL_EXIT", "0"))
            if exit_code == 0:
                output = Path(option("--output"))
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(os.environ["FAKE_EVAL_METRICS"], encoding="utf-8")
            raise SystemExit(exit_code)
            """
        ).strip()
        (root / "train_actor_critic.py").write_text(fake_program, encoding="utf-8")
        (root / "evaluate.py").write_text(fake_program, encoding="utf-8")
        return temporary, root, output_directory

    def run_launcher(
        self,
        *,
        interpreter=True,
        checkpoint=True,
        existing_output=False,
        train_exit=0,
        create_best=True,
        evaluation_exit=0,
        metrics=None,
    ):
        temporary, root, output_directory = self.stage_launcher(
            interpreter=interpreter,
            checkpoint=checkpoint,
            existing_output=existing_output,
        )
        try:
            call_log = root / "calls.jsonl"
            environment = os.environ.copy()
            environment.update(
                {
                    "ACCEPTANCE_CALL_LOG": str(call_log),
                    "FAKE_TRAIN_EXIT": str(train_exit),
                    "FAKE_CREATE_BEST": "1" if create_best else "0",
                    "FAKE_EVAL_EXIT": str(evaluation_exit),
                    "FAKE_EVAL_METRICS": json.dumps(
                        metrics
                        or {
                            "active_win_rate": 38.0,
                            "win_rate": 60.5,
                            "draw_rate": 23.5,
                            "suicide_rate": 1.5,
                            "sudden_death_death_rate": 18.5,
                        }
                    ),
                }
            )
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(root / self.launcher_name),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            calls = []
            if call_log.is_file():
                calls = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
            evaluation_path = output_directory / "evaluation-200.json"
            return {
                "completed": completed,
                "calls": calls,
                "output": completed.stdout + completed.stderr,
                "output_directory": str(output_directory),
                "evaluation_exists": evaluation_path.is_file(),
                "sentinel": (output_directory / "do-not-overwrite.txt").read_text(encoding="utf-8")
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

    def assert_gate_status(self, output, metric, status):
        matching = [line for line in output.splitlines() if metric.lower() in line.lower()]
        self.assertTrue(matching, f"Missing gate output for {metric}:\n{output}")
        self.assertTrue(
            any(status.lower() in line.lower() for line in matching),
            f"Expected {status} for {metric}: {matching}",
        )

    def test_launcher_training_contract_is_exact(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        self.assertEqual([call["kind"] for call in result["calls"]], ["train_actor_critic", "evaluate"])
        arguments = result["calls"][0]["argv"]
        expected = {
            "--phase": "C",
            "--episodes": "750",
            "--board_size": "9",
            "--num_agents": "2",
            "--max_steps": "500",
            "--lr": "0.00005",
            "--entropy_coef": "0.01",
            "--wall_destroy_bonus": "0.15",
            "--attack_lookahead": "2",
            "--min_pressure_gain": "0.15",
            "--late_game_start": "400",
            "--late_game_attack_multiplier": "1.5",
            "--escape_loss_coef": "0.5",
            "--combat_loss_coef": "0.75",
            "--navigation_loss_coef": "0.35",
            "--validation_every": "250",
            "--validation_episodes": "100",
            "--sudden_death_probability": "0.75",
            "--sudden_death_start": "475",
            "--sudden_death_interval": "25",
            "--sudden_death_warning_steps": "40",
        }
        for option, value in expected.items():
            self.assert_option(arguments, option, value)
        launcher_root = Path(result["completed"].args[-1]).parent
        self.assert_option(
            arguments,
            "--resume_from",
            str(launcher_root / "runs" / "phaseC-sd-1500-e" / "best_phaseC.pt"),
        )
        self.assert_option(arguments, "--checkpoint_dir", result["output_directory"])
        self.assertEqual(Path(result["output_directory"]).name, "phaseC-sd-attack-750")

    def test_success_path_evaluates_generated_best_with_schedule_e(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        self.assertEqual([call["kind"] for call in result["calls"]], ["train_actor_critic", "evaluate"])
        arguments = result["calls"][1]["argv"]
        expected = {
            "--mode": "seeded",
            "--num_episodes": "200",
            "--board_size": "9",
            "--num_agents": "2",
            "--max_steps": "500",
            "--device": "cpu",
            "--sudden_death_start": "475",
            "--sudden_death_interval": "25",
            "--sudden_death_warning_steps": "40",
            "--ac_model": str(Path(result["output_directory"]) / "best_phaseC.pt"),
            "--output": str(Path(result["output_directory"]) / "evaluation-200.json"),
        }
        for option, value in expected.items():
            self.assert_option(arguments, option, value)
        self.assertEqual(arguments.count("--sudden_death"), 1)
        self.assertEqual(arguments.count("--seed"), 1)
        int(arguments[arguments.index("--seed") + 1])
        self.assertTrue(result["evaluation_exists"])

    def test_fail_fast_and_no_overwrite_contract(self):
        preflight_cases = (
            {"interpreter": False},
            {"checkpoint": False},
            {"existing_output": True},
        )
        for options in preflight_cases:
            with self.subTest(stage="preflight", **options):
                result = self.run_launcher(**options)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])
                self.assertEqual(result["calls"], [])
                if options.get("existing_output"):
                    self.assertEqual(result["sentinel"], "preserve")

        for options in ({"train_exit": 17}, {"create_best": False}):
            with self.subTest(stage="training", **options):
                result = self.run_launcher(**options)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])
                self.assertEqual([call["kind"] for call in result["calls"]], ["train_actor_critic"])

        result = self.run_launcher(evaluation_exit=19)
        self.assertNotEqual(result["completed"].returncode, 0, result["output"])
        self.assertEqual([call["kind"] for call in result["calls"]], ["train_actor_critic", "evaluate"])

    def test_existing_five_gate_boundaries_are_preserved(self):
        passing = self.run_launcher()
        self.assertEqual(passing["completed"].returncode, 0, passing["output"])
        metrics = (
            "active_win_rate",
            "win_rate",
            "draw_rate",
            "suicide_rate",
            "sudden_death_death_rate",
        )
        for metric in metrics:
            self.assert_gate_status(passing["output"], metric, "PASS")

        failing = self.run_launcher(
            metrics={
                "active_win_rate": 37.9,
                "win_rate": 60.4,
                "draw_rate": 23.6,
                "suicide_rate": 1.6,
                "sudden_death_death_rate": 18.6,
            }
        )
        self.assertEqual(failing["completed"].returncode, 0, failing["output"])
        for metric in metrics:
            self.assert_gate_status(failing["output"], metric, "FAIL")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
