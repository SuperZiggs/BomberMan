"""Phase C Sudden Death acceptance tests: python -m unittest -v test_sudden_death.py."""

import inspect
import json
import math
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
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STOP,
    EMPTY,
    POWERUP_AMMO,
    RIGID_WALL,
    WOODEN_WALL,
    Bomb,
    BombermanFFAEnv,
)
from evaluate import evaluate_seeded
from train_actor_critic import ActorCriticNet, get_action_mask, smart_load_weights, train


LEGACY_INFO_KEYS = {
    "episode_done",
    "alive",
    "step",
    "walls_destroyed",
    "kills",
    "self_blast_deaths",
    "bombs_placed",
    "items_collected",
}


def arena(num_agents=2, board_size=7, **kwargs):
    sd_options = tuple(name for name in kwargs if name.startswith("sudden_death"))
    require_parameters(BombermanFFAEnv, *sd_options)
    env = BombermanFFAEnv(
        board_size=board_size,
        num_agents=num_agents,
        max_steps=20,
        wooden_wall_density=0,
        powerup_prob=0,
        seed=7,
        **kwargs,
    )
    env.reset()
    return env


def ring_cells(board_size, depth):
    low = depth
    high = board_size - 1 - depth
    return {
        (row, col)
        for row in range(low, high + 1)
        for col in range(low, high + 1)
        if row in (low, high) or col in (low, high)
    }


def ring_plane(board_size, depth, value=1.0):
    plane = np.zeros((board_size, board_size), dtype=np.float32)
    for row, col in ring_cells(board_size, depth):
        plane[row, col] = value
    return plane


def metric(mapping, *names):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise AssertionError(f"Missing required metric; expected one of {names}, got {sorted(mapping)}")


def require_parameters(callable_object, *names):
    parameters = inspect.signature(callable_object).parameters
    missing = set(names) - parameters.keys()
    if missing:
        raise AssertionError(f"{callable_object.__name__} is missing required options: {sorted(missing)}")


def configure_bomb_escape_corridor(env):
    env.grid[1:-1, 1:-1] = RIGID_WALL
    for pos in ((1, 1), (2, 1), (3, 1), (3, 2), (3, 3)):
        env.grid[pos] = EMPTY
    env.grid[3, 4] = WOODEN_WALL
    env.agents[0].row, env.agents[0].col = 3, 2
    env.agents[1].row, env.agents[1].col = 5, 5


class StopOpponent:
    def select_action(self, *args, **kwargs):
        return ACTION_STOP


class SuddenDeathEnvironmentAcceptance(unittest.TestCase):
    def test_public_configuration_and_invalid_boundaries(self):
        parameters = inspect.signature(BombermanFFAEnv).parameters
        required = {
            "sudden_death",
            "sudden_death_start",
            "sudden_death_interval",
            "sudden_death_warning_steps",
        }
        self.assertTrue(required <= parameters.keys(), f"Missing constructor options: {required - parameters.keys()}")
        self.assertIs(parameters["sudden_death"].default, False)

        cases = (
            {"sudden_death_start": 0},
            {"sudden_death_interval": 0},
            {"sudden_death_warning_steps": -1},
        )
        for invalid in cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, AssertionError)):
                    BombermanFFAEnv(sudden_death=True, **invalid)

    def test_default_observation_is_thirteen_channels_with_zero_warning(self):
        env = arena()
        observations = env.reset()
        self.assertEqual(observations[0].shape, (13, 7, 7))
        self.assertEqual(observations[0].dtype, np.float32)
        np.testing.assert_array_equal(observations[0][12], np.zeros((7, 7), dtype=np.float32))

    def test_disabled_mode_preserves_legacy_trajectory_reward_and_info(self):
        default_env = arena()
        disabled_env = arena(
            sudden_death=False,
            sudden_death_start=1,
            sudden_death_interval=1,
            sudden_death_warning_steps=3,
        )
        default_obs = default_env.reset()
        disabled_obs = disabled_env.reset()
        for left, right in zip(default_obs, disabled_obs):
            np.testing.assert_array_equal(left[:12], right[:12])
            np.testing.assert_array_equal(left[12], 0)
            np.testing.assert_array_equal(right[12], 0)

        for actions in ([ACTION_STOP, ACTION_STOP], [ACTION_RIGHT, ACTION_LEFT], [ACTION_STOP, ACTION_STOP]):
            default_step = default_env.step(actions)
            disabled_step = disabled_env.step(actions)
            for left, right in zip(default_step[0], disabled_step[0]):
                np.testing.assert_array_equal(left[:12], right[:12])
                np.testing.assert_array_equal(left[12], 0)
                np.testing.assert_array_equal(right[12], 0)
            self.assertEqual(default_step[1:], disabled_step[1:])
            self.assertEqual(set(default_step[3]), LEGACY_INFO_KEYS)
            self.assertEqual(set(disabled_step[3]), LEGACY_INFO_KEYS)

    def test_due_closure_precedes_actions_and_clears_every_ring_layer(self):
        env = arena(
            sudden_death=True,
            sudden_death_start=1,
            sudden_death_interval=3,
            sudden_death_warning_steps=1,
        )
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 3, 3
        env.grid[1, 2] = WOODEN_WALL
        env.powerups[(1, 3)] = POWERUP_AMMO
        env.bombs.append(Bomb(1, 4, 1, 8, 2))
        env.agents[1].bombs_placed = 1
        env.flames[(2, 1)] = 2
        env._flame_owners[(2, 1)] = 0
        env._flame_sources[(2, 1)] = {0: 2}

        _, _, _, info = env.step([ACTION_RIGHT, ACTION_STOP])

        first_ring = ring_cells(7, 1)
        self.assertFalse(env.agents[0].alive)
        self.assertEqual((env.agents[0].row, env.agents[0].col), (3, 1), "Closure must happen before movement")
        self.assertTrue(all(env.grid[pos] == RIGID_WALL for pos in first_ring))
        self.assertFalse(any((bomb.row, bomb.col) in first_ring for bomb in env.bombs))
        self.assertEqual(env.agents[1].bombs_placed, 0)
        self.assertFalse(first_ring.intersection(env.flames))
        self.assertFalse(first_ring.intersection(env._flame_owners))
        self.assertFalse(first_ring.intersection(env._flame_sources))
        self.assertFalse(first_ring.intersection(env.powerups))
        self.assertEqual(metric(info, "sudden_death_deaths", "sd_deaths"), [0])

    def test_ring_death_has_no_bomb_credit_suicide_credit_or_terminal_bonus(self):
        env = arena(
            sudden_death=True,
            sudden_death_start=1,
            sudden_death_interval=3,
            sudden_death_warning_steps=1,
        )
        env.agents[0].row, env.agents[0].col = 3, 1
        env.agents[1].row, env.agents[1].col = 3, 3

        _, rewards, dones, info = env.step([ACTION_STOP, ACTION_STOP])

        self.assertEqual(metric(info, "sudden_death_deaths", "sd_deaths"), [0])
        self.assertEqual(info["kills"], [])
        self.assertEqual(info["self_blast_deaths"], [])
        self.assertEqual([agent.kills for agent in env.agents], [0, 0])
        self.assertEqual(rewards, [-1.0, -0.001])
        self.assertEqual(dones, [True, True])
        self.assertTrue(info["episode_done"])

    def test_interval_closes_exactly_one_next_inner_ring_when_due(self):
        env = arena(
            board_size=9,
            sudden_death=True,
            sudden_death_start=1,
            sudden_death_interval=2,
            sudden_death_warning_steps=0,
        )
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 5

        env.step([ACTION_STOP, ACTION_STOP])
        self.assertTrue(all(env.grid[pos] == RIGID_WALL for pos in ring_cells(9, 1)))
        second_before = env.grid.copy()
        env.step([ACTION_STOP, ACTION_STOP])
        np.testing.assert_array_equal(env.grid, second_before)
        env.step([ACTION_STOP, ACTION_STOP])
        self.assertTrue(all(env.grid[pos] == RIGID_WALL for pos in ring_cells(9, 2)))

    def test_warning_marks_only_next_ring_and_ramps_to_one(self):
        env = arena(
            board_size=9,
            sudden_death=True,
            sudden_death_start=3,
            sudden_death_interval=4,
            sudden_death_warning_steps=2,
        )
        initial = env.reset()
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 3, 5
        np.testing.assert_array_equal(initial[0][12], 0)

        after_one = env.step([ACTION_STOP, ACTION_STOP])[0][0]
        after_two = env.step([ACTION_STOP, ACTION_STOP])[0][0]
        after_close = env.step([ACTION_STOP, ACTION_STOP])[0][0]
        np.testing.assert_allclose(after_one[12], ring_plane(9, 1, 0.5))
        np.testing.assert_array_equal(after_two[12], ring_plane(9, 1, 1.0))
        np.testing.assert_array_equal(after_close[12], 0)

    def test_zero_warning_steps_keeps_channel_zero(self):
        env = arena(
            sudden_death=True,
            sudden_death_start=1,
            sudden_death_interval=2,
            sudden_death_warning_steps=0,
        )
        observations = env.reset()
        np.testing.assert_array_equal(observations[0][12], 0)
        env.agents[0].row, env.agents[0].col = 3, 3
        env.agents[1].row, env.agents[1].col = 2, 3
        observations = env.step([ACTION_STOP, ACTION_STOP])[0]
        np.testing.assert_array_equal(observations[0][12], 0)


class SuddenDeathSafetyAcceptance(unittest.TestCase):
    def test_mask_ignores_closure_beyond_current_survival_horizon(self):
        disabled = arena(sudden_death=False)
        enabled = arena(
            sudden_death=True,
            sudden_death_start=2,
            sudden_death_interval=3,
            sudden_death_warning_steps=5,
        )
        for env in (disabled, enabled):
            env.agents[0].row, env.agents[0].col = 3, 2
            env.agents[1].row, env.agents[1].col = 5, 5

        self.assertEqual(get_action_mask(enabled, 0), get_action_mask(disabled, 0))

    def test_mask_rejects_move_into_ring_that_closes_before_action(self):
        disabled = arena(sudden_death=False)
        enabled = arena(
            sudden_death=True,
            sudden_death_start=1,
            sudden_death_interval=3,
            sudden_death_warning_steps=1,
        )
        for env in (disabled, enabled):
            env.agents[0].row, env.agents[0].col = 3, 2
            env.agents[1].row, env.agents[1].col = 5, 5

        self.assertTrue(get_action_mask(disabled, 0)[ACTION_LEFT])
        self.assertFalse(get_action_mask(enabled, 0)[ACTION_LEFT])

    def test_mask_changes_only_in_warning_window_and_rejects_closing_escape(self):
        disabled = arena(sudden_death=False)
        enabled = arena(
            sudden_death=True,
            sudden_death_start=2,
            sudden_death_interval=3,
            sudden_death_warning_steps=1,
        )
        for env in (disabled, enabled):
            configure_bomb_escape_corridor(env)

        disabled_before = get_action_mask(disabled, 0)
        enabled_before = get_action_mask(enabled, 0)
        self.assertEqual(enabled_before, disabled_before, "Outside the warning window behavior must be unchanged")
        self.assertTrue(enabled_before[ACTION_BOMB], "The baseline escape uses the soon-closing ring")

        disabled.step([ACTION_STOP, ACTION_STOP])
        enabled.step([ACTION_STOP, ACTION_STOP])
        self.assertTrue(get_action_mask(disabled, 0)[ACTION_BOMB])
        self.assertFalse(get_action_mask(enabled, 0)[ACTION_BOMB])


class SuddenDeathCheckpointAcceptance(unittest.TestCase):
    def test_twelve_channel_checkpoint_migrates_exactly_at_zero_warning(self):
        torch.manual_seed(14)
        old = ActorCriticNet(in_channels=12, board_size=7).eval()
        new = ActorCriticNet(board_size=7).eval()
        self.assertEqual(new.conv[0].in_channels, 13)

        with patch("train_actor_critic.torch.load", return_value=old.state_dict()):
            smart_load_weights(new, "unused-12-channel.pt", "cpu")

        torch.testing.assert_close(new.conv[0].weight[:, :12], old.conv[0].weight)
        self.assertEqual(int(torch.count_nonzero(new.conv[0].weight[:, 12])), 0)
        observations = torch.randn(2, 12, 7, 7)
        expanded = torch.cat((observations, torch.zeros(2, 1, 7, 7)), dim=1)
        with torch.no_grad():
            old_logits, old_values = old(observations)
            new_logits, new_values = new(expanded)
        torch.testing.assert_close(new_logits, old_logits)
        torch.testing.assert_close(new_values, old_values)

    def test_incompatible_checkpoint_tensors_raise(self):
        new = ActorCriticNet(board_size=7)
        bad_inputs = {
            "eleven input channels": ActorCriticNet(in_channels=11, board_size=7).state_dict(),
            "wrong actor shape": {**new.state_dict(), "actor_head.weight": torch.zeros(7, 256)},
        }
        for label, state_dict in bad_inputs.items():
            with self.subTest(label=label):
                with patch("train_actor_critic.torch.load", return_value=state_dict):
                    with self.assertRaises((ValueError, RuntimeError)):
                        smart_load_weights(new, "malformed.pt", "cpu")


class SuddenDeathTrainingAcceptance(unittest.TestCase):
    def quiet_train(self, **kwargs):
        sd_options = tuple(name for name in kwargs if name.startswith("sudden_death"))
        require_parameters(train, *sd_options)
        with patch("train_actor_critic.torch.save"), \
             patch("train_actor_critic.os.makedirs"), \
             patch("builtins.print"):
            return train(
                episodes=kwargs.pop("episodes", 1),
                board_size=7,
                num_agents=2,
                max_steps=1,
                device="cpu",
                checkpoint_dir="unused-sudden-death-checkpoints",
                validation_every=0,
                run_report=False,
                **kwargs,
            )

    def test_trainer_public_options_and_probability_validation(self):
        parameters = inspect.signature(train).parameters
        required = {
            "sudden_death_probability",
            "sudden_death_start",
            "sudden_death_interval",
            "sudden_death_warning_steps",
        }
        self.assertTrue(required <= parameters.keys(), f"Missing train options: {required - parameters.keys()}")
        for probability in (-0.01, 1.01):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    self.quiet_train(sudden_death_probability=probability)

    def test_episode_enablement_is_seed_reproducible_and_honors_endpoints(self):
        def enabled_sequence(seed, probability):
            enabled = []

            def observe(env, episode, learner, actions, opponents):
                if env.step_count == 0:
                    enabled.append(bool(env.sudden_death))

            self.quiet_train(
                episodes=8,
                seed=seed,
                sudden_death_probability=probability,
                sudden_death_start=20,
                sudden_death_interval=2,
                sudden_death_warning_steps=1,
                on_step=observe,
            )
            return enabled

        first = enabled_sequence(91, 0.5)
        second = enabled_sequence(91, 0.5)
        self.assertEqual(first, second)
        self.assertIn(False, first)
        self.assertIn(True, first)
        self.assertEqual(enabled_sequence(91, 0.0), [False] * 8)
        self.assertEqual(enabled_sequence(91, 1.0), [True] * 8)

    def test_probability_endpoints_do_not_consume_curriculum_random_draws(self):
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                with patch("train_actor_critic.load_opponent_pool", return_value=[StopOpponent()]), \
                     patch("train_actor_critic.np.random.random", side_effect=[0.1] * 20) as random_draw:
                    self.quiet_train(
                        phase="D",
                        episodes=3,
                        seed=27,
                        sudden_death_probability=probability,
                        sudden_death_start=20,
                        sudden_death_interval=2,
                        sudden_death_warning_steps=1,
                    )
                self.assertEqual(random_draw.call_count, 3, "Only one curriculum draw per episode is allowed")

    def test_episode_metrics_separate_sd_active_passive_and_draw_outcomes(self):
        metrics = []

        def arrange(env, episode, learner, actions, opponents):
            if env.step_count != 0:
                return
            enemy = 1 - learner
            env.grid[3, 1] = EMPTY
            env.grid[3, 3] = EMPTY
            if episode == 1:
                env.agents[learner].row, env.agents[learner].col = 3, 3
                env.agents[enemy].row, env.agents[enemy].col = 3, 1
            else:
                env.agents[learner].row, env.agents[learner].col = 3, 1
                env.agents[enemy].row, env.agents[enemy].col = 3, 3

        self.quiet_train(
            episodes=2,
            seed=8,
            sudden_death_probability=1.0,
            sudden_death_start=1,
            sudden_death_interval=2,
            sudden_death_warning_steps=1,
            on_step=arrange,
            on_episode=metrics.append,
        )

        self.assertEqual(len(metrics), 2)
        self.assertEqual([metric(row, "sudden_death_enabled", "sd_enabled") for row in metrics], [1, 1])
        self.assertEqual([metric(row, "sudden_death_death", "sd_death") for row in metrics], [0, 1])
        self.assertEqual([metric(row, "sudden_death_win", "sd_win") for row in metrics], [1, 0])
        self.assertEqual([row["active_win"] for row in metrics], [0, 0])
        self.assertEqual([metric(row, "passive_win", "passive") for row in metrics], [0, 0])
        self.assertEqual([row["draw"] for row in metrics], [0, 0])

    def test_smoke_training_writes_finite_checkpoints(self):
        require_parameters(
            train,
            "sudden_death_probability",
            "sudden_death_start",
            "sudden_death_interval",
            "sudden_death_warning_steps",
        )
        with tempfile.TemporaryDirectory() as checkpoint_dir, patch("builtins.print"):
            net, rewards, wins = train(
                episodes=2,
                board_size=7,
                num_agents=2,
                max_steps=1,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                validation_every=0,
                run_report=False,
                seed=19,
                sudden_death_probability=1.0,
                sudden_death_start=1,
                sudden_death_interval=2,
                sudden_death_warning_steps=1,
            )
            self.assertTrue(all(torch.isfinite(parameter).all() for parameter in net.parameters()))
            self.assertTrue(all(math.isfinite(value) for value in rewards))
            self.assertTrue(all(value in (0, 1) for value in wins))
            for name in ("final_phaseC.pt", "best_phaseC.pt"):
                path = Path(checkpoint_dir, name)
                self.assertTrue(path.is_file(), f"Missing checkpoint {name}")
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                self.assertTrue(all(torch.isfinite(tensor).all() for tensor in state_dict.values()))

    def test_validation_candidates_use_enabled_schedule_e_from_episode_zero(self):
        validation_events = []
        validation_result = {
            "active_win_rate": 40.0,
            "suicide_rate": 1.0,
            "draw_rate": 20.0,
            "avg_kills": 0.4,
            "win_rate": 61.0,
            "games": [],
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir, \
             patch("evaluate.evaluate_seeded", return_value=validation_result) as evaluate_mock, \
             patch("builtins.print"):
            train(
                episodes=1,
                board_size=7,
                num_agents=2,
                max_steps=1,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
                validation_every=1,
                validation_episodes=100,
                validation_seed=31000,
                run_report=False,
                seed=31,
                sudden_death_probability=0.75,
                sudden_death_start=475,
                sudden_death_interval=25,
                sudden_death_warning_steps=40,
                on_validation=validation_events.append,
            )

        completed_episodes = [event["episode"] for event in validation_events if "result" in event]
        self.assertEqual(completed_episodes, [0, 1])
        self.assertEqual(evaluate_mock.call_count, 2)
        for call in evaluate_mock.call_args_list:
            positional, keywords = call
            self.assertEqual(positional[1:6], (7, 2, 100, 31000, 1))
            self.assertIs(keywords.get("sudden_death"), True)
            self.assertEqual(keywords.get("sudden_death_start"), 475)
            self.assertEqual(keywords.get("sudden_death_interval"), 25)
            self.assertEqual(keywords.get("sudden_death_warning_steps"), 40)


class PhaseCSD1500LauncherAcceptance(unittest.TestCase):
    launcher_name = "run_train_phaseC_sd_1500.ps1"

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

        checkpoint_path = root / "runs" / "pressure-C-next" / "final_phaseC.pt"
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
                metrics = json.loads(os.environ["FAKE_EVAL_METRICS"])
                output.write_text(json.dumps(metrics), encoding="utf-8")
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
            result = {
                "completed": completed,
                "calls": calls,
                "output": completed.stdout + completed.stderr,
                "output_directory": str(output_directory),
                "evaluation_exists": evaluation_path.is_file(),
                "sentinel": (output_directory / "do-not-overwrite.txt").read_text(encoding="utf-8")
                if (output_directory / "do-not-overwrite.txt").is_file()
                else None,
            }
            return result
        finally:
            temporary.cleanup()

    def assert_option(self, arguments, name, expected):
        self.assertEqual(arguments.count(name), 1, f"Expected exactly one {name}: {arguments}")
        index = arguments.index(name)
        self.assertLess(index + 1, len(arguments), f"Missing value for {name}")
        self.assertEqual(arguments[index + 1], str(expected))

    def assert_gate_status(self, output, metric, status):
        matching_lines = [line for line in output.splitlines() if metric.lower() in line.lower()]
        self.assertTrue(matching_lines, f"Missing gate output for {metric}:\n{output}")
        self.assertTrue(
            any(status.lower() in line.lower() for line in matching_lines),
            f"Expected {status} for {metric}: {matching_lines}",
        )

    def test_launcher_training_contract_is_exact(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        self.assertEqual([call["kind"] for call in result["calls"]], ["train_actor_critic", "evaluate"])
        arguments = result["calls"][0]["argv"]
        expected = {
            "--phase": "C",
            "--episodes": "1500",
            "--board_size": "9",
            "--num_agents": "2",
            "--max_steps": "500",
            "--lr": "0.0001",
            "--wall_destroy_bonus": "0.15",
            "--sudden_death_probability": "0.75",
            "--sudden_death_start": "475",
            "--sudden_death_interval": "25",
            "--sudden_death_warning_steps": "40",
        }
        for option, value in expected.items():
            self.assert_option(arguments, option, value)
        self.assert_option(
            arguments,
            "--resume_from",
            str(Path(result["completed"].args[-1]).parent / "runs" / "pressure-C-next" / "final_phaseC.pt"),
        )
        self.assert_option(arguments, "--checkpoint_dir", result["output_directory"])

    def test_launcher_validation_contract_is_exact(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        training_arguments = result["calls"][0]["argv"]
        self.assert_option(training_arguments, "--validation_every", 250)
        self.assert_option(training_arguments, "--validation_episodes", 100)

    def test_success_path_trains_then_evaluates_best_checkpoint(self):
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
        seed_index = arguments.index("--seed")
        int(arguments[seed_index + 1])
        self.assertTrue(result["evaluation_exists"])

    def test_preflight_failures_stop_before_training(self):
        cases = (
            {"interpreter": False},
            {"checkpoint": False},
            {"existing_output": True},
        )
        for options in cases:
            with self.subTest(**options):
                result = self.run_launcher(**options)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])
                self.assertEqual(result["calls"], [])
                if options.get("existing_output"):
                    self.assertEqual(result["sentinel"], "preserve")

    def test_training_failure_or_missing_best_stops_before_evaluation(self):
        cases = (
            {"train_exit": 17},
            {"create_best": False},
        )
        for options in cases:
            with self.subTest(**options):
                result = self.run_launcher(**options)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])
                self.assertEqual([call["kind"] for call in result["calls"]], ["train_actor_critic"])

    def test_evaluation_failure_is_process_failure(self):
        result = self.run_launcher(evaluation_exit=19)
        self.assertNotEqual(result["completed"].returncode, 0, result["output"])
        self.assertEqual([call["kind"] for call in result["calls"]], ["train_actor_critic", "evaluate"])

    def test_model_gates_report_boundaries_without_failing_process(self):
        passing = self.run_launcher()
        self.assertEqual(passing["completed"].returncode, 0, passing["output"])
        for metric in (
            "active_win_rate",
            "win_rate",
            "draw_rate",
            "suicide_rate",
            "sudden_death_death_rate",
        ):
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
        for metric in (
            "active_win_rate",
            "win_rate",
            "draw_rate",
            "suicide_rate",
            "sudden_death_death_rate",
        ):
            self.assert_gate_status(failing["output"], metric, "FAIL")


class SuddenDeathEvaluationAcceptance(unittest.TestCase):
    def assert_summary_rates(self, summary, *, sd_win, sd_death):
        self.assertEqual(summary["active_win_rate"], 0)
        self.assertEqual(
            metric(summary, "sudden_death_only_win_rate", "sudden_death_win_rate", "sd_only_win_rate", "sd_win_rate"),
            sd_win,
        )
        self.assertEqual(summary["passive_win_rate"], 0)
        self.assertEqual(summary["suicide_rate"], 0)
        self.assertEqual(metric(summary, "sudden_death_death_rate", "sd_death_rate"), sd_death)
        self.assertEqual(summary["draw_rate"], 0)

    def test_seeded_evaluator_accepts_sd_config_and_separates_all_rates(self):
        parameters = inspect.signature(evaluate_seeded).parameters
        required = {
            "sudden_death",
            "sudden_death_start",
            "sudden_death_interval",
            "sudden_death_warning_steps",
        }
        self.assertTrue(required <= parameters.keys(), f"Missing evaluator options: {required - parameters.keys()}")

        def evaluate(learner_survives):
            def policy(env, observations, learner):
                enemy = 1 - learner
                env.grid[3, 1] = EMPTY
                env.grid[3, 3] = EMPTY
                survivor, victim = (learner, enemy) if learner_survives else (enemy, learner)
                env.agents[survivor].row, env.agents[survivor].col = 3, 3
                env.agents[victim].row, env.agents[victim].col = 3, 1
                return ACTION_STOP

            return evaluate_seeded(
                policy,
                board_size=7,
                num_agents=2,
                num_episodes=2,
                seed=300,
                max_steps=2,
                sudden_death=True,
                sudden_death_start=1,
                sudden_death_interval=2,
                sudden_death_warning_steps=1,
            )

        wins = evaluate(True)
        deaths = evaluate(False)
        self.assert_summary_rates(wins, sd_win=100, sd_death=0)
        self.assert_summary_rates(deaths, sd_win=0, sd_death=100)
        for result in (wins, deaths):
            for corner in result["by_corner"].values():
                self.assertIn("passive_win_rate", corner)
                metric(corner, "sudden_death_only_win_rate", "sudden_death_win_rate", "sd_only_win_rate", "sd_win_rate")
                metric(corner, "sudden_death_death_rate", "sd_death_rate")
            for game in result["games"]:
                metric(game, "sudden_death_only_win", "sudden_death_win", "sd_only_win", "sd_win")
                metric(game, "sudden_death_death", "sd_death")
                self.assertIn("passive_win", game)

    def test_training_and_evaluation_clis_expose_sd_configuration(self):
        root = Path(__file__).resolve().parent
        commands = {
            "train_actor_critic.py": (
                "--sudden_death_probability",
                "--sudden_death_start",
                "--sudden_death_interval",
                "--sudden_death_warning_steps",
            ),
            "evaluate.py": (
                "--sudden_death",
                "--sudden_death_start",
                "--sudden_death_interval",
                "--sudden_death_warning_steps",
            ),
        }
        for script, flags in commands.items():
            with self.subTest(script=script):
                completed = subprocess.run(
                    [sys.executable, str(root / script), "--help"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                for flag in flags:
                    self.assertIn(flag, completed.stdout)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
