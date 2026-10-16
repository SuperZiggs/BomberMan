"""Independent acceptance tests for the Phase C-SD temporal 250 launcher.

Run with: venv\Scripts\python.exe -m unittest -v test_phase_c_sd_temporal_250.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest


class PhaseCSDTemporal250LauncherAcceptance(unittest.TestCase):
    launcher_name = "run_train_phaseC_sd_temporal_250.ps1"
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
                    "seed": 33000 + index // 2,
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
                    "steps": 100.0,
                }
            )
        payload = {
            "seed": 33000,
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
        warm_start=True,
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

        warm_start_path = root / "runs" / "phaseC-sd-target-750" / "best_phaseC.pt"
        if warm_start:
            warm_start_path.parent.mkdir(parents=True)
            warm_start_path.write_bytes(b"warm-start")

        output_directory = root / "runs" / "phaseC-sd-temporal-250"
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
        warm_start=True,
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
            warm_start=warm_start,
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
            sentinel_path = output_directory / "do-not-overwrite.txt"
            return {
                "completed": completed,
                "calls": calls,
                "output": completed.stdout + completed.stderr,
                "root": root,
                "output_directory": output_directory,
                "evaluation_exists": evaluation_path.is_file(),
                "sentinel": (
                    sentinel_path.read_text(encoding="utf-8")
                    if sentinel_path.is_file()
                    else None
                ),
            }
        finally:
            temporary.cleanup()

    def assert_option(self, arguments, name, expected):
        self.assertEqual(arguments.count(name), 1, f"Expected exactly one {name}: {arguments}")
        index = arguments.index(name)
        self.assertLess(index + 1, len(arguments), f"Missing value for {name}")
        self.assertEqual(arguments[index + 1], str(expected))

    def assert_recovery_options(self, arguments):
        self.assertEqual(arguments.count("--resume_partial"), 1, arguments)
        self.assert_option(arguments, "--partial_every", 10)

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

    def test_launcher_declares_evaluate_only_switch(self):
        source = self.launcher_source().read_text(encoding="utf-8-sig")
        parameter_block = re.search(r"(?is)\A\s*param\s*\((.*?)\)", source)
        self.assertIsNotNone(parameter_block, "Missing launcher parameter block")
        self.assertRegex(parameter_block.group(1), r"(?i)\[switch\]\s*\$EvaluateOnly\b")
        self.assertRegex(parameter_block.group(1), r"(?i)\[string\]\s*\$OutputDirectory\b")

    def test_training_probability_is_evaluator_aligned_without_launching(self):
        source = self.launcher_source().read_text(encoding="utf-8-sig")
        invocation = re.search(
            r'(?ms)^\s*& \$Python \(Join-Path \$PSScriptRoot "train_actor_critic.py"\) `(.*?)^\s*--checkpoint_dir \$OutputDirectory',
            source,
        )
        self.assertIsNotNone(invocation, "Missing normal training invocation")
        arguments = invocation.group(1).replace("`", "").split()
        self.assert_option(arguments, "--sudden_death_probability", "1.0")

    def test_training_invocation_matches_frozen_temporal_contract(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        self.assertEqual(
            [call["kind"] for call in result["calls"]],
            ["train_actor_critic", "evaluate"],
        )
        arguments = result["calls"][0]["argv"]
        expected = {
            "--phase": "C",
            "--episodes": "250",
            "--board_size": "9",
            "--num_agents": "2",
            "--max_steps": "500",
            "--lr": "0.00001",
            "--entropy_coef": "0.005",
            "--temporal_trap_horizon": "8",
            "--temporal_trap_max_bombs": "2",
            "--interception_horizon": "4",
            "--sequence_pressure_gain": "0.35",
            "--ring_safety_coef": "0.6",
            "--seed": "2026101504",
            "--validation_every": "125",
            "--validation_episodes": "200",
            "--validation_seed": "32000",
            "--sudden_death_probability": "1.0",
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
            result["root"] / "runs" / "phaseC-sd-target-750" / "best_phaseC.pt",
        )
        self.assert_option(arguments, "--checkpoint_dir", result["output_directory"])

    def test_final_evaluation_uses_best_checkpoint_and_exact_recovery_contract(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        arguments = result["calls"][1]["argv"]
        expected = {
            "--mode": "seeded",
            "--ac_model": result["output_directory"] / "best_phaseC.pt",
            "--board_size": "9",
            "--num_agents": "2",
            "--num_episodes": "200",
            "--max_steps": "500",
            "--seed": "33000",
            "--device": "cpu",
            "--sudden_death_start": "475",
            "--sudden_death_interval": "25",
            "--sudden_death_warning_steps": "40",
            "--output": result["output_directory"] / "evaluation-200.json",
        }
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assert_option(arguments, option, value)
        self.assertEqual(arguments.count("--sudden_death"), 1)
        self.assert_recovery_options(arguments)
        self.assertTrue(result["evaluation_exists"])

    def test_evaluate_only_skips_training_and_requires_existing_output_best(self):
        success = self.run_launcher(
            warm_start=False,
            existing_output=True,
            output_best=True,
            evaluate_only=True,
        )
        self.assertEqual(success["completed"].returncode, 0, success["output"])
        self.assertEqual([call["kind"] for call in success["calls"]], ["evaluate"])
        self.assert_option(
            success["calls"][0]["argv"],
            "--ac_model",
            success["output_directory"] / "best_phaseC.pt",
        )
        self.assert_recovery_options(success["calls"][0]["argv"])

        cases = (
            ({"warm_start": False, "evaluate_only": True}, r"output|directory|phaseC-sd-temporal-250"),
            (
                {
                    "warm_start": False,
                    "existing_output": True,
                    "evaluate_only": True,
                },
                r"best_phaseC\.pt|best checkpoint",
            ),
        )
        for options, pattern in cases:
            with self.subTest(options=options):
                result = self.run_launcher(**options)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])
                self.assertEqual(result["calls"], [])
                self.assertRegex(result["output"], pattern)

    def test_stage_messages_and_window_warning_precede_subprocesses(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        train_marker = "FAKE_SUBPROCESS_START:train_actor_critic"
        eval_marker = "FAKE_SUBPROCESS_START:evaluate"
        train_index = result["output"].find(train_marker)
        eval_index = result["output"].find(eval_marker)
        self.assertGreaterEqual(train_index, 0, result["output"])
        self.assertGreater(eval_index, train_index, result["output"])
        before_train = result["output"][:train_index].lower()
        before_eval = result["output"][:eval_index].lower()
        self.assertRegex(before_train, r"stage|giai doan|giai đoạn")
        self.assertRegex(before_train, r"train|huan luyen|huấn luyện")
        self.assertRegex(before_eval, r"stage|giai doan|giai đoạn")
        self.assertRegex(before_eval, r"evaluat|danh gia|đánh giá")
        self.assertRegex(before_eval, r"do not close|khong dong|không đóng")
        self.assertIn("evaluation-200.json", before_eval)

    def test_normal_mode_preflight_and_subprocess_failures_are_operational(self):
        for options in ({"warm_start": False}, {"existing_output": True}):
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

        evaluation_failure = self.run_launcher(evaluation_exit=19)
        self.assertNotEqual(
            evaluation_failure["completed"].returncode,
            0,
            evaluation_failure["output"],
        )
        self.assertEqual(
            [call["kind"] for call in evaluation_failure["calls"]],
            ["train_actor_critic", "evaluate"],
        )

    def test_final_metadata_mismatches_are_operational_failures(self):
        mismatches = {
            "seed": 33001,
            "board_size": 11,
            "num_agents": 4,
            "max_steps": 499,
            "sudden_death": False,
            "sudden_death_start": 474,
            "sudden_death_interval": 24,
            "sudden_death_warning_steps": 39,
            "num_episodes": 199,
        }
        for field, value in mismatches.items():
            with self.subTest(field=field):
                result = self.run_launcher(
                    evaluation_payload=self.valid_evaluation_payload(**{field: value})
                )
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])

    def test_incomplete_out_of_order_missing_or_metricless_final_is_operational(self):
        incomplete = self.valid_evaluation_payload()
        incomplete["games"] = incomplete["games"][:-1]
        out_of_order = self.valid_evaluation_payload()
        out_of_order["games"][0], out_of_order["games"][1] = (
            out_of_order["games"][1],
            out_of_order["games"][0],
        )
        cases = [
            {"evaluation_payload": incomplete},
            {"evaluation_payload": out_of_order},
            {"evaluation_artifact": "missing"},
        ]
        for field in (
            "win_rate",
            "active_win_rate",
            "draw_rate",
            "sudden_death_death_rate",
            "suicide_rate",
        ):
            payload = self.valid_evaluation_payload()
            del payload[field]
            cases.append({"evaluation_payload": payload})

        for options in cases:
            with self.subTest(options=tuple(options)):
                result = self.run_launcher(**options)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])

    def test_every_game_requires_the_seeded_outcome_schema(self):
        seed_and_aid_only = self.valid_evaluation_payload()
        seed_and_aid_only["games"] = [
            {"seed": game["seed"], "aid": game["aid"]}
            for game in seed_and_aid_only["games"]
        ]
        result = self.run_launcher(evaluation_payload=seed_and_aid_only)
        self.assertNotEqual(result["completed"].returncode, 0, result["output"])

        for field in self.outcome_fields:
            with self.subTest(missing_field=field):
                payload = self.valid_evaluation_payload()
                del payload["games"][0][field]
                result = self.run_launcher(evaluation_payload=payload)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])

        invalid_binary = self.valid_evaluation_payload()
        invalid_binary["games"][0]["win"] = 2
        result = self.run_launcher(evaluation_payload=invalid_binary)
        self.assertNotEqual(result["completed"].returncode, 0, result["output"])

    def test_top_level_gate_metrics_must_match_recomputed_game_outcomes(self):
        for field in (
            "win_rate",
            "active_win_rate",
            "draw_rate",
            "sudden_death_death_rate",
            "suicide_rate",
        ):
            with self.subTest(field=field):
                payload = self.valid_evaluation_payload()
                payload[field] = 99.0
                result = self.run_launcher(evaluation_payload=payload)
                self.assertNotEqual(result["completed"].returncode, 0, result["output"])

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
    unittest.main()
