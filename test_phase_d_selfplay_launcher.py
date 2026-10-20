"""Independent acceptance tests for the Phase D 1v1 self-play launcher.

The staged launcher uses recording Python stubs.  It never starts real training
or evaluates a real checkpoint.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parent
LAUNCHER_NAME = "run_train_phaseD_sd_selfplay_v1.ps1"
OUTPUT_RELATIVE = Path("runs") / "phaseD-sd-selfplay-v1"
WARM_START_RELATIVE = Path("runs") / "phaseC-sd-best-fallback" / "best_phaseC.pt"


class PhaseDSelfPlayLauncherAcceptance(unittest.TestCase):
    """Black-box process-contract coverage for the new launcher."""

    def launcher_source(self):
        launcher = ROOT / LAUNCHER_NAME
        self.assertTrue(launcher.is_file(), f"Missing required launcher: {launcher}")
        return launcher, launcher.read_text(encoding="utf-8-sig")

    @staticmethod
    def valid_evaluation_payload(seed=33000):
        games = []
        for index in range(200):
            games.append(
                {
                    "seed": seed + index // 2,
                    "aid": index % 2,
                    "win": int(index < 140),
                    "active_win": int(index < 110),
                    "sudden_death_win": int(110 <= index < 130),
                    "passive_win": int(130 <= index < 140),
                    "suicide": int(index >= 197),
                    "sudden_death_death": int(index >= 180),
                    "draw": int(index >= 170),
                    "kills": float(index < 110),
                    "bombs": 1.0,
                    "items": 0.0,
                    "steps": 100,
                }
            )
        summary = {
            "num_episodes": 200,
            "win_rate": 70.0,
            "active_win_rate": 55.0,
            "draw_rate": 15.0,
            "sudden_death_death_rate": 10.0,
            "suicide_rate": 1.5,
        }
        return {
            "seed": seed,
            "board_size": 9,
            "num_agents": 2,
            "max_steps": 500,
            "sudden_death": True,
            "sudden_death_start": 475,
            "sudden_death_interval": 25,
            "sudden_death_warning_steps": 40,
            **summary,
            "by_corner": {"0": dict(summary), "1": dict(summary)},
            "games": games,
        }

    def stage_launcher(self, *, checkpoint=True, existing_output=False):
        launcher, source = self.launcher_source()
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        root = Path(temporary.name)
        shutil.copy2(launcher, root / launcher.name)

        scripts_dir = root / "venv" / "Scripts"
        scripts_dir.mkdir(parents=True)
        repository_venv = ROOT / "venv"
        shutil.copy2(repository_venv / "Scripts" / "python.exe", scripts_dir / "python.exe")
        shutil.copy2(repository_venv / "pyvenv.cfg", root / "venv" / "pyvenv.cfg")

        warm_start = root / WARM_START_RELATIVE
        if checkpoint:
            warm_start.parent.mkdir(parents=True)
            warm_start.write_bytes(b"fallback")

        output_directory = root / OUTPUT_RELATIVE
        if existing_output:
            output_directory.mkdir(parents=True)
            (output_directory / "do-not-overwrite.txt").write_text("preserve", encoding="utf-8")

        scripts = set(re.findall(r'["\']([^"\'\r\n]+\.py)["\']', source))
        self.assertIn("train_actor_critic.py", scripts)
        self.assertGreaterEqual(len(scripts), 2, "launcher needs training plus evaluation tooling")
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
                output = Path(option("--checkpoint_dir"))
                output.mkdir(parents=True, exist_ok=True)
                (output / "best_phaseD.pt").write_bytes(b"best")
                raise SystemExit(0)

            if "--output" in sys.argv:
                output = Path(option("--output"))
                output.parent.mkdir(parents=True, exist_ok=True)
                payload = json.loads(os.environ["FAKE_EVALUATION_PAYLOAD"])
                payload["diagnostic_script"] = kind
                output.write_text(json.dumps(payload), encoding="utf-8")
            raise SystemExit(0)
            """
        ).strip()
        for script in scripts:
            (root / script).write_text(fake_program, encoding="utf-8")
        return temporary, root, output_directory

    def run_launcher(self, *, evaluation_payload=None, **stage_options):
        temporary, root, output_directory = self.stage_launcher(**stage_options)
        try:
            call_log = root / "calls.jsonl"
            environment = os.environ.copy()
            environment.update(
                {
                    "ACCEPTANCE_CALL_LOG": str(call_log),
                    "FAKE_EVALUATION_PAYLOAD": json.dumps(
                        self.valid_evaluation_payload() if evaluation_payload is None else evaluation_payload
                    ),
                }
            )
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(root / LAUNCHER_NAME)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            calls = []
            if call_log.is_file():
                calls = [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
            artifacts = {
                str(path): json.loads(path.read_text(encoding="utf-8"))
                for path in output_directory.glob("*.json")
            } if output_directory.is_dir() else {}
            sentinel = output_directory / "do-not-overwrite.txt"
            return {
                "completed": completed,
                "output": completed.stdout + completed.stderr,
                "calls": calls,
                "root": root,
                "output_directory": output_directory,
                "artifacts": artifacts,
                "sentinel": sentinel.read_text(encoding="utf-8") if sentinel.is_file() else None,
            }
        finally:
            temporary.cleanup()

    def assert_option(self, arguments, name, expected):
        self.assertEqual(arguments.count(name), 1, f"Expected one {name}: {arguments}")
        index = arguments.index(name)
        self.assertLess(index + 1, len(arguments), f"Missing value for {name}")
        self.assertEqual(arguments[index + 1], str(expected))

    def assert_gate_status(self, output, label, status):
        patterns = {
            "WR": r"(?<![A-Za-z])WR(?![A-Za-z])|win[_ ]?rate",
            "ActiveWR": r"ActiveWR|active[_ ]?win[_ ]?rate",
            "Draw": r"Draw|draw[_ ]?rate",
            "Suicide": r"Suicide|suicide[_ ]?rate",
            "SD death": r"SD\s*deaths?|sudden[_ ]?death[_ ]?death",
        }
        matching = [
            line for line in output.splitlines()
            if re.search(patterns[label], line, flags=re.IGNORECASE)
        ]
        self.assertTrue(matching, f"Missing {label} gate: {output}")
        self.assertTrue(any(status in line.upper() for line in matching), matching)

    def test_training_contract_is_exact(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        self.assertGreaterEqual(len(result["calls"]), 4, result["calls"])
        train = result["calls"][0]
        self.assertEqual(train["kind"], "train_actor_critic")
        expected = {
            "--phase": "D",
            "--episodes": "2000",
            "--board_size": "9",
            "--num_agents": "2",
            "--max_steps": "500",
            "--lr": "0.000025",
            "--entropy_coef": "0.0075",
            "--snapshot_every": "200",
            "--seed": "2026101601",
            "--validation_every": "250",
            "--validation_episodes": "200",
            "--validation_seed": "32000",
            "--sudden_death_probability": "0.75",
            "--sudden_death_start": "475",
            "--sudden_death_interval": "25",
            "--sudden_death_warning_steps": "40",
        }
        for name, value in expected.items():
            with self.subTest(option=name):
                self.assert_option(train["argv"], name, value)
        self.assert_option(train["argv"], "--resume_from", result["root"] / WARM_START_RELATIVE)
        self.assert_option(train["argv"], "--checkpoint_dir", result["output_directory"])

    def test_two_rulebot_evaluations_are_ordered_and_share_the_frozen_environment(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        rulebot = [call for call in result["calls"] if call["kind"] == "evaluate"]
        self.assertEqual(len(rulebot), 2, result["calls"])
        expected = {
            "--mode": "seeded",
            "--board_size": "9",
            "--num_agents": "2",
            "--num_episodes": "200",
            "--max_steps": "500",
            "--seed": "33000",
            "--device": "cpu",
            "--sudden_death_start": "475",
            "--sudden_death_interval": "25",
            "--sudden_death_warning_steps": "40",
        }
        outputs = []
        for call in rulebot:
            for name, value in expected.items():
                with self.subTest(call=call, option=name):
                    self.assert_option(call["argv"], name, value)
            self.assertEqual(call["argv"].count("--sudden_death"), 1)
            output = Path(call["argv"][call["argv"].index("--output") + 1])
            self.assertIn(str(output), result["artifacts"])
            outputs.append(output)
        self.assertEqual(len(set(outputs)), 2)
        self.assert_option(rulebot[0]["argv"], "--ac_model", result["root"] / WARM_START_RELATIVE)
        self.assert_option(rulebot[1]["argv"], "--ac_model", result["output_directory"] / "best_phaseD.pt")

    def test_gates_are_reported_and_rulebot_artifacts_keep_corner_data(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        for label in ("WR", "ActiveWR", "Draw", "Suicide", "SD death"):
            self.assert_gate_status(result["output"], label, "PASS")

        for call in (call for call in result["calls"] if call["kind"] == "evaluate"):
            artifact = Path(call["argv"][call["argv"].index("--output") + 1])
            payload = result["artifacts"][str(artifact)]
            self.assertEqual(len(payload["games"]), 200)
            self.assertEqual(set(payload["by_corner"]), {"0", "1"})

        failing_payload = self.valid_evaluation_payload()
        failing_payload.update(
            {
                "win_rate": 69.5,
                "active_win_rate": 54.5,
                "draw_rate": 15.5,
                "sudden_death_death_rate": 10.5,
                "suicide_rate": 1.51,
            }
        )
        failing = self.run_launcher(evaluation_payload=failing_payload)
        self.assertEqual(failing["completed"].returncode, 0, failing["output"])
        for label in ("WR", "ActiveWR", "Draw", "Suicide", "SD death"):
            self.assert_gate_status(failing["output"], label, "FAIL")

    def test_snapshot_pool_diagnostic_is_distinct_deterministic_and_bounded(self):
        result = self.run_launcher()
        self.assertEqual(result["completed"].returncode, 0, result["output"])
        diagnostics = [call for call in result["calls"] if call["kind"] not in {"train_actor_critic", "evaluate"}]
        self.assertEqual(len(diagnostics), 1, result["calls"])
        arguments = diagnostics[0]["argv"]
        self.assertIn("--output", arguments)
        self.assertIn("--seed", arguments, "diagnostic must pin its random seed")
        bounded_names = ("--num_episodes", "--games_per_snapshot", "--episodes_per_opponent")
        bounded = next((name for name in bounded_names if name in arguments), None)
        self.assertIsNotNone(bounded, f"diagnostic needs an explicit bounded game count: {arguments}")
        count = int(arguments[arguments.index(bounded) + 1])
        self.assertGreater(count, 0)
        self.assertLessEqual(count, 200)
        artifact = Path(arguments[arguments.index("--output") + 1])
        self.assertIn(str(artifact), result["artifacts"])
        self.assertTrue(str(artifact).startswith(str(result["output_directory"])))

    def test_fail_fast_and_never_overwrites_existing_output(self):
        missing = self.run_launcher(checkpoint=False)
        self.assertNotEqual(missing["completed"].returncode, 0, missing["output"])
        self.assertEqual(missing["calls"], [])

        existing = self.run_launcher(existing_output=True)
        self.assertNotEqual(existing["completed"].returncode, 0, existing["output"])
        self.assertEqual(existing["calls"], [])
        self.assertEqual(existing["sentinel"], "preserve")
        _, source = self.launcher_source()
        self.assertNotRegex(source, r"(?i)\bRemove-Item\b")
        self.assertNotRegex(source, r"(?i)\bClear-Content\b")


if __name__ == "__main__":
    unittest.main()
