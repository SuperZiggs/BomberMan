"""Acceptance tests for resumable seeded evaluation artifacts and lifecycle."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluate


SCHEMA_VERSION = 1


def evaluation_config(*, seed=9100, num_episodes=4):
    return {
        "board_size": 7,
        "num_agents": 2,
        "num_episodes": num_episodes,
        "seed": seed,
        "max_steps": 12,
        "sudden_death": True,
        "sudden_death_start": 8,
        "sudden_death_interval": 2,
        "sudden_death_warning_steps": 3,
    }


def game_record(index, *, seed=9100, num_agents=2):
    board_seed = seed + index // num_agents
    aid = index % num_agents
    won = int((board_seed + aid) % 2 == 0)
    return {
        "seed": board_seed,
        "aid": aid,
        "win": won,
        "active_win": won,
        "sudden_death_win": 0,
        "passive_win": 0,
        "suicide": 0,
        "sudden_death_death": 0,
        "draw": 1 - won,
        "kills": float(won),
        "bombs": float(index + 1),
        "items": 0.0,
        "steps": float(10 + index),
    }


def result_payload(config, games):
    return {
        "seed": config["seed"],
        "board_size": config["board_size"],
        "num_agents": config["num_agents"],
        "max_steps": config["max_steps"],
        "sudden_death": config["sudden_death"],
        "sudden_death_start": config["sudden_death_start"],
        "sudden_death_interval": config["sudden_death_interval"],
        "sudden_death_warning_steps": config["sudden_death_warning_steps"],
        "num_episodes": len(games),
        "games": deepcopy(games),
    }


def partial_payload(checkpoint_identity, config, games, *, requested_count=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": deepcopy(checkpoint_identity),
        "config": deepcopy(config),
        "games": deepcopy(games),
        "completed_count": len(games),
        "requested_count": config["num_episodes"] if requested_count is None else requested_count,
    }


class ContractCase(unittest.TestCase):
    def require_callable(self, name):
        value = getattr(evaluate, name, None)
        self.assertTrue(callable(value), f"evaluate.{name} public contract is missing")
        return value

    def write_json(self, path, payload):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def independent_checkpoint_identity(self, checkpoint):
        stat = checkpoint.stat()
        return {
            "path": os.path.normcase(str(checkpoint.resolve())),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def expected_partial(self, checkpoint, config):
        return {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": self.independent_checkpoint_identity(checkpoint),
            "config": deepcopy(config),
            "requested_count": config["num_episodes"],
        }

    def require_seeded_resume_contract(self):
        parameters = inspect.signature(evaluate.evaluate_seeded).parameters
        self.assertIn("initial_games", parameters, "evaluate_seeded.initial_games public contract is missing")
        self.assertIn("on_checkpoint", parameters, "evaluate_seeded.on_checkpoint public contract is missing")

    def call_runner(self, runner, checkpoint, output, config, **overrides):
        arguments = {
            "agent_fn": object(),
            "checkpoint_path": str(checkpoint),
            "output_path": str(output),
            **config,
        }
        arguments.update(overrides)
        return runner(**arguments)


class PublicContractAcceptance(ContractCase):
    def test_public_contracts_and_defaults_are_exposed(self):
        for name in (
            "_checkpoint_identity",
            "_atomic_write_json",
            "_load_partial_games",
            "run_seeded_evaluation_to_file",
        ):
            self.require_callable(name)

        seeded_parameters = inspect.signature(evaluate.evaluate_seeded).parameters
        self.assertIn("initial_games", seeded_parameters)
        self.assertIn("on_checkpoint", seeded_parameters)

        runner = self.require_callable("run_seeded_evaluation_to_file")
        runner_parameters = inspect.signature(runner).parameters
        self.assertIn("partial_every", runner_parameters)
        self.assertEqual(runner_parameters["partial_every"].default, 10)

    def test_cli_help_exposes_resume_flags(self):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "evaluate.py"), "--help"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--resume_partial", completed.stdout)
        self.assertIn("--partial_every", completed.stdout)


class AtomicArtifactAcceptance(ContractCase):
    def test_checkpoint_identity_is_normalized_and_detects_same_path_replacement(self):
        identity_for = self.require_callable("_checkpoint_identity")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory, "model.pt")
            checkpoint.write_bytes(b"old!")
            first = identity_for(str(checkpoint.parent / "." / checkpoint.name))

            self.assertEqual(first["path"], os.path.normcase(str(checkpoint.resolve())))
            self.assertEqual(first["size"], checkpoint.stat().st_size)
            self.assertTrue(
                "mtime_ns" in first or "mtime" in first,
                f"Checkpoint identity must include mtime: {first}",
            )

            prior_mtime = checkpoint.stat().st_mtime_ns
            checkpoint.write_bytes(b"new!")
            os.utime(checkpoint, ns=(prior_mtime + 2_000_000_000, prior_mtime + 2_000_000_000))
            second = identity_for(str(checkpoint))
            self.assertNotEqual(first, second)

    def test_atomic_json_write_succeeds_and_preserves_old_target_on_write_failure(self):
        atomic_write = self.require_callable("_atomic_write_json")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "artifact.json")
            atomic_write(str(target), {"status": "complete", "count": 4})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"status": "complete", "count": 4})

            old_bytes = target.read_bytes()
            with self.assertRaises((TypeError, ValueError)):
                atomic_write(str(target), {"not_json": object()})
            self.assertEqual(target.read_bytes(), old_bytes)

    def test_atomic_json_write_preserves_old_target_on_replace_failure(self):
        atomic_write = self.require_callable("_atomic_write_json")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "artifact.json")
            target.write_text('{"old": true}', encoding="utf-8")
            old_bytes = target.read_bytes()

            with patch.object(evaluate.os, "replace", side_effect=OSError("replace denied")):
                with self.assertRaisesRegex(OSError, "replace denied"):
                    atomic_write(str(target), {"new": True})
            self.assertEqual(target.read_bytes(), old_bytes)


class PartialCompatibilityAcceptance(ContractCase):
    def make_fixture(self, directory):
        checkpoint = Path(directory, "model.pt")
        checkpoint.write_bytes(b"checkpoint")
        config = evaluation_config()
        games = [game_record(0), game_record(1)]
        expected = self.expected_partial(checkpoint, config)
        partial = Path(directory, "evaluation.partial.json")
        self.write_json(partial, partial_payload(expected["checkpoint"], config, games))
        return checkpoint, config, games, expected, partial

    def assert_rejected_without_mutation(self, partial, expected, pattern):
        loader = self.require_callable("_load_partial_games")
        original = partial.read_bytes()
        with self.assertRaisesRegex(ValueError, pattern):
            loader(str(partial), expected)
        self.assertEqual(partial.read_bytes(), original)

    def test_compatible_partial_returns_ordered_games(self):
        loader = self.require_callable("_load_partial_games")
        with tempfile.TemporaryDirectory() as directory:
            _, _, games, expected, partial = self.make_fixture(directory)
            self.assertEqual(loader(str(partial), expected), games)

    def test_schema_config_seed_and_requested_count_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, games, expected, partial = self.make_fixture(directory)
            cases = []

            wrong_schema = partial_payload(expected["checkpoint"], config, games)
            wrong_schema["schema_version"] = SCHEMA_VERSION + 1
            cases.append((wrong_schema, r"schema|version"))

            wrong_seed = partial_payload(expected["checkpoint"], config, games)
            wrong_seed["config"]["seed"] += 1
            cases.append((wrong_seed, r"config|seed"))

            wrong_config = partial_payload(expected["checkpoint"], config, games)
            wrong_config["config"]["max_steps"] += 1
            cases.append((wrong_config, r"config|max_steps"))

            wrong_requested = partial_payload(
                expected["checkpoint"], config, games, requested_count=config["num_episodes"] + 2
            )
            cases.append((wrong_requested, r"requested|count"))

            for payload, pattern in cases:
                with self.subTest(pattern=pattern):
                    self.write_json(partial, payload)
                    self.assert_rejected_without_mutation(partial, expected, pattern)

    def test_replaced_checkpoint_at_same_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, config, games, expected, partial = self.make_fixture(directory)
            previous_mtime = checkpoint.stat().st_mtime_ns
            checkpoint.write_bytes(b"replacemen")
            os.utime(checkpoint, ns=(previous_mtime + 2_000_000_000, previous_mtime + 2_000_000_000))
            current_expected = {
                **expected,
                "checkpoint": self.independent_checkpoint_identity(checkpoint),
            }
            self.assert_rejected_without_mutation(partial, current_expected, r"checkpoint")

    def test_duplicate_out_of_order_and_count_mismatch_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, games, expected, partial = self.make_fixture(directory)
            duplicate = partial_payload(expected["checkpoint"], config, [games[0], games[0]])
            out_of_order = partial_payload(expected["checkpoint"], config, [games[1], games[0]])
            wrong_count = partial_payload(expected["checkpoint"], config, games)
            wrong_count["completed_count"] = len(games) + 1

            for payload, pattern in (
                (duplicate, r"duplicate|unique"),
                (out_of_order, r"order"),
                (wrong_count, r"completed|count"),
            ):
                with self.subTest(pattern=pattern):
                    self.write_json(partial, payload)
                    self.assert_rejected_without_mutation(partial, expected, pattern)

    def test_incomplete_or_non_object_game_is_rejected_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, games, expected, partial = self.make_fixture(directory)
            missing_metric = deepcopy(games)
            del missing_metric[0]["kills"]
            for bad_games in (missing_metric, [None, games[1]]):
                with self.subTest(games=bad_games):
                    self.write_json(
                        partial,
                        partial_payload(expected["checkpoint"], config, bad_games),
                    )
                    self.assert_rejected_without_mutation(partial, expected, r"game|record")


class SeededResumeAcceptance(ContractCase):
    class FakeEnv:
        def __init__(self, **kwargs):
            self.num_agents = kwargs["num_agents"]
            self.seed = kwargs["seed"]

    def run_seeded_with_fake_match(self, *, initial_games=None, on_checkpoint=None):
        policy = object()
        calls = []

        def fake_rule(aid):
            return ("rule", aid)

        def fake_match(env, functions, num_episodes=1):
            aid = functions.index(policy)
            calls.append((env.seed, aid))
            game = game_record((env.seed - 9100) * 2 + aid)
            wins = [0, 0]
            active_wins = [0, 0]
            suicide_rates = [0.0, 0.0]
            sudden_death_death_rates = [0.0, 0.0]
            kills = [0.0, 0.0]
            bombs = [0.0, 0.0]
            items = [0.0, 0.0]
            wins[aid] = game["win"]
            active_wins[aid] = game["active_win"]
            kills[aid] = game["kills"]
            bombs[aid] = game["bombs"]
            return {
                "wins": wins,
                "active_wins": active_wins,
                "sudden_death_wins": [0, 0],
                "passive_wins": [0, 0],
                "suicide_rates": suicide_rates,
                "sudden_death_death_rates": sudden_death_death_rates,
                "draws": game["draw"],
                "avg_kills": kills,
                "avg_bombs": bombs,
                "avg_items": items,
                "avg_steps": game["steps"],
            }

        with patch.object(evaluate, "BombermanFFAEnv", self.FakeEnv), \
             patch.object(evaluate, "make_rule_based_fn", side_effect=fake_rule), \
             patch.object(evaluate, "run_match", side_effect=fake_match):
            result = evaluate.evaluate_seeded(
                policy,
                **evaluation_config(),
                initial_games=initial_games,
                on_checkpoint=on_checkpoint,
            )
        return result, calls

    def test_resume_matches_uninterrupted_payload_and_runs_only_missing_games(self):
        self.require_seeded_resume_contract()
        uninterrupted, all_calls = self.run_seeded_with_fake_match()
        initial = deepcopy(uninterrupted["games"][:2])
        callbacks = []
        resumed, resumed_calls = self.run_seeded_with_fake_match(
            initial_games=initial,
            on_checkpoint=lambda games: callbacks.append(deepcopy(games)),
        )

        self.assertEqual(all_calls, [(9100, 0), (9100, 1), (9101, 0), (9101, 1)])
        self.assertEqual(resumed_calls, [(9101, 0), (9101, 1)])
        self.assertEqual(resumed, uninterrupted)
        self.assertEqual(initial, uninterrupted["games"][:2])
        self.assertEqual([len(games) for games in callbacks], [3, 4])
        self.assertEqual(callbacks[-1], uninterrupted["games"])

    def test_duplicate_or_out_of_order_initial_games_fail_before_matches(self):
        self.require_seeded_resume_contract()
        first = game_record(0)
        second = game_record(1)
        cases = (
            ([first, first], r"duplicate|unique"),
            ([second, first], r"order"),
        )
        for initial_games, pattern in cases:
            with self.subTest(pattern=pattern), \
                 patch.object(evaluate, "run_match") as run_match_mock:
                with self.assertRaisesRegex(ValueError, pattern):
                    evaluate.evaluate_seeded(
                        object(),
                        **evaluation_config(),
                        initial_games=initial_games,
                    )
                run_match_mock.assert_not_called()


class OutputLifecycleAcceptance(ContractCase):
    def paths(self, directory):
        output = Path(directory, "evaluation-200.json")
        partial = Path(directory, "evaluation-200.partial.json")
        promote = Path(str(output) + ".tmp")
        return output, partial, promote

    def test_success_writes_schema_at_cadence_then_promotes_complete_final(self):
        runner = self.require_callable("run_seeded_evaluation_to_file")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory, "best.pt")
            checkpoint.write_bytes(b"best")
            output, partial, promote = self.paths(directory)
            config = evaluation_config()
            games = [game_record(index) for index in range(config["num_episodes"])]
            final_payload = result_payload(config, games)
            snapshots = []

            def fake_evaluate(*args, initial_games=None, on_checkpoint=None, **kwargs):
                cumulative = list(initial_games or [])
                for game in games[len(cumulative):]:
                    cumulative.append(deepcopy(game))
                    self.assertFalse(output.exists())
                    on_checkpoint(deepcopy(cumulative))
                    if partial.exists():
                        snapshots.append(json.loads(partial.read_text(encoding="utf-8")))
                return deepcopy(final_payload)

            replace_calls = []
            real_replace = os.replace

            def recording_replace(source, destination):
                replace_calls.append((Path(source), Path(destination)))
                return real_replace(source, destination)

            with patch.object(evaluate, "evaluate_seeded", side_effect=fake_evaluate), \
                 patch.object(evaluate.os, "replace", side_effect=recording_replace), \
                 patch("builtins.print") as print_mock:
                returned = self.call_runner(
                    runner,
                    checkpoint,
                    output,
                    config,
                    partial_every=2,
                    resume_partial=True,
                )

            self.assertEqual(returned, final_payload)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), final_payload)
            self.assertFalse(partial.exists())
            self.assertFalse(promote.exists())
            self.assertTrue(
                any(source == promote and destination == output for source, destination in replace_calls),
                f"Final payload was not promoted from {promote}: {replace_calls}",
            )

            cadence_snapshots = [item for item in snapshots if item["completed_count"] in (2, 4)]
            self.assertTrue(cadence_snapshots)
            snapshot = cadence_snapshots[0]
            self.assertEqual(snapshot["schema_version"], SCHEMA_VERSION)
            self.assertEqual(snapshot["checkpoint"], self.require_callable("_checkpoint_identity")(str(checkpoint)))
            self.assertEqual(snapshot["config"], config)
            self.assertEqual(snapshot["games"], games[:2])
            self.assertEqual(snapshot["completed_count"], 2)
            self.assertEqual(snapshot["requested_count"], 4)
            self.assertTrue(
                any(
                    "2" in " ".join(str(value) for value in call.args)
                    and "4" in " ".join(str(value) for value in call.args)
                    and call.kwargs.get("flush") is True
                    for call in print_mock.call_args_list
                ),
                f"Missing flushed cadence progress: {print_mock.call_args_list}",
            )

    def test_default_cadence_flushes_at_ten_not_before(self):
        runner = self.require_callable("run_seeded_evaluation_to_file")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory, "best.pt")
            checkpoint.write_bytes(b"best")
            output, partial, _ = self.paths(directory)
            config = evaluation_config(seed=9200, num_episodes=12)
            games = [game_record(index, seed=9200) for index in range(12)]
            final_payload = result_payload(config, games)
            persisted_counts = []

            def fake_evaluate(*args, initial_games=None, on_checkpoint=None, **kwargs):
                cumulative = list(initial_games or [])
                for game in games[len(cumulative):]:
                    cumulative.append(deepcopy(game))
                    on_checkpoint(deepcopy(cumulative))
                    if partial.exists():
                        persisted_counts.append(json.loads(partial.read_text(encoding="utf-8"))["completed_count"])
                return deepcopy(final_payload)

            with patch.object(evaluate, "evaluate_seeded", side_effect=fake_evaluate), \
                 patch("builtins.print") as print_mock:
                self.call_runner(runner, checkpoint, output, config, resume_partial=True)

            self.assertNotIn(1, persisted_counts)
            self.assertNotIn(9, persisted_counts)
            self.assertIn(10, persisted_counts)
            self.assertTrue(
                any(
                    "10" in " ".join(str(value) for value in call.args)
                    and "12" in " ".join(str(value) for value in call.args)
                    and call.kwargs.get("flush") is True
                    for call in print_mock.call_args_list
                ),
                f"Missing flushed default-cadence progress: {print_mock.call_args_list}",
            )

    def test_handled_failure_or_interrupt_flushes_all_completed_records(self):
        runner = self.require_callable("run_seeded_evaluation_to_file")
        for failure in (RuntimeError("evaluation failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory, "best.pt")
                checkpoint.write_bytes(b"best")
                output, partial, promote = self.paths(directory)
                config = evaluation_config()
                games = [game_record(index) for index in range(3)]

                def fake_evaluate(*args, initial_games=None, on_checkpoint=None, **kwargs):
                    cumulative = list(initial_games or [])
                    for game in games:
                        cumulative.append(deepcopy(game))
                        on_checkpoint(deepcopy(cumulative))
                    raise failure

                with patch.object(evaluate, "evaluate_seeded", side_effect=fake_evaluate):
                    with self.assertRaises(type(failure)):
                        self.call_runner(
                            runner,
                            checkpoint,
                            output,
                            config,
                            partial_every=10,
                            resume_partial=True,
                        )

                saved = json.loads(partial.read_text(encoding="utf-8"))
                self.assertEqual(saved["games"], games)
                self.assertEqual(saved["completed_count"], 3)
                self.assertFalse(output.exists())
                self.assertFalse(promote.exists())

    def test_incomplete_or_duplicate_final_payload_never_creates_final(self):
        runner = self.require_callable("run_seeded_evaluation_to_file")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory, "best.pt")
            checkpoint.write_bytes(b"best")
            config = evaluation_config()
            valid_games = [game_record(index) for index in range(4)]
            cases = (
                (valid_games[:3], r"complete|count|requested"),
                ([valid_games[0], valid_games[0], valid_games[2], valid_games[3]], r"duplicate|unique|order"),
            )
            for index, (games, pattern) in enumerate(cases):
                with self.subTest(index=index):
                    output = Path(directory, f"invalid-{index}.json")
                    partial = output.with_suffix(".partial.json")
                    promote = Path(str(output) + ".tmp")
                    with patch.object(
                        evaluate,
                        "evaluate_seeded",
                        return_value=result_payload(config, games),
                    ):
                        with self.assertRaisesRegex(ValueError, pattern):
                            self.call_runner(runner, checkpoint, output, config, resume_partial=True)
                    self.assertFalse(output.exists())
                    self.assertFalse(promote.exists())
                    if partial.exists():
                        self.assertLessEqual(
                            json.loads(partial.read_text(encoding="utf-8"))["completed_count"],
                            config["num_episodes"],
                        )

    def test_compatible_partial_is_passed_to_evaluator_and_final_matches_uninterrupted(self):
        runner = self.require_callable("run_seeded_evaluation_to_file")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory, "best.pt")
            checkpoint.write_bytes(b"best")
            output, partial, _ = self.paths(directory)
            config = evaluation_config()
            games = [game_record(index) for index in range(4)]
            expected_final = result_payload(config, games)
            identity = self.require_callable("_checkpoint_identity")(str(checkpoint))
            self.write_json(partial, partial_payload(identity, config, games[:2]))
            received_initial = []

            def fake_evaluate(*args, initial_games=None, on_checkpoint=None, **kwargs):
                received_initial.extend(deepcopy(initial_games or []))
                cumulative = list(initial_games or [])
                for game in games[len(cumulative):]:
                    cumulative.append(deepcopy(game))
                    on_checkpoint(deepcopy(cumulative))
                return deepcopy(expected_final)

            with patch.object(evaluate, "evaluate_seeded", side_effect=fake_evaluate):
                returned = self.call_runner(
                    runner,
                    checkpoint,
                    output,
                    config,
                    partial_every=2,
                    resume_partial=True,
                )

            self.assertEqual(received_initial, games[:2])
            self.assertEqual(returned, expected_final)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected_final)
            self.assertFalse(partial.exists())

    def test_mismatched_partial_is_rejected_without_running_or_mutating_it(self):
        runner = self.require_callable("run_seeded_evaluation_to_file")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory, "best.pt")
            checkpoint.write_bytes(b"best")
            output, partial, _ = self.paths(directory)
            config = evaluation_config()
            identity = self.require_callable("_checkpoint_identity")(str(checkpoint))
            incompatible = partial_payload(identity, config, [game_record(0)])
            incompatible["config"]["seed"] += 10
            self.write_json(partial, incompatible)
            original = partial.read_bytes()

            with patch.object(evaluate, "evaluate_seeded") as evaluate_mock:
                with self.assertRaisesRegex(ValueError, r"config|seed"):
                    self.call_runner(
                        runner,
                        checkpoint,
                        output,
                        config,
                        resume_partial=True,
                    )
            evaluate_mock.assert_not_called()
            self.assertEqual(partial.read_bytes(), original)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
