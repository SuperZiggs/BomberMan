"""Focused behavior tests for the bounded Phase D snapshot diagnostic."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluate
import evaluate_phase_d_snapshot_pool as snapshot_pool


class PhaseDSnapshotEvaluationTests(unittest.TestCase):
    def test_pair_evaluation_reports_candidate_metrics_at_both_corners(self):
        self.assertTrue(
            hasattr(evaluate, "evaluate_pair_seeded"),
            "snapshot evaluation must support a seeded candidate-vs-opponent pairing",
        )
        result = evaluate.evaluate_pair_seeded(
            lambda env, obs, aid: 0,
            lambda env, obs, aid: 0,
            board_size=9,
            num_episodes=4,
            seed=44000,
            max_steps=2,
            sudden_death=True,
            sudden_death_start=1,
            sudden_death_interval=1,
            sudden_death_warning_steps=1,
        )

        self.assertEqual(result["num_episodes"], 4)
        self.assertEqual([game["aid"] for game in result["games"]], [0, 1, 0, 1])
        self.assertEqual([game["seed"] for game in result["games"]], [44000, 44000, 44001, 44001])
        self.assertEqual(set(result["by_corner"]), {"0", "1"})
        self.assertTrue(result["sudden_death"])
        for metric in ("win_rate", "active_win_rate", "draw_rate", "suicide_rate",
                       "sudden_death_death_rate", "avg_kills", "avg_bombs", "avg_steps"):
            self.assertIn(metric, result)

    def test_snapshot_diagnostic_rejects_an_empty_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(sys, "argv", [
                "evaluate_phase_d_snapshot_pool.py", "--ac_model", "candidate.pt",
                "--snapshot_dir", str(root), "--output", str(root / "diagnostic.json"),
            ]), patch.object(snapshot_pool, "make_ac_fn") as load_policy:
                with self.assertRaisesRegex(ValueError, "at least one snapshot"):
                    snapshot_pool.main()
                load_policy.assert_not_called()

    def test_snapshot_diagnostic_uses_newest_five_and_records_total_games(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for episode in (0, 200, 400, 600, 800, 1000):
                (root / f"snapshot_ep{episode:06d}.pt").touch()
            output = root / "diagnostic.json"
            fake_result = {"num_episodes": 40, "games": [{"aid": 0}, {"aid": 1}]}
            with patch.object(sys, "argv", [
                "evaluate_phase_d_snapshot_pool.py", "--ac_model", "candidate.pt",
                "--snapshot_dir", str(root), "--output", str(output),
                "--games_per_snapshot", "40", "--max_snapshots", "5", "--seed", "34000",
            ]), patch.object(snapshot_pool, "make_ac_fn", return_value=lambda *_: 0), \
                    patch.object(snapshot_pool, "evaluate_pair_seeded", return_value=fake_result) as evaluate_pair:
                snapshot_pool.main()

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([match["snapshot"] for match in payload["matches"]], [
                "snapshot_ep000200.pt", "snapshot_ep000400.pt", "snapshot_ep000600.pt",
                "snapshot_ep000800.pt", "snapshot_ep001000.pt",
            ])
            self.assertEqual(payload["total_games"], 200)
            self.assertEqual(evaluate_pair.call_count, 5)


if __name__ == "__main__":
    unittest.main()
