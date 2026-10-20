"""Bounded, reproducible Phase D checkpoint-versus-snapshot diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import evaluate_pair_seeded, make_ac_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Phase D best checkpoint against its snapshot pool")
    parser.add_argument("--ac_model", required=True)
    parser.add_argument("--snapshot_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--board_size", type=int, default=9)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--games_per_snapshot", type=int, default=40)
    parser.add_argument("--max_snapshots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=34000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sudden_death", action="store_true")
    parser.add_argument("--sudden_death_start", type=int, default=475)
    parser.add_argument("--sudden_death_interval", type=int, default=25)
    parser.add_argument("--sudden_death_warning_steps", type=int, default=40)
    args = parser.parse_args()

    if args.games_per_snapshot < args.num_agents or args.games_per_snapshot % args.num_agents:
        raise ValueError("games_per_snapshot must be positive and divide evenly across corners")
    if args.max_snapshots < 1 or args.games_per_snapshot * args.max_snapshots > 200:
        raise ValueError("snapshot diagnostic must use 1..200 total games")

    snapshots = sorted(Path(args.snapshot_dir).glob("snapshot_ep*.pt"))[-args.max_snapshots:]
    if not snapshots:
        raise ValueError("snapshot diagnostic requires at least one snapshot_ep*.pt file")
    candidate = make_ac_fn(args.ac_model, args.board_size, args.device)
    matches = []
    for index, snapshot in enumerate(snapshots):
        opponent = make_ac_fn(str(snapshot), args.board_size, args.device)
        result = evaluate_pair_seeded(
            candidate, opponent, args.board_size, args.num_agents,
            args.games_per_snapshot, args.seed + index * args.games_per_snapshot // args.num_agents,
            args.max_steps, args.sudden_death, args.sudden_death_start,
            args.sudden_death_interval, args.sudden_death_warning_steps,
        )
        matches.append({"snapshot": snapshot.name, **result})

    total_games = sum(match["num_episodes"] for match in matches)
    expected_games = len(matches) * args.games_per_snapshot
    if total_games != expected_games:
        raise RuntimeError(f"snapshot diagnostic expected {expected_games} games, got {total_games}")

    output = {
        "seed": args.seed,
        "games_per_snapshot": args.games_per_snapshot,
        "max_snapshots": args.max_snapshots,
        "num_snapshots": len(matches),
        "total_games": total_games,
        "matches": matches,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "matches"}, indent=2))


if __name__ == "__main__":
    main()
