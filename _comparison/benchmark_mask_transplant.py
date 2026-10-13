"""Isolate replacing only the selected current checkpoint's safety mask."""
import json
import time
import compare_branches as c


def main():
    checkpoint = c.ROOT / "runs" / "pressure-C-next" / "final_phaseC.pt"
    fn = c.remote_policy(checkpoint, current_channels=True)
    started = time.perf_counter()
    def progress(done, total):
        if done % 20 == 0:
            print("current_checkpoint_Loc_update_mask", done, "/", total, flush=True)
    result = c.evaluate_seeded(fn, num_episodes=100, seed=18000, max_steps=500,
                               on_progress=progress)
    result["seconds"] = time.perf_counter() - started
    data = {
        "protocol": "Same current engine, RuleBot, 12-channel selected checkpoint, boards/corners/seeds; only mask replaced by Loc_update; no sudden death",
        "checkpoint_sha256": c.sha(checkpoint),
        "result": result,
    }
    path = c.ROOT / "_comparison" / "mask_transplant_18000.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("games", "by_corner")}), flush=True)


if __name__ == "__main__":
    main()
