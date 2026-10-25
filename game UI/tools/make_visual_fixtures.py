"""Build labeled render fixtures from the real engine, never quality evaluation."""
from pathlib import Path
import json
import sys

UI = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(UI / ".python-deps"), str(UI), str(UI.parent)]
from bomberman_env import BombermanFFAEnv, Bomb
from bridge.snapshots import capture_state


def frame(env):
    return dict(capture_state(env, ["AI · Actor-Critic", "RuleBot", "RuleBot", "RuleBot"]),
                paused=True, running=False, mode="watch", speed=8, session_id="fixture", match_id="fixture")


def main():
    normal = BombermanFFAEnv(seed=912770, num_agents=4)
    normal.reset()
    bomb = BombermanFFAEnv(seed=912771, num_agents=4)
    bomb.reset()
    bomb.bombs.append(Bomb(1, 1, 0, 2, 3))
    bomb.agents[0].bombs_placed = 1
    bomb.powerups[(1, 2)] = "ammo"
    bomb.grid[1, 2] = 0
    blast = BombermanFFAEnv(seed=912772, num_agents=4, bomb_timer=1)
    blast.reset()
    blast.grid[3, 1:8] = 0
    blast.grid[1:8, 3] = 0
    blast.bombs.extend([Bomb(3, 3, 0, 1, 4), Bomb(3, 6, 1, 3, 2)])
    blast.agents[0].bombs_placed = 1
    blast.agents[1].bombs_placed = 1
    blast.step([0] * 4)
    ring = BombermanFFAEnv(seed=912773, num_agents=4, sudden_death=True, sudden_death_start=3,
                          sudden_death_interval=20, sudden_death_warning_steps=4)
    ring.reset()
    ring.step([0] * 4)
    ring.step([0] * 4)
    ending = BombermanFFAEnv(seed=912774, num_agents=4, max_steps=1)
    ending.reset()
    ending.step([0] * 4)
    states = {name: frame(env) for name, env in [("normal", normal), ("bomb-under-owner", bomb),
              ("chain-explosion", blast), ("ring-warning", ring), ("draw-result", ending)]}
    destination = UI / "tests/fixtures/scenes.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
    print("5 real-engine render fixtures; not agent-quality evidence.")


if __name__ == "__main__":
    main()
