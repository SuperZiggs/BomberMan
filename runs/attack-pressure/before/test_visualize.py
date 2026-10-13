"""One runnable integration check: python test_visualize.py (requires Tk and torch)."""
import json
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

import torch

from bomberman_env import BombermanFFAEnv, ACTION_BOMB
from train_actor_critic import get_action_mask, train
from visualize import App, ROOT, capture_state, load_agent, rolling_metrics


def check():
    torch.set_num_threads(1)
    model = ROOT / "checkpoints" / "best_phaseE_4a.pt"
    env = BombermanFFAEnv(num_agents=4, seed=7)
    obs = env.reset()
    policy = load_agent(model, 11)
    for aid in range(4):
        assert get_action_mask(env, aid)[policy.select_action(obs[aid], env, aid)]
    frame = capture_state(env, ["AC", "Bot", "Bot", "Bot"])
    env.grid[1, 1] = 2
    assert frame["grid"][1][1] == 0, "Frame must be independent of the worker's environment"
    small = BombermanFFAEnv(board_size=9, num_agents=2, seed=0)
    small_obs = small.reset()
    transferred = load_agent(model, 9)
    assert get_action_mask(small, 0)[transferred.select_action(small_obs[0], small, 0)]

    rows = [{"episode": i, "reward": float(i), "win": i % 2,
             "bombs": i, "loss": float(i)} for i in range(1, 4)]
    smoothed = rolling_metrics(rows, window=2)
    assert smoothed[0] == (1, 1.0, 100.0, 1.0, 1.0)
    assert smoothed[-1] == (3, 2.5, 50.0, 2.5, 2.5)

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        metrics, frames = [], []
        corners = []
        cfg = dict(phase="C", episodes=2, board_size=9, num_agents=2, max_steps=8,
                   checkpoint_dir=directory, device="cpu", run_report=False,
                   resume_from=str(ROOT / "checkpoints" / "best_phaseC.pt"))
        def observe_train(env, ep, aid, acts, opp):
            frames.append((ep, env.step_count))
            if env.step_count == 0:
                corners.append(aid)
        _, rewards, wins = train(**cfg, on_episode=metrics.append, on_step=observe_train)
        assert len(metrics) == len(rewards) == len(wins) == 2
        assert frames[0] == (1, 0) and frames[-1] == (2, 8)
        assert corners == [0, 1], "Phase C must rotate the learning corner"
        assert {"loss", "lr", "walls", "reward", "win", "bombs", "kills", "active_win", "suicide",
                "items", "learner_steps", "best_episode", "attack_opportunities", "attack_bombs",
                "attack_labels", "missing_attack_labels"} <= metrics[0].keys()
        assert all(0 <= m['attack_bombs'] <= m['attack_opportunities'] for m in metrics)
        assert all(m["active_win"] <= m["win"] and m["kills"] >= 0 for m in metrics)
        assert (Path(directory) / "final_phaseC.pt").is_file()
        stop = threading.Event()
        cancelled = []
        _, rewards, wins = train(**cfg, stop_event=stop, on_episode=cancelled.append,
                                  on_step=lambda *args: stop.set())
        assert not rewards and not wins and not cancelled
        load_agent(Path(directory) / "final_phaseC.pt", 9)

        learning_ids = []
        stop.clear()
        def observe_multi(env, ep, aid, actions, opponents):
            if env.step_count == 0:
                learning_ids.append(aid)
        cfg_multi = dict(cfg, phase="E", num_agents=3, snapshot_every=1)
        train(**cfg_multi, on_step=observe_multi)
        assert learning_ids == [0, 1], "Phase E must rotate the learning corner"
        assert len(list(Path(directory).glob("snapshot_phaseE_3a_*.pt"))) == 2

    root = tk.Tk()
    root.withdraw()
    app = App(root)
    try:
        root.update()
        assert len(app.tabs.tabs()) == 3
        pane = app.training
        assert tuple(map(str, pane.agent_box["values"])) == ("2", "3", "4")
        for phase in ("C", "D"):
            pane.phase.set(phase)
            assert pane.agent_count.get() == "2" and str(pane.agent_box["state"]) == "disabled"
        pane.phase.set("E")
        assert str(pane.agent_box["state"]) == "readonly"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            with patch("visualize.ROOT", Path(directory)), patch("visualize.threading.Thread.start") as dispatch:
                pane.episodes.set("1")
                pane.warm.set(False)
                for count in (2, 3, 4):
                    pane.agent_box.set(str(count))
                    pane.start()
                    config = json.loads((pane.run_dir / "config.json").read_text())
                    assert config["num_agents"] == count
                    assert config["board_size"] == (11 if count == 4 else 9)
                    assert config["lr"] == 3e-4
                assert dispatch.call_count == 3
                pane.phase.set("C")
                pane.warm.set(True)
                app.model.set(str(model))
                pane.start()
                config = json.loads((pane.run_dir / "config.json").read_text())
                assert config["lr"] == 1e-4 and config["resume_from"] == str(model)
        pane.thread = None
        app.watch.new_match()
        app.watch.step(schedule=False)
        assert app.watch.env.step_count == 1
        assert len(app.watch.board.table.get_children()) == 4
        assert app.watch.board.canvas.find_all(), "Board must draw actual tiles"
        board = app.watch.board
        assert {"brick", "crate", "robot_green", "robot_blue", "robot_pink", "robot_yellow",
                "bomb_green", "bomb_blue", "bomb_pink", "bomb_yellow", "chest", "gold",
                "explosion_0", "explosion_1", "explosion_2", "explosion_3", "smoke_0", "smoke_1"} <= board.sources.keys()
        images = {name: str(im) for name, im in board.sprites.items()}
        assert all(im.width() == board.sprite_size for im in board.sprites.values())
        assert any(board.canvas.type(item) == "image" for item in board.canvas.find_withtag("terrain"))
        rendered = {board.canvas.itemcget(item, "image") for item in board.canvas.find_withtag("dynamic")
                    if board.canvas.type(item) == "image"}
        assert {images[f"robot_{color}"] for color in ("green", "blue", "pink", "yellow")} <= rendered
        terrain = app.watch.board.canvas.find_withtag("terrain")
        app.watch.board.show(app.watch.board.state)
        assert images == {name: str(im) for name, im in board.sprites.items()}, "Sprites must stay cached between ticks"
        assert app.watch.board.canvas.find_withtag("terrain") == terrain, "Unchanged terrain must be reused"
        edited = capture_state(app.watch.env, app.watch.labels)
        edited["grid"][1][1] = 2
        app.watch.board.show(edited)
        assert app.watch.board.canvas.find_withtag("terrain") != terrain, "Terrain changes must redraw"
        app.watch.board.show(capture_state(app.watch.env, app.watch.labels))
        effect_env = BombermanFFAEnv(num_agents=4, bomb_timer=1, seed=7)
        effect_env.reset()
        board.show(capture_state(effect_env, ["A", "B", "C", "D"]))
        effect_env.step([ACTION_BOMB, 0, 0, 0])
        board.show(capture_state(effect_env, ["A", "B", "C", "D"]))
        assert {kind for _, kind in board.effects} == {"explosion", "smoke"}
        deadline = time.monotonic() + 1
        while board.animation_id is not None and time.monotonic() < deadline:
            root.update()
            time.sleep(.005)
        assert board.animation_id is None, "Effects must stop after their finite animation"
        for item, kind in board.effects:
            assert board.canvas.itemcget(item, "image") == str(board.sprites[f"{kind}_{3 if kind == 'explosion' else 1}"])
        board.show(capture_state(app.watch.env, app.watch.labels))
        app.play.new_match()
        assert app.play.speed.get() == 12
        app.play.running = True
        app.play.key_press(SimpleNamespace(keysym="d"))
        app.play.key_release(SimpleNamespace(keysym="d"))
        app.play.step()
        assert app.play.env.agents[0].col == 2, "A short tap released before the tick must still move A0"
        tick_id = app.play.after_id
        _, timer_type = root.tk.call("after", "info", tick_id)
        assert timer_type == "timer"
        app.play.key_press(SimpleNamespace(keysym="space"))
        app.play.key_release(SimpleNamespace(keysym="space"))
        app.play.key_press(SimpleNamespace(keysym="s"))
        app.play.step()
        assert app.play.board.state["actions"][0] == ACTION_BOMB
        assert tick_id not in root.tk.call("after", "info"), "Only one game tick may be scheduled"
        assert any(b.owner == 0 for b in app.play.env.bombs)
        app.play.step()
        assert app.play.board.state["actions"][0] == 2, "Held movement must resume after the queued bomb"
        app.play.speed.set(60)
        app.play.step()
        app.play.pause()
        assert app.play.after_id is None and not app.play.pressed
        # Timer expiry must produce a draw with multiple survivors.
        app.watch.env.max_steps = app.watch.env.step_count + 1
        app.watch.step(schedule=False)
        assert app.watch.env._episode_done and "Hòa" in app.watch.status.get()

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            pane = app.training
            pane.run_dir = Path(directory)
            pane.gate.clear()
            pane.progress.configure(maximum=2)
            cfg["checkpoint_dir"] = directory
            cfg_validation = dict(cfg, validation_every=1, validation_episodes=2)
            pane.thread = threading.Thread(target=pane.worker, args=(cfg_validation,))
            pane.thread.start()
            time.sleep(.15)
            assert pane.thread.is_alive() and pane.events.empty(), "Pause must block training steps"
            pane.gate.set()
            deadline = time.monotonic() + 30
            while pane.thread.is_alive() and time.monotonic() < deadline:
                root.update()
                time.sleep(.02)
            if pane.thread.is_alive():
                pane.stop()
                pane.thread.join(timeout=10)
                raise AssertionError("GUI worker did not finish")
            pane.poll()
            assert len(pane.metrics) == 2 and pane.board.state is not None
            logged = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            assert len(logged) == 2 and logged[-1]["episode"] == 2
            assert "Hoàn tất" in pane.status.get()
            assert "Best kiểm tra" in pane.status.get()
            validations = [json.loads(s) for s in (Path(directory)/"validation_metrics.jsonl").read_text().splitlines()]
            assert [v['episode'] for v in validations] == [0, 1, 2]
            pane.start_button.configure(state="disabled")
            pane.events.put(("validation", {"episode":250,"completed":1,"total":48}))
            pane.poll()
            assert "1/48" in pane.status.get() and str(pane.start_button['state']) == 'disabled'
    finally:
        app.watch.pause()
        app.play.pause()
        root.destroy()
    print("PASS: checkpoints, training, controls, cached PNG terrain/robots and finite explosion/smoke effects")


if __name__ == "__main__":
    check()
