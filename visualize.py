"""Desktop views for live training, spectating and human FFA. Run: python visualize.py."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from math import gcd
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import traceback

import numpy as np
import torch

from bomberman_env import BombermanFFAEnv, ACTION_STOP, ACTION_BOMB
from rule_based_bot import RuleBasedAgent, _build_danger_map
from train_actor_critic import ActorCriticNet, SnapshotOpponent, smart_load_weights, train

ROOT = Path(__file__).resolve().parent
BG, PANEL, TEXT, MUTED = "#101827", "#172338", "#edf3ff", "#a4b6d0"
COLORS = ("#65d6b7", "#7db3ff", "#ed9eee", "#ffd071")
SPRITE_COLORS = ("green", "blue", "pink", "yellow")
ACTION_NAMES = ("Đứng", "Lên", "Xuống", "Trái", "Phải", "Bom")
KEY_ACTIONS = {"w": 1, "Up": 1, "s": 2, "Down": 2,
               "a": 3, "Left": 3, "d": 4, "Right": 4}


def capture_state(env, labels, actions=None, rewards=None):
    """Copy a frame; Tk never reads a training environment while it is changing."""
    return {"grid": env.grid.tolist(), "agents": [asdict(a) for a in env.agents],
            "bombs": [asdict(b) for b in env.bombs], "flames": list(env.flames),
            "powerups": list(env.powerups.items()), "danger": _build_danger_map(env).tolist(),
            "step": env.step_count, "max_steps": env.max_steps,
            "done": env._episode_done, "labels": list(labels),
            "actions": list(actions or [ACTION_STOP] * env.num_agents),
            "rewards": list(rewards) if rewards is not None else None}


def load_agent(path, board_size):
    if not Path(path).is_file():
        raise ValueError("Chọn checkpoint .pt có sẵn trước khi bắt đầu.")
    net = ActorCriticNet(board_size=board_size)
    smart_load_weights(net, str(path), "cpu")
    return SnapshotOpponent(net, "cpu")


class BoardView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.state = None
        self.layout = None
        self.sources = {path.stem: tk.PhotoImage(master=self, file=str(path))
                        for path in (ROOT / "assets").glob("*.png")}
        self.sprites, self.small_bombs = {}, {}
        self.sprite_size = None
        self.animation_id = None
        self.effects = []
        self.just_died = []
        self.danger = tk.BooleanVar(value=False)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.caption = ttk.Label(header, text="Chưa có ván đấu")
        self.caption.pack(side="left")
        ttk.Checkbutton(header, text="Vùng nguy hiểm", variable=self.danger,
                        command=self.draw).pack(side="right")
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, width=520, height=460,
                                takefocus=True)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _: self.draw())
        ttk.Label(self, text="Gạch: không phá · Thùng: phá được · Rương +A · Vàng +R · Bom: timer",
                  style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=6)
        self.table = ttk.Treeview(self, columns=("state", "action", "ammo", "range", "kills", "reward"),
                                  height=4, selectmode="none")
        for col, label, width in (("#0", "Tác tử", 180), ("state", "Trạng thái", 90),
                                  ("action", "Hành động", 80), ("ammo", "Bom / sức chứa", 100),
                                  ("range", "Tầm nổ", 65), ("kills", "Hạ gục", 65),
                                  ("reward", "Reward", 75)):
            self.table.heading(col, text=label)
            self.table.column(col, width=width, minwidth=55, stretch=True)
        self.table.grid(row=3, column=0, sticky="ew")
        for i, color in enumerate(COLORS):
            self.table.tag_configure(str(i), foreground=color)

    def show(self, state):
        self.just_died = []
        if self.state is not None and state["step"] > self.state["step"]:
            previous_alive = {a["agent_id"] for a in self.state["agents"] if a["alive"]}
            self.just_died = [a for a in state["agents"] if not a["alive"] and a["agent_id"] in previous_alive]
        self.state = state
        self.caption.configure(text=f"Bước {state['step']} / {state['max_steps']}")
        for row in self.table.get_children():
            if int(row) >= len(state["agents"]):
                self.table.delete(row)
        for a in state["agents"]:
            i = a["agent_id"]
            reward = f"{state['rewards'][i]:+.2f}" if state["rewards"] is not None else "—"
            if not self.table.exists(str(i)):
                self.table.insert("", "end", iid=str(i))
            self.table.item(str(i), text=f"A{i} · {state['labels'][i]}", tags=(str(i),),
                            values=("Còn sống" if a["alive"] else "Đã bị loại",
                                    ACTION_NAMES[state["actions"][i]],
                                    f"{a['bombs_placed']} / {a['ammo']}",
                                    a["blast_range"], a["kills"], reward))
        self.draw()

    def draw(self):
        c, state = self.canvas, self.state
        if state is None:
            return
        if self.animation_id is not None:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        self.effects.clear()
        size = len(state["grid"])
        cell = max(1, (min(c.winfo_width(), c.winfo_height()) - 12) // size)
        ox = (c.winfo_width() - size * cell) / 2
        oy = (c.winfo_height() - size * cell) / 2
        font = ("Segoe UI", max(8, int(cell * .2)), "bold")

        pixels = max(1, round(cell))
        if pixels != self.sprite_size:
            def resized(source, target):
                factor = gcd(source.width(), target)
                return source.zoom(target // factor).subsample(source.width() // factor)

            # PNGs are square pixel art: native nearest-neighbor scaling, cached per cell size.
            self.sprites = {name: resized(source, pixels) for name, source in self.sources.items()}
            self.small_bombs = {name: resized(source, max(1, pixels // 2))
                                for name, source in self.sources.items() if name.startswith("bomb_")}
            self.sprite_size = pixels

        def sprite(name, r, col, offset=.5, small=False):
            images = self.small_bombs if small else self.sprites
            return c.create_image(ox + (col + offset) * cell, oy + (r + offset) * cell,
                                  image=images[name], tags="dynamic")

        def box(r, col, inset=0):
            return (ox + (col + inset) * cell, oy + (r + inset) * cell,
                    ox + (col + 1 - inset) * cell, oy + (r + 1 - inset) * cell)

        layout = (c.winfo_width(), c.winfo_height(), state["grid"])
        if layout != self.layout:
            c.delete("all")
            self.layout = layout
            # Composite static tiles once: one opaque canvas image is cheaper to repaint.
            self.terrain_image = tk.PhotoImage(master=c, width=size * pixels, height=size * pixels)
            self.terrain_image.put("#1b3040", to=(0, 0, size * pixels, size * pixels))
            for r, row in enumerate(state["grid"]):
                for col, tile in enumerate(row):
                    if tile:
                        image = self.sprites["brick" if tile == 1 else "crate"]
                        self.terrain_image.tk.call(str(self.terrain_image), "copy", str(image),
                                                   "-to", col * pixels, r * pixels)
            c.create_image(ox, oy, image=self.terrain_image, anchor="nw", tags="terrain")
        c.delete("dynamic")
        if self.danger.get():
            for r, row in enumerate(state["grid"]):
                for col, tile in enumerate(row):
                    if tile == 0 and state["danger"][r][col]:
                        c.create_rectangle(*box(r, col, .07), fill="#a38a39", stipple="gray50",
                                           outline="", tags="dynamic")
        for (r, col), kind in state["powerups"]:
            sprite("chest" if kind == "ammo" else "gold", r, col)
            c.create_text(ox + (col + .5) * cell, oy + (r + .85) * cell,
                          text="+A" if kind == "ammo" else "+R", fill=TEXT, font=font, tags="dynamic")
        for r, col in state["flames"]:
            c.create_rectangle(*box(r, col, .04), fill="#ff983f", stipple="gray50",
                               outline="", tags="dynamic")
            self.effects.append((sprite("explosion_0", r, col), "explosion"))
        for a in self.just_died:
            self.effects.append((sprite("smoke_0", a["row"], a["col"]), "smoke"))
        for a in state["agents"]:
            if a["alive"]:
                r, col, i = a["row"], a["col"], a["agent_id"]
                sprite(f"robot_{SPRITE_COLORS[i]}", r, col)
                c.create_rectangle(ox + col * cell, oy + r * cell,
                                   ox + (col + .5) * cell, oy + (r + .34) * cell,
                                   fill=BG, outline="", tags="dynamic")
                c.create_text(ox + (col + .04) * cell, oy + (r + .01) * cell,
                              text=f"A{i}", anchor="nw", fill=COLORS[i], font=font, tags="dynamic")
        # Draw timers last so a bomb under its owner stays visible.
        occupied = {(a["row"], a["col"]) for a in state["agents"] if a["alive"]}
        for b in state["bombs"]:
            small = (b["row"], b["col"]) in occupied
            offset = .75 if small else .5
            sprite(f"bomb_{SPRITE_COLORS[b['owner']]}", b["row"], b["col"], offset=offset, small=small)
            c.create_text(ox + (b["col"] + offset) * cell, oy + (b["row"] + offset + .05) * cell,
                          text=str(b["timer"]), fill=BG, font=font, tags="dynamic")
        if self.effects:
            self.animation_id = self.after(25, self.animate_effects, 1)

    def animate_effects(self, frame):
        self.animation_id = None
        for item, kind in self.effects:
            index = frame if kind == "explosion" else min(1, frame // 2)
            self.canvas.itemconfigure(item, image=self.sprites[f"{kind}_{index}"])
        if frame < 3:
            self.animation_id = self.after(25, self.animate_effects, frame + 1)


class GamePane(ttk.Frame):
    def __init__(self, parent, app, human=False):
        super().__init__(parent, padding=14)
        self.app, self.human = app, human
        self.env = self.policy = None
        self.running = False
        self.pending = ACTION_STOP
        self.pressed = []
        self.after_id = None
        self.agents = tk.StringVar(value="4")
        self.speed = tk.IntVar(value=12 if human else 8)
        self.status = tk.StringVar(value="Chọn checkpoint phía trên, rồi tạo ván mới.")
        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Số tác tử").pack(side="left")
        ttk.Combobox(controls, textvariable=self.agents, values=(3, 4) if human else (2, 3, 4),
                     width=3, state="readonly").pack(side="left", padx=8)
        ttk.Button(controls, text="Ván mới", command=self.new_match).pack(side="left", padx=4)
        self.run_button = ttk.Button(controls, text="Chạy", command=self.toggle)
        self.run_button.pack(side="left", padx=4)
        ttk.Button(controls, text="1 bước", command=lambda: self.step(schedule=False)).pack(side="left", padx=4)
        ttk.Label(controls, text="Bước / giây").pack(side="left", padx=(16, 4))
        ttk.Spinbox(controls, from_=1, to=60, width=4, textvariable=self.speed).pack(side="left")
        ttk.Label(self, textvariable=self.status, wraplength=1000).pack(fill="x", pady=(0, 8))
        self.board = BoardView(self)
        self.board.pack(fill="both", expand=True)
        self.board.canvas.bind("<Button-1>", lambda _: self.board.canvas.focus_set())
        self.board.canvas.bind("<KeyPress>", self.key_press)
        self.board.canvas.bind("<KeyRelease>", self.key_release)
        self.board.canvas.bind("<FocusOut>", lambda _: self.clear_keys())
        if human:
            pad = ttk.Frame(self)
            pad.pack(fill="x", pady=(8, 0))
            ttk.Label(pad, text="Bạn là A0 · Click bàn cờ · WASD / mũi tên · Space / E đặt bom").pack(side="left")
            for text, action in (("↑", 1), ("↓", 2), ("←", 3), ("→", 4), ("Đặt bom", 5)):
                ttk.Button(pad, text=text, width=8 if action == 5 else 3,
                           command=lambda a=action: self.queue_action(a)).pack(side="left", padx=2)

    def clear_keys(self):
        self.pressed.clear()
        self.pending = ACTION_STOP

    def queue_action(self, action):
        # Keep a short tap until the next tick; movement must not overwrite a queued bomb.
        if self.running and (self.pending != ACTION_BOMB or action == ACTION_BOMB):
            self.pending = action
        self.board.canvas.focus_set()

    def key_press(self, event):
        key = event.keysym.lower() if len(event.keysym) == 1 else event.keysym
        if self.human and self.running and key not in self.pressed:
            self.pressed.append(key)
            if key in ("space", "e"):
                self.queue_action(ACTION_BOMB)
            elif key in KEY_ACTIONS:
                self.queue_action(KEY_ACTIONS[key])
        return "break" if key in (*KEY_ACTIONS, "space", "e") else None

    def key_release(self, event):
        key = event.keysym.lower() if len(event.keysym) == 1 else event.keysym
        if key in self.pressed:
            self.pressed.remove(key)

    def pause(self):
        self.running = False
        self.run_button.configure(text="Chạy")
        self.clear_keys()
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    def new_match(self):
        self.pause()
        try:
            count = int(self.agents.get())
            if count not in ((3, 4) if self.human else (2, 3, 4)):
                raise ValueError("Số tác tử không hợp lệ.")
            self.policy = load_agent(self.app.model.get(), 11)
            self.env = BombermanFFAEnv(board_size=11, num_agents=count, max_steps=500)
            self.obs = self.env.reset()
            self.rewards = [0.0] * count
            self.trained_id = 1 if self.human else 0
            self.bots = {i: RuleBasedAgent(i) for i in range(count)
                         if i != self.trained_id and not (self.human and i == 0)}
            self.labels = ["Bạn" if self.human and i == 0 else
                           "Agent train" if i == self.trained_id else "RuleBot" for i in range(count)]
            self.board.show(capture_state(self.env, self.labels, rewards=self.rewards))
            name = Path(self.app.model.get()).name
            self.status.set(f"{name} · Agent A{self.trained_id} đấu {len(self.bots)} bot"
                            + (" và bạn. Bấm Chạy để chơi." if self.human else ". Bấm Chạy để xem."))
        except Exception as exc:
            self.env = None
            messagebox.showerror("Không tạo được ván", str(exc), parent=self.app.root)

    def toggle(self):
        if self.running:
            self.pause()
        else:
            if self.env is None or self.env._episode_done:
                self.new_match()
            if self.env is not None:
                self.running = True
                self.run_button.configure(text="Tạm dừng")
                self.board.canvas.focus_set()
                self.step()

    def step(self, schedule=True):
        if schedule and not self.running:
            return
        if self.env is None or self.env._episode_done:
            return
        if not schedule:
            self.pause()
        elif self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        tick_started = time.perf_counter()
        try:
            fps = int(self.speed.get())
            if not 1 <= fps <= 60:
                raise ValueError("Tốc độ phải từ 1 đến 60 bước / giây.")
            actions = [ACTION_STOP] * self.env.num_agents
            for i, a in enumerate(self.env.agents):
                if not a.alive:
                    continue
                if self.human and i == 0:
                    actions[i] = self.pending or next(
                        (KEY_ACTIONS[k] for k in reversed(self.pressed) if k in KEY_ACTIONS), ACTION_STOP)
                    self.pending = ACTION_STOP
                elif i == self.trained_id:
                    actions[i] = self.policy.select_action(self.obs[i], self.env, i)
                else:
                    actions[i] = self.bots[i].select_action(self.env)
            self.obs, rewards, _, info = self.env.step(actions)
            self.rewards = [r + delta for r, delta in zip(self.rewards, rewards)]
            self.board.show(capture_state(self.env, self.labels, actions, self.rewards))
            if info["episode_done"]:
                survivors = [i for i, alive in enumerate(info["alive"]) if alive]
                result = f"A{survivors[0]} · {self.labels[survivors[0]]} thắng" if len(survivors) == 1 else "Hòa"
                self.status.set(f"Kết thúc sau {info['step']} bước · {result}. Bấm Ván mới để chơi tiếp.")
                self.pause()
            elif schedule and self.running:
                elapsed_ms = (time.perf_counter() - tick_started) * 1000
                self.after_id = self.after(max(1, round(1000 / fps - elapsed_ms)), self.step)
        except Exception as exc:
            self.pause()
            messagebox.showerror("Lỗi ván đấu", str(exc), parent=self.app.root)


def rolling_metrics(rows, window=50):
    """Trailing means include early episodes instead of padding them with fake data."""
    result = []
    recent = deque()
    totals = np.zeros(4)
    for row in rows:
        values = np.array([row["reward"], row["win"] * 100, row["bombs"], row["loss"]])
        recent.append(values)
        totals += values
        if len(recent) > window:
            totals -= recent.popleft()
        result.append((row["episode"], *(totals / len(recent))))
    return result


class TrainPane(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=14)
        self.app = app
        self.thread = None
        self.stop_event, self.gate = threading.Event(), threading.Event()
        self.events, self.frames = queue.SimpleQueue(), deque(maxlen=1)
        self.metrics = []
        self.phase = tk.StringVar(value="E")
        self.episodes = tk.StringVar(value="100")
        self.agent_count = tk.StringVar(value="4")
        self.warm = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Train trực tiếp · Mỗi lần chạy lưu checkpoint và metrics vào runs/.")
        self.delay = 0.0
        self.run_dir = None
        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Label(controls, text="Phase").pack(side="left")
        phase = ttk.Combobox(controls, textvariable=self.phase, values=("C", "D", "E"), width=3, state="readonly")
        phase.pack(side="left", padx=6)
        ttk.Label(controls, text="Episodes").pack(side="left")
        ttk.Entry(controls, textvariable=self.episodes, width=7).pack(side="left", padx=6)
        ttk.Label(controls, text="Tác tử").pack(side="left")
        self.agent_box = ttk.Combobox(controls, textvariable=self.agent_count,
                                     values=(2, 3, 4), width=3, state="readonly")
        self.agent_box.pack(side="left", padx=6)
        self.phase.trace_add("write", self.sync_agent_count)
        self.sync_agent_count()
        ttk.Checkbutton(controls, text="Warm-start từ checkpoint đã chọn", variable=self.warm).pack(side="left", padx=8)
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=8)
        self.start_button = ttk.Button(buttons, text="Bắt đầu train", command=self.start)
        self.start_button.pack(side="left")
        self.pause_button = ttk.Button(buttons, text="Tạm dừng", command=self.toggle, state="disabled")
        self.pause_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(buttons, text="Dừng & lưu", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Label(buttons, text="Chậm để quan sát").pack(side="left", padx=(18, 5))
        ttk.Scale(buttons, from_=0, to=250, length=150,
                  command=lambda v: setattr(self, "delay", float(v) / 1000)).pack(side="left")
        ttk.Label(buttons, text="0–250 ms / bước", style="Muted.TLabel").pack(side="left", padx=5)
        ttk.Label(self, textvariable=self.status, wraplength=1040).pack(fill="x", pady=(0, 6))
        self.progress = ttk.Progressbar(self)
        self.progress.pack(fill="x", pady=(0, 8))
        self.split = ttk.Panedwindow(self, orient="horizontal")
        self.split.pack(fill="both", expand=True)
        self.board = BoardView(self.split)
        self.split.add(self.board, weight=3)
        charts = ttk.Frame(self.split, padding=(14, 0, 0, 0))
        self.split.add(charts, weight=2)
        ttk.Label(charts, text="Learning curves · Trung bình 50 episode gần nhất").pack(fill="x")
        self.chart = tk.Canvas(charts, bg=BG, highlightthickness=0, width=400)
        self.chart.pack(fill="both", expand=True)
        self.chart.bind("<Configure>", lambda _: self.draw_charts())
        self.after(100, self.poll)

    def sync_agent_count(self, *_):
        if self.phase.get() in ("C", "D"):
            self.agent_count.set("2")
            self.agent_box.configure(state="disabled")
        else:
            self.agent_box.configure(state="readonly")

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        try:
            episodes, phase = int(self.episodes.get()), self.phase.get()
            count = int(self.agent_count.get()) if phase == "E" else 2
            if not 1 <= episodes <= 100000 or phase not in ("C", "D", "E") or (phase == "E" and count not in (2, 3, 4)):
                raise ValueError("Episodes: 1–100000; Phase C/D: 2 tác tử; Phase E: 2, 3 hoặc 4.")
            resume = self.app.model.get() if self.warm.get() else ""
            if self.warm.get() and not Path(resume).is_file():
                raise ValueError("Chọn checkpoint cho warm-start hoặc bỏ chọn warm-start.")
            stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d-%H%M%S-%f")
            self.run_dir = ROOT / "runs" / stamp
            self.run_dir.mkdir(parents=True)
            config = dict(phase=phase, episodes=episodes, num_agents=count,
                          board_size=11 if count == 4 else 9, max_steps=500,
                          device="cpu", checkpoint_dir=str(self.run_dir), resume_from=resume,
                          lr=1e-4 if resume else 3e-4,
                          log_every=50, run_report=False,
                          seed=int(time.time_ns() % (2**32)), validation_every=250,
                          validation_episodes=48, validation_seed=6000)
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Không bắt đầu được", str(exc), parent=self.app.root)
            return
        self.metrics.clear()
        self.frames.clear()
        self.stop_event.clear()
        self.gate.set()
        self.progress.configure(maximum=episodes, value=0)
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal", text="Tạm dừng")
        self.stop_button.configure(state="normal")
        self.status.set(f"Phase {phase} · {count} tác tử · Đang khởi tạo · {self.run_dir}")
        self.draw_charts()
        self.thread = threading.Thread(target=self.worker, args=(config,), daemon=False)
        self.thread.start()

    def worker(self, config):
        last_frame = 0.0

        def on_step(env, ep, aid, actions, opponents):
            nonlocal last_frame
            while not self.gate.wait(.1):
                if self.stop_event.is_set():
                    return
            if self.stop_event.is_set():
                return
            now = time.monotonic()
            if now - last_frame >= .08 or env.step_count == 0 or env._episode_done:
                labels = ["Đang học" if i == aid else "RuleBot" if opponents[i] == "rule"
                          else "Snapshot" for i in range(env.num_agents)]
                self.frames.append((ep, aid, capture_state(env, labels, actions)))
                last_frame = now
            if self.delay and env.step_count:
                self.stop_event.wait(self.delay)

        try:
            with (Path(config["checkpoint_dir"]) / "metrics.jsonl").open("w", encoding="utf-8") as log:
                def on_episode(data):
                    log.write(json.dumps(data) + "\n")
                    log.flush()
                    self.events.put(("metric", data))

                def on_validation(data):
                    while not self.gate.wait(.1):
                        if self.stop_event.is_set():
                            return
                    if not self.stop_event.is_set():
                        self.events.put(("validation", data))

                train(**config, on_step=on_step, on_episode=on_episode,
                      on_validation=on_validation, stop_event=self.stop_event)
            self.events.put(("done", self.stop_event.is_set()))
        except Exception as exc:
            traceback.print_exc()
            self.events.put(("error", str(exc)))

    def toggle(self):
        if self.gate.is_set():
            self.gate.clear()
            self.pause_button.configure(text="Tiếp tục")
            self.status.set("Đã tạm dừng train. Bấm Tiếp tục hoặc Dừng & lưu.")
        else:
            self.gate.set()
            self.pause_button.configure(text="Tạm dừng")
            self.status.set("Đang tiếp tục train…")

    def stop(self):
        self.stop_event.set()
        self.gate.set()
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status.set("Đang dừng và lưu các cập nhật đã hoàn thành…")

    def poll(self):
        if self.frames:
            ep, aid, frame = self.frames.popleft()
            self.board.show(frame)
            self.board.caption.configure(text=f"Episode {ep} · A{aid} đang học · Bước {frame['step']}")
        changed = False
        while not self.events.empty():
            kind, data = self.events.get()
            if kind == "metric":
                self.metrics.append(data)
                self.progress.configure(value=data["episode"])
                recent = self.metrics[-50:]
                wr = sum(m["win"] for m in recent) / len(recent) * 100
                active_wr = sum(m["active_win"] for m in recent) / len(recent) * 100
                kills = sum(m["kills"] for m in recent) / len(recent)
                suicide = sum(m["suicide"] for m in recent) / len(recent) * 100
                items = sum(m["items"] for m in recent) / len(recent)
                reward = sum(m["reward"] for m in recent) / len(recent)
                if self.gate.is_set() and not self.stop_event.is_set():
                    self.status.set(f"Episode {data['episode']} · WR {wr:.1f}% · WR có hạ gục {active_wr:.1f}%"
                                    f" · Hạ gục {kills:.2f}/ván · Reward {reward:+.2f}"
                                    f" · Tự nổ {suicide:.1f}% · Item {items:.2f}/ván"
                                    f" · Loss {data['loss']:.3f} · LR {data['lr']:.2g}")
                changed = True
            elif kind == "validation":
                if self.gate.is_set() and not self.stop_event.is_set():
                    result = data.get("result")
                    if result is None:
                        self.status.set(f"Đánh giá checkpoint · Episode {data['episode']}"
                                        f" · {data['completed']}/{data['total']} ván · Luân phiên góc")
                    else:
                        self.status.set(f"Đánh giá · WR có hạ gục {result['active_win_rate']:.1f}%"
                                        f" · Tự nổ {result['suicide_rate']:.1f}%"
                                        f" · Hòa {result['draw_rate']:.1f}% · Best ở ep {data['best_episode']}")
            else:
                self.start_button.configure(state="normal")
                self.pause_button.configure(state="disabled")
                self.stop_button.configure(state="disabled")
                self.app.refresh_models()
                if kind == "error":
                    self.status.set(f"Train lỗi: {data} · Log đã ghi: {self.run_dir}")
                    messagebox.showerror("Train lỗi", data, parent=self.app.root)
                else:
                    # UI configuration may have changed during training.
                    model = next(self.run_dir.glob("best_*.pt"), None)
                    if model is None:
                        model = next(self.run_dir.glob("final_*.pt"), None)
                    if model is not None:
                        self.app.model.set(str(model))
                    recent = self.metrics[-200:]
                    count = max(len(recent), 1)
                    active_wr = sum(m["active_win"] for m in recent) / count * 100
                    suicide = sum(m["suicide"] for m in recent) / count * 100
                    status = (f"{'Đã dừng' if data else 'Hoàn tất'} · {len(self.metrics)} episode"
                              f" · Train: WR có hạ gục {active_wr:.1f}% · Tự nổ {suicide:.1f}%")
                    selected = self.run_dir / "best_validation.json"
                    if selected.is_file():
                        validation = json.loads(selected.read_text(encoding="utf-8"))
                        status += (f" · Best kiểm tra: WR có hạ gục {validation['active_win_rate']:.1f}%"
                                   f" · Tự nổ {validation['suicide_rate']:.1f}%")
                    self.status.set(status + f" · Checkpoint và metrics: {self.run_dir}")
        if changed:
            self.draw_charts()
        self.after(100, self.poll)

    def draw_charts(self):
        c = self.chart
        c.delete("all")
        width, height = c.winfo_width(), c.winfo_height()
        if width < 100 or height < 100:
            return
        curves = rolling_metrics(self.metrics)
        # ponytail: O(episodes) redraw; downsample long runs if the GUI starts lagging.
        for index, title in enumerate(("Reward", "Win rate (%)", "Bom / episode", "Loss")):
            top, panel_height = index * height / 4, height / 4
            c.create_text(12, top + 12, text=title, anchor="w", fill=TEXT, font=("Segoe UI", 10, "bold"))
            x0, x1, y0, y1 = 62, width - 24, top + 34, top + panel_height - 25
            if y1 <= y0:
                continue
            c.create_line(x0, y0, x0, y1, x1, y1, fill="#40506a")
            if not curves:
                c.create_text((x0+x1)/2, (y0+y1)/2, text="Chưa có dữ liệu train", fill=MUTED)
                continue
            values = [row[index + 1] for row in curves]
            low, high = min(values), max(values)
            if index == 1:
                low, high = 0, 100
            else:
                pad = max((high-low) * .1, .1)
                low, high = low-pad, high+pad
            for value, y in ((high, y0), (low, y1)):
                c.create_text(x0-7, y, text=f"{value:.1f}", anchor="e", fill=MUTED)
            c.create_text(x0, y1+12, text="1", anchor="w", fill=MUTED)
            c.create_text(x1, y1+12, text=f"Episode {curves[-1][0]}", anchor="e", fill=MUTED)
            points = []
            for ep, *means in curves:
                points.extend((x0+(ep-1)/max(curves[-1][0]-1, 1)*(x1-x0),
                               y1-(means[index]-low)/(high-low)*(y1-y0)))
            if len(points) >= 4:
                c.create_line(*points, fill=COLORS[index], width=2)
            else:
                x, y = points
                c.create_oval(x-3, y-3, x+3, y+3, fill=COLORS[index], outline="")
            c.create_text(x1, top+12, text=f"{values[-1]:.2f}", anchor="e", fill=COLORS[index])


class App:
    def __init__(self, root):
        self.root = root
        root.title("Bomberman RL · Train / Xem đấu / Chơi")
        root.geometry("1180x880")
        root.minsize(980, 760)
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TButton", background=PANEL, padding=(10, 6))
        style.map("TButton", background=[("active", "#2c4260")])
        style.configure("TNotebook", borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, padding=(18, 10))
        style.map("TNotebook.Tab", background=[("selected", "#2c4260")])
        style.configure("TEntry", fieldbackground=PANEL, insertcolor=TEXT)
        style.configure("TCombobox", fieldbackground=PANEL, arrowcolor=TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)])
        style.configure("TSpinbox", fieldbackground=PANEL, arrowcolor=TEXT)
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, rowheight=27)
        style.configure("Treeview.Heading", background="#293951", padding=4)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("TProgressbar", background=COLORS[0], troughcolor=PANEL)
        root.option_add("*TCombobox*Listbox.background", PANEL)
        root.option_add("*TCombobox*Listbox.foreground", TEXT)
        header = ttk.Frame(root, padding=(16, 12))
        header.pack(fill="x")
        title_font = ("Segoe UI", 15, "bold")
        font_path = ROOT / "assets" / "emulogic.ttf"
        if font_path.is_file():
            import sys
            if sys.platform == "win32":
                import ctypes
                # Private to this process; no font installation or system change.
                if ctypes.windll.gdi32.AddFontResourceExW(str(font_path), 0x10, 0):
                    title_font = ("Emulogic", 12)
        ttk.Label(header, text="BOMBERMAN / RL", font=title_font).pack(side="left")
        ttk.Label(header, text="Checkpoint", style="Muted.TLabel").pack(side="left", padx=(25, 8))
        self.model = tk.StringVar()
        self.model_box = ttk.Combobox(header, textvariable=self.model, state="readonly")
        self.model_box.pack(side="left", fill="x", expand=True)
        ttk.Button(header, text="Làm mới", command=self.refresh_models).pack(side="left", padx=(8, 0))
        self.refresh_models()
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)
        self.training = TrainPane(self.tabs, self)
        self.watch = GamePane(self.tabs, self)
        self.play = GamePane(self.tabs, self, human=True)
        for pane, label in ((self.training, "01 · Train trực tiếp"),
                            (self.watch, "02 · Agent vs Bots"), (self.play, "03 · Bạn vs Agent + Bots")):
            self.tabs.add(pane, text=label)
        self.tabs.bind("<<NotebookTabChanged>>", self.tab_changed)
        self.tabs.select(self.watch)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def refresh_models(self):
        models = sorted(ROOT.glob("checkpoints/*.pt")) + sorted(ROOT.glob("runs/*/*.pt"))
        models = [str(p) for p in models if p.name.startswith(("best_", "final_"))]
        self.model_box.configure(values=models)
        if not self.model.get() and models:
            preferred = str(ROOT / "checkpoints" / "best_phaseE_4a.pt")
            self.model.set(preferred if preferred in models else models[0])

    def tab_changed(self, _):
        selected = self.tabs.select()
        for pane in (self.watch, self.play):
            if selected != str(pane):
                pane.pause()
            elif pane.env is None and self.model.get():
                pane.new_match()

    def close(self):
        self.watch.pause()
        self.play.pause()
        if self.training.thread is not None and self.training.thread.is_alive():
            self.training.stop()
            self.root.after(100, self.close)
        else:
            self.root.destroy()


if __name__ == "__main__":
    # Tiny CNN batches run faster without a large CPU thread pool.
    torch.set_num_threads(1)
    App(tk.Tk()).root.mainloop()
