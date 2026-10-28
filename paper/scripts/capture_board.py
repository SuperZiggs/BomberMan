"""Screenshot real game frames from the repo's game UI (visualize.BoardView) around one S3 kill.

Plays S3 (sampled policy + mask) against RuleBot on validation boards (seed 21000.., sudden death E)
until S3 wins by killing with its own bomb, then redraws four frames of that game in the Tk board
widget and grabs them from the screen: bomb placed, two intermediate steps, the explosion.
Validation seeds only. Usage: python paper/scripts/capture_board.py
Outputs paper/figures/board/frame_*.png and board/frames.json.
"""
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'paper' / 'figures' / 'board'
S3 = ROOT / 'runs/sd9-20261023/train/epoch_12.pt'

try:  # grab real pixels on scaled Windows displays
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402
from PIL import ImageGrab  # noqa: E402

import visualize  # noqa: E402
from bomberman_env import BombermanFFAEnv, ACTION_BOMB  # noqa: E402
from rule_based_bot import RuleBasedAgent  # noqa: E402


def play(seed, learner=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=seed, sudden_death=True,
                          sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
    obs = env.reset()
    s3, bot = visualize.load_agent(S3, 9), RuleBasedAgent(1 - learner)
    labels = ['S3', 'RuleBot'] if learner == 0 else ['RuleBot', 'S3']
    frames = [visualize.capture_state(env, labels)]
    kill_step = None
    for _ in range(500):
        actions = [0, 0]
        actions[learner] = s3.select_action(obs[learner], env, learner)
        actions[1 - learner] = bot.select_action(env)
        obs, _, _, info = env.step(actions)
        frames.append(visualize.capture_state(env, labels, actions))
        if any(k == learner and v != learner for k, v in info['kills']) and env.agents[learner].alive:
            kill_step = env.step_count
        if info['episode_done']:
            break
    won = env.agents[learner].alive and not env.agents[1 - learner].alive
    return frames, (kill_step if won else None)


def placement_step(frames, kill_step, learner=0):
    """Step at which the learner placed the bomb that was closest to exploding just before the kill."""
    mine = [b for b in frames[kill_step - 1]['bombs'] if b['owner'] == learner]
    if not mine:
        raise ValueError('no learner bomb before the kill')
    first = min(mine, key=lambda b: b['timer'])
    t = kill_step - 1
    while t > 0 and any(b['owner'] == learner and (b['row'], b['col']) == (first['row'], first['col'])
                        for b in frames[t - 1]['bombs']):
        t -= 1
    assert frames[t]['actions'][learner] == ACTION_BOMB or t == 0, 'placement step should be a BOMB action'
    return t


def grab(frames, picks):
    root = tk.Tk()
    root.attributes('-topmost', True)
    ttk.Style(root).configure('Muted.TLabel')
    view = visualize.BoardView(root)
    view.pack(fill='both', expand=True)
    view.canvas.configure(width=450, height=450)
    for _ in range(5):
        root.update()
    paths = []
    for name, t, danger in picks:
        view.state = None
        view.danger.set(danger)
        view.show(frames[t])
        view.draw()
        for _ in range(5):
            root.update()
        c = view.canvas
        size = 9 * max(1, (min(c.winfo_width(), c.winfo_height()) - 12) // 9)   # same cell maths as BoardView.draw
        x = c.winfo_rootx() + (c.winfo_width() - size) // 2
        y = c.winfo_rooty() + (c.winfo_height() - size) // 2
        img = ImageGrab.grab(bbox=(x, y, x + size, y + size), all_screens=True)
        assert len(set(img.resize((16, 16)).getdata())) > 8, 'blank or uniform grab'
        path = OUT / f'frame_{name}.png'
        img.save(path)
        paths.append(str(path.relative_to(ROOT)))
    root.destroy()
    return paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in range(21000, 21050):
        frames, kill_step = play(seed)
        if kill_step is not None:
            break
    else:
        raise SystemExit('no active win found')
    t0 = placement_step(frames, kill_step)
    picks = [('a_place', t0, False), ('b_mid', min(t0 + 4, kill_step - 1), True),
             ('c_before', kill_step - 1, True), ('d_kill', kill_step, False)]
    paths = grab(frames, picks)
    meta = {'seed': seed, 'learner': 'S3 (A0)', 'opponent': 'RuleBot (A1)', 'kill_step': kill_step,
            'game_length': frames[-1]['step'], 'frames': [{'name': n, 'step': t, 'danger_overlay': d, 'png': p}
                                                          for (n, t, d), p in zip(picks, paths)]}
    (OUT / 'frames.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
