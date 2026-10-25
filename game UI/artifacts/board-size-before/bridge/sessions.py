"""One serialized engine owner per match. Rendering never drives the engine."""
from collections import deque
from copy import deepcopy
import threading
import time
import uuid

from bomberman_env import BombermanFFAEnv
from rule_based_bot import RuleBasedAgent
from .models import load_policy
from .snapshots import capture_state

DIRECTIONS = {"stop": 0, "up": 1, "down": 2, "left": 3, "right": 4}


class GameSession:
    def __init__(self, config):
        self.config = config
        self.session_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.closed = False
        self.running = False
        self.speed = config.speed
        self.revision = 0
        self.connections = 0
        self.last_seen = time.monotonic()
        self.command_ids = deque(maxlen=256)
        self.input_seq = -1
        self.held = 0
        self.pending = 0
        self.events = deque(maxlen=100)
        self.event_seq = 0
        self.tick_ms = 0.
        self.error = None
        self.policy = load_policy(config.model_id, 11)
        self._reset_locked()
        self.thread = threading.Thread(target=self._loop, name=f"match-{self.session_id[:8]}", daemon=True)
        self.thread.start()

    def _reset_locked(self):
        self.match_id = uuid.uuid4().hex
        self.env = BombermanFFAEnv(board_size=11, num_agents=self.config.num_agents,
                                  max_steps=self.config.max_steps, seed=self.config.seed,
                                  sudden_death=self.config.sudden_death)
        self.obs = self.env.reset()
        self.trained_id = 1 if self.config.mode == "play" else 0
        self.bots = {i: RuleBasedAgent(i) for i in range(self.env.num_agents)
                     if i != self.trained_id and not (self.config.mode == "play" and i == 0)}
        self.labels = ["Bạn" if self.config.mode == "play" and i == 0 else
                       "AI · Actor-Critic" if i == self.trained_id else "RuleBot"
                       for i in range(self.env.num_agents)]
        self.rewards = [0.] * self.env.num_agents
        self.actions = [0] * self.env.num_agents
        self.running = False
        self.held = self.pending = 0
        self.events.clear()
        self.error = None
        self._capture()

    def _capture(self):
        self.frame = capture_state(self.env, self.labels, self.actions, self.rewards)
        self.revision += 1

    def state(self):
        with self.lock:
            return dict(deepcopy(self.frame), session_id=self.session_id, match_id=self.match_id,
                        paused=not self.running, running=self.running, speed=self.speed,
                        mode=self.config.mode, revision=self.revision, tick_ms=round(self.tick_ms, 2),
                        events=list(deepcopy(self.events)), error=self.error)

    def _event(self, kind, **fields):
        self.event_seq += 1
        self.events.append(dict(id=self.event_seq, tick=self.env.step_count, kind=kind, **fields))

    def _step_locked(self):
        if self.env._episode_done or self.closed:
            self.running = False
            return
        started = time.perf_counter()
        old_bombs = {(b.row, b.col) for b in self.env.bombs}
        old_alive = {a.agent_id for a in self.env.agents if a.alive}
        old_items = set(self.env.powerups)
        old_grid = self.env.grid.copy()
        acts = [0] * self.env.num_agents
        for i, agent in enumerate(self.env.agents):
            if not agent.alive:
                continue
            if self.config.mode == "play" and i == 0:
                acts[i] = self.pending or self.held
                self.pending = 0
            elif i == self.trained_id:
                acts[i] = self.policy.select_action(self.obs[i], self.env, i)
            else:
                acts[i] = self.bots[i].select_action(self.env)
        self.obs, reward, _, info = self.env.step(acts)
        self.actions = acts
        self.rewards = [float(a + b) for a, b in zip(self.rewards, reward)]
        if any(action == 5 for action in acts):
            self._event("bomb")
        if old_bombs - {(b.row, b.col) for b in self.env.bombs}:
            self._event("explosion", cells=[list(p) for p in self.env.flames])
        for agent in self.env.agents:
            if agent.agent_id in old_alive and not agent.alive:
                self._event("defeat", row=agent.row, col=agent.col, agent=agent.agent_id)
        for pos in old_items - set(self.env.powerups):
            self._event("pickup", row=pos[0], col=pos[1])
        for r, c in zip(*((old_grid == 2) & (self.env.grid != 2)).nonzero()):
            self._event("crate", row=int(r), col=int(c))
        if info["episode_done"]:
            self.running = False
            self.held = self.pending = 0
            self._event("end")
        self.tick_ms = (time.perf_counter() - started) * 1000
        self._capture()

    def pause(self):
        with self.condition:
            self.running = False
            self.held = self.pending = 0
            self.revision += 1
            self.condition.notify_all()

    def input(self, message):
        seq, direction, bomb = message.get("seq"), message.get("direction"), message.get("bomb", False)
        if type(seq) is not int or seq < 0 or direction not in DIRECTIONS or type(bomb) is not bool:
            raise ValueError("Input không hợp lệ.")
        with self.lock:
            if seq <= self.input_seq:
                return
            self.input_seq = seq
            self.held = DIRECTIONS[direction]
            if bomb:
                self.pending = 5
            elif self.held and self.pending != 5:
                self.pending = self.held

    def command(self, message):
        name, cid = message.get("command"), message.get("command_id")
        if not isinstance(cid, str) or not 1 <= len(cid) <= 128:
            raise ValueError("Lệnh cần command_id hợp lệ.")
        if name not in ("pause", "resume", "step", "reset", "set_speed"):
            raise ValueError("Lệnh không được hỗ trợ.")
        with self.condition:
            if cid in self.command_ids:
                return dict(type="ack", command_id=cid, duplicate=True)
            if name == "set_speed":
                speed = message.get("speed")
                if type(speed) is not int or not 1 <= speed <= 60:
                    raise ValueError("Tốc độ phải từ 1 đến 60 bước/giây.")
                self.speed = speed
            elif name == "pause":
                self.pause()
            elif name == "resume":
                if not self.env._episode_done:
                    self.running = True
            elif name == "step":
                self.running = False
                self._step_locked()
            elif name == "reset":
                self._reset_locked()
            self.command_ids.append(cid)
            self.revision += 1
            self.condition.notify_all()
            return dict(type="ack", command_id=cid, tick=self.env.step_count)

    def _loop(self):
        due = time.monotonic()
        while True:
            # Release between ticks even when inference is slower than the target rate.
            with self.condition:
                if self.closed:
                    return
                if not self.running:
                    self.condition.wait(.1)
                    due = time.monotonic()
                    continue
                if time.monotonic() - self.last_seen > 2:
                    self.pause()
                    continue
                delay = due - time.monotonic()
                if delay > 0:
                    self.condition.wait(min(delay, .1))
                    continue
                started = time.monotonic()
                try:
                    self._step_locked()
                except Exception as exc:
                    self.error = f"Không chạy được bước game: {exc}"
                    self.pause()
                due = max(started + 1 / self.speed, time.monotonic() + .001)

    def close(self):
        with self.condition:
            self.closed = True
            self.pause()
            self.condition.notify_all()
        self.thread.join(timeout=3)
