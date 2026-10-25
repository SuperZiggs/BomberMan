"""Owned UI jobs only; existing experiment processes and outputs are untouched."""
from collections import deque
from copy import deepcopy
from datetime import datetime
import json
import threading
import time
import uuid

from . import UI_ROOT
from .models import resolve_model
from .snapshots import capture_state

TERMINAL = {"completed", "cancelled", "error"}


class TrainingJob:
    def __init__(self, config):
        self.config = config
        self.job_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.gate = threading.Event()
        self.gate.set()
        self.stop_event = threading.Event()
        self.status = "initializing"
        self.episode = 0
        self.metrics = deque(maxlen=5000)
        self.frame = None
        self.error = None
        self.started = time.time()
        self.output = UI_ROOT / "runs" / (datetime.now().strftime("ui-%Y%m%d-%H%M%S-") + self.job_id[:8])
        self.output.mkdir(parents=True, exist_ok=False)
        self.thread = threading.Thread(target=self._worker, name=f"ui-train-{self.job_id[:8]}", daemon=True)

    def state(self):
        with self.lock:
            return dict(job_id=self.job_id, status=self.status, episode=self.episode,
                        episodes=self.config.episodes, metrics=list(deepcopy(self.metrics)),
                        frame=deepcopy(self.frame), error=self.error,
                        output_dir=self.output.relative_to(UI_ROOT).as_posix(),
                        elapsed=round(time.time() - self.started, 1))

    def control(self, command):
        with self.lock:
            if command not in ("pause", "resume", "stop"):
                raise ValueError("Thao tác huấn luyện không được hỗ trợ.")
            if self.status in TERMINAL:
                return
            if command == "pause":
                self.gate.clear()
                self.status = "paused"
            elif command == "resume":
                self.status = "running"
                self.gate.set()
            else:
                self.status = "stopping"
                self.stop_event.set()
                self.gate.set()

    def _wait(self):
        while not self.gate.wait(.1):
            if self.stop_event.is_set():
                break

    def _worker(self):
        try:
            self._run()
        except Exception as exc:
            with self.lock:
                self.error = f"Huấn luyện thất bại: {exc}"
                self.status = "error"

    def _run(self):
        from train_actor_critic import train
        cfg = self.config
        resume = str(resolve_model(cfg.model_id)) if cfg.warm_start else ""
        kwargs = dict(phase=cfg.phase, episodes=cfg.episodes, num_agents=cfg.num_agents,
                      board_size=11 if cfg.num_agents == 4 else 9, max_steps=cfg.max_steps,
                      checkpoint_dir=str(self.output), device="cpu", resume_from=resume,
                      lr=1e-4 if resume else 3e-4, log_every=50, run_report=False,
                      seed=912760, validation_every=0)
        (self.output / "config.json").write_text(json.dumps(kwargs, ensure_ascii=False, indent=2), encoding="utf-8")
        last_frame = 0.

        def on_step(env, ep, aid, acts, opponents):
            nonlocal last_frame
            self._wait()
            if self.stop_event.is_set():
                return
            now = time.monotonic()
            if now - last_frame >= .08 or env.step_count == 0 or env._episode_done:
                labels = ["Đang học" if i == aid else "RuleBot" if opponents[i] == "rule" else "Snapshot"
                          for i in range(env.num_agents)]
                frame = capture_state(env, labels, acts)
                frame.update(paused=not self.gate.is_set(), mode="training", speed=12,
                             match_id=f"{self.job_id}-{ep}", session_id=self.job_id)
                with self.lock:
                    self.frame = frame
                last_frame = now

        with self.lock:
            if self.status == "initializing":
                self.status = "running"
        with (self.output / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            def on_episode(data):
                self._wait()
                row = dict(data)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                with self.lock:
                    self.metrics.append(row)
                    self.episode = int(row["episode"])
            train(**kwargs, on_step=on_step, on_episode=on_episode, stop_event=self.stop_event)
        with self.lock:
            self.status = "cancelled" if self.stop_event.is_set() else "completed"


class TrainingManager:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.RLock()

    def start(self, config):
        if config.warm_start:
            resolve_model(config.model_id)
        with self.lock:
            if any(j.state()["status"] not in TERMINAL for j in self.jobs.values()):
                raise RuntimeError("Đang có một job huấn luyện của giao diện. Dừng và lưu trước khi tạo job mới.")
            job = TrainingJob(config)
            self.jobs[job.job_id] = job
            try:
                job.thread.start()
            except Exception as exc:
                self.jobs.pop(job.job_id, None)
                raise RuntimeError(f"Không khởi tạo được worker huấn luyện: {exc}") from exc
            return job

    def close(self):
        for job in self.jobs.values():
            job.control("stop")
        for job in self.jobs.values():
            job.thread.join(timeout=5)
