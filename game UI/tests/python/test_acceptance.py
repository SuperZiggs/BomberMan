"""Frozen public-contract acceptance tests; no production implementation imports.

Run with unittest discovery; all generated files and subprocess outputs stay in UI.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from dataclasses import asdict
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

UI = Path(__file__).resolve().parents[2]
ROOT = UI.parent
sys.path[:0] = [str(UI / ".python-deps"), str(UI), str(ROOT)]
ARTIFACTS = UI / "tests" / "_artifacts"
_SERVER = None
_SERVER_LOG = None
_BASE = os.environ.get("BOMBERMAN_API_URL")
TERMINAL = {"completed", "complete", "cancelled", "canceled", "stopped", "error", "failed"}


def tick(state):
    return state.get("tick", state.get("step"))


def paused(state):
    if "paused" in state:
        return state["paused"] is True
    if "running" in state:
        return state["running"] is False
    return state.get("status") == "paused"


def request(method, path, body=None, origin=None, timeout=35):
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    req = Request(_BASE + path, data=None if body is None else json.dumps(body).encode(),
                  method=method, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except HTTPError as error:
        raw = error.read()
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            data = raw.decode(errors="replace")
        return error.code, data


def stop_server():
    global _SERVER
    if _SERVER is not None:
        _SERVER.terminate()
        try:
            _SERVER.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _SERVER.kill()
            _SERVER.wait(timeout=5)
        _SERVER = None
    if _SERVER_LOG is not None:
        _SERVER_LOG.close()


atexit.register(stop_server)


def require_bridge(case):
    """Missing behavior is an assertion, never an import/dependency error."""
    try:
        importlib.import_module("bridge.app")
    except ModuleNotFoundError as error:
        case.fail(f"Required public ASGI package bridge.app is not available: {error}")
    except Exception as error:
        case.fail(f"Required public ASGI app cannot initialize: {type(error).__name__}: {error}")


def ensure_server(case):
    global _SERVER, _SERVER_LOG, _BASE
    require_bridge(case)
    if _BASE is not None:
        return
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    _BASE = f"http://127.0.0.1:{port}"
    _SERVER_LOG = (ARTIFACTS / "acceptance-server.log").open("w", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (UI / ".python-deps", UI, ROOT))),
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    _SERVER = subprocess.Popen([sys.executable, "-m", "uvicorn", "bridge.app:app", "--host", "127.0.0.1",
                                "--port", str(port)], cwd=UI, env=env,
                               stdout=_SERVER_LOG, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _SERVER.poll() is not None:
            case.fail("Public ASGI server exited during startup; see tests/_artifacts/acceptance-server.log")
        try:
            if request("GET", "/api/health", timeout=1)[0] == 200:
                return
        except (URLError, TimeoutError):
            pass
        time.sleep(.1)
    case.fail("Public ASGI server did not provide GET /api/health within 30 seconds")


def cell_values(value):
    """Normalize copied cell collections without imposing a presentation schema."""
    if isinstance(value, dict):
        return sorted(map(str, value.items()))
    return sorted(map(str, value))


class SnapshotTests(unittest.TestCase):
    def adapter(self):
        try:
            module = importlib.import_module("bridge.snapshots")
        except ModuleNotFoundError as error:
            self.fail(f"Required public snapshot adapter is absent: {error}")
        self.assertTrue(callable(getattr(module, "capture_state", None)), "capture_state must be callable")
        return module.capture_state

    def test_snapshot_is_json_copied_and_does_not_mutate_or_reseed(self):
        from bomberman_env import BombermanFFAEnv, Bomb
        capture = self.adapter()
        env = BombermanFFAEnv(board_size=11, num_agents=4, seed=912743)
        env.reset()
        env.bombs.append(Bomb(1, 1, 0, 7, 2))
        env.flames[(3, 3)] = 2
        env.powerups[(3, 5)] = "ammo"
        labels, actions, rewards = ["Bạn", "AC", "Bot", "Bot"], [5, 0, 0, 0], [0., .1, .2, .3]
        before = pickle.dumps(env.__dict__)
        state = capture(env, labels, actions, rewards)
        self.assertEqual(pickle.dumps(env.__dict__), before, "Capturing must not mutate/reseed engine or RNG")
        json.dumps(state, allow_nan=False)
        self.assertEqual(state["grid"], env.grid.tolist())
        self.assertEqual(state["agents"], [asdict(a) for a in env.agents])
        self.assertEqual(state["bombs"], [asdict(b) for b in env.bombs])
        self.assertTrue(state["flames"], "Active engine flames must appear in the snapshot")
        self.assertTrue(state["powerups"], "Engine pickups must appear in the snapshot")
        self.assertEqual(state["labels"], labels)
        self.assertEqual(state["actions"], actions)
        self.assertEqual(state["rewards"], rewards)
        self.assertEqual(tick(state), 0)
        self.assertFalse(state["done"])
        self.assertIn("result", state)
        frozen = json.dumps(state, ensure_ascii=False, sort_keys=True)
        env.grid[1, 1] = 2
        env.agents[0].ammo = 3
        env.bombs[0].timer = 1
        env.flames.clear()
        env.powerups.clear()
        labels[0], actions[0], rewards[0] = "changed", 1, 99
        self.assertEqual(json.dumps(state, ensure_ascii=False, sort_keys=True), frozen,
                         "Every nested snapshot field must be copied")
        state["agents"][1]["col"] = 99
        self.assertNotEqual(env.agents[1].col, 99)

    def test_small_training_frame_and_terminal_result(self):
        from bomberman_env import BombermanFFAEnv
        capture = self.adapter()
        env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=1, seed=912744)
        env.reset()
        _, rewards, _, _ = env.step([0, 0])
        state = capture(env, ["AC", "Bot"], [0, 0], rewards)
        json.dumps(state, allow_nan=False)
        self.assertEqual(len(state["grid"]), 9)
        self.assertEqual(len(state["agents"]), 2)
        self.assertEqual(tick(state), 1)
        self.assertTrue(state["done"])
        self.assertIsNotNone(state["result"], "Terminal snapshots must contain a readable result")


class APIBase(unittest.TestCase):
    def setUp(self):
        ensure_server(self)
        self.sessions, self.jobs = [], []

    def tearDown(self):
        for job in self.jobs:
            request("POST", f"/api/training/{job}/stop", {})
            self.wait_job(job, terminal=True)
        for sid in self.sessions:
            request("DELETE", f"/api/sessions/{sid}")

    def model(self):
        code, data = request("GET", "/api/models")
        self.assertEqual(code, 200)
        self.assertIsInstance(data.get("models"), list)
        self.assertGreater(len(data["models"]), 0, "Catalog must contain a real allowed checkpoint")
        return data["models"][0]["id"]

    def create(self, **changes):
        config = dict(mode="play", model_id=self.model(), num_agents=3, speed=1,
                      seed=912745, max_steps=8, sudden_death=False)
        config.update(changes)
        code, data = request("POST", "/api/sessions", config)
        self.assertIn(code, (200, 201), str(data))
        self.assertIn("session_id", data)
        self.sessions.append(data["session_id"])
        self.assertIn("snapshot", data)
        return data["session_id"], data["snapshot"]

    def state(self, sid):
        code, data = request("GET", f"/api/sessions/{sid}")
        self.assertEqual(code, 200, str(data))
        return data["snapshot"]

    def wait_state(self, sid, predicate, seconds=4):
        end = time.monotonic() + seconds
        while True:
            state = self.state(sid)
            if predicate(state):
                return state
            if time.monotonic() > end:
                self.fail(f"Expected authoritative state was not observed: {state}")
            time.sleep(.025)

    @contextmanager
    def ws(self, sid, origin=None):
        from websockets.sync.client import connect
        url = _BASE.replace("http://", "ws://") + f"/api/sessions/{sid}/ws"
        with connect(url, origin=origin, open_timeout=10, close_timeout=2) as ws:
            hello = json.loads(ws.recv(timeout=5))
            self.assertEqual(hello.get("type"), "hello")
            self.assertEqual(hello.get("protocol_version"), 1)
            self.assertEqual(hello.get("session_id"), sid)
            frame = json.loads(ws.recv(timeout=5))
            self.assertEqual(frame.get("type"), "snapshot")
            self.assertEqual(frame["state"]["session_id"], sid)
            yield ws

    def until(self, ws, predicate, seconds=5):
        end = time.monotonic() + seconds
        seen = []
        while time.monotonic() < end:
            try:
                msg = json.loads(ws.recv(timeout=max(.01, end - time.monotonic())))
            except TimeoutError:
                self.fail(f"Expected WebSocket response not received; observed {seen}")
            seen.append(msg)
            if predicate(msg):
                return msg
        self.fail(f"Expected WebSocket response not received; observed {seen}")

    def command(self, ws, command, command_id=None, **fields):
        command_id = command_id or uuid.uuid4().hex
        ws.send(json.dumps(dict(type="command", command_id=command_id, command=command, **fields)))
        self.until(ws, lambda m: m.get("type") == "ack" and m.get("command_id") == command_id)
        return command_id

    def input(self, ws, seq, direction="stop", bomb=False):
        ws.send(json.dumps(dict(type="input", seq=seq, direction=direction, bomb=bomb)))

    def wait_job(self, job, terminal=False, seconds=40):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            code, data = request("GET", f"/api/training/{job}")
            self.assertEqual(code, 200, str(data))
            if not terminal or data.get("status") in TERMINAL:
                return data
            time.sleep(.1)
        self.fail(f"Bounded training job did not finish within {seconds}s: {data}")


class HealthCatalogTests(APIBase):
    def test_health_protocol_and_opaque_catalog(self):
        code, data = request("GET", "/api/health")
        self.assertEqual(code, 200)
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("protocol_version"), 1)
        code, catalog = request("GET", "/api/models")
        self.assertEqual(code, 200)
        self.assertGreater(len(catalog["models"]), 0)
        ids = []
        for model in catalog["models"]:
            self.assertIsInstance(model.get("id"), str)
            self.assertTrue(model.get("name"))
            self.assertFalse(any(c in model["id"] for c in ("/", "\\", ":", "..")), "Model ID must be opaque")
            ids.append(model["id"])
        self.assertEqual(len(ids), len(set(ids)))


class SessionTests(APIBase):
    def test_supported_modes_and_counts(self):
        for mode, counts in (("play", (3, 4)), ("watch", (2, 3, 4))):
            for count in counts:
                with self.subTest(mode=mode, count=count):
                    sid, state = self.create(mode=mode, num_agents=count)
                    self.assertEqual(len(state["grid"]), 11)
                    self.assertTrue(all(len(row) == 11 for row in state["grid"]))
                    self.assertEqual(len(state["agents"]), count)
                    self.assertEqual(state["session_id"], sid)
                    self.assertTrue(state["match_id"])
                    self.assertEqual(tick(state), 0)
                    self.assertTrue(paused(state), "Sessions start paused")
                    self.assertTrue({"bombs", "flames", "powerups", "labels", "actions", "rewards", "done", "result", "max_steps"} <= state.keys())

    def test_invalid_session_configs_are_recoverable(self):
        base = dict(mode="play", model_id=self.model(), num_agents=3, max_steps=8)
        bad = [dict(mode="unknown"), dict(num_agents=2), dict(num_agents=5), dict(speed=0),
               dict(speed=61), dict(max_steps=0), dict(model_id="missing-model"),
               dict(model_id="../checkpoints/best_phaseE_4a.pt"),
               dict(model_id=str(ROOT / "checkpoints" / "best_phaseE_4a.pt")), dict(model_id="")]
        for patch in bad:
            with self.subTest(patch=patch):
                code, data = request("POST", "/api/sessions", dict(base, **patch))
                self.assertTrue(400 <= code < 500, f"Bad config {patch} must fail with 4xx: {code}: {data}")
        missing = dict(base)
        del missing["model_id"]
        self.assertTrue(400 <= request("POST", "/api/sessions", missing)[0] < 500)
        self.create()

    def test_ws_handshake_ping_step_and_duplicate_command(self):
        sid, _ = self.create()
        with self.ws(sid) as ws:
            ws.send('{"type":"ping"}')
            self.until(ws, lambda m: m.get("type") == "pong")
            cid = self.command(ws, "step")
            self.assertEqual(tick(self.wait_state(sid, lambda s: tick(s) == 1)), 1)
            self.command(ws, "step", command_id=cid)
            time.sleep(.15)
            state = self.state(sid)
            self.assertEqual(tick(state), 1, "Repeated command ID must not execute twice")
            self.assertTrue(paused(state))

    def test_tap_and_old_input_sequence(self):
        sid, initial = self.create()
        with self.ws(sid) as ws:
            self.input(ws, 10, "right")
            self.input(ws, 9, "down")
            self.input(ws, 11, "stop")
            self.command(ws, "step")
            state = self.wait_state(sid, lambda s: tick(s) == 1)
            self.assertEqual(state["actions"][0], 4, "Short right tap survives release; old sequence cannot replace it")
            self.assertEqual(state["agents"][0]["col"], initial["agents"][0]["col"] + 1)
            self.command(ws, "step")
            state = self.wait_state(sid, lambda s: tick(s) == 2)
            self.assertEqual(state["actions"][0], 0, "Released tap is consumed once")

    def test_bomb_is_not_replaced_by_movement(self):
        sid, _ = self.create()
        with self.ws(sid) as ws:
            self.input(ws, 1, "stop", bomb=True)
            self.input(ws, 2, "right")
            self.command(ws, "step")
            state = self.wait_state(sid, lambda s: tick(s) == 1)
            self.assertEqual(state["actions"][0], 5)
            self.assertTrue(any(b["owner"] == 0 for b in state["bombs"]))

    def test_pause_clears_controls(self):
        sid, _ = self.create()
        with self.ws(sid) as ws:
            self.input(ws, 1, "right", bomb=True)
            self.command(ws, "pause")
            self.command(ws, "step")
            state = self.wait_state(sid, lambda s: tick(s) == 1)
            self.assertEqual(state["actions"][0], 0, "Pause clears held and queued controls")
            self.assertFalse(any(b["owner"] == 0 for b in state["bombs"]))

    def test_pause_resume_speed_and_reset(self):
        sid, initial = self.create(max_steps=100)
        with self.ws(sid) as ws:
            self.command(ws, "set_speed", speed=60)
            self.command(ws, "resume")
            self.wait_state(sid, lambda s: tick(s) >= 1)
            self.command(ws, "pause")
            state = self.wait_state(sid, paused)
            saved_tick = tick(state)
            time.sleep(.15)
            self.assertEqual(tick(self.state(sid)), saved_tick)
            self.command(ws, "step")
            self.assertEqual(tick(self.wait_state(sid, lambda s: tick(s) == saved_tick + 1)), saved_tick + 1)
            self.command(ws, "set_speed", speed=1)
            for speed in (0, 61):
                ws.send(json.dumps(dict(type="command", command_id=uuid.uuid4().hex, command="set_speed", speed=speed)))
                error = self.until(ws, lambda m: m.get("type") == "error")
                self.assertTrue(error.get("code") and error.get("message"))
            self.command(ws, "reset")
            state = self.wait_state(sid, lambda s: tick(s) == 0)
            self.assertNotEqual(state["match_id"], initial["match_id"])
            self.assertTrue(paused(state))

    def test_terminal_never_advances(self):
        sid, _ = self.create(max_steps=1)
        with self.ws(sid) as ws:
            self.command(ws, "step")
            state = self.wait_state(sid, lambda s: s["done"])
            self.assertEqual(tick(state), 1)
            self.assertIsNotNone(state["result"])
            self.command(ws, "step")
            self.command(ws, "resume")
            time.sleep(.15)
            self.assertEqual(tick(self.state(sid)), 1)

    def test_disconnect_pauses_and_reconnects(self):
        sid, _ = self.create(max_steps=100)
        with self.ws(sid) as ws:
            self.command(ws, "resume")
            self.wait_state(sid, lambda s: tick(s) >= 1)
            self.input(ws, 3, "right", bomb=True)
        state = self.wait_state(sid, paused)
        saved_tick = tick(state)
        time.sleep(.15)
        self.assertEqual(tick(self.state(sid)), saved_tick)
        with self.ws(sid) as ws:
            self.command(ws, "step")
            state = self.wait_state(sid, lambda s: tick(s) == saved_tick + 1)
            self.assertEqual(state["actions"][0], 0)

    def test_invalid_messages_and_unknown_version(self):
        sid, _ = self.create()
        with self.ws(sid) as ws:
            for bad in ("not JSON", "[]", '{"type":"surprise"}',
                        '{"type":"input","seq":1,"direction":"diagonal"}',
                        json.dumps(dict(type="command", command="resume", command_id="future-version", protocol_version=999))):
                ws.send(bad)
                error = self.until(ws, lambda m: m.get("type") == "error")
                self.assertTrue(error.get("code") and error.get("message"))
            self.assertEqual(tick(self.state(sid)), 0, "Unsupported protocol must never start simulation")
            ws.send('{"type":"ping"}')
            self.until(ws, lambda m: m.get("type") == "pong")
            self.command(ws, "step")
            self.assertEqual(tick(self.wait_state(sid, lambda s: tick(s) == 1)), 1)

    def test_foreign_origins_rejected(self):
        from websockets.sync.client import connect
        from websockets.exceptions import InvalidStatus
        config = dict(mode="play", model_id=self.model(), num_agents=3, max_steps=8)
        code, _ = request("POST", "/api/sessions", config, origin="https://foreign.example")
        self.assertIn(code, (400, 403))
        sid, _ = self.create()
        self.assertIn(request("DELETE", f"/api/sessions/{sid}", origin="https://foreign.example")[0], (400, 403))
        url = _BASE.replace("http://", "ws://") + f"/api/sessions/{sid}/ws"
        try:
            with connect(url, origin="https://foreign.example", open_timeout=3, close_timeout=1):
                self.fail("Foreign browser WebSocket Origin must be rejected before handshake succeeds")
        except InvalidStatus as error:
            self.assertIn(error.response.status_code, (400, 403))
        with self.ws(sid, origin=_BASE) as ws:
            ws.send('{"type":"ping"}')
            self.until(ws, lambda m: m.get("type") == "pong")

    def test_unknown_and_deleted_session(self):
        self.assertEqual(request("GET", "/api/sessions/not-a-ui-session")[0], 404)
        sid, _ = self.create()
        self.assertIn(request("DELETE", f"/api/sessions/{sid}")[0], (200, 204))
        self.assertEqual(request("GET", f"/api/sessions/{sid}")[0], 404)

    def test_authoritative_action_tape_matches_real_engine(self):
        from bomberman_env import BombermanFFAEnv
        sid, state = self.create(seed=912746, max_steps=8)
        reference = BombermanFFAEnv(board_size=11, num_agents=3, seed=912746, max_steps=8, sudden_death=False)
        reference.reset()
        self.assertEqual(state["grid"], reference.grid.tolist())
        self.assertEqual(state["agents"], [asdict(a) for a in reference.agents])
        cumulative = [0.] * 3
        with self.ws(sid) as ws:
            for i, direction in enumerate(("right", "stop", "down", "stop", "left", "stop", "up", "stop"), 1):
                self.input(ws, i, direction, bomb=i == 2)
                self.command(ws, "step")
                state = self.wait_state(sid, lambda s: tick(s) == i)
                _, rewards, _, info = reference.step(state["actions"])
                self.assertEqual(state["grid"], reference.grid.tolist())
                self.assertEqual(state["agents"], [asdict(a) for a in reference.agents])
                self.assertEqual(state["bombs"], [asdict(b) for b in reference.bombs])
                self.assertEqual(state["done"], info["episode_done"])
                cumulative = [total + value for total, value in zip(cumulative, rewards)]
                self.assertTrue(any(all(abs(a - b) < 1e-8 for a, b in zip(state["rewards"], values))
                                    for values in (rewards, cumulative)),
                                "Displayed rewards must be actual step rewards or their cumulative sums")
                # Rewards can be per-tick or cumulative; verify authoritative adapter for both.
                from bridge.snapshots import capture_state
                expected = capture_state(reference, state["labels"], state["actions"], state["rewards"])
                self.assertEqual(cell_values(state["flames"]), cell_values(expected["flames"]))
                self.assertEqual(cell_values(state["powerups"]), cell_values(expected["powerups"]))
                self.assertTrue(all(math.isfinite(r) for r in state["rewards"]))
                if state["done"]:
                    break
        self.assertTrue(state["done"])


class TrainingTests(APIBase):
    def test_validation_and_job_isolation(self):
        base = dict(phase="C", episodes=1, num_agents=2, max_steps=8, warm_start=False)
        bad = [dict(phase="A"), dict(phase="F"), dict(episodes=0), dict(episodes=-1),
               dict(episodes=1.5), dict(episodes="bad"), dict(phase="C", num_agents=3),
               dict(phase="D", num_agents=4), dict(phase="E", num_agents=1),
               dict(phase="E", num_agents=5), dict(max_steps=0),
               dict(warm_start=True), dict(warm_start=True, model_id="missing"),
               dict(warm_start=True, model_id="../checkpoints/best_phaseC.pt")]
        for patch in bad:
            with self.subTest(patch=patch):
                code, data = request("POST", "/api/training", dict(base, **patch))
                if code in (200, 201) and isinstance(data, dict) and "job_id" in data:
                    self.jobs.append(data["job_id"])
                self.assertTrue(400 <= code < 500, f"Invalid training config must fail: {patch}: {code}: {data}")
        self.assertEqual(request("GET", "/api/training/existing-experiment")[0], 404)
        for action in ("pause", "resume", "stop", "unknown-action"):
            code, _ = request("POST", f"/api/training/existing-experiment/{action}", {})
            self.assertTrue(400 <= code < 500)
        self.assertIn(request("POST", "/api/training", base, origin="https://foreign.example")[0], (400, 403))

    def start_job(self, **changes):
        config = dict(phase="C", episodes=2, num_agents=2, max_steps=8,
                      warm_start=True, model_id=self.model())
        config.update(changes)
        code, data = request("POST", "/api/training", config)
        self.assertIn(code, (200, 201), str(data))
        self.assertTrue(data.get("job_id"))
        self.jobs.append(data["job_id"])
        return data["job_id"]

    def test_real_bounded_training_metrics_and_artifacts(self):
        job = self.start_job()
        data = self.wait_job(job, terminal=True)
        self.assertTrue({"job_id", "status", "episode", "episodes", "metrics", "frame", "error", "output_dir"} <= data.keys())
        self.assertNotIn(data["status"], ("error", "failed"), str(data.get("error")))
        self.assertEqual(data["episode"], 2)
        self.assertEqual(data["episodes"], 2)
        frame = data["frame"]
        self.assertIsInstance(frame, dict, "Actual on-step callback must supply latest board")
        self.assertEqual(len(frame["grid"]), 9)
        self.assertEqual(len(frame["agents"]), 2)
        self.assertLessEqual(tick(frame), 8)
        self.assertLessEqual(frame["max_steps"], 8)
        output = Path(data["output_dir"])
        if not output.is_absolute():
            output = UI / output
        self.assertTrue(output.resolve().is_relative_to((UI / "runs").resolve()))
        self.assertTrue(output.is_dir())
        self.assertTrue(list(output.rglob("*.pt")), "Completed real worker saves model weights")
        def callback_rows(value):
            if isinstance(value, dict):
                if {"reward", "loss", "win", "active_win"} <= value.keys():
                    yield value
                else:
                    for nested in value.values():
                        yield from callback_rows(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from callback_rows(nested)
        rows = list(callback_rows(data["metrics"]))
        self.assertTrue(rows, "Public metrics must contain actual episode callback values")
        for row in rows:
            self.assertLessEqual(row["steps"], 8)
            self.assertTrue(all(math.isfinite(row[key]) for key in ("reward", "loss", "win", "active_win")))
            self.assertLessEqual(row["active_win"], row["win"])
        self.assertTrue(data["metrics"], "API must expose actual callback metrics")
        # Parse the artifact with torch rather than trusting a filename placeholder.
        import torch
        weights = torch.load(next(output.rglob("*.pt")), map_location="cpu", weights_only=True)
        self.assertIsInstance(weights, dict)
        self.assertTrue(any(torch.is_tensor(v) for v in weights.values()), "Smoke must save learned network tensors")

    def test_job_controls_and_competing_start(self):
        job = self.start_job(warm_start=False)
        self.assertTrue(200 <= request("POST", f"/api/training/{job}/pause", {})[0] < 300)
        data = self.wait_job(job)
        if data["status"] not in TERMINAL:
            end = time.monotonic() + 4
            while data["status"] != "paused" and time.monotonic() < end:
                time.sleep(.025)
                data = self.wait_job(job)
            self.assertEqual(data["status"], "paused")
            ep = data["episode"]
            time.sleep(.15)
            self.assertEqual(self.wait_job(job)["episode"], ep)
            code, competing = request("POST", "/api/training", dict(phase="C", episodes=1, num_agents=2, max_steps=8))
            if code in (200, 201):
                self.jobs.append(competing["job_id"])
            self.assertEqual(code, 409, "A paused UI job still owns the one-active-job slot")
        self.assertTrue(200 <= request("POST", f"/api/training/{job}/resume", {})[0] < 300)
        self.assertTrue(200 <= request("POST", f"/api/training/{job}/stop", {})[0] < 300)
        data = self.wait_job(job, terminal=True)
        self.assertIn(data["status"], TERMINAL)
        self.assertNotIn(data["status"], ("error", "failed"), str(data.get("error")))
        output = Path(data["output_dir"])
        if not output.is_absolute():
            output = UI / output
        self.assertTrue(output.resolve().is_relative_to((UI / "runs").resolve()))
        self.assertTrue(list(output.rglob("*.pt")), "Stop-and-save must save current completed work")
        self.assertTrue(400 <= request("POST", f"/api/training/{job}/attach-existing", {})[0] < 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
