"""Loopback HTTP/WS entry point. Run via ../run.py or uvicorn bridge.app:app."""
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import pickle
import time
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch

from . import UI_ROOT
from .models import catalog
from .protocol import SessionConfig, TrainingConfig, VERSION
from .sessions import GameSession
from .training import TrainingManager

torch.set_num_threads(1)
sessions = {}
training = TrainingManager()


@asynccontextmanager
async def lifespan(app):
    yield
    await asyncio.to_thread(training.close)
    for session in list(sessions.values()):
        await asyncio.to_thread(session.close)
    sessions.clear()


app = FastAPI(title="Bomberman Foundry · Local bridge", lifespan=lifespan)


def allowed_origin(origin):
    if origin is None:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in ("http", "https") and parsed.hostname in ("127.0.0.1", "localhost", "::1")


@app.middleware("http")
async def local_writes(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS") and not allowed_origin(request.headers.get("origin")):
        return JSONResponse({"detail": "Chỉ giao diện local được phép điều khiển game."}, status_code=403)
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"status": "ok", "protocol_version": VERSION, "sessions": len(sessions)}


@app.get("/api/models")
def models():
    return {"models": catalog()}


def get_session(sid):
    if sid not in sessions:
        raise HTTPException(404, "Không tìm thấy ván đấu này.")
    return sessions[sid]


@app.post("/api/sessions", status_code=201)
async def create_session(config: SessionConfig):
    if len(sessions) >= 12:
        raise HTTPException(409, "Có quá nhiều ván đang mở. Hãy đóng ván cũ trước.")
    try:
        session = await asyncio.to_thread(GameSession, config)
    except (ValueError, OSError, RuntimeError, pickle.UnpicklingError, EOFError, IndexError) as exc:
        raise HTTPException(422, f"Không nạp được mô hình: {exc}") from exc
    sessions[session.session_id] = session
    return {"session_id": session.session_id, "snapshot": session.state()}


@app.get("/api/sessions/{sid}")
async def read_session(sid: str):
    session = get_session(sid)
    return {"session_id": sid, "snapshot": await asyncio.to_thread(session.state)}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    session = get_session(sid)
    sessions.pop(sid)
    await asyncio.to_thread(session.close)
    return {"status": "closed"}


@app.websocket("/api/sessions/{sid}/ws")
async def socket(ws: WebSocket, sid: str):
    if not allowed_origin(ws.headers.get("origin")) or sid not in sessions:
        await ws.close(code=1008)
        return
    session = sessions[sid]
    await ws.accept()
    send_lock = asyncio.Lock()

    async def send(message):
        async with send_lock:
            await ws.send_json(message)

    session.last_seen = time.monotonic()
    session.connections += 1
    await send({"type": "hello", "protocol_version": VERSION, "session_id": sid})
    initial = await asyncio.to_thread(session.state)
    await send({"type": "snapshot", "state": initial})

    async def updates():
        revision = initial["revision"]
        while not session.closed:
            state = await asyncio.to_thread(session.state)
            if state["revision"] != revision:
                await send({"type": "snapshot", "state": state})
                revision = state["revision"]
            await asyncio.sleep(.025)

    sender = asyncio.create_task(updates())
    try:
        while not session.closed:
            raw = await ws.receive_text()
            session.last_seen = time.monotonic()
            try:
                if len(raw) > 4096:
                    raise ValueError("Message quá lớn.")
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("Message phải là một JSON object.")
                if message.get("protocol_version", VERSION) != VERSION:
                    await send({"type": "error", "code": "PROTOCOL_VERSION", "message": "Phiên bản giao thức không được hỗ trợ."})
                    continue
                kind = message.get("type")
                if kind == "ping":
                    await send({"type": "pong"})
                elif kind == "input":
                    await asyncio.to_thread(session.input, message)
                elif kind == "command":
                    ack = await asyncio.to_thread(session.command, message)
                    await send(ack)
                    await send({"type": "snapshot", "state": await asyncio.to_thread(session.state)})
                else:
                    raise ValueError("Loại message không được hỗ trợ.")
            except (ValueError, TypeError, KeyError) as exc:
                await send({"type": "error", "code": "INVALID_MESSAGE", "message": str(exc)})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sender.cancel()
        with suppress(asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
            await sender
        session.connections = max(0, session.connections - 1)
        await asyncio.to_thread(session.pause)


def get_job(jid):
    job = training.jobs.get(jid)
    if job is None:
        raise HTTPException(404, "Không tìm thấy job do giao diện tạo.")
    return job


@app.post("/api/training", status_code=201)
async def create_training(config: TrainingConfig):
    try:
        job = await asyncio.to_thread(training.start, config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"job_id": job.job_id}


@app.get("/api/training/{jid}")
def read_training(jid: str):
    return get_job(jid).state()


@app.post("/api/training/{jid}/{command}")
def training_control(jid: str, command: str):
    job = get_job(jid)
    try:
        job.control(command)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return job.state()


dist = UI_ROOT / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="game")
else:
    @app.get("/")
    def not_built():
        return HTMLResponse("<h1>Giao diện chưa được build</h1><p>Chạy npm run build trong folder game UI, sau đó khởi động lại.</p>", status_code=503)
