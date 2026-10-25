"""Opaque, allowlisted model catalog; inference never accepts a client path."""
from functools import lru_cache
from hashlib import sha256

from . import REPO_ROOT, UI_ROOT


def catalog():
    paths = list((REPO_ROOT / "checkpoints").glob("*.pt"))
    paths += list((REPO_ROOT / "runs").glob("*/*.pt"))
    paths += list((UI_ROOT / "runs").glob("*/*.pt"))
    paths = [p for p in paths if p.name.startswith(("best_", "final_"))
             and p.is_file() and p.stat().st_size > 1024
             and p.resolve().is_relative_to(REPO_ROOT.resolve())]
    paths.sort(key=lambda p: (p.name != "best_phaseE_4a.pt", "checkpoints" not in p.parts,
                              p.name, str(p)))
    rows = []
    for p in paths:
        relative = p.relative_to(REPO_ROOT).as_posix()
        rows.append({"id": sha256(relative.encode()).hexdigest()[:20], "name": p.name,
                     "source": relative, "size": p.stat().st_size})
    return rows


def resolve_model(model_id):
    if not isinstance(model_id, str) or not model_id or len(model_id) != 20:
        raise ValueError("Checkpoint không hợp lệ. Hãy chọn một mô hình trong danh sách.")
    for row in catalog():
        if row["id"] == model_id:
            path = (REPO_ROOT / row["source"]).resolve()
            if not path.is_relative_to(REPO_ROOT.resolve()):
                break
            return path
    raise ValueError("Không tìm thấy checkpoint đã chọn. Hãy làm mới danh sách mô hình.")


@lru_cache(maxsize=3)
def _load(path, board_size, modified):
    from train_actor_critic import ActorCriticNet, SnapshotOpponent, smart_load_weights
    net = ActorCriticNet(board_size=board_size)
    smart_load_weights(net, path, "cpu")
    return SnapshotOpponent(net, "cpu")


def load_policy(model_id, board_size):
    path = resolve_model(model_id)
    return _load(str(path), board_size, path.stat().st_mtime_ns)
