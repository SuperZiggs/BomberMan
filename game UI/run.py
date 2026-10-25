"""Run the local web app with the project's Torch runtime and isolated web deps."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / ".python-deps"), str(ROOT), str(ROOT.parent)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bomberman web UI (local only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run("bridge.app:app", host="127.0.0.1", port=args.port, ws_max_size=4096)
