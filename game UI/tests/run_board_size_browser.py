"""Build and run only the board-size browser acceptance on a disposable server."""
from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

UI = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (UI / ".python-deps", UI, UI.parent))),
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def main():
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required; use the existing project setup.")
    subprocess.run([node, "node_modules/vite/bin/vite.js", "build"], cwd=UI, env=ENV, check=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    artifacts = UI / "tests/_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / "board-size-server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "run.py", "--port", str(port)], cwd=UI, env=ENV,
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError("Disposable server exited; inspect tests/_artifacts/board-size-server.log")
                try:
                    with urlopen(base + "/api/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError("Disposable server did not become healthy")
            print(f"Disposable browser server: {base}", flush=True)
            return subprocess.run([node, "node_modules/@playwright/test/cli.js", "test", "--config",
                                   "tests/browser/playwright.config.ts", "board-size.spec.ts"],
                                  cwd=UI, env=dict(ENV, BOMBERMAN_BASE_URL=base)).returncode
        finally:
            server.terminate()
            try:
                server.wait(timeout=8)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    raise SystemExit(main())
