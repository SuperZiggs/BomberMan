"""Run all web-project checks against a disposable loopback server."""
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


def run(args, env=ENV):
    print("\n> " + " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=UI, env=env, check=True)


def main():
    node = shutil.which("node")
    if not node:
        raise SystemExit("Cần Node.js >=22.18 và npm. Chạy setup.cmd trước.")
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests/python", "-p", "test_*.py", "-v"])
    run([node, "--experimental-strip-types", "--test", "tests/client.test.ts"])
    run([node, "node_modules/typescript/bin/tsc", "--noEmit"])
    run([node, "node_modules/vite/bin/vite.js", "build"])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    (UI / "artifacts").mkdir(exist_ok=True)
    with (UI / "artifacts/verify-server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "run.py", "--port", str(port)], cwd=UI, env=ENV,
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError("Server kiểm thử không khởi động được; xem artifacts/verify-server.log")
                try:
                    with urlopen(base + "/api/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError("Hết thời gian chờ server local")
            run([node, "node_modules/@playwright/test/cli.js", "test", "--config", "tests/browser/playwright.config.ts"],
                dict(ENV, BOMBERMAN_BASE_URL=base))
        finally:
            server.terminate()
            try:
                server.wait(timeout=8)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    print("\nPASS: Python contracts/assets, browser input, typecheck/build and browser journeys.")


if __name__ == "__main__":
    main()
