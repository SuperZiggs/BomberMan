import json
import os
from pathlib import Path
import sys

def option(name):
    index = sys.argv.index(name)
    return sys.argv[index + 1]

kind = Path(__file__).stem
log_path = Path(os.environ["ACCEPTANCE_CALL_LOG"])
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"kind": kind, "argv": sys.argv[1:]}) + "\n")
print(f"FAKE_SUBPROCESS_START:{kind}", flush=True)

if kind == "train_actor_critic":
    exit_code = int(os.environ.get("FAKE_TRAIN_EXIT", "0"))
    if exit_code == 0 and os.environ.get("FAKE_CREATE_BEST", "1") == "1":
        output = Path(option("--checkpoint_dir"))
        output.mkdir(parents=True, exist_ok=True)
        (output / "best_phaseC.pt").write_bytes(b"best")
    raise SystemExit(exit_code)

exit_code = int(os.environ.get("FAKE_EVAL_EXIT", "0"))
if exit_code == 0 and os.environ.get("FAKE_EVAL_ARTIFACT", "write") == "write":
    output = Path(option("--output"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(os.environ["FAKE_EVAL_PAYLOAD"], encoding="utf-8")
raise SystemExit(exit_code)