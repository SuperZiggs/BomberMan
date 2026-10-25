"""Local browser presentation for the existing Bomberman Python engine."""

from pathlib import Path
import sys

UI_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = UI_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
