"""Script entry point; use python -m graphbench no-pair."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphbench.workflows.pair_novelty_control import main

if __name__ == "__main__":
    raise SystemExit(main())
