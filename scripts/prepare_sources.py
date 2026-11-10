"""Script entry point; use python -m graphbench prepare."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphbench.workflows.prepare_sources import main

if __name__ == "__main__":
    raise SystemExit(main())
