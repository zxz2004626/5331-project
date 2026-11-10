"""Compatibility entry point; use python -m graphbench pdf."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphbench.workflows.build_pdf_report import main

if __name__ == "__main__":
    raise SystemExit(main())
