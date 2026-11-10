"""Locate project inputs without depending on a user's current directory."""

import os
from pathlib import Path


def project_root() -> Path:
    """Prefer GRAPHBENCH_ROOT; otherwise find the source checkout or working tree."""
    configured = os.environ.get("GRAPHBENCH_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "configs/experiment.json").is_file():
            raise ValueError(f"GRAPHBENCH_ROOT has no configs/experiment.json: {root}")
        return root
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for candidate in (start, *start.parents):
            if (candidate / "configs/experiment.json").is_file():
                return candidate
    raise RuntimeError("Project data not found. Set GRAPHBENCH_ROOT or use graphbench --root PATH.")


ROOT = project_root()
