"""Experiment settings, identifiers, and relocatable project paths."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Prefer GRAPH_ANOMALY_ROOT; otherwise find the source checkout or working tree."""
    configured = os.environ.get("GRAPH_ANOMALY_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "configs/experiment.json").is_file():
            raise ValueError(f"GRAPH_ANOMALY_ROOT has no configs/experiment.json: {root}")
        return root
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for candidate in (start, *start.parents):
            if (candidate / "configs/experiment.json").is_file():
                return candidate
    raise RuntimeError(
        "Project data not found. Set GRAPH_ANOMALY_ROOT or use graph_anomaly --root PATH."
    )


ROOT = project_root()

DATASETS = ["movielens", "ucimessages"]

SEEDS = [11, 23, 37, 51, 71]

GENERATORS = ["G0", "G1", "G2", "G3"]

CORE_FEATURES = ["tgf", "extended"]

ABLATIONS = ["local", "temporal", "tgf_local", "tgf_temporal", "local_temporal"]

SENSITIVITIES = ["ratio01", "ratio10", "window500", "window2000", "past_only"]

SCHEMA_VERSION = 1
