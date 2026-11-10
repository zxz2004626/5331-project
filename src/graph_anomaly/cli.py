"""Command dispatch and full experiment orchestration."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

COMMANDS = {
    "prepare": ("official", "prepare_main"),
    "generate": ("generation", "main"),
    "evaluate": ("evaluation", "main"),
    "no-pair": ("controls", "main"),
    "official": ("official", "main"),
    "summarize": ("results", "main"),
    "verify": ("verify", "main"),
    "run": ("cli", "pipeline_main"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="COMP5331 graph anomaly benchmark")
    parser.add_argument("--root", type=Path, help="Project checkout (or set GRAPH_ANOMALY_ROOT)")
    parser.add_argument(
        "command", choices=COMMANDS, help="Workflow to execute; use COMMAND --help for options"
    )
    parser.add_argument("options", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.root is not None:
        os.environ["GRAPH_ANOMALY_ROOT"] = str(args.root.expanduser().resolve())
    # Limit numerical libraries before importing any NumPy/scikit-learn modules.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    module = importlib.import_module(f"graph_anomaly.{COMMANDS[args.command][0]}")
    previous = sys.argv
    try:
        sys.argv = [f"graph_anomaly {args.command}", *args.options]
        return getattr(module, COMMANDS[args.command][1])()
    finally:
        sys.argv = previous


def pipeline_steps(workers=1, model_jobs=2, no_pair_jobs=1, skip_official=False):
    from .config import SENSITIVITIES

    steps = [["prepare"]]
    if not skip_official:
        # These flags select mutually exclusive modes in the original wrapper.
        steps += [
            ["official"],
            ["official", "--paired-timing"],
            ["official", "--fresh-pipeline"],
            ["official", "--verify-output"],
        ]
    steps += [
        ["generate", "--workers", str(workers)],
        ["evaluate", "--ablations", "--n-jobs", str(model_jobs)],
    ]
    for scenario in SENSITIVITIES:
        steps += [
            [
                "generate",
                "--scenario",
                scenario,
                "--generators",
                "G0",
                "G3",
                "--workers",
                str(workers),
            ],
            [
                "evaluate",
                "--scenario",
                scenario,
                "--generators",
                "G0",
                "G3",
                "--models",
                "RF",
                "--features",
                "tgf",
                "extended",
                "--same-generator-only",
                "--n-jobs",
                str(model_jobs),
            ],
        ]
    steps += [["no-pair", "--n-jobs", str(no_pair_jobs)], ["verify"], ["summarize"]]
    return steps


def pipeline_main():
    from .config import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-jobs", type=int, default=2)
    parser.add_argument("--no-pair-jobs", type=int, default=1)
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help="Skip official reproduction; its saved results must exist for reports",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without downloads or computation"
    )
    args = parser.parse_args()
    if min(args.workers, args.model_jobs, args.no_pair_jobs) < 1:
        parser.error("Worker counts must be positive")
    steps = pipeline_steps(args.workers, args.model_jobs, args.no_pair_jobs, args.skip_official)
    environment = os.environ.copy()
    environment["GRAPH_ANOMALY_ROOT"] = str(ROOT)
    source = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [source, environment.get("PYTHONPATH")])
    )
    for number, step in enumerate(steps, 1):
        command = [sys.executable, "-m", "graph_anomaly", *step]
        print(f"[{number}/{len(steps)}] {subprocess.list2cmdline(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0
