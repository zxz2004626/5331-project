"""One command dispatcher; workflow-specific options are forwarded unchanged."""

import argparse
import importlib
import os
import sys
from pathlib import Path

COMMANDS = {
    "prepare": ("prepare_sources", "Fetch and verify the pinned TGF source"),
    "generate": ("build_experiments", "Build timestamp-safe feature caches"),
    "evaluate": ("evaluate_experiments", "Fit detectors and evaluate transfer"),
    "no-pair": ("no_pair_ablation", "Run the prior-pair attribution control"),
    "official": ("official_reproduction", "Reproduce the released TGF protocol"),
    "report": ("report_results", "Regenerate tables and result figures"),
    "write-report": ("write_report", "Write the Chinese experiment summary"),
    "pdf": ("build_pdf_report", "Build the optional English experiment PDF"),
    "verify": ("verify_artifacts", "Independently audit experiment artifacts"),
    "run": ("pipeline", "Run the complete pipeline, or print its plan"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="COMP5331 graph anomaly benchmark")
    parser.add_argument("--root", type=Path, help="Project checkout (or set GRAPHBENCH_ROOT)")
    parser.add_argument(
        "command", choices=COMMANDS, help="; ".join(f"{k}: {v[1]}" for k, v in COMMANDS.items())
    )
    parser.add_argument("options", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.root is not None:
        os.environ["GRAPHBENCH_ROOT"] = str(args.root.expanduser().resolve())
    # Limit numerical libraries before importing any NumPy/scikit-learn modules.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    module = importlib.import_module(f"graphbench.workflows.{COMMANDS[args.command][0]}")
    previous = sys.argv
    try:
        sys.argv = [f"graphbench {args.command}", *args.options]
        return module.main()
    finally:
        sys.argv = previous
