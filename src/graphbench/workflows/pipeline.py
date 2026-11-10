"""Cross-platform orchestration, with a reviewable dry-run plan."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from graphbench.constants import SENSITIVITIES
from graphbench.paths import ROOT


def pipeline_steps(workers=1, model_jobs=2, no_pair_jobs=1, skip_official=False):
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
    steps += [["no-pair", "--n-jobs", str(no_pair_jobs)], ["verify"], ["report"], ["write-report"]]
    return steps


def main():
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
    environment["GRAPHBENCH_ROOT"] = str(ROOT)
    source = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [source, environment.get("PYTHONPATH")])
    )
    for number, step in enumerate(steps, 1):
        command = [sys.executable, "-m", "graphbench", *step]
        print(f"[{number}/{len(steps)}] {subprocess.list2cmdline(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0
