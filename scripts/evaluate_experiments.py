"""Run resumable detector fits on already-built graphbench feature caches."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Set limits before importing NumPy/scikit-learn or linked numerical libraries.
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphbench.evaluate import (ABLATIONS, CORE_FEATURES, GENERATORS, SEEDS,
                                 aggregate_results, run_evaluations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scenario", default="main")
    parser.add_argument("--datasets", nargs="+", default=["movielens", "ucimessages"],
                        choices=["movielens", "ucimessages"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--models", nargs="+", default=["RF", "LR"], choices=["RF", "LR"])
    parser.add_argument("--generators", nargs="+", default=GENERATORS, choices=GENERATORS)
    parser.add_argument("--features", nargs="+", default=CORE_FEATURES,
                        choices=CORE_FEATURES + ABLATIONS)
    parser.add_argument("--ablations", action="store_true",
                        help="Add five RF-only feature-group experiments with diagonal tests")
    parser.add_argument("--same-generator-only", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    if args.aggregate_only:
        rows, summary = aggregate_results(args.root / "results" / "evaluation" / args.scenario)
        print(json.dumps({"rows": len(rows), "summary_rows": len(summary)}))
        return 0
    counts = run_evaluations(args.root, args.scenario, args.datasets, args.seeds,
                             args.models, args.features, args.ablations, args.n_jobs,
                             not args.same_generator_only, args.generators)
    print(json.dumps(counts, sort_keys=True), flush=True)
    return 1 if counts["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
