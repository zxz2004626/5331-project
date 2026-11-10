"""Build full timestamp-safe experimental streams and feature caches."""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

from graphbench.constants import DATASETS, GENERATORS, SEEDS, SENSITIVITIES
from graphbench.data import write_data_manifest
from graphbench.generators import build_cache
from graphbench.paths import ROOT


def recorded_build(job):
    try:
        return build_cache(*job)
    except Exception as exc:
        directory = ROOT / "results/failures"
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "job": job,
            "status": "failed",
            "error": repr(exc),
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        (directory / ("_".join(map(str, job[:4])) + ".json")).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="main", choices=["main", *SENSITIVITIES])
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--generators", nargs="+", default=GENERATORS, choices=GENERATORS)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    write_data_manifest()
    jobs = [
        (d, s, g, args.scenario, args.force)
        for d in args.datasets
        for s in args.seeds
        for g in args.generators
    ]
    if args.workers == 1:
        for job in jobs:
            recorded_build(job)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(recorded_build, job): job for job in jobs}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"FAILED {futures[f]}: {e}", flush=True)
                    raise


if __name__ == "__main__":
    main()
