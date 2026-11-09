"""Build full experimental streams/features. Run from repository root."""
import os
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import argparse
import json
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from graphbench.data import write_data_manifest
from graphbench.generate import build_cache

def recorded_build(job):
    try:return build_cache(*job)
    except Exception as exc:
        directory=Path(__file__).resolve().parents[1]/"results/failures"
        directory.mkdir(parents=True,exist_ok=True)
        record={"job":job,"status":"failed","error":repr(exc),"utc":datetime.now(timezone.utc).isoformat()}
        (directory/("_".join(map(str,job[:4]))+".json")).write_text(json.dumps(record,indent=2),encoding="utf-8")
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument("--scenario",default="main");p.add_argument("--datasets",nargs="+",default=["movielens","ucimessages"])
    p.add_argument("--seeds",nargs="+",type=int,default=[11,23,37,51,71]);p.add_argument("--generators",nargs="+",default=["G0","G1","G2","G3"])
    p.add_argument("--workers",type=int,default=2);p.add_argument("--force",action="store_true");args=p.parse_args();write_data_manifest()
    jobs=[(d,s,g,args.scenario,args.force) for d in args.datasets for s in args.seeds for g in args.generators]
    if args.workers==1:
        for job in jobs:recorded_build(job)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(recorded_build,job):job for job in jobs}
            for f in as_completed(futures):
                try:f.result()
                except Exception as e:print(f"FAILED {futures[f]}: {e}",flush=True);raise

if __name__=="__main__":main()
