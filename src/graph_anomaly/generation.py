"""G0-G3 anomaly sampling, timestamp batching, and cache generation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from .config import DATASETS, GENERATORS, ROOT, SEEDS, SENSITIVITIES
from .data import code_hash, load_dataset, write_data_manifest
from .features import GraphHistory, feature_groups


def scenario_config(scenario):
    cfg = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    for key, val in cfg["scenarios"][scenario].items():
        cfg["history" if key == "window" else "injection"][key] = val
    cfg["scenario"] = scenario
    return cfg


def injection_schedule(t, split, valid, fraction, seed):
    """Exactly the declared mixed-data prior in each split, up to integer rounding."""
    rng = np.random.default_rng(seed)
    refs = []
    for s in range(3):
        indices = np.flatnonzero((split == s) & valid)
        count = int(round(len(indices) * fraction / (1 - fraction)))
        if count and not len(indices):
            raise ValueError("empty split")
        refs.extend(rng.choice(indices, size=count, replace=True).tolist())
    return sorted(refs, key=lambda i: (int(t[i]), i))


def _profile_vector(hist, t, u, v):
    p = hist.profile(t, int(u), int(v))
    return np.log1p([p["u_degree"], p["v_degree"], p["u_weighted_degree"], p["v_weighted_degree"]])


def _endpoint_vectors(hist, pairs):
    a = np.asarray(pairs, dtype=np.int64).copy()
    if not hist.bipartite:
        swap = hist.degree[a[:, 0]] > hist.degree[a[:, 1]]
        a[swap] = a[swap, ::-1]
    return np.log1p(
        np.column_stack(
            (
                hist.degree[a[:, 0]],
                hist.degree[a[:, 1]],
                hist.strength[a[:, 0]],
                hist.strength[a[:, 1]],
            )
        )
    )


def _local_vector(hist, u, v):
    return np.log1p(list(hist.local_features(int(u), int(v)).values()))


def _temporal_vector(hist, t, u, v):
    d = hist.temporal_features(t, int(u), int(v))
    cap = np.log1p(30 * hist.recent_seconds)
    return np.array(
        [
            np.log1p(d["u_recent_count"]),
            np.log1p(d["v_recent_count"]),
            d["u_recency_log"] / cap,
            d["v_recency_log"] / cap,
            d["u_active"],
            d["v_active"],
        ]
    )


def choose_edge(hist, t, ru, rv, generator, rng, all_pairs, injected, cfg, n_nodes, partition):
    p = cfg["injection"]
    pool = []
    pool_set = set()
    rejected = Counter()
    pool_goal = 1 if generator == "G0" else p["legal_candidate_pool"]
    draws = 0
    while len(pool) < pool_goal and draws < p["max_draws"]:
        draws += 1
        if partition is None:
            u, v = rng.integers(0, n_nodes, size=2).tolist()
            if u == v:
                rejected["self_loop"] += 1
                continue
            u, v = min(u, v), max(u, v)
        else:
            u = int(rng.integers(0, partition))
            v = int(rng.integers(partition, n_nodes))
        pair = (u, v)
        if pair in (all_pairs if p["absence"] == "global" else hist.seen_pairs):
            rejected["observed_pair"] += 1
            continue
        # For past-only condition current normal pairs remain legal by definition;
        # prevent an exact same-timestamp labelled duplicate via caller exclusions.
        if pair in injected:
            rejected["injected_duplicate"] += 1
            continue
        if pair in pool_set:
            rejected["candidate_duplicate"] += 1
            continue
        pool.append(pair)
        pool_set.add(pair)
    if len(pool) != pool_goal:
        raise RuntimeError(
            f"support-limited {generator}: {len(pool)}/{pool_goal} after {draws} draws"
        )
    ref_endpoint = _profile_vector(hist, t, ru, rv)
    endpoint_dist = np.sqrt(np.mean((_endpoint_vectors(hist, pool) - ref_endpoint) ** 2, axis=1))
    ep_keep = np.argsort(endpoint_dist, kind="stable")[: p["endpoint_keep"]]
    if generator == "G0":
        chosen = 0
        local_ids = ep_keep
    elif generator == "G1":
        chosen = int(rng.choice(ep_keep))
        local_ids = ep_keep
    else:
        ref_local = _local_vector(hist, ru, rv)
        ld = np.array(
            [np.sqrt(np.mean((_local_vector(hist, *pool[i]) - ref_local) ** 2)) for i in ep_keep]
        )
        # Random tie-break to avoid unintentionally retaining endpoint-distance order.
        tie = rng.random(len(ep_keep))
        local_ids = ep_keep[np.lexsort((tie, ld))[: p["local_keep"]]]
        if generator == "G2":
            chosen = int(rng.choice(local_ids))
        else:
            ref_temp = _temporal_vector(hist, t, ru, rv)
            td = np.array(
                [
                    np.sqrt(np.mean((_temporal_vector(hist, t, *pool[i]) - ref_temp) ** 2))
                    for i in local_ids
                ]
            )
            chosen = int(local_ids[np.lexsort((rng.random(len(local_ids)), td))[0]])
    u, v = pool[chosen]
    ep = float(endpoint_dist[chosen])
    ld = float(np.sqrt(np.mean((_local_vector(hist, u, v) - _local_vector(hist, ru, rv)) ** 2)))
    td = float(
        np.sqrt(np.mean((_temporal_vector(hist, t, u, v) - _temporal_vector(hist, t, ru, rv)) ** 2))
    )
    return (
        u,
        v,
        {
            "candidate_pool_size": len(pool),
            "draws": draws,
            "rejections": dict(rejected),
            "partition_violations": 0,
            "endpoint_pool_size": len(ep_keep),
            "local_pool_size": len(local_ids),
            "endpoint_distance": ep,
            "local_distance": ld,
            "temporal_distance": td,
            "endpoint_best_distance": float(endpoint_dist.min()),
            "endpoint_top16_max_distance": float(endpoint_dist[ep_keep].max()),
            "endpoint_distance_over_log2": bool(ep > np.log(2)),
            "status": "valid",
        },
    )


def diagnostics(X, y, split, names, groups):
    rows = []
    for s in range(3):
        for j, name in enumerate(names):
            a = X[(split == s) & (y == 0), j].astype(float)
            b = X[(split == s) & (y == 1), j].astype(float)
            if not len(a) or not len(b):
                continue
            pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            smd = (
                (b.mean() - a.mean()) / pooled
                if pooled > 1e-12
                else (0.0 if b.mean() == a.mean() else None)
            )
            group = next(k for k, vals in groups.items() if name in vals)
            rows.append(
                {
                    "split": s,
                    "feature": name,
                    "group": group,
                    "normal_n": len(a),
                    "anomaly_n": len(b),
                    "normal_mean": float(a.mean()),
                    "anomaly_mean": float(b.mean()),
                    "smd": smd,
                    "ks": float(ks_2samp(a, b).statistic),
                    "wasserstein": float(wasserstein_distance(a, b)),
                    "wasserstein_normal_std": float(
                        wasserstein_distance(a, b) / max(a.std(), 1e-9)
                    ),
                }
            )
    return rows


class StreamBuilder:
    """Own the frozen history, reserved pairs, and output arrays for one generator run."""

    def __init__(self, cfg, t, u, v, split, meta, seed, generator):
        """Allocate the schedule, history, and fixed-size feature arrays."""
        self.cfg, self.seed, self.generator = (cfg, seed, generator)
        self.t, self.u, self.v, self.split, self.meta = (t, u, v, split, meta)
        hc = self.cfg["history"]
        self.hist = GraphHistory(
            self.meta["n_nodes"],
            self.meta["bipartite"],
            self.meta["partition"],
            hc["window"],
            hc["recent_seconds"],
        )
        self.warmup_t = int(self.t[min(hc["warmup_observed_events"] - 1, len(self.t) - 1)])
        self.valid = self.t > self.warmup_t
        self.refs = injection_schedule(
            self.t, self.split, self.valid, self.cfg["injection"]["anomaly_fraction"], self.seed
        )
        self.scheduled = {}
        for ordinal, idx in enumerate(self.refs):
            self.scheduled.setdefault(int(self.t[idx]), []).append((ordinal, idx))
        self.all_pairs = set(zip(self.u.tolist(), self.v.tolist()))
        self.injected = set()
        self.groups = feature_groups(self.meta["bipartite"])
        self.names = sum([list(self.groups[k]) for k in ("tgf", "local", "temporal")], [])
        self.n = int(self.valid.sum()) + len(self.refs)
        self.X = np.empty((self.n, len(self.names)), np.float32)
        self.ts = np.empty(self.n, np.int64)
        self.us = np.empty(self.n, np.int32)
        self.vs = np.empty(self.n, np.int32)
        self.ys = np.empty(self.n, np.int8)
        self.ss = np.empty(self.n, np.int8)
        self.histories = np.empty(self.n, np.int64)
        self.audit = []
        self.row = 0

    def _store(self, x, y, label, s):
        """Extract features without inserting the scored event."""
        values = self.hist.extract(self.now, int(x), int(y))
        self.X[self.row] = [values[name] for name in self.names]
        self.ts[self.row] = self.now
        self.us[self.row] = x
        self.vs[self.row] = y
        self.ys[self.row] = label
        self.ss[self.row] = s
        self.histories[self.row] = self.last_before
        self.row += 1

    def _inject(self, ordinal, ref, same_time_pairs):
        """Reserve a legal candidate and record its frozen historical context."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, ordinal, 93017]))
        x, y, detail = choose_edge(
            self.hist,
            self.now,
            int(self.u[ref]),
            int(self.v[ref]),
            self.generator,
            rng,
            self.all_pairs,
            self.injected | same_time_pairs,
            self.cfg,
            len(self.hist.degree),
            self.meta["partition"],
        )
        assert (x, y) not in self.injected and (x, y) not in same_time_pairs
        if self.cfg["injection"]["absence"] == "global":
            assert (x, y) not in self.all_pairs
        else:
            assert (x, y) not in self.hist.seen_pairs
        self._store(x, y, 1, self.split[ref])
        self.pending.append((x, y))
        self.injected.add((x, y))
        detail.update(
            {
                "t": self.now,
                "u": x,
                "v": y,
                "reference_index": int(ref),
                "reference_u": int(self.u[ref]),
                "reference_v": int(self.v[ref]),
                "split": int(self.split[ref]),
                "history_max_timestamp": self.last_before,
            }
        )
        detail["reference_endpoint_context"] = _profile_vector(
            self.hist, self.now, self.u[ref], self.v[ref]
        ).tolist()
        detail["reference_local_context"] = _local_vector(
            self.hist, self.u[ref], self.v[ref]
        ).tolist()
        detail["reference_temporal_context"] = _temporal_vector(
            self.hist, self.now, self.u[ref], self.v[ref]
        ).tolist()
        self.audit.append(detail)

    def build(self):
        """Score all equal-time events before inserting the observed and injected batch."""
        starts = np.r_[0, np.flatnonzero(np.diff(self.t)) + 1, len(self.t)]
        for k in range(len(starts) - 1):
            a, b = (int(starts[k]), int(starts[k + 1]))
            self.now = int(self.t[a])
            self.hist.advance(self.now)
            self.pending = []
            self.last_before = int(self.t[a - 1]) if a else -1
            if self.valid[a]:
                for i in range(a, b):
                    self._store(self.u[i], self.v[i], 0, self.split[i])
            same_time_pairs = set(zip(self.u[a:b].tolist(), self.v[a:b].tolist()))
            for ordinal, ref in self.scheduled.get(self.now, []):
                self._inject(ordinal, ref, same_time_pairs)
            for i in range(a, b):
                self.hist.add(self.now, int(self.u[i]), int(self.v[i]))
            for x, y in self.pending:
                self.hist.add(self.now, x, y)
        assert self.row == self.n and np.all(self.histories < self.ts) and np.isfinite(self.X).all()
        assert len(self.injected) == len(self.refs)
        for s in range(3):
            assert ((self.ys == 1) & (self.ss == s)).sum() == sum(
                (self.split[r] == s for r in self.refs)
            )


def _generation_manifest(state, signature, start):
    """Summarize quotas, validity checks, and matching diagnostics for the completed cache."""
    rejected = Counter()
    for d in state.audit:
        rejected.update(d["rejections"])
    manifest = {
        "status": "complete",
        "signature": signature,
        "dataset": state.meta,
        "seed": state.seed,
        "generator": state.generator,
        "config": state.cfg,
        "warmup_last_timestamp": state.warmup_t,
        "observed_scored": int(state.valid.sum()),
        "anomalies": len(state.refs),
        "split_counts": [
            {
                "normal": int(((state.ss == s) & (state.ys == 0)).sum()),
                "anomaly": int(((state.ss == s) & (state.ys == 1)).sum()),
            }
            for s in range(3)
        ],
        "feature_names": state.names,
        "feature_groups": state.groups,
        "code_sha256": code_hash(),
        "candidate_draws": sum((d["draws"] for d in state.audit)),
        "rejections": dict(rejected),
        "matching_mean_distances": {
            k: float(np.mean([d[k] for d in state.audit]))
            for k in ("endpoint_distance", "local_distance", "temporal_distance")
        },
        "endpoint_distance_over_log2_fraction": float(
            np.mean([d["endpoint_distance_over_log2"] for d in state.audit])
        ),
        "runtime_seconds": time.perf_counter() - start,
        "python": platform.python_version(),
        "checks": {
            "finite_features": True,
            "strict_prearrival": True,
            "requested_count_met": True,
            "unique_injected_pairs": True,
            "graph_constraints": True,
        },
    }
    return manifest


def _persist_cache(dataset, state, out, path, signature, start):
    """Save feature arrays, event audit records, diagnostics, and their manifest."""
    manifest_path = path.with_suffix(".json")
    scenario = state.cfg["scenario"]
    np.savez_compressed(
        path,
        X=state.X,
        names=np.asarray(state.names),
        t=state.ts,
        u=state.us,
        v=state.vs,
        y=state.ys,
        split=state.ss,
        history_max_timestamp=state.histories,
    )
    with gzip.open(out / f"{state.generator}_audit.jsonl.gz", "wt", encoding="utf-8") as f:
        for d in state.audit:
            f.write(json.dumps(d, separators=(",", ":")) + "\n")
    pd.DataFrame(diagnostics(state.X, state.ys, state.ss, state.names, state.groups)).to_csv(
        out / f"{state.generator}_diagnostics.csv", index=False
    )
    manifest = _generation_manifest(state, signature, start)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"complete {scenario}/{dataset}/{state.seed}/{state.generator}: {state.n} rows {manifest['runtime_seconds']:.1f}s",
        flush=True,
    )
    return manifest


def build_cache(dataset, seed, generator, scenario="main", force=False):
    cfg = scenario_config(scenario)
    t, u, v, split, meta = load_dataset(dataset)
    out = ROOT / f"results/cache/{scenario}/{dataset}/seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{generator}.npz"
    manifest_path = path.with_suffix(".json")
    signature = hashlib.sha256(
        json.dumps(
            {
                "cfg": cfg,
                "seed": seed,
                "generator": generator,
                "data": meta["sha256"],
                "code": code_hash(),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if path.exists() and manifest_path.exists() and (not force):
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("signature") == signature and old.get("status") == "complete":
            print(f"cached {scenario}/{dataset}/{seed}/{generator}", flush=True)
            return old
        raise RuntimeError(f"Stale cache {path}; use --force to rebuild explicitly")
    start = time.perf_counter()
    builder = StreamBuilder(cfg, t, u, v, split, meta, seed, generator)
    builder.build()
    return _persist_cache(dataset, builder, out, path, signature, start)


for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


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
