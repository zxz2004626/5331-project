"""Causal, timestamp-batched injection with auditable finite candidate pools."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from .data import ROOT, code_hash, load_dataset
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
    if path.exists() and manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("signature") == signature and old.get("status") == "complete":
            print(f"cached {scenario}/{dataset}/{seed}/{generator}", flush=True)
            return old
        raise RuntimeError(f"Stale cache {path}; use --force to rebuild explicitly")
    start = time.perf_counter()
    hc = cfg["history"]
    hist = GraphHistory(
        meta["n_nodes"], meta["bipartite"], meta["partition"], hc["window"], hc["recent_seconds"]
    )
    warmup_t = int(t[min(hc["warmup_observed_events"] - 1, len(t) - 1)])
    valid = t > warmup_t  # whole warmup timestamp stays excluded
    refs = injection_schedule(t, split, valid, cfg["injection"]["anomaly_fraction"], seed)
    scheduled = {}
    for ordinal, idx in enumerate(refs):
        scheduled.setdefault(int(t[idx]), []).append((ordinal, idx))
    all_pairs = set(zip(u.tolist(), v.tolist()))
    injected = set()
    groups = feature_groups(meta["bipartite"])
    names = sum([list(groups[k]) for k in ("tgf", "local", "temporal")], [])
    n = int(valid.sum()) + len(refs)
    X = np.empty((n, len(names)), np.float32)
    ts = np.empty(n, np.int64)
    us = np.empty(n, np.int32)
    vs = np.empty(n, np.int32)
    ys = np.empty(n, np.int8)
    ss = np.empty(n, np.int8)
    histories = np.empty(n, np.int64)
    audit = []
    row = 0
    starts = np.r_[0, np.flatnonzero(np.diff(t)) + 1, len(t)]
    for k in range(len(starts) - 1):
        a, b = int(starts[k]), int(starts[k + 1])
        now = int(t[a])
        hist.advance(now)
        pending = []
        last_before = int(t[a - 1]) if a else -1

        def store_edge(x, y, label, s):
            nonlocal row
            values = hist.extract(now, int(x), int(y))
            X[row] = [values[name] for name in names]
            ts[row] = now
            us[row] = x
            vs[row] = y
            ys[row] = label
            ss[row] = s
            histories[row] = last_before
            row += 1

        if valid[a]:
            for i in range(a, b):
                store_edge(u[i], v[i], 0, split[i])
        same_time_pairs = set(zip(u[a:b].tolist(), v[a:b].tolist()))
        for ordinal, ref in scheduled.get(now, []):
            # Common candidate random stream per anchor across generators;
            # legality can differ once mixed histories differ in past-only sensitivity.
            rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal, 93017]))
            x, y, detail = choose_edge(
                hist,
                now,
                int(u[ref]),
                int(v[ref]),
                generator,
                rng,
                all_pairs,
                injected | same_time_pairs,
                cfg,
                len(hist.degree),
                meta["partition"],
            )
            assert (x, y) not in injected and (x, y) not in same_time_pairs
            if cfg["injection"]["absence"] == "global":
                assert (x, y) not in all_pairs
            else:
                assert (x, y) not in hist.seen_pairs
            store_edge(x, y, 1, split[ref])
            pending.append((x, y))
            injected.add((x, y))
            detail.update(
                {
                    "t": now,
                    "u": x,
                    "v": y,
                    "reference_index": int(ref),
                    "reference_u": int(u[ref]),
                    "reference_v": int(v[ref]),
                    "split": int(split[ref]),
                    "history_max_timestamp": last_before,
                }
            )
            # Explicit context stored to reconstruct why a candidate was chosen.
            detail["reference_endpoint_context"] = _profile_vector(
                hist, now, u[ref], v[ref]
            ).tolist()
            detail["reference_local_context"] = _local_vector(hist, u[ref], v[ref]).tolist()
            detail["reference_temporal_context"] = _temporal_vector(
                hist, now, u[ref], v[ref]
            ).tolist()
            audit.append(detail)
        # Every edge at now was scored before ANY now edge entered either history.
        for i in range(a, b):
            hist.add(now, int(u[i]), int(v[i]))
        for x, y in pending:
            hist.add(now, x, y)
    assert row == n and np.all(histories < ts) and np.isfinite(X).all()
    assert len(injected) == len(refs)
    for s in range(3):
        assert ((ys == 1) & (ss == s)).sum() == sum(split[r] == s for r in refs)
    np.savez_compressed(
        path,
        X=X,
        names=np.asarray(names),
        t=ts,
        u=us,
        v=vs,
        y=ys,
        split=ss,
        history_max_timestamp=histories,
    )
    with gzip.open(out / f"{generator}_audit.jsonl.gz", "wt", encoding="utf-8") as f:
        for d in audit:
            f.write(json.dumps(d, separators=(",", ":")) + "\n")
    pd.DataFrame(diagnostics(X, ys, ss, names, groups)).to_csv(
        out / f"{generator}_diagnostics.csv", index=False
    )
    rejected = Counter()
    for d in audit:
        rejected.update(d["rejections"])
    manifest = {
        "status": "complete",
        "signature": signature,
        "dataset": meta,
        "seed": seed,
        "generator": generator,
        "config": cfg,
        "warmup_last_timestamp": warmup_t,
        "observed_scored": int(valid.sum()),
        "anomalies": len(refs),
        "split_counts": [
            {
                "normal": int(((ss == s) & (ys == 0)).sum()),
                "anomaly": int(((ss == s) & (ys == 1)).sum()),
            }
            for s in range(3)
        ],
        "feature_names": names,
        "feature_groups": groups,
        "code_sha256": code_hash(),
        "candidate_draws": sum(d["draws"] for d in audit),
        "rejections": dict(rejected),
        "matching_mean_distances": {
            k: float(np.mean([d[k] for d in audit]))
            for k in ("endpoint_distance", "local_distance", "temporal_distance")
        },
        "endpoint_distance_over_log2_fraction": float(
            np.mean([d["endpoint_distance_over_log2"] for d in audit])
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
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"complete {scenario}/{dataset}/{seed}/{generator}: {n} rows {manifest['runtime_seconds']:.1f}s",
        flush=True,
    )
    return manifest
