"""Use saved scientific summaries when bulky regeneration caches are absent."""

import json
from pathlib import Path


def load_split_counts(root: Path, dataset: str):
    """Read fresh metadata, or the same counts recorded in the historical audit."""
    manifest = root / f"results/cache/main/{dataset}/seed_11/G0.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))["split_counts"]
    audit = json.loads((root / "results/validation/final_audit.json").read_text(encoding="utf-8"))
    matches = [
        row
        for row in audit["cache_checksums_and_counts"]
        if row["scenario"] == "main"
        and row["dataset"] == dataset
        and row["seed"] == 11
        and row["generator"] == "G0"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one audited split-count record for {dataset}")
    return matches[0]["split_counts"]


def load_diagnostics(root: Path):
    """Prefer complete cache diagnostics; otherwise read checked-in summaries.

    Summary reuse supports moving a results-only checkout. It never claims that
    event-level diagnostics were recomputed without their source feature caches.
    """
    import pandas as pd

    from .constants import DATASETS, GENERATORS, SEEDS

    manifests = [
        root / f"results/cache/main/{d}/seed_{seed}/{g}.json"
        for d in DATASETS
        for seed in SEEDS
        for g in GENERATORS
    ]
    if not all(path.exists() for path in manifests):
        tables = root / "results/report/tables"
        print("Using saved diagnostic summaries; generator caches are not complete.")
        return (
            pd.read_csv(tables / "diagnostic_group_statistics.csv"),
            pd.read_csv(tables / "matching_statistics.csv"),
        )
    rows = []
    match = []
    endpoint = ["u_degree", "v_degree", "u_weighted_degree", "v_weighted_degree"]
    temporal = [
        "u_recent_count",
        "v_recent_count",
        "u_recency_log",
        "v_recency_log",
        "u_active",
        "v_active",
    ]
    for d in DATASETS:
        for seed in [11, 23, 37, 51, 71]:
            for g in GENERATORS:
                p = root / f"results/cache/main/{d}/seed_{seed}/{g}"
                meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
                x = pd.read_csv(p.parent / f"{g}_diagnostics.csv")
                for name, features in [
                    ("endpoint", endpoint),
                    ("local", meta["feature_groups"]["local"]),
                    ("temporal", temporal),
                    ("prior_pair", ["pair_seen_before"]),
                ]:
                    z = x[(x.split == 2) & x.feature.isin(features)]
                    rows.append(
                        {
                            "dataset": d,
                            "generator": g,
                            "seed": seed,
                            "group": name,
                            "mean_ks": z.ks.mean(),
                            "mean_abs_smd": z.smd.abs().mean(),
                            "mean_wasserstein_normal_std": z.wasserstein_normal_std.mean(),
                        }
                    )
                match.append(
                    {
                        "dataset": d,
                        "generator": g,
                        "seed": seed,
                        **meta["matching_mean_distances"],
                        "endpoint_distance_over_log2_fraction": meta[
                            "endpoint_distance_over_log2_fraction"
                        ],
                        "candidate_draws": meta["candidate_draws"],
                        "legal_acceptance": sum(
                            __import__("gzip")
                            .open(p.parent / f"{g}_audit.jsonl.gz", "rt")
                            .read()
                            .count('"candidate_pool_size":' + str(i))
                            * i
                            for i in [1, 128]
                        )
                        / meta["candidate_draws"],
                    }
                )
    diag = pd.DataFrame(rows)
    return diag, pd.DataFrame(match)
