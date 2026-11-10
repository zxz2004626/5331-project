"""Aggregate experiment records and generate numerical tables and figures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    _canonical,
    _digest,
    _file_hash,
    _read_json,
    _write_csv,
    artifact_path,
    evaluation_code_hash,
)
from .config import ROOT


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "scenario",
        "dataset",
        "seed",
        "source_generator",
        "target_generator",
        "model",
        "features",
        "n_train",
        "n_val",
        "n_test",
        "run_id",
        "selected_validation_average_precision",
        "normal_previously_seen_fraction",
    ]
    row = {key: record.get(key) for key in fields}
    row.update(record["metrics"])
    row.update(
        {f"novel_{key}": value for key, value in (record.get("novel_normal_metrics") or {}).items()}
    )
    row.update(record.get("times", {}))
    row["params"] = _canonical(record["params"])
    return row


def aggregate_results(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild tables, excluding successful but now-invalidated old records."""
    output = Path(output)
    root = output.parents[2]
    source_hash = evaluation_code_hash()
    cache_fingerprints: dict[Path, str | None] = {}

    def current_cache_fingerprint(record: dict[str, Any], generator: str) -> str | None:
        path = (
            root
            / "results"
            / "cache"
            / record["scenario"]
            / record["dataset"]
            / f"seed_{record['seed']}"
            / f"{generator}.npz"
        )
        if path not in cache_fingerprints:
            metadata_path = path.with_suffix(".json")
            metadata = _read_json(metadata_path)
            cache_fingerprints[path] = (
                _digest(
                    {"npz_sha256": _file_hash(path), "metadata_sha256": _file_hash(metadata_path)}
                )
                if path.exists() and metadata and metadata.get("status") == "complete"
                else None
            )
        return cache_fingerprints[path]

    records = []
    for path in sorted((output / "runs").glob("*.json")):
        record = _read_json(path)
        if not record or record.get("status") != "complete":
            continue
        if record.get("evaluation_source_sha256") != source_hash:
            continue
        if record.get("source_cache_fingerprint") != current_cache_fingerprint(
            record, record["source_generator"]
        ) or record.get("target_cache_fingerprint") != current_cache_fingerprint(
            record, record["target_generator"]
        ):
            continue
        model_metadata = _read_json(artifact_path(root, record["model_metadata_file"]))
        if (
            not model_metadata
            or model_metadata.get("model_fingerprint") != record.get("model_fingerprint")
            or not artifact_path(root, record["prediction_file"]).exists()
        ):
            continue
        records.append(record)
    rows = [_flatten_record(record) for record in records]
    if not rows and any((output / "runs").glob("*.json")):
        raise ValueError(
            "No run matches the current source and available artifacts. "
            "Existing summary tables were preserved. Rebuild in a separate checkout "
            "or use graph_anomaly summarize to read the saved historical results."
        )
    _write_csv(output / "metrics.csv", rows)
    dimensions = [
        "scenario",
        "dataset",
        "source_generator",
        "target_generator",
        "model",
        "features",
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in dimensions)].append(row)
    summary = []
    metrics = [
        "average_precision",
        "pr_auc_trapezoid",
        "roc_auc",
        "precision_at_k",
        "recall_at_k",
        "anomaly_fraction",
        "novel_average_precision",
        "novel_roc_auc",
        "novel_precision_at_k",
        "normal_previously_seen_fraction",
    ]
    for key, group in sorted(groups.items()):
        row = dict(zip(dimensions, key))
        row["n_seeds"] = len(group)
        row["seeds"] = ";".join(
            str(item["seed"]) for item in sorted(group, key=lambda v: v["seed"])
        )
        for metric in metrics:
            values = [item[metric] for item in group if item.get(metric) is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    _write_csv(output / "summary.csv", summary)
    return rows, summary


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

    from .config import DATASETS, GENERATORS, SEEDS

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


OUT = ROOT / "results/report"

FIG = OUT / "figures"

TABLE = OUT / "tables"

DS = ["movielens", "ucimessages"]

GEN = ["G0", "G1", "G2", "G3"]

LABEL = {"movielens": "MovieLens 100K", "ucimessages": "UCI Messages"}

COLORS = ["#67757b", "#377eb8", "#e69536", "#bb4b53"]


plt = None


def _configure_plotting():
    """Load the noninteractive plotting backend only for the summarize command."""
    global plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 200,
        }
    )


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def meanstd(values, digits=3):
    a = np.asarray(values, dtype=float)
    return f"{np.mean(a):.{digits}f} ± {np.std(a, ddof=1):.{digits}f}"


def load_frames():
    main = pd.read_csv(ROOT / "results/evaluation/main/metrics.csv")
    assert len(main) == 840, f"Incomplete main: {len(main)}/840"
    assert (
        len(main.drop_duplicates(["dataset", "seed", "source_generator", "model", "features"]))
        == 360
    )
    sensitivities = []
    for s in ["ratio01", "ratio10", "window500", "window2000", "past_only"]:
        d = pd.read_csv(ROOT / f"results/evaluation/{s}/metrics.csv")
        assert len(d) == 40, f"Incomplete {s}: {len(d)}/40"
        sensitivities.append(d)
    return main, pd.concat(sensitivities, ignore_index=True)


def _paired_difference_tables(core):
    """Write paired generator and feature differences without changing seed alignment."""
    paired = []
    for d in DS:
        for m in ["RF", "LR"]:
            for f in ["tgf", "extended"]:
                x = core[(core.dataset == d) & (core.model == m) & (core.features == f)]
                p = x.pivot(index="seed", columns="source_generator", values="average_precision")
                for g in GEN[1:]:
                    dif = p[g] - p.G0
                    paired.append(
                        {
                            "dataset": d,
                            "model": m,
                            "features": f,
                            "contrast": f"{g}-G0",
                            "mean": dif.mean(),
                            "std": dif.std(),
                            **{f"seed_{s}": v for s, v in dif.items()},
                        }
                    )
    paired = pd.DataFrame(paired)
    paired.to_csv(TABLE / "paired_generator_differences.csv", index=False)
    additions = []
    for (d, g, m), x in core.groupby(["dataset", "source_generator", "model"]):
        p = x.pivot(index="seed", columns="features", values="average_precision")
        dif = p.extended - p.tgf
        additions.append(
            {
                "dataset": d,
                "generator": g,
                "model": m,
                "contrast": "extended-tgf",
                "mean": dif.mean(),
                "std": dif.std(),
                **{f"seed_{s}": v for s, v in dif.items()},
            }
        )
    pd.DataFrame(additions).to_csv(TABLE / "paired_feature_differences.csv", index=False)
    return paired


def make_tables(main, sens):
    diagonal = main[main.source_generator == main.target_generator]
    core = diagonal[diagonal.features.isin(["tgf", "extended"])]
    rows = []
    for d in DS:
        for g in GEN:
            r = {"Dataset": LABEL[d], "Generator": g}
            for m in ["RF", "LR"]:
                for f in ["tgf", "extended"]:
                    x = core[
                        (core.dataset == d)
                        & (core.source_generator == g)
                        & (core.model == m)
                        & (core.features == f)
                    ]
                    assert len(x) == 5
                    r[f"{m}-{f}"] = meanstd(x.average_precision)
            rows.append(r)
    primary = pd.DataFrame(rows)
    primary.to_csv(TABLE / "main_ap.csv", index=False)
    metric_table = core.groupby(["dataset", "source_generator", "model", "features"])[
        ["average_precision", "pr_auc_trapezoid", "roc_auc", "precision_at_k", "recall_at_k"]
    ].agg(["mean", "std"])
    metric_table.to_csv(TABLE / "main_all_metrics.csv")
    paired = _paired_difference_tables(core)
    sen = sens.groupby(["scenario", "dataset", "source_generator", "features"])[
        ["average_precision", "roc_auc", "anomaly_fraction"]
    ].agg(["mean", "std"])
    sen.to_csv(TABLE / "sensitivity_metrics.csv")
    novel = (
        core[(core.model == "RF")]
        .groupby(["dataset", "source_generator", "features"])[
            [
                "average_precision",
                "novel_average_precision",
                "novel_anomaly_fraction",
                "novel_n",
                "normal_previously_seen_fraction",
            ]
        ]
        .agg(["mean", "std"])
    )
    novel.to_csv(TABLE / "novel_normal_control.csv")
    no_pair_path = ROOT / "results/no_pair/metrics.csv"
    if no_pair_path.exists():
        np_ = pd.read_csv(no_pair_path)
        full = core[(core.model == "RF") & (core.features == "extended")]
        matched = np_.merge(
            full,
            on=["dataset", "seed", "source_generator"],
            suffixes=("_no_pair", "_full"),
            validate="one_to_one",
        )
        assert len(matched) == 40
        matched["paired_ap_difference"] = (
            matched.average_precision_no_pair - matched.average_precision_full
        )
        matched["paired_novel_ap_difference"] = (
            matched.novel_average_precision_no_pair - matched.novel_average_precision_full
        )
        cols = [
            "dataset",
            "seed",
            "source_generator",
            "average_precision_full",
            "average_precision_no_pair",
            "paired_ap_difference",
            "paired_novel_ap_difference",
        ]
        matched[cols].to_csv(TABLE / "no_pair_paired_differences.csv", index=False)
        matched.groupby(["dataset", "source_generator"])[cols[3:]].agg(["mean", "std"]).to_csv(
            TABLE / "no_pair_summary.csv"
        )
    return diagonal, core, primary, paired


def primary_fig(core):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, sharex=True, layout="constrained")
    for i, d in enumerate(DS):
        for j, m in enumerate(["RF", "LR"]):
            ax = axes[i, j]
            for f, color, marker in [("tgf", "#377eb8", "o"), ("extended", "#bb4b53", "s")]:
                x = (
                    core[(core.dataset == d) & (core.model == m) & (core.features == f)]
                    .groupby("source_generator")
                    .average_precision.agg(["mean", "std"])
                    .reindex(GEN)
                )
                ax.errorbar(
                    GEN, x["mean"], yerr=x["std"], color=color, marker=marker, capsize=3, label=f
                )
            ax.axhline(0.05, c="gray", ls=":", lw=1, label="5% prevalence")
            ax.set_title(f"{LABEL[d]} / {m}")
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.2)
            if j == 0:
                ax.set_ylabel("Average precision (mean ± seed SD)")
    axes[0, 0].legend(loc="lower left", frameon=False)
    save(fig, "main_average_precision")


def heatmap(ax, table, title, vmin=0, vmax=1):
    image = ax.imshow(table.values, aspect="auto", vmin=vmin, vmax=vmax, cmap="YlGnBu")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_title(title)
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            x = table.iloc[i, j]
            ax.text(
                j,
                i,
                f"{x:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if x > 0.6 else "#163344",
            )
    return image


def cross_fig(main):
    for m in ["RF", "LR"]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
        for i, d in enumerate(DS):
            for j, f in enumerate(["tgf", "extended"]):
                x = main[(main.dataset == d) & (main.model == m) & (main.features == f)]
                tab = x.pivot_table(
                    index="source_generator", columns="target_generator", values="average_precision"
                ).reindex(index=GEN, columns=GEN)
                tab.to_csv(TABLE / f"cross_{d}_{m}_{f}.csv")
                im = heatmap(axes[i, j], tab, f"{LABEL[d]} / {f}")
                axes[i, j].set_xlabel("Test generator")
                axes[i, j].set_ylabel("Train generator")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Mean AP (5 seeds)")
        save(fig, f"cross_generator_{m}")


def ablation_fig(diagonal):
    features = [
        "local",
        "temporal",
        "tgf",
        "local_temporal",
        "tgf_local",
        "tgf_temporal",
        "extended",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, d in zip(axes, DS):
        x = diagonal[(diagonal.dataset == d) & (diagonal.model == "RF")]
        tab = x.pivot_table(
            index="features", columns="source_generator", values="average_precision"
        ).reindex(index=features, columns=GEN)
        tab.to_csv(TABLE / f"ablation_{d}.csv")
        im = heatmap(ax, tab, LABEL[d])
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="Mean AP (5 seeds)")
    save(fig, "feature_ablation")


def sensitivity_fig(core, sens):
    b = core[(core.model == "RF") & core.source_generator.isin(["G0", "G3"])].copy()
    all_ = pd.concat([b, sens], ignore_index=True)
    scenarios = ["main", "ratio01", "ratio10", "window500", "window2000", "past_only"]
    xlabels = ["5%, H1000", "1%", "10%", "H500", "H2000", "Past-only"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), layout="constrained", sharey=True)
    for i, d in enumerate(DS):
        for j, f in enumerate(["tgf", "extended"]):
            ax = axes[i, j]
            for g, color, offset in [("G0", COLORS[0], -0.08), ("G3", COLORS[3], 0.08)]:
                x = (
                    all_[(all_.dataset == d) & (all_.features == f) & (all_.source_generator == g)]
                    .groupby("scenario")
                    .average_precision.agg(["mean", "std"])
                    .reindex(scenarios)
                )
                ax.errorbar(
                    np.arange(6) + offset,
                    x["mean"],
                    yerr=x["std"],
                    fmt="o" if g == "G0" else "s",
                    capsize=3,
                    color=color,
                    label=g,
                )
            ax.set_xticks(range(6), xlabels, rotation=25, ha="right")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{LABEL[d]} / RF {f}")
            ax.grid(axis="y", alpha=0.2)
            ax.set_ylabel("Average precision")
            ax.legend(frameon=False)
    save(fig, "sensitivity")


def diagnostic_figures():
    diag, matching = load_diagnostics(ROOT)
    diag.to_csv(TABLE / "diagnostic_group_statistics.csv", index=False)
    matching.to_csv(TABLE / "matching_statistics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", sharey=True)
    for ax, d in zip(axes, DS):
        for group, color, mark in [
            ("endpoint", COLORS[1], "o"),
            ("local", COLORS[2], "s"),
            ("temporal", COLORS[3], "^"),
            ("prior_pair", COLORS[0], "x"),
        ]:
            z = (
                diag[(diag.dataset == d) & (diag.group == group)]
                .groupby("generator")
                .mean_ks.agg(["mean", "std"])
                .reindex(GEN)
            )
            ax.errorbar(
                GEN, z["mean"], yerr=z["std"], marker=mark, color=color, label=group, capsize=3
            )
        ax.set_title(LABEL[d])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean feature KS (lower = closer)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "generator_diagnostics")
    feature_caches = [
        ROOT / f"results/cache/main/{d}/seed_11/{g}.npz" for d in DS for g in ["G0", "G3"]
    ]
    if not all(path.exists() for path in feature_caches):
        print(
            "Skipped event-level feature CDF regeneration: rebuild feature caches first. Existing CDF assets are retained."
        )
        return diag, matching
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for i, d in enumerate(DS):
        for j, name in enumerate(
            ["u_degree", "path3_count" if d == "movielens" else "common_neighbors", "v_recency_log"]
        ):
            ax = axes[i, j]
            for g in ["G0", "G3"]:
                with np.load(ROOT / f"results/cache/main/{d}/seed_11/{g}.npz") as data:
                    x = data["X"][:, list(data["names"]).index(name)]
                    mask = data["split"] == 2
                    for y, label, color, style in [
                        (0, f"Normal ({g} history)", "#202f38", "-" if g == "G0" else "--"),
                        (1, g, COLORS[GEN.index(g)], "-"),
                    ]:
                        vals = x[mask & (data["y"] == y)].astype(float)
                        if name != "v_recency_log":
                            vals = np.log1p(vals)
                        vals = np.sort(vals)
                        ax.step(
                            vals,
                            np.arange(1, len(vals) + 1) / len(vals),
                            where="post",
                            label=label,
                            color=color,
                            ls=style,
                            lw=1.5,
                        )
            ax.set_title(f"{LABEL[d]} / {name}")
            ax.set_ylabel("Empirical CDF")
            ax.set_xlabel(name if name == "v_recency_log" else "log(1 + feature)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False)
    save(fig, "feature_distributions_seed11")
    return diag, matching


def importance_fig(main):
    records = []
    for p in (ROOT / "results/evaluation/main/runs").glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if (
            r["model"] != "RF"
            or r["features"] != "extended"
            or r["source_generator"] != r["target_generator"]
        ):
            continue
        for f, val in r["feature_importance_impurity"].items():
            records.append(
                {
                    "dataset": r["dataset"],
                    "generator": r["source_generator"],
                    "seed": r["seed"],
                    "feature": f,
                    "importance": val,
                }
            )
    x = pd.DataFrame(records)
    x.to_csv(TABLE / "rf_feature_importance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, d in zip(axes, DS):
        z = (
            x[x.dataset == d]
            .groupby(["feature", "generator"])
            .importance.mean()
            .unstack()
            .fillna(0)
        )
        z = z.loc[z.max(axis=1).nlargest(10).index].iloc[::-1]
        pos = np.arange(len(z))
        for j, g in enumerate(GEN):
            ax.barh(
                pos + (j - 1.5) * 0.18,
                z[g],
                height=0.18,
                label=g,
                color=COLORS[j],
                hatch=["", "///", "...", "xx"][j],
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_yticks(pos, z.index, fontsize=8)
        ax.set_title(LABEL[d])
        ax.set_xlabel("Mean RF impurity importance")
        ax.legend(frameon=False, ncol=2)
    save(fig, "feature_importance")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    _configure_plotting()
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    main, sens = load_frames()
    diagonal, core, primary, paired = make_tables(main, sens)
    primary_fig(core)
    cross_fig(main)
    ablation_fig(diagonal)
    sensitivity_fig(core, sens)
    diag, matching = diagnostic_figures()
    importance_fig(main)
    no_pair = pd.read_csv(ROOT / "results/no_pair/metrics.csv")
    assert len(no_pair) == 40
    facts = {
        "main_trained_models": 360,
        "core_trained_models": 160,
        "cross_generator_evaluations": 640,
        "ablation_evaluations": 200,
        "sensitivity_evaluations": 200,
        "main_total_evaluations": 840,
        "total_extension_evaluations": 1040,
        "datasets": json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8")),
        "main_ap": primary.to_dict("records"),
        "paired": paired.to_dict("records"),
    }
    facts.update(
        no_pair_evaluations=40,
        total_extension_evaluations=1080,
        total_extension_fitted_models=600,
        persisted_extension_models=560,
        official_and_timing_models=20,
    )
    (OUT / "summary.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "figures": len(list(FIG.glob("*.png"))),
                "tables": len(list(TABLE.glob("*.csv"))),
            }
        )
    )
