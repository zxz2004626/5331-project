"""Aggregate only records whose source and artifact provenance still match."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import _canonical, _digest, _file_hash, _read_json, _write_csv, artifact_path
from .provenance import evaluation_code_hash


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
            "or use graphbench report to read the saved historical results."
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
