"""Independent validation of scientific artifacts and saved predictions."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import artifact_path
from .config import ROOT

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

SEEDS = [11, 23, 37, 51, 71]

DATASETS = ["movielens", "ucimessages"]

GENERATORS = ["G0", "G1", "G2", "G3"]

SENSITIVITIES = ["ratio01", "ratio10", "window500", "window2000", "past_only"]

ABLATIONS = ["local", "temporal", "tgf_local", "tgf_temporal", "local_temporal"]

NO_PAIR_CONFIG_KEYS = [
    "schema_version",
    "scenario",
    "dataset",
    "seed",
    "source_generator",
    "target_generator",
    "model",
    "features",
    "feature_names",
    "removed_features",
    "n_jobs",
    "source_cache_fingerprint",
    "source_model_metadata_file",
    "source_model_metadata_sha256",
    "source_model_fingerprint",
    "evaluation_source_sha256",
    "ablation_source_sha256",
    "python_version",
    "numpy_version",
    "sklearn_version",
    "hyperparameter_origin",
    "selected_min_samples_leaf",
]


def digest(value):
    data = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def independent_metrics(y, scores):
    """Threshold-group AP/PR, Mann–Whitney ROC, and stable-order top-K."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if (
        y.ndim != 1
        or y.shape != scores.shape
        or not len(y)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Invalid labels or probabilities")
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        raise ValueError("Both classes are required")
    order = np.argsort(-scores, kind="stable")
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(y) - 1]
    starts = np.r_[0, ends[:-1] + 1]
    cumulative = np.cumsum(ys)[ends]
    group_positives = np.diff(np.r_[0, cumulative])
    recall = cumulative / positives
    precision = cumulative / (ends + 1)
    ap = np.sum(precision * np.diff(np.r_[0.0, recall]))
    pr = np.trapz(np.r_[1.0, precision], np.r_[0.0, recall])
    # Convert each descending equal-score group to its average ascending rank.
    ranks = len(y) - (starts + ends) / 2
    rank_sum = np.sum(ranks * group_positives)
    roc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    true_positives = int(ys[:positives].sum())
    return {
        "average_precision": float(ap),
        "pr_auc_trapezoid": float(pr),
        "roc_auc": float(roc),
        "precision_at_k": true_positives / positives,
        "recall_at_k": true_positives / positives,
        "k": positives,
        "true_positives_at_k": true_positives,
        "n": len(y),
        "n_anomalies": positives,
        "anomaly_fraction": positives / len(y),
    }


def expected_cells():
    cells = set()
    for dataset, seed, source, model, features, target in itertools.product(
        DATASETS, SEEDS, GENERATORS, ["RF", "LR"], ["tgf", "extended"], GENERATORS
    ):
        cells.add(("main", dataset, seed, source, target, model, features))
    for dataset, seed, source, features in itertools.product(
        DATASETS, SEEDS, GENERATORS, ABLATIONS
    ):
        cells.add(("main", dataset, seed, source, source, "RF", features))
    for scenario, dataset, seed, source, features in itertools.product(
        SENSITIVITIES, DATASETS, SEEDS, ["G0", "G3"], ["tgf", "extended"]
    ):
        cells.add((scenario, dataset, seed, source, source, "RF", features))
    for dataset, seed, source in itertools.product(DATASETS, SEEDS, GENERATORS):
        cells.add(("no_pair", dataset, seed, source, source, "RF", "extended_no_pair"))
    return cells


def implementation_hash(root, names):
    hasher = hashlib.sha256()
    for name in sorted(names):
        hasher.update(name.encode("utf-8"))
        hasher.update((root / "src/graph_anomaly" / name).read_bytes().replace(b"\r\n", b"\n"))
    return hasher.hexdigest()


def generation_hash(root):
    return implementation_hash(
        root, ("data.py", "generation.py", "features.py", "config.py", "artifacts.py")
    )


def load_original(root, dataset, metadata):
    """Independent reconstruction of the official cleaned file's integer IDs."""
    path = artifact_path(root, metadata["source_path"])
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            timestamp, u, v = line.split()
            rows.append(
                (
                    int(timestamp),
                    u if dataset == "movielens" else int(u),
                    v if dataset == "movielens" else int(v),
                )
            )
    rows.sort(key=lambda row: row[0])
    if dataset == "movielens":
        left = sorted({row[1] for row in rows})
        right = sorted({row[2] for row in rows})
        mapping = {node: index for index, node in enumerate(left + right)}
        partition = len(left)
    else:
        nodes = sorted({row[1] for row in rows} | {row[2] for row in rows})
        mapping = {node: index for index, node in enumerate(nodes)}
        partition = None
    t = np.array([row[0] for row in rows], dtype=np.int64)
    u = np.array([mapping[row[1]] for row in rows], dtype=np.int32)
    v = np.array([mapping[row[2]] for row in rows], dtype=np.int32)
    if partition is None:
        u, v = np.minimum(u, v), np.maximum(u, v)
    n_nodes = len(mapping)
    pairs = u.astype(np.int64) * n_nodes + v
    unique_pairs, first_indices = np.unique(pairs, return_index=True)
    return {
        "t": t,
        "u": u,
        "v": v,
        "pairs": unique_pairs,
        "first_pair_t": t[first_indices],
        "n_nodes": n_nodes,
        "partition": partition,
        "sha256": file_hash(path),
    }


class Audit:
    def __init__(self, root, available_only=False, models=False):
        self.root, self.available_only, self.models = root, available_only, models
        self.errors, self.excluded, self.missing, self.failure_history = [], [], [], []
        self.cache_info, self.models_seen, self.records, self.checked_records = {}, {}, [], []
        self.cache_results, self.model_results = [], []
        self.max_metric_error = self.max_model_error = self.max_csv_error = 0.0
        self.eval_hash = implementation_hash(
            root,
            (
                "artifacts.py",
                "data.py",
                "config.py",
                "evaluation.py",
                "metrics.py",
                "models.py",
                "results.py",
            ),
        )
        self.no_pair_hash = file_hash(root / "src/graph_anomaly/controls.py")
        self.gen_hash = generation_hash(root)
        self.configuration = read_json(root / "configs/experiment.json")
        self.n_models_rescored = 0
        self.csv_rows_checked = self.summary_groups_checked = 0

    def check(self, condition, message):
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def unavailable(self, name):
        self.missing.append(name)
        if not self.available_only:
            self.errors.append("Missing/incomplete required artifact: " + name)

    def scenario_configuration(self, scenario):
        configuration = json.loads(json.dumps(self.configuration))
        for key, value in configuration["scenarios"][scenario].items():
            configuration["history" if key == "window" else "injection"][key] = value
        configuration["scenario"] = scenario
        return configuration

    def inventory(self):
        actual = []
        for collection in ["main"] + SENSITIVITIES + ["no_pair"]:
            directory = (
                self.root / "results/no_pair"
                if collection == "no_pair"
                else self.root / "results/evaluation" / collection
            )
            for path in sorted((directory / "runs").glob("*.json")):
                try:
                    record = read_json(path)
                    if record.get("status") != "complete":
                        self.excluded.append(
                            {
                                "path": str(path.relative_to(self.root)),
                                "status": record.get("status"),
                            }
                        )
                        continue
                    record["_collection"] = collection
                    cell = (collection,) + tuple(
                        record[field]
                        for field in [
                            "dataset",
                            "seed",
                            "source_generator",
                            "target_generator",
                            "model",
                            "features",
                        ]
                    )
                    actual.append(cell)
                    self.records.append(record)
                except Exception as error:
                    self.errors.append(f"Unreadable run {path}: {error}")
            for path in sorted((directory / "failures").glob("*.json")):
                record = read_json(path)
                self.failure_history.append({"path": str(path.relative_to(self.root)), **record})
                if not record.get("resolved", False):
                    self.check(False, f"Unresolved evaluation failure: {path}")
        self.check(len(actual) == len(set(actual)), "Duplicate factorial evaluation cells")
        required = expected_cells()
        for cell in sorted(required - set(actual)):
            self.unavailable("evaluation " + "/".join(map(str, cell)))
        self.check(not (set(actual) - required), "Unexpected factorial evaluation cells")
        for path in sorted((self.root / "results/failures").glob("*.json")):
            record = read_json(path)
            self.failure_history.append({"path": str(path.relative_to(self.root)), **record})
            self.check(
                record.get("status") == "failed_then_fixed" or record.get("resolved", False),
                f"Unresolved generation failure: {path}",
            )
        for scenario in ["main"] + SENSITIVITIES:
            generators = GENERATORS if scenario == "main" else ["G0", "G3"]
            for dataset, seed, generator in itertools.product(DATASETS, SEEDS, generators):
                key = (scenario, dataset, seed, generator)
                path = self.root / f"results/cache/{scenario}/{dataset}/seed_{seed}/{generator}.npz"
                try:
                    manifest = read_json(path.with_suffix(".json"))
                    if manifest.get("status") != "complete" or not path.exists():
                        self.unavailable("cache " + "/".join(map(str, key)))
                        continue
                    npz_hash, manifest_hash = file_hash(path), file_hash(path.with_suffix(".json"))
                    self.cache_info[key] = {
                        "path": path,
                        "manifest": manifest,
                        "fingerprint": digest(
                            {"npz_sha256": npz_hash, "metadata_sha256": manifest_hash}
                        ),
                        "npz_sha256": npz_hash,
                        "manifest_sha256": manifest_hash,
                    }
                except (OSError, json.JSONDecodeError) as error:
                    self.unavailable("cache " + "/".join(map(str, key)) + f": {error}")

    def validate_model_metadata(self, path):
        path = Path(path)
        if path in self.models_seen:
            return self.models_seen[path]
        metadata = read_json(path)
        identity = str(path.relative_to(self.root))
        config = {
            key: value
            for key, value in metadata.items()
            if key not in {"model_fingerprint", "fitting"}
        }
        self.check(
            metadata["model_fingerprint"] == digest(config), f"Model config hash: {identity}"
        )
        self.check(
            metadata["evaluation_source_sha256"] == self.eval_hash,
            f"Model evaluator source hash: {identity}",
        )
        cache_key = tuple(
            metadata[key] for key in ["scenario", "dataset", "seed", "source_generator"]
        )
        cache = self.cache_info.get(cache_key)
        self.check(
            cache is not None and metadata["source_cache_fingerprint"] == cache["fingerprint"],
            f"Model source cache fingerprint: {identity}",
        )
        fit = metadata["fitting"]
        chosen = max(
            fit["validation_trials"], key=lambda trial: trial["validation_average_precision"]
        )
        parameter = "min_samples_leaf" if metadata["model"] == "RF" else "C"
        self.check(
            fit["selected_params"][parameter] == chosen[parameter]
            and fit["selected_validation_average_precision"]
            == chosen["validation_average_precision"],
            f"Validation-only hyperparameter selection: {identity}",
        )
        self.check(
            fit["refit_on_train_plus_validation"] and metadata["n_jobs"] == 2,
            f"Refit/execution settings: {identity}",
        )
        self.check(
            path.with_suffix(".joblib").exists(), f"Missing serialized estimator: {identity}"
        )
        self.models_seen[path] = metadata
        return metadata

    def validate_record_provenance(self, record, cache):
        name = record["_collection"] + "/" + record["run_id"]
        self.check(record["evaluation_source_sha256"] == self.eval_hash, f"Run source hash: {name}")
        expected_jobs = 1 if record["_collection"] == "no_pair" else 2
        self.check(record["n_jobs"] == expected_jobs, f"Run execution settings: {name}")
        source_key = tuple(
            record[key] for key in ["scenario", "dataset", "seed", "source_generator"]
        )
        source = self.cache_info.get(source_key)
        self.check(
            source is not None and record["source_cache_fingerprint"] == source["fingerprint"],
            f"Run source cache fingerprint: {name}",
        )
        if record["_collection"] == "no_pair":
            model_path = artifact_path(self.root, record["source_model_metadata_file"])
            metadata = self.validate_model_metadata(model_path)
            self.check(
                record["source_model_metadata_sha256"] == file_hash(model_path),
                f"No-pair source metadata checksum: {name}",
            )
            self.check(
                record["source_model_fingerprint"] == metadata["model_fingerprint"],
                f"No-pair source model fingerprint: {name}",
            )
            self.check(
                record["ablation_source_sha256"] == self.no_pair_hash,
                f"No-pair source-code hash: {name}",
            )
            self.check(
                record["run_fingerprint"]
                == digest({key: record[key] for key in NO_PAIR_CONFIG_KEYS}),
                f"No-pair run fingerprint: {name}",
            )
            self.check(
                record["feature_names"] == [n for n in cache["names"] if n != "pair_seen_before"],
                f"No-pair feature omission: {name}",
            )
            self.check(
                record["selected_min_samples_leaf"]
                == metadata["fitting"]["selected_params"]["min_samples_leaf"]
                and not record["hyperparameter_search_performed"],
                f"No-pair no-retuning control: {name}",
            )
            prediction_path = artifact_path(self.root, record["prediction_file"])
            self.check(
                record["prediction_sha256"] == file_hash(prediction_path),
                f"No-pair prediction checksum: {name}",
            )
        else:
            metadata = self.validate_model_metadata(
                artifact_path(self.root, record["model_metadata_file"])
            )
            self.check(
                record["model_fingerprint"] == metadata["model_fingerprint"],
                f"Run/model fingerprint linkage: {name}",
            )
            self.check(
                record["target_cache_fingerprint"] == cache["fingerprint"],
                f"Run target cache fingerprint: {name}",
            )
            self.check(
                record["run_fingerprint"]
                == digest({"model": metadata["model_fingerprint"], "target": cache["fingerprint"]}),
                f"Run fingerprint: {name}",
            )
            self.check(
                record["feature_names"] == metadata["feature_names"]
                and record["params"] == metadata["fitting"]["selected_params"],
                f"Run/model feature and parameter linkage: {name}",
            )
        if source:
            counts = source["manifest"]["split_counts"]
            self.check(
                record["n_train"] == sum(counts[0].values())
                and record["n_val"] == sum(counts[1].values()),
                f"Train/validation counts: {name}",
            )

    def _validate_cache_timing(self, values, manifest, configuration, original, name):
        """Check feature schema, elapsed-time splits, strict history, and warm-up preservation."""
        X, t, u, v, y, split = (values[field] for field in ["X", "t", "u", "v", "y", "split"])
        self.check(
            X.shape == (len(y), len(values["names"])) and np.isfinite(X).all(),
            f"Finite feature shape: {name}",
        )
        self.check(
            values["names"] == manifest["feature_names"]
            and len(set(values["names"])) == len(values["names"]),
            f"Feature schema: {name}",
        )
        self.check(
            np.isin(y, [0, 1]).all() and np.isin(split, [0, 1, 2]).all(),
            f"Binary labels/split codes: {name}",
        )
        self.check(np.all(t[:-1] <= t[1:]), f"Chronological event ordering: {name}")
        raw_t = original["t"]
        bounds = [float(raw_t[0] + f * (raw_t[-1] - raw_t[0])) for f in (0.6, 0.8)]
        self.check(
            np.array_equal(split, np.searchsorted(bounds, t, side="right")),
            f"Elapsed-time split/tie integrity: {name}",
        )
        previous = np.searchsorted(raw_t, t, side="left") - 1
        self.check(
            np.all(previous >= 0) and np.all(values["history_max_timestamp"] < t),
            f"Strict pre-arrival timestamp lineage: {name}",
        )
        self.check(
            np.array_equal(values["history_max_timestamp"], raw_t[previous]),
            f"Recorded history timestamp lineage: {name}",
        )
        warmup = int(
            raw_t[min(configuration["history"]["warmup_observed_events"] - 1, len(raw_t) - 1)]
        )
        valid = raw_t > warmup
        self.check(
            manifest["warmup_last_timestamp"] == warmup and np.all(t > warmup),
            f"Whole-timestamp warmup: {name}",
        )
        normal = y == 0
        for field in ["t", "u", "v"]:
            self.check(
                np.array_equal(values[field][normal], original[field][valid]),
                f"Observed event preservation/{field}: {name}",
            )
        return normal

    def _record_cache_validation(self, key, info, counts, y):
        """Record the accepted cache identity and class counts for the audit summary."""
        scenario, dataset, seed, generator = key
        self.cache_results.append(
            {
                "scenario": scenario,
                "dataset": dataset,
                "seed": seed,
                "generator": generator,
                "npz_sha256": info["npz_sha256"],
                "manifest_sha256": info["manifest_sha256"],
                "fingerprint": info["fingerprint"],
                "split_counts": counts,
                "n_events": len(y),
            }
        )

    def validate_cache(self, key, info, original):
        name = "/".join(map(str, key))
        scenario, dataset, seed, generator = key
        manifest = info["manifest"]
        configuration = self.scenario_configuration(scenario)
        self.check(manifest["config"] == configuration, f"Declared configuration: {name}")
        self.check(
            manifest["dataset"]["sha256"] == original["sha256"], f"Original data checksum: {name}"
        )
        self.check(manifest["code_sha256"] == self.gen_hash, f"Generator code checksum: {name}")
        signature = hashlib.sha256(
            json.dumps(
                {
                    "cfg": configuration,
                    "seed": seed,
                    "generator": generator,
                    "data": original["sha256"],
                    "code": self.gen_hash,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.check(manifest["signature"] == signature, f"Generator signature: {name}")
        with np.load(info["path"], allow_pickle=False) as data:
            values = {
                field: data[field]
                for field in ["X", "names", "t", "u", "v", "y", "split", "history_max_timestamp"]
            }
        values["names"] = values["names"].tolist()
        X, t, u, v, y, split = (values[field] for field in ["X", "t", "u", "v", "y", "split"])
        normal = self._validate_cache_timing(values, manifest, configuration, original, name)
        n_nodes = original["n_nodes"]
        self.check(
            np.all((u >= 0) & (u < n_nodes) & (v >= 0) & (v < n_nodes) & (u != v)),
            f"Endpoint range/self-loops: {name}",
        )
        if original["partition"] is None:
            self.check(np.all(u < v), f"Undirected canonical endpoints: {name}")
        else:
            self.check(
                np.all((u < original["partition"]) & (v >= original["partition"])),
                f"Bipartite partitions: {name}",
            )
        positive_rows = np.flatnonzero(y == 1)
        pair_codes = u[positive_rows].astype(np.int64) * n_nodes + v[positive_rows]
        self.check(len(np.unique(pair_codes)) == len(pair_codes), f"Unique injected pairs: {name}")
        locations = np.searchsorted(original["pairs"], pair_codes)
        in_range = locations < len(original["pairs"])
        exists = np.zeros(len(pair_codes), dtype=bool)
        exists[in_range] = original["pairs"][locations[in_range]] == pair_codes[in_range]
        if configuration["injection"]["absence"] == "global":
            self.check(not exists.any(), f"Full-observed-stream non-edge constraint: {name}")
        else:
            self.check(
                np.all(original["first_pair_t"][locations[exists]] > t[positive_rows][exists]),
                f"Past-only and same-timestamp observed-pair exclusion: {name}",
            )
        fraction = configuration["injection"]["anomaly_fraction"]
        counts = []
        for s in range(3):
            n_normal = int(((split == s) & normal).sum())
            n_anomaly = int(((split == s) & ~normal).sum())
            counts.append({"normal": n_normal, "anomaly": n_anomaly})
            self.check(
                n_normal > 0
                and n_anomaly == round(n_normal * fraction / (1 - fraction))
                and n_anomaly > 0,
                f"Both classes and exact anomaly quota split{s}: {name}",
            )
        self.check(
            counts == manifest["split_counts"]
            and len(positive_rows) == manifest["anomalies"]
            and int(normal.sum()) == manifest["observed_scored"],
            f"Manifest class counts: {name}",
        )
        for flag in [
            "finite_features",
            "strict_prearrival",
            "requested_count_met",
            "unique_injected_pairs",
            "graph_constraints",
        ]:
            self.check(
                manifest["checks"].get(flag) is True, f"Manifest validity flag/{flag}: {name}"
            )
        self.validate_generator_audit(info, values, original, positive_rows, configuration, name)
        values["fingerprint"] = info["fingerprint"]
        self._record_cache_validation(key, info, counts, y)
        return values

    def validate_generator_audit(self, info, values, original, positive_rows, configuration, name):
        audit_path = info["path"].with_name(info["path"].stem + "_audit.jsonl.gz")
        rejection_totals = Counter()
        total_draws = 0
        n_records = 0
        with gzip.open(audit_path, "rt", encoding="utf-8") as stream:
            for n_records, line in enumerate(stream, 1):
                record = json.loads(line)
                if not self.check(
                    n_records <= len(positive_rows), f"Excess generator audit record: {name}"
                ):
                    break
                row = positive_rows[n_records - 1]
                self.check(
                    all(
                        record[field] == int(values[field][row])
                        for field in ["t", "u", "v", "split"]
                    ),
                    f"Generated event/audit alignment row{n_records}: {name}",
                )
                reference = int(record["reference_index"])
                self.check(0 <= reference < len(original["t"]), f"Reference event range: {name}")
                self.check(
                    record["t"] == int(original["t"][reference])
                    and record["reference_u"] == int(original["u"][reference])
                    and record["reference_v"] == int(original["v"][reference]),
                    f"Reference event context: {name}",
                )
                self.check(
                    record["history_max_timestamp"] == int(values["history_max_timestamp"][row]),
                    f"Generator history audit timestamp: {name}",
                )
                pool = (
                    1
                    if info["manifest"]["generator"] == "G0"
                    else configuration["injection"]["legal_candidate_pool"]
                )
                self.check(
                    record["candidate_pool_size"] == pool
                    and record["status"] == "valid"
                    and record["partition_violations"] == 0,
                    f"Candidate pool/support status: {name}",
                )
                self.check(
                    pool <= record["draws"] <= configuration["injection"]["max_draws"],
                    f"Candidate draw budget: {name}",
                )
                self.check(
                    all(
                        math.isfinite(record[field]) and record[field] >= 0
                        for field in ["endpoint_distance", "local_distance", "temporal_distance"]
                    ),
                    f"Finite matching distances: {name}",
                )
                rejection_totals.update(record["rejections"])
                total_draws += record["draws"]
        self.check(n_records == len(positive_rows), f"Generator audit event count: {name}")
        self.check(
            total_draws == info["manifest"]["candidate_draws"]
            and dict(rejection_totals) == info["manifest"]["rejections"],
            f"Generator rejection accounting: {name}",
        )
        diagnostic_path = info["path"].with_name(info["path"].stem + "_diagnostics.csv")
        with diagnostic_path.open(newline="", encoding="utf-8-sig") as stream:
            diagnostic_rows = list(csv.DictReader(stream))
        actual = {(int(row["split"]), row["feature"]) for row in diagnostic_rows}
        expected = set(itertools.product(range(3), values["names"]))
        self.check(
            actual == expected and len(diagnostic_rows) == len(expected),
            f"Diagnostic feature/split coverage: {name}",
        )
        for row in diagnostic_rows:
            counts = info["manifest"]["split_counts"][int(row["split"])]
            self.check(
                int(row["normal_n"]) == counts["normal"]
                and int(row["anomaly_n"]) == counts["anomaly"],
                f"Diagnostic denominators: {name}",
            )

    def validate_prediction(self, record, cache):
        name = record["_collection"] + "/" + record["run_id"]
        self.validate_record_provenance(record, cache)
        test = cache["split"] == 2
        rows = np.flatnonzero(test)
        y = cache["y"][test]
        seen = cache["X"][test, cache["names"].index("pair_seen_before")] > 0
        novel = (y == 1) | ~seen
        with np.load(
            artifact_path(self.root, record["prediction_file"]), allow_pickle=False
        ) as prediction:
            for field in ["y", "t", "u", "v"]:
                self.check(
                    np.array_equal(prediction[field], cache[field][test]),
                    f"Prediction event identity/{field}: {name}",
                )
            self.check(
                np.array_equal(prediction["row_id"], rows), f"Prediction row alignment: {name}"
            )
            self.check(
                np.array_equal(prediction["novel_normal_subset"], novel),
                f"Novel-normal subset mask: {name}",
            )
            scores = prediction["scores"]
            self.check(
                scores.shape == y.shape
                and np.isfinite(scores).all()
                and np.all((scores >= 0) & (scores <= 1)),
                f"Finite probability vector: {name}",
            )
            for field, mask in [
                ("metrics", np.ones(len(y), dtype=bool)),
                ("novel_normal_metrics", novel),
            ]:
                computed = independent_metrics(y[mask], scores[mask])
                for metric, value in computed.items():
                    difference = abs(value - float(record[field][metric]))
                    self.max_metric_error = max(self.max_metric_error, difference)
                    self.check(difference < 1e-10, f"Independent metric/{field}/{metric}: {name}")
            self.check(record["n_test"] == len(y), f"Test set size: {name}")
            self.check(
                abs(record["normal_previously_seen_fraction"] - float(seen[y == 0].mean())) < 1e-12,
                f"Normal-repeat denominator: {name}",
            )
            if (
                self.models
                and record["_collection"] != "no_pair"
                and record["source_generator"] == record["target_generator"]
            ):
                self.rescore_model(record, cache, rows, scores)
        self.checked_records.append(record)

    def rescore_model(self, record, cache, rows, scores):
        # Imported only when the user requests serialization/rescoring checks.
        import joblib

        sys.path.insert(0, str(self.root / "src"))
        path = (artifact_path(self.root, record["model_metadata_file"])).with_suffix(".joblib")
        model = joblib.load(path)
        take = np.unique(np.linspace(0, len(rows) - 1, 32, dtype=int))
        columns = [cache["names"].index(name) for name in record["feature_names"]]
        if record["model"] == "RF":
            self.check(
                model.n_jobs == 2 and model.n_estimators == 100,
                f"Serialized RF execution parameters: {record['run_id']}",
            )
        rescored = model.predict_proba(cache["X"][rows[take]][:, columns])[
            :, int(np.flatnonzero(model.classes_ == 1)[0])
        ]
        difference = float(np.max(np.abs(rescored - scores[take])))
        self.max_model_error = max(self.max_model_error, difference)
        self.check(difference < 1e-12, f"Serialized estimator rescore: {record['run_id']}")
        self.n_models_rescored += 1
        self.model_results.append(
            {"path": str(path.relative_to(self.root)), "sha256": file_hash(path)}
        )
        del model

    def verify_all_caches_and_predictions(self):
        by_cache = defaultdict(list)
        for record in self.records:
            key = tuple(
                record[field] for field in ["scenario", "dataset", "seed", "target_generator"]
            )
            by_cache[key].append(record)
        for dataset in DATASETS:
            keys = [key for key in self.cache_info if key[1] == dataset]
            if not keys:
                continue
            original = load_original(
                self.root, dataset, self.cache_info[keys[0]]["manifest"]["dataset"]
            )
            for key in keys:
                before = len(self.errors)
                try:
                    cache = self.validate_cache(key, self.cache_info[key], original)
                    for record in by_cache[key]:
                        try:
                            self.validate_prediction(record, cache)
                        except Exception as error:
                            self.errors.append(
                                f"Prediction verification exception {record['run_id']}: {type(error).__name__}: {error}"
                            )
                    del cache
                except Exception as error:
                    self.errors.append(
                        f"Cache verification exception {key}: {type(error).__name__}: {error}"
                    )
                if len(self.cache_results) % 10 == 0 or len(self.errors) > before:
                    print(
                        f"CHECKED caches={len(self.cache_results)}/140 predictions={len(self.checked_records)}/1080 errors={len(self.errors)}",
                        flush=True,
                    )
            del original

    def verify_csv_tables(self):
        records_by_collection = defaultdict(list)
        for record in self.checked_records:
            records_by_collection[record["_collection"]].append(record)
        for collection, records in records_by_collection.items():
            directory = (
                self.root / "results/no_pair"
                if collection == "no_pair"
                else self.root / "results/evaluation" / collection
            )
            table = directory / "metrics.csv"
            if not table.exists():
                self.unavailable(str(table.relative_to(self.root)))
                continue
            with table.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            indexed = {row["run_id"]: row for row in rows}
            self.check(len(indexed) == len(rows), f"Duplicate CSV run IDs: {collection}")
            if not self.available_only:
                self.check(
                    set(indexed) == {record["run_id"] for record in records},
                    f"CSV/run coverage: {collection}",
                )
            for record in records:
                if record["run_id"] not in indexed:
                    self.unavailable(f"CSV row {collection}/{record['run_id']}")
                    continue
                row = indexed[record["run_id"]]
                for prefix, metrics in [
                    ("", record["metrics"]),
                    ("novel_", record["novel_normal_metrics"]),
                ]:
                    for metric, value in metrics.items():
                        difference = abs(float(row[prefix + metric]) - value)
                        self.max_csv_error = max(self.max_csv_error, difference)
                        self.check(
                            difference < 1e-10,
                            f"CSV metric {collection}/{record['run_id']}/{prefix}{metric}",
                        )
                self.csv_rows_checked += 1
            if collection == "no_pair":
                continue
            dimensions = [
                "scenario",
                "dataset",
                "source_generator",
                "target_generator",
                "model",
                "features",
            ]
            groups = defaultdict(list)
            for record in records:
                if record["run_id"] in indexed:
                    groups[tuple(record[field] for field in dimensions)].append(
                        indexed[record["run_id"]]
                    )
            summary_path = directory / "summary.csv"
            if not summary_path.exists():
                self.unavailable(str(summary_path.relative_to(self.root)))
                continue
            with summary_path.open(newline="", encoding="utf-8-sig") as stream:
                summaries = list(csv.DictReader(stream))
            if not self.available_only:
                self.check(len(summaries) == len(groups), f"Summary group count: {collection}")
            for row in summaries:
                key = tuple(row[field] for field in dimensions)
                if key not in groups:
                    if not self.available_only:
                        self.check(False, f"Summary unknown group: {collection}/{key}")
                    continue
                group = groups[key]
                self.check(
                    int(row["n_seeds"]) == len(group), f"Summary seed count: {collection}/{key}"
                )
                if not self.available_only:
                    self.check(
                        {int(item["seed"]) for item in group} == set(SEEDS),
                        f"Exact five-seed coverage: {collection}/{key}",
                    )
                for field, saved in row.items():
                    if field.endswith("_mean") or field.endswith("_std"):
                        metric, statistic = field.rsplit("_", 1)
                        values = [
                            float(item[metric]) for item in group if item.get(metric, "") != ""
                        ]
                        if statistic == "std" and len(values) < 2:
                            continue
                        expected = (
                            statistics.mean(values)
                            if statistic == "mean"
                            else statistics.stdev(values)
                        )
                        difference = abs(float(saved) - expected)
                        self.max_csv_error = max(self.max_csv_error, difference)
                        self.check(
                            difference < 1e-10, f"Summary aggregation: {collection}/{key}/{field}"
                        )
                self.summary_groups_checked += 1

    def run(self):
        start = time.perf_counter()
        self.inventory()
        self.verify_all_caches_and_predictions()
        self.verify_csv_tables()
        counts = dict(Counter(record["_collection"] for record in self.checked_records))
        if not self.available_only:
            self.check(
                len(self.cache_results) == 140, "Expected exactly140 verified generator caches"
            )
            self.check(
                len(self.checked_records) == 1080, "Expected exactly1080 verified evaluations"
            )
            self.check(
                len(self.models_seen) == 560, "Expected exactly560 persisted model metadata files"
            )
        status = (
            "failed" if self.errors else ("partial" if self.missing or self.excluded else "passed")
        )
        return {
            "status": status,
            "verification_utc": datetime.now(timezone.utc).isoformat(),
            "available_only": self.available_only,
            "models_option": self.models,
            "expected_evaluations": 1080,
            "verified_evaluations": len(self.checked_records),
            "evaluation_counts": counts,
            "expected_caches": 140,
            "verified_caches": len(self.cache_results),
            "verified_model_metadata": len(self.models_seen),
            "serialized_models_rescored": self.n_models_rescored,
            "csv_rows_checked": self.csv_rows_checked,
            "summary_groups_checked": self.summary_groups_checked,
            "max_absolute_metric_error": self.max_metric_error,
            "max_absolute_csv_error": self.max_csv_error,
            "max_absolute_model_probability_error": self.max_model_error if self.models else None,
            "elapsed_seconds": time.perf_counter() - start,
            "seeds": SEEDS,
            "evaluation_source_sha256": self.eval_hash,
            "generator_source_sha256": self.gen_hash,
            "no_pair_source_sha256": self.no_pair_hash,
            "verifier_source_sha256": file_hash(Path(__file__)),
            "missing_artifacts": self.missing,
            "excluded_incomplete_or_failed_runs": self.excluded,
            "historical_failures": self.failure_history,
            "errors": self.errors,
            "cache_checksums_and_counts": self.cache_results,
            "optional_model_checksums": self.model_results,
            "check_notes": [
                "No model fitting occurs. At most one feature cache and one optional estimator are resident.",
                "All main and novel-normal metrics are recomputed independently of graph_anomaly.evaluation.compute_metrics.",
                "Legality is checked against independently remapped original data, including warmup, full/past absence, unique injections, partitions, quotas and observed-event preservation.",
                "Timestamp lineage arrays and audit logs are checked; all feature vectors are not regenerated from scratch.",
                "Configuration/source/cache/model provenance fingerprints are checked. Serialized estimator rescoring is optional; main_audit.json already documents all360 main estimators.",
                "No-pair models were intentionally not persisted; their fixed source-selected parameters, feature omission, prediction hashes and metrics are verified.",
                "Resolved pilot failures remain in history and never enter successful-run counts.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument(
        "--models",
        action="store_true",
        help="Additionally load each persisted estimator and rescore32 diagonal test events",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: results/validation/final_audit.json (available_audit.json in partial mode)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "results/validation" / (
        "available_audit.json" if args.available_only else "final_audit.json"
    )
    audit = Audit(root, args.available_only, args.models)
    try:
        report = audit.run()
    except Exception as error:
        report = {
            "status": "failed",
            "errors": audit.errors + [f"Unhandled verifier error: {type(error).__name__}: {error}"],
            "verified_evaluations": len(audit.checked_records),
            "verified_caches": len(audit.cache_results),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in [
                    "status",
                    "verified_evaluations",
                    "verified_caches",
                    "verified_model_metadata",
                    "csv_rows_checked",
                    "summary_groups_checked",
                    "max_absolute_metric_error",
                    "elapsed_seconds",
                    "errors",
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    return int(
        report["status"] == "failed" or (report["status"] != "passed" and not args.available_only)
    )
