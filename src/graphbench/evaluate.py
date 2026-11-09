"""Leakage-aware model selection and cached cross-generator evaluation.

Run metadata and predictions are checkpointed independently. Changing a cache,
feature schema, configuration or this implementation invalidates its checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEEDS = [11, 23, 37, 51, 71]
GENERATORS = ["G0", "G1", "G2", "G3"]
CORE_FEATURES = ["tgf", "extended"]
ABLATIONS = ["local", "temporal", "tgf_local", "tgf_temporal", "local_temporal"]
SCHEMA_VERSION = 1


class NonnegativeLog1p(BaseEstimator, TransformerMixin):
    """A stateless, explicitly checked log transform, preceding a fitted scaler."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self._check(X)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    @staticmethod
    def _check(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Features must be a finite two-dimensional array")
        if (values < 0).any():
            raise ValueError("The log1p feature protocol requires nonnegative features")
        return values

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = self._check(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from fitted transform")
        return np.log1p(values)


def compute_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Anomaly-positive metrics; rank ties use the preserved cache row order."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != scores.shape or not len(y):
        raise ValueError("Labels and scores must be equal nonempty vectors")
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Scores must be finite and labels must be binary")
    y = y.astype(np.int8, copy=False)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        raise ValueError("Both normal and anomaly examples are required")
    ranking = np.lexsort((np.arange(len(y)), -scores))
    tp = int(y[ranking[:positives]].sum())
    precision, recall, _ = precision_recall_curve(y, scores)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "pr_auc_trapezoid": float(auc(recall, precision)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision_at_k": float(tp / positives),
        "recall_at_k": float(tp / positives),
        "k": positives,
        "true_positives_at_k": tp,
        "n": int(len(y)),
        "n_anomalies": positives,
        "anomaly_fraction": float(positives / len(y)),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class Cache:
    path: Path
    X: np.ndarray
    names: list[str]
    t: np.ndarray
    u: np.ndarray
    v: np.ndarray
    y: np.ndarray
    split: np.ndarray
    fingerprint: str
    metadata: dict[str, Any]


def load_cache(path: Path) -> Cache:
    if not path.exists():
        raise FileNotFoundError(f"Required generator cache is missing: {path}")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Required cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata.get("status") != "complete":
        raise ValueError(f"Cache manifest is not complete: {metadata_path}")
    with np.load(path, allow_pickle=False) as cached:
        required = ["X", "names", "t", "u", "v", "y", "split"]
        missing = set(required).difference(cached.files)
        if missing:
            raise ValueError(f"Cache is missing arrays: {sorted(missing)}")
        values = {key: cached[key] for key in required}
    names = [str(name) for name in values.pop("names").tolist()]
    n = len(values["y"])
    if values["X"].shape != (n, len(names)) or len(set(names)) != len(names):
        raise ValueError("Invalid cache feature dimensions or duplicate names")
    if "feature_names" in metadata and metadata["feature_names"] != names:
        raise ValueError("Cache feature schema differs from its manifest")
    if not np.isfinite(values["X"]).all():
        raise ValueError("Cache features contain non-finite values")
    for key in ["t", "u", "v", "split"]:
        if values[key].shape != (n,):
            raise ValueError(f"Invalid cache array shape: {key}")
    if not np.isin(values["y"], [0, 1]).all():
        raise ValueError("Cache labels must be binary")
    if not np.isin(values["split"], [0, 1, 2]).all():
        raise ValueError("Cache split codes must be 0, 1 or 2")
    previous_max = None
    for split in [0, 1, 2]:
        mask = values["split"] == split
        if not mask.any() or len(np.unique(values["y"][mask])) != 2:
            raise ValueError(f"Split {split} must contain both classes")
        if "split_counts" in metadata:
            counts = metadata["split_counts"][split]
            if (counts["normal"] != int((values["y"][mask] == 0).sum()) or
                    counts["anomaly"] != int((values["y"][mask] == 1).sum())):
                raise ValueError(f"Split {split} class counts differ from manifest")
        lo, hi = int(values["t"][mask].min()), int(values["t"][mask].max())
        if previous_max is not None and lo <= previous_max:
            raise ValueError("Chronological splits overlap in timestamp")
        previous_max = hi
    fingerprint = _digest({"npz_sha256": _file_hash(path),
                           "metadata_sha256": _file_hash(metadata_path)})
    return Cache(path=path, names=names, fingerprint=fingerprint,
                 metadata=metadata, **values)


def selected_names(cache: Cache, dataset: str, feature_set: str) -> list[str]:
    from .history import feature_groups

    groups = feature_groups(dataset == "movielens")
    if feature_set == "extended":
        return list(cache.names)
    if feature_set not in CORE_FEATURES + ABLATIONS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    requested = feature_set.split("_")
    wanted = {name for group in requested for name in groups[group]}
    missing = wanted.difference(cache.names)
    if missing:
        raise ValueError(f"Feature group is missing cache columns: {sorted(missing)}")
    names = [name for name in cache.names if name in wanted]
    if not names:
        raise ValueError(f"Empty feature set: {feature_set}")
    return names


def select_columns(cache: Cache, names: list[str]) -> np.ndarray:
    missing = set(names).difference(cache.names)
    if missing:
        raise ValueError(f"Target cache is missing model features: {sorted(missing)}")
    return cache.X[:, [cache.names.index(name) for name in names]]


def build_model(model_name: str, parameter: float | int, seed: int,
                n_jobs: int = 2):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=100, n_jobs=n_jobs, min_samples_leaf=int(parameter),
            class_weight="balanced_subsample", random_state=seed,
        )
    if model_name == "LR":
        return Pipeline([
            ("log1p", NonnegativeLog1p()),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="liblinear", max_iter=1000, class_weight="balanced",
                C=float(parameter), random_state=seed,
            )),
        ])
    raise ValueError(f"Unknown model: {model_name}")


def anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    positive = np.flatnonzero(model.classes_ == 1)
    if len(positive) != 1:
        raise ValueError("Fitted model does not have one anomaly-positive class")
    return model.predict_proba(X)[:, int(positive[0])]


def fit_selected_model(cache: Cache, names: list[str], model_name: str,
                       seed: int, n_jobs: int = 2) -> tuple[Any, dict[str, Any]]:
    X = select_columns(cache, names)
    train, validation = cache.split == 0, cache.split == 1
    candidates = [1, 5] if model_name == "RF" else [0.1, 1.0, 10.0]
    parameter_name = "min_samples_leaf" if model_name == "RF" else "C"
    trials = []
    selection_start = time.perf_counter()
    for parameter in candidates:
        model = build_model(model_name, parameter, seed, n_jobs)
        start = time.perf_counter()
        model.fit(X[train], cache.y[train])
        scores = anomaly_scores(model, X[validation])
        trials.append({parameter_name: parameter,
                       "validation_average_precision": float(
                           average_precision_score(cache.y[validation], scores)),
                       "fit_and_validation_seconds": time.perf_counter() - start})
    # Ties retain the first predeclared candidate, without test-dependent choices.
    chosen = max(trials, key=lambda trial: trial["validation_average_precision"])
    model = build_model(model_name, chosen[parameter_name], seed, n_jobs)
    selection_seconds = time.perf_counter() - selection_start
    refit_start = time.perf_counter()
    development = train | validation
    model.fit(X[development], cache.y[development])
    details = {
        "selected_params": {
            **({"n_estimators": 100, "class_weight": "balanced_subsample",
                "n_jobs": n_jobs} if model_name == "RF" else
               {"solver": "liblinear", "max_iter": 1000,
                "class_weight": "balanced", "transform": "log1p_then_StandardScaler"}),
            parameter_name: chosen[parameter_name], "random_state": seed,
        },
        "validation_trials": trials,
        "selected_validation_average_precision": chosen["validation_average_precision"],
        "n_train": int(train.sum()), "n_val": int(validation.sum()),
        "n_refit": int(development.sum()),
        "selection_seconds": selection_seconds,
        "refit_seconds": time.perf_counter() - refit_start,
        "refit_on_train_plus_validation": True,
    }
    if model_name == "RF":
        details["feature_importance_impurity"] = {
            name: float(value) for name, value in zip(names, model.feature_importances_)
        }
    return model, details


def _model_id(dataset: str, seed: int, generator: str, model: str,
              feature_set: str) -> str:
    return f"{dataset}_seed{seed}_{generator}_{model}_{feature_set}"


def run_evaluations(project_root: Path, scenario: str = "main",
                    datasets: Iterable[str] = ("movielens", "ucimessages"),
                    seeds: Iterable[int] = SEEDS,
                    models: Iterable[str] = ("RF", "LR"),
                    features: Iterable[str] = CORE_FEATURES,
                    ablations: bool = False, n_jobs: int = 2,
                    cross_generator: bool = True,
                    generators: Iterable[str] = GENERATORS) -> dict[str, int]:
    root = Path(project_root).resolve()
    output = root / "results" / "evaluation" / scenario
    for subdir in ["runs", "predictions", "models", "failures"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)
    source_hash = _file_hash(Path(__file__))
    models, features, datasets, seeds = list(models), list(features), list(datasets), list(seeds)
    generators = list(generators)
    if not generators or len(set(generators)) != len(generators) or any(
            generator not in GENERATORS for generator in generators):
        raise ValueError("Generators must be a nonempty unique selection from G0, G1, G2, G3")
    counts = {"models_fitted": 0, "models_resumed": 0,
              "evaluations_written": 0, "evaluations_resumed": 0, "failures": 0}
    with threadpool_limits(limits=n_jobs):
        for dataset in datasets:
            for seed in seeds:
                cache_dir = root / "results" / "cache" / scenario / dataset / f"seed_{seed}"
                cache_failure_path = output / "failures" / f"cache_{dataset}_seed{seed}.json"
                try:
                    caches = {generator: load_cache(cache_dir / f"{generator}.npz")
                              for generator in generators}
                except Exception as error:
                    counts["failures"] += 1
                    _write_json(cache_failure_path, {
                        "status": "failed", "resolved": False,
                        "scenario": scenario, "dataset": dataset, "seed": seed,
                        "stage": "cache_validation", "error_type": type(error).__name__,
                        "error": str(error),
                    })
                    print(f"FAILED cache {dataset} seed={seed}: {error}", flush=True)
                    continue
                if cache_failure_path.exists():
                    _write_json(cache_failure_path, {
                        **(_read_json(cache_failure_path) or {}), "resolved": True,
                    })
                work = [(model, feature) for model in models for feature in features]
                if ablations and "RF" in models:
                    work.extend(("RF", feature) for feature in ABLATIONS
                                if ("RF", feature) not in work)
                for source_generator, source in caches.items():
                    for model_name, feature_set in work:
                        identifier = _model_id(dataset, seed, source_generator,
                                               model_name, feature_set)
                        target_generator = None
                        try:
                            names = selected_names(source, dataset, feature_set)
                            config = {
                                "schema_version": SCHEMA_VERSION, "scenario": scenario,
                                "dataset": dataset, "seed": seed,
                                "source_generator": source_generator,
                                "model": model_name, "features": feature_set,
                                "feature_names": names, "n_jobs": n_jobs,
                                "source_cache_fingerprint": source.fingerprint,
                                "evaluation_source_sha256": source_hash,
                                "python_version": platform.python_version(),
                                "numpy_version": np.__version__,
                                "sklearn_version": sklearn.__version__,
                            }
                            model_fingerprint = _digest(config)
                            model_path = output / "models" / f"{identifier}.joblib"
                            model_metadata_path = model_path.with_suffix(".json")
                            saved = _read_json(model_metadata_path)
                            if (saved and saved.get("model_fingerprint") == model_fingerprint
                                    and model_path.exists()):
                                model = joblib.load(model_path)
                                fitted = saved["fitting"]
                                counts["models_resumed"] += 1
                            else:
                                print(f"FIT {scenario} {identifier}", flush=True)
                                model, fitted = fit_selected_model(source, names, model_name,
                                                                   seed, n_jobs)
                                temporary = model_path.with_suffix(".joblib.tmp")
                                joblib.dump(model, temporary, compress=3)
                                os.replace(temporary, model_path)
                                _write_json(model_metadata_path, {
                                    **config, "model_fingerprint": model_fingerprint,
                                    "fitting": fitted,
                                })
                                counts["models_fitted"] += 1
                            targets = (generators if cross_generator and feature_set in CORE_FEATURES
                                       else [source_generator])
                            for target_generator in targets:
                                target = caches[target_generator]
                                run_id = f"{identifier}_to_{target_generator}"
                                run_path = output / "runs" / f"{run_id}.json"
                                prediction_path = output / "predictions" / f"{run_id}.npz"
                                run_fingerprint = _digest({
                                    "model": model_fingerprint, "target": target.fingerprint})
                                previous = _read_json(run_path)
                                if (previous and previous.get("run_fingerprint") == run_fingerprint
                                        and prediction_path.exists()):
                                    counts["evaluations_resumed"] += 1
                                    continue
                                start = time.perf_counter()
                                mask = target.split == 2
                                row_ids = np.flatnonzero(mask)
                                scores = anomaly_scores(model, select_columns(target, names)[mask])
                                y = target.y[mask]
                                metrics = compute_metrics(y, scores)
                                if "pair_seen_before" not in target.names:
                                    raise ValueError("Novel-normal analysis requires pair_seen_before")
                                seen = target.X[mask, target.names.index("pair_seen_before")] > 0
                                novel_mask = (y == 1) | ~seen
                                if len(np.unique(y[novel_mask])) == 2:
                                    novel_metrics = compute_metrics(y[novel_mask], scores[novel_mask])
                                    novel_status = "complete"
                                else:
                                    novel_metrics = None
                                    novel_status = "single_class_after_filter"
                                temporary = prediction_path.with_suffix(".npz.tmp")
                                with temporary.open("wb") as stream:
                                    np.savez_compressed(stream, y=y, scores=scores,
                                                        row_id=row_ids, t=target.t[mask],
                                                        u=target.u[mask], v=target.v[mask],
                                                        novel_normal_subset=novel_mask)
                                os.replace(temporary, prediction_path)
                                record = {
                                    **config, "status": "complete", "run_id": run_id,
                                    "run_fingerprint": run_fingerprint,
                                    "model_fingerprint": model_fingerprint,
                                    "target_generator": target_generator,
                                    "target_cache_fingerprint": target.fingerprint,
                                    "n_train": fitted["n_train"], "n_val": fitted["n_val"],
                                    "n_test": int(mask.sum()), "params": fitted["selected_params"],
                                    "selected_validation_average_precision": fitted[
                                        "selected_validation_average_precision"],
                                    "metrics": metrics,
                                    "novel_normal_metrics": novel_metrics,
                                    "novel_normal_status": novel_status,
                                    "normal_previously_seen_fraction": float(seen[y == 0].mean()),
                                    "times": {key: fitted[key] for key in
                                              ["selection_seconds", "refit_seconds"]},
                                    "prediction_file": str(prediction_path.relative_to(root)),
                                    "model_metadata_file": str(model_metadata_path.relative_to(root)),
                                    "tie_breaking": "ascending preserved target-cache row index",
                                    "feature_importance_impurity": fitted.get(
                                        "feature_importance_impurity"),
                                }
                                record["times"]["evaluation_seconds"] = time.perf_counter() - start
                                _write_json(run_path, record)
                                counts["evaluations_written"] += 1
                            failure_path = output / "failures" / f"{identifier}.json"
                            if failure_path.exists():
                                # Preserve the failed record while marking successful recovery.
                                failed = _read_json(failure_path) or {}
                                _write_json(failure_path, {**failed, "resolved": True,
                                                          "resolved_model_fingerprint": model_fingerprint})
                        except Exception as error:
                            counts["failures"] += 1
                            _write_json(output / "failures" / f"{identifier}.json", {
                                "status": "failed", "resolved": False,
                                "scenario": scenario, "dataset": dataset, "seed": seed,
                                "source_generator": source_generator, "model": model_name,
                                "target_generator": target_generator,
                                "features": feature_set, "error_type": type(error).__name__,
                                "error": str(error),
                            })
                            print(f"FAILED {identifier}: {type(error).__name__}: {error}", flush=True)
                aggregate_results(output)
                print(f"CHECKPOINT {scenario} {dataset} seed={seed} {_canonical(counts)}", flush=True)
    aggregate_results(output)
    return counts


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    fields = ["scenario", "dataset", "seed", "source_generator", "target_generator",
              "model", "features", "n_train", "n_val", "n_test", "run_id",
              "selected_validation_average_precision", "normal_previously_seen_fraction"]
    row = {key: record.get(key) for key in fields}
    row.update(record["metrics"])
    row.update({f"novel_{key}": value for key, value in
                (record.get("novel_normal_metrics") or {}).items()})
    row.update(record.get("times", {}))
    row["params"] = _canonical(record["params"])
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (list(dict.fromkeys(key for row in rows for key in row)) or
              ["scenario", "dataset", "source_generator", "target_generator", "model", "features"])
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate_results(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild tables, excluding successful but now-invalidated old records."""
    output = Path(output)
    root = output.parents[2]
    source_hash = _file_hash(Path(__file__))
    cache_fingerprints: dict[Path, str | None] = {}

    def current_cache_fingerprint(record: dict[str, Any], generator: str) -> str | None:
        path = (root / "results" / "cache" / record["scenario"] / record["dataset"] /
                f"seed_{record['seed']}" / f"{generator}.npz")
        if path not in cache_fingerprints:
            metadata_path = path.with_suffix(".json")
            metadata = _read_json(metadata_path)
            cache_fingerprints[path] = (
                _digest({"npz_sha256": _file_hash(path),
                         "metadata_sha256": _file_hash(metadata_path)})
                if path.exists() and metadata and metadata.get("status") == "complete"
                else None)
        return cache_fingerprints[path]

    records = []
    for path in sorted((output / "runs").glob("*.json")):
        record = _read_json(path)
        if not record or record.get("status") != "complete":
            continue
        if record.get("evaluation_source_sha256") != source_hash:
            continue
        if (record.get("source_cache_fingerprint") != current_cache_fingerprint(
                record, record["source_generator"]) or
                record.get("target_cache_fingerprint") != current_cache_fingerprint(
                    record, record["target_generator"])):
            continue
        model_metadata = _read_json(root / record["model_metadata_file"])
        if (not model_metadata or model_metadata.get("model_fingerprint") !=
                record.get("model_fingerprint") or
                not (root / record["prediction_file"]).exists()):
            continue
        records.append(record)
    rows = [_flatten_record(record) for record in records]
    _write_csv(output / "metrics.csv", rows)
    dimensions = ["scenario", "dataset", "source_generator", "target_generator", "model", "features"]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in dimensions)].append(row)
    summary = []
    metrics = ["average_precision", "pr_auc_trapezoid", "roc_auc",
               "precision_at_k", "recall_at_k", "anomaly_fraction",
               "novel_average_precision", "novel_roc_auc", "novel_precision_at_k",
               "normal_previously_seen_fraction"]
    for key, group in sorted(groups.items()):
        row = dict(zip(dimensions, key))
        row["n_seeds"] = len(group)
        row["seeds"] = ";".join(str(item["seed"]) for item in sorted(group, key=lambda v: v["seed"]))
        for metric in metrics:
            values = [item[metric] for item in group if item.get(metric) is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    _write_csv(output / "summary.csv", summary)
    return rows, summary
