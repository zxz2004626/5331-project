"""Prior-pair feature removal and first-seen interaction controls."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from . import artifacts
from .artifacts import artifact_path
from .config import GENERATORS, SEEDS
from .config import ROOT as PROJECT_ROOT
from .data import load_cache, select_columns
from .metrics import compute_metrics
from .models import anomaly_scores, build_model

DATASETS = ["movielens", "ucimessages"]

REMOVED_FEATURE = "pair_seen_before"

FEATURE_SET = "extended_no_pair"


def model_id(dataset, seed, generator, feature_set=FEATURE_SET):
    return f"{dataset}_seed{seed}_{generator}_RF_{feature_set}"


def source_metadata_path(root, dataset, seed, generator):
    return (
        Path(root)
        / "results/evaluation/main/models"
        / f"{model_id(dataset, seed, generator, 'extended')}.json"
    )


def validate_source(source, cache, dataset, seed, generator, evaluation_hash):
    expected = {
        "scenario": "main",
        "dataset": dataset,
        "seed": seed,
        "source_generator": generator,
        "model": "RF",
        "features": "extended",
        "source_cache_fingerprint": cache.fingerprint,
        "evaluation_source_sha256": evaluation_hash,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f"Source extended model mismatch: {key}")
    if source.get("feature_names") != cache.names:
        raise ValueError("Source model does not use the complete cached extended feature schema")
    if cache.names.count(REMOVED_FEATURE) != 1:
        raise ValueError("Exactly one pair_seen_before column is required")
    fit = source["fitting"]
    params = fit["selected_params"]
    if (
        params.get("n_estimators") != 100
        or params.get("class_weight") != "balanced_subsample"
        or params.get("random_state") != seed
        or params.get("min_samples_leaf") not in (1, 5)
    ):
        raise ValueError("Source RF parameters differ from the predeclared experiment")
    if not fit.get("refit_on_train_plus_validation"):
        raise ValueError("Source model was not refitted on train plus validation")
    if not source.get("model_fingerprint"):
        raise ValueError("Source model lacks its provenance fingerprint")
    return int(params["min_samples_leaf"])


def _prepare_control(root, dataset, seed, generator, n_jobs):
    """Validate the source model and choose the exact ablated feature set and settings."""
    source_path = source_metadata_path(root, dataset, seed, generator)
    source = artifacts._read_json(source_path)
    if source is None or not source_path.with_suffix(".joblib").exists():
        raise FileNotFoundError(f"Completed source extended model is unavailable: {source_path}")
    cache_path = root / f"results/cache/main/{dataset}/seed_{seed}/{generator}.npz"
    cache = load_cache(cache_path)
    evaluation_hash = artifacts.evaluation_code_hash()
    parameter = validate_source(source, cache, dataset, seed, generator, evaluation_hash)
    names = [name for name in cache.names if name != REMOVED_FEATURE]
    configuration = {
        "schema_version": 1,
        "scenario": "main",
        "dataset": dataset,
        "seed": seed,
        "source_generator": generator,
        "target_generator": generator,
        "model": "RF",
        "features": FEATURE_SET,
        "feature_names": names,
        "removed_features": [REMOVED_FEATURE],
        "n_jobs": n_jobs,
        "source_cache_fingerprint": cache.fingerprint,
        "source_model_metadata_file": source_path.relative_to(root).as_posix(),
        "source_model_metadata_sha256": artifacts._file_hash(source_path),
        "source_model_fingerprint": source["model_fingerprint"],
        "evaluation_source_sha256": evaluation_hash,
        "ablation_source_sha256": artifacts._file_hash(Path(__file__)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "hyperparameter_origin": "source_extended_RF_validation_selection_reused_without_retuning",
        "selected_min_samples_leaf": parameter,
    }
    return (source, cache, names, parameter, configuration)


def _fit_control_predictions(cache, names, parameter, seed, n_jobs, identifier, prediction_path):
    """Refit the fixed control settings, score the test subset, and save aligned predictions."""
    X = select_columns(cache, names)
    development, test = (cache.split != 2, cache.split == 2)
    model = build_model("RF", parameter, seed, n_jobs)
    start = time.perf_counter()
    print(f"FIT no-pair {identifier} min_samples_leaf={parameter}", flush=True)
    model.fit(X[development], cache.y[development])
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scores = anomaly_scores(model, X[test])
    y = cache.y[test]
    metrics = compute_metrics(y, scores)
    seen = cache.X[test, cache.names.index(REMOVED_FEATURE)] > 0
    novel = (y == 1) | ~seen
    novel_metrics = (
        compute_metrics(y[novel], scores[novel]) if len(np.unique(y[novel])) == 2 else None
    )
    artifacts._write_npz(
        prediction_path,
        y=y,
        scores=scores,
        row_id=np.flatnonzero(test),
        t=cache.t[test],
        u=cache.u[test],
        v=cache.v[test],
        novel_normal_subset=novel,
    )
    return {
        "n_refit": int(development.sum()),
        "n_test": int(test.sum()),
        "metrics": metrics,
        "novel_normal_metrics": novel_metrics,
        "novel_normal_status": "complete" if novel_metrics else "single_class_after_filter",
        "normal_previously_seen_fraction": float(seen[y == 0].mean()),
        "times": {"refit_seconds": fit_seconds, "evaluation_seconds": time.perf_counter() - start},
        "feature_importance_impurity": {
            name: float(value) for name, value in zip(names, model.feature_importances_)
        },
    }


def run_one(root, dataset, seed, generator, n_jobs=2):
    root = Path(root).resolve()
    source, cache, names, parameter, configuration = _prepare_control(
        root, dataset, seed, generator, n_jobs
    )
    fingerprint = artifacts._digest(configuration)
    identifier = model_id(dataset, seed, generator)
    output = root / "results/no_pair"
    run_path = output / "runs" / f"{identifier}.json"
    prediction_path = output / "predictions" / f"{identifier}.npz"
    previous = artifacts._read_json(run_path)
    if (
        previous
        and previous.get("status") == "complete"
        and (previous.get("run_fingerprint") == fingerprint)
        and prediction_path.exists()
        and (previous.get("prediction_sha256") == artifacts._file_hash(prediction_path))
    ):
        print(f"RESUME no-pair {identifier}", flush=True)
        return (previous, False)
    artifacts._write_json(
        run_path,
        {
            **configuration,
            "status": "running",
            "run_id": identifier,
            "run_fingerprint": fingerprint,
        },
    )
    try:
        analysis = _fit_control_predictions(
            cache, names, parameter, seed, n_jobs, identifier, prediction_path
        )
        record = {
            **configuration,
            "status": "complete",
            "run_id": identifier,
            "run_fingerprint": fingerprint,
            "params": {
                "n_estimators": 100,
                "class_weight": "balanced_subsample",
                "random_state": seed,
                "min_samples_leaf": parameter,
                "n_jobs": n_jobs,
            },
            "n_train": int((cache.split == 0).sum()),
            "n_val": int((cache.split == 1).sum()),
            "n_refit": analysis["n_refit"],
            "n_test": analysis["n_test"],
            "refit_on_train_plus_validation": True,
            "hyperparameter_search_performed": False,
            "source_selected_validation_average_precision": source["fitting"][
                "selected_validation_average_precision"
            ],
            "metrics": analysis["metrics"],
            "novel_normal_metrics": analysis["novel_normal_metrics"],
            "novel_normal_status": analysis["novel_normal_status"],
            "normal_previously_seen_fraction": analysis["normal_previously_seen_fraction"],
            "times": analysis["times"],
            "prediction_file": prediction_path.relative_to(root).as_posix(),
            "prediction_sha256": artifacts._file_hash(prediction_path),
            "tie_breaking": "ascending preserved cache row index",
            "feature_importance_impurity": analysis["feature_importance_impurity"],
        }
        artifacts._write_json(run_path, record)
        return (record, True)
    except Exception as error:
        artifacts._write_json(
            run_path,
            {
                **configuration,
                "status": "failed",
                "run_id": identifier,
                "run_fingerprint": fingerprint,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def aggregate(root):
    """Publish metrics only if every one of the 40 current controls is complete."""
    root = Path(root).resolve()
    output = root / "results/no_pair"
    rows, missing, stale = [], [], []
    script_hash = artifacts._file_hash(Path(__file__))
    evaluation_hash = artifacts.evaluation_code_hash()
    for dataset, seed, generator in itertools.product(DATASETS, SEEDS, GENERATORS):
        identifier = model_id(dataset, seed, generator)
        record = artifacts._read_json(output / "runs" / f"{identifier}.json")
        if not record or record.get("status") != "complete":
            missing.append(identifier)
            continue
        source_path = source_metadata_path(root, dataset, seed, generator)
        prediction_path = artifact_path(root, record["prediction_file"])
        cache_path = root / f"results/cache/main/{dataset}/seed_{seed}/{generator}.npz"
        manifest_path = cache_path.with_suffix(".json")
        paths = [source_path, prediction_path, cache_path, manifest_path]
        current = all(path.exists() for path in paths)
        if current:
            cache_fingerprint = artifacts._digest(
                {
                    "npz_sha256": artifacts._file_hash(cache_path),
                    "metadata_sha256": artifacts._file_hash(manifest_path),
                }
            )
            current = (
                record.get("ablation_source_sha256") == script_hash
                and record.get("evaluation_source_sha256") == evaluation_hash
                and record.get("source_cache_fingerprint") == cache_fingerprint
                and record.get("source_model_metadata_sha256") == artifacts._file_hash(source_path)
                and record.get("prediction_sha256") == artifacts._file_hash(prediction_path)
            )
        if not current:
            stale.append(identifier)
            continue
        row = {
            key: record[key]
            for key in [
                "run_id",
                "dataset",
                "seed",
                "source_generator",
                "target_generator",
                "model",
                "features",
                "n_train",
                "n_val",
                "n_refit",
                "n_test",
                "selected_min_samples_leaf",
                "normal_previously_seen_fraction",
            ]
        }
        row.update(record["metrics"])
        row.update(
            {"novel_" + key: value for key, value in (record["novel_normal_metrics"] or {}).items()}
        )
        row.update(record["times"])
        rows.append(row)
    complete = not missing and not stale
    status = {
        "status": "complete" if complete else "partial",
        "expected_runs": len(DATASETS) * len(SEEDS) * len(GENERATORS),
        "completed_current_runs": len(rows),
        "missing_runs": missing,
        "stale_runs": stale,
        "ablation_source_sha256": script_hash,
        "evaluation_source_sha256": evaluation_hash,
    }
    artifacts._write_json(output / "status.json", status)
    if complete:
        artifacts._write_csv(output / "metrics.csv", rows)
    elif (output / "metrics.csv").exists():
        # Generated summary is recoverable; never leave a stale complete table.
        (output / "metrics.csv").unlink()
    return status


def run_no_pair(
    root=PROJECT_ROOT,
    datasets=DATASETS,
    seeds=SEEDS,
    generators=GENERATORS,
    n_jobs=2,
    available_only=False,
):
    root = Path(root).resolve()
    counts = {"fitted": 0, "resumed": 0, "unavailable": 0, "failed": 0}
    with threadpool_limits(limits=n_jobs):
        for dataset, seed, generator in itertools.product(datasets, seeds, generators):
            source = source_metadata_path(root, dataset, seed, generator)
            if not source.exists() or not source.with_suffix(".joblib").exists():
                counts["unavailable"] += 1
                if available_only:
                    continue
                print(f"UNAVAILABLE {source}", flush=True)
                continue
            try:
                _, fitted = run_one(root, dataset, seed, generator, n_jobs)
                counts["fitted" if fitted else "resumed"] += 1
            except Exception as error:
                counts["failed"] += 1
                print(f"FAILED no-pair {dataset}/{seed}/{generator}: {error}", flush=True)
    status = aggregate(root)
    counts["complete_total"] = status["completed_current_runs"]
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--generators", nargs="+", choices=GENERATORS, default=GENERATORS)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    if args.aggregate_only:
        print(json.dumps(aggregate(args.root), indent=2))
        return 0
    counts = run_no_pair(
        args.root, args.datasets, args.seeds, args.generators, args.n_jobs, args.available_only
    )
    print(json.dumps(counts, indent=2))
    return int(counts["failed"] > 0 or (counts["unavailable"] > 0 and not args.available_only))
