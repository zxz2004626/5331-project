"""Coordinate resumable fitting and cross-generator evaluation.

Cache validation, metrics, model selection, result I/O and aggregation live in
separate modules. Public imports here remain available to existing callers and
serialized estimators from the previous layout.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from .artifacts import _canonical, _digest, _file_hash, _read_json, _write_csv, _write_json
from .cache import Cache, load_cache, select_columns, selected_names
from .constants import ABLATIONS, CORE_FEATURES, GENERATORS, SCHEMA_VERSION, SEEDS
from .metrics import compute_metrics
from .models import NonnegativeLog1p, anomaly_scores, build_model, fit_selected_model
from .provenance import evaluation_code_hash
from .summaries import aggregate_results


def _model_id(dataset: str, seed: int, generator: str, model: str, feature_set: str) -> str:
    return f"{dataset}_seed{seed}_{generator}_{model}_{feature_set}"


def run_evaluations(
    project_root: Path,
    scenario: str = "main",
    datasets: Iterable[str] = ("movielens", "ucimessages"),
    seeds: Iterable[int] = SEEDS,
    models: Iterable[str] = ("RF", "LR"),
    features: Iterable[str] = CORE_FEATURES,
    ablations: bool = False,
    n_jobs: int = 2,
    cross_generator: bool = True,
    generators: Iterable[str] = GENERATORS,
) -> dict[str, int]:
    root = Path(project_root).resolve()
    output = root / "results" / "evaluation" / scenario
    for subdir in ["runs", "predictions", "models", "failures"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)
    source_hash = evaluation_code_hash()
    models, features, datasets, seeds = list(models), list(features), list(datasets), list(seeds)
    generators = list(generators)
    if (
        not generators
        or len(set(generators)) != len(generators)
        or any(generator not in GENERATORS for generator in generators)
    ):
        raise ValueError("Generators must be a nonempty unique selection from G0, G1, G2, G3")
    counts = {
        "models_fitted": 0,
        "models_resumed": 0,
        "evaluations_written": 0,
        "evaluations_resumed": 0,
        "failures": 0,
    }
    with threadpool_limits(limits=n_jobs):
        for dataset in datasets:
            for seed in seeds:
                cache_dir = root / "results" / "cache" / scenario / dataset / f"seed_{seed}"
                cache_failure_path = output / "failures" / f"cache_{dataset}_seed{seed}.json"
                try:
                    caches = {
                        generator: load_cache(cache_dir / f"{generator}.npz")
                        for generator in generators
                    }
                except Exception as error:
                    counts["failures"] += 1
                    _write_json(
                        cache_failure_path,
                        {
                            "status": "failed",
                            "resolved": False,
                            "scenario": scenario,
                            "dataset": dataset,
                            "seed": seed,
                            "stage": "cache_validation",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
                    print(f"FAILED cache {dataset} seed={seed}: {error}", flush=True)
                    continue
                if cache_failure_path.exists():
                    _write_json(
                        cache_failure_path,
                        {
                            **(_read_json(cache_failure_path) or {}),
                            "resolved": True,
                        },
                    )
                work = [(model, feature) for model in models for feature in features]
                if ablations and "RF" in models:
                    work.extend(
                        ("RF", feature) for feature in ABLATIONS if ("RF", feature) not in work
                    )
                for source_generator, source in caches.items():
                    for model_name, feature_set in work:
                        identifier = _model_id(
                            dataset, seed, source_generator, model_name, feature_set
                        )
                        target_generator = None
                        try:
                            names = selected_names(source, dataset, feature_set)
                            config = {
                                "schema_version": SCHEMA_VERSION,
                                "scenario": scenario,
                                "dataset": dataset,
                                "seed": seed,
                                "source_generator": source_generator,
                                "model": model_name,
                                "features": feature_set,
                                "feature_names": names,
                                "n_jobs": n_jobs,
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
                            if (
                                saved
                                and saved.get("model_fingerprint") == model_fingerprint
                                and model_path.exists()
                            ):
                                model = joblib.load(model_path)
                                fitted = saved["fitting"]
                                counts["models_resumed"] += 1
                            else:
                                print(f"FIT {scenario} {identifier}", flush=True)
                                model, fitted = fit_selected_model(
                                    source, names, model_name, seed, n_jobs
                                )
                                temporary = model_path.with_suffix(".joblib.tmp")
                                joblib.dump(model, temporary, compress=3)
                                os.replace(temporary, model_path)
                                _write_json(
                                    model_metadata_path,
                                    {
                                        **config,
                                        "model_fingerprint": model_fingerprint,
                                        "fitting": fitted,
                                    },
                                )
                                counts["models_fitted"] += 1
                            targets = (
                                generators
                                if cross_generator and feature_set in CORE_FEATURES
                                else [source_generator]
                            )
                            for target_generator in targets:
                                target = caches[target_generator]
                                run_id = f"{identifier}_to_{target_generator}"
                                run_path = output / "runs" / f"{run_id}.json"
                                prediction_path = output / "predictions" / f"{run_id}.npz"
                                run_fingerprint = _digest(
                                    {"model": model_fingerprint, "target": target.fingerprint}
                                )
                                previous = _read_json(run_path)
                                if (
                                    previous
                                    and previous.get("run_fingerprint") == run_fingerprint
                                    and prediction_path.exists()
                                ):
                                    counts["evaluations_resumed"] += 1
                                    continue
                                start = time.perf_counter()
                                mask = target.split == 2
                                row_ids = np.flatnonzero(mask)
                                scores = anomaly_scores(model, select_columns(target, names)[mask])
                                y = target.y[mask]
                                metrics = compute_metrics(y, scores)
                                if "pair_seen_before" not in target.names:
                                    raise ValueError(
                                        "Novel-normal analysis requires pair_seen_before"
                                    )
                                seen = target.X[mask, target.names.index("pair_seen_before")] > 0
                                novel_mask = (y == 1) | ~seen
                                if len(np.unique(y[novel_mask])) == 2:
                                    novel_metrics = compute_metrics(
                                        y[novel_mask], scores[novel_mask]
                                    )
                                    novel_status = "complete"
                                else:
                                    novel_metrics = None
                                    novel_status = "single_class_after_filter"
                                temporary = prediction_path.with_suffix(".npz.tmp")
                                with temporary.open("wb") as stream:
                                    np.savez_compressed(
                                        stream,
                                        y=y,
                                        scores=scores,
                                        row_id=row_ids,
                                        t=target.t[mask],
                                        u=target.u[mask],
                                        v=target.v[mask],
                                        novel_normal_subset=novel_mask,
                                    )
                                os.replace(temporary, prediction_path)
                                record = {
                                    **config,
                                    "status": "complete",
                                    "run_id": run_id,
                                    "run_fingerprint": run_fingerprint,
                                    "model_fingerprint": model_fingerprint,
                                    "target_generator": target_generator,
                                    "target_cache_fingerprint": target.fingerprint,
                                    "n_train": fitted["n_train"],
                                    "n_val": fitted["n_val"],
                                    "n_test": int(mask.sum()),
                                    "params": fitted["selected_params"],
                                    "selected_validation_average_precision": fitted[
                                        "selected_validation_average_precision"
                                    ],
                                    "metrics": metrics,
                                    "novel_normal_metrics": novel_metrics,
                                    "novel_normal_status": novel_status,
                                    "normal_previously_seen_fraction": float(seen[y == 0].mean()),
                                    "times": {
                                        key: fitted[key]
                                        for key in ["selection_seconds", "refit_seconds"]
                                    },
                                    "prediction_file": prediction_path.relative_to(root).as_posix(),
                                    "model_metadata_file": model_metadata_path.relative_to(
                                        root
                                    ).as_posix(),
                                    "tie_breaking": "ascending preserved target-cache row index",
                                    "feature_importance_impurity": fitted.get(
                                        "feature_importance_impurity"
                                    ),
                                }
                                record["times"]["evaluation_seconds"] = time.perf_counter() - start
                                _write_json(run_path, record)
                                counts["evaluations_written"] += 1
                            failure_path = output / "failures" / f"{identifier}.json"
                            if failure_path.exists():
                                # Preserve the failed record while marking successful recovery.
                                failed = _read_json(failure_path) or {}
                                _write_json(
                                    failure_path,
                                    {
                                        **failed,
                                        "resolved": True,
                                        "resolved_model_fingerprint": model_fingerprint,
                                    },
                                )
                        except Exception as error:
                            counts["failures"] += 1
                            _write_json(
                                output / "failures" / f"{identifier}.json",
                                {
                                    "status": "failed",
                                    "resolved": False,
                                    "scenario": scenario,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "source_generator": source_generator,
                                    "model": model_name,
                                    "target_generator": target_generator,
                                    "features": feature_set,
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                },
                            )
                            print(
                                f"FAILED {identifier}: {type(error).__name__}: {error}", flush=True
                            )
                aggregate_results(output)
                print(
                    f"CHECKPOINT {scenario} {dataset} seed={seed} {_canonical(counts)}", flush=True
                )
    aggregate_results(output)
    return counts
