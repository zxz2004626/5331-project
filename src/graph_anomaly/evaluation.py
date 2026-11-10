"""Resumable model fitting, same-generator and cross-generator evaluation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from .artifacts import (
    _canonical,
    _digest,
    _read_json,
    _write_json,
    _write_npz,
    evaluation_code_hash,
)
from .config import ABLATIONS, CORE_FEATURES, GENERATORS, ROOT, SCHEMA_VERSION, SEEDS
from .data import load_cache, select_columns, selected_names
from .metrics import compute_metrics
from .models import anomaly_scores, fit_selected_model
from .results import aggregate_results


def _model_id(dataset: str, seed: int, generator: str, model: str, feature_set: str) -> str:
    return f"{dataset}_seed{seed}_{generator}_{model}_{feature_set}"


@dataclass
class FittedRun:
    """A selected estimator and the metadata required to evaluate it."""

    identifier: str
    config: dict
    model: object
    fitting: dict
    fingerprint: str
    metadata_path: Path


class EvaluationSession:
    """Share output paths, provenance, and progress across related model jobs."""

    def __init__(self, root, scenario, n_jobs, generators, cross_generator):
        self.root = Path(root).resolve()
        self.output = self.root / "results/evaluation" / scenario
        self.scenario = scenario
        self.n_jobs = n_jobs
        self.generators = generators
        self.cross_generator = cross_generator
        self.source_hash = evaluation_code_hash()
        self.counts = dict.fromkeys(
            [
                "models_fitted",
                "models_resumed",
                "evaluations_written",
                "evaluations_resumed",
                "failures",
            ],
            0,
        )
        for name in ["runs", "predictions", "models", "failures"]:
            (self.output / name).mkdir(parents=True, exist_ok=True)

    def load_caches(self, dataset, seed):
        directory = self.root / f"results/cache/{self.scenario}/{dataset}/seed_{seed}"
        failure = self.output / "failures" / f"cache_{dataset}_seed{seed}.json"
        try:
            caches = {name: load_cache(directory / f"{name}.npz") for name in self.generators}
        except Exception as error:
            self.counts["failures"] += 1
            _write_json(
                failure,
                {
                    "status": "failed",
                    "resolved": False,
                    "scenario": self.scenario,
                    "dataset": dataset,
                    "seed": seed,
                    "stage": "cache_validation",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            print(f"FAILED cache {dataset} seed={seed}: {error}", flush=True)
            return None
        if failure.exists():
            _write_json(failure, {**(_read_json(failure) or {}), "resolved": True})
        return caches

    def fit_or_resume(self, source, config):
        identifier = _model_id(
            config["dataset"],
            config["seed"],
            config["source_generator"],
            config["model"],
            config["features"],
        )
        fingerprint = _digest(config)
        path = self.output / "models" / f"{identifier}.joblib"
        metadata = path.with_suffix(".json")
        saved = _read_json(metadata)
        if saved and saved.get("model_fingerprint") == fingerprint and path.exists():
            model, fitting = joblib.load(path), saved["fitting"]
            self.counts["models_resumed"] += 1
        else:
            print(f"FIT {self.scenario} {identifier}", flush=True)
            model, fitting = fit_selected_model(
                source, config["feature_names"], config["model"], config["seed"], self.n_jobs
            )
            temporary = path.with_suffix(".joblib.tmp")
            joblib.dump(model, temporary, compress=3)
            os.replace(temporary, path)
            _write_json(metadata, {**config, "model_fingerprint": fingerprint, "fitting": fitting})
            self.counts["models_fitted"] += 1
        return FittedRun(identifier, config, model, fitting, fingerprint, metadata)

    def run_model(self, dataset, seed, source_generator, source, model_name, feature_set, caches):
        identifier = _model_id(dataset, seed, source_generator, model_name, feature_set)
        target_generator = None
        config = {
            "schema_version": SCHEMA_VERSION,
            "scenario": self.scenario,
            "dataset": dataset,
            "seed": seed,
            "source_generator": source_generator,
            "model": model_name,
            "features": feature_set,
            "n_jobs": self.n_jobs,
            "source_cache_fingerprint": source.fingerprint,
            "evaluation_source_sha256": self.source_hash,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        }
        failure = self.output / "failures" / f"{identifier}.json"
        try:
            config["feature_names"] = selected_names(source, dataset, feature_set)
            run = self.fit_or_resume(source, config)
            targets = (
                self.generators
                if self.cross_generator and feature_set in CORE_FEATURES
                else [source_generator]
            )
            for target_generator in targets:
                self.evaluate_target(run, target_generator, caches[target_generator])
            if failure.exists():
                _write_json(
                    failure,
                    {
                        **(_read_json(failure) or {}),
                        "resolved": True,
                        "resolved_model_fingerprint": run.fingerprint,
                    },
                )
        except Exception as error:
            self.counts["failures"] += 1
            _write_json(
                failure,
                {
                    "status": "failed",
                    "resolved": False,
                    "scenario": self.scenario,
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
            print(f"FAILED {identifier}: {type(error).__name__}: {error}", flush=True)

    def evaluate_target(self, run, target_generator, target):
        run_id = f"{run.identifier}_to_{target_generator}"
        path = self.output / "runs" / f"{run_id}.json"
        prediction = self.output / "predictions" / f"{run_id}.npz"
        fingerprint = _digest({"model": run.fingerprint, "target": target.fingerprint})
        previous = _read_json(path)
        if previous and previous.get("run_fingerprint") == fingerprint and prediction.exists():
            self.counts["evaluations_resumed"] += 1
            return
        start = time.perf_counter()
        arrays, analysis = _score_target(run.model, target, run.config["feature_names"])
        _write_npz(prediction, **arrays)
        fitting = run.fitting
        record = {
            **run.config,
            **analysis,
            "status": "complete",
            "run_id": run_id,
            "run_fingerprint": fingerprint,
            "model_fingerprint": run.fingerprint,
            "target_generator": target_generator,
            "target_cache_fingerprint": target.fingerprint,
            "n_train": fitting["n_train"],
            "n_val": fitting["n_val"],
            "n_test": len(arrays["y"]),
            "params": fitting["selected_params"],
            "selected_validation_average_precision": fitting[
                "selected_validation_average_precision"
            ],
            "times": {name: fitting[name] for name in ["selection_seconds", "refit_seconds"]},
            "prediction_file": prediction.relative_to(self.root).as_posix(),
            "model_metadata_file": run.metadata_path.relative_to(self.root).as_posix(),
            "tie_breaking": "ascending preserved target-cache row index",
            "feature_importance_impurity": fitting.get("feature_importance_impurity"),
        }
        record["times"]["evaluation_seconds"] = time.perf_counter() - start
        _write_json(path, record)
        self.counts["evaluations_written"] += 1


def _score_target(model, target, names):
    """Score the test period and its first-seen-normal subset with one estimator."""
    mask = target.split == 2
    scores = anomaly_scores(model, select_columns(target, names)[mask])
    y = target.y[mask]
    if "pair_seen_before" not in target.names:
        raise ValueError("Novel-normal analysis requires pair_seen_before")
    seen = target.X[mask, target.names.index("pair_seen_before")] > 0
    novel = (y == 1) | ~seen
    has_both_classes = len(np.unique(y[novel])) == 2
    arrays = dict(
        y=y,
        scores=scores,
        row_id=np.flatnonzero(mask),
        t=target.t[mask],
        u=target.u[mask],
        v=target.v[mask],
        novel_normal_subset=novel,
    )
    analysis = {
        "metrics": compute_metrics(y, scores),
        "novel_normal_metrics": compute_metrics(y[novel], scores[novel])
        if has_both_classes
        else None,
        "novel_normal_status": "complete" if has_both_classes else "single_class_after_filter",
        "normal_previously_seen_fraction": float(seen[y == 0].mean()),
    }
    return arrays, analysis


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
    """Load each dataset/seed once, then fit and cross-test the selected model jobs."""
    models, features, datasets, seeds = list(models), list(features), list(datasets), list(seeds)
    generators = list(generators)
    if (
        not generators
        or len(set(generators)) != len(generators)
        or any(g not in GENERATORS for g in generators)
    ):
        raise ValueError("Generators must be a nonempty unique selection from G0, G1, G2, G3")
    session = EvaluationSession(project_root, scenario, n_jobs, generators, cross_generator)
    work = [(model, feature) for model in models for feature in features]
    if ablations and "RF" in models:
        work.extend(("RF", feature) for feature in ABLATIONS if ("RF", feature) not in work)
    with threadpool_limits(limits=n_jobs):
        for dataset in datasets:
            for seed in seeds:
                caches = session.load_caches(dataset, seed)
                if caches is None:
                    continue
                for source_generator, source in caches.items():
                    for model_name, feature_set in work:
                        session.run_model(
                            dataset, seed, source_generator, source, model_name, feature_set, caches
                        )
                aggregate_results(session.output)
                print(
                    f"CHECKPOINT {scenario} {dataset} seed={seed} {_canonical(session.counts)}",
                    flush=True,
                )
    aggregate_results(session.output)
    return session.counts


for variable in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[variable] = "2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scenario", default="main")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["movielens", "ucimessages"],
        choices=["movielens", "ucimessages"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--models", nargs="+", default=["RF", "LR"], choices=["RF", "LR"])
    parser.add_argument("--generators", nargs="+", default=GENERATORS, choices=GENERATORS)
    parser.add_argument(
        "--features", nargs="+", default=CORE_FEATURES, choices=CORE_FEATURES + ABLATIONS
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Add five RF-only feature-group experiments with diagonal tests",
    )
    parser.add_argument("--same-generator-only", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    if args.aggregate_only:
        rows, summary = aggregate_results(args.root / "results" / "evaluation" / args.scenario)
        print(json.dumps({"rows": len(rows), "summary_rows": len(summary)}))
        return 0
    counts = run_evaluations(
        args.root,
        args.scenario,
        args.datasets,
        args.seeds,
        args.models,
        args.features,
        args.ablations,
        args.n_jobs,
        not args.same_generator_only,
        args.generators,
    )
    print(json.dumps(counts, sort_keys=True), flush=True)
    return 1 if counts["failures"] else 0
