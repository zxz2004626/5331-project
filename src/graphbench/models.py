"""Classical estimators and training/validation-only model selection."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cache import Cache, select_columns


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


def build_model(model_name: str, parameter: float | int, seed: int, n_jobs: int = 2):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=100,
            n_jobs=n_jobs,
            min_samples_leaf=int(parameter),
            class_weight="balanced_subsample",
            random_state=seed,
        )
    if model_name == "LR":
        return Pipeline(
            [
                ("log1p", NonnegativeLog1p()),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        class_weight="balanced",
                        C=float(parameter),
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    positive = np.flatnonzero(model.classes_ == 1)
    if len(positive) != 1:
        raise ValueError("Fitted model does not have one anomaly-positive class")
    return model.predict_proba(X)[:, int(positive[0])]


def fit_selected_model(
    cache: Cache, names: list[str], model_name: str, seed: int, n_jobs: int = 2
) -> tuple[Any, dict[str, Any]]:
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
        trials.append(
            {
                parameter_name: parameter,
                "validation_average_precision": float(
                    average_precision_score(cache.y[validation], scores)
                ),
                "fit_and_validation_seconds": time.perf_counter() - start,
            }
        )
    # Ties retain the first predeclared candidate, without test-dependent choices.
    chosen = max(trials, key=lambda trial: trial["validation_average_precision"])
    model = build_model(model_name, chosen[parameter_name], seed, n_jobs)
    selection_seconds = time.perf_counter() - selection_start
    refit_start = time.perf_counter()
    development = train | validation
    model.fit(X[development], cache.y[development])
    details = {
        "selected_params": {
            **(
                {"n_estimators": 100, "class_weight": "balanced_subsample", "n_jobs": n_jobs}
                if model_name == "RF"
                else {
                    "solver": "liblinear",
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "transform": "log1p_then_StandardScaler",
                }
            ),
            parameter_name: chosen[parameter_name],
            "random_state": seed,
        },
        "validation_trials": trials,
        "selected_validation_average_precision": chosen["validation_average_precision"],
        "n_train": int(train.sum()),
        "n_val": int(validation.sum()),
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
