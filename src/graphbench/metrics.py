"""Anomaly-positive ranking metrics with deterministic treatment of ties."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


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
