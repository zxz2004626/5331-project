"""Independent ranking examples and a training-only transformation check."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from graph_anomaly.data import load_cache, select_columns
from graph_anomaly.evaluation import run_evaluations
from graph_anomaly.features import feature_groups
from graph_anomaly.metrics import compute_metrics
from graph_anomaly.models import build_model
from graph_anomaly.results import aggregate_results


class MetricTests(unittest.TestCase):
    def test_perfect_and_reverse_rankings(self):
        y = np.array([0, 0, 1, 1])
        perfect = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(perfect["average_precision"], 1.0)
        self.assertEqual(perfect["roc_auc"], 1.0)
        self.assertEqual(perfect["precision_at_k"], 1.0)
        reverse = compute_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))
        self.assertEqual(reverse["roc_auc"], 0.0)
        self.assertEqual(reverse["precision_at_k"], 0.0)
        # Positives occur at ranks 3 and 4: AP = (1/3 + 2/4) / 2.
        self.assertAlmostEqual(reverse["average_precision"], 5 / 12)

    def test_constant_scores_are_prevalence_ap_and_half_auc(self):
        y = np.array([0, 1, 0, 0, 1, 0, 0, 0])
        metrics = compute_metrics(y, np.ones(8))
        self.assertEqual(metrics["average_precision"], 0.25)
        self.assertEqual(metrics["roc_auc"], 0.5)
        # Cache row order is the documented deterministic tie break.
        self.assertEqual(metrics["precision_at_k"], 0.5)
        self.assertEqual(metrics["recall_at_k"], 0.5)

    def test_k_identity_and_pr_integral_distinction(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        metrics = compute_metrics(y, np.arange(6))
        self.assertEqual(metrics["k"], 3)
        self.assertEqual(metrics["precision_at_k"], metrics["recall_at_k"])
        self.assertEqual(metrics["precision_at_k"], 2 / 3)
        self.assertNotEqual(metrics["average_precision"], metrics["pr_auc_trapezoid"])

    def test_invalid_evaluations_are_not_silently_scored(self):
        for y, scores in [
            (np.ones(4), np.arange(4)),
            (np.array([0, 1]), np.array([np.nan, 1])),
            (np.array([0, 1]), np.array([0.1])),
            (np.array([0, 0.5]), np.array([0.1, 0.3])),
        ]:
            with self.assertRaises(ValueError):
                compute_metrics(y, scores)

    def test_prediction_does_not_fit_scaler_to_test_distribution(self):
        X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 3.0]])
        y = np.array([0, 0, 1, 0, 1, 1])
        model = build_model("LR", 1.0, 11)
        model.fit(X, y)
        expected_mean = np.log1p(X).mean(axis=0)
        fitted_mean = model.named_steps["scale"].mean_.copy()
        fitted_scale = model.named_steps["scale"].scale_.copy()
        model.predict_proba(np.array([[100000.0, 100000.0], [50000.0, 90000.0]]))
        np.testing.assert_allclose(fitted_mean, expected_mean)
        np.testing.assert_array_equal(model.named_steps["scale"].mean_, fitted_mean)
        np.testing.assert_array_equal(model.named_steps["scale"].scale_, fitted_scale)


class CacheEvaluationTests(unittest.TestCase):
    def _fixture(self, root):
        groups = feature_groups(False)
        names = sum(groups.values(), [])
        directory = root / "results/cache/test/ucimessages/seed_11"
        directory.mkdir(parents=True)
        for i in range(4):
            rng = np.random.default_rng(100 + i)
            X = rng.uniform(0, 10, size=(18, len(names)))
            X[:, names.index("pair_seen_before")] = 0
            y = np.tile([0, 0, 1, 0, 1, 0], 3)
            # Target generator labels differ; all three splits retain both classes.
            y = np.roll(y, i)
            split = np.repeat([0, 1, 2], 6)
            t = np.repeat([1, 2, 3], 6)
            order = np.arange(len(names))[::-1] if i == 3 else np.arange(len(names))
            np.savez(
                directory / f"G{i}.npz",
                X=X[:, order],
                names=np.array(names)[order],
                y=y,
                split=split,
                t=t,
                u=np.arange(18),
                v=np.arange(18) + 20,
            )
            (directory / f"G{i}.json").write_text(json.dumps({"status": "complete"}))
        return directory

    def test_cache_status_split_and_column_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = self._fixture(Path(temp))
            cache = load_cache(directory / "G3.npz")
            names = [cache.names[-1], cache.names[0]]
            selected = select_columns(cache, names)
            np.testing.assert_array_equal(selected[:, 0], cache.X[:, -1])
            np.testing.assert_array_equal(selected[:, 1], cache.X[:, 0])
            (directory / "G3.json").write_text('{"status":"failed"}')
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_cache(directory / "G3.npz")
            # A nominally complete cache with overlapping split timestamps is invalid.
            path = directory / "G2.npz"
            with np.load(path) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["t"][:] = 1
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_cache(path)

    def test_cross_target_identity_resume_and_stale_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = self._fixture(root)
            counts = run_evaluations(root, "test", ["ucimessages"], [11], ["LR"], ["tgf"], n_jobs=1)
            self.assertEqual(counts["models_fitted"], 4)
            self.assertEqual(counts["evaluations_written"], 16)
            self.assertEqual(counts["failures"], 0)
            output = root / "results/evaluation/test"
            for target in range(4):
                cache = load_cache(directory / f"G{target}.npz")
                prediction = (
                    output / "predictions" / f"ucimessages_seed11_G0_LR_tgf_to_G{target}.npz"
                )
                with np.load(prediction) as data:
                    np.testing.assert_array_equal(data["y"], cache.y[cache.split == 2])
                    np.testing.assert_array_equal(data["row_id"], np.flatnonzero(cache.split == 2))
            resumed = run_evaluations(
                root, "test", ["ucimessages"], [11], ["LR"], ["tgf"], n_jobs=1
            )
            self.assertEqual(resumed["models_resumed"], 4)
            self.assertEqual(resumed["evaluations_resumed"], 16)
            self.assertEqual(resumed["models_fitted"], 0)
            (directory / "G2.json").write_text('{"status":"failed"}')
            rows, _ = aggregate_results(output)
            # Source G2 and target G2 each invalidate their row/column, with one overlap.
            self.assertEqual(len(rows), 9)

    def test_failed_fit_is_recorded_and_successful_retry_resolves_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._fixture(root)
            with patch(
                "graph_anomaly.evaluation.fit_selected_model",
                side_effect=RuntimeError("test fit failure"),
            ):
                counts = run_evaluations(
                    root, "test", ["ucimessages"], [11], ["LR"], ["tgf"], n_jobs=1
                )
            self.assertEqual(counts["failures"], 4)
            failure = (
                root / "results/evaluation/test/failures" / "ucimessages_seed11_G0_LR_tgf.json"
            )
            failed = json.loads(failure.read_text())
            self.assertEqual(failed["error"], "test fit failure")
            self.assertFalse(failed["resolved"])
            counts = run_evaluations(
                root,
                "test",
                ["ucimessages"],
                [11],
                ["LR"],
                ["tgf"],
                n_jobs=1,
                cross_generator=False,
            )
            self.assertEqual(counts["failures"], 0)
            self.assertTrue(json.loads(failure.read_text())["resolved"])


if __name__ == "__main__":
    unittest.main()
