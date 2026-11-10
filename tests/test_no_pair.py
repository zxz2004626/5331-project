"""Attribution control must drop pair history and reuse source parameters only."""

import json
import platform

import numpy as np
import pytest
import sklearn

from graphbench import evaluate
from graphbench.workflows import no_pair_ablation as ablation


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    directory = tmp_path / "results/cache/main/ucimessages/seed_11"
    directory.mkdir(parents=True)
    path = directory / "G0.npz"
    names = ["signal", "pair_seen_before", "recent"]
    X = np.tile(np.array([[1, 1, 2], [8, 0, 4], [2, 0, 3], [7, 0, 5]], dtype=np.float32), (3, 1))
    y = np.tile([0, 1, 0, 1], 3).astype(np.int8)
    np.savez_compressed(
        path,
        X=X,
        names=np.array(names),
        y=y,
        t=np.arange(12),
        u=np.zeros(12, dtype=int),
        v=np.ones(12, dtype=int),
        split=np.repeat([0, 1, 2], 4),
    )
    path.with_suffix(".json").write_text(json.dumps({"status": "complete", "feature_names": names}))
    cache = evaluate.load_cache(path)
    source_path = ablation.source_metadata_path(tmp_path, "ucimessages", 11, "G0")
    source_path.parent.mkdir(parents=True)
    source_path.with_suffix(".joblib").write_bytes(b"metadata-only-fixture")
    source = {
        "scenario": "main",
        "dataset": "ucimessages",
        "seed": 11,
        "source_generator": "G0",
        "model": "RF",
        "features": "extended",
        "source_cache_fingerprint": cache.fingerprint,
        "feature_names": names,
        "evaluation_source_sha256": evaluate.evaluation_code_hash(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "model_fingerprint": "source-model",
        "fitting": {
            "selected_params": {
                "n_estimators": 100,
                "class_weight": "balanced_subsample",
                "random_state": 11,
                "min_samples_leaf": 5,
            },
            "refit_on_train_plus_validation": True,
            "selected_validation_average_precision": 0.77,
        },
    }
    source_path.write_text(json.dumps(source))
    calls = []

    class SpyModel:
        def fit(self, X_fit, y_fit):
            calls.append(("fit", X_fit.copy(), y_fit.copy()))
            self.classes_ = np.array([0, 1])
            self.feature_importances_ = np.ones(X_fit.shape[1]) / X_fit.shape[1]
            return self

        def predict_proba(self, X_test):
            calls.append(("predict", X_test.copy()))
            score = X_test[:, 0] / 10
            return np.column_stack([1 - score, score])

    def spy_build(model, parameter, seed, n_jobs):
        calls.append(("build", model, parameter, seed, n_jobs))
        return SpyModel()

    monkeypatch.setattr(ablation, "build_model", spy_build)
    return tmp_path, X, y, source_path, source, calls


def test_reuses_source_parameter_once_without_seen_or_test_training(fixture):
    root, X, y, source_path, source, calls = fixture
    record, fitted = ablation.run_one(root, "ucimessages", 11, "G0", n_jobs=2)
    assert fitted and record["status"] == "complete"
    assert calls[0] == ("build", "RF", 5, 11, 2)
    assert len([call for call in calls if call[0] == "fit"]) == 1
    np.testing.assert_array_equal(calls[1][1], X[:8, [0, 2]])
    np.testing.assert_array_equal(calls[1][2], y[:8])
    np.testing.assert_array_equal(calls[2][1], X[8:, [0, 2]])
    assert record["feature_names"] == ["signal", "recent"]
    assert record["params"]["min_samples_leaf"] == 5
    assert record["hyperparameter_search_performed"] is False
    assert record["n_refit"] == 8 and record["n_test"] == 4
    assert record["metrics"]["average_precision"] == 1
    assert record["novel_normal_metrics"]["n"] == 3
    assert record["normal_previously_seen_fraction"] == 0.5
    with np.load(root / record["prediction_file"]) as predictions:
        assert predictions["row_id"].tolist() == [8, 9, 10, 11]
        assert predictions["novel_normal_subset"].tolist() == [False, True, True, True]
    resumed, fitted = ablation.run_one(root, "ucimessages", 11, "G0", n_jobs=2)
    assert not fitted and resumed == record and len(calls) == 3


def test_source_parameter_and_prediction_integrity_invalidate_checkpoint(fixture):
    root, X, y, source_path, source, calls = fixture
    first, _ = ablation.run_one(root, "ucimessages", 11, "G0")
    (root / first["prediction_file"]).write_bytes(b"corrupt")
    repaired, fitted = ablation.run_one(root, "ucimessages", 11, "G0")
    assert fitted and repaired["run_fingerprint"] == first["run_fingerprint"]
    source["fitting"]["selected_params"]["min_samples_leaf"] = 1
    source["model_fingerprint"] = "changed-source-model"
    source_path.write_text(json.dumps(source))
    changed, fitted = ablation.run_one(root, "ucimessages", 11, "G0")
    assert fitted and changed["params"]["min_samples_leaf"] == 1
    assert changed["run_fingerprint"] != repaired["run_fingerprint"]
    assert len([call for call in calls if call[0] == "fit"]) == 3


def test_mismatched_source_does_not_fit(fixture):
    root, X, y, source_path, source, calls = fixture
    source["source_cache_fingerprint"] = "another-cache"
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="source_cache_fingerprint"):
        ablation.run_one(root, "ucimessages", 11, "G0")
    assert calls == []


def test_available_only_skips_missing_sources_without_fitting(tmp_path):
    counts = ablation.run_no_pair(
        tmp_path, datasets=["ucimessages"], seeds=[11], generators=["G0"], available_only=True
    )
    assert counts == {"fitted": 0, "resumed": 0, "unavailable": 1, "failed": 0, "complete_total": 0}
    assert not (tmp_path / "results/no_pair/metrics.csv").exists()


def test_summary_requires_current_complete_outputs_and_removes_stale_table(fixture, monkeypatch):
    root, X, y, source_path, source, calls = fixture
    ablation.run_one(root, "ucimessages", 11, "G0")
    status = ablation.aggregate(root)
    assert status["status"] == "partial" and status["completed_current_runs"] == 1
    assert status["expected_runs"] == 40
    assert not (root / "results/no_pair/metrics.csv").exists()
    # Narrow the expected run universe for a small complete-aggregation fixture.
    monkeypatch.setattr(ablation, "DATASETS", ["ucimessages"])
    monkeypatch.setattr(ablation, "SEEDS", [11])
    monkeypatch.setattr(ablation, "GENERATORS", ["G0"])
    status = ablation.aggregate(root)
    assert status["status"] == "complete" and status["expected_runs"] == 1
    csv = root / "results/no_pair/metrics.csv"
    assert csv.exists() and "novel_average_precision" in csv.read_text()
    source_path.write_text(source_path.read_text() + "\n")
    status = ablation.aggregate(root)
    assert status["status"] == "partial" and len(status["stale_runs"]) == 1
    assert not csv.exists()
