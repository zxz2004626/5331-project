"""Pinned TGF preparation, baseline reproduction, and timing controls."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.util
import io
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import ROOT


def find_bash() -> str:
    configured = os.environ.get("GRAPH_ANOMALY_BASH")
    if configured:
        executable = shutil.which(configured) or configured
        if not Path(executable).is_file():
            raise FileNotFoundError(f"GRAPH_ANOMALY_BASH does not exist: {configured}")
        return str(executable)
    if os.name == "nt":
        # Prefer Git's Bash over the Windows/WSL launcher named bash.exe.
        git = shutil.which("git")
        candidates = []
        if git:
            install = Path(git).resolve().parent.parent
            candidates += [install / "bin/bash.exe", install / "usr/bin/bash.exe"]
        for variable in ("ProgramFiles", "LOCALAPPDATA"):
            if os.environ.get(variable):
                candidates.append(
                    Path(os.environ[variable])
                    / (
                        "Programs/Git/bin/bash.exe"
                        if variable == "LOCALAPPDATA"
                        else "Git/bin/bash.exe"
                    )
                )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    else:
        executable = shutil.which("bash")
        if executable:
            return executable
    raise FileNotFoundError(
        "Bash is required only for official --fresh-pipeline. Install Git Bash (Windows) or bash/gzip/coreutils (Unix), or set GRAPH_ANOMALY_BASH."
    )


def injection_command(bash, python, script, dataset, percentage):
    # All user paths travel through argv; the shell program contains no interpolation.
    launcher = 'interpreter="$1"; export interpreter; shift; python3() { set -o pipefail; "$interpreter" "$@" | tr -d "\\r"; }; export -f python3; bash "$@"'
    return [
        str(bash),
        "-c",
        launcher,
        "official-injection",
        Path(python).as_posix(),
        script,
        dataset,
        percentage,
    ]


REV = "af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b"

TARGET = ROOT / "vendor/TGF"


def prepare_main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not shutil.which("git"):
        raise FileNotFoundError("Git is required for prepare. Install Git and add it to PATH.")
    if not TARGET.exists():
        TARGET.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/StephanyRajeh/TGF.git", str(TARGET)], check=True
        )
        subprocess.run(["git", "-C", str(TARGET), "checkout", "--detach", REV], check=True)
    rev = subprocess.check_output(
        ["git", "-C", str(TARGET), "rev-parse", "HEAD"], text=True
    ).strip()
    if rev != REV:
        raise RuntimeError(
            f"Existing upstream is {rev}, expected {REV}. Preserve its changes and select the pinned revision explicitly."
        )
    print(f"Verified upstream {rev}")


UPSTREAM = ROOT / "vendor" / "TGF"

OUT = ROOT / "results" / "official"

CONFIG = {
    "movielens": ("Bipartite", "MovieLens", "MovieLens-5Perc-H1000.ipynb"),
    "ucimessages": ("Unipartite", "UCIMessages", "UCIMessages-5Perc-H1000.ipynb"),
}

SEEDS = [42, 0, 1, 2, 3]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def evaluate(y, score, pred):
    n = max(1, int(np.ceil(len(y) * 0.01)))
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "precision_at_1pct": float(np.asarray(y)[np.argsort(-score, kind="stable")[:n]].mean()),
        "precision_threshold_05": float(precision_score(y, pred, zero_division=0)),
        "recall_threshold_05": float(recall_score(y, pred, zero_division=0)),
        "f1_threshold_05": float(f1_score(y, pred, zero_division=0)),
    }


def generate_features(dataset, injected, dest):
    cmd = [sys.executable, str(UPSTREAM / "3-FeatureGeneration" / "main.py"), "-H", "1000"]
    if dataset == "movielens":
        cmd.append("-bip")
    # Passing the official decompressed bytes through stdin preserves row order/tie handling.
    with gzip.open(injected, "rb") as f:
        source = f.read()
    start = time.perf_counter()
    result = subprocess.run(cmd, input=source, capture_output=True, check=True)
    elapsed = time.perf_counter() - start
    with gzip.open(dest, "wb") as f:
        f.write(result.stdout)
    (OUT / f"{dataset}_feature_generation.log").write_bytes(result.stderr)
    return [json.loads(line) for line in result.stdout.splitlines()], elapsed, cmd


def read_cached(path):
    with gzip.open(path, "rb") as f:
        payload = f.read()
    encoding = "utf-16" if payload[:2] in [b"\xff\xfe", b"\xfe\xff"] else "utf-8-sig"
    return [json.loads(line) for line in payload.decode(encoding).splitlines()], encoding


def _evaluate_official_seeds(dataset, x_train, y_train, x_test, y_test, env, learning_seconds):
    """Evaluate the official RF seeds on the unchanged chronological test rows."""
    rows = []
    feature_importances = []
    for seed in SEEDS:
        print(f"[{dataset}] RF seed {seed}", flush=True)
        if seed == 42:
            rf, fit_seconds = (env["rf_classifier"], learning_seconds)
        else:
            start = time.perf_counter()
            rf = RandomForestClassifier(random_state=seed).fit(x_train, y_train)
            fit_seconds = time.perf_counter() - start
        pred_start = time.perf_counter()
        score = rf.predict_proba(x_test)[:, 1]
        pred = rf.predict(x_test)
        predict_seconds = time.perf_counter() - pred_start
        rows.append(
            {
                "dataset": dataset,
                "protocol": "official_post_update_H1000",
                "rf_seed": seed,
                "n_train_balanced": len(y_train),
                "n_test": len(y_test),
                "test_prevalence": float(y_test.mean()),
                "n_features": x_train.shape[1],
                "fit_seconds": fit_seconds,
                "predict_seconds_two_passes": predict_seconds,
                **evaluate(y_test, score, pred),
            }
        )
        for name, importance in zip(x_train.columns, rf.feature_importances_):
            feature_importances.append(
                {
                    "dataset": dataset,
                    "rf_seed": seed,
                    "feature": name,
                    "importance": float(importance),
                }
            )
        np.savez_compressed(
            OUT / f"{dataset}_seed{seed}_predictions.npz",
            row_index=x_test.index.values,
            y=y_test.values,
            score=score,
        )
    return (rows, feature_importances)


def _execute_official_notebook(dataset, notebook_dir, notebook_path, data):
    """Execute the specified upstream learning cells and preserve their console output."""
    spec = importlib.util.spec_from_file_location(
        f"official_utilities_{dataset}", notebook_dir / "utilitiesMe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nb = json.loads(notebook_path.read_text(encoding="utf8"))
    env = {
        "pd": pd,
        "np": np,
        "data": data,
        "flatten_dict": mod.flatten_dict,
        "RandomForestClassifier": RandomForestClassifier,
        "roc_auc_score": roc_auc_score,
    }
    executed = [8, 11, 13, 17, 20, 22, 25, 31, 35, 36, 42, 43, 47, 49]
    log = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(log):
        for i in executed:
            exec(
                compile("".join(nb["cells"][i]["source"]), f"{notebook_path.name}:cell{i}", "exec"),
                env,
            )
    learning_seconds = time.perf_counter() - start
    (OUT / f"{dataset}_notebook_classical.log").write_text(log.getvalue(), encoding="utf8")
    return (env, learning_seconds, executed)


def _compare_official_features(data, cached, dataset):
    """Compare regenerated and released feature records, excluding computation cost."""
    mismatch_count, first_mismatch = (0, None)
    for i, (a, b) in enumerate(zip(data, cached)):
        aa, bb = (
            {k: v for k, v in a.items() if k != "cost"},
            {k: v for k, v in b.items() if k != "cost"},
        )
        if aa != bb:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "row": i,
                    "fields": {
                        k: [aa.get(k), bb.get(k)]
                        for k in aa.keys() | bb.keys()
                        if aa.get(k) != bb.get(k)
                    },
                }
    assert len(data) == len(cached), "Regenerated/cached row counts differ"
    print(f"[{dataset}] {len(data)} rows generated; cache mismatches={mismatch_count}", flush=True)
    return (mismatch_count, first_mismatch)


def run_dataset(dataset):
    OUT.mkdir(parents=True, exist_ok=True)
    kind, title, notebook_name = CONFIG[dataset]
    injected = UPSTREAM / "2-Injection" / kind / dataset / f"{dataset}_5_injected.gz"
    notebook_dir = UPSTREAM / "4-LearningAndTesting" / "HTypeHistoryGraphs" / kind / title
    notebook_path = notebook_dir / notebook_name
    generated_path = OUT / f"{dataset}_5_injected_H1000_regenerated.json.gz"
    print(f"[{dataset}] generating upstream features", flush=True)
    data, generation_seconds, command = generate_features(dataset, injected, generated_path)
    cached, encoding = read_cached(notebook_dir / f"{dataset}_5_injected_H1000.json.gz")
    mismatch_count, first_mismatch = _compare_official_features(data, cached, dataset)

    env, learning_seconds, executed = _execute_official_notebook(
        dataset, notebook_dir, notebook_path, data
    )
    df, x_train, y_train, x_test, y_test = [
        env[k] for k in ["df", "X_train", "y_train", "X_test", "y_test"]
    ]
    assert "is_fraud" not in x_train.columns
    assert set(x_train.index).isdisjoint(x_test.index)
    assert max(x_train.index) < min(x_test.index)
    rows, feature_importances = _evaluate_official_seeds(
        dataset, x_train, y_train, x_test, y_test, env, learning_seconds
    )
    pd.DataFrame(rows).to_csv(OUT / f"{dataset}_metrics.csv", index=False)
    pd.DataFrame(feature_importances).to_csv(OUT / f"{dataset}_feature_importance.csv", index=False)
    # Preserve complete row-aligned matrices for a paired causal comparison.
    np.savez_compressed(
        OUT / f"{dataset}_official_matrix.npz",
        X=df[x_train.columns].to_numpy(dtype=np.float64),
        y=df["is_fraud"].to_numpy(),
        row_index=df.index.to_numpy(),
        columns=np.asarray(x_train.columns, dtype=str),
    )
    metadata = {
        "dataset": dataset,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "command": command,
        "generation_wall_seconds": generation_seconds,
        "injected_sha256": sha256(injected),
        "main_py_sha256": sha256(UPSTREAM / "3-FeatureGeneration" / "main.py"),
        "DSC_lib_py_sha256": sha256(UPSTREAM / "3-FeatureGeneration" / "DSC_lib.py"),
        "notebook_sha256": sha256(notebook_path),
        "executed_unchanged_cell_indices_zero_based": executed,
        "upstream_cached_encoding": encoding,
        "upstream_cache_row_count": len(cached),
        "cache_non_cost_row_mismatches": mismatch_count,
        "cache_first_mismatch": first_mismatch,
        "n_input": len(data),
        "n_anomalies_input": sum(row["is_fraud"] for row in data),
        "n_after_dropna": len(df),
        "n_dropped_warmup": len(data) - len(df),
        "n_train_before_balance": len(env["train_df"]),
        "n_test": len(y_test),
        "train_positive_before_balance": int(env["train_df"]["is_fraud"].sum()),
        "test_positive": int(y_test.sum()),
        "features": list(x_train.columns),
        "train_normal_sampling_seed": 12,
        "train_shuffle_seed": 42,
        "rf_seeds": SEEDS,
        "scope": "Classical chronological 70/30; no sliding-window learning rerun; official provided injection reused.",
    }
    (OUT / f"{dataset}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(
        f"[{dataset}] complete AUC={rows[0]['roc_auc']:.6f}, AP={rows[0]['average_precision']:.6f}",
        flush=True,
    )
    return rows


def snapshot_causal_features(env, u, v, bip):
    """Same definitions as official fill_result, extended to absent endpoints/edges.

    DSC includes active nodes/edges only. An absent object's value and equal count
    are zero; every active object is strictly greater. No zero-valued phantom
    node/edge is inserted to compute statistics.
    """
    out = {}

    def val(counter, key):
        return counter.val(key) if counter.has_counter(key) else 0

    def rank(counter, value):
        return counter.val2pos[value] if value in counter.val2pos else counter.len()

    def stats(prefix, degrees, weighted):
        out[prefix + "number_of_nodes"] = degrees.len()
        for name, counter in [("degrees", degrees), ("weighted_degrees", weighted)]:
            for k in [1, 2]:
                out[f"{prefix}{name}_nb_{'one' if k == 1 else 'two'}"] = counter.distrib.get(k, 0)
            for attr in ["min", "max", "median"]:
                out[f"{prefix}{name}_{attr}"] = getattr(counter, attr)()

    if bip:
        du, dv = env["top_degrees"], env["bot_degrees"]
        wu, wv = env["top_weighted_degrees"], env["bot_weighted_degrees"]
        stats("top_", du, wu)
        stats("bot_", dv, wv)
    else:
        du = dv = env["degrees"]
        wu = wv = env["weighted_degrees"]
        stats("", du, wu)
        if val(du, u) > val(dv, v):
            u, v = v, u
    for endpoint, node, degree, weighted in [("u", u, du, wu), ("v", v, dv, wv)]:
        for name, counter in [("degree", degree), ("weighted_degree", weighted)]:
            value = val(counter, node)
            out[f"{endpoint}_{name}"] = value
            out[f"{endpoint}_same_{name}"] = counter.distrib.get(value, 0)
            out[f"{endpoint}_greater_{name}"] = rank(counter, value)
    links = env["link_weights"]
    out["number_of_links"], out["total_weight"] = links.len(), links.total
    for k in [1, 2]:
        out[f"link_weights_nb_{'one' if k == 1 else 'two'}"] = links.distrib.get(k, 0)
    for attr in ["min", "max", "median"]:
        out[f"link_weights_{attr}"] = getattr(links, attr)()
    # fill_result's endpoint swap is local: original pair key stays canonical.
    key = (u, v) if bip else tuple(sorted((u, v)))
    weight = val(links, key)
    out["u_v_weight"] = weight
    out["u_v_same_weight"] = links.distrib.get(weight, 0)
    out["u_v_greater_weight"] = rank(links, weight)
    return out


def _evaluate_paired_seeds(dataset, x_train, y_train, x_test, y_test):
    """Fit the fixed RF settings on strict pre-arrival features."""
    rows = []
    for seed in SEEDS:
        start = time.perf_counter()
        rf = RandomForestClassifier(random_state=seed).fit(x_train, y_train)
        elapsed = time.perf_counter() - start
        score = rf.predict_proba(x_test)[:, 1]
        pred = rf.predict(x_test)
        rows.append(
            {
                "dataset": dataset,
                "protocol": "strict_timestamp_causal_pre_update_H1000",
                "rf_seed": seed,
                "n_train_balanced": len(y_train),
                "n_test": len(y_test),
                "test_prevalence": float(y_test.mean()),
                "n_features": x_train.shape[1],
                "fit_seconds": elapsed,
                **evaluate(y_test, score, pred),
            }
        )
        np.savez_compressed(
            OUT / f"{dataset}_pre_update_seed{seed}_predictions.npz",
            row_index=x_test.index.values,
            y=y_test.values,
            score=score,
        )
    return rows


def _paired_feature_matrix(dataset, injected, reference_x, columns, env):
    """Batch equal-time feature extraction and verify the original post-arrival matrix."""
    matrix = np.empty_like(reference_x)
    n_checked = 0
    start = time.perf_counter()
    timestamps = []
    with gzip.open(injected, "rt", encoding="utf8") as f:
        for timestamp, group_iter in itertools.groupby(
            enumerate(f), key=lambda item: int(item[1].split()[0])
        ):
            group = []
            for i, line in group_iter:
                t, u, v, label = line.split()
                t = int(t)
                if dataset != "movielens":
                    u, v = (min(u, v), max(u, v))
                group.append((i, t, u, v))
                if i >= 1000:
                    before = snapshot_causal_features(env, u, v, dataset == "movielens")
                    matrix[i - 1000] = [before[c] for c in columns]
                    timestamps.append(t)
            for i, t, u, v in group:
                env["link_arrival"](t, u, v)
                while len(env["Q"]) > 1000:
                    env["link_departure"](*env["Q"].popleft())
                if i >= 1000:
                    after = snapshot_causal_features(env, u, v, dataset == "movielens")
                    actual = np.asarray([after[c] for c in columns])
                    if not np.array_equal(actual, reference_x[i - 1000]):
                        raise AssertionError(
                            f"Post-update verification mismatch at row {i}: "
                            + str(
                                {
                                    c: (actual[j], reference_x[i - 1000, j])
                                    for j, c in enumerate(columns)
                                    if actual[j] != reference_x[i - 1000, j]
                                }
                            )
                        )
                    n_checked += 1
    generation_seconds = time.perf_counter() - start
    assert n_checked == len(matrix)
    return (matrix, n_checked, timestamps, generation_seconds)


def _initialize_upstream_history(dataset):
    """Initialize upstream history functions without executing the original input loop."""
    env = {}
    feature_dir = UPSTREAM / "3-FeatureGeneration"
    source = (feature_dir / "main.py").read_text(encoding="utf8")
    prefix = source.split("# process input links one by one")[0]
    saved_argv, saved_path = (sys.argv[:], sys.path[:])
    try:
        sys.argv = ["main.py", "-H", "1000"] + (["-bip"] if dataset == "movielens" else [])
        sys.path.insert(0, str(feature_dir))
        exec(compile(prefix, str(feature_dir / "main.py"), "exec"), env)
    finally:
        sys.argv, sys.path[:] = (saved_argv, saved_path)
    return env


def run_paired_timing(dataset):
    """Hold stream, row split, training samples, RF fixed; vary current-event inclusion."""
    print(f"[{dataset}] paired timing contrast", flush=True)
    kind = CONFIG[dataset][0]
    injected = UPSTREAM / "2-Injection" / kind / dataset / f"{dataset}_5_injected.gz"
    reference = np.load(OUT / f"{dataset}_official_matrix.npz")
    # Cache once: NpzFile.__getitem__ decompresses a member on every access.
    reference_x = reference["X"]
    columns = list(reference["columns"])
    env = _initialize_upstream_history(dataset)
    matrix, n_checked, timestamps, generation_seconds = _paired_feature_matrix(
        dataset, injected, reference_x, columns, env
    )
    np.savez_compressed(
        OUT / f"{dataset}_pre_update_matrix.npz",
        X=matrix,
        y=reference["y"],
        row_index=reference["row_index"],
        columns=reference["columns"],
    )
    df = pd.DataFrame(matrix, columns=columns, index=reference["row_index"])
    df["is_fraud"] = reference["y"]
    split = round(len(df) / 100 * 70)
    train, test = df.iloc[:split], df.iloc[split:]
    positives = train[train.is_fraud == 1]
    balanced = pd.concat(
        [train[train.is_fraud == 0].sample(n=len(positives), random_state=12), positives]
    ).sample(frac=1, random_state=42)
    x_train, y_train = balanced.drop(columns="is_fraud"), balanced.is_fraud
    x_test, y_test = test.drop(columns="is_fraud"), test.is_fraud
    rows = _evaluate_paired_seeds(dataset, x_train, y_train, x_test, y_test)
    pd.DataFrame(rows).to_csv(OUT / f"{dataset}_paired_timing_metrics.csv", index=False)
    (OUT / f"{dataset}_paired_timing_metadata.json").write_text(
        json.dumps(
            {
                "post_update_rows_checked_exactly": n_checked,
                "generation_plus_post_update_verification_seconds": generation_seconds,
                "absent_value_convention": "value=0, same count=0, greater count=active population size",
                "history": "Last 1000 rows with timestamp strictly earlier than the candidate; score all equal-timestamp candidates before adding any of them.",
                "held_fixed": [
                    "official injected stream",
                    "row order",
                    "warmup 1000",
                    "chronological70/30",
                    "training indices",
                    "RF parameters and seeds",
                ],
                "split_boundary_timestamps": [timestamps[split - 1], timestamps[split]],
                "split_boundary_shares_timestamp": bool(timestamps[split - 1] == timestamps[split]),
                "scope": "Strict timestamp-causal extraction contrast only; official injection uses full-stream node and timestamp universe. Original row split retained, potentially sharing a timestamp at boundary.",
            },
            indent=2,
        ),
        encoding="utf8",
    )
    print(
        f"[{dataset}] paired timing complete: AUC={rows[0]['roc_auc']:.6f}, AP={rows[0]['average_precision']:.6f}",
        flush=True,
    )


def _check_fresh_injection(dest, dataset, actual):
    """Validate counts, ordering, pair legality, and preservation of all observed records."""
    with gzip.open(dest / f"{dataset}_5_injected.gz", "rt") as f:
        events = [tuple(line.split()) for line in f]
    normals = [x[:3] for x in events if x[3] == "0"]
    injected = [x[:3] for x in events if x[3] == "1"]
    assert len(normals) == len(actual) and set(normals) == set(actual)
    assert len(injected) == len(actual) * 5 // 100
    assert len(set((x[:3] for x in events))) == len(events)
    assert all((int(events[i][0]) <= int(events[i + 1][0]) for i in range(len(events) - 1)))
    assert all((x[1] != x[2] for x in events))
    assert not set(injected) & set(actual)
    if dataset == "movielens":
        assert all((x[1].startswith("bottom_") and x[2].startswith("top_") for x in events))
    return (events, normals, injected)


def _run_original_injection_scripts(dest, build, check, dataset):
    """Run exact upstream Git blobs and retain each script console log."""
    source_hashes = {}
    for filename in [build, "inject.sh", check, "random_sample.py"]:
        blob = subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "show", f"HEAD:2-Injection/{filename}"]
        )
        (dest / filename).write_bytes(blob)
        source_hashes[filename] = hashlib.sha256(blob).hexdigest()
    commands, logs = ([], [])
    bash = find_bash()
    for script, pct in [(build, "20"), ("inject.sh", "5"), (check, "5")]:
        command = injection_command(bash, sys.executable, script, dataset, pct)
        run = subprocess.run(
            command, cwd=dest, capture_output=True, text=True, encoding="utf8", errors="replace"
        )
        text = run.stdout + "\n[stderr]\n" + run.stderr
        (dest / f"{script}.log").write_text(text, encoding="utf8")
        if run.returncode != 0 or "PROBLEM!!!" in text:
            raise RuntimeError(f"Official {script} failed: {text}")
        commands.append(
            {"script": script, "arguments": [dataset, pct], "returncode": run.returncode}
        )
        logs.append(text)
    assert "everything seems ok" in logs[-1]
    return (source_hashes, commands)


def _run_preprocessing_notebook(pre, notebook_name, dest):
    """Execute raw-data cleaning in the isolated dataset output directory."""
    notebook = json.loads((pre / notebook_name).read_text(encoding="utf8"))
    executed = []
    log = io.StringIO()
    cwd = Path.cwd()
    try:
        os.chdir(dest)
        env = {}
        with contextlib.redirect_stdout(log):
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"{notebook_name}:cell{i}", "exec"), env)
                    executed.append(i)
    finally:
        os.chdir(cwd)
    (dest / "preprocessing.log").write_text(log.getvalue(), encoding="utf8")
    return executed


def run_fresh_preprocessing_injection(dataset):
    """Execute original raw-data preprocessing and injection in isolated output dirs.

    Git's source blobs avoid Windows checkout CRLF conversion in shell scripts.
    Their logic is not edited. The shell function only resolves python3 to this
    installed interpreter rather than Windows' Microsoft Store alias.
    """
    print(f"[{dataset}] fresh raw preprocessing and official random injection", flush=True)
    if dataset == "movielens":
        pre = UPSTREAM / "1-PreprocessingOfDatasets/BipartiteNetworks/movielens"
        raw_name, notebook_name, build, check = (
            "rel.rating",
            "CleaningMovieLens.ipynb",
            "bip_build_injectable.sh",
            "bip_check.sh",
        )
    else:
        pre = UPSTREAM / "1-PreprocessingOfDatasets/UnipartiteNetworks/opsahl-ucsocial"
        raw_name, notebook_name, build, check = (
            "out.opsahl-ucsocial",
            "CleaningUCIMessages.ipynb",
            "build_injectable.sh",
            "check.sh",
        )
    dest = OUT / "fresh_pipeline" / dataset
    dest.mkdir(parents=True, exist_ok=True)
    prior_check_log = dest / f"{check}.log"
    if prior_check_log.exists() and "PROBLEM!!!" in prior_check_log.read_text(encoding="utf8"):
        shutil.copyfile(prior_check_log, dest / f"{check}.initial_windows_crlf_failure.log")
    shutil.copyfile(pre / raw_name, dest / raw_name)
    executed = _run_preprocessing_notebook(pre, notebook_name, dest)
    cleaned = dest / f"{dataset}.txt"
    upstream_cleaned = pre / f"{dataset}.txt"
    actual = [tuple(line.split()) for line in cleaned.read_text(encoding="utf-8").splitlines()]
    expected = [
        tuple(line.split()) for line in upstream_cleaned.read_text(encoding="utf-8").splitlines()
    ]
    assert actual == expected, "Fresh original preprocessing differs from committed cleaned rows"
    with gzip.open(dest / f"{dataset}.gz", "wb") as f:
        f.write(cleaned.read_bytes())
    source_hashes, commands = _run_original_injection_scripts(dest, build, check, dataset)
    events, normals, injected = _check_fresh_injection(dest, dataset, actual)
    metadata = {
        "dataset": dataset,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
        "raw_input_sha256": sha256(dest / raw_name),
        "preprocessing_notebook_sha256": sha256(pre / notebook_name),
        "preprocessing_unchanged_cells_zero_based": executed,
        "cleaned_file_sha256": sha256(cleaned),
        "upstream_cleaned_file_sha256": sha256(upstream_cleaned),
        "cleaned_rows_equal_in_order": actual == expected,
        "normal_count": len(normals),
        "injected_count": len(injected),
        "total_count": len(events),
        "prevalence": len(injected) / len(events),
        "source_git_blob_sha256": source_hashes,
        "script_commands": commands,
        "source_adaptation": "Exact Git blobs avoid checkout CRLF. The python3 shell function uses the active Python interpreter and removes stdout CR; no algorithm changes.",
        "injected_sha256": sha256(dest / f"{dataset}_5_injected.gz"),
        "original_check_passed": True,
        "independent_checks": [
            "all original normal rows retained",
            "floor(5% normal_count) injected",
            "unique triples",
            "nondecreasing time",
            "no loops",
            "no injected triple in normals",
            "bipartite partitions preserved when applicable",
        ],
        "randomness": "Original sampler is unseeded and shuf is unseeded; this fresh generated stream is saved with SHA256, but regeneration produces another realization.",
        "learning_scope": "No fresh-stream retraining; faithful feature/learning results use the committed stream for exact cached comparison.",
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(
        f"[{dataset}] fresh pipeline passed: {len(normals)} normals + {len(injected)} injected",
        flush=True,
    )


def verify_saved_outputs():
    table = pd.read_csv(OUT / "paired_timing_metrics.csv")
    assert len(table) == 20 and not table.duplicated(["dataset", "protocol", "rf_seed"]).any()
    checked, totals = 0, 0
    for dataset in CONFIG:
        original = np.load(OUT / f"{dataset}_official_matrix.npz")
        causal = np.load(OUT / f"{dataset}_pre_update_matrix.npz")
        for key in ["y", "row_index", "columns"]:
            assert np.array_equal(original[key], causal[key]), (dataset, key)
        assert original["X"].shape == causal["X"].shape
        assert np.isfinite(causal["X"]).all()
        for seed in SEEDS:
            paired = []
            for prefix, protocol in [
                ("", "official_post_update_H1000"),
                ("pre_update_", "strict_timestamp_causal_pre_update_H1000"),
            ]:
                data = np.load(OUT / f"{dataset}_{prefix}seed{seed}_predictions.npz")
                row = table[
                    (table.dataset == dataset)
                    & (table.protocol == protocol)
                    & (table.rf_seed == seed)
                ].iloc[0]
                assert len(data["score"]) == int(row.n_test)
                assert np.isfinite(data["score"]).all() and np.all(
                    (data["score"] >= 0) & (data["score"] <= 1)
                )
                assert abs(roc_auc_score(data["y"], data["score"]) - row.roc_auc) < 1e-12
                assert (
                    abs(average_precision_score(data["y"], data["score"]) - row.average_precision)
                    < 1e-12
                )
                paired.append(data)
                checked += 1
                totals += len(data["score"])
            assert np.array_equal(paired[0]["row_index"], paired[1]["row_index"])
            assert np.array_equal(paired[0]["y"], paired[1]["y"])
    summary = table.groupby(["dataset", "protocol"])[
        ["roc_auc", "average_precision", "precision_at_1pct"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / "paired_timing_summary_flat.csv", index=False)
    a = table[table.protocol == "official_post_update_H1000"].set_index(["dataset", "rf_seed"])
    b = table[table.protocol == "strict_timestamp_causal_pre_update_H1000"].set_index(
        ["dataset", "rf_seed"]
    )
    (
        b[["roc_auc", "average_precision", "precision_at_1pct"]]
        - a[["roc_auc", "average_precision", "precision_at_1pct"]]
    ).reset_index().to_csv(OUT / "paired_timing_deltas.csv", index=False)
    result = {
        "status": "passed",
        "prediction_files_checked": checked,
        "prediction_rows_checked": totals,
        "metrics_recomputed": ["roc_auc", "average_precision"],
        "absolute_tolerance": 1e-12,
        "checks": [
            "exactly 5 RF seeds per dataset/protocol",
            "finite bounded scores",
            "same test labels and row indices across paired protocols",
            "same complete matrix labels/row indices/columns",
            "all prior-timestamp matrix values finite",
        ],
    }
    (OUT / "integrity_checks.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["all", *CONFIG], default="all")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--paired-timing",
        action="store_true",
        help="Only run same-stream pre-update contrast; official matrices must already exist.",
    )
    mode.add_argument(
        "--fresh-pipeline",
        action="store_true",
        help="Only rerun original raw preprocessing and random injection in isolated output directories.",
    )
    mode.add_argument(
        "--verify-output",
        action="store_true",
        help="Recompute metrics from all saved scores and check paired row alignment.",
    )
    args = ap.parse_args()
    if args.verify_output:
        verify_saved_outputs()
        return
    for dataset in CONFIG if args.dataset == "all" else [args.dataset]:
        if args.fresh_pipeline:
            run_fresh_preprocessing_injection(dataset)
        elif args.paired_timing:
            run_paired_timing(dataset)
        else:
            run_dataset(dataset)
    if args.fresh_pipeline:
        return
    frames = [
        pd.read_csv(OUT / f"{d}_metrics.csv") for d in CONFIG if (OUT / f"{d}_metrics.csv").exists()
    ]
    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(OUT / "metrics.csv", index=False)
    merged.groupby(["dataset", "protocol"])[
        ["roc_auc", "average_precision", "precision_at_1pct"]
    ].agg(["mean", "std"]).to_csv(OUT / "summary.csv")
    if args.paired_timing:
        paired = pd.concat(
            [
                pd.read_csv(OUT / f"{d}_paired_timing_metrics.csv")
                for d in CONFIG
                if (OUT / f"{d}_paired_timing_metrics.csv").exists()
            ],
            ignore_index=True,
        )
        pd.concat([merged, paired], ignore_index=True).to_csv(
            OUT / "paired_timing_metrics.csv", index=False
        )
        pd.concat([merged, paired], ignore_index=True).groupby(["dataset", "protocol"])[
            ["roc_auc", "average_precision", "precision_at_1pct"]
        ].agg(["mean", "std"]).to_csv(OUT / "paired_timing_summary.csv")
