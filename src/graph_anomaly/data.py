"""Cleaned dataset loading, chronological splits, and cache validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import GENERATION_SOURCES, _digest, _file_hash, source_digest
from .config import ABLATIONS, CORE_FEATURES, ROOT

SOURCES = {
    "movielens": "1-PreprocessingOfDatasets/BipartiteNetworks/movielens/movielens.txt",
    "ucimessages": "1-PreprocessingOfDatasets/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def code_hash() -> str:
    """Only generation dependencies invalidate expensive feature caches."""
    return source_digest(Path(__file__).parent, GENERATION_SOURCES)


def load_dataset(name: str):
    p = ROOT / "vendor/TGF" / SOURCES[name]
    df = pd.read_csv(p, sep=r"\s+", names=["t", "source", "target"])
    df = df.sort_values("t", kind="stable")
    bip = name == "movielens"
    if bip:
        # Official cleaned data is movie first (bottom_*), user second (top_*).
        left = sorted(df.source.unique())
        right = sorted(df.target.unique())
        mapping = {str(x): i for i, x in enumerate(left + right)}
        partition = len(left)
    else:
        labels = sorted(set(df.source) | set(df.target))
        mapping = {str(x): i for i, x in enumerate(labels)}
        partition = None
    t = df.t.to_numpy(np.int64)
    u = df.source.astype(str).map(mapping).to_numpy(np.int32)
    v = df.target.astype(str).map(mapping).to_numpy(np.int32)
    if not bip:
        u, v = np.minimum(u, v), np.maximum(u, v)
    assert np.all(u != v)
    assert len(set(zip(t.tolist(), u.tolist(), v.tolist()))) == len(t)
    boundaries = [float(t[0] + f * (t[-1] - t[0])) for f in (0.6, 0.8)]
    split = np.searchsorted(boundaries, t, side="right").astype(np.int8)
    metadata = {
        "dataset": name,
        "source_path": p.relative_to(ROOT).as_posix(),
        "source_url": f"https://github.com/StephanyRajeh/TGF/blob/af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b/{SOURCES[name]}",
        "sha256": sha256(p),
        "n_observed": len(t),
        "n_nodes": len(mapping),
        "n_unique_pairs": len(set(zip(u.tolist(), v.tolist()))),
        "n_timestamps": len(np.unique(t)),
        "bipartite": bip,
        "partition": partition,
        "left_partition": "movies" if bip else None,
        "right_partition": "users" if bip else None,
        "first_timestamp": int(t[0]),
        "last_timestamp": int(t[-1]),
        "split_boundaries": boundaries,
        "split_observed_counts_before_warmup": np.bincount(split, minlength=3).tolist(),
    }
    return t, u, v, split, metadata


def write_data_manifest():
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    manifests = [load_dataset(name)[-1] for name in SOURCES]
    (out / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    return manifests


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
            if counts["normal"] != int((values["y"][mask] == 0).sum()) or counts["anomaly"] != int(
                (values["y"][mask] == 1).sum()
            ):
                raise ValueError(f"Split {split} class counts differ from manifest")
        lo, hi = int(values["t"][mask].min()), int(values["t"][mask].max())
        if previous_max is not None and lo <= previous_max:
            raise ValueError("Chronological splits overlap in timestamp")
        previous_max = hi
    fingerprint = _digest(
        {"npz_sha256": _file_hash(path), "metadata_sha256": _file_hash(metadata_path)}
    )
    return Cache(path=path, names=names, fingerprint=fingerprint, metadata=metadata, **values)


def selected_names(cache: Cache, dataset: str, feature_set: str) -> list[str]:
    from .features import feature_groups

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
