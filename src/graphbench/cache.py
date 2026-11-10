"""Validate feature caches and align model columns by name."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import _digest, _file_hash
from .constants import ABLATIONS, CORE_FEATURES


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
