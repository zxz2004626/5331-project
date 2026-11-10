"""Load the author's complete cleaned examples; preserve timestamp ties."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import ROOT
from .provenance import GENERATION_SOURCES, source_digest

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
