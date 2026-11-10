"""Independent generator-stage and end-to-end causal-cache invariants."""

import copy
import gzip
import json

import numpy as np
import pytest

from graphbench import data, generators
from graphbench.features import GraphHistory


def config():
    return {
        "history": {"window": 7, "recent_seconds": 10, "warmup_observed_events": 3},
        "injection": {
            "anomaly_fraction": 0.2,
            "absence": "global",
            "legal_candidate_pool": 6,
            "endpoint_keep": 3,
            "local_keep": 2,
            "max_draws": 10000,
        },
        "scenario": "test",
    }


class ScriptedRng:
    """A controlled proposal stream, with deterministic non-minimum choice."""

    def __init__(self, pairs):
        self.pairs = iter(pairs)

    def integers(self, low, high, size=None):
        assert size == 2
        return np.asarray(next(self.pairs), dtype=int)

    def choice(self, values):
        return values[-1]

    def random(self, size):
        return np.linspace(0.01, 0.99, size)


def test_schedule_exact_per_split_prior_only_valid_anchors_and_reproducible():
    t = np.repeat(np.arange(100, 160), 2)
    split = np.searchsorted([136, 148], t, side="right")
    valid = t > 101
    refs = generators.injection_schedule(t, split, valid, 0.2, 11)
    assert refs == generators.injection_schedule(t, split, valid, 0.2, 11)
    assert all(valid[i] for i in refs)
    assert refs == sorted(refs, key=lambda i: (int(t[i]), i))
    for s in range(3):
        normal_n = np.count_nonzero((split == s) & valid)
        expected = round(normal_n * 0.2 / 0.8)
        assert sum(split[i] == s for i in refs) == expected


@pytest.mark.parametrize("generator,expected", [("G0", 0), ("G1", 3), ("G2", 2), ("G3", 3)])
def test_stages_are_nested_ranked_pools_not_global_optimization(monkeypatch, generator, expected):
    pairs = [(i, i + 10) for i in range(6)]
    endpoint = dict(zip(pairs, [9, 1, 2, 3, 8, 7]))
    local = dict(zip(pairs, [0, 9, 2, 1, 0, 0]))
    temporal = dict(zip(pairs, [0, 0, 5, 1, 0, 0]))
    reference = (8, 18)
    # Controlled scalar distances make every stage's required selection clear.
    monkeypatch.setattr(generators, "_profile_vector", lambda *args: np.zeros(4))
    monkeypatch.setattr(
        generators, "_endpoint_vectors", lambda hist, p: np.array([[endpoint[e]] * 4 for e in p])
    )
    monkeypatch.setattr(
        generators, "_local_vector", lambda hist, u, v: np.array([local.get((u, v), 0)])
    )
    monkeypatch.setattr(
        generators, "_temporal_vector", lambda hist, t, u, v: np.array([temporal.get((u, v), 0)])
    )
    h = GraphHistory(20, False)
    u, v, audit = generators.choose_edge(
        h, 10, *reference, generator, ScriptedRng(pairs), set(), set(), config(), 20, None
    )
    assert (u, v) == pairs[expected]
    assert audit["status"] == "valid"
    assert audit["candidate_pool_size"] == (1 if generator == "G0" else 6)
    assert audit["endpoint_pool_size"] == (1 if generator == "G0" else 3)
    assert audit["local_pool_size"] == (1 if generator == "G0" else 3 if generator == "G1" else 2)
    assert audit["endpoint_distance"] == endpoint[pairs[expected]]
    if generator != "G0":
        assert audit["endpoint_best_distance"] == 1
        assert audit["endpoint_top16_max_distance"] == 3
    # G3 cannot select zero local/temporal distance candidates excluded earlier;
    # it minimizes temporal distance only within the retained local candidates.
    if generator == "G3":
        assert audit["local_distance"] == audit["temporal_distance"] == 1


def test_rejection_accounting_and_no_silent_caliper_guarantee():
    cfg = config()
    cfg["injection"]["legal_candidate_pool"] = 3
    h = GraphHistory(6, False)
    proposed = [(0, 0), (0, 1), (1, 2), (3, 4), (4, 3), (0, 2), (2, 5)]
    u, v, audit = generators.choose_edge(
        h, 10, 0, 1, "G1", ScriptedRng(proposed), {(0, 1)}, {(1, 2)}, cfg, 6, None
    )
    assert (u, v) == (2, 5)
    assert audit["draws"] == 7
    assert audit["rejections"] == {
        "self_loop": 1,
        "observed_pair": 1,
        "injected_duplicate": 1,
        "candidate_duplicate": 1,
    }
    assert audit["candidate_pool_size"] + sum(audit["rejections"].values()) == audit["draws"]


def test_past_absence_allows_future_observed_pair_but_excludes_seen_and_current():
    h = GraphHistory(6, False)
    h.add(0, 0, 1)
    cfg = config()
    cfg["injection"]["absence"] = "past"
    # The caller puts current-timestamp normal pairs in the duplicate set.
    u, v, audit = generators.choose_edge(
        h,
        10,
        0,
        1,
        "G0",
        ScriptedRng([(0, 1), (1, 2), (2, 3)]),
        {(0, 1), (2, 3)},
        {(1, 2)},
        cfg,
        6,
        None,
    )
    assert (u, v) == (2, 3)
    assert audit["rejections"] == {"observed_pair": 1, "injected_duplicate": 1}


def test_support_exhaustion_fails_instead_of_reducing_count():
    cfg = config()
    cfg["injection"].update(legal_candidate_pool=2, max_draws=4)
    with pytest.raises(RuntimeError, match="support-limited G1: 1/2 after 4 draws"):
        generators.choose_edge(
            GraphHistory(4, False),
            10,
            0,
            1,
            "G1",
            ScriptedRng([(0, 2)] * 4),
            set(),
            set(),
            cfg,
            4,
            None,
        )


def synthetic_dataset(bipartite):
    t = np.repeat(np.arange(100, 160), 2).astype(np.int64)
    u = (np.arange(len(t)) % 10).astype(np.int32)
    v = (10 + np.arange(len(t)) * 7 % 20).astype(np.int32)
    split = np.searchsorted([136, 148], t, side="right").astype(np.int8)
    meta = {
        "n_nodes": 30,
        "bipartite": bipartite,
        "partition": 10 if bipartite else None,
        "sha256": "synthetic-fixture",
    }
    return t, u, v, split, meta


@pytest.mark.parametrize("bipartite", [False, True])
def test_full_cache_strict_ties_counts_legality_and_reproducibility(
    monkeypatch, tmp_path, bipartite
):
    dataset = synthetic_dataset(bipartite)
    cfg = config()
    monkeypatch.setattr(generators, "ROOT", tmp_path)
    monkeypatch.setattr(generators, "scenario_config", lambda scenario: copy.deepcopy(cfg))
    monkeypatch.setattr(generators, "load_dataset", lambda name: dataset)
    monkeypatch.setattr(generators, "code_hash", lambda: "test-code-version")
    manifest = generators.build_cache("synthetic", 11, "G3", "test")
    folder = tmp_path / "results/cache/test/synthetic/seed_11"
    with np.load(folder / "G3.npz") as result:
        rows = {key: result[key].copy() for key in result.files}
    with gzip.open(folder / "G3_audit.jsonl.gz", "rt", encoding="utf-8") as stream:
        audit = [json.loads(line) for line in stream]
    assert manifest["status"] == "complete"
    assert all(manifest["checks"].values())
    assert manifest["warmup_last_timestamp"] == 101  # t[2], then whole tie.
    assert manifest["observed_scored"] == 116
    assert np.all(rows["history_max_timestamp"] < rows["t"])
    assert np.isfinite(rows["X"]).all()
    assert len(audit) == manifest["anomalies"]
    t, observed_u, observed_v, split, meta = dataset
    observed = set(zip(observed_u.tolist(), observed_v.tolist()))
    anomalies = [
        (int(u), int(v)) for u, v in zip(rows["u"][rows["y"] == 1], rows["v"][rows["y"] == 1])
    ]
    assert len(set(anomalies)) == len(anomalies)
    assert not set(anomalies) & observed
    assert all(u < v for u, v in anomalies)
    if bipartite:
        assert all(u < 10 <= v for u, v in anomalies)
    for s in range(3):
        normal = int(np.count_nonzero((split == s) & (t > 101)))
        anomaly = round(normal * 0.2 / 0.8)
        assert manifest["split_counts"][s] == {"normal": normal, "anomaly": anomaly}
        assert np.count_nonzero((rows["split"] == s) & (rows["y"] == 1)) == anomaly
    assert all(
        d["draws"] == d["candidate_pool_size"] + sum(d["rejections"].values()) for d in audit
    )
    assert all(
        d["candidate_pool_size"] == 6 and d["endpoint_pool_size"] == 3 and d["local_pool_size"] == 2
        for d in audit
    )
    # Reconstruct every timestamp from a separate history. Score the entire
    # batch before insertion, including normals and injections at the same t.
    h = GraphHistory(30, bipartite, meta["partition"], 7, 10)
    for now in np.unique(t):
        h.advance(int(now))
        selected = np.flatnonzero(rows["t"] == now)
        for idx in selected:
            feature = h.extract(int(now), int(rows["u"][idx]), int(rows["v"][idx]))
            expected = np.asarray([feature[name] for name in rows["names"]], dtype=np.float32)
            np.testing.assert_array_equal(rows["X"][idx], expected)
        # The build's stable order is real events, then injected events.
        for idx in np.flatnonzero(t == now):
            h.add(int(now), int(observed_u[idx]), int(observed_v[idx]))
        for idx in selected[rows["y"][selected] == 1]:
            h.add(int(now), int(rows["u"][idx]), int(rows["v"][idx]))
    assert generators.build_cache("synthetic", 11, "G3", "test") == manifest
    generators.build_cache("synthetic", 11, "G3", "test", force=True)
    with np.load(folder / "G3.npz") as repeated:
        for key, value in rows.items():
            np.testing.assert_array_equal(repeated[key], value)
    with gzip.open(folder / "G3_audit.jsonl.gz", "rt", encoding="utf-8") as stream:
        assert [json.loads(line) for line in stream] == audit
    monkeypatch.setattr(generators, "code_hash", lambda: "changed-version")
    with pytest.raises(RuntimeError, match="Stale cache"):
        generators.build_cache("synthetic", 11, "G3", "test")


@pytest.mark.parametrize("bipartite", [False, True])
def test_data_loader_stable_ties_elapsed_splits_and_namespaces(monkeypatch, tmp_path, bipartite):
    name = "movielens" if bipartite else "ucimessages"
    source = tmp_path / "vendor/TGF/fixture.txt"
    source.parent.mkdir(parents=True)
    # Input deliberately unsorted, with ties exactly at 60%/80% boundaries.
    raw = [(80, 3, 4), (0, 1, 2), (60, 2, 3), (100, 4, 1), (60, 3, 1), (80, 2, 4), (20, 4, 2)]

    def label(a, left):
        return ("bottom_" if left else "top_") + str(a) if bipartite else str(a)

    source.write_text("".join(f"{t} {label(u, True)} {label(v, False)}\n" for t, u, v in raw))
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setitem(data.SOURCES, name, "fixture.txt")
    t, u, v, split, meta = data.load_dataset(name)
    assert t.tolist() == [0, 20, 60, 60, 80, 80, 100]
    assert split.tolist() == [0, 0, 1, 1, 2, 2, 2]
    assert meta["split_boundaries"] == [60.0, 80.0]
    assert meta["split_observed_counts_before_warmup"] == [2, 2, 3]
    assert meta["n_timestamps"] == 5
    assert meta["n_nodes"] == (8 if bipartite else 4)
    if bipartite:
        assert meta["partition"] == 4
        assert np.all(u < 4) and np.all(v >= 4)
        assert u[2:4].tolist() == [1, 2]  # Stable order within t=60.
    else:
        assert np.all(u < v)


def test_data_loader_rejects_same_timestamp_undirected_duplicate(monkeypatch, tmp_path):
    source = tmp_path / "vendor/TGF/fixture.txt"
    source.parent.mkdir(parents=True)
    source.write_text("0 1 2\n0 2 1\n100 2 3\n")
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setitem(data.SOURCES, "ucimessages", "fixture.txt")
    with pytest.raises(AssertionError):
        data.load_dataset("ucimessages")


def test_diagnostics_separate_all_groups_and_flag_constant_separation():
    names = ["degree", "path3", "recent", "pair_seen_before"]
    groups = {"tgf": ["degree"], "local": ["path3"], "temporal": ["recent", "pair_seen_before"]}
    X = np.asarray([[0, 1, 2, 1], [0, 1, 2, 1], [0, 3, 2, 0], [0, 3, 2, 0]])
    rows = generators.diagnostics(X, np.array([0, 0, 1, 1]), np.zeros(4), names, groups)
    by_name = {row["feature"]: row for row in rows}
    assert by_name["degree"]["smd"] == by_name["degree"]["ks"] == 0
    assert by_name["path3"]["smd"] is None
    assert by_name["pair_seen_before"]["ks"] == 1
    assert by_name["pair_seen_before"]["wasserstein"] == 1
