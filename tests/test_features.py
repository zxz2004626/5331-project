"""Arithmetic checks, official implementation parity, and causal invariants."""

import json
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from graph_anomaly.features import GraphHistory, feature_groups


def test_cold_start_finite_and_immutable():
    history = GraphHistory(6, False, window=3, recent_seconds=10)
    before = history.extract(100, 0, 5)
    assert before["number_of_nodes"] == before["number_of_links"] == 0
    assert before["u_v_weight"] == before["u_degree"] == 0
    assert before["u_recency_log"] == pytest.approx(math.log1p(300))
    assert all(math.isfinite(x) for x in before.values())
    assert history.extract(100, 5, 0) == before
    assert not history.events and not history.seen_pairs
    assert set(before) == set(sum(feature_groups(False).values(), []))
    assert len(feature_groups(False)["tgf"]) == 33
    assert len(feature_groups(True)["tgf"]) == 44


def test_score_tied_timestamp_before_any_insertion():
    h = GraphHistory(5, False, window=4)
    h.add(1, 0, 1)
    events = [(10, 0, 2), (10, 1, 2), (10, 0, 2)]
    rows = [h.extract(*event) for event in events]
    assert rows[0] == rows[2]
    assert all(row["number_of_links"] == 1 for row in rows)
    assert all(row["pair_seen_before"] == 0 for row in rows)
    for event in events:
        h.add(*event)
    after = h.extract(11, 0, 2)
    assert after["u_v_weight"] == 2 and after["pair_seen_before"] == 1
    assert after["common_neighbors"] == 1


def test_event_window_and_recent_window_are_independent():
    h = GraphHistory(5, False, window=2, recent_seconds=10)
    h.add(0, 0, 1)
    h.add(5, 0, 2)
    h.add(10, 2, 3)
    assert h.degree.tolist() == [1, 0, 2, 1, 0]
    assert h.strength.tolist() == [1, 0, 2, 1, 0]
    assert h.recent_counts.tolist() == [2, 1, 2, 1, 0]
    # Boundary event at t-recent_seconds remains; H eviction does not expire it.
    h.advance(10)
    assert h.recent_counts[1] == 1
    h.advance(11)
    assert h.recent_counts.tolist() == [1, 0, 2, 1, 0]
    row = h.extract(16, 0, 1)
    assert row["pair_seen_before"] == 1
    assert row["u_v_weight"] == 0
    assert h.last_active[1] == 0
    assert h.recent_counts.tolist() == [0, 0, 1, 1, 0]
    assert len(h.events) == 2


def test_multiplicity_and_eviction():
    h = GraphHistory(4, False, window=3)
    for event in [(0, 0, 1), (1, 1, 0), (2, 0, 2)]:
        h.add(*event)
    assert h.degree.tolist() == [2, 1, 1, 0]
    assert h.strength.tolist() == [3, 2, 1, 0]
    assert h.edge_counts[(0, 1)] == 2
    h.add(3, 2, 3)
    assert h.edge_counts[(0, 1)] == 1 and h.degree[0] == 2
    h.add(4, 2, 3)
    assert (0, 1) not in h.edge_counts and h.degree[1] == 0
    row = h.extract(5, 0, 1)
    assert row["u_degree"] == row["u_same_degree"] == 0
    assert row["u_greater_degree"] == 3
    assert row["u_v_same_weight"] == 0
    assert row["u_v_greater_weight"] == 2


def test_official_lower_median_and_population_ranks():
    h = GraphHistory(7, False, window=10)
    for t, edge in enumerate([(0, 1), (0, 2), (0, 3), (4, 5)]):
        h.add(t, *edge)
    row = h.extract(4, 0, 1)
    assert row["degrees_median"] == 1
    assert row["u_same_degree"] == 5
    assert row["u_greater_degree"] == 1
    assert row["v_greater_degree"] == 0
    # Weight population [2,2,1,1]: lower median is 1, not mean 1.5.
    h.add(5, 0, 1)
    h.add(6, 0, 2)
    row = h.extract(7, 0, 3)
    assert row["link_weights_median"] == 1
    assert row["u_v_greater_weight"] == 2  # Objects, not distinct values.


def test_unipartite_local_scores_and_symmetric_endpoint_order():
    h = GraphHistory(6, False, window=20)
    for t, edge in enumerate([(0, 2), (1, 2), (0, 3), (1, 3), (0, 4)]):
        h.add(t, *edge)
    result = h.extract(10, 0, 1)
    assert result == h.extract(10, 1, 0)
    assert result["u_degree"] == 2 and result["v_degree"] == 3
    assert result["common_neighbors"] == 2
    assert result["jaccard"] == pytest.approx(2 / 3)
    assert result["adamic_adar"] == pytest.approx(2 / math.log(2))


def test_bipartite_simple_paths_exclude_backtracking():
    h = GraphHistory(6, True, partition=3, window=20)
    for t, edge in enumerate([(0, 4), (1, 4), (1, 3), (0, 5), (2, 5), (2, 3)]):
        h.add(t, *edge)
    row = h.extract(10, 0, 3)
    assert row["path3_count"] == 2
    assert row["path3_normalized"] == 0.5
    assert row["u_degree"] == row["v_degree"] == 2
    assert row == h.extract(10, 3, 0)
    assert "common_neighbors" not in row
    h.add(10, 0, 3)
    row = h.extract(11, 0, 3)
    assert row["path3_count"] == 2
    assert row["path3_normalized"] == pytest.approx(2 / 9)
    assert row["top_number_of_nodes"] == row["bot_number_of_nodes"] == 3


@pytest.mark.parametrize("bipartite", [False, True])
def test_random_stream_matches_original_tgf(bipartite):
    """Compare all exact TGF names/values with the independently run vendor CLI.

    We deliberately insert BEFORE extraction only in this parity test: it
    validates arithmetic, not the causal benchmark's timestamp protocol.
    """
    vendor = Path(__file__).resolve().parents[1] / "vendor/TGF/3-FeatureGeneration/main.py"
    if not vendor.exists():
        pytest.skip("Vendored official reference unavailable")
    rng = random.Random(621)
    events = []
    for t in range(120):
        if bipartite:
            u, v = rng.randrange(8), rng.randrange(8, 20)
        else:
            u, v = rng.sample(range(20), 2)
        events.append((t, u, v))
    text = "".join(f"{t} {u} {v} 0\n" for t, u, v in events)
    command = [sys.executable, str(vendor), "-H", "11", "-int"]
    if bipartite:
        command.append("-bip")
    completed = subprocess.run(command, input=text, capture_output=True, text=True, check=True)
    official = [json.loads(line) for line in completed.stdout.splitlines()]
    h = GraphHistory(20, bipartite, partition=8 if bipartite else None, window=11)
    for event, reference in zip(events, official):
        h.add(*event)
        result = h.extract(*event)
        if "number_of_links" in reference:
            for name in feature_groups(bipartite)["tgf"]:
                assert result[name] == reference[name], (event, name, result[name], reference[name])


def test_random_evictions_agree_with_bruteforce_support():
    rng = random.Random(54)
    h = GraphHistory(12, False, window=7)
    stream = []
    for t in range(100):
        edge = tuple(sorted(rng.sample(range(12), 2)))
        h.add(t, *edge)
        stream.append(edge)
        expected = Counter(stream[-7:])
        assert h.edge_counts == expected
        strengths = np.zeros(12, dtype=int)
        degrees = np.zeros(12, dtype=int)
        for (u, v), count in expected.items():
            strengths[u] += count
            strengths[v] += count
            degrees[u] += 1
            degrees[v] += 1
        np.testing.assert_array_equal(h.strength, strengths)
        np.testing.assert_array_equal(h.degree, degrees)
        row = h.extract(t, 0, 1)
        assert row["total_weight"] == min(t + 1, 7)
        assert row["number_of_nodes"] == np.count_nonzero(strengths)


def test_single_event_window_and_validation():
    h = GraphHistory(4, False, window=1)
    h.add(0, 0, 1)
    h.add(1, 2, 3)
    row = h.extract(1, 0, 1)
    assert row["degrees_nb_two"] == row["link_weights_nb_two"] == 0
    assert h.degree.tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        h.add(0, 0, 1)
    with pytest.raises(ValueError):
        h.add(2, 0, 0)
    with pytest.raises(ValueError):
        h.extract(2, -1, 1)
    with pytest.raises(ValueError):
        GraphHistory(6, True, partition=3).add(0, 0, 1)
