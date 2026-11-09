"""Causal sliding-history graph features; see ``docs/features.md``.

Call ``advance(t)``, score every event at timestamp t, then ``add`` that
timestamp's observed (including injected) events.  Extraction never inserts an
edge.  Dense integer IDs refer to a fixed vocabulary, not an active population.
"""
from __future__ import annotations

from collections import deque
import math
import numpy as np


_NODE_GLOBALS = (
    "number_of_nodes", "degrees_nb_one", "degrees_nb_two", "degrees_min",
    "degrees_max", "degrees_median", "weighted_degrees_nb_one",
    "weighted_degrees_nb_two", "weighted_degrees_min", "weighted_degrees_max",
    "weighted_degrees_median",
)
_ENDPOINTS = (
    "u_degree", "v_degree", "u_same_degree", "v_same_degree",
    "u_greater_degree", "v_greater_degree", "u_weighted_degree",
    "v_weighted_degree", "u_same_weighted_degree", "v_same_weighted_degree",
    "u_greater_weighted_degree", "v_greater_weighted_degree",
)
_EDGE_GLOBALS = (
    "number_of_links", "total_weight", "link_weights_nb_one",
    "link_weights_nb_two", "link_weights_min", "link_weights_max",
    "link_weights_median", "u_v_weight", "u_v_same_weight", "u_v_greater_weight",
)
_TEMPORAL = (
    "u_recent_count", "v_recent_count", "u_recency_log", "v_recency_log",
    "u_active", "v_active", "pair_seen_before",
)


def feature_groups(bipartite: bool) -> dict[str, list[str]]:
    """Stable, disjoint groups, in output-column order (33/44 TGF columns)."""
    globals_ = ([prefix + name for prefix in ("top_", "bot_")
                 for name in _NODE_GLOBALS] if bipartite else list(_NODE_GLOBALS))
    return {
        "tgf": globals_ + list(_ENDPOINTS) + list(_EDGE_GLOBALS),
        "local": (["path3_count", "path3_normalized"] if bipartite else
                  ["common_neighbors", "jaccard", "adamic_adar"]),
        "temporal": list(_TEMPORAL),
    }


class _Distribution:
    """Positive-integer histogram and Fenwick order statistics, without sorting."""

    def __init__(self, limit: int):
        self.limit = limit
        self.counts = [0] * (limit + 1)
        self.tree = [0] * (limit + 1)
        self.n = 0
        self.total = 0
        self._summary: dict[str, int] | None = None
        self._greater: dict[int, int] = {}

    def change(self, old: int, new: int) -> None:
        if old == new:
            return
        for value, delta in ((old, -1), (new, 1)):
            if value == 0:
                continue
            if not 1 <= value <= self.limit:
                raise ValueError("Counter exceeds history window")
            self.counts[value] += delta
            self.n += delta
            self.total += delta * value
            i = value
            while i <= self.limit:
                self.tree[i] += delta
                i += i & -i
        self._summary = None
        self._greater.clear()

    def same(self, value: int) -> int:
        # Absent objects are not inserted into the reference population.
        return self.counts[value] if value else 0

    def greater(self, value: int) -> int:
        cached = self._greater.get(value)
        if cached is not None:
            return cached
        leq, i = 0, value
        while i:
            leq += self.tree[i]
            i -= i & -i
        result = self.n - leq
        self._greater[value] = result
        return result

    def _select(self, kth: int) -> int:
        """Return one-based kth ascending value; caller ensures nonempty."""
        index = 0
        step = 1 << (self.limit.bit_length() - 1)
        while step:
            nxt = index + step
            if nxt <= self.limit and self.tree[nxt] < kth:
                index = nxt
                kth -= self.tree[nxt]
            step >>= 1
        return index + 1

    def summary(self) -> dict[str, int]:
        if self._summary is None:
            self._summary = {
                "nb_one": self.same(1),
                "nb_two": self.same(2) if self.limit >= 2 else 0,
                "min": self._select(1) if self.n else 0,
                "max": self._select(self.n) if self.n else 0,
                # Official DSC: descending values[int(n/2)], LOWER median.
                "median": self._select(self.n - self.n // 2) if self.n else 0,
            }
        return self._summary


class GraphHistory:
    """Undirected or bipartite multigraph over the last ``window`` events.

    ``neighbors`` and ``degree`` describe the simple support graph; ``strength``
    and ``edge_counts`` retain event multiplicities. Recent activity uses its
    own time window, while ``last_active`` and ``seen_pairs`` cover all history.
    Public arrays/sets are intended for read access by a candidate generator.
    """

    def __init__(self, n_nodes: int, bipartite: bool, partition: int | None = None,
                 window: int = 1000, recent_seconds: float = 86400):
        if n_nodes < 1 or window < 1 or int(window) != window:
            raise ValueError("n_nodes and window must be positive integers")
        if not math.isfinite(recent_seconds) or recent_seconds <= 0:
            raise ValueError("recent_seconds must be finite and positive")
        if bipartite and (partition is None or not 0 < partition < n_nodes):
            raise ValueError("Bipartite graphs require 0 < partition < n_nodes")
        self.n_nodes = n_nodes
        self.bipartite = bool(bipartite)
        self.partition = partition
        self.window = int(window)
        self.recent_seconds = float(recent_seconds)
        self.recency_cap = 30.0 * self.recent_seconds
        self.degree = np.zeros(n_nodes, dtype=np.int64)
        self.strength = np.zeros(n_nodes, dtype=np.int64)
        self.neighbors: list[set[int]] = [set() for _ in range(n_nodes)]
        self.edge_counts: dict[tuple[int, int], int] = {}
        self.last_active = np.full(n_nodes, -np.inf)
        self.recent_counts = np.zeros(n_nodes, dtype=np.int64)
        self.seen_pairs: set[tuple[int, int]] = set()
        self.events: deque[tuple[float, int, int]] = deque()
        self._recent: deque[tuple[float, int, int]] = deque()
        self._time = -math.inf
        groups = 2 if bipartite else 1
        self._degrees = [_Distribution(self.window) for _ in range(groups)]
        self._strengths = [_Distribution(self.window) for _ in range(groups)]
        self._weights = _Distribution(self.window)
        self._global_cache: dict[str, float] | None = None

    def canonical(self, u: int, v: int) -> tuple[int, int]:
        u, v = int(u), int(v)
        if not (0 <= u < self.n_nodes and 0 <= v < self.n_nodes):
            raise ValueError("Node ID is outside the fixed vocabulary")
        if u == v:
            raise ValueError("Self-loops are excluded")
        if u > v:
            u, v = v, u
        if self.bipartite and not (u < self.partition <= v):
            raise ValueError("Bipartite pairs must cross the partition")
        return u, v

    def _ordered(self, u: int, v: int) -> tuple[int, int]:
        u, v = self.canonical(u, v)
        if not self.bipartite and self.degree[u] > self.degree[v]:
            return v, u
        return u, v

    def advance(self, t: float) -> None:
        """Expire activity strictly older than t-recent_seconds; keep boundary."""
        t = float(t)
        if not math.isfinite(t) or t < self._time:
            raise ValueError("Timestamps must be finite and nondecreasing")
        cutoff = t - self.recent_seconds
        while self._recent and self._recent[0][0] < cutoff:
            _, u, v = self._recent.popleft()
            self.recent_counts[u] -= 1
            self.recent_counts[v] -= 1
        self._time = t

    expire_recent = advance

    def _node_change(self, node: int, degree_delta: int, strength_delta: int) -> None:
        group = int(self.bipartite and node >= self.partition)
        d, s = int(self.degree[node]), int(self.strength[node])
        self._degrees[group].change(d, d + degree_delta)
        self._strengths[group].change(s, s + strength_delta)
        self.degree[node] += degree_delta
        self.strength[node] += strength_delta

    def _remove_oldest(self) -> None:
        _, u, v = self.events.popleft()
        pair = (u, v)
        weight = self.edge_counts[pair]
        self._weights.change(weight, weight - 1)
        last = weight == 1
        if last:
            del self.edge_counts[pair]
            self.neighbors[u].remove(v)
            self.neighbors[v].remove(u)
        else:
            self.edge_counts[pair] = weight - 1
        self._node_change(u, -int(last), -1)
        self._node_change(v, -int(last), -1)

    def add(self, t: float, u: int, v: int) -> None:
        """Insert one observed event, evicting the oldest event beyond H."""
        u, v = self.canonical(u, v)
        self.advance(t)
        if len(self.events) == self.window:
            self._remove_oldest()
        pair = u, v
        old = self.edge_counts.get(pair, 0)
        self._weights.change(old, old + 1)
        self.edge_counts[pair] = old + 1
        if not old:
            self.neighbors[u].add(v)
            self.neighbors[v].add(u)
        self._node_change(u, int(not old), 1)
        self._node_change(v, int(not old), 1)
        self.events.append((float(t), u, v))
        self._recent.append((float(t), u, v))
        self.recent_counts[u] += 1
        self.recent_counts[v] += 1
        self.last_active[u] = self.last_active[v] = float(t)
        self.seen_pairs.add(pair)
        self._global_cache = None

    def _globals(self) -> dict[str, float]:
        if self._global_cache is None:
            result: dict[str, float] = {}
            prefixes = ("top_", "bot_") if self.bipartite else ("",)
            for prefix, degrees, strengths in zip(prefixes, self._degrees, self._strengths):
                result[prefix + "number_of_nodes"] = degrees.n
                for family, dist in (("degrees_", degrees), ("weighted_degrees_", strengths)):
                    result.update({prefix + family + key: value
                                   for key, value in dist.summary().items()})
            result["number_of_links"] = self._weights.n
            result["total_weight"] = self._weights.total
            result.update({"link_weights_" + k: v for k, v in self._weights.summary().items()})
            self._global_cache = result
        return self._global_cache

    def local_features(self, u: int, v: int) -> dict[str, float]:
        u, v = self.canonical(u, v)
        nu, nv = self.neighbors[u], self.neighbors[v]
        if self.bipartite:
            # Simple u-a-b-v paths: forbid backtracking through u or v.
            if len(nu) > len(nv):
                u, v, nu, nv = v, u, nv, nu
            count = sum(len(self.neighbors[a] & nv) - int(u in nv)
                        for a in nu if a != v)
            denominator = len(nu) * len(nv)
            return {"path3_count": count,
                    "path3_normalized": count / denominator if denominator else 0.0}
        common = nu & nv
        union = len(nu) + len(nv) - len(common)
        return {"common_neighbors": len(common),
                "jaccard": len(common) / union if union else 0.0,
                "adamic_adar": sum(1.0 / math.log(int(self.degree[w])) for w in common)}

    def temporal_features(self, t: float, u: int, v: int) -> dict[str, float]:
        self.advance(t)
        u, v = self._ordered(u, v)
        ru = min(self.recency_cap, max(0.0, float(t) - self.last_active[u]))
        rv = min(self.recency_cap, max(0.0, float(t) - self.last_active[v]))
        return {"u_recent_count": int(self.recent_counts[u]),
                "v_recent_count": int(self.recent_counts[v]),
                "u_recency_log": math.log1p(ru), "v_recency_log": math.log1p(rv),
                "u_active": int(self.recent_counts[u] > 0),
                "v_active": int(self.recent_counts[v] > 0),
                "pair_seen_before": int(self.canonical(u, v) in self.seen_pairs)}

    def profile(self, t: float, u: int, v: int) -> dict[str, float]:
        """Cheap endpoint profile for proposal matching; excludes path searches."""
        u, v = self._ordered(u, v)
        result = {"u_degree": int(self.degree[u]), "v_degree": int(self.degree[v]),
                  "u_weighted_degree": int(self.strength[u]),
                  "v_weighted_degree": int(self.strength[v])}
        result.update(self.temporal_features(t, u, v))
        return result

    def extract(self, t: float, u: int, v: int) -> dict[str, float]:
        """Return all feature groups without inserting the scored event."""
        self.advance(t)
        pair = self.canonical(u, v)
        u, v = self._ordered(u, v)
        result = self._globals().copy()
        for name, node in (("u", u), ("v", v)):
            group = int(self.bipartite and node >= self.partition)
            for suffix, values, dist in (("degree", self.degree, self._degrees[group]),
                                         ("weighted_degree", self.strength, self._strengths[group])):
                value = int(values[node])
                result[name + "_" + suffix] = value
                result[name + "_same_" + suffix] = dist.same(value)
                result[name + "_greater_" + suffix] = dist.greater(value)
        weight = self.edge_counts.get(pair, 0)
        result.update(u_v_weight=weight, u_v_same_weight=self._weights.same(weight),
                      u_v_greater_weight=self._weights.greater(weight))
        result.update(self.local_features(u, v))
        result.update(self.temporal_features(t, u, v))
        return result
