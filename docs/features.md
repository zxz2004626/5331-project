# Causal feature implementation

`src/graphbench/history.py` computes features from observed history. Each
timestamp is a batch: call `advance(t)`, extract every real and injected event's
features, then call `add(t,u,v)` for all events in that batch. Thus neither an
event's own edge nor other events sharing its timestamp appear in its features.
The caller must implement this batching; `add` intentionally allows equal
timestamps and cannot infer whether the caller has finished scoring a batch.

The structural history is the last H mixed-stream events, with H=1000 by default.
Repeated events count as one support edge with multiplicity. Degree is support
degree; weighted degree/strength is incident event count. An event inserted
after scoring can evict the oldest event. Graph orientation is undirected for
UCI and bipartite for MovieLens. Self-loops and within-partition bipartite pairs
are rejected. Node IDs are dense global integers, with all left-partition IDs
below `partition`. Merely assigning an ID never makes a node active.

## Original TGF arithmetic

The module preserves all 33 unipartite / 44 bipartite feature names from
`vendor/TGF/3-FeatureGeneration/main.py`. It has 11 graph-wide node distribution
statistics (per partition for bipartite graphs), 12 endpoint statistics, and 10
edge distribution/endpoint-pair statistics. `feature_groups(bipartite)['tgf']`
returns their names in a stable order. Metadata, timestamps, fraud labels,
history parameters and computation cost are not predictor features.

The reference implementation's `DSC_lib.py` defines several details that must
not be silently replaced by common alternatives:

- Populations contain only positive-count nodes/edges currently in H history.
- `*_same_*` counts population members with equal value.
- `*_greater_*` counts population members with strictly greater value; this is
  not the rank among distinct values.
- Median is `sorted(values, reverse=True)[len(values)//2]`, the lower middle
  value for even-length populations, rather than the arithmetic mean.
- Unipartite endpoints are ordered by increasing support degree. Their weighted
  degree statistics follow that same orientation. Ties use increasing integer
  ID, making extraction invariant to input endpoint reversal. Bipartite endpoints
  retain left/right orientation. A numeric ID order may differ from the official
  CLI's lexical tie order if that CLI is run without `-int`.

The original CLI inserts the current event before feature extraction. The causal
benchmark changes the observation time to before timestamp-batch insertion.
Therefore an unseen endpoint or pair can occur. We define its value and equal
count as zero; its greater count is the number of active reference-population
members. Missing objects are not added as artificial zero-valued population
members. Empty-population minimum, maximum and median are zero. These extensions
are explicit modelling choices; they are not claims about unspecified official
cold-start behaviour.

Positive counter histograms and Fenwick trees support updates and ranks in
O(log H). Global statistics and queried ranks are cached until their population
changes. Extraction performs no full node/edge sort. Local path searches still
depend on the endpoint neighbourhood sizes.

## Local feature group

For undirected UCI graphs the features are `common_neighbors`, `jaccard` and
`adamic_adar`: respectively |N(u) intersection N(v)|, the intersection divided
by the union, and the sum of 1/log(degree(w)) over common neighbours. An empty
union has Jaccard zero. Self-loops are absent, so a common neighbour's degree
is at least two.

For MovieLens, a user and an item cannot have ordinary common neighbours.
`path3_count` instead counts **simple length-three paths** u-a-b-v, excluding
u=v-like endpoint revisits/backtracking. `path3_normalized` divides this count
by degree(u)*degree(v), or returns zero if either endpoint has degree zero.
Thus the local group has two features for bipartite graphs. Edge multiplicities
are not used in these local support-graph statistics.

## Temporal feature group

The recent interval is [t-R,t), R=86400 seconds by default, where its open right
boundary follows from the timestamp-batch scoring protocol. Internally an event
expires only when its timestamp is strictly below t-R. This recent queue is
independent of the last-H structural queue, and includes all observed mixed
events in the interval even if they have already left H.

- `u_recent_count`, `v_recent_count`: incident event counts in the recent interval.
- `u_recency_log`, `v_recency_log`: log1p(min(time since lifetime last activity,
  30R)). Never-observed nodes receive log1p(30R). Recency is in seconds before
  transformation.
- `u_active`, `v_active`: whether the corresponding recent count is positive.
- `pair_seen_before`: whether this pair has occurred anywhere in observed
  lifetime history, including events older than either window.

Temporal endpoint orientation follows TGF orientation, including its degree
ordering for undirected graphs. No future timestamp, future interaction or
clean/fraud label is used to construct these quantities.

## Public interface and verification

`GraphHistory(n_nodes, bipartite, partition=None, window=1000,
recent_seconds=86400)` exposes `degree`, `strength`, `neighbors`, `edge_counts`,
`last_active`, `recent_counts`, and `seen_pairs` for read-only generator use.
`canonical(u,v)` validates and orients an edge; `advance(t)` (alias
`expire_recent`) expires recent activity; `add` inserts an event. `extract`
returns all groups; `profile` returns degree/strength endpoints plus the temporal
group without neighbourhood searches; `local_features` and `temporal_features`
allow separate matching stages. Extraction advances the recent clock but never
inserts the scored event. Calls must have nondecreasing finite times.

`tests/test_history.py` covers cold start, timestamp ties, separate window
expiration, repeated-edge eviction, ranks with tied values, official lower
medians, endpoint symmetry, bipartite simple paths and random sliding-history
invariants. It also runs the original vendor CLI as an independent subprocess
on randomized unipartite and bipartite streams and compares every TGF feature
after insertion. That paired check isolates arithmetic equivalence; the causal
batch-order check separately verifies the intended observation-time change.
