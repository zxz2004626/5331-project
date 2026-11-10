# Official TGF reproduction and protocol audit

This is a representative reproduction of the released H1000 classical-learning experiment on **both complete example datasets**, followed by a paired extraction-timing sensitivity experiment. It does not claim to reproduce every dataset, every history size, the sliding-learning experiment, or every table in the paper.

## Provenance and scope

- Repository: <https://github.com/stephanyrajeh/TGF>, commit `af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`.
- Paper: Latapy and Rajeh, *Trivial Graph Features and Classical Learning are Enough to Detect Random Anomalies*, ICDM 2025, DOI `10.1109/ICDM65498.2025.00142`.
- The faithful learning reproduction uses dataset files and the authors' already-injected 5% streams directly from that commit. Reusing committed streams permits an exact comparison to the authors' cached features. A separate fresh preprocessing/injection run executes the original raw-data notebooks and shell scripts, as detailed below; this fresh random realization is not substituted into the exact-cache learning reproduction.
- The original `3-FeatureGeneration/main.py` is executed as a subprocess **without source edits**. The original notebook's data-frame preparation, chronological split, undersampling, training and evaluation cells are executed verbatim. Plotting and display-only cells are omitted. Inputs are supplied locally and imports are provided by the wrapper.
- Zero-based unchanged notebook cell indices are `8, 11, 13, 17, 20, 22, 25, 31, 35, 36, 42, 43, 47, 49`. Cell 17's deprecated `DataFrame.append` remains available under the actual pandas 1.5.3 environment; no compatibility patch was needed.
- Runtime: Python 3.11.4, NumPy 1.24.3, pandas 1.5.3, scikit-learn 1.3.0. Input, source and notebook SHA-256 hashes are recorded in `results/official/*_metadata.json`.
- Source-controlled upstream files remain unchanged. Python imports may create untracked bytecode cache files.

## Actual official protocol

The notebook uses a **chronological 70%/30% split**, not a shuffled train/test split. After dropping the first 1,000 featureless warm-up rows, the split position is `round(len(df)/100*70)`. Normal training rows are sampled without replacement to match the number of training positives, using seed 12. The balanced training rows are shuffled using seed 42. `RandomForestClassifier(random_state=42)` is fit with its library defaults (100 trees in the recorded version). Labels, timestamps, node IDs, history metadata and runtime cost are excluded from the RF inputs.

The faithful run uses RF seed 42. Four additional runs change **only** the RF seed to 0, 1, 2 or 3. Consequently, the standard deviations below measure RF randomness on one fixed injected stream and one fixed split; they do not measure injection or dataset uncertainty.

| Dataset | Normal rows | Injected rows | Total rows | Feature rows | RF features | Balanced training rows | Test rows / positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens 100K | 100,000 | 5,000 | 105,000 | 104,000 | 44 | 6,934 | 31,200 / 1,489 |
| UCI Messages | 59,795 | 2,989 | 62,784 | 61,784 | 33 | 4,148 | 18,535 / 871 |

"5% injection" means the injected count is 5% of the **normal** count, rounded down. Thus the final positive prevalence is approximately 4.762%, not 5% of the combined stream. MovieLens here is the original **100K ratings** example, not MovieLens 1M. The upstream bipartite preprocessing prefixes both partitions, then sorts endpoint strings, resulting in `bottom_*` movie nodes in the first stream column and `top_*` user nodes in the second. Preserve this convention when matching upstream feature names.

## Reproduction results

The regenerated feature JSON was compared against every upstream cached row, excluding only the measured `cost` field. There were **zero mismatching rows** among all 105,000 MovieLens rows and all 62,784 UCI rows. The upstream cache is UTF-16; regenerated output is UTF-8 JSON with the same parsed values.

| Dataset | Stored notebook ROC-AUC (rounded) | Reproduced seed-42 ROC-AUC | Seed-42 AP | 5-RF-seed ROC-AUC mean ± sample SD | 5-RF-seed AP mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| MovieLens | 0.9880 | 0.987971 | 0.815704 | 0.988070 ± 0.000445 | 0.812675 ± 0.003847 |
| UCI Messages | 0.9635 | 0.963540 | 0.484083 | 0.963900 ± 0.000920 | 0.489404 ± 0.009986 |

AP is scikit-learn `average_precision_score`, the non-interpolated average precision; it is not trapezoidal integration of the PR curve. Precision at the highest-scored 1% is also reported. No test-set threshold tuning is used; optional precision/recall/F1 columns use the classifier's fixed 0.5 decision rule.

## Audited feature definitions

The history contains the last H **events**, with repeated pairs contributing link weights. Unweighted degree counts distinct neighbors; weighted degree counts incident events. Only nodes and pairs active in the current history contribute to the distributions.

| Feature family | Exact released meaning |
|---|---|
| Degree / weighted degree / pair weight | Values in the current maintained H-event graph |
| `*_same_*` | Number of active nodes/pairs having exactly the endpoint/pair's value |
| `*_greater_*` | Number of active nodes/pairs having a strictly greater value; `DSC.val2pos`, not a normalized percentile |
| `*_nb_one`, `*_nb_two` | Number of active counters equal to 1 or 2 |
| `*_min`, `*_max`, `*_median` | Active-counter summary; median is decreasing-sorted index `floor(n/2)`, i.e. the lower central value for even n, not the average of two central values |
| Unipartite endpoint ordering | First canonicalize endpoint labels lexicographically, then locally swap feature endpoints iff the first unweighted degree exceeds the second. Weighted-degree features follow that same degree-based swap. |
| Bipartite endpoint ordering | Keep the two stream partitions distinct; do not swap by degree |

The original program calls `link_arrival(t,u,v)`, evicts excess history, then emits the current event's features. Thus H1000 contains the current event plus up to 999 preceding events. In particular, its current pair weight is at least one and its endpoints necessarily occur in the graph. This is **current-event inclusion in the feature definition**. The anomaly label is not used to maintain the graph and is not passed to the classifier; this audit does not establish label leakage. The implementation is valid for an after-arrival representation, but differs from a detector required to score using strictly prior history.

Timestamp ties are processed sequentially in the official file order. The official README's informal strictly-increasing-time description should therefore not be interpreted as a guarantee that input timestamps are unique.

## Injection semantics and limits

The shell injection scripts draw timestamps from all unique timestamps and endpoints from the full-stream node sets. They therefore know the complete node/time universe, including future activity. Bipartite endpoints are sampled from their separate partitions. Loops and duplicate **triples** are removed, and only exact triples `(t,u,v)` already in the normal stream are excluded. An injected node pair can be a genuine pair at another timestamp. This is weaker than excluding every historically or globally observed pair.

These labels mean "inserted by this synthetic generator," not confirmed real fraud. High ROC-AUC on this stream is evidence of synthetic anomaly discrimination under the stated protocol. It should not be presented as real-world fraud accuracy.

## Paired timing experiment

`--paired-timing` keeps the official injected stream, endpoint strings, ordering, 1,000-row warm-up, 70/30 split, balanced training indices and all five RF seeds fixed. It changes extraction to use only events with timestamps **strictly earlier** than the candidate. All rows sharing a timestamp are scored before any row in that timestamp group is inserted. The maintained history contains at most the previous 1,000 events.

This wrapper executes the original initialization, `DSC`, arrival and departure definitions verbatim. Its absent-aware feature extractor is validated by recomputing the original after-update features and comparing every feature of every eligible row exactly to the upstream-generated matrix. An unseen endpoint/pair has value zero, equal-count zero and greater-count equal to the active population size. It is not inserted as a phantom zero-valued member of the graph.

The original row split is deliberately retained to isolate extraction, so it can divide an equal-timestamp group; boundary timestamps are recorded. This contrast is only about extraction timing and tied timestamps. It does not fix full-universe injection assumptions or substitute the project's new structure-preserving generator. The strict-causal benchmark in the main experiment should be reported separately.

Machine-readable outcomes are `results/official/paired_timing_metrics.csv`, `paired_timing_summary.csv` and `*_paired_timing_metadata.json`. Per-test-row scores and labels are retained in compressed NPZ files for additional paired analysis.

| Dataset | Extraction protocol | ROC-AUC mean ± sample SD | AP mean ± sample SD | Precision at top 1% mean ± sample SD |
|---|---|---:|---:|---:|
| MovieLens | Official after-arrival | 0.988070 ± 0.000445 | 0.812675 ± 0.003847 | 0.858974 ± 0.010869 |
| MovieLens | Strict prior timestamp | 0.983856 ± 0.000650 | 0.747420 ± 0.004984 | 0.823718 ± 0.023985 |
| UCI Messages | Official after-arrival | 0.963900 ± 0.000920 | 0.489404 ± 0.009986 | 0.537634 ± 0.020116 |
| UCI Messages | Strict prior timestamp | 0.962814 ± 0.000847 | 0.491464 ± 0.003436 | 0.523656 ± 0.012377 |

MovieLens AP decreases by 0.065256 (6.53 percentage points) under strict prior-timestamp extraction. UCI AP changes by +0.002060, which is small compared with its observed RF-seed variability. Thus current-event/tie handling affects the MovieLens result materially but does not explain away strong random-injection discrimination on either dataset. This paired comparison changes both current-event inclusion and visibility of equal-timestamp peers; it cannot separate those two effects. Neither recorded 70/30 boundary actually divides a timestamp group. Post-update extraction was verified exactly on all 104,000 eligible MovieLens rows and 61,784 UCI rows before the alternative models were fit.

## Fresh original preprocessing and injection execution

The `--fresh-pipeline` command runs both original preprocessing notebooks from raw data in `results/official/fresh_pipeline/<dataset>/`, executing **all code cells unchanged**. The generated cleaned rows are checked against the committed cleaned data in exact order. It then executes the original `bip_build_injectable.sh`/`build_injectable.sh` at 20%, `inject.sh` at 5%, and `bip_check.sh`/`check.sh` at 5%.

Windows Git checkout converted shell files to CRLF, so the wrapper copies their exact LF **Git blobs** from the pinned commit into the isolated directory. Git Bash's `python3` resolves to a Microsoft Store alias on this host; a shell function routes that command to the then-installed Python interpreter and normalizes its stdout from Windows CRLF to Linux LF. Without the latter adjustment, embedded carriage returns survive `paste` and cause the original checker's temporal-order check to fail; the initial failure log is retained. No sampling algorithm or source logic is patched. Every script's Git-blob SHA-256, original cell indices, command arguments, stdout/stderr, input/output hashes and checks are recorded beside its output.

The unseeded original `random.seed()` sampler and `shuf` produce a new realization on each run. This execution validates the original preprocessing and injection stages; fixed committed-stream features and learning remain the basis for the faithful metric table. Fresh injected streams are retained with hashes, and no fresh-stream training result is claimed. The original checker must report `everything seems ok`; additional Python assertions verify original-normal preservation, exact injected counts, unique triples, nondecreasing timestamps, no loops, no injected original triple, and MovieLens partition preservation.

| Dataset | Reprocessed cleaned file SHA-256 equals committed file | Fresh normals | Fresh injected | Original shell checker | Independent assertions |
|---|---|---:|---:|---|---|
| MovieLens | Yes, byte-for-byte | 100,000 | 5,000 | Passed | Passed |
| UCI Messages | Yes, byte-for-byte | 59,795 | 2,989 | Passed | Passed |

## Output integrity and schemas

`--verify-output` independently reloads all 20 saved prediction files (497,350 scored rows across models), recomputes ROC-AUC and AP to an absolute tolerance of `1e-12`, checks paired test labels/row indices, validates finite bounded scores, and verifies complete matrix column/label/row alignment. All checks passed; see `results/official/integrity_checks.json`.

- `paired_timing_metrics.csv`: one row per `(dataset, protocol, rf_seed)`, 20 rows, with sample counts, prevalence, feature count and evaluation metrics.
- `paired_timing_summary_flat.csv`: one row per `(dataset, protocol)` with metric `_mean` and `_std` columns; convenient for report generation.
- `paired_timing_deltas.csv`: one row per `(dataset, rf_seed)`, containing strict-prior minus official metric changes.
- `*_predictions.npz`: arrays `row_index`, `y`, `score`; row indices refer to original committed-stream positions, zero-based.
- `*_matrix.npz`: arrays `X`, `y`, `row_index`, `columns`, including every row after the fixed 1,000-row warm-up.
- `fresh_pipeline/<dataset>/metadata.json`: hashes, raw preprocessing cell provenance, original shell command arguments and validation results.

## Rerun

From the project root:

```powershell
python scripts/official_reproduction.py
python scripts/official_reproduction.py --paired-timing
python scripts/official_reproduction.py --fresh-pipeline
python scripts/official_reproduction.py --verify-output
```

The first command reruns both complete datasets. `--dataset movielens` or `--dataset ucimessages` restricts either command. Runtime measurements in the first run include IO and feature serialization for feature generation; seed-42 `fit_seconds` includes unchanged notebook preparation and evaluation cells, while the extra-seed values time only fitting. Therefore these columns are diagnostic elapsed times, not a fair model speed comparison. The released `cost` field itself times arrival/eviction but not the subsequent full feature extraction/serialization.
