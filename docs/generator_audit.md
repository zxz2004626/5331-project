# Independent generator audit

Audit scope: `src/graphbench/generate.py`, `src/graphbench/data.py`, all 40 main
cache manifests and NPZ files (2 datasets × 5 seeds × 4 generators). Generation
source and caches were not modified by the audit. The dedicated tests are in
`tests/test_generate.py`.

## Checks and findings

All 40 main caches passed the following checks against the complete source
datasets and their manifests:

- Complete status and generation source hash match.
- Every feature is finite, and every stored maximum history timestamp is
  strictly less than the scored event timestamp.
- Exact requested injection count in each train/validation/test split:
  round(N_normal × 0.05 / 0.95), using only post-warmup normal events.
- Injected pairs are unique across the entire generated stream, are absent
  from the full original stream, contain no self-loops, and satisfy the
  MovieLens partition constraints.
- Every output event's timestamp belongs to its declared original-data split.
- Every injected pair's lifetime prior-interaction feature is zero, as required
  by the global-absence definition.

Synthetic end-to-end tests independently replay a second graph history and
compare **every stored feature vector** before inserting any current-timestamp
normal or anomalous event. They also rebuild identical caches and compare all
NPZ arrays and every audit record, test rejection accounting and exhausted
support, and require stale caches to fail explicitly. The test fixtures cover
both undirected and bipartite graphs.

Data loader tests verify stable ordering within tied timestamps and exact
60%/80% **elapsed-time** boundaries. A timestamp equal to a boundary belongs to
the later split. No timestamp is divided between splits. Warmup excludes the
timestamp of event H−1 (zero-based) and all other events sharing that timestamp;
thus at least H observed events are excluded. Warmup affects scored rows, not
the graph's capacity to ingest those observations.

The causal feature history uses only earlier timestamps. Full-stream node
vocabulary, full-stream observed-pair exclusion in the main condition, and the
elapsed-time split endpoints are known benchmark construction information.
They are not an online generator claim. In the past-only sensitivity, future
observed pairs are eligible, but the caller still excludes current-timestamp
normal pairs to prevent contradictory labelled copies at the same time.

## What the matching algorithm guarantees

The fixed algorithm samples a reservoir of 128 legal candidate pairs, retains
16 with smallest endpoint log-degree/strength distance, then retains 4 with
smallest local-feature distance, and finally chooses the temporal-distance
minimum among those 4. G1 randomly chooses from the endpoint 16; G2 randomly
chooses from the local 4; G3 chooses the temporal minimum. G0 samples one legal
pair. Stable endpoint sorting and seeded randomized local/temporal ties make
these operations reproducible for fixed software/data/configuration.

These are nested rankings **within a sampled reservoir**, not a search of the
complete legal graph and not hard matching calipers. Later stages are not
required to choose the best endpoint match. A zero-distance candidate excluded
at an earlier stage cannot be recovered. Synthetic tests explicitly exercise
these distinctions. The four generated streams subsequently have different
mixed histories, so aggregate distances do not have a mathematical monotonicity
guarantee across generators.

## Actual matching quality

The table below averages the manifest's per-anchor distances over the five
seeds. Endpoint distance is RMS difference in log1p(degree/strength), local
distance is RMS difference in log1p(local features), and temporal distance is
RMS difference in the six matching quantities defined in `generate.py`.
The “> log 2” column is the fraction of injected events whose endpoint RMS
distance exceeds log(2). It is a diagnostic threshold, not an enforced caliper
or a bound on individual degree ratios.

| Dataset | Generator | Endpoint distance | Local distance | Temporal distance | Endpoint > log 2 |
|---|---|---:|---:|---:|---:|
| MovieLens | G0 | 2.6187 | 1.3356 | 1.6070 | 96.84% |
| MovieLens | G1 | 2.2018 | 1.3300 | 1.4350 | 90.60% |
| MovieLens | G2 | 2.1320 | 1.2138 | 1.4056 | 88.40% |
| MovieLens | G3 | 1.6168 | 1.0558 | 1.0721 | 76.33% |
| UCI messages | G0 | 1.9304 | 0.0520 | 1.3720 | 92.39% |
| UCI messages | G1 | 1.2655 | 0.0546 | 0.9860 | 76.53% |
| UCI messages | G2 | 1.2660 | 0.0508 | 0.9882 | 76.54% |
| UCI messages | G3 | 1.1161 | 0.0490 | 0.7779 | 69.06% |

The next table describes **test-set marginal distribution gaps**, averaging
KS statistics across the five seeds and the columns in each named group.
Endpoint has only four columns: u/v degree and u/v weighted degree. Local
contains three UCI columns or two MovieLens columns. Temporal contains six
columns, with `pair_seen_before` explicitly excluded and shown separately.
These are descriptive averages, not multivariate equivalence tests.

| Dataset | Generator | Endpoint mean KS | Local mean KS | Temporal mean KS | Pair seen KS |
|---|---|---:|---:|---:|---:|
| MovieLens | G0 | 0.6397 | 0.6077 | 0.6118 | 0.0000 |
| MovieLens | G1 | 0.4797 | 0.5628 | 0.4809 | 0.0000 |
| MovieLens | G2 | 0.4484 | 0.5031 | 0.4655 | 0.0000 |
| MovieLens | G3 | 0.3501 | 0.4007 | 0.3671 | 0.0000 |
| UCI messages | G0 | 0.8096 | 0.0949 | 0.7421 | 0.8209 |
| UCI messages | G1 | 0.6477 | 0.0982 | 0.6387 | 0.8209 |
| UCI messages | G2 | 0.6421 | 0.1065 | 0.6462 | 0.8209 |
| UCI messages | G3 | 0.5753 | 0.1035 | 0.5190 | 0.8209 |

The generators make MovieLens candidates closer on these diagnostics, but
substantial distribution gaps remain even for G3. On UCI, local marginal
separation is not reduced relative to G0. Therefore the results support an
evaluation of **progressively ranked, approximately matched injections**, not
a claim that endpoint/local/temporal shortcuts have been neutralized.

In UCI test data, 82.09% of normal events have a previously seen pair, whereas
all globally absent injected pairs have value zero. This shortcut is unchanged
across generators and cannot be matched under the global-absence constraint.
The MovieLens source has no corresponding test normal-pair repeat signal
(0%). A temporal feature-set gain on UCI must therefore be interpreted with
this strong prior-pair signal in mind; an ablation excluding this column is
needed to attribute gains specifically to recent activity or recency.

## Verification command

From the project root in PowerShell:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_generate.py tests/test_history.py -q
```

Observed result: **25 passed in 2.60 seconds**. This includes 14 generator/data
cases and 11 history cases. Tests are synthetic and do not regenerate or
invalidate the complete experiment caches.

## Additional attribution control: remove prior-pair history

`scripts/no_pair_ablation.py` implements the audit-motivated control for all
40 main dataset/generator/seed combinations. Its predictor set is the complete
extended feature set minus the single `pair_seen_before` column. It reuses the
corresponding **same-generator source extended RF's** validation-selected
`min_samples_leaf`, forest size, class weighting and random seed. It performs
one fit on train plus validation, with no new parameter search and no use of
test outcomes to choose parameters. Thus the control estimates the effect of
removing this explicit feature under the original selected model configuration.

Test scores and metrics are stored both for the complete test set and the
novel-normal subset (all anomalies plus only previously unseen normal pairs).
The removed column is used solely to define that evaluation subset, never as
an input to the ablated estimator. TGF's within-history pair weight remains in
the model, so this is not removal of every possible novelty signal.

```powershell
python scripts/no_pair_ablation.py --n-jobs 1 --available-only
```

The recorded experiment uses one execution thread to limit concurrent memory;
all statistical model parameters are reused as described above. The command
processes available completed source models sequentially and then
exits; rerunning resumes completed controls after checking source metadata,
cache, implementation and prediction hashes. Outputs are
`results/no_pair/runs/*_RF_extended_no_pair.json` and corresponding prediction
NPZ files. `results/no_pair/status.json` tracks completeness. A current
`results/no_pair/metrics.csv` is published only after all 40 controls pass
integrity checks. Its feature-set label is `extended_no_pair`, with the main
evaluation's metric names and `novel_` prefixes for the subset metrics.

`tests/test_no_pair.py` verifies that the fit receives only train/validation
rows, excludes exactly the intended column, reuses the source parameter with
one fit, scores the correct novel-normal subset, resumes without fitting,
invalidates corrupt or changed outputs, skips unavailable source models, and
withholds stale/incomplete summary tables. All five tests passed using a spy
estimator, so testing did not compete with the experiment's RF fitting work.

## Completed no-pair control and independent verification

All **40/40 controls completed**. A separate audit checked every run's source
metadata hash, generation-cache fingerprint, implementation hashes and
prediction hash; verified that statistical RF parameters equal the matching
full-extended source model; compared test row IDs, labels, endpoints,
timestamps and novel-subset masks with the corresponding main evaluation;
and recomputed all saved full-test and novel-subset metrics from prediction
arrays. All checks passed. The machine-readable verification is
`results/validation/no_pair_audit.json`.

The final full test suite also passed: **38 passed in 8.82 seconds**, recorded
in `results/validation/tests.txt`.

The following are five-seed mean AP values. Δ is **no-pair minus full extended**;
its standard deviation is computed from the five paired seed differences,
not from independently pooled runs. All comparisons use the same dataset,
generator, seed, test rows and source-selected statistical RF parameters.

| Dataset | Generator | Full extended AP | No-pair AP | Paired Δ AP ± SD |
|---|---|---:|---:|---:|
| MovieLens | G0 | 0.8787 | 0.8773 | −0.0014 ± 0.0030 |
| MovieLens | G1 | 0.7649 | 0.7666 | +0.0018 ± 0.0033 |
| MovieLens | G2 | 0.7615 | 0.7646 | +0.0030 ± 0.0035 |
| MovieLens | G3 | 0.6905 | 0.6965 | +0.0060 ± 0.0043 |
| UCI messages | G0 | 0.7032 | 0.6271 | −0.0762 ± 0.0185 |
| UCI messages | G1 | 0.5998 | 0.5155 | −0.0843 ± 0.0364 |
| UCI messages | G2 | 0.5689 | 0.4766 | −0.0923 ± 0.0280 |
| UCI messages | G3 | 0.5371 | 0.4589 | −0.0782 ± 0.0329 |

For UCI, deleting the explicit prior-pair flag reduces full-test AP by about
**7.6–9.2 percentage points**. The effect is much smaller on the subset that
removes previously seen normal pairs:

| UCI generator | Full extended novel-subset AP | No-pair novel-subset AP | Paired mean Δ AP |
|---|---:|---:|---:|
| G0 | 0.7068 | 0.7167 | +0.0099 |
| G1 | 0.6000 | 0.5993 | −0.0007 |
| G2 | 0.5691 | 0.5632 | −0.0059 |
| G3 | 0.5371 | 0.5411 | +0.0040 |

This contrast supports a specific interpretation: much of the flag's measured
benefit comes from distinguishing repeated normal interactions from synthetic
never-seen anomalies. It does not demonstrate better discrimination between
new normal and new anomalous interactions. This is an attribution control
under a fixed selected configuration, rather than a claim about the optimal
retuned model without the feature.

The UCI novel-normal subset has anomaly prevalence **0.227053**, versus about
0.05 in the complete test set. AP comparisons between feature sets are valid
within each identical subset; directly treating an AP increase from filtering
as a model improvement would change the random-ranking baseline. MovieLens's
novel subset equals its full test set (prevalence 0.05), so its two sets of
metrics coincide. Small MovieLens differences after removing the uninformative
column are compatible with the changed feature-subsampling sequence of the
random forest; removing a constant column need not preserve the trained forest.

The recorded controls use `--n-jobs 1`, while main/sensitivity RF experiments
use two execution threads. This is a resource setting, with tree count,
minimum leaf size, class weighting and random seed reused. Since checkpoint
fingerprints include `n_jobs`, reproduction should explicitly use one thread
for this control to resume the existing 40 results; passing two threads can
legitimately trigger their refitting even though the statistical specification
is unchanged.
