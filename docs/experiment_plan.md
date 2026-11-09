# Experimental protocol (fixed before detector results)

The authoritative specification is `kdd_proposal.pdf`, September 2026. The earlier `proposal.md` is background. This implementation tests synthetic edge detection, not real fraud detection.

## Data and provenance

Use the complete cleaned examples in the official TGF repository at commit `af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`: MovieLens 100K (100,000 events, 1,682 movies and 943 users), and UCI Messages (59,795 cleaned undirected events, 1,899 users). UCI originally contains 59,835 events; the official cleaning removes duplicate timestamp/pair rows. Directions and ratings are not used. MovieLens's official cleaned columns are **movie then user**; disjoint integer namespaces preserve this orientation. `data/manifest.json` records source URLs, SHA-256 hashes, time boundaries and counts.

## Official baseline and timing audit

Run the pinned author's feature generator unchanged with H=1000 and execute the original classical RF notebook cells against released 5% injected streams. Compare every non-timing feature against the author's cached features. The official RF notebook uses a chronological 70/30 row split, normal downsampling on training data (sample seed12), and RF100 with seed42. Repeat RF seeds for model variability, clearly distinguished from independent injection seeds. The official 5% means anomalies/normal count, giving 4.762% mixed prevalence. Audit the source before execution.

The timing sensitivity uses the *same* released injection, training indices, feature names, and RF settings, but freezes all features at a timestamp until its complete batch has been scored. Thus it changes only feature history timing, not global absence restrictions or model hyperparameters. Original feature extraction includes the current edge and preceding same-timestamp rows; that is self-inclusion and order dependence, not automatically label leakage.

## Causal extension and chronology

1. Split by elapsed time: `t_min+0.6*(t_max-t_min)` and `t_min+0.8*(t_max-t_min)`. Each timestamp belongs to one split. This is not an event-count quantile split. UCI's sparse late period makes its test set small; report actual counts.
2. Warm up through the complete timestamp containing observed event1000, score only later timestamps, and preserve all warmup events in history. Warmup selection is fixed across history sensitivities.
3. Main H history contains the last1000 **mixed normal and injected events**. Recent endpoint activity has an independent86,400-second window. Lifetime last-active time and prior-pair presence use only earlier events. Every event at t reads a frozen history with timestamp strictly less than t; after scoring, add normal rows in their stable source order, then injected rows in their saved order. Keeping a subset of an old timestamp at the H boundary is allowed and deterministic.
4. Features are evaluated on all observed events after warmup; there is no subsampling of the real data. Test features may use earlier unlabeled validation/test interactions, as in an online stream. Labels do not update history differently.
5. The node vocabulary and main global forbidden pair set use the complete observed dataset. This is an explicit **oracle benchmark construction constraint**, not a claim that the generator itself is deployable without future knowledge. Feature extraction remains causal. In the past-only sensitivity the forbidden set contains prior observed/injected pairs; exact current-timestamp observed duplicates remain excluded.

## Injection and finite candidate matching

The main anomaly fraction is5% of the final mixed scored data. In every split request `round(n_normal*p/(1-p))` anomalies; report achieved prevalence. Seeds are11,23,37,51,71. Sampling normal reference events with replacement in each split gives matched timestamps and paired reference schedules across generators. An injected pair is unique within its condition, is legal for the graph, has no self-loop, and meets the declared absence rule. No generated anomaly silently disappears. An inability to fill the finite legal pool raises a recorded support-limited error.

- G0: one uniformly sampled legal pair.
- G1: sample128 distinct legal pairs; retain16 with smallest RMS distance between `log1p(degree_u,degree_v,strength_u,strength_v)` and the reference profile; choose one uniformly.
- G2: retain that same endpoint set, then retain4 with smallest RMS distance over `log1p(local_features)` and choose uniformly. Distance ties use seeded random tie breaking.
- G3: retain both candidate sets, then choose minimum temporal RMS distance. The temporal vector contains log1p endpoint recent counts, capped log recencies divided by log1p(30days), and recent active flags. Prior-pair presence is intentionally **not** matched; global non-edges cannot match repeated normal pairs.

For UCI, endpoint orientation is ascending degree with numeric-ID tie breaking, local features are common neighbors, Jaccard and Adamic-Adar. For MovieLens, local features are simple three-edge paths and paths divided by endpoint degree product, avoiding the identically zero common-neighbor measure between partitions. All transformations and candidate quotas are fixed constants, with no test-label fitting. Candidate draws are deterministic per seed/reference ordinal. Mixed histories may subsequently differ between generators.

These are **ranked finite-pool approximate controls**, not hard tolerance or exact marginal distribution matching. G2/G3 retain candidate subsets within an event, but population distributions need not be nested or monotonically closer to normals. Report endpoint distance>log2 as a diagnostic, not a hidden rejection. Do not rename a poorly matched condition as a proven structure-preserving anomaly. No generator hyperparameter is selected from detector test scores.

Save t/u/v/label/split/features for every scored event, and per-injection reference ID/context, candidate support, stage distances, rejections and maximum history timestamp. Normal events before warmup remain reconstructable from the pinned input. Save every run's code hash, configuration, dataset hash and execution status.

## Models and model selection

Main factorial: 2 datasets ×4 generators ×2 feature sets (33/44 TGF vs TGF+local+temporal) ×2 classifiers ×5 seeds =160 independently fitted final models. Each is evaluated on all4 target generators with the same seed and split, producing640 cross-generator cells, including160 diagonal results.

- RF:100 trees, balanced_subsample weights, validation-select min_samples_leaf in {1,5}, other scikit-learn1.3 defaults. Use2 threads.
- LR: nonnegative log1p features, a StandardScaler fitted within the training pipeline, balanced class weights, liblinear, max_iter1000, validation-select C in {0.1,1,10}.
- Select by validation average precision only; deterministic candidate order breaks exact ties. Refit the selected configuration on train+validation before final test. Target-generator data never chooses a source-generator model. Test labels enter metrics only.
- RF diagonal ablations: local-only, temporal-only, TGF+local, TGF+temporal, local+temporal;5 seeds and both datasets/all generators (200 further final models). Main TGF and full models complete feature sufficiency and leave-one-group-out contrasts. The groups contain2/3 local and7 temporal features.

## Sensitivity scope

Each sensitivity changes one parameter from main, retaining5 seeds, both datasets, G0/G3, RF, and both feature sets. Conditions: anomaly fraction1%; anomaly fraction10%; H500; H2000; past-only pair exclusion. This is200 additional diagonal model evaluations; it intentionally does not repeat LR or the full cross-generator grid. The same main G0/G3 results serve as controls. The separate official timing audit isolates the update-order difference. Optional MIDAS and coordinated/micro-cluster experiments are outside the required single-edge study and are not run.

## Supplementary attribution control

The generator audit identified lifetime prior-pair presence as an unmatched shortcut, motivating an additional40-model RF control (two datasets, four generators, five seeds). Remove only `pair_seen_before` from the extended features; reuse each source generator's complete extended model's validation-selected hyperparameter, then fit train+validation once and score the same test rows. No new hyperparameters are selected. This is a follow-up attribution control, not part of the original160-model factorial. The run uses one execution thread to limit memory. Predictions, source-model provenance and metrics are saved; these40 fitted estimators are not separately serialized. Removing a column can also change RF feature subsampling, and remaining TGF features still encode some repeated-edge information.

## Metrics, diagnostics, and interpretation

Primary metric is **average precision (AP)**, the step-weighted precision-recall area summary from sklearn. Also save trapezoidal PR-AUC separately, ROC-AUC, Precision@K and Recall@K. K equals the true test anomaly count; these last two numbers are therefore identical, and are retrospective ranking metrics, not deployment threshold selection. Score ties use stable cache-row order, which is recorded. Mean±sample SD across seeds describes conditional injection/model randomness, not uncertainty across real datasets or independent chronological folds.

Before interpreting detector performance, compute signed SMD (pooled class variance), two-sample KS and Wasserstein distances for every feature/split, along with reference matching distances, candidate acceptance and rejection categories. Plot endpoint/local/temporal distributions; show prior-pair presence separately because it is an unmatched novelty shortcut. Retain paired per-seed differences for G3-G0 and extended-TGF without assuming any sign. Cross-generator effects compare a fixed trained model on different generators. RF impurity importance is descriptive, not a causal attribution.

Supplementary novel-normal-only metrics filter repeated normal pairs from each test set using `pair_seen_before`. The changed class prior is reported; AP values under this filter cannot be compared directly with full-set AP as a difficulty claim. Pair filtering uses lifetime prior mixed history and does not retrain the detector. A future benchmark could instead match an equal novelty prior.

## Validation requirements

Tests compare causal-engine TGF formulas to the original CLI, validate graph/window/recency semantics, timestamp-batch invariance, legal negative generation, class counts, checkpoint fingerprints, metric definitions, source-only model selection and feature-column alignment. Failed/incomplete artifacts are retained in failure records and excluded until successfully rerun. Tables/figures are generated from saved metrics, not hand-entered or simulated numbers.
