# Independent methodological review

Source of requirements: `proposal.pdf`, pages 3–7. The PDF supersedes the older `archive/proposal_draft.md`. This note is a design review, not a record that an experiment has already run.

## Required scope and completion accounting

- Run the released TGF implementation unchanged on MovieLens and UCI Messages, recording the exact source revision, dataset hashes, arguments, logs and returned metrics. A reimplementation or a changed train/test split is not an unchanged official reproduction.
- Keep the official reproduction separate from the pre-arrival extension. Audit the order of history insertion and feature extraction rather than assuming the official implementation is causal.
- The core extension has 2 datasets × 4 generators × 2 classifiers × 2 feature sets × 5 seeds = **160 same-generator test evaluations** at a declared 5% anomaly ratio. Validation and repeated fitting are additional work, not additional independent datasets.
- Reuse each trained model to obtain the full 4 × 4 generator matrix. This gives 640 cells over the complete factorial design, including the 160 diagonal cells; it does not require 640 distinct model fits.
- Required analyses include generator fidelity, endpoint/local/temporal feature sufficiency and leave-one-group-out ablations, cross-generator transfer and the declared ratio/history/protocol sensitivity checks. Optional MIDAS, micro-clusters and neural methods should not displace these.
- Record any unexecuted part explicitly. A small demonstration with fewer seeds, a smaller data subset or a narrower model matrix is a pilot, even when the code exposes all planned options.

## A defensible strict-history implementation

1. Sort events stably and group by exact timestamp. Compute all generator statistics and detector features for time `t` from a frozen state containing only events with timestamp `< t`. Insert the complete timestamp batch only after every event at `t` has been processed.
2. Assign an entire timestamp to one chronological split. Interpret “60/20/20 by time” as boundaries at 60% and 80% of the observed time span unless a different interpretation is explicitly documented. Fractions of rows are not fractions of elapsed time.
3. Fix the history definition, units, warm-up handling and eviction boundary. For a duration `W`, choose and document a convention such as `[t-W, t)`. A last-`N`-events history must explain how ties at the oldest boundary are handled.
4. Fit scales, bins, transformations and model hyperparameters only on the training/validation periods appropriate to their role. Parameters for generator matching should be declared in advance or calibrated on an early training-history segment before applying them to later events. There should be no fit on test labels or test feature distributions.
5. The proposal's main **global non-edge** rule excludes pairs found anywhere in the observed stream. This deliberately uses future edge identities for synthetic label construction. Features can remain pre-arrival causal, but the generator is not a fully online deployable procedure. Call this distinction out explicitly. The past-only exclusion sensitivity removes this future edge-identity filter and permits eventual real pairs.
6. Declare whether synthetic edges enter future history. A mixed-stream history includes previous observed and injected events and models contamination. A clean observed-only history isolates generator effects and allows efficient common caches, but it uses the benchmark labels to exclude injected events. It is an oracle-history experimental control, not a complete simulation of an online detector. Report the choice and avoid silently exchanging the two protocols.

## Injection design and legal support

Use a common interface returning each injected event and an audit record. To isolate endpoint matching, use the same reference observed-event timestamps and the same split-level anomaly quotas for G0–G3 within a seed. State that this timestamp-anchored extension control may differ from the unchanged official random-timestamp procedure.

Define the anomaly ratio unambiguously. If `p` is the fraction of anomalies among all evaluated events, `n_anomaly ≈ p / (1-p) * n_observed`; `p * n_observed` has a different denominator. Publish realized ratios separately for train, validation and test. Use exact, predeclared rounding and prevent silent quota reduction.

- G0 samples valid endpoints and does not intentionally match their profiles.
- G1 matches endpoint degree/weighted degree/activity profiles, using frozen historical values and explicit log transforms or training-calibrated scales.
- G2 retains the declared G1 constraints and additionally matches local context. In the unipartite graph, common neighbors, Jaccard or Adamic–Adar are natural candidates. In a bipartite graph, the common-neighbor count of a user and an item is identically zero. Use length-three path/co-interaction features or a separately declared same-partition projection instead.
- G3 retains G1/G2 controls and matches recent endpoint counts, activity indicators and recency. A large common timestamp sentinel for never-active nodes can itself become a shortcut; include an explicit inactivity indicator and inspect its distribution.

Exact hard constraints can be support-limited. Soft weighted nearest-candidate objectives trade one discrepancy for another and do not mathematically preserve earlier matching constraints. In that case use labels such as “endpoint/local/temporal targeted sampling” unless empirical diagnostics justify the nested interpretation.

For every generated event, save dataset, seed, generator, timestamp, split, reference event ID, reference endpoints, sampled endpoints, candidate-pool size, eligible pool size, per-group distances, selected distance, attempt count and rejection counters. Distinguish “no candidate found in a finite sampled pool” from “no legal candidate exists.” Record whether duplicate means an identical timestamped event or an endpoint pair appearing anywhere in the injected stream. Use one definition consistently.

Reject or explicitly classify self-loops, partition violations, prohibited observed pairs and duplicate injections. Do not quietly replace an unsupported G3 sample by an unconstrained G0 edge. Predeclare a support failure rule and record failures as machine-readable artifacts.

## Diagnostics that determine what the benchmark measures

Global non-edges always have historical pair count zero. Observed interactions may be repeated pairs. Consequently even a strongly matched generator can leave a powerful novelty shortcut that endpoint/local/temporal matching cannot remove. For every split and generator, report:

- the fraction of normal events with previous pair history;
- distributions of historical pair count and pair recency;
- normal and anomaly rates of unseen/inactive endpoints;
- local-feature zero rates and recent-activity zero rates;
- diagnostic differences both to the selected reference contexts and to the full normal comparison population.

A useful supplemental evaluation restricts the normal test population to first-seen pairs, preserving synthetic anomalies and reporting the resulting changed prevalence. Treat this as a separate analysis, not a replacement for the predeclared main test. If prevalence changes, AP numbers cannot be directly compared without noting that change.

Compute standardized mean differences, KS statistics, Wasserstein distances and empirical distributions for each controlled group and for potential unrelated shifts. Report the transformations and units. For normalized Wasserstein distance, use a training-period scale; handle zero variance explicitly rather than dividing by zero. Distribution plots should show anomalies and normal events with comparable normalization despite the class imbalance.

Also report candidate support, acceptance rates, matching distances, endpoint reuse, duplicate rates, injection times and split-level quotas. A lower mean matching objective alone is not evidence that every intended feature distribution became closer. Do not declare monotonically harder anomalies from G0–G3 by design. Results from a legal generator with failed fidelity can still be reported as a distinct sampling mechanism, with the failed preservation claim removed.

## Evaluation and uncertainty

- Use the same split, observed rows, anomaly quota, model search budget and seed pairing across generators. Tune hyperparameters using validation data only; save the selected settings for every condition.
- Label `average_precision_score` as **average precision (AP)**, a common PR-area summary. Trapezoidal area under a PR curve is a different statistic; do not silently interchange them.
- ROC-AUC and AP require both classes. Fail clearly on a single-class evaluation set. Confirm score direction is anomaly-positive.
- At `K = number of test anomalies`, Precision@K equals Recall@K algebraically because both denominators are `K`. Reporting both is compliant with the proposal but does not provide two independent perspectives. Specify deterministic tie breaking, especially for small forests and score ties.
- Summarize seeds with mean, standard deviation and paired differences. Reusing one real stream across five seeds measures randomness of injection/model fitting; it is not evidence of uncertainty over five independent datasets.
- Keep feature-set experiments on identical cached events. Sufficiency models and leave-one-group-out models must be refit. Raw Random Forest impurity importances can be biased toward variables with many possible split points; feature-group ablations or permutation importance provide more useful corroboration.
- Cross-generator tests reuse the row generator's trained model without refitting on the column generator. Any probability threshold must come from training/validation. Save every test prediction and its event ID so matrix cells can be reconstructed.
- A diagonal advantage suggests generator-specific signals within these synthetic mechanisms. It does not establish real fraud/attack performance.

## Minimum meaningful correctness tests

1. **Current-event exclusion:** changing the current pair's target label or adding its current event must not change its pre-arrival features.
2. **Equal-time permutation invariance:** reorder a timestamp batch and verify that all event features and legal candidate history statistics are unchanged.
3. **Future append invariance:** adding later observed events leaves prior features unchanged. Under global non-edge injection, it may change candidate legality; test that exception separately. Under past-only injection, prior generator outputs should also be unchanged when seeded deterministically.
4. **History eviction boundaries:** construct events exactly at `t-W`, just before it and at `t` and verify the declared convention.
5. **Graph semantics:** verify MovieLens endpoint partitions and meaningful nonzero length-three path features on a small hand-built bipartite graph; verify UCI orientation/symmetrization consistently.
6. **Legality and quotas:** exact split anomaly counts, no prohibited pair, no self-loop and no duplicate under the declared definition; explicit support-failure behavior.
7. **Training-only fitting:** a shifted test distribution must not change fitted scaler/bin parameters or chosen training-only generator calibration.
8. **Metric sanity:** perfect/reversed/constant scores, deterministic ranking ties, correct anomaly-positive direction and Precision@K = Recall@K under the proposal's K.
9. **Cache identity:** predictions map one-to-one to the expected event IDs; stale caches with different dataset/config/source hashes are rejected.

## Reproducibility artifacts and reporting language

Every completed run should retain its config, dataset checksums, source revision or source-file hashes, software versions, seed, split boundaries, generator audits, feature schema, model parameters, predictions, metrics, timing and status. Aggregate reports should be generated from these files rather than manually copied values.

Keep at least three statements separate in the final report: (a) the unchanged official implementation's result, (b) the audited differences in official versus pre-arrival behavior, and (c) the extension's synthetic-generator results. A reimplementation that resembles TGF is a “TGF-style baseline” until its feature equivalence is verified. Synthetic injected edges are known labels by construction; observed edges are assumed normal and may contain natural anomalies.
