# Independent review of the completed main experiment

This review concerns `results/evaluation/main`, the 5% anomaly experiment with a last-1,000-event history. It does not treat sensitivity runs or the unchanged official TGF reproduction as interchangeable with this experiment. Machine-readable validation is in `results/validation/main_audit.json`.

## Completion and independent validation

The main output contains exactly 360 fitted estimators and 840 test prediction files:

- 160 core estimators: 2 datasets × 5 seeds × 4 training generators × 2 classifiers × 2 feature sets.
- 640 core evaluations: each core estimator tested on all four generators.
- 200 additional RF estimators and evaluations: five feature ablations, tested on the training generator, for each dataset/seed/generator.
- 168 summary groups, each with the five declared seeds 11, 23, 37, 51 and 71.

The audit independently recomputes every main and novel-normal-subset metric from all 840 saved prediction files. AP is reconstructed from equal-score threshold groups, ROC-AUC from the Mann–Whitney rank statistic, trapezoidal PR area from threshold interpolation, and top-K metrics from a stable descending-score ordering. This checks the results without calling the evaluator's metric function. The largest absolute metric discrepancy is 3.33 × 10⁻¹⁶. All 840 flattened CSV rows and 168 summary groups also agree with independently reaggregated raw records to floating-point precision.

The audit checks target event IDs, endpoint IDs, timestamps, labels, novel-subset masks, cache hashes, configuration hashes, source-code hashes, selected validation hyperparameters and run/model fingerprint links. It also loads every serialized estimator and compares its predictions on 32 deterministic diagonal-test examples with the saved predictions; the maximum absolute probability discrepancy is 4.44 × 10⁻¹⁶. RF execution used 2 jobs and 100 trees; an earlier discussion of increasing the number of jobs was not an executed configuration change. No model was retrained for this audit. All checks pass, with no unresolved failures.

## Main result: measured performance depends strongly on the generator

The following values are same-generator RF average precision, mean ± sample standard deviation over five seeds. AP is the primary PR-area summary used here; the separately stored trapezoidal PR area is a different statistic.

| Dataset | Features | G0 | G1 | G2 | G3 |
|---|---|---:|---:|---:|---:|
| MovieLens | TGF | 0.8298 ± 0.0091 | 0.6018 ± 0.0067 | 0.5623 ± 0.0054 | 0.3719 ± 0.0218 |
| MovieLens | Extended | 0.8787 ± 0.0073 | 0.7649 ± 0.0111 | 0.7615 ± 0.0091 | 0.6905 ± 0.0137 |
| UCI Messages | TGF | 0.5988 ± 0.0352 | 0.4428 ± 0.0624 | 0.4126 ± 0.0473 | 0.3568 ± 0.0161 |
| UCI Messages | Extended | 0.7032 ± 0.0396 | 0.5998 ± 0.0429 | 0.5689 ± 0.0963 | 0.5371 ± 0.0336 |

The paired same-seed AP change from G0 to G3 is −0.4579 ± 0.0200 for MovieLens TGF and −0.2421 ± 0.0502 for UCI TGF. The corresponding extended-feature changes are −0.1882 ± 0.0189 and −0.1662 ± 0.0585. These numbers support sensitivity to the benchmark generator. They do not establish the realism of G3 or performance on real fraud.

RF is not uniformly superior to Logistic Regression. For example, UCI G0 with extended features has LR AP 0.7732 ± 0.0226 versus RF 0.7032 ± 0.0396. The report should present both predeclared classifiers rather than select one after observing the test results.

## Transfer is asymmetric and generator specific

The cleanest transfer comparison fixes the **test generator** and changes only the training generator. For the G3 test data:

| Dataset | Features | Train G0, test G3 | Train G3, test G3 | Paired AP advantage of G3 training |
|---|---|---:|---:|---:|
| MovieLens | TGF | 0.2583 | 0.3719 | 0.1136 ± 0.0164 |
| MovieLens | Extended | 0.4391 | 0.6905 | 0.2515 ± 0.0162 |
| UCI Messages | TGF | 0.2273 | 0.3568 | 0.1294 ± 0.0054 |
| UCI Messages | Extended | 0.3233 | 0.5371 | 0.2138 ± 0.0413 |

Training on the more constrained generator does not automatically provide a detector that transfers back to random anomalies. For example, UCI RF with extended features scores AP 0.2569 when trained on G3 and tested on G0, versus 0.7032 for G0-to-G0. That is evidence against treating the generator sequence as a universal curriculum of increasingly general anomaly signals.

Do not claim that every column has its best value on the diagonal. UCI RF with TGF features trained on G1 reaches 0.3681 on G3, slightly above the G3-trained value of 0.3568. UCI RF with extended features trained on G1 reaches 0.6004 on G2, above G2-to-G2 at 0.5689. The supported statement is that transfer is often asymmetric and sometimes substantially worse, with specific exceptions visible in the full matrix.

## Temporal features drive most of the feature-extension benefit

On G3, adding all extended features to RF increases AP by a paired 0.3186 ± 0.0213 on MovieLens and 0.1803 ± 0.0278 on UCI. The ablations locate most of this improvement in the temporal group:

| Dataset, G3 | TGF | TGF + local | TGF + temporal | All features | Temporal only | Local only |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens | 0.3719 | 0.3756 | 0.6923 | 0.6905 | 0.6497 | 0.1036 |
| UCI Messages | 0.3568 | 0.3610 | 0.5505 | 0.5371 | 0.4608 | 0.0554 |

Local-only UCI performance is close to the approximately 0.05 AP of a constant-score baseline. Adding local features does not consistently improve the temporal model: on UCI G2, TGF + temporal reaches AP 0.6145 while the full extension reaches 0.5689 with substantial seed variation. Thus “a few temporal features recover much of the lost performance” is better supported than “richer local structure reliably improves detection.” RF impurity importance can supplement these results but cannot establish a causal feature contribution on its own.

The temporal group includes lifetime `pair_seen_before` as well as recent counts, recencies and activity indicators. These experiments do not independently identify the contribution of each temporal variable. In particular, a global-non-edge injection rule makes pair novelty a known synthetic shortcut.

## The novel-normal subset changes the class prior

| Dataset | Full normal events | Anomalies | Full anomaly fraction | First-seen normal events | Subset anomaly fraction |
|---|---:|---:|---:|---:|---:|
| MovieLens | 18,886 | 994 | 5.0000% | 18,886 | 5.0000% |
| UCI Messages | 1,787 | 94 | 4.9973% | 320 | 22.7053% |

MovieLens has no repeated normal test pairs, so its subset results are identical to the full test. In UCI, 82.0929% of normal test events are historically repeated pairs. Removing them retains all 94 anomalies but reduces the total from 1,881 to 414 examples. The constant-score AP reference therefore rises from approximately 0.05 to approximately 0.227.

For UCI G3 RF, the extended-feature ROC-AUC decreases from 0.9684 on the full test to 0.8234 on the first-seen-normal subset. TGF ROC-AUC decreases from 0.9358 to 0.7433. This shows that comparison against repeated normal interactions contributes to the full-test discrimination.

AP behaves differently: UCI G3 extended AP remains approximately 0.537, while TGF AP rises from 0.3568 to 0.4343. This is not evidence that restricting the normal class made the problem intrinsically easier. AP depends on prevalence, and removing low-ranked normal events may have little effect on precision at the ranked anomalies. Always present the changed class prior alongside this supplemental analysis.

## Generator fidelity limits the structural interpretation

The generators use ranked finite candidate pools. They reduce some discrepancies but do not achieve exact distribution matching. Averaging test diagnostics over the specified endpoint features and five seeds, mean absolute standardized mean difference decreases from 1.039 to 0.641 for MovieLens G0→G3 and from 1.171 to 1.010 for UCI G0→G3. These are residual differences large enough to matter.

The fraction of G3 samples whose endpoint RMS log-distance exceeds log(2) remains 76.3% for MovieLens and 69.1% for UCI. This is a diagnostic threshold, not a claim that matching support was sufficient under a hard factor-two constraint.

UCI specifically does not support a clean local-preservation interpretation for G2. The mean selected-reference local distance improves from 0.05463 under G1 to 0.05078 under G2, but the test marginal local-feature KS statistic, averaged across local features and seeds, worsens from 0.09815 to 0.10655. A reference-distance objective and marginal population matching are different checks. Report this as a local-targeted sampling mechanism with incomplete fidelity, rather than evidence that only local structure was controlled.

Other limits to carry into the final report are:

- The global-non-edge filter uses the complete observed edge set. Detector features are pre-arrival, but this label-construction rule is not fully online. The past-only sensitivity is a separate protocol.
- Synthetic edges are treated as anomalies by construction; observed interactions are assumed normal. Neither is a verified real-fraud label.
- Histories contain past injected and observed events. Changing the generator also changes later graph state, so normal-event feature vectors are not identical across generator conditions.
- Chronological splitting uses elapsed time. UCI has only 94 test anomalies per seed, and its temporal distribution is highly imbalanced across splits. Five injection seeds on the same stream do not represent five independent real datasets.
- Precision@K equals Recall@K here because K is defined as the number of true test anomalies. These are two names for the same value under this protocol.
- These feature ablations and synthetic generator controls support association and mechanism-specific comparisons, not causal claims about real anomaly behavior.

The main experiment is complete at the predeclared factorial scope. The strongest defensible conclusion is that the synthetic injection mechanism substantially changes measured detector performance and transfer, and that temporal features account for much of the recoverable signal in these two streams. The experiment also documents residual matching artifacts and prevalence effects that prevent a stronger claim of realistic anomaly detection.
