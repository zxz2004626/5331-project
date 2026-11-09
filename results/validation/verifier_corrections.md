# Verification-script corrections

The first complete independent verification recomputed all1080 evaluation metrics correctly (maximum error3.33e-16), but its own protocol assertions had two mistakes. Its output is preserved in `verifier_first_audit.json` and `verifier_first_run.log`.

1. It treated the1000th warmup event as zero-based index1000. The declared configuration and generator use index999, excluding that event's complete timestamp. The generator was corrected before any final main or sensitivity cache was constructed; the final cache timestamps/counts and tests already reflect this protocol.
2. It required two execution threads for every result, although the supplemental no-pair run explicitly used one thread to constrain memory. All core/sensitivity models used two threads; the no-pair metadata, README and protocol document one thread. Model hyperparameters were otherwise reused as specified.

Only these assertions in `scripts/verify_artifacts.py` were corrected. No dataset, feature cache, fitted model, prediction, metric, or experimental result was edited. The corrected complete audit is `final_audit.json`.
