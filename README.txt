COMP5331: STRUCTURE- AND TIME-AWARE GRAPH ANOMALY DETECTION
Repository: https://github.com/zxz2004626/5331-project

1. SETUP / COMPILATION

Use 64-bit Python 3.10 or 3.11 (tested with 3.11). From this directory:

  python -m venv .venv
  .\.venv\Scripts\Activate.ps1                 (Windows PowerShell)
  source .venv/bin/activate                   (Linux/macOS alternative)
  python -m pip install -e ".[dev]"

This is Python code; no native compilation is required. Optional bytecode check:
  python -m compileall -q src scripts

Git is required for prepare. The official --fresh-pipeline mode additionally
requires Bash, gzip, and coreutils; use Git Bash on Windows.

2. EXECUTION AND EXAMPLES

Show options or preview the complete experiment:
  python -m graphbench --help
  python -m graphbench run --dry-run

Verify supplied official predictions without training or downloading data:
  python -m graphbench official --verify-output

Run one dataset, generator, model, and seed:
  python -m graphbench prepare
  python -m graphbench generate --datasets ucimessages --generators G0 --seeds 11 --workers 1
  python -m graphbench evaluate --datasets ucimessages --generators G0 --seeds 11 --models RF --features tgf --same-generator-only --n-jobs 1

Run the full experiment, or rebuild result tables/plots from saved records:
  python -m graphbench run --workers 1 --model-jobs 2 --no-pair-jobs 1
  python -m graphbench summarize

Other commands: no-pair (prior-pair control), verify (artifact audit).
Each command accepts --help. Full regeneration is computationally expensive;
use a separate checkout for new runs rather than overwriting historical outputs.

3. SOURCE FILES AND THEIR RELATION TO THE REPORT

Core implementation under src/graphbench/:
  data.py          Input loading and chronological splits (Section 4.1).
  features.py      Strict history, TGF/local/temporal features (Sections 3.1, 3.3).
  generators.py    G0-G3 injection schedules and candidate matching (Section 3.2).
  models.py        RF/LR model selection (Sections 3.3, 4.2).
  metrics.py       AP, PR-AUC, ROC-AUC, and top-K metrics (Section 4.2).
  evaluation.py    Same-/cross-generator evaluation and feature ablations (Section 5).
  result_inputs.py Saved split counts and diagnostics for result summaries.
  cache.py         Cache validation and feature-column alignment.
  summaries.py     Aggregate valid evaluation records.
  artifacts.py     Atomic file I/O and saved-path resolution.
  provenance.py    Source fingerprints and artifact lineage.
  paths.py         Relocatable project-root discovery.
  constants.py     Dataset, generator, seed, and sensitivity identifiers.
  shell.py         Portable upstream Bash invocation.
  cli.py           Command dispatch and --root override.
  __main__.py      Entry point for python -m graphbench.
  __init__.py      Package initialization.

Workflows under src/graphbench/workflows/; scripts/ contains matching entry
files with the same names, forwarding to the corresponding workflow:
  prepare_sources.py       Fetch pinned upstream TGF inputs/code.
  generate_anomalies.py    Schedule generator jobs (Section 3.2).
  evaluate_models.py       Schedule training and evaluation (Sections 4-5).
  pair_novelty_control.py  Prior-pair feature removal control (Section 5.6).
  official_reproduction.py Official baseline and feature-timing contrast (Section 5.1).
  summarize_results.py     CSV/JSON tables and PNG/PDF result figures (Section 5).
  verify_artifacts.py      Independent checks of saved predictions and provenance.
  pipeline.py              Orchestrate main, sensitivity, and control runs.
  __init__.py              Workflow package initialization (no script counterpart).
scripts/run_all.py calls pipeline.py; scripts/run_all.ps1 is its PowerShell wrapper.

Tests under tests/:
  test_features.py      History arithmetic, timing, and feature definitions.
  test_generators.py    Sampling, matching, legality, and cache lineage.
  test_evaluation.py    Training/evaluation, metrics, and artifact handling.
  test_pair_novelty.py  Prior-pair control and selected-parameter reuse.
  test_portability.py   Relocation, CLI entry points, source hashes, and Bash calls.

Archived upstream helpers under results/official/fresh_pipeline/:
  movielens/bip_check.sh and ucimessages/check.sh: input checks.
  movielens/bip_build_injectable.sh and ucimessages/build_injectable.sh: candidates.
  Each dataset's inject.sh and random_sample.py: original injection and sampling.

configs/experiment.json defines the protocol. pyproject.toml defines packaging
and test settings; requirements.txt pins dependencies. .github/workflows/checks.yml
configures CI. .gitignore and .gitattributes control artifacts and line endings.

4. DATA, RESULTS, AND LOGS

data/manifest.json records exact input URLs, checksums, and statistics. Cleaned
data and preprocessing intermediates are retained in results/official/fresh_pipeline/.
The normal loader reads the same inputs from vendor/TGF/ after prepare.

results/evaluation/ contains metrics and run records; results/official/ contains
baseline/timing results; results/no_pair/ contains the novelty control;
results/report/ contains tables and figures; results/validation/ contains audits.
All existing logs, including results/logs/, are retained. results/failures/ retains
failed-run records. Figure PDFs are results, not prose report documents.

Large main feature caches, fitted extension models, and main prediction arrays
must be rebuilt for complete event-level rescoring. Historical hashes/paths are
preserved; renamed or edited source files require new cache fingerprints.
Use verify --available-only to audit the supplied subset with coverage reporting.

Relocation: run python -m graphbench --root "PATH TO PROJECT" COMMAND.
GRAPHBENCH_ROOT is the equivalent environment override; GRAPHBENCH_BASH selects Bash.

5. TESTED SYSTEM AND CHECKS

Tested on 64-bit Windows build 26200, Python 3.11.9, NumPy 1.24.3,
pandas 1.5.3, and scikit-learn 1.3.0. Run:
  python -m pytest -q
  python -m ruff check src scripts tests
  python -m ruff format --check src scripts tests

Local tests: 51 passed, 3 skipped. A real UCI G0/RF generation and evaluation
run completed with zero failures in an isolated copy.

Packaging, installed-package execution in a relocated path containing spaces,
official prediction verification, result generation, and the dry-run pipeline
were checked. The full expensive experiment matrix was not rerun for cleanup.
Linux/macOS are configured in CI but were not locally tested. Tests requiring
the absent upstream checkout or Bash are skipped explicitly.

Upstream: https://github.com/StephanyRajeh/TGF (GPL-2.0), pinned revision
af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b. The experiments use synthetic labels;
matching is approximate and does not establish real-world fraud detection.
