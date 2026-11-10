# Graph Anomaly Detection

[中文说明](README.zh-CN.md) · [Repository](https://github.com/zxz2004626/5331-project)

COMP5331 Group 5: compare G0–G3 synthetic anomaly generators, RF/LR detectors,
and interpretable graph features on MovieLens and UCI Messages.

## Install and compile

Use **64-bit Python 3.10 or 3.11**. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On Linux/macOS, activate with `source .venv/bin/activate` instead. This is Python
code; no native compilation is needed. Optional bytecode check:
`python -m compileall -q src`.

Git is required for `prepare`. The official `--fresh-pipeline` mode also needs
Bash, gzip, and coreutils; use Git Bash on Windows.

## Run

Quick check using supplied predictions, without training or downloading data:

```sh
python -m graph_anomaly official --verify-output
```

Example: one dataset, generator, seed, and model:

```sh
python -m graph_anomaly prepare
python -m graph_anomaly generate --datasets ucimessages --generators G0 --seeds 11 --workers 1
python -m graph_anomaly evaluate --datasets ucimessages --generators G0 --seeds 11 --models RF --features tgf --same-generator-only --n-jobs 1
```

```sh
python -m graph_anomaly run --dry-run  # Preview the complete experiment
python -m graph_anomaly run            # Execute the complete experiment
python -m graph_anomaly summarize      # Rebuild result tables and figures
python -m graph_anomaly --help
```

The installed console command is `graph-anomaly`. Other commands are `no-pair`
(prior-pair control) and `verify` (artifact audit). Each accepts `--help`.

## Source files

All implementation files are under `src/graph_anomaly/`; numerical logic and its
CLI workflow stay together instead of using forwarding scripts.

| File | Responsibility |
|---|---|
| `config.py` | Experiment identifiers and relocatable project paths. |
| `data.py` | Dataset loading, chronological splits, cache validation and feature alignment. |
| `features.py` | Strict graph history and TGF/local/temporal features (report §§3.1, 3.3). |
| `generation.py` | G0–G3 candidate selection and timestamp-batched generation (§3.2). |
| `models.py` | RF/LR estimators and validation-based model selection (§4.2). |
| `metrics.py` | AP, PR-AUC, ROC-AUC, and top-K metrics (§4.2). |
| `evaluation.py` | Model checkpointing, same-/cross-generator tests and ablations (§5). |
| `controls.py` | Prior-pair feature removal control (§5.6). |
| `official.py` | Upstream preparation, baseline reproduction and timing comparison (§5.1). |
| `results.py` | Aggregate runs and produce CSV/JSON tables and PNG/PDF figures. |
| `verify.py` | Independently check predictions, cache lineage, counts and summaries. |
| `artifacts.py` | Atomic file I/O and source fingerprints. |
| `cli.py` | Command dispatch and full experiment scheduling. |
| `__main__.py` | `python -m graph_anomaly` entry point. |
| `__init__.py` | Package initialization. |

Tests: `test_features.py` checks history/features; `test_generators.py` checks
sampling and causal caches; `test_evaluation.py` checks metrics/training/resume;
`test_pair_novelty.py` checks the control; `test_portability.py` checks relocation
and command entry points. Configuration and dependencies are in
`configs/experiment.json`, `pyproject.toml`, and `requirements.txt`.

Archived upstream helpers in `results/official/fresh_pipeline/`: `bip_check.sh`
and `check.sh` check inputs; `bip_build_injectable.sh` and `build_injectable.sh`
build candidates; each dataset's `inject.sh` and `random_sample.py` perform
original injection and sampling.

## Data, results and logs

`data/manifest.json` records exact sources/checksums. Cleaned data and preprocessing
intermediates are retained in `results/official/fresh_pipeline/`; `prepare` fetches
the same inputs into `vendor/TGF/`. Results are under `results/evaluation/`,
`official/`, `no_pair/`, `report/`, and `validation/`. All original logs are kept,
including `results/logs/`; failed-run records remain in `results/failures/`.

Large main caches, extension models and main predictions require regeneration.
Use a separate checkout for new runs. Historical results retain their original
hashes; changed source requires fresh caches. `verify --available-only` reports
the coverage of the supplied subset.

To relocate: `python -m graph_anomaly --root "PATH TO PROJECT" COMMAND`.
Environment overrides: `GRAPH_ANOMALY_ROOT` and `GRAPH_ANOMALY_BASH`.

## Tests and tested system

```sh
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Tested on **64-bit Windows build 26200, Python 3.11.9**, NumPy 1.24.3,
pandas 1.5.3 and scikit-learn 1.3.0. Checks cover unit/portability tests,
packaging, installed execution from a path containing spaces, saved predictions,
result generation and a real UCI G0/RF run. The full experiment matrix is not
rerun for refactoring. Tests requiring absent upstream code or Bash are skipped.
Linux/macOS CI is configured but was not locally tested.

Upstream: [TGF](https://github.com/StephanyRajeh/TGF), GPL-2.0, pinned commit
`af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`.
