param([string]$Python = 'python', [int]$GeneratorWorkers = 1, [int]$ModelJobs = 2, [int]$NoPairJobs = 1)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
function Run-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw ('Command failed: ' + ($args -join ' ')) }
}
Run-Python scripts/prepare_sources.py
$env:PYTHONPATH = 'src'
Run-Python -m pytest tests -q
Run-Python scripts/official_reproduction.py --paired-timing --fresh-pipeline --verify-output
Run-Python scripts/build_experiments.py --workers $GeneratorWorkers
Run-Python scripts/evaluate_experiments.py --ablations --n-jobs $ModelJobs
foreach ($scenario in @('ratio01','ratio10','window500','window2000','past_only')) {
    Run-Python scripts/build_experiments.py --scenario $scenario --generators G0 G3 --workers $GeneratorWorkers
    Run-Python scripts/evaluate_experiments.py --scenario $scenario --generators G0 G3 --models RF --features tgf extended --same-generator-only --n-jobs $ModelJobs
}
Run-Python scripts/no_pair_ablation.py --n-jobs $NoPairJobs
Run-Python scripts/verify_artifacts.py
Run-Python scripts/report_results.py
Run-Python scripts/write_report.py
Write-Output 'Completed experiments, tables, plots and Chinese Markdown report. The optional PDF renderer uses reportlab; see README.'
